from functools import partial
from time import time

import numpy as np
import torch

from .roadrunner_nn.actor import FF_Stochastic_Actor, LSTM_Stochastic_Actor

def get_model_class(model_key):
    # TODO: make these opcc models?
    if model_key == 'ff':
        return FF_Stochastic_Actor
    elif model_key == 'lstm':
        return LSTM_Stochastic_Actor
    else:
        raise ValueError(f"model_key {model_key} not recognized.")

def get_model_factory(config):
    model_key = config.model_key
    model_class = get_model_class(model_key)
    # TODO: don't hardcode this, react to the actual env.
    from ..envs.cartpole_env import Cartpole_Env
    env_class = Cartpole_Env
    env = env_class()
    observation_size = env.observation_size
    action_size = env.action_size
    state_action_size = observation_size + action_size
    return partial(model_class, input_dim=state_action_size, action_dim=observation_size, layers=config.model_layers)



class Ensemble_Network:

    def __init__(self, obs_size, action_size, config):
        self.obs_size = obs_size
        self.action_size = action_size
        model_lr = config.model_lr
        model_eps = config.model_eps

        model_factory = get_model_factory(config)

        self.inference_model = model_factory()
        self.inference_model_optim = torch.optim.Adam(self.inference_model.parameters(), lr=model_lr, eps=model_eps)

        self.ensemble = []
        self.ensemble_optims = []
        for i in range(config.model_num_ensemble):
            net = model_factory()
            self.ensemble.append(net)
            optim = torch.optim.Adam(net.parameters(), lr=model_lr, eps=model_eps)
            self.ensemble_optims.append(optim)

        self.nets = [self.inference_model] + self.ensemble

    
    def get_all_parameters(self):
        return [list(net.parameters()) for net in self.nets] # Casting to list may be unnecessary.

    
    def propagate(self, method_name, *args, **kwargs):
        # for net in [self.inference_model] + self.ensemble:
        for net in self.nets:
            method = getattr(net, method_name)
            method(*args, **kwargs)

    
    def train(self, *args, **kwargs):
        self.propagate('train', *args, **kwargs)
    def init_hidden_state(self):
        for net in self.nets:
            if hasattr(net, 'init_hidden_state'):
                net.init_hidden_state()


    def __call__(self, *args, **kwargs):
        self.step(*args, **kwargs)
    def step(self, state, action, update_norm=False, debug=False):
        # TODO: verify state and action shapes.
        # TODO: remove / verify presence of 'done'?
        
        state = torch.Tensor(state)
        action = torch.Tensor(action)
        assert state.dim() == 1
        assert action.dim() == 1

        state_action = torch.cat([state, action], dim=-1)

        predicted_next_state = self.inference_model(state_action, update_norm=update_norm)
        
        for ensemble_idx in range(len(self.ensemble)):
            net = self.ensemble[ensemble_idx]
            # TODO: slice & unsqueeze right?
            net_next_state = net(state_action, update_norm=False)
            net_next_state = net_next_state.unsqueeze(0)
            if update_norm: net.copy_normalizer_stats(self.inference_model)
            if ensemble_idx == 0:
                ensemble_next_state = net_next_state
            else:
                ensemble_next_state = torch.cat([ensemble_next_state, net_next_state], dim=0)
        # uncertainty is the ensemble's average element-wise stdev of the predicted next state.
        uncertainty = torch.std(ensemble_next_state, dim=0).mean()

        output = predicted_next_state, uncertainty
        if debug: 
            print("\n\nNOTICE ME\n\n")
            print(f"output is None: {output is None}")
            print(f"output: {output}")
        return output
        return predicted_next_state, uncertainty


    def optimize(self, env_sample_buffer, epochs: int,
               batch_size: int, kl_thresh):
        torch.set_num_threads(1)
        info = {}
        terminate_early = False
        losses = []
        for epoch in range(epochs):
            epoch_start_time = time()
            # Update self.inference_model and self.ensemble.
            ensemble_indices = [None] + list(range(len(self.ensemble))) # nets = [self.inference_model] + self.ensemble
            # for ensemble_index, net in zip(nets, ensemble_indices):
            for ensemble_index in ensemble_indices:
                for batch in env_sample_buffer.sample(batch_size=batch_size, recurrent=self.recurrent, ensemble_index=ensemble_index):
                    states, actions, are_real, traj_mask = batch
                    loss, kl_div = self._update(ensemble_index, states, actions, are_real, traj_mask)
                    losses.append(loss)
                    # Note: kl_div not implemented atm.
                    # if max(kl_divs) > kl_thresh:
                    #     print(f"Batch had KL divergence {max(kl_divs)} (threshold {kl_thresh})."
                    #         "Stopping optimization early.")
                    #     terminate_early = True
                    #     break
            if terminate_early:
                break
        info['avg_loss'] = np.average(losses)
        return info

    # TODO: propagate use of traj_mask.
    def _update(self, ensemble_index, states, actions, are_real, traj_mask):
        """
        Compute loss and invoke the corresponding network's optimizer.
        Assumes everything is normalized already.
        """
        all_state_actions = torch.cat([states, actions], dim=-1)

        state_actions = all_state_actions[:-1] # Skip last time step.
        net = self.inference_model if ensemble_index is None else self.ensemble[ensemble_index]
        
        # TODO: add in old_actor counterpart(s) to compute kl_divergence.
        # with torch.no_grad():
        #     old_pdf = net.pdf(state_actions)
        kl_div = 0 # This is just not implemented.

        # TODO: adjust collection to include the terminating state and not waste the last state-action input.
        all_state_actions_normalized = self.inference_model.normalize_state(all_state_actions, update=False)
        states_normalized = all_state_actions_normalized[:-1, ..., :states.shape[-1]] # Skip last time step.
        next_states_normalized = all_state_actions_normalized[1:, ..., :states.shape[-1]] # Skip first time step.
        next_state_preds = net(state_actions, update_norm=False)
        # next_state_preds_normalized = self.inference_model.normalize_state(next_state_preds, update=False)

        # Take the state error and normalize across the state dimension.
        prediction_errors = torch.linalg.norm((next_state_preds - next_states_normalized)*traj_mask[:-1], dim=-1)
        loss = prediction_errors.mean()

        optim = self.inference_model_optim if ensemble_index is None else self.ensemble_optims[ensemble_index]
        optim.zero_grad()
        loss.backward()
        optim.step()

        return loss, kl_div