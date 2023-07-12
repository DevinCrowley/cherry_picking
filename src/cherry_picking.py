"""Proximal Policy Optimization (clip objective)."""
import os
from time import time, sleep
from copy import deepcopy

import ray
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import kl_divergence

from .buffers import Model_Buffer as Env_Sample_Buffer, Policy_Buffer as Model_Sample_Buffer
from .buffers import Policy_Buffer as Model_Sample_Buffer
from .nn.ensemble import Ensemble_Network, get_recurrent
from .util import train_normalizer, train_model_normalizer

from .nn.roadrunner_nn.actor import LSTM_Stochastic_Actor, GRU_Stochastic_Actor, FF_Stochastic_Actor
from .nn.roadrunner_nn.critic import LSTM_V, GRU_V, FF_V

"""
Notes:
    - Currently making each iteration gather all new data, this could be adjusted
        to partly replace the buffer rather than fully, as in it will randomly replace
        data with new data once it's at a larger overall size, and/or could use 
        long-term file storage. In which case it may have to mark which samples
        are predicted and not real.
        - This should be reflected in Model_Sampler.collect_experience's loop, 
            amount of data based on a given size rather than all collected env data.
    - Could make env.step output standardized rather than doing things piecewise.
    - Could have the model predict the reward as well as the full state.
    - TODO: normalize states and actions
- The Env_Sampler, which collects env data for model learning, could:
    -> collect a batch and train on it
    - accumulate a batch and train on all data up to that point
    - The Model_Sampler, which collects synthetic data for policy learning, could:
        -> predict from this batch of trajectories from Env_Sampler
        - choose a random set of trajectories to predict from
        - predict trajectories from reset states
    - Can we make this more memory efficient by interleaving policy updates with data collection?
    - Support ensemble mixture from opcc?, randomly sampling from ensemble for each step call
    - Is it necessary to model 'done'?
    - Predict reward along with next_state?
    - Move all use of config down into actual use, not into __init__ methods.
    - Have ensemble learn to predict its own disagreement?
        - incorporate opcc's min/max log_var?
    - Note: should really make the networks only normalize the state and not the action for state-action inputs.

Big TODOs:
    - Get the model to incorporate an ensemble, and have a step/__call__ method that 
        invokes the ensemble to get both the regular prediction and the bootstrap uncertainty, 
        and a train method that has its ensemble components get bootstrapped versions of the data 
        and outputs loss.
        - Write Env_Sample_Buffer, handle bootstrapping with ensemble_index
        - Add lstm net, give update method
    - Normalize states across the board.

Plan:
    - Model_Optimizer --> call model.update, update self with new model params.

Thoughts:
    - Leverage ensemble for inference by choosing the next_state that all models agree with the most? Requires more noodling.


Daimler TODOs:
    - OPCC
        - Learn normalization
        - Learn inherent variance
    - Dreamer
        - Separate deterministic and stochastic components in the model with RSSM
"""


class Worker:
    """
    Template for a worker, either a sampler or an optimizer.

    Includes actor, critic, and model.
    """

    def __init__(self, model, actor, critic, env_factory) -> None:
        self.model = deepcopy(model)
        self.actor = deepcopy(actor)
        self.critic = deepcopy(critic)
        self.env = env_factory()


    def retrieve_parameters(self):
        return (
            list(self.model.get_all_parameters()),
            list(self.actor.parameters()),
            list(self.critic.parameters()),
            (self.model.inference_model.welford_state_mean, self.model.inference_model.welford_state_mean_diff, self.model.inference_model.welford_state_n), # input_norm
        )


    def sync(self, new_model_params=None, new_actor_params=None, new_critic_params=None, input_norm=None):
        """
        Function to sync the mode, actor, and critic parameters with new parameters.

        Args:
            new_model_params (torch dictionary): New model parameters to copy over
            new_actor_params (torch dictionary): New actor parameters to copy over
            new_critic_params (torch dictionary): New critic parameters to copy over
            input_norm (int): Running counter of states for normalization 
        """
        if new_model_params is not None:
            for net_params, new_net_params in zip(self.model.get_all_parameters(), new_model_params):
                for p, new_p in zip(net_params, new_net_params):
                    p.data.copy_(new_p)

        if new_actor_params is not None:
            for p, new_p in zip(self.actor.parameters(), new_actor_params):
                p.data.copy_(new_p)

        if new_critic_params is not None:
            for p, new_p in zip(self.critic.parameters(), new_critic_params):
                p.data.copy_(new_p)

        if input_norm is not None:
            self.model.propagate('set_normalizer_stats', input_norm)
            self.actor.copy_normalizer_stats(self.model.inference_model)
            self.critic.copy_normalizer_stats(self.actor)


@ray.remote
class Env_Sampler(Worker):
    """
    Class for collecting data from the environment for model learning.
    """

    def __init__(self, model, actor, critic, env_factory, config) -> None:
        super().__init__(model, actor, critic, env_factory)
        self.uncertainty_max = config.uncertainty_max
        self.learn_norm = config.model_learn_norm

        self.model_recurrent = get_recurrent(config.model_arch_key)
        self.override_confidence = config.override_confidence

    def collect_experience(self, max_traj_len, min_steps):
        """
        Function to sample experience

        Args:
            max_traj_len: maximum trajectory length of an episode
            min_steps: minimum total steps to sample
        """
        torch.set_num_threads(1)
        with torch.no_grad():
            start = time()

            num_steps = 0
            env_sample_buffer = Env_Sample_Buffer()

            while num_steps < min_steps:
                # self.env.dynamics_randomization = self.dynamics_randomization
                state = torch.Tensor(self.env.reset())

                done = False
                value = 0
                traj_len = 0

                if hasattr(self.model, 'init_hidden_state'):
                    self.model.init_hidden_state()
                if hasattr(self.actor, 'init_hidden_state'):
                    self.actor.init_hidden_state()
                if hasattr(self.critic, 'init_hidden_state'):
                    self.critic.init_hidden_state()

                state_is_real = True
                while not done and traj_len < max_traj_len:
                    state = torch.Tensor(state)
                    action = self.actor(state, deterministic=False)
                    # value = self.critic(state)

                    # TODO: check how opcc does this?
                    next_state_pred, uncertainty = self.model(state, action, update_norm=self.learn_norm)
                    if (uncertainty > self.uncertainty_max) or (np.random.random() < self.override_confidence):
                        next_state, reward, done, info = self.env.step(action.numpy())
                        next_state = torch.Tensor(next_state)
                        next_state_is_real = 1
                    else:
                        next_state = next_state_pred.clone()
                        next_state_denormalized = self.model.inference_model.denormalize_state(next_state)
                        self.env.set_state(next_state_denormalized.numpy())
                        next_state = next_state_denormalized
                        done = self.env.compute_done()
                        next_state_is_real = 0
                    
                    if self.model_recurrent:
                        env_sample_buffer.push(state.numpy(), action.numpy(), state_is_real)
                    elif next_state_is_real:
                        env_sample_buffer.push(state.numpy(), action.numpy(), next_state_is_real, next_state.numpy())
                    
                    state = next_state
                    state_is_real = next_state_is_real
                    traj_len += 1
                    num_steps += 1
                env_sample_buffer.end_trajectory()

        return env_sample_buffer, num_steps
    
    def evaluate_model(self, trajs=1, max_traj_len=400):
        """
        Function to evaluate the model.

        Args:
            max_traj_len: maximum trajectory length of an episode
            trajs: minimum trajectories to evaluate for
        """
        with torch.no_grad():
            ep_avg_model_errors = []
            ep_avg_model_uncertainties = []
            ep_returns = []
            traj_lens = []
            for traj in range(trajs):
                self.env.dynamics_randomization = False
                state = torch.Tensor(self.env.reset())

                done = False
                model_errors = []
                model_uncertainties = []
                traj_len = 0
                ep_return = 0

                
                if hasattr(self.model, 'init_hidden_state'):
                    self.model.init_hidden_state()
                if hasattr(self.actor, 'init_hidden_state'):
                    self.actor.init_hidden_state()

                while not done and traj_len < max_traj_len:
                    action = self.actor(state, deterministic=True)

                    next_state_pred, uncertainty = self.model(state, action)
                    next_state, reward, done, _ = self.env.step(action.numpy())
                    next_state_normalized = self.model.inference_model.normalize_state(next_state, update=False)

                    state = torch.Tensor(next_state)
                    model_error = np.linalg.norm(next_state_pred - next_state_normalized).mean()
                    model_errors.append(model_error)
                    model_uncertainties.append(uncertainty)
                    ep_return += reward
                    traj_len += 1
                ep_avg_model_errors.append(np.mean(model_errors))
                ep_avg_model_uncertainties.append(np.mean(model_uncertainties))
                ep_returns += [ep_return]
                traj_lens += [traj_len]
            error_uncertainty_correlation = np.corrcoef(model_errors, model_uncertainties)[0, 1]

        return {
            'avg_model_error': np.mean(ep_avg_model_errors),
            'avg_model_uncertainty': np.mean(ep_avg_model_uncertainties),
            'avg_return': np.mean(ep_returns),
            'avg_traj_len': np.mean(traj_lens),
            'error_uncertainty_correlation': error_uncertainty_correlation,
        }


@ray.remote
class Model_Optimizer(Worker):
    """
    Worker for optimizing the model.
    """
    
    def __init__(self, model, actor, critic, env_factory, config) -> None:
        super().__init__(model, actor, critic, env_factory)
        # self.model_optim = optim.Adam(self.model.parameters(), lr=config.model_lr)
        self.recurrent = get_recurrent(config.model_arch_key)
        # self.grad_clip = config.model_grad_clip

    def optimize(
        self,
        env_sample_buffer,
        epochs=4,
        batch_size=32,
        kl_thresh=0.02,
    ):
        """
        Does a single optimization step for the model.
        """
        torch.set_num_threads(1)
        info = self.model.optimize(env_sample_buffer=env_sample_buffer, epochs=epochs, batch_size=batch_size, kl_thresh=kl_thresh)
        return info


@ray.remote
class Model_Sampler(Worker):
    """
    Class for collecting data from the model for policy learning.
    """

    def __init__(self, model, actor, critic, env_factory, config) -> None:
        super().__init__(model, actor, critic, env_factory)
        self.discount = config.policy_discount

    def collect_experience(self, start_states, max_traj_len):
        """
        Function to sample experience. 
        Predicts trajectories from each state in start_states.

        Args:
            max_traj_len: maximum trajectory length of an episode
            min_steps: minimum total steps to sample
        """
        torch.set_num_threads(1)
        with torch.no_grad():
            start = time()

            num_steps = 0
            model_sample_buffer = Model_Sample_Buffer(self.discount)

            # while num_steps < min_steps:
            for start_state in start_states:
                # self.env.dynamics_randomization = self.dynamics_randomization
                state = torch.Tensor(start_state)
                self.env.reset(state)

                done = False
                value = 0
                traj_len = 0


                if hasattr(self.model, 'init_hidden_state'):
                    self.model.init_hidden_state()
                if hasattr(self.actor, 'init_hidden_state'):
                    self.actor.init_hidden_state()
                if hasattr(self.critic, 'init_hidden_state'):
                    self.critic.init_hidden_state()

                while not done and traj_len < max_traj_len:
                    state = torch.Tensor(state)
                    action = self.actor(state, deterministic=False)
                    value = self.critic(state)

                    next_state_pred, uncertainty = self.model(state, action, update_norm=False)
                    next_state = next_state_pred
                    next_state_denormalized = self.model.inference_model.denormalize_state(next_state_pred)
                    self.env.set_state(next_state_denormalized.numpy())
                    reward = self.env.compute_reward(action)
                    reward = np.array([reward])
                    done = self.env.compute_done()
                    
                    model_sample_buffer.push(state.numpy(), action.numpy(), reward, value.numpy())
                    
                    state = next_state_denormalized
                    traj_len += 1
                    # num_steps += 1

                value = (not done) * self.critic(torch.Tensor(state)).numpy()
                model_sample_buffer.end_trajectory(terminal_value=value)

        return model_sample_buffer
    

@ray.remote
class Policy_Optimizer(Worker):
    """
    Worker for optimizing the policy (actor and critic).
    """
    
    def __init__(self, model, actor, critic, env_factory, config) -> None:
        super().__init__(model, actor, critic, env_factory)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=config.actor_lr, eps=config.policy_eps)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=config.critic_lr, eps=config.policy_eps)
        self.recurrent = get_recurrent(config.policy_arch_key)
        self.grad_clip = config.policy_grad_clip
        self.clip = config.policy_clip
        self.entropy_coeff = config.policy_entropy_coeff
        self.old_actor = deepcopy(actor)

    def optimize(
        self,
        model_sample_buffer,
        epochs=4,
        batch_size=32,
        kl_thresh=0.02,
    ):
        """
        Does a single optimization step for the policy (actor and critic).
        """
        self.old_actor.load_state_dict(self.actor.state_dict())
        torch.set_num_threads(1)
        kl_divs, actor_losses, critic_losses = [], [], []
        done = False
        for epoch in range(epochs):
            epoch_start_time = time()
            for batch in model_sample_buffer.sample(batch_size=batch_size, recurrent=self.recurrent):
                states, actions, returns, advantages, mask = batch
                kl_div, actor_loss, critic_loss = self._update_policy(states, actions, returns, advantages, mask)
                
                kl_divs.append(kl_div)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                if max(kl_divs) > kl_thresh:
                    print(f"Batch had KL divergence {max(kl_divs)} (threshold {kl_thresh})."
                          "Stopping optimization early.")
                    done = True
                    break
            if done:
                break
        info = {
            'avg_kl_div': np.mean(kl_divs),
            'avg_actor_loss': np.mean(actor_losses),
            'avg_critic_loss': np.mean(critic_losses),
        }
        return info

    # TODO: check whether to include all the PPO stuff like the clipping, 
    # is it appropriate if we're backpropagating through the world model?
    def _update_policy(self, states, actions, returns, advantages, mask):
        """
        Update the policy (actor and critic) and return losses and kl divergence.
        """
        with torch.no_grad():
            old_pdf = self.old_actor.pdf(states)
            old_log_probs = old_pdf.log_prob(actions).sum(-1, keepdim=True)
        
        # Get new action distribution and log probabilities.
        pdf = self.actor.pdf(states)
        log_probs = pdf.log_prob(actions).sum(-1, keepdim=True)

        ratio = ((log_probs - old_log_probs) * mask).exp()
        cpi_loss   = ratio * advantages * mask
        clip_loss  = ratio.clamp(1.0 - self.clip, 1 + self.clip) * advantages * mask
        actor_loss = -torch.min(cpi_loss, clip_loss).mean()

        critic_loss = 0.5 * ((returns - self.critic(states)) * mask).pow(2).mean()

        entropy_penalty = -(self.entropy_coeff * pdf.entropy() * mask).mean()

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

        (actor_loss + entropy_penalty).backward()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip)
        self.actor_optim.step()
        self.critic_optim.step()

        with torch.no_grad():
            kl_div = kl_divergence(pdf, old_pdf).mean().numpy()

        return kl_div, (actor_loss + entropy_penalty).item(), critic_loss.item()


class Cherry_Picking_Algo(Worker):

    def __init__(self, model, actor, critic, env_factory, config) -> None:
        super().__init__(model, actor, critic, env_factory)
        self.discount = config.policy_discount
        self.entropy_coeff = config.policy_entropy_coeff
        self.grad_clip = config.policy_grad_clip

        # Initialize ray.
        if not ray.is_initialized():
            if config.redis_address is not None:
                ray.init(redis_address=config.redis_address)
            else:
                ray.init(num_cpus=config.workers)
        
        self.env_samplers = [Env_Sampler.remote(model, actor, critic, env_factory, config) for _ in range(config.workers)]
        self.model_samplers = [Model_Sampler.remote(model, actor, critic, env_factory, config) for _ in range(config.workers)]
        self.model_optimizer = Model_Optimizer.remote(model, actor, critic, env_factory, config)
        if config.learn_policy: self.policy_optimizer = Policy_Optimizer.remote(model, actor, critic, env_factory, config)

        self.env_sample_buffer_queue = Env_Sample_Buffer()

    def do_iteration(self, config): 
        """
        Function to do a single iteration of Cherry Picking

        Args:
            max_traj_len (int): maximum trajectory length of an episode
            num_steps (int): number of steps to collect experience for
            epochs (int): optimization epochs
            batch_size (int): optimzation batch size
            mirror (bool): Mirror loss enabled or not
            kl_thresh (float): threshold for max kl divergence
            verbose (bool): verbose logging output
        """
        model_info = {}
        policy_info = {}

        start = time()
        model_param_id = ray.put(list(self.model.get_all_parameters()))
        actor_param_id  = ray.put(list(self.actor.parameters()))
        critic_param_id = ray.put(list(self.critic.parameters()))
        norm_id = ray.put([self.model.inference_model.welford_state_mean, self.model.inference_model.welford_state_mean_diff, self.model.inference_model.welford_state_n])

        env_samples_per_worker      = max(config.num_env_steps // len(self.model_samplers), config.env_max_traj_len)

        if config.verbose:
            print("\t{:5.4f}s to copy model & policy params to workers.".format(time() - start))

        # Evaluate model & policy.
        # Only model for now.

        eval_info_of_workers = ray.get([env_sampler.evaluate_model.remote(trajs=config.evaluate_model_trajs, max_traj_len=config.env_max_traj_len) for env_sampler in self.env_samplers])
        eval_info_avg = {key: np.average([info[key] for info in eval_info_of_workers]) for key in eval_info_of_workers[0].keys()}
        model_info.update(eval_info_avg)

        torch.set_num_threads(1)

        # Train model.

        # Sync env_samplers and model_optimizer to current model & policy.
        for env_sampler in self.env_samplers:
            env_sampler.sync.remote(model_param_id, actor_param_id, critic_param_id, input_norm=norm_id)
        self.model_optimizer.sync.remote(model_param_id, actor_param_id, critic_param_id, input_norm=norm_id)

        # Collect env experience for model learning.
        env_sample_start_time = time()
        collection_output = ray.get([
            env_sampler.collect_experience.remote(max_traj_len=config.env_max_traj_len, min_steps=env_samples_per_worker) 
            for env_sampler in self.env_samplers])
        env_sample_buffers, total_steps = list(zip(*collection_output))
        total_steps = np.sum(total_steps)
        env_sample_buffer = Env_Sample_Buffer.merge_buffers(env_sample_buffers)
        # Delete buffers to free up memory? Might not be necessary
        del env_sample_buffers
        self.env_sample_buffer_queue.assimilate_buffer(env_sample_buffer, max_size=config.model_buffer_max_size)
        # TODO: make total_steps reflect the number of steps actually taken, not just recorded.
        num_env_steps = len(env_sample_buffer)
        env_sample_duration = time() - env_sample_start_time
        env_sample_rate = num_env_steps / env_sample_duration
        model_info['env_sample_rate'] = env_sample_rate
        model_info['num_env_samples'] = num_env_steps
        assert num_env_steps == np.sum(env_sample_buffer.are_real)
        model_info['proportion_env_samples'] = model_info['num_env_samples'] / max(1, total_steps)
        del env_sample_buffer

        # Update model with env_sample_buffer_queue and sync to algo.
        model_optimizer_info = ray.get(self.model_optimizer.optimize.remote(ray.put(self.env_sample_buffer_queue), epochs=config.model_epochs, batch_size=config.model_batch_size))
        model_info.update(model_optimizer_info)
        model_params, actor_params, critic_params, input_norm = ray.get(self.model_optimizer.retrieve_parameters.remote())
        self.sync(new_model_params=model_params, input_norm=input_norm)

        # Train policy.

        if config.learn_policy:
            # Sync model_samplers and policy_optimizer to current model & policy.
            for model_sampler in self.model_samplers:
                model_sampler.sync.remote(model_param_id, actor_param_id, critic_param_id, input_norm=norm_id)
            self.policy_optimizer.sync.remote(model_param_id, actor_param_id, critic_param_id, input_norm=norm_id)
            
            # Collect model experience for policy learning.
            # TODO: make this smarter, respect config.policy_batch_size.
            model_sample_start_time = time()
            start_states_generator = self.env_sample_buffer_queue.sample(batch_size=len(self.env_sample_buffer_queue)//len(self.model_samplers), recurrent=False)
            worker_start_states_generator = (next(start_states_generator)[0].reshape(-1, self.env.observation_size) for _ in range(len(self.model_samplers)))
            # worker_start_state_partitions = [???? for each in self.model_samplers]
            model_sample_buffers = ray.get([
                # model_sampler.collect_experience.remote(start_states=worker_start_state_partitions[i], max_traj_len=config.env_max_traj_len) 
                model_sampler.collect_experience.remote(start_states=next(worker_start_states_generator), max_traj_len=config.env_max_traj_len) 
                for i, model_sampler in enumerate(self.model_samplers)
            ])
            model_sample_buffer = Model_Sample_Buffer.merge_buffers(model_sample_buffers)
            total_model_steps = len(model_sample_buffer)
            model_sample_duration = time() - model_sample_start_time
            model_sample_rate = total_model_steps / model_sample_duration
            avg_batch_reward = np.mean(model_sample_buffer.ep_returns)

            # Update policy with model_sample_buffer and sync to algo.
            policy_optimizer_info = ray.get(self.policy_optimizer.optimize.remote(ray.put(model_sample_buffer), epochs=config.policy_epochs, batch_size=config.policy_batch_size))
            policy_info.update(policy_optimizer_info)
            model_params, actor_params, critic_params, input_norm = ray.get(self.policy_optimizer.retrieve_parameters.remote())
            self.sync(new_actor_params=actor_params, new_critic_params=critic_params, input_norm=input_norm)

        return total_steps, model_info, policy_info

    @classmethod
    def run(cls, config):
        
        # Set up env.

        # from .envs import get_env_factory
        # env_factory = get_env_factory(env_key=?)
        from .envs.cartpole_env import Cartpole_Env
        from functools import partial
        env_factory = partial(Cartpole_Env)
        env = env_factory()
        observation_size = env.observation_size
        action_size = env.action_size

        # Set up model ensemble.


        if hasattr(config, "previous_model") and config.previous_model is not None:
            # TODO: copy optimizer states also???
            model   = torch.load(os.path.join(config.previous_model, "model.pt"))
            print(f"loaded model from {config.previous_model}")
        else:
            model = Ensemble_Network(
                observation_size,
                action_size,
                config,
            )

        # Set up policy: actor and critic.
        class Rand_Policy:
            action_dim = 1
            def forward(self, *args, **kwargs):
                return torch.Tensor([np.random.random() * 2 - 1])
            def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
            def train(self, *args, **kwargs): pass
            def copy_normalizer_stats(self, *args, **kwargs): pass
            def parameters(self, *args, **kwargs): return []
        actor = Rand_Policy()
        critic = actor

        if config.learn_policy:
            # Disabling actual policy and critic
            policy_fixed_std = torch.ones(action_size)*torch.Tensor(config.policy_fixed_std).item() if np.array(config.policy_fixed_std).size == 1 else torch.Tensor(config.policy_fixed_std)
            assert policy_fixed_std.size(0) == action_size, policy_fixed_std.size(0)
            assert policy_fixed_std.dim() == 1
            # layers = [int(x) for x in config.policy_layers.split(',')]

            if hasattr(config, "previous_policy") and config.previous_policy is not None:
                # TODO: copy optimizer states also???
                actor   = torch.load(os.path.join(config.previous_policy, "actor.pt"))
                critic  = torch.load(os.path.join(config.previous_policy, "critic.pt"))
                print(f"loaded policy from {config.previous_policy}")
            else:
                if config.policy_arch_key.lower() == 'lstm':
                    actor   = LSTM_Stochastic_Actor(observation_size, action_size, env_name=config.env, fixed_std=policy_fixed_std, bounded=False, layers=config.policy_layers)
                    critic  = LSTM_V(observation_size, layers=config.policy_layers)
                elif config.policy_arch_key.lower() == 'gru':
                    actor   = GRU_Stochastic_Actor(observation_size, action_size, env_name=config.env, fixed_std=policy_fixed_std, bounded=False, layers=config.policy_layers)
                    critic  = GRU_V(observation_size, layers=config.policy_layers)
                elif config.policy_arch_key.lower() == 'ff':
                    actor   = FF_Stochastic_Actor(observation_size, action_size, env_name=config.env, fixed_std=policy_fixed_std, bounded=False, layers=config.policy_layers)
                    critic  = FF_V(observation_size, layers=config.policy_layers)
                else:
                    raise ValueError(f"arch config.policy_arch_key not recognized: {config.policy_arch_key}")
        
        # Prenormalization
        if config.do_prenorm:
            print(f"Collecting normalization statistics with {config.prenormalize_steps} states...")
            train_model_normalizer(env_factory, model, actor, min_timesteps=config.prenormalize_steps, max_traj_len=config.env_max_traj_len, noise=1)
            actor.copy_normalizer_stats(model.inference_model)
            critic.copy_normalizer_stats(actor)
            # model.propagate('copy_normalizer_stats', actor)

        model.train(True)
        actor.train(True)
        critic.train(True)

        # if args.wandb:
        #     import wandb
        #     wandb.init(group = config.run_name, project=config.wandb_project_name, config=config, sync_tensorboard=True)

        algo = cls(model, actor, critic, env_factory, config)

        # create a tensorboard logging object
        from .util import create_logger
        if not config.no_log:
            logger = create_logger(config)
        else:
            logger = None

        if not config.no_log:
            config.save_model = os.path.join(logger.dir, 'model.pt')
            config.save_actor = os.path.join(logger.dir, 'actor.pt')
            config.save_critic = os.path.join(logger.dir, 'critic.pt')

        print()
        print("Cherry Picking:")
        print("\trun_name:           {}".format(config.run_name))
        print("\tseed:               {}".format(config.seed))
        print("\tenv:                {}".format(config.env))
        # print("\ttimesteps:          {:n}".format(int(config.timesteps)))
        # print("\titeration steps:    {:n}".format(int(config.num_steps)))
        # print("\tprenormalize steps: {}".format(int(config.prenormalize_steps)))
        # print("\ttraj_len:           {}".format(config.traj_len))
        # print("\tdiscount:           {}".format(config.discount))
        # print("\tactor_lr:           {}".format(config.a_lr))
        # print("\tcritic_lr:          {}".format(config.c_lr))
        # print("\tadam eps:           {}".format(config.eps))
        # print("\tentropy coeff:      {}".format(config.entropy_coeff))
        # print("\tgrad clip:          {}".format(config.grad_clip))
        # print("\tbatch size:         {}".format(config.batch_size))
        # print("\tepochs:             {}".format(config.epochs))
        print("\tworkers:            {}".format(config.workers))
        print()

        itr = 0
        timesteps = 0
        best_model_loss = None
        best_policy_loss = None
        cumulative_env_samples = 0
        while timesteps < config.timesteps:
            steps, model_info, policy_info = algo.do_iteration(config)

            timesteps += steps
            print(f"iter {itr:4d}")
            for key, val in model_info.items(): print(f"{key}: {val:5.4f} | ", end='')
            # print(f"iter {itr:4d} | return: {info['eval_reward']:5.2f} | KL {info['kl']:5.4f} | Actor loss {info['actor_loss']:5.4f} | Critic loss {info['critic_loss']:5.4f} | Model loss {info['model_loss']:5.4f} | ", end='')
            print("timesteps {:n}".format(timesteps))

            if not config.no_log and (best_model_loss is None or model_info['avg_loss'] < best_model_loss):
                print("\t(best model so far! saving to {})".format(config.save_model))
                best_model_loss = model_info['avg_loss']
                torch.save(algo.model, config.save_model)
            if config.learn_policy and not config.no_log and (best_policy_loss is None or policy_info['avg_loss'] < best_policy_loss):
                print("\t(best policy so far! saving to {} and {})".format(config.save_actor, config.save_critic))
                if config.save_actor is not None:
                    torch.save(algo.actor, config.save_actor)
                if config.save_critic is not None:
                    torch.save(algo.critic, config.save_critic)

            if logger is not None:
                # TODO: make info dicts recursive.
                for subtitle, info in zip(['model', 'policy'], [model_info, policy_info]):
                    for key, value in info.items():
                        logger.add_scalar(f"{subtitle}/{key}", value, itr)
                cumulative_env_samples += model_info['num_env_samples']
                logger.add_scalar(f"model/cumulative_env_samples", cumulative_env_samples, itr)

            if config.model_loss_goal is not None and model_info['avg_loss'] < config.model_loss_goal:
                print(f"Terminating on model_loss condition: \nmodel avg_loss {model_info['avg_loss']:5.4f} < {config.model_loss_goal:5.4f}")
                break

            itr += 1
        print(f"Finished ({timesteps} of {config.timesteps}).")

        # if config.wandb:
            # wandb.join()
