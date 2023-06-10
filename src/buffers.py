import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence


class Policy_Buffer:
    """
    Generic Buffer class to hold samples for PPO. 
    Note: that it is assumed that trajectories are stored
    consecutively, next to each other. The list traj_idx stores the indices where individual trajectories are started.

    Args:
        discount (float): Discount factor 

    Attributes:
        states (list): List of stored sampled observation states
        actions (list): List of stored sampled actions
        rewards (list): List of stored sampled rewards
        values (list): List of stored sampled values
        returns (list): List of stored computed returns
        advantages (list): List of stored computed advantages
        ep_returns (list): List of trajectories returns (summed rewards over whole trajectory)
        ep_lens (list): List of trajectory lengths
        size (int): Number of currently stored states
        traj_idx (list): List of indices where individual trajectories start
        buffer_read (bool): Whether or not the buffer is ready to be used for optimization.
    """
    def __init__(self, discount=0.99):
        self.discount = discount
        self.clear()

    def __len__(self):
        return len(self.states)

    def clear(self):
        """
        Clear out/reset all buffer values. Should always be called before starting new sampling iteration
        """
        self.states     = []
        self.actions    = []
        self.rewards    = []
        self.values     = []
        self.returns    = []
        self.advantages = []

        self.ep_returns = []
        self.ep_lens = []

        self.size = 0

        self.traj_idx = [0]
        self.buffer_ready = False

    def push(self, state, action, reward, value, done=False):
        """
        Store new PPO state (state, action, reward, value, termination)

        Args:
            state (numpy vector):  observation
            action (numpy vector): policy action
            reward (numpy vector): reward
            value (numpy vector): value function value
            return (numpy vector): return
            done (bool): last mdp tuple in rollout
        """
        self.states  += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values  += [value]

        self.size += 1

    def end_trajectory(self, terminal_value=0):
        """
        Finish a stored trajectory, i.e. calculate return for each step by adding a termination value to the last state 
        and backing up return based on discount factor. 

        Args:
            terminal_value (float): Estimated value at the final state in the trajectory. Used to back up and calculate returns for the whole trajectory
        """
        self.traj_idx += [self.size]
        rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]

        returns = []

        R = terminal_value
        for reward in reversed(rewards):
            R = self.discount * R + reward
            returns.insert(0, R)

        self.returns += returns

        self.ep_returns += [np.sum(rewards)]
        self.ep_lens    += [len(rewards)]

    def _finish_buffer(self):
        """
        Get a buffer ready for optimization by turning each list into torch Tensor. Also calculate mirror states and normalized advantages. Must be called before 
        sampling from the buffer for optimization. While make "buffer_ready" variable true.

        Args:
            mirror (function pointer): Pointer to the state mirroring function that while mirror observation states
        """
        with torch.no_grad():
            self.states =  np.array(self.states)
            self.actions = np.array(self.actions)
            self.rewards = np.array(self.rewards)
            self.returns = np.array(self.returns)
            self.values  = np.array(self.values)

            self.states  = torch.Tensor(self.states)
            self.actions = torch.Tensor(self.actions)
            self.rewards = torch.Tensor(self.rewards)
            self.returns = torch.Tensor(self.returns)
            self.values  = torch.Tensor(self.values)

            # Calculate and normalize advantages
            a = self.returns - self.values
            a = (a - a.mean()) / (a.std() + 1e-4)
            self.advantages = a
            self.buffer_ready = True

    def sample(self, batch_size=64, recurrent=False):
        """
        Returns a randomly sampled batch from the buffer to be used for optimization. If "recurrent" is true, will return a random batch of trajectories to be used
        for backprop through time. Otherwise will return randomly selected states from the buffer

        Args:
            batch_size (int): Size of the batch. If recurrent is True then the number of trajectories to return. Otherwise is the number of states to return.
            recurrent (bool): Whether to return a recurrent batch (trajectories) or not
            mirror (function pointer): Pointer to the state mirroring function. If is None, the no mirroring will be done.
        """
        if not self.buffer_ready:
            self._finish_buffer()

        if recurrent:
            random_indices = SubsetRandomSampler(range(len(self.traj_idx)-1))
            sampler = BatchSampler(random_indices, batch_size, drop_last=False)

            for traj_indices in sampler:
                states     = [self.states[self.traj_idx[i]:self.traj_idx[i+1]]     for i in traj_indices]
                actions    = [self.actions[self.traj_idx[i]:self.traj_idx[i+1]]    for i in traj_indices]
                returns    = [self.returns[self.traj_idx[i]:self.traj_idx[i+1]]    for i in traj_indices]
                advantages = [self.advantages[self.traj_idx[i]:self.traj_idx[i+1]] for i in traj_indices]
                traj_mask  = [torch.ones_like(r) for r in returns]

                states     = pad_sequence(states,     batch_first=False)
                actions    = pad_sequence(actions,    batch_first=False)
                returns    = pad_sequence(returns,    batch_first=False)
                advantages = pad_sequence(advantages, batch_first=False)
                traj_mask  = pad_sequence(traj_mask,  batch_first=False)

                yield states, actions, returns, advantages, traj_mask

        else:
            random_indices = SubsetRandomSampler(range(self.size))
            sampler = BatchSampler(random_indices, batch_size, drop_last=True)

            for i, idxs in enumerate(sampler):
                states     = self.states[idxs]
                actions    = self.actions[idxs]
                returns    = self.returns[idxs]
                advantages = self.advantages[idxs]

                yield states, actions, returns, advantages, 1

    @classmethod
    def merge_buffers(cls, buffers):
        """
        Function to merge a list of buffers into a single Buffer object. Used for merging buffers received from multiple remote workers into a simple Buffer object to sample from

        Args:
            buffers (list): List of Buffer objects to merge

        Returns:
            A single Buffer object
        """
        assert cls is Policy_Buffer

        for buffer in buffers:
            assert isinstance(buffer, cls)

        memory = Policy_Buffer()

        for b in buffers:
            offset = len(memory)

            memory.states  += b.states
            memory.actions += b.actions
            memory.rewards += b.rewards
            memory.values  += b.values
            memory.returns += b.returns

            memory.ep_returns += b.ep_returns
            memory.ep_lens    += b.ep_lens

            memory.traj_idx += [offset + i for i in b.traj_idx[1:]]
            memory.size     += b.size

        return memory


"""
================================================================================
================================================================================
================================================================================
"""


class Model_Buffer:
    """
    Buffer class to hold samples for model learning in Cherry Picking.
    Note that pushed values are expected to be Tensors and must retain their gradients.

    Note: that it is assumed that trajectories are stored
    consecutively, next to each other. The list traj_idx stores the indices where individual trajectories are started.

    Args:
        discount (float): Discount factor 

    Attributes:
        states (list): List of stored sampled observation states
        actions (list): List of stored sampled actions
        rewards (list): List of stored sampled rewards
        values (list): List of stored sampled values
        returns (list): List of stored computed returns
        advantages (list): List of stored computed advantages
        ep_returns (list): List of trajectories returns (summed rewards over whole trajectory)
        ep_lens (list): List of trajectory lengths
        size (int): Number of currently stored states
        traj_idx (list): List of indices where individual trajectories start
        buffer_read (bool): Whether or not the buffer is ready to be used for optimization.
    """
    def __init__(self, discount=0.99):
        # self.discount = discount
        self.clear()
        self.ff_bootstrap_indices_map = dict() # Populated at sample time.
        self.rnn_bootstrap_indices_map = dict() # Populated at sample time.

    def __len__(self):
        return len(self.states)

    def clear(self):
        """
        Clear out/reset all buffer values. Should always be called before starting new sampling iteration
        """
        self.states      = []
        self.actions     = []
        self.is_real     = []

        self.ep_lens = []

        self.size = 0

        self.traj_idx = [0]
        self.buffer_ready = False

    def push(self, state, action, is_real):
        """
        Store new PPO state (state, action, reward, value, termination)

        Args:
            state (numpy vector):  observation
            action (numpy vector): policy action
            reward (numpy vector): reward
            value (numpy vector): value function value
            return (numpy vector): return
            done (bool): last mdp tuple in rollout
        """
        self.states += [state]
        self.actions += [action]
        self.are_real += [is_real]

        self.size += 1

    def end_trajectory(self):
        """
        Finish a stored trajectory, i.e. calculate return for each step by adding a termination value to the last state 
        and backing up return based on discount factor. 

        Args:
            terminal_value (float): Estimated value at the final state in the trajectory. Used to back up and calculate returns for the whole trajectory
        """
        self.traj_idx += [self.size]
        # next_states = self.next_states[self.traj_idx[-2]:self.traj_idx[-1]]

        self.ep_lens    += [self.traj_idx[-1] - self.traj_idx[-2]]

    def _finish_buffer(self):
        """
        Get a buffer ready for optimization by turning each list into torch Tensor. Also calculate mirror states and normalized advantages. Must be called before 
        sampling from the buffer for optimization. While make "buffer_ready" variable true.

        Args:
            mirror (function pointer): Pointer to the state mirroring function that while mirror observation states
        """
        self.buffer_ready = True

    def sample(self, ensemble_index=None, batch_size=64, recurrent=False):
        """
        Returns a randomly sampled batch from the buffer to be used for optimization. If "recurrent" is true, will return a random batch of trajectories to be used
        for backprop through time. Otherwise will return randomly selected states from the buffer

        Args:
            batch_size (int): Size of the batch. If recurrent is True then the number of trajectories to return. Otherwise is the number of states to return.
            recurrent (bool): Whether to return a recurrent batch (trajectories) or not
            mirror (function pointer): Pointer to the state mirroring function. If is None, the no mirroring will be done.
        """
        if not self.buffer_ready:
            self._finish_buffer()

        if recurrent: # (rnn), get random trajectories.
            num_trajectories = len(self.traj_idx)-1

            random_indices = SubsetRandomSampler(range(num_trajectories))
            sampler = BatchSampler(random_indices, batch_size, drop_last=False)

            # Bootstrap trajectories based on ensemble_index.
            if ensemble_index not in self.rnn_bootstrap_indices_map:
                if ensemble_index is None:
                    self.rnn_bootstrap_indices_map[ensemble_index] = np.arange(num_trajectories)
                else:
                    self.rnn_bootstrap_indices_map[ensemble_index] = np.random.randint(num_trajectories)
            rnn_bootstrap_indices = self.rnn_bootstrap_indices_map[ensemble_index]

            for traj_indices in sampler:
                bootstrapped_traj_indices = rnn_bootstrap_indices[traj_indices]

                states          = [self.states[    self.traj_idx[i]:self.traj_idx[i+1]] for i in bootstrapped_traj_indices]
                actions         = [self.actions[   self.traj_idx[i]:self.traj_idx[i+1]] for i in bootstrapped_traj_indices]
                are_real         = [self.are_real[   self.traj_idx[i]:self.traj_idx[i+1]] for i in bootstrapped_traj_indices]
                traj_mask       = [torch.ones_like(is_real) for is_real in are_real]

                states          = pad_sequence(states,         batch_first=False)
                actions         = pad_sequence(actions,             batch_first=False)
                are_real        = pad_sequence(are_real,    batch_first=False)
                traj_mask       = pad_sequence(traj_mask,           batch_first=False)

                yield states, actions, are_real, traj_mask

        else: # Not recurrent (ff), get random samples.
            num_samples = self.size
            
            random_indices = SubsetRandomSampler(range(num_samples))
            sampler = BatchSampler(random_indices, batch_size, drop_last=True)

            # Bootstrap samples based on ensemble_index.
            if ensemble_index not in self.ff_bootstrap_indices_map:
                if ensemble_index is None:
                    self.ff_bootstrap_indices_map[ensemble_index] = np.arange(num_samples)
                else:
                    self.ff_bootstrap_indices_map[ensemble_index] = np.random.randint(num_samples)
            ff_bootstrap_indices = self.ff_bootstrap_indices_map[ensemble_index]

            for sample_indices in sampler:
                bootstrapped_sample_indices = ff_bootstrap_indices[sample_indices]

                states         = self.states        [bootstrapped_sample_indices]
                actions             = self.actions  [bootstrapped_sample_indices]
                are_real            = self.are_real [bootstrapped_sample_indices]
                traj_mask           = 1

                yield states, actions, are_real, traj_mask

    @classmethod
    def merge_buffers(cls, buffers):
        """
        Function to merge a list of buffers into a single Buffer object. Used for merging buffers received from multiple remote workers into a simple Buffer object to sample from

        Args:
            buffers (list): List of Buffer objects to merge

        Returns:
            A single Buffer object
        """
        assert cls is Model_Buffer
        
        for buffer in buffers:
            assert isinstance(buffer, cls)

        memory = Model_Buffer()

        for b in buffers:
            offset = len(memory)

            memory.next_states          += b.next_states
            memory.next_state_preds     += b.next_state_preds
            memory.actions              += b.actions

            memory.ep_lens += b.ep_lens

            memory.traj_idx += [offset + i for i in b.traj_idx[1:]]
            memory.size     += b.size

        return memory