"""
GPU-based Replay Buffer for Soft Actor-Critic (SAC)

This module provides a replay buffer implementation that stores transitions
directly on the GPU using PyTorch tensors for efficient sampling during training.

Whole class created by Opus 4.5. With Prompt: "pls create a replay buffer implementation
that can store values from an gymnasium instance via pytorch in the gpu. The replaybuffer
should be used for an soft actor critic system."
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
import gymnasium as gym


class ReplayBuffer:
    """
    A GPU-based replay buffer for storing and sampling transitions.
    
    Stores transitions (state, action, reward, next_state, done) directly on GPU
    memory using PyTorch tensors. Uses a circular buffer approach for memory efficiency.
    
    Attributes:
        capacity (int): Maximum number of transitions to store
        device (torch.device): Device to store tensors on (cuda/cpu)
        state_dim (int): Dimension of the state space
        action_dim (int): Dimension of the action space
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space  
            device: Device to store tensors on. If None, uses CUDA if available,
                   otherwise falls back to CPU
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Current position in buffer and size
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate GPU tensors for the buffer
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
    
    @classmethod
    def from_env(
        cls,
        env: gym.Env,
        capacity: int,
        device: Optional[Union[str, torch.device]] = None
    ) -> "ReplayBuffer":
        """
        Create a replay buffer from a Gymnasium environment.
        
        Automatically infers state and action dimensions from the environment.
        
        Args:
            env: Gymnasium environment
            capacity: Maximum number of transitions to store
            device: Device to store tensors on
            
        Returns:
            ReplayBuffer instance configured for the environment
        """
        # Get state dimension
        if isinstance(env.observation_space, gym.spaces.Box):
            state_dim = int(np.prod(env.observation_space.shape))
        elif isinstance(env.observation_space, gym.spaces.Discrete):
            state_dim = 1
        else:
            raise ValueError(f"Unsupported observation space: {type(env.observation_space)}")
        
        # Get action dimension
        if isinstance(env.action_space, gym.spaces.Box):
            action_dim = int(np.prod(env.action_space.shape))
        elif isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = 1
        else:
            raise ValueError(f"Unsupported action space: {type(env.action_space)}")
        
        return cls(capacity, state_dim, action_dim, device)
    
    def add(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        reward: float,
        next_state: Union[np.ndarray, torch.Tensor],
        done: bool
    ) -> None:
        """
        Add a single transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after taking action
            done: Whether the episode terminated
        """
        # Convert to tensors if necessary and move to device
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state).float()
        
        # Flatten if needed
        state = state.flatten()
        action = action.flatten()
        next_state = next_state.flatten()
        
        # Store transition
        self.states[self.ptr] = state.to(self.device)
        self.actions[self.ptr] = action.to(self.device)
        self.rewards[self.ptr, 0] = reward
        self.next_states[self.ptr] = next_state.to(self.device)
        self.dones[self.ptr, 0] = float(done)
        
        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def add_batch(
        self,
        states: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
        rewards: Union[np.ndarray, torch.Tensor],
        next_states: Union[np.ndarray, torch.Tensor],
        dones: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """
        Add a batch of transitions to the buffer.
        
        More efficient than adding transitions one by one when collecting
        from vectorized environments.
        
        Args:
            states: Batch of current states (batch_size, state_dim)
            actions: Batch of actions (batch_size, action_dim)
            rewards: Batch of rewards (batch_size,) or (batch_size, 1)
            next_states: Batch of next states (batch_size, state_dim)
            dones: Batch of done flags (batch_size,) or (batch_size, 1)
        """
        # Convert to tensors
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()
        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards).float()
        if isinstance(next_states, np.ndarray):
            next_states = torch.from_numpy(next_states).float()
        if isinstance(dones, np.ndarray):
            dones = torch.from_numpy(dones).float()
        
        batch_size = states.shape[0]
        
        # Reshape rewards and dones if needed
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        # Flatten state and action dimensions
        states = states.reshape(batch_size, -1)
        actions = actions.reshape(batch_size, -1)
        next_states = next_states.reshape(batch_size, -1)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Calculate indices for circular buffer
        if self.ptr + batch_size <= self.capacity:
            # Simple case: no wraparound
            self.states[self.ptr:self.ptr + batch_size] = states
            self.actions[self.ptr:self.ptr + batch_size] = actions
            self.rewards[self.ptr:self.ptr + batch_size] = rewards
            self.next_states[self.ptr:self.ptr + batch_size] = next_states
            self.dones[self.ptr:self.ptr + batch_size] = dones
        else:
            # Wraparound case
            first_part = self.capacity - self.ptr
            second_part = batch_size - first_part
            
            self.states[self.ptr:] = states[:first_part]
            self.states[:second_part] = states[first_part:]
            
            self.actions[self.ptr:] = actions[:first_part]
            self.actions[:second_part] = actions[first_part:]
            
            self.rewards[self.ptr:] = rewards[:first_part]
            self.rewards[:second_part] = rewards[first_part:]
            
            self.next_states[self.ptr:] = next_states[:first_part]
            self.next_states[:second_part] = next_states[first_part:]
            
            self.dones[self.ptr:] = dones[:first_part]
            self.dones[:second_part] = dones[first_part:]
        
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
    
    def sample(
        self,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            All tensors are on the configured device with shape (batch_size, dim)
        """
        # Generate random indices
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def sample_with_indices(
        self,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of transitions and return the indices.
        
        Useful for algorithms like Prioritized Experience Replay where
        you need to update priorities after training.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices)
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices
        )
    
    def __len__(self) -> int:
        """Return the current number of transitions in the buffer."""
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if the buffer has enough samples for training."""
        return self.size >= batch_size
    
    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self.ptr = 0
        self.size = 0
        # Reset tensors to zero
        self.states.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.next_states.zero_()
        self.dones.zero_()
    
    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all transitions currently in the buffer.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            with actual buffer size (not capacity)
        """
        return (
            self.states[:self.size],
            self.actions[:self.size],
            self.rewards[:self.size],
            self.next_states[:self.size],
            self.dones[:self.size]
        )
    
    def save(self, path: str) -> None:
        """
        Save the buffer to disk.
        
        Args:
            path: Path to save the buffer to
        """
        torch.save({
            'capacity': self.capacity,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'ptr': self.ptr,
            'size': self.size,
            'states': self.states.cpu(),
            'actions': self.actions.cpu(),
            'rewards': self.rewards.cpu(),
            'next_states': self.next_states.cpu(),
            'dones': self.dones.cpu(),
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[Union[str, torch.device]] = None) -> "ReplayBuffer":
        """
        Load a buffer from disk.
        
        Args:
            path: Path to load the buffer from
            device: Device to load tensors to. If None, uses CUDA if available
            
        Returns:
            ReplayBuffer instance with loaded data
        """
        data = torch.load(path, weights_only=False)
        
        buffer = cls(
            capacity=data['capacity'],
            state_dim=data['state_dim'],
            action_dim=data['action_dim'],
            device=device
        )
        
        buffer.ptr = data['ptr']
        buffer.size = data['size']
        buffer.states = data['states'].to(buffer.device)
        buffer.actions = data['actions'].to(buffer.device)
        buffer.rewards = data['rewards'].to(buffer.device)
        buffer.next_states = data['next_states'].to(buffer.device)
        buffer.dones = data['dones'].to(buffer.device)
        
        return buffer
    
    @property
    def memory_usage_mb(self) -> float:
        """Return the approximate GPU memory usage in MB."""
        total_elements = (
            self.states.numel() +
            self.actions.numel() +
            self.rewards.numel() +
            self.next_states.numel() +
            self.dones.numel()
        )
        # float32 = 4 bytes
        return (total_elements * 4) / (1024 * 1024)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer for SAC.
    
    Extends the base ReplayBuffer with priority-based sampling,
    which can improve sample efficiency by replaying important
    transitions more frequently.
    
    Reference: Schaul et al. "Prioritized Experience Replay" (2015)
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Amount to increase beta each sampling step
            epsilon: Small constant to ensure non-zero priorities
            device: Device to store tensors on
        """
        super().__init__(capacity, state_dim, action_dim, device)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Priority storage (kept on CPU for efficient updates)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        reward: float,
        next_state: Union[np.ndarray, torch.Tensor],
        done: bool,
        priority: Optional[float] = None
    ) -> None:
        """
        Add a transition with priority.
        
        New transitions are added with max priority to ensure they get
        sampled at least once.
        
        Args:
            state, action, reward, next_state, done: Transition data
            priority: Optional priority value. If None, uses max priority
        """
        # Store priority before adding (to get correct index)
        if priority is None:
            priority = self.max_priority
        
        self.priorities[self.ptr] = priority
        
        # Call parent add method
        super().add(state, action, reward, next_state, done)
    
    def sample(
        self,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch based on priorities.
        
        Returns transitions along with importance sampling weights.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
            weights can be used to correct for the bias introduced by prioritized sampling
        """
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(self.size, size=batch_size, replace=False, p=probs)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        weights = torch.from_numpy(weights).float().to(self.device).unsqueeze(1)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        indices_tensor = torch.from_numpy(indices).long().to(self.device)
        
        return (
            self.states[indices_tensor],
            self.actions[indices_tensor],
            self.rewards[indices_tensor],
            self.next_states[indices_tensor],
            self.dones[indices_tensor],
            weights,
            indices_tensor
        )
    
    def update_priorities(
        self,
        indices: Union[np.ndarray, torch.Tensor],
        priorities: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """
        Update priorities for sampled transitions.
        
        Call this after computing TD errors during training.
        
        Args:
            indices: Indices of transitions to update
            priorities: New priority values (typically |TD error| + epsilon)
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.cpu().numpy()
        
        priorities = np.abs(priorities.flatten()) + self.epsilon
        
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
