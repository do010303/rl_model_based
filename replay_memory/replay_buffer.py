"""
Experience Replay Buffer for Reinforcement Learning
"""

import numpy as np
from typing import Dict, Any
import random
from collections import deque

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    
    Features:
    - Circular buffer with fixed capacity
    - Efficient random sampling
    - Support for goal-conditioned RL (HER)
    """
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool, **kwargs):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            **kwargs: Additional fields (e.g., goals for HER)
        """
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            **kwargs
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary containing batched transitions
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer. Have {len(self.buffer)}, need {batch_size}")
        
        batch = random.sample(self.buffer, batch_size)
        
        # Convert to numpy arrays
        states = np.array([t['state'] for t in batch])
        actions = np.array([t['action'] for t in batch])
        rewards = np.array([t['reward'] for t in batch])
        next_states = np.array([t['next_state'] for t in batch])
        dones = np.array([t['done'] for t in batch], dtype=np.float32)
        
        result = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
        
        # Add any additional fields
        if batch and len(batch[0]) > 5:
            for key in batch[0].keys():
                if key not in result:
                    result[key] = np.array([t[key] for t in batch])
        
        return result
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.position = 0
    
    def get_all_transitions(self) -> list:
        """Get all transitions in the buffer."""
        return list(self.buffer)

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer.
    
    Samples transitions based on their TD-error for more efficient learning.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer capacity
            alpha: Prioritization exponent
            beta: Importance sampling exponent  
            beta_increment: Beta increment per sampling step
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        # Priority storage
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool, priority: float = None, **kwargs):
        """Add transition with priority."""
        super().add(state, action, reward, next_state, done, **kwargs)
        
        if priority is None:
            priority = self.max_priority
        
        if len(self.priorities) < self.capacity:
            self.priorities.append(priority)
        else:
            self.priorities[self.position - 1] = priority
        
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch with prioritized sampling."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer. Have {len(self.buffer)}, need {batch_size}")
        
        # Calculate sampling probabilities
        priorities = np.array(list(self.priorities))
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Get transitions
        batch = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Convert to numpy arrays
        states = np.array([t['state'] for t in batch])
        actions = np.array([t['action'] for t in batch])
        rewards = np.array([t['reward'] for t in batch])
        next_states = np.array([t['next_state'] for t in batch])
        dones = np.array([t['done'] for t in batch], dtype=np.float32)
        
        result = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'indices': indices,
            'weights': weights.astype(np.float32)
        }
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return result
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for given indices."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
