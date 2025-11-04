"""
Base Agent Interface for Reinforcement Learning Algorithms
"""

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple, Optional

class BaseAgent(ABC):
    """
    Abstract base class for all RL agents.
    
    This class defines the interface that all agents must implement,
    ensuring consistency across different algorithm implementations.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: Configuration dictionary with hyperparameters
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        self.total_reward = 0.0
        
    @abstractmethod
    def act(self, state: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """
        Select an action given the current state.
        
        Args:
            state: Current state observation
            add_noise: Whether to add exploration noise
            
        Returns:
            Action to take
        """
        pass
    
    @abstractmethod
    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Batch of experiences from replay buffer
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """
        Load the model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        pass
    
    def update_training_stats(self, reward: float, done: bool) -> None:
        """
        Update training statistics.
        
        Args:
            reward: Reward received
            done: Whether episode is done
        """
        self.total_reward += reward
        if done:
            self.episode_count += 1
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get current training statistics.
        
        Returns:
            Dictionary of training statistics
        """
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'total_reward': self.total_reward,
            'average_reward': self.total_reward / max(1, self.episode_count)
        }
    
    def reset_stats(self) -> None:
        """Reset training statistics."""
        self.training_step = 0
        self.episode_count = 0
        self.total_reward = 0.0
