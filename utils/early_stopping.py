"""
Early Stopping Utility for RL Training
Prevents performance degradation by stopping when agent performance peaks.
"""

import numpy as np
from typing import List, Dict, Optional

class EarlyStopping:
    """
    Early stopping mechanism for reinforcement learning training.
    
    Monitors performance metrics and stops training when performance 
    starts consistently degrading to prevent catastrophic forgetting.
    """
    
    def __init__(self, 
                 patience: int = 50,
                 min_improvement: float = 0.01,
                 metric: str = 'success_rate',
                 restore_best: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of episodes to wait for improvement
            min_improvement: Minimum improvement to consider as progress
            metric: Metric to monitor ('success_rate', 'avg_reward', etc.)
            restore_best: Whether to restore best model when stopping
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.metric = metric
        self.restore_best = restore_best
        
        self.best_score = -np.inf
        self.best_episode = 0
        self.wait_count = 0
        self.stopped = False
        self.best_model_path = None
        
        self.history = []
        
    def __call__(self, current_score: float, episode: int, model_path: Optional[str] = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_score: Current performance score
            episode: Current episode number
            model_path: Path to current model (for restoration)
            
        Returns:
            True if training should stop, False otherwise
        """
        self.history.append(current_score)
        
        # Check if current score is the best so far
        if current_score > self.best_score + self.min_improvement:
            self.best_score = current_score
            self.best_episode = episode
            self.wait_count = 0
            if model_path:
                self.best_model_path = model_path
            print(f"🎯 New best {self.metric}: {current_score:.3f} at episode {episode}")
            
        else:
            self.wait_count += 1
            
        # Check if we should stop
        if self.wait_count >= self.patience:
            self.stopped = True
            print(f"\n⏹️ Early stopping triggered after {episode} episodes")
            print(f"   Best {self.metric}: {self.best_score:.3f} at episode {self.best_episode}")
            print(f"   No improvement for {self.patience} episodes")
            
            if self.restore_best and self.best_model_path:
                print(f"   🔄 Restoring best model from episode {self.best_episode}")
                
            return True
            
        return False
        
    def get_stats(self) -> Dict:
        """Get early stopping statistics."""
        return {
            'stopped': self.stopped,
            'best_score': self.best_score,
            'best_episode': self.best_episode,
            'total_wait': self.wait_count,
            'patience': self.patience,
            'improvement_threshold': self.min_improvement
        }
        
    def should_save_checkpoint(self, current_score: float) -> bool:
        """Check if current model should be saved as checkpoint."""
        return current_score >= self.best_score


class AdaptiveEarlyStopping(EarlyStopping):
    """
    Adaptive early stopping that adjusts patience based on training progress.
    """
    
    def __init__(self, 
                 initial_patience: int = 50,
                 max_patience: int = 200,
                 patience_increase: float = 1.2,
                 **kwargs):
        """
        Initialize adaptive early stopping.
        
        Args:
            initial_patience: Starting patience value
            max_patience: Maximum patience allowed
            patience_increase: Factor to increase patience when performance improves
        """
        super().__init__(patience=initial_patience, **kwargs)
        self.initial_patience = initial_patience
        self.max_patience = max_patience
        self.patience_increase = patience_increase
        
    def __call__(self, current_score: float, episode: int, model_path: Optional[str] = None) -> bool:
        """Adaptive early stopping with dynamic patience adjustment."""
        old_best = self.best_score
        should_stop = super().__call__(current_score, episode, model_path)
        
        # Increase patience if we found a new best (agent is still improving)
        if current_score > old_best + self.min_improvement:
            new_patience = min(int(self.patience * self.patience_increase), self.max_patience)
            if new_patience > self.patience:
                print(f"   📈 Increasing patience: {self.patience} → {new_patience}")
                self.patience = new_patience
                
        return should_stop


def calculate_performance_metrics(episode_rewards: List[float], 
                                success_history: List[bool], 
                                window_size: int = 100) -> Dict[str, float]:
    """
    Calculate performance metrics for early stopping decisions.
    
    Args:
        episode_rewards: List of episode rewards
        success_history: List of success flags
        window_size: Window size for rolling averages
        
    Returns:
        Dictionary of performance metrics
    """
    if len(episode_rewards) == 0:
        return {'avg_reward': 0.0, 'success_rate': 0.0, 'reward_stability': 0.0}
    
    # Recent performance (last window_size episodes)
    recent_rewards = episode_rewards[-window_size:]
    recent_success = success_history[-window_size:]
    
    avg_reward = np.mean(recent_rewards)
    success_rate = np.mean(recent_success) if recent_success else 0.0
    
    # Reward stability (lower std = more stable)
    reward_std = np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0
    reward_stability = 1.0 / (1.0 + reward_std)  # Higher is better
    
    # Combined score that balances reward and success rate
    combined_score = avg_reward * 0.7 + success_rate * 1000 * 0.3
    
    return {
        'avg_reward': avg_reward,
        'success_rate': success_rate,
        'reward_stability': reward_stability,
        'combined_score': combined_score,
        'episodes_evaluated': len(recent_rewards)
    }