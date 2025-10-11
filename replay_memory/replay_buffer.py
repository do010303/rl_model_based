"""
Experience Replay Buffer for Reinforcement Learning
"""

import numpy as np
from typing import Dict, Any
import random
from collections import deque

class ReplayBuffer:
    def save(self, filename: str):
        """Save the replay buffer to a file using pickle."""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({
                'capacity': self.capacity,
                'buffer': list(self.buffer),
                'position': self.position
            }, f)

    def load(self, filename: str):
        """Load the replay buffer from a file using pickle."""
        import pickle
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.capacity = data['capacity']
            self.buffer = deque(data['buffer'], maxlen=self.capacity)
            self.position = data['position']
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
            
            # Trigger cleanup when buffer reaches 90% capacity
            if len(self.buffer) >= int(0.9 * self.capacity):
                print(f"🧹 Auto-cleanup triggered at {len(self.buffer)}/{self.capacity} ({len(self.buffer)/self.capacity*100:.1f}%)")
                self.cleanup_competitive_buffer(keep_ratio=0.5, episode=0)
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
    
    def cleanup_old_samples(self, keep_ratio: float = 0.5):
        """
        Keep only the most recent samples to free up space.
        
        Args:
            keep_ratio: Ratio of samples to keep (0.5 = keep 50% most recent)
        """
        if len(self.buffer) == 0:
            return
            
        current_size = len(self.buffer)
        keep_count = int(current_size * keep_ratio)
        
        # Convert deque to list for slicing
        buffer_list = list(self.buffer)
        
        # Keep only the most recent samples
        recent_samples = buffer_list[-keep_count:]
        
        # Clear and refill with recent samples
        self.buffer.clear()
        for sample in recent_samples:
            self.buffer.append(sample)
        
        # Reset position
        self.position = len(self.buffer)
        
        # Update priorities if using PER
        if hasattr(self, 'priorities'):
            old_priorities = self.priorities.copy()
            self.priorities = np.zeros(self.capacity)
            # Keep priorities for recent samples
            self.priorities[:keep_count] = old_priorities[-keep_count:]

    def cleanup_competitive_buffer(self, keep_ratio: float = 0.5, episode: int = 0):
        """
        Success-prioritized buffer cleanup: Keeps successful experiences first, then high rewards.
        
        Args:
            keep_ratio: Ratio of buffer to keep (e.g., 0.5 = keep 50%)
            episode: Current episode for tracking improvement
        """
        if len(self.buffer) == 0:
            return
            
        current_size = len(self.buffer)
        target_size = int(current_size * keep_ratio)
        
        if target_size >= current_size:
            return
            
        print(f"🏆 Success-prioritized cleanup (Episode {episode}): {current_size} -> {target_size}")
        
        # Convert to list for faster indexing
        experiences = list(self.buffer)
        
        # Extract rewards and check for success indicators
        rewards = np.array([exp['reward'] for exp in experiences])
        
        # Identify successful experiences (high reward indicates success)
        # Based on reward structure: 100+ = acceptable precision (2cm), 200+ = excellent, 500+ = perfect
        success_threshold = 100.0  # Keep experiences with acceptable precision or better
        successful_mask = rewards >= success_threshold
        successful_indices = np.where(successful_mask)[0]
        
        print(f"   Found {len(successful_indices)} successful experiences ({len(successful_indices)/current_size*100:.1f}%)")
        
        # Strategy: SUCCESS FIRST, then high rewards, minimal recent exploration
        keep_indices = set()
        
        # Phase 1: ALWAYS keep ALL successful experiences (highest priority)
        if len(successful_indices) > 0:
            success_kept = min(len(successful_indices), int(target_size * 0.8))  # Up to 80% can be successful
            # Sort successful experiences by reward (best successes first)
            success_rewards = rewards[successful_indices]
            sorted_success_idx = successful_indices[np.argsort(success_rewards)[::-1]]
            keep_indices.update(sorted_success_idx[:success_kept])
            print(f"   Phase 1 - Kept {len(keep_indices)} successful experiences")
        
        # Phase 2: Fill remaining slots with highest reward experiences (non-successful)
        remaining_slots = target_size - len(keep_indices)
        if remaining_slots > 0:
            # Get non-successful experiences
            non_success_mask = ~successful_mask
            non_success_indices = np.where(non_success_mask)[0]
            
            if len(non_success_indices) > 0:
                # Sort by reward (highest first)
                non_success_rewards = rewards[non_success_indices]
                sorted_non_success_idx = non_success_indices[np.argsort(non_success_rewards)[::-1]]
                
                # Take top rewards to fill remaining slots
                high_reward_count = min(remaining_slots, len(sorted_non_success_idx))
                keep_indices.update(sorted_non_success_idx[:high_reward_count])
                print(f"   Phase 2 - Added {high_reward_count} high-reward non-successful experiences")
        
        # Phase 3: If still need more, add very recent experiences (last 5% of buffer)
        remaining_slots = target_size - len(keep_indices)
        if remaining_slots > 0:
            recent_start = max(0, current_size - max(10, int(current_size * 0.05)))
            recent_indices = list(range(recent_start, current_size))
            recent_indices = [i for i in recent_indices if i not in keep_indices]
            
            recent_added = min(remaining_slots, len(recent_indices))
            keep_indices.update(recent_indices[-recent_added:])
            print(f"   Phase 3 - Added {recent_added} recent exploration experiences")
        
        # Build final selection
        final_indices = sorted(list(keep_indices))[:target_size]
        kept_experiences = [experiences[i] for i in final_indices]
        
        # Statistics
        kept_rewards = np.array([exp['reward'] for exp in kept_experiences])
        kept_successes = len([r for r in kept_rewards if r >= success_threshold])
        avg_reward = np.mean(kept_rewards)
        max_reward = np.max(kept_rewards)
        success_rate = kept_successes / len(kept_experiences) * 100
        
        print(f"   Results: {kept_successes} successes ({success_rate:.1f}%), Avg: {avg_reward:.1f}, Max: {max_reward:.1f}")
        
        # Rebuild buffer efficiently
        self.buffer.clear()
        for exp in kept_experiences:
            self.buffer.append(exp)
        
        # Reset position
        self.position = len(self.buffer)
        
        # Update priorities if using PER - SUCCESS gets highest priority
        if hasattr(self, 'priorities'):
            self.priorities = np.zeros(self.capacity)
            for i, exp in enumerate(kept_experiences):
                reward = exp['reward']
                if reward >= success_threshold:
                    self.priorities[i] = 1.0  # Maximum priority for successful experiences
                elif reward >= np.percentile(kept_rewards, 90):
                    self.priorities[i] = 0.8  # High priority for top 10% rewards
                elif reward >= np.percentile(kept_rewards, 75):
                    self.priorities[i] = 0.6  # Medium-high for top 25%
                elif reward >= np.percentile(kept_rewards, 50):
                    self.priorities[i] = 0.4  # Medium for above median
                else:
                    self.priorities[i] = 0.2  # Lower priority for below median

    def cleanup_by_progressive_success(self, keep_ratio: float = 0.5, episode: int = 0, total_episodes: int = 1000):
        """
        Legacy progressive method - redirects to competitive method for better performance.
        """
        print(f"🔄 Using competitive buffer cleanup (episode {episode})...")
        self.cleanup_competitive_buffer(keep_ratio, episode)

    def cleanup_by_success_rate(self, keep_ratio: float = 0.5, min_success_ratio: float = 0.3):
        """
        Legacy method for backward compatibility - uses fixed thresholds.
        """
        current_size = len(self.buffer)
        # Call progressive method with default episode settings
        self.cleanup_by_progressive_success(keep_ratio, episode=100, total_episodes=1000)
        print(f"🧽 Legacy cleanup: {current_size} -> {len(self.buffer)} samples")

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
    
    def clear(self):
        """Clear all samples from the buffer."""
        self.buffer.clear()
        self.position = 0
        if hasattr(self, 'priorities'):
            self.priorities = np.zeros(self.capacity)
            self.max_priority = 1.0
        print(f"🧹 Buffer cleared completely")
