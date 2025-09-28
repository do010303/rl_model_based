"""
Hindsight Experience Replay (HER) Implementation
"""

import numpy as np
from typing import Dict, List, Any, Optional
from replay_memory.replay_buffer import ReplayBuffer

class HER:
    """
    Hindsight Experience Replay for goal-conditioned reinforcement learning.
    
    HER allows agents to learn from failed attempts by relabeling the goals
    in stored experiences, dramatically improving sample efficiency for
    sparse reward environments.
    """
    
    def __init__(self, replay_buffer: ReplayBuffer, k: int = 4, strategy: str = 'future'):
        """
        Initialize HER.
        
        Args:
            replay_buffer: Replay buffer to store experiences
            k: Number of additional goals to sample per transition
            strategy: Goal selection strategy ('future', 'episode', 'random')
        """
        self.replay_buffer = replay_buffer
        self.k = k
        self.strategy = strategy
        
        # Validate strategy
        valid_strategies = ['future', 'episode', 'random']
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}, got {strategy}")
    
    def store_episode(self, episode_data: Dict[str, np.ndarray]) -> None:
        """
        Store an episode with HER goal relabeling.
        
        Args:
            episode_data: Dictionary containing episode trajectory
                - states: Array of states
                - actions: Array of actions  
                - rewards: Array of rewards
                - next_states: Array of next states
                - dones: Array of done flags
                - achieved_goals: Array of achieved goals (end-effector positions)
                - desired_goals: Array of desired goals (target positions)
        """
        episode_length = len(episode_data['states'])
        
        # Store original episode
        for t in range(episode_length):
            self.replay_buffer.add(
                state=episode_data['states'][t],
                action=episode_data['actions'][t],
                reward=episode_data['rewards'][t],
                next_state=episode_data['next_states'][t],
                done=episode_data['dones'][t]
            )
        
        # Generate HER experiences
        for t in range(episode_length):
            # Sample k additional goals according to strategy
            sampled_goals = self._sample_goals(episode_data, t)
            
            for new_goal in sampled_goals:
                # Relabel reward based on new goal
                new_reward = self._compute_reward(
                    achieved_goal=episode_data['achieved_goals'][t] if t < len(episode_data.get('achieved_goals', [])) else episode_data['states'][t][-3:],
                    desired_goal=new_goal,
                    sparse=True
                )
                
                # Create new transition with relabeled goal
                new_state = self._replace_goal_in_obs(episode_data['states'][t], new_goal)
                new_next_state = self._replace_goal_in_obs(episode_data['next_states'][t], new_goal)
                
                # Check if episode should terminate with new goal
                distance_to_new_goal = np.linalg.norm(
                    episode_data['achieved_goals'][t] if t < len(episode_data.get('achieved_goals', [])) else episode_data['states'][t][-3:] - new_goal
                )
                new_done = distance_to_new_goal < 0.05  # Success threshold
                
                self.replay_buffer.add(
                    state=new_state,
                    action=episode_data['actions'][t],
                    reward=new_reward,
                    next_state=new_next_state,
                    done=new_done
                )
    
    def _sample_goals(self, episode_data: Dict[str, np.ndarray], t: int) -> List[np.ndarray]:
        """
        Sample goals according to the specified strategy.
        
        Args:
            episode_data: Episode trajectory data
            t: Current timestep
            
        Returns:
            List of sampled goals
        """
        episode_length = len(episode_data['states'])
        achieved_goals = episode_data.get('achieved_goals', 
                                        [state[-3:] for state in episode_data['states']])
        
        sampled_goals = []
        
        for _ in range(self.k):
            if self.strategy == 'future':
                # Sample from future timesteps in the same episode
                if t < episode_length - 1:
                    future_t = np.random.randint(t + 1, episode_length)
                    goal = achieved_goals[future_t] if future_t < len(achieved_goals) else achieved_goals[-1]
                else:
                    goal = achieved_goals[-1]
                    
            elif self.strategy == 'episode':
                # Sample from any timestep in the episode
                sample_t = np.random.randint(0, episode_length)
                goal = achieved_goals[sample_t] if sample_t < len(achieved_goals) else achieved_goals[-1]
                
            elif self.strategy == 'random':
                # Sample random goals from workspace
                goal = self._sample_random_goal()
            
            sampled_goals.append(goal)
        
        return sampled_goals
    
    def _sample_random_goal(self) -> np.ndarray:
        """Sample a random goal from the workspace."""
        # Sample in cylindrical coordinates for better distribution
        radius = np.random.uniform(0.2, 0.8)
        angle = np.random.uniform(0, 2 * np.pi)
        height = np.random.uniform(0.1, 0.8)
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        
        return np.array([x, y, z])
    
    def _compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, 
                       sparse: bool = True, distance_threshold: float = 0.05) -> float:
        """
        Compute reward based on achieved and desired goals.
        
        Args:
            achieved_goal: Position achieved by the robot
            desired_goal: Target position
            sparse: Whether to use sparse rewards
            distance_threshold: Success distance threshold
            
        Returns:
            Computed reward
        """
        distance = np.linalg.norm(achieved_goal - desired_goal)
        
        if sparse:
            # Sparse reward: +1 for success, -1 for failure
            return 1.0 if distance < distance_threshold else -1.0
        else:
            # Dense reward: negative distance with success bonus
            reward = -distance
            if distance < distance_threshold:
                reward += 10.0
            return reward
    
    def _replace_goal_in_obs(self, obs: np.ndarray, new_goal: np.ndarray) -> np.ndarray:
        """
        Replace goal in observation.
        
        Assumes observation format: [joint_pos(4), joint_vel(4), end_effector_pos(3), target_pos(3)]
        
        Args:
            obs: Original observation
            new_goal: New goal to insert
            
        Returns:
            Modified observation with new goal
        """
        new_obs = obs.copy()
        new_obs[-3:] = new_goal  # Replace last 3 elements (target position)
        return new_obs


class GoalConditionedReplayBuffer(ReplayBuffer):
    """
    Goal-conditioned replay buffer that stores goal information.
    
    Extends the basic replay buffer to handle goal-conditioned experiences
    used in HER and other goal-conditioned RL algorithms.
    """
    
    def __init__(self, capacity: int):
        """Initialize goal-conditioned replay buffer."""
        super().__init__(capacity)
    
    def add_goal_conditioned(self, state: np.ndarray, action: np.ndarray, reward: float,
                           next_state: np.ndarray, done: bool, achieved_goal: np.ndarray,
                           desired_goal: np.ndarray, **kwargs):
        """
        Add goal-conditioned transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            achieved_goal: Goal achieved in this step
            desired_goal: Target goal
            **kwargs: Additional fields
        """
        super().add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
            **kwargs
        )
    
    def sample_goal_conditioned(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch with goal information."""
        batch_data = super().sample(batch_size)
        
        # Ensure goal information is included
        if 'achieved_goal' not in batch_data:
            # Extract goals from states if not explicitly stored
            # Assumes observation format: [..., achieved_goal(3), desired_goal(3)]
            batch_data['achieved_goals'] = batch_data['states'][:, -6:-3]
            batch_data['desired_goals'] = batch_data['states'][:, -3:]
        
        return batch_data
