#!/usr/bin/env python3
"""
ROS-Gazebo Environment Adapter for MBPO Integration

This adapter makes the ROS-Gazebo robot arm environment compatible with
the existing MBPO framework by implementing the gymnasium interface.
"""

import sys
import os
import numpy as np
import rospy
from gymnasium import spaces
import gymnasium as gym
from typing import Dict, Any, Tuple, Optional

# Import ROS environment
from robot_arm_env import RobotArmEnv

class RosGazeboAdapter(gym.Env):
    """
    Gymnasium adapter for ROS-Gazebo robot arm environment.
    
    This adapter bridges the ROS-Gazebo simulation with your existing 
    MBPO training framework by providing the standard gymnasium interface.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.config = config or {}
        
        # Initialize ROS node if not already initialized
        if not rospy.get_node_uri():
            rospy.init_node('ros_gazebo_adapter', anonymous=True)
        
        # Create ROS-Gazebo environment
        self.ros_env = RobotArmEnv(
            headless=self.config.get('headless', True),
            real_time=self.config.get('real_time', False)
        )
        
        # Wait for environment to be ready
        rospy.sleep(2.0)
        
        print(f"🤖 ROS-Gazebo Environment Initialized")
        print(f"   ROS Environment Dimensions:")
        print(f"   - Observations: {self.ros_env.n_observations}")
        print(f"   - Actions: {self.ros_env.n_actions}")
        
        # Define gymnasium spaces to match your existing MBPO framework
        # Action space: 4 joint commands (normalized to [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.ros_env.n_actions,), 
            dtype=np.float32
        )
        
        # Observation space: joint positions + velocities + target + end-effector position
        # This matches the observation structure expected by your Robot4DOFEnv
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.ros_env.n_observations,), 
            dtype=np.float32
        )
        
        print(f"   Gymnasium Spaces:")
        print(f"   - Action space: {self.action_space}")
        print(f"   - Observation space: {self.observation_space}")
        
        # Internal state tracking
        self._current_observation = None
        self._episode_step = 0
        self._max_episode_steps = self.config.get('max_steps', 200)
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.
        
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        if seed is not None:
            self.seed(seed)
        
        # Reset ROS environment
        observation = self.ros_env.reset()
        
        if observation is None:
            # Fallback to zero observation if reset fails
            print("⚠️ ROS environment reset failed, using fallback observation")
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        self._current_observation = np.array(observation, dtype=np.float32)
        self._episode_step = 0
        
        # Create info dict compatible with your MBPO framework
        info = {
            'episode_step': self._episode_step,
            'max_episode_steps': self._max_episode_steps
        }
        
        return self._current_observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action array (should be in [-1, 1] range)
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        # Ensure action is in correct format
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        
        # Take step in ROS environment
        next_observation, reward, done, info = self.ros_env.step(action)
        
        self._episode_step += 1
        
        # Handle invalid observations
        if next_observation is None:
            print("⚠️ Invalid observation from ROS environment")
            # Use previous observation as fallback
            next_observation = self._current_observation
            reward = -10.0  # Penalty for invalid step
            done = True
        
        self._current_observation = np.array(next_observation, dtype=np.float32)
        
        # Determine termination conditions
        terminated = done  # Episode ended due to success/failure
        truncated = self._episode_step >= self._max_episode_steps  # Time limit
        
        # Enhanced info dictionary for MBPO compatibility
        info.update({
            'episode_step': self._episode_step,
            'max_episode_steps': self._max_episode_steps,
            'TimeLimit.truncated': truncated and not terminated
        })
        
        return self._current_observation, reward, terminated, truncated, info
    
    def render(self, mode: str = 'human'):
        """Render the environment (if supported by ROS environment)."""
        try:
            return self.ros_env.render()
        except:
            pass  # Rendering not supported in headless mode
    
    def close(self):
        """Close the environment."""
        try:
            self.ros_env.cleanup()
            print("🧹 ROS-Gazebo environment cleaned up")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    
    def seed(self, seed: int = None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
        return [seed]
    
    # Properties to match your existing Robot4DOFEnv interface
    @property
    def unwrapped(self):
        return self
    
    def get_wrapper_attr(self, name):
        """Get attribute from wrapped environment."""
        return getattr(self.ros_env, name, None)


class ROSMBPOIntegration:
    """
    Direct integration with your existing MBPO framework.
    
    This class creates a ROS-compatible environment and plugs it directly
    into your existing MBPOTrainer without any modifications.
    """
    
    def __init__(self, env_config: Dict = None, agent_config: Dict = None):
        """Initialize ROS-MBPO integration."""
        
        # Add parent directory to path to import MBPO components
        parent_dir = '/home/ducanh/rl_model_based'
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Import your existing MBPO components
        from mbpo_trainer import MBPOTrainer
        
        # Default configurations
        self.env_config = env_config or {
            'headless': True,
            'real_time': False,
            'max_steps': 200,
            'success_distance': 0.02,
            'dense_reward': True
        }
        
        self.agent_config = agent_config or {
            'lr_actor': 0.0001,
            'lr_critic': 0.001,
            'gamma': 0.99,
            'tau': 0.005,
            'noise_std': 0.2,
            'noise_decay': 0.9995,
            'hidden_dims': [512, 256, 128]
        }
        
        print("🚀 Initializing ROS-MBPO Integration...")
        print(f"   Environment config: {self.env_config}")
        print(f"   Agent config: {self.agent_config}")
        
        # Create the adapter environment that your MBPO trainer will use
        self.create_adapted_environment()
        
        # Create MBPO trainer with ROS environment
        self.mbpo_trainer = self.create_mbpo_trainer()
        
    def create_adapted_environment(self):
        """Create ROS-Gazebo environment adapter."""
        print("🔧 Creating ROS-Gazebo environment adapter...")
        
        # Replace the Robot4DOFEnv import in your MBPO trainer temporarily
        # We'll monkey-patch it to use our ROS adapter
        import importlib
        import environments.robot_4dof_env
        
        # Store original Robot4DOFEnv
        self._original_env_class = environments.robot_4dof_env.Robot4DOFEnv
        
        # Replace with our adapter
        environments.robot_4dof_env.Robot4DOFEnv = RosGazeboAdapter
        
        print("✅ Environment adapter configured")
    
    def create_mbpo_trainer(self):
        """Create MBPO trainer with ROS environment."""
        print("🧠 Creating MBPO trainer...")
        
        # Import and create trainer (will now use our adapted environment)
        from mbpo_trainer import MBPOTrainer
        
        trainer = MBPOTrainer(
            env_config=self.env_config,
            agent_config=self.agent_config,
            buffer_capacity=100000,
            ensemble_size=3,  # Use ensemble for better model learning
            buffer_path='checkpoints/ros_mbpo_replay_buffer.pkl'
        )
        
        print("✅ MBPO trainer created successfully")
        return trainer
    
    def train(self, episodes: int = 1000, max_steps: int = 200, rollout_every: int = 10):
        """
        Run MBPO training using your existing framework.
        
        Args:
            episodes: Number of training episodes
            max_steps: Maximum steps per episode
            rollout_every: Frequency of model rollouts (episodes)
        """
        print(f"🎯 Starting MBPO Training")
        print(f"   Episodes: {episodes}")
        print(f"   Max steps per episode: {max_steps}")
        print(f"   Model rollout frequency: every {rollout_every} episodes")
        print("=" * 60)
        
        try:
            # Use your existing MBPO trainer's run method
            self.mbpo_trainer.run(
                episodes=episodes,
                max_steps=max_steps,
                rollout_every=rollout_every
            )
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Restore original environment and cleanup."""
        try:
            # Restore original Robot4DOFEnv class
            import environments.robot_4dof_env
            environments.robot_4dof_env.Robot4DOFEnv = self._original_env_class
            
            # Close environment
            if hasattr(self.mbpo_trainer, 'env'):
                self.mbpo_trainer.env.close()
                
            print("🧹 Cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")


def main():
    """Main function for direct ROS-MBPO training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ROS-MBPO Integration Training')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=200,
                       help='Maximum steps per episode')
    parser.add_argument('--rollout-freq', type=int, default=10,
                       help='Model rollout frequency (episodes)')
    parser.add_argument('--headless', action='store_true', default=True,
                       help='Run Gazebo in headless mode')
    
    args = parser.parse_args()
    
    # Environment configuration
    env_config = {
        'headless': args.headless,
        'real_time': False,
        'max_steps': args.max_steps,
        'success_distance': 0.02,
        'dense_reward': True
    }
    
    # Agent configuration (optimized for robot arm)
    agent_config = {
        'lr_actor': 0.0001,
        'lr_critic': 0.001,
        'gamma': 0.99,
        'tau': 0.005,
        'noise_std': 0.2,
        'noise_decay': 0.9995,
        'hidden_dims': [512, 256, 128]
    }
    
    print("🤖 ROS-MBPO Integration")
    print("=" * 30)
    
    # Create and run training
    integration = ROSMBPOIntegration(env_config, agent_config)
    integration.train(
        episodes=args.episodes,
        max_steps=args.max_steps,
        rollout_every=args.rollout_freq
    )

if __name__ == '__main__':
    main()