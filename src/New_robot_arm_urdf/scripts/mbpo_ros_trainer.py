#!/usr/bin/env python3
"""
MBPO Training Integration for ROS-Gazebo Robot Arm

This script integrates the existing MBPO (Model-Based Policy Optimization) framework
with the ROS-Gazebo robot arm simulation environment.
"""

import sys
import os
import numpy as np
import rospy
import argparse
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, '/home/ducanh/rl_model_based')

# Import MBPO components
from mbpo_trainer import MBPOTrainer
from agents.ddpg_agent import DDPGAgent
from models.dynamics_model import DynamicsModel
from replay_memory.replay_buffer import ReplayBuffer

# Import ROS-Gazebo environment wrapper
from robot_arm_env import RobotArmEnv

class RosMBPOWrapper:
    """
    Wrapper that adapts the ROS-Gazebo robot arm environment to work with MBPO.
    
    This creates a bridge between:
    - ROS-Gazebo simulation (robot_arm_env.py)  
    - MBPO framework (existing implementation)
    """
    
    def __init__(self, mbpo_config: Dict[str, Any]):
        """Initialize the ROS-MBPO integration."""
        
        self.config = mbpo_config
        
        # Initialize ROS node
        if not rospy.get_node_uri():
            rospy.init_node('mbpo_robot_arm_trainer', anonymous=True)
        
        # Create ROS-Gazebo environment
        self.ros_env = RobotArmEnv(
            headless=mbpo_config.get('headless', True),
            real_time=mbpo_config.get('real_time', False)
        )
        
        # Get environment dimensions
        self.state_dim = self.ros_env.n_observations  # 12 dimensions
        self.action_dim = self.ros_env.n_actions      # 4 joints
        
        print(f"🤖 ROS Environment initialized:")
        print(f"   State dimension: {self.state_dim}")
        print(f"   Action dimension: {self.action_dim}")
        
        # Create DDPG agent with robust configuration
        agent_config = {
            'lr_actor': mbpo_config.get('lr_actor', 0.0001),
            'lr_critic': mbpo_config.get('lr_critic', 0.001), 
            'gamma': mbpo_config.get('gamma', 0.99),
            'tau': mbpo_config.get('tau', 0.005),
            'noise_std': mbpo_config.get('noise_std', 0.2),
            'noise_decay': mbpo_config.get('noise_decay', 0.9995),
            'hidden_dims': [512, 256, 128]
        }
        
        self.agent = DDPGAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=agent_config
        )
        
        # Create replay buffer
        buffer_capacity = mbpo_config.get('buffer_capacity', 100000)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Create ensemble of dynamics models
        ensemble_size = mbpo_config.get('ensemble_size', 3)
        self.dynamics_models = []
        for i in range(ensemble_size):
            model = DynamicsModel(self.state_dim, self.action_dim)
            self.dynamics_models.append(model)
        
        print(f"🧠 MBPO Components initialized:")
        print(f"   Dynamics ensemble size: {ensemble_size}")
        print(f"   Replay buffer capacity: {buffer_capacity}")
        
        # Training parameters
        self.model_train_freq = mbpo_config.get('model_train_freq', 250)
        self.rollout_freq = mbpo_config.get('rollout_freq', 100)
        self.rollout_length = mbpo_config.get('rollout_length', 5)
        self.num_rollouts = mbpo_config.get('num_rollouts', 100)
        self.policy_train_freq = mbpo_config.get('policy_train_freq', 1)
        
        # Metrics tracking
        self.episode_rewards = []
        self.success_rates = []
        self.distances = []
        
    def collect_real_experience(self, num_steps: int = 200) -> Dict[str, float]:
        """
        Collect real environment experience using current policy.
        
        Args:
            num_steps: Maximum steps per episode
            
        Returns:
            Episode statistics dictionary
        """
        # Reset environment
        observation = self.ros_env.reset()
        if observation is None:
            print("⚠️ Failed to reset ROS environment")
            return {'reward': -1000, 'success': False, 'distance': 1.0, 'steps': 0}
        
        episode_reward = 0.0
        episode_success = False
        episode_distances = []
        step_count = 0
        
        for step in range(num_steps):
            # Get action from agent
            action = self.agent.act(observation, add_noise=True)
            
            # Take step in ROS environment
            next_observation, reward, done, info = self.ros_env.step(action)
            
            if next_observation is None:
                print(f"⚠️ Invalid observation at step {step}, terminating episode")
                break
            
            # Store transition in replay buffer
            self.replay_buffer.add(observation, action, reward, next_observation, done)
            
            # Update episode statistics
            episode_reward += reward
            current_distance = info.get('distance_to_target', 1.0)
            episode_distances.append(current_distance)
            
            if info.get('goal_reached', False):
                episode_success = True
            
            observation = next_observation
            step_count += 1
            
            if done:
                break
        
        avg_distance = np.mean(episode_distances) if episode_distances else 1.0
        
        return {
            'reward': episode_reward,
            'success': episode_success, 
            'distance': avg_distance,
            'steps': step_count
        }
    
    def train_dynamics_models(self, batch_size: int = 256, epochs: int = 5):
        """
        Train the ensemble of dynamics models on collected data.
        
        Args:
            batch_size: Batch size for training
            epochs: Number of training epochs
        """
        if len(self.replay_buffer) < batch_size:
            print(f"⚠️ Not enough data for dynamics training: {len(self.replay_buffer)} < {batch_size}")
            return
        
        print(f"🧠 Training dynamics models on {len(self.replay_buffer)} samples...")
        
        # Get all transitions
        transitions = self.replay_buffer.get_all_transitions()
        states = np.array([t['state'] for t in transitions])
        actions = np.array([t['action'] for t in transitions])
        next_states = np.array([t['next_state'] for t in transitions])
        rewards = np.array([t['reward'] for t in transitions]).reshape(-1, 1)
        
        # Train each model in ensemble
        for i, model in enumerate(self.dynamics_models):
            try:
                model.train(states, actions, next_states, rewards, epochs=epochs, batch_size=batch_size)
            except Exception as e:
                print(f"⚠️ Error training dynamics model {i}: {e}")
    
    def generate_model_rollouts(self):
        """
        Generate synthetic rollouts using trained dynamics models.
        """
        if len(self.replay_buffer) == 0:
            print("⚠️ No real data available for model rollouts")
            return
        
        print(f"🎲 Generating {self.num_rollouts} model rollouts of length {self.rollout_length}...")
        
        # Get random starting states from real data
        transitions = self.replay_buffer.get_all_transitions()
        real_states = [t['state'] for t in transitions]
        
        rollouts_generated = 0
        
        for _ in range(self.num_rollouts):
            # Sample random starting state
            start_state = real_states[np.random.randint(len(real_states))]
            
            # Perform rollout
            state = start_state.copy()
            
            for step in range(self.rollout_length):
                # Get action from current policy
                action = self.agent.act(state, add_noise=True)
                
                # Sample random dynamics model from ensemble
                model_idx = np.random.randint(len(self.dynamics_models))
                model = self.dynamics_models[model_idx]
                
                try:
                    # Predict next state and reward
                    next_state, reward = model.predict(state, action)
                    
                    # Check for valid predictions
                    if (np.any(np.isnan(next_state)) or np.isnan(reward) or 
                        np.any(np.isinf(next_state)) or np.isinf(reward)):
                        break  # Skip invalid rollouts
                    
                    # Add synthetic transition to replay buffer
                    # Mark as done randomly to simulate episode endings
                    done = np.random.random() < 0.05  # 5% chance of episode termination
                    self.replay_buffer.add(state, action, reward, next_state, done)
                    
                    state = next_state
                    rollouts_generated += 1
                    
                except Exception as e:
                    print(f"⚠️ Error in model rollout: {e}")
                    break
        
        print(f"✅ Generated {rollouts_generated} synthetic transitions")
    
    def train_policy(self, num_updates: int = 50):
        """
        Train the policy using mixed real and synthetic data.
        
        Args:
            num_updates: Number of policy update steps
        """
        if len(self.replay_buffer) < 256:
            return
        
        print(f"🎯 Training policy for {num_updates} updates...")
        
        for _ in range(num_updates):
            # Sample batch from combined real+synthetic data
            batch = self.replay_buffer.sample(256)
            
            # Train agent
            try:
                self.agent.train_step(batch)
            except Exception as e:
                print(f"⚠️ Error in policy training: {e}")
                break
    
    def train(self, episodes: int = 1000, max_steps: int = 200):
        """
        Main MBPO training loop for ROS-Gazebo robot arm.
        
        Args:
            episodes: Number of training episodes
            max_steps: Maximum steps per episode
        """
        print(f"🚀 Starting MBPO training for {episodes} episodes")
        print(f"   Max steps per episode: {max_steps}")
        print(f"   Model training frequency: {self.model_train_freq}")
        print(f"   Rollout frequency: {self.rollout_freq}")
        print("=" * 60)
        
        total_steps = 0
        
        for episode in range(episodes):
            # Collect real environment experience
            episode_stats = self.collect_real_experience(max_steps)
            
            total_steps += episode_stats['steps']
            
            # Store episode statistics
            self.episode_rewards.append(episode_stats['reward'])
            success_rate = np.mean([s for s in [episode_stats['success']] + 
                                  [stats['success'] for stats in 
                                   getattr(self, '_recent_episodes', [])[-99:])])
            self.success_rates.append(success_rate)
            self.distances.append(episode_stats['distance'])
            
            # Train policy every episode
            if episode % self.policy_train_freq == 0:
                self.train_policy(num_updates=40)
            
            # Train dynamics models periodically
            if episode > 0 and episode % self.model_train_freq == 0:
                self.train_dynamics_models()
            
            # Generate model rollouts periodically
            if episode > 0 and episode % self.rollout_freq == 0:
                self.generate_model_rollouts()
            
            # Print progress
            if episode % 10 == 0 or episode < 10:
                status_icon = "✅" if episode_stats['success'] else "🔄"
                print(f"{status_icon} Episode {episode:4d} | "
                      f"Reward: {episode_stats['reward']:7.1f} | "
                      f"Success: {success_rate:5.1%} | "
                      f"Distance: {episode_stats['distance']:.3f}m | "
                      f"Steps: {episode_stats['steps']:3d} | "
                      f"Buffer: {len(self.replay_buffer):5d}")
        
        # Save final results
        self.save_results()
        print(f"\n🎉 MBPO training completed!")
        print(f"   Total environment steps: {total_steps}")
        print(f"   Final success rate: {self.success_rates[-1]:.1%}")
        print(f"   Final average reward: {np.mean(self.episode_rewards[-100:]):.1f}")
    
    def save_results(self):
        """Save training results and models."""
        os.makedirs('/home/ducanh/rl_model_based/results', exist_ok=True)
        
        # Save replay buffer
        buffer_path = '/home/ducanh/rl_model_based/results/mbpo_ros_buffer.pkl'
        try:
            self.replay_buffer.save(buffer_path)
            print(f"💾 Saved replay buffer to {buffer_path}")
        except Exception as e:
            print(f"⚠️ Failed to save replay buffer: {e}")
        
        # Save training metrics
        metrics = {
            'episode_rewards': self.episode_rewards,
            'success_rates': self.success_rates,
            'distances': self.distances
        }
        
        metrics_path = '/home/ducanh/rl_model_based/results/mbpo_ros_metrics.npy'
        try:
            np.save(metrics_path, metrics)
            print(f"📊 Saved training metrics to {metrics_path}")
        except Exception as e:
            print(f"⚠️ Failed to save metrics: {e}")
        
        # Save agent
        agent_path = '/home/ducanh/rl_model_based/results/mbpo_ros_agent.keras'
        try:
            self.agent.save(agent_path)
            print(f"🤖 Saved trained agent to {agent_path}")
        except Exception as e:
            print(f"⚠️ Failed to save agent: {e}")
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            self.ros_env.cleanup()
            print("🧹 Cleaned up ROS environment")
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")

def main():
    """Main training function with command line arguments."""
    
    parser = argparse.ArgumentParser(description='MBPO Training for ROS-Gazebo Robot Arm')
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--max-steps', type=int, default=200,
                       help='Maximum steps per episode (default: 200)')
    parser.add_argument('--headless', action='store_true', default=True,
                       help='Run Gazebo in headless mode (default: True)')
    parser.add_argument('--real-time', action='store_true', default=False,
                       help='Run simulation in real-time (default: False)')
    parser.add_argument('--buffer-capacity', type=int, default=100000,
                       help='Replay buffer capacity (default: 100000)')
    parser.add_argument('--ensemble-size', type=int, default=3,
                       help='Dynamics model ensemble size (default: 3)')
    parser.add_argument('--model-train-freq', type=int, default=250,
                       help='Model training frequency (episodes) (default: 250)')
    parser.add_argument('--rollout-freq', type=int, default=100,
                       help='Model rollout frequency (episodes) (default: 100)')
    
    args = parser.parse_args()
    
    # Configuration for MBPO
    mbpo_config = {
        'headless': args.headless,
        'real_time': args.real_time,
        'buffer_capacity': args.buffer_capacity,
        'ensemble_size': args.ensemble_size,
        'model_train_freq': args.model_train_freq,
        'rollout_freq': args.rollout_freq,
        'rollout_length': 5,
        'num_rollouts': 100,
        'policy_train_freq': 1,
        # Agent hyperparameters optimized for robot arm
        'lr_actor': 0.0001,
        'lr_critic': 0.001,
        'gamma': 0.99,
        'tau': 0.005,
        'noise_std': 0.2,
        'noise_decay': 0.9995
    }
    
    print("🤖 MBPO Training for ROS-Gazebo Robot Arm")
    print("=" * 50)
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Headless mode: {args.headless}")
    print(f"Ensemble size: {args.ensemble_size}")
    print(f"Model training frequency: {args.model_train_freq}")
    print(f"Rollout frequency: {args.rollout_freq}")
    
    # Initialize trainer
    trainer = None
    
    try:
        trainer = RosMBPOWrapper(mbpo_config)
        
        # Run training
        trainer.train(episodes=args.episodes, max_steps=args.max_steps)
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if trainer:
            trainer.cleanup()
    
    print(f"\n🏁 Training session ended")

if __name__ == '__main__':
    main()