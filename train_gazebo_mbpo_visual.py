#!/usr/bin/env python3
"""
MBPO Training Integration with Gazebo Visual RL Environment
==========================================================

Integrates the visual Gazebo RL environment with MBPO training
Compatible with existing project structure and ROS Noetic

Usage:
1. First launch visual training: roslaunch new_robot_arm_urdf visual_training.launch
2. Then run this training script: python3 train_gazebo_mbpo_visual.py

Author: Adapted for visual RL training
Date: October 2025
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing MBPO components  
from mbpo_trainer import MBPOTrainer
from agents.ddpg_agent import DDPGAgent
from gazebo_rl_environment import GazeboRLEnvironment

# ROS imports
import rospy

class VisualMBPOTrainer:
    """
    MBPO Trainer adapted for visual Gazebo RL environment
    """
    
    def __init__(self, config=None):
        # Initialize ROS
        rospy.init_node('mbpo_visual_trainer', anonymous=True)
        
        # Initialize Gazebo environment first to get correct dimensions
        self.env = GazeboRLEnvironment()
        
        # Default configuration with correct dimensions from environment
        default_config = {
            'state_dim': self.env.get_observation_space_size(),  # 15: joints(4) + vels(4) + ee_pos(3) + target(3) + distance(1)
            'action_dim': self.env.get_action_space_size(),      # 4: 4-DOF robot joints
            'max_episodes': 100,     # Training episodes (reduced for initial testing)
            'max_steps': 50,         # Steps per episode
            'model_train_freq': 250, # How often to train dynamics model
            'policy_train_freq': 1,  # How often to train policy
            'rollout_length': 5,     # Model-based rollout length
            'batch_size': 256,       # Training batch size
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'noise_scale': 0.1,
            'buffer_size': 1000000,
            'save_freq': 50,         # Save model every N episodes
            'log_freq': 10,          # Log progress every N episodes
        }
        
        # Update with user config
        if config:
            default_config.update(config)
        self.config = default_config
        
        rospy.loginfo("🚀 Visual MBPO Trainer initialized with Gazebo environment!")
        
        # Initialize MBPO trainer with existing interface
        env_config = {
            'action_space_size': self.config['action_dim'],
            'observation_space_size': self.config['state_dim'],
            'max_episode_steps': self.config['max_steps']
        }
        
        agent_config = {
            'actor_lr': self.config['learning_rate'],
            'critic_lr': self.config['learning_rate'],
            'gamma': self.config['gamma'],
            'tau': self.config['tau'],
            'noise_scale': self.config['noise_scale']
        }
        
        self.mbpo_trainer = MBPOTrainer(
            env_config=env_config,
            agent_config=agent_config,
            buffer_capacity=self.config['buffer_size']
        )
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate_history = []
        self.distance_history = []
        
        # Success tracking
        self.recent_successes = deque(maxlen=100)  # Track last 100 episodes
        
        rospy.loginfo("✅ Visual MBPO Trainer initialized!")
    
    def train(self):
        """Main training loop"""
        rospy.loginfo("🎯 Starting visual MBPO training...")
        
        best_success_rate = 0.0
        
        for episode in range(self.config['max_episodes']):
            episode_start_time = time.time()
            
            # Reset environment
            state = self.env.reset_environment_request()
            episode_reward = 0
            episode_length = 0
            min_distance = float('inf')
            episode_success = False
            
            for step in range(self.config['max_steps']):
                # Select action using MBPO policy (via DDPG agent)
                action = self.mbpo_trainer.agent.select_action(state)
                
                # Add exploration noise during training
                if episode < self.config['max_episodes'] * 0.8:  # Decay exploration
                    noise = np.random.normal(0, self.config['noise_scale'], self.config['action_dim'])
                    action = np.clip(action + noise, -1.0, 1.0)
                
                # Execute action
                next_state, reward, done, info = self.env.action_step_service(action)
                
                # Store transition in replay buffer
                self.mbpo_trainer.replay_buffer.push(state, action, reward, next_state, done)
                
                # Update statistics
                episode_reward += reward
                episode_length += 1
                
                # Calculate current distance from state (last element in state vector)
                current_distance = state[-1] if len(state) > 0 else float('inf')
                min_distance = min(min_distance, current_distance)
                
                # Check for success (reached target within threshold)
                if current_distance < 0.02:  # 2cm threshold
                    episode_success = True
                
                # Train MBPO components
                if len(self.mbpo_trainer.replay_buffer) > self.config['batch_size']:
                    # Train dynamics model periodically
                    if step % self.config['model_train_freq'] == 0:
                        self.mbpo_trainer.train_dynamics(epochs=3, batch_size=self.config['batch_size'])
                        if episode % self.config['log_freq'] == 0:
                            rospy.loginfo(f"📊 Dynamics model training completed")
                    
                    # Train policy every step  
                    if step % self.config['policy_train_freq'] == 0:
                        loss_dict = self.mbpo_trainer.agent.update_policy(self.mbpo_trainer.replay_buffer)
                        if episode % self.config['log_freq'] == 0 and loss_dict:
                            rospy.loginfo(f"🎯 Policy updated")
                
                state = next_state
                
                if done:
                    break
            
            # Record episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.distance_history.append(min_distance)
            self.recent_successes.append(episode_success)
            
            # Calculate success rate
            current_success_rate = sum(self.recent_successes) / len(self.recent_successes)
            self.success_rate_history.append(current_success_rate)
            
            # Logging
            if episode % self.config['log_freq'] == 0:
                avg_reward = np.mean(self.episode_rewards[-self.config['log_freq']:])
                avg_length = np.mean(self.episode_lengths[-self.config['log_freq']:])
                avg_distance = np.mean(self.distance_history[-self.config['log_freq']:])
                episode_time = time.time() - episode_start_time
                
                rospy.loginfo(f"📈 Episode {episode + 1}/{self.config['max_episodes']}")
                rospy.loginfo(f"   Reward: {episode_reward:.2f} (avg: {avg_reward:.2f})")
                rospy.loginfo(f"   Length: {episode_length} (avg: {avg_length:.1f})")
                rospy.loginfo(f"   Distance: {min_distance:.4f} (avg: {avg_distance:.4f})")
                rospy.loginfo(f"   Success Rate: {current_success_rate:.2%}")
                rospy.loginfo(f"   Time: {episode_time:.2f}s")
                rospy.loginfo(f"   Buffer Size: {len(self.mbpo_trainer.replay_buffer)}")
            
            # Save best model
            if current_success_rate > best_success_rate:
                best_success_rate = current_success_rate
                self.save_model(f"best_model_sr_{current_success_rate:.2%}")
                rospy.loginfo(f"🏆 New best success rate: {current_success_rate:.2%}")
            
            # Periodic model saving
            if episode % self.config['save_freq'] == 0:
                self.save_model(f"checkpoint_episode_{episode}")
                self.plot_training_progress()
        
        rospy.loginfo("🎉 Training completed!")
        self.save_model("final_model")
        self.plot_training_progress(save=True)
    
    def evaluate(self, num_episodes=10):
        """Evaluate trained policy"""
        rospy.loginfo(f"🧪 Evaluating policy for {num_episodes} episodes...")
        
        eval_rewards = []
        eval_successes = []
        eval_distances = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            min_distance = float('inf')
            episode_success = False
            
            for step in range(self.config['max_steps']):
                # Use deterministic policy (no noise)
                action = self.mbpo_trainer.select_action(state, deterministic=True)
                
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                min_distance = min(min_distance, info['distance_to_target'])
                
                if info['success']:
                    episode_success = True
                
                state = next_state
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_successes.append(episode_success)
            eval_distances.append(min_distance)
            
            rospy.loginfo(f"Eval Episode {episode + 1}: Reward={episode_reward:.2f}, "
                         f"Distance={min_distance:.4f}, Success={episode_success}")
        
        # Summary statistics
        avg_reward = np.mean(eval_rewards)
        success_rate = np.mean(eval_successes)
        avg_distance = np.mean(eval_distances)
        
        rospy.loginfo(f"📊 Evaluation Results:")
        rospy.loginfo(f"   Average Reward: {avg_reward:.2f}")
        rospy.loginfo(f"   Success Rate: {success_rate:.2%}")
        rospy.loginfo(f"   Average Min Distance: {avg_distance:.4f}")
        
        return {
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'avg_distance': avg_distance,
            'all_rewards': eval_rewards,
            'all_successes': eval_successes,
            'all_distances': eval_distances
        }
    
    def save_model(self, name="model"):
        """Save trained models"""
        save_dir = "checkpoints/visual_mbpo"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save MBPO components
        self.mbpo_trainer.save_models(os.path.join(save_dir, name))
        
        # Save training statistics
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rate_history': self.success_rate_history,
            'distance_history': self.distance_history,
            'config': self.config
        }
        
        import pickle
        with open(os.path.join(save_dir, f"{name}_stats.pkl"), 'wb') as f:
            pickle.dump(stats, f)
        
        rospy.loginfo(f"💾 Model saved: {save_dir}/{name}")
    
    def load_model(self, path):
        """Load trained model"""
        self.mbpo_trainer.load_models(path)
        rospy.loginfo(f"📂 Model loaded: {path}")
    
    def plot_training_progress(self, save=False):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Success rate  
        axes[0, 1].plot(self.success_rate_history)
        axes[0, 1].set_title('Success Rate')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].grid(True)
        
        # Episode lengths
        axes[1, 0].plot(self.episode_lengths)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)
        
        # Distance to target
        axes[1, 1].plot(self.distance_history)
        axes[1, 1].set_title('Min Distance to Target')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Distance (m)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('logs/visual_mbpo_training_progress.png', dpi=300, bbox_inches='tight')
            rospy.loginfo("📊 Training progress plot saved")
        else:
            plt.show()


def main():
    """Main training function"""
    try:
        # Configuration
        config = {
            'max_episodes': 300,
            'max_steps': 50,
            'model_train_freq': 100,
            'policy_train_freq': 1,
            'batch_size': 256,
            'learning_rate': 3e-4,
            'noise_scale': 0.15,
            'save_freq': 25,
            'log_freq': 5,
            'use_cuda': False  # Set to True if you have CUDA
        }
        
        # Initialize trainer
        trainer = VisualMBPOTrainer(config)
        
        # Check if we should load existing model
        if len(sys.argv) > 1 and sys.argv[1] == '--load':
            model_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/visual_mbpo/best_model"
            trainer.load_model(model_path)
            rospy.loginfo("Loaded existing model for continued training")
        
        # Check if we should only evaluate
        if len(sys.argv) > 1 and sys.argv[1] == '--eval':
            model_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/visual_mbpo/best_model"
            trainer.load_model(model_path)
            trainer.evaluate(num_episodes=20)
        else:
            # Run training
            trainer.train()
            
            # Final evaluation
            rospy.loginfo("🧪 Running final evaluation...")
            trainer.evaluate(num_episodes=20)
    
    except rospy.ROSInterruptException:
        rospy.loginfo("Training interrupted by user")
    except Exception as e:
        rospy.logerr(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rospy.loginfo("Training session ended")


if __name__ == "__main__":
    main()