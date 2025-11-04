#!/usr/bin/env python3
"""
Visual Training Script for Robot Arm Target Reaching
==================================================

This script demonstrates visual training where you can watch the robot
learn to reach red target spheres in Gazebo simulation.

Usage:
1. Launch Gazebo with: roslaunch new_robot_arm_urdf visual_training.launch
2. Run this script to start visual training
"""

import os
import sys
import time
import numpy as np
import random
import rospy

# Add project root to path
sys.path.append('/home/ducanh/rl_model_based')

from environments.visual_target_env import make_visual_env

class SimpleTargetReachingAgent:
    """Simple agent that learns to reach targets using basic policy"""
    
    def __init__(self, action_dim=4):
        self.action_dim = action_dim
        self.learning_rate = 0.01
        self.exploration_rate = 0.3
        self.exploration_decay = 0.995
        self.min_exploration = 0.05
        
        # Simple policy: move toward target
        self.best_distance = float('inf')
        self.best_action = None
        
    def select_action(self, observation):
        """Select action based on current observation"""
        # Extract positions from observation
        joint_pos = observation[:4]
        joint_vel = observation[4:8] 
        target_pos = observation[8:11]
        ee_pos = observation[11:14]
        distance = observation[14]
        
        # Calculate direction to target
        direction = target_pos - ee_pos
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm > 0:
            direction = direction / direction_norm
        
        # Much gentler movements to prevent jumping
        action = np.zeros(4)
        
        # Reduced exploration for stability
        if random.random() < self.exploration_rate:
            action = np.random.uniform(-0.1, 0.1, 4)  # Much smaller random movements
        else:
            # Very gentle heuristic policy
            if target_pos[0] != 0 or target_pos[1] != 0:
                target_angle = np.arctan2(target_pos[1], target_pos[0])
                current_angle = joint_pos[0]
                angle_diff = target_angle - current_angle
                # Much smaller movement
                action[0] = np.clip(angle_diff * 0.1, -0.2, 0.2)
            
            # Gentle vertical movement toward target
            height_diff = target_pos[2] - ee_pos[2]
            if height_diff > 0.02:  # Target above by more than 2cm
                action[1] = 0.1   # Very gentle upward motion
                action[2] = 0.05
                action[3] = 0.05
            elif height_diff < -0.02:  # Target below by more than 2cm
                action[1] = -0.1  # Very gentle downward motion
                action[2] = -0.05
                action[3] = -0.05
            
            # Gentle horizontal adjustments
            if abs(direction[0]) > 0.1:
                action[1] += direction[0] * 0.05
            if abs(direction[1]) > 0.1:
                action[0] += direction[1] * 0.05
        
        # Update exploration rate (slower decay for stability)
        self.exploration_rate = max(
            self.min_exploration, 
            self.exploration_rate * 0.999  # Much slower decay
        )
        
        # Clip to smaller range for stability
        return np.clip(action, -0.3, 0.3)
    
    def update(self, observation, action, reward, next_observation, done):
        """Update agent based on experience (placeholder for RL algorithm)"""
        # Simple tracking of best performance
        distance = observation[14]
        if distance < self.best_distance:
            self.best_distance = distance
            self.best_action = action.copy()

def visual_training_demo(episodes=20):
    """Run visual training demonstration"""
    
    print("🤖 Starting Stable Visual Target Reaching Training")
    print("=" * 50)
    print("You should see:")
    print("🔴 Red sphere = Target to reach")
    print("🔵 Blue sphere = Robot end effector") 
    print("🤖 Robot arm = Smooth, controlled movements")
    print("🟢 Green circle = Workspace boundary")
    print("💡 Fixed: No more erratic jumping!")
    print("=" * 50)
    
    # Create environment with stable settings
    env = make_visual_env(use_gui=True, real_time=True)
    agent = SimpleTargetReachingAgent()
    
    # Start from a safe initial position
    print("🔧 Initializing robot to safe position...")
    rospy.sleep(2.0)  # Wait for systems to settle
    
    # Training statistics
    episode_rewards = []
    success_count = 0
    
    for episode in range(episodes):
        print(f"\n📊 Episode {episode + 1}/{episodes}")
        
        # Reset environment
        observation = env.reset()
        episode_reward = 0
        step = 0
        
        while True:
            step += 1
            
            # Select action
            action = agent.select_action(observation)
            
            # Take step
            next_observation, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Update agent
            agent.update(observation, action, reward, next_observation, done)
            
            # Print step info
            if step % 20 == 0:
                distance = info['distance_to_target']
                print(f"  Step {step}: Distance = {distance:.3f}m, Reward = {reward:.2f}")
            
            observation = next_observation
            
            if done:
                break
        
        # Episode summary
        episode_rewards.append(episode_reward)
        if info['success']:
            success_count += 1
            print(f"🎯 SUCCESS! Target reached in {step} steps")
        else:
            print(f"❌ Episode ended. Final distance: {info['distance_to_target']:.3f}m")
        
        print(f"📈 Episode reward: {episode_reward:.2f}")
        print(f"🎲 Exploration rate: {agent.exploration_rate:.3f}")
        
        # Print progress
        if episode > 0:
            success_rate = (success_count / (episode + 1)) * 100
            avg_reward = np.mean(episode_rewards[-10:])  # Last 10 episodes
            print(f"📊 Success rate: {success_rate:.1f}% | Avg reward (last 10): {avg_reward:.2f}")
    
    # Final statistics
    print("\n" + "=" * 50)
    print("🏁 Training Complete!")
    print(f"Total episodes: {episodes}")
    print(f"Total successes: {success_count}")
    print(f"Final success rate: {(success_count / episodes) * 100:.1f}%")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print("=" * 50)
    
    env.close()

if __name__ == "__main__":
    print("🚀 Visual Robot Arm Training")
    print("\nMAKE SURE TO:")
    print("1. Launch Gazebo first:")
    print("   roslaunch new_robot_arm_urdf visual_training.launch")
    print("2. Wait for Gazebo to fully load")
    print("3. Then run this training script")
    print()
    
    input("Press Enter when Gazebo is ready...")
    
    try:
        # Run training demo
        visual_training_demo(episodes=20)
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()