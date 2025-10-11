#!/usr/bin/env python3
"""
Simple demo script for 4-DOF Robot Arm
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.robot_4dof_env import Robot4DOFEnv

def demo_environment(episodes=3, render=False):
    """
    Simple demo of the 4-DOF robot arm environment.
    
    Args:
        episodes: Number of episodes to run
        render: Whether to render the environment
    """
    
    print("🤖 4-DOF Robot Arm Demo")
    print("=" * 40)
    
    # Create environment
    config = {
        'max_steps': 50,  # Short episodes for demo
        'success_distance': 0.05,
        'dense_reward': True
    }
    
    env = Robot4DOFEnv(config=config)
    
    print(f"Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Success threshold: {config['success_distance']}m")
    
    total_rewards = []
    success_count = 0
    
    for episode in range(episodes):
        print(f"\n🎯 Episode {episode + 1}/{episodes}")
        
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"Initial position: [{obs[8]:.3f}, {obs[9]:.3f}, {obs[10]:.3f}]")
        print(f"Target position:  [{obs[11]:.3f}, {obs[12]:.3f}, {obs[13]:.3f}]")
        
        initial_distance = np.linalg.norm(obs[8:11] - obs[11:14])
        print(f"Initial distance: {initial_distance:.3f}m")
        
        while True:
            # Random action for demo (replace with trained agent)
            action = env.action_space.sample()
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Render if requested
            if render:
                env.render(mode='human')
            
            if terminated or truncated:
                break
        
        final_distance = info['distance_to_target']
        success = info.get('goal_reached', False)
        
        if success:
            success_count += 1
            status = "✅ SUCCESS"
        else:
            status = "❌ Failed"
        
        print(f"{status} - Steps: {step_count}, Final distance: {final_distance:.3f}m, Reward: {episode_reward:.1f}")
        total_rewards.append(episode_reward)
    
    # Summary
    print(f"\n📊 Demo Summary")
    print(f"Episodes: {episodes}")
    print(f"Success rate: {success_count}/{episodes} ({success_count/episodes:.1%})")
    print(f"Average reward: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
    
    env.close()
    
    print(f"\n🎉 Demo completed!")
    print(f"\n💡 Next steps:")
    print(f"   • Train an agent: python3 examples/train_ddpg.py")
    print(f"   • Test different algorithms")
    print(f"   • Customize the environment")

if __name__ == "__main__":
    # Run demo
    demo_environment(episodes=3, render=False)
