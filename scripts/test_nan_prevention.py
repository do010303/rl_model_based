#!/usr/bin/env python3
"""
Test script để kiểm tra NaN issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.robot_4dof_env import Robot4DOFEnv
from agents.ddpg_agent import DDPGAgent
import numpy as np

def test_nan_prevention():
    """Test cấu hình ultra-stable để ngăn chặn NaN."""
    
    # Ultra-stable environment config
    env_config = {
        'max_steps': 50,          # Ngắn để test nhanh
        'success_distance': 0.05,
        'dense_reward': True,
        'success_reward': 100.0,
        'workspace_radius': 0.5   # Nhỏ hơn để dễ reach
    }
    
    # Ultra-stable agent config
    agent_config = {
        'lr_actor': 0.00001,      # Cực thấp
        'lr_critic': 0.00005,     # Cực thấp
        'gamma': 0.99,            # Standard
        'tau': 0.001,             # Slow
        'noise_std': 0.1,         # Thấp
        'noise_decay': 0.999,     # Slow decay
        'hidden_dims': [128, 64]  # Network nhỏ
    }
    
    # Initialize
    env = Robot4DOFEnv(config=env_config)
    agent = DDPGAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        config=agent_config
    )
    
    print("🧪 Testing NaN prevention...")
    
    for episode in range(3):
        state, info = env.reset()
        episode_reward = 0.0
        
        print(f"\n📍 Episode {episode+1}")
        print(f"   Initial state: {state[:4]} (joint positions)")
        
        for step in range(10):  # Short episodes
            # Test action
            action = agent.act(state, add_noise=True)
            
            if np.any(np.isnan(action)):
                print(f"   ❌ Step {step}: NaN action detected: {action}")
                break
            else:
                print(f"   ✅ Step {step}: Valid action: {action}")
            
            # Test step
            next_state, reward, terminated, truncated, info = env.step(action)
            
            if np.isnan(reward):
                print(f"   ❌ Step {step}: NaN reward: {reward}")
                break
            else:
                print(f"   ✅ Step {step}: Valid reward: {reward:.2f}")
            
            if np.any(np.isnan(next_state)):
                print(f"   ❌ Step {step}: NaN in next_state")
                break
            
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        print(f"   💰 Episode reward: {episode_reward:.2f}")
    
    print("\n🎉 NaN prevention test completed!")
    env.close()

if __name__ == "__main__":
    test_nan_prevention()