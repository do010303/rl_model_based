#!/usr/bin/env python3
"""
Simple project verification
"""

def main():
    print("🚀 Robotarm-RL-4DoF Project Verification")
    print("=" * 45)
    
    # Test 1: Import environment
    try:
        from environments.robot_4dof_env import Robot4DOFEnv
        print("✅ Environment import successful")
    except Exception as e:
        print(f"❌ Environment import failed: {e}")
        return False
    
    # Test 2: Create environment
    try:
        env = Robot4DOFEnv()
        print("✅ Environment creation successful")
        print(f"   - Observation space: {env.observation_space.shape}")
        print(f"   - Action space: {env.action_space.shape}")
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return False
    
    # Test 3: Environment basic functionality
    try:
        obs, info = env.reset()
        print("✅ Environment reset successful")
        
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print("✅ Environment step successful")
        print(f"   - Reward: {reward:.3f}")
        
        env.close()
    except Exception as e:
        print(f"❌ Environment functionality failed: {e}")
        return False
    
    # Test 4: Import agent
    try:
        from agents.ddpg_agent import DDPGAgent
        print("✅ Agent import successful")
    except Exception as e:
        print(f"❌ Agent import failed: {e}")
        return False
    
    # Test 5: Import replay buffer  
    try:
        from replay_memory.replay_buffer import ReplayBuffer
        print("✅ Replay buffer import successful")
    except Exception as e:
        print(f"❌ Replay buffer import failed: {e}")
        return False
    
    print("\n🎉 All basic tests passed!")
    print("\n📋 Project ready for:")
    print("   • Training: python3 examples/train_ddpg.py")  
    print("   • Testing: python3 examples/test_model.py")
    print("   • Curriculum: python3 examples/train_curriculum.py")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
