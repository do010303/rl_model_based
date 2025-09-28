#!/usr/bin/env python3
"""
Quick test script to verify project functionality
"""

import sys
import os
import traceback

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test all imports work correctly."""
    print("🧪 Testing imports...")
    
    try:
        from environments import Robot4DOFEnv
        print("✅ Environment import successful")
        
        from agents import DDPGAgent, BaseAgent
        print("✅ Agent imports successful")
        
        from replay_memory import ReplayBuffer
        print("✅ Replay buffer import successful")
        
        from utils import HER
        print("✅ HER import successful")
        
        from training import CurriculumTrainer
        print("✅ Training imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_environment():
    """Test environment creation and basic functionality."""
    print("\n🤖 Testing environment...")
    
    try:
        from environments import Robot4DOFEnv
        
        env = Robot4DOFEnv()
        print(f"✅ Environment created successfully")
        print(f"   - Action space: {env.action_space}")
        print(f"   - Observation space: {env.observation_space}")
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Environment reset successful")
        print(f"   - Observation shape: {obs.shape}")
        print(f"   - Info keys: {list(info.keys())}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Environment step successful")
        print(f"   - Reward: {reward:.3f}")
        print(f"   - Distance to target: {info.get('distance_to_target', 'N/A'):.3f}m")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment error: {e}")
        traceback.print_exc()
        return False

def test_agent():
    """Test agent creation."""
    print("\n🧠 Testing agent...")
    
    try:
        from agents import DDPGAgent
        from environments import Robot4DOFEnv
        
        env = Robot4DOFEnv()
        agent = DDPGAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0]
        )
        
        print("✅ DDPG agent created successfully")
        print(f"   - State dim: {agent.state_dim}")
        print(f"   - Action dim: {agent.action_dim}")
        
        # Test action selection
        obs, _ = env.reset()
        action = agent.act(obs, add_noise=False)
        print(f"✅ Action selection successful")
        print(f"   - Action shape: {action.shape}")
        print(f"   - Action range: [{action.min():.3f}, {action.max():.3f}]")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Agent error: {e}")
        traceback.print_exc()
        return False

def test_training_setup():
    """Test training components."""
    print("\n🎯 Testing training setup...")
    
    try:
        from replay_memory import ReplayBuffer
        from utils import HER
        
        # Test replay buffer
        buffer = ReplayBuffer(capacity=1000)
        print("✅ Replay buffer created successfully")
        
        # Test HER
        her = HER(replay_buffer=buffer, k=4, strategy='future')
        print("✅ HER created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Training setup error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Running Robotarm-RL-4DoF Project Tests")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Environment Test", test_environment),
        ("Agent Test", test_agent),
        ("Training Setup Test", test_training_setup),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project is ready for use.")
        print("\n🚀 Next steps:")
        print("   python examples/train_ddpg.py        # Start training")
        print("   python examples/test_model.py        # Test trained model")
        print("   python examples/train_curriculum.py  # Curriculum learning")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
