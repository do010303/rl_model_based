#!/usr/bin/env python3
"""
Quick test script for Gazebo MBPO integration
Tests basic functionality without full training
"""

import os
import sys
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Test core imports
        import numpy as np
        import rospy
        print("✅ Core dependencies imported")
        
        # Test project imports
        from environments.gazebo_robot_4dof_env import GazeboRobot4DOFEnv
        from agents.ddpg_agent import DDPGAgent
        from models.dynamics_model import DynamicsModel
        from replay_memory.replay_buffer import ReplayBuffer
        print("✅ Project modules imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_environment_standalone():
    """Test environment without Gazebo (fallback mode)"""
    print("🧪 Testing environment in standalone mode...")
    
    try:
        from environments.gazebo_robot_4dof_env import GazeboRobot4DOFEnv
        
        # Create environment with ROS disabled
        env_config = {'use_ros': False, 'max_steps': 10}
        env = GazeboRobot4DOFEnv(config=env_config)
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Environment reset - obs shape: {obs.shape}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Environment step - reward: {reward:.3f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_agent():
    """Test DDPG agent initialization"""
    print("🧪 Testing DDPG agent...")
    
    try:
        from agents.ddpg_agent import DDPGAgent
        import numpy as np
        
        # Create agent
        agent = DDPGAgent(state_dim=14, action_dim=4, config={})
        print("✅ DDPG agent created")
        
        # Test action selection
        state = np.random.randn(14).astype(np.float32)
        action = agent.act(state, add_noise=False)
        print(f"✅ Action generated - shape: {action.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

def test_dynamics_model():
    """Test dynamics model"""
    print("🧪 Testing dynamics model...")
    
    try:
        from models.dynamics_model import DynamicsModel
        import numpy as np
        
        # Create model (using correct initialization)
        model = DynamicsModel(state_dim=14, action_dim=4)
        print("✅ Dynamics model created")
        
        # Test prediction
        state = np.random.randn(14).astype(np.float32)
        action = np.random.randn(4).astype(np.float32)
        next_state, reward = model.predict(state, action)
        print(f"✅ Model prediction - next_state shape: {next_state.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dynamics model test failed: {e}")
        return False

def test_replay_buffer():
    """Test replay buffer"""
    print("🧪 Testing replay buffer...")
    
    try:
        from replay_memory.replay_buffer import ReplayBuffer
        import numpy as np
        
        # Create buffer (using correct initialization)
        buffer = ReplayBuffer(capacity=10000)
        print("✅ Replay buffer created")
        
        # Add sample data
        state = np.random.randn(14).astype(np.float32)
        action = np.random.randn(4).astype(np.float32)
        reward = 1.0
        next_state = np.random.randn(14).astype(np.float32)
        done = False
        
        buffer.add(state, action, reward, next_state, done)
        print(f"✅ Data added to buffer - size: {len(buffer)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Replay buffer test failed: {e}")
        return False

def test_workspace_build():
    """Test if the robot workspace can be built"""
    print("🧪 Testing workspace build...")
    
    try:
        import subprocess
        
        robot_ws_path = "/home/ducanh/rl_model_based/robot_ws"
        
        # Check if workspace exists
        if not os.path.exists(robot_ws_path):
            print(f"❌ Robot workspace not found: {robot_ws_path}")
            return False
        
        # Test catkin_make with proper bash sourcing
        build_cmd = [
            "bash", "-c", 
            f"cd {robot_ws_path} && source /opt/ros/noetic/setup.bash && catkin_make --dry-run"
        ]
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Workspace build test passed")
            return True
        else:
            print(f"❌ Workspace build test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Workspace build test error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Running Gazebo MBPO integration tests...\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Environment Test", test_environment_standalone),
        ("Agent Test", test_agent),
        ("Dynamics Model Test", test_dynamics_model),
        ("Replay Buffer Test", test_replay_buffer),
        ("Workspace Build Test", test_workspace_build),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for Gazebo MBPO training.")
        return True
    else:
        print("⚠️  Some tests failed. Please fix issues before training.")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)