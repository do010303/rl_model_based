#!/usr/bin/env python3
"""
Quick test script to verify ROS-MBPO integration works
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, '/home/ducanh/rl_model_based')

def test_imports():
    """Test that all required components can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Test ROS environment import
        from robot_arm_env import RobotArmEnv
        print("✅ RobotArmEnv imported successfully")
    except Exception as e:
        print(f"❌ RobotArmEnv import failed: {e}")
        return False
    
    try:
        # Test parent MBPO components
        from mbpo_trainer import MBPOTrainer
        print("✅ MBPOTrainer imported successfully")
    except Exception as e:
        print(f"❌ MBPOTrainer import failed: {e}")
        return False
    
    try:
        from agents.ddpg_agent import DDPGAgent
        print("✅ DDPGAgent imported successfully")
    except Exception as e:
        print(f"❌ DDPGAgent import failed: {e}")
        return False
        
    try:
        from models.dynamics_model import DynamicsModel
        print("✅ DynamicsModel imported successfully")
    except Exception as e:
        print(f"❌ DynamicsModel import failed: {e}")
        return False
        
    try:
        from replay_memory.replay_buffer import ReplayBuffer
        print("✅ ReplayBuffer imported successfully")
    except Exception as e:
        print(f"❌ ReplayBuffer import failed: {e}")
        return False
    
    try:
        # Test integration adapter
        from ros_mbpo_integration import ROSMBPOIntegration, RosGazeboAdapter
        print("✅ ROS-MBPO integration imported successfully")
    except Exception as e:
        print(f"❌ ROS-MBPO integration import failed: {e}")
        return False
    
    return True

def test_environment_creation():
    """Test that the environment adapter can be created."""
    print("\n🤖 Testing environment creation...")
    
    try:
        import rospy
        from ros_mbpo_integration import RosGazeboAdapter
        
        # Initialize ROS node
        if not rospy.get_node_uri():
            rospy.init_node('test_ros_mbpo', anonymous=True)
        
        # Create adapter (without starting Gazebo)
        config = {
            'headless': True,
            'real_time': False,
            'max_steps': 50
        }
        
        print("   Creating ROS-Gazebo adapter...")
        # Note: This might fail if Gazebo isn't running, which is expected
        env = RosGazeboAdapter(config)
        print("✅ RosGazeboAdapter created successfully")
        
        # Check spaces
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"⚠️  Environment creation test failed (expected if Gazebo not running): {e}")
        return False

def test_mbpo_integration():
    """Test that the MBPO integration can be created."""
    print("\n🧠 Testing MBPO integration...")
    
    try:
        from ros_mbpo_integration import ROSMBPOIntegration
        
        env_config = {
            'headless': True,
            'real_time': False,
            'max_steps': 50
        }
        
        agent_config = {
            'lr_actor': 0.0001,
            'lr_critic': 0.001,
            'gamma': 0.99
        }
        
        print("   Creating ROS-MBPO integration...")
        # This will also fail if Gazebo isn't running, but we can test the setup
        integration = ROSMBPOIntegration(env_config, agent_config)
        print("✅ ROSMBPOIntegration created successfully")
        
        integration.cleanup()
        return True
        
    except Exception as e:
        print(f"⚠️  MBPO integration test failed (expected if Gazebo not running): {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 ROS-MBPO Integration Test Suite")
    print("=" * 40)
    
    # Test 1: Imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Import tests failed. Please check your Python path and dependencies.")
        return False
    
    # Test 2: Environment (may fail without Gazebo)
    test_environment_creation()
    
    # Test 3: MBPO Integration (may fail without Gazebo)  
    test_mbpo_integration()
    
    print("\n📋 Test Summary:")
    print("✅ All import tests passed")
    print("⚠️  Environment tests may fail without running Gazebo (this is normal)")
    print("\n💡 To run actual training:")
    print("   1. Start ROS and Gazebo simulation")
    print("   2. Run: python3 scripts/train_robot_arm.py --algorithm MBPO")
    print("   3. Or run: python3 scripts/ros_mbpo_integration.py")
    
    return True

if __name__ == '__main__':
    main()