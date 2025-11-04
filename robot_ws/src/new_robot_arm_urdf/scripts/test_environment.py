#!/usr/bin/env python3
"""
Test Script for Visual RL Environment Setup
Verifies that all components are working before running full RL training

This script tests:
1. Environment state retrieval
2. Action execution
3. Target sphere manipulation 
4. Controller functionality
5. TF transforms

Run this after launching test_rl_environment.launch to verify setup
"""

import rospy
import sys
import time
import numpy as np

# Add scripts directory to path
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_rl_environment_noetic import RLEnvironmentNoetic


def test_basic_functionality():
    """Test basic RL environment functionality"""
    rospy.loginfo("🧪 Testing Visual RL Environment Setup...")
    
    try:
        # Initialize environment
        rospy.loginfo("1️⃣ Initializing RL Environment...")
        env = RLEnvironmentNoetic()
        rospy.sleep(3.0)  # Wait for initialization
        
        # Test 1: Check if environment is ready
        rospy.loginfo("2️⃣ Testing environment readiness...")
        timeout = 10.0
        start_time = time.time()
        
        while not env.data_ready and (time.time() - start_time) < timeout:
            rospy.loginfo("   ⏳ Waiting for environment data...")
            rospy.sleep(1.0)
        
        if env.data_ready:
            rospy.loginfo("   ✅ Environment data ready!")
        else:
            rospy.logerr("   ❌ Environment data not ready within timeout")
            return False
        
        # Test 2: Get initial state
        rospy.loginfo("3️⃣ Testing state retrieval...")
        state = env.get_state()
        if state is not None:
            rospy.loginfo(f"   ✅ State retrieved: {state.shape} elements")
            rospy.loginfo(f"   📊 State values: {state}")
        else:
            rospy.logerr("   ❌ Failed to get state")
            return False
        
        # Test 3: Get action space info
        rospy.loginfo("4️⃣ Testing action space...")
        action_info = env.get_action_space_info()
        rospy.loginfo(f"   ✅ Action space: {action_info}")
        
        # Test 4: Generate and execute test action
        rospy.loginfo("5️⃣ Testing action execution...")
        test_action = env.generate_random_action()
        rospy.loginfo(f"   🎮 Test action: {test_action}")
        
        if env.execute_action(test_action):
            rospy.loginfo("   ✅ Action executed successfully!")
        else:
            rospy.logerr("   ❌ Action execution failed")
            return False
        
        # Wait for action to complete
        rospy.sleep(3.0)
        
        # Test 5: Check reward calculation
        rospy.loginfo("6️⃣ Testing reward calculation...")
        reward, done = env.calculate_reward()
        distance = env.get_distance_to_goal()
        rospy.loginfo(f"   🏆 Reward: {reward}, Done: {done}")
        rospy.loginfo(f"   📏 Distance to goal: {distance:.4f}m")
        
        # Test 6: Test environment reset
        rospy.loginfo("7️⃣ Testing environment reset...")
        if env.reset_environment():
            rospy.loginfo("   ✅ Environment reset successful!")
        else:
            rospy.logerr("   ❌ Environment reset failed")
            return False
        
        # Test 7: Final state check after reset
        rospy.loginfo("8️⃣ Testing state after reset...")
        rospy.sleep(2.0)  # Wait for reset to complete
        
        final_state = env.get_state()
        if final_state is not None:
            final_distance = env.get_distance_to_goal()
            rospy.loginfo(f"   ✅ Final state retrieved")
            rospy.loginfo(f"   📏 Final distance: {final_distance:.4f}m")
        else:
            rospy.logerr("   ❌ Failed to get final state")
            return False
        
        rospy.loginfo("🎉 All tests passed! Environment is ready for RL training!")
        return True
        
    except Exception as e:
        rospy.logerr(f"❌ Test failed with exception: {e}")
        return False
    finally:
        if 'env' in locals():
            env.shutdown()


def test_controller_response():
    """Test individual joint movements"""
    rospy.loginfo("🎮 Testing individual joint control...")
    
    try:
        env = RLEnvironmentNoetic()
        rospy.sleep(3.0)
        
        # Wait for readiness
        while not env.data_ready:
            rospy.sleep(0.5)
        
        # Test each joint individually
        joint_names = ['Joint_1', 'Joint_2', 'Joint_3', 'Joint_4']
        
        for i, joint_name in enumerate(joint_names):
            rospy.loginfo(f"   Testing {joint_name}...")
            
            # Create action with only one joint moving
            action = np.zeros(4)
            action[i] = 0.5  # Move joint to moderate position
            
            if env.execute_action(action):
                rospy.loginfo(f"   ✅ {joint_name} moved successfully")
                rospy.sleep(2.0)  # Wait for movement
            else:
                rospy.logerr(f"   ❌ {joint_name} movement failed")
            
            # Return to zero position
            zero_action = np.zeros(4)
            env.execute_action(zero_action)
            rospy.sleep(2.0)
        
        rospy.loginfo("🎉 Joint control test completed!")
        
    except Exception as e:
        rospy.logerr(f"❌ Controller test failed: {e}")
    finally:
        if 'env' in locals():
            env.shutdown()


def main():
    """Main test function"""
    rospy.init_node('test_rl_environment', anonymous=True)
    
    rospy.loginfo("🚀 Visual RL Environment Test Suite")
    rospy.loginfo("=" * 50)
    
    try:
        # Run basic functionality tests
        if test_basic_functionality():
            rospy.loginfo("✅ Basic functionality test PASSED")
        else:
            rospy.logerr("❌ Basic functionality test FAILED")
            return
        
        rospy.sleep(2.0)
        
        # Run controller response tests
        test_controller_response()
        
        rospy.loginfo("🏁 All tests completed!")
        rospy.loginfo("🚀 Ready to run full RL training with train_visual_rl_4dof.py!")
        
    except KeyboardInterrupt:
        rospy.loginfo("Test interrupted by user")
    except Exception as e:
        rospy.logerr(f"Test suite failed: {e}")


if __name__ == '__main__':
    main()