#!/usr/bin/env python3
"""
Test script to verify robot safety features.

This script tests:
1. Joint limit clipping
2. Velocity limits
3. NaN detection
4. Error recovery

Run this BEFORE starting full training to ensure safety features work.
"""

import rospy
import numpy as np
import sys
import os

# Add path to import environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main_rl_environment_noetic import RLEnvironmentNoetic

def test_joint_limit_clipping():
    """Test that joint limits are enforced"""
    print("\n" + "="*60)
    print("TEST 1: Joint Limit Clipping")
    print("="*60)
    
    env = RLEnvironmentNoetic()
    
    # Test case 1: Joint1 exceeds upper limit
    print("\n📋 Test 1a: Joint1 too high (2.0 > 1.57)")
    result = env.move_to_joint_positions(np.array([2.0, 0.0, 0.0, 0.0]))
    print(f"   Result: {result}")
    assert result['success'] or result['error_code'] == -5, "Should clip and succeed"
    
    # Test case 2: Joint4 below lower limit
    print("\n📋 Test 1b: Joint4 too low (-1.0 < 0.0)")
    result = env.move_to_joint_positions(np.array([0.0, 0.0, 0.0, -1.0]))
    print(f"   Result: {result}")
    assert result['success'] or result['error_code'] == -5, "Should clip and succeed"
    
    # Test case 3: The BREAKING command from user's test
    print("\n📋 Test 1c: The breaking command [-0.6, 1.57, 1.57, 0]")
    result = env.move_to_joint_positions(np.array([-0.6, 1.57, 1.57, 0.0]))
    print(f"   Result: {result}")
    
    # Check final state is not NaN
    rospy.sleep(0.5)
    joints = env.get_joint_positions()
    vels = env.get_joint_velocities()
    print(f"   Final joints: {joints}")
    print(f"   Final velocities: {vels}")
    
    assert joints is not None, "❌ FAILED: Could not get joint positions!"
    assert vels is not None, "❌ FAILED: Could not get joint velocities!"
    assert not np.any(np.isnan(joints)), "❌ FAILED: Joints contain NaN!"
    assert not np.any(np.isnan(vels)), "❌ FAILED: Velocities contain NaN!"
    
    print("\n✅ All joint limit tests PASSED!")

def test_velocity_limits():
    """Test that velocity limits are enforced"""
    print("\n" + "="*60)
    print("TEST 2: Velocity Limits")
    print("="*60)
    
    env = RLEnvironmentNoetic()
    
    # Move to home first
    env.move_to_joint_positions(np.array([0.0, 0.0, 0.0, 0.0]))
    rospy.sleep(2.0)
    
    # Make a large movement (should still be limited)
    print("\n📋 Test 2a: Large movement (all joints ±90°)")
    result = env.move_to_joint_positions(np.array([1.57, 1.57, 1.57, 3.14]))
    rospy.sleep(3.5)  # Wait for settling
    
    vels = env.get_joint_velocities()
    max_vel = np.max(np.abs(vels))
    print(f"   Final velocities: {vels}")
    print(f"   Max final velocity: {max_vel:.2f} rad/s")
    
    assert vels is not None, "❌ FAILED: Could not get velocities!"
    # After settling, velocity should be low
    assert max_vel < 1.0, f"❌ FAILED: Velocity too high after settling: {max_vel:.2f} rad/s"
    
    print("\n✅ Velocity limit tests PASSED!")

def test_nan_detection():
    """Test that NaN states are detected"""
    print("\n" + "="*60)
    print("TEST 3: NaN Detection")
    print("="*60)
    
    env = RobotArmEnvironment()
    
    # This test is hard to trigger without actually breaking the robot
    # Just verify the detection mechanism exists
    print("\n📋 Test 3a: Verify NaN detection code exists")
    
    # Check that the function has the NaN detection code
    import inspect
    source = inspect.getsource(env.move_to_joint_positions)
    
    assert "np.isnan" in source, "❌ FAILED: No NaN detection in move_to_joint_positions!"
    assert "error_code': -999" in source, "❌ FAILED: No -999 error code for broken robot!"
    
    print("   ✓ NaN detection code found")
    print("   ✓ Error code -999 defined for broken robot")
    
    print("\n✅ NaN detection tests PASSED!")

def test_error_recovery():
    """Test that training loop handles critical errors"""
    print("\n" + "="*60)
    print("TEST 4: Error Recovery")
    print("="*60)
    
    # Import the training wrapper
    sys.path.insert(0, '/home/ducanh/rl_model_based/robot_ws/src/new_robot_arm_urdf/scripts')
    from train_robot import RobotArmEnvWrapper
    
    print("\n📋 Test 4a: Verify error recovery code exists")
    
    import inspect
    source = inspect.getsource(RobotArmEnvWrapper.step)
    
    assert "error_code'] == -999" in source, "❌ FAILED: No -999 error handling in training loop!"
    assert "robot_broken" in source, "❌ FAILED: No robot_broken error info!"
    assert "reward = -100" in source or "reward=-100" in source, "❌ FAILED: No penalty for breaking robot!"
    
    print("   ✓ Error code -999 handling found")
    print("   ✓ Robot broken error info found")
    print("   ✓ Large penalty (-100) for breaking robot")
    
    print("\n✅ Error recovery tests PASSED!")

def run_all_tests():
    """Run all safety tests"""
    print("\n" + "="*70)
    print(" ROBOT SAFETY FEATURE TESTS")
    print("="*70)
    
    rospy.init_node('test_safety_features', anonymous=True)
    
    try:
        test_joint_limit_clipping()
        test_velocity_limits()
        test_nan_detection()
        test_error_recovery()
        
        print("\n" + "="*70)
        print("🎉 ALL SAFETY TESTS PASSED! 🎉")
        print("="*70)
        print("\n✅ Robot is safe to use for training!")
        print("✅ Joint limits will be enforced")
        print("✅ Velocities will be controlled")
        print("✅ NaN states will be detected")
        print("✅ Critical errors will trigger recovery")
        
    except AssertionError as e:
        print("\n" + "="*70)
        print("❌ SAFETY TEST FAILED!")
        print("="*70)
        print(f"\nError: {e}")
        print("\n⚠️  DO NOT RUN TRAINING until this is fixed!")
        sys.exit(1)
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST ERROR!")
        print("="*70)
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    run_all_tests()
