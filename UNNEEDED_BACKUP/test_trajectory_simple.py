#!/usr/bin/env python3

"""
Simple trajectory test to debug the controller issue.
Test basic trajectory execution with proper timing.
"""

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

def test_simple_trajectory():
    """Test a simple trajectory to debug the controller."""
    rospy.init_node('simple_trajectory_test')
    
    # Connect to action server
    client = actionlib.SimpleActionClient('/joint_trajectory_controller/follow_joint_trajectory', 
                                          FollowJointTrajectoryAction)
    
    print("Waiting for trajectory action server...")
    if not client.wait_for_server(timeout=rospy.Duration(10.0)):
        print("❌ Failed to connect to trajectory action server!")
        return False
    
    print("✅ Connected to trajectory action server")
    
    # Create a simple trajectory - move joint 1 by a small amount
    point = JointTrajectoryPoint()
    point.positions = [0.1, 0.0, 0.0, 0.0]  # Move joint 1 by 0.1 radians
    point.velocities = [0.0] * 4
    point.accelerations = [0.0] * 4
    point.time_from_start = rospy.Duration(2.0)  # Take 2 seconds
    
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.1)
    goal.trajectory.joint_names = ['Joint1', 'Joint2', 'Joint3', 'Joint4']
    goal.trajectory.points = [point]
    goal.goal_time_tolerance = rospy.Duration(3.0)
    
    print("Sending simple trajectory goal...")
    client.send_goal_and_wait(goal, rospy.Duration(5.0))
    
    result = client.get_result()
    if result and result.error_code == 0:
        print("✅ Trajectory executed successfully!")
        return True
    else:
        print(f"❌ Trajectory failed with error code: {result.error_code if result else 'No result'}")
        return False

if __name__ == '__main__':
    try:
        success = test_simple_trajectory()
        print(f"Test result: {'PASS' if success else 'FAIL'}")
    except Exception as e:
        print(f"Test failed with exception: {e}")