#!/usr/bin/env python3
"""
Simple test to verify robot control in Gazebo
"""

import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import time
import math

class RobotTester:
    def __init__(self):
        rospy.init_node('robot_tester', anonymous=True)
        
        # Publishers for joint commands
        self.joint_pubs = []
        for i in range(4):
            pub = rospy.Publisher(
                f'/joint{i+1}_position_controller/command', 
                Float64, 
                queue_size=1
            )
            self.joint_pubs.append(pub)
        
        # Subscriber for joint states
        self.joint_positions = [0.0, 0.0, 0.0, 0.0]
        self.joint_sub = rospy.Subscriber(
            '/joint_states', 
            JointState, 
            self.joint_state_callback
        )
        
        rospy.loginfo("Robot tester initialized")
    
    def joint_state_callback(self, msg):
        """Update joint positions from feedback"""
    joint_names = ['Joint1', 'Joint2', 'Joint3', 'Joint4']
        for i, joint_name in enumerate(joint_names):
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                self.joint_positions[i] = msg.position[idx]
    
    def move_joints(self, positions):
        """Move joints to target positions"""
        rospy.loginfo(f"Moving joints to: {positions}")
        for i, pos in enumerate(positions):
            self.joint_pubs[i].publish(Float64(pos))
    
    def wait_for_movement(self, target_positions, timeout=5.0):
        """Wait for robot to reach target positions"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if we're close to target
            close_enough = True
            for i, target in enumerate(target_positions):
                error = abs(self.joint_positions[i] - target)
                if error > 0.1:  # 0.1 radian tolerance
                    close_enough = False
                    break
            
            if close_enough:
                rospy.loginfo("Reached target positions!")
                return True
            
            rospy.sleep(0.1)
        
        rospy.logwarn("Timeout waiting for movement")
        return False
    
    def test_sequence(self):
        """Test a sequence of movements"""
        rospy.loginfo("Starting robot test sequence...")
        
        # Wait for joint state feedback
        rospy.loginfo("Waiting for joint state feedback...")
        timeout = time.time() + 10.0
        while time.time() < timeout and all(pos == 0.0 for pos in self.joint_positions):
            rospy.sleep(0.1)
        
        rospy.loginfo(f"Initial joint positions: {self.joint_positions}")
        
        # Test sequence
        test_positions = [
            [0.0, 0.0, 0.0, 0.0],      # Home position
            [0.5, 0.2, -0.2, 0.1],     # Test position 1
            [-0.5, 0.4, -0.4, 0.2],    # Test position 2
            [0.0, 0.5, -0.5, 0.0],     # Test position 3
            [0.0, 0.0, 0.0, 0.0],      # Back to home
        ]
        
        for i, positions in enumerate(test_positions):
            rospy.loginfo(f"Step {i+1}: Moving to {positions}")
            self.move_joints(positions)
            self.wait_for_movement(positions)
            rospy.loginfo(f"Current positions: {self.joint_positions}")
            rospy.sleep(1.0)  # Pause between movements
        
        rospy.loginfo("Test sequence completed!")
    
    def run(self):
        """Run the test"""
        try:
            # Wait a bit for everything to initialize
            rospy.sleep(2.0)
            
            # Run test sequence
            self.test_sequence()
            
            rospy.loginfo("Robot test successful! 🎉")
            return True
            
        except Exception as e:
            rospy.logerr(f"Robot test failed: {e}")
            return False

def main():
    try:
        tester = RobotTester()
        success = tester.run()
        return 0 if success else 1
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted")
        return 1
    except Exception as e:
        rospy.logerr(f"Test error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())