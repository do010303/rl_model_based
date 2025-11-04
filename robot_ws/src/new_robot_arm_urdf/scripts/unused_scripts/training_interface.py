#!/usr/bin/env python3
"""
Visual Training Interface for Robot Arm Target Reaching
======================================================

This node manages:
- Target sphere position updates
- Visual feedback for training progress
- End effector position tracking
- Distance calculations and rewards
"""

import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import random
from geometry_msgs.msg import Pose, Point, TransformStamped
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Bool, String
from visualization_msgs.msg import Marker
import tf.transformations as tf_trans

class VisualTrainingInterface:
    def __init__(self):
        rospy.init_node('visual_training_interface', anonymous=False)
        
        # TF setup for getting actual end effector position
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Services for Gazebo model manipulation
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/get_model_state')
        
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
        # Publishers
        self.target_pos_pub = rospy.Publisher('/training/target_position', Point, queue_size=1)
        self.ee_pos_pub = rospy.Publisher('/training/end_effector_position', Point, queue_size=1)
        self.distance_pub = rospy.Publisher('/training/distance_to_target', Float64, queue_size=1)
        self.success_pub = rospy.Publisher('/training/success', Bool, queue_size=1)
        self.episode_info_pub = rospy.Publisher('/training/episode_info', String, queue_size=1)
        
        # Subscribers
        self.joint_states_sub = rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        self.reset_target_sub = rospy.Subscriber('/training/reset_target', Bool, self.reset_target_callback)
        
        # Robot parameters (4-DOF arm)
        self.joint_names = ['Joint_1', 'Joint_2', 'Joint_3', 'Joint_4']
        self.joint_positions = [0.0, 0.0, 0.0, 0.0]
        
        # Link lengths for forward kinematics (approximate)
        self.link_lengths = [0.1, 0.1, 0.1, 0.08]  # Base height, link1, link2, link3+ee
        
        # Target management - initial position within cylinder
        self.target_position = [0.20, 0.0, 0.12]  # Safe initial position
        # Cylindrical workspace parameters (for reference, actual sampling in reset_target)
        self.cylinder_center = [0.10, 0.0]  # X,Y center of workspace cylinder
        self.cylinder_radius = 0.18          # Maximum reach radius  
        self.min_radius = 0.06               # Minimum radius (avoid base collision)
        self.z_limits = [0.05, 0.20]        # Height range
        
        # Keep old format for compatibility (though now using cylindrical sampling)
        self.workspace_limits = {
            'x': [0.06, 0.28],     # Bounding box for cylinder
            'y': [-0.18, 0.18],    # Bounding box for cylinder  
            'z': [0.05, 0.20]      # Height range
        }
        
        # Training statistics
        self.episode_count = 0
        self.success_count = 0
        self.success_threshold = 0.08  # Increased to 8cm for easier success
        self.last_distance = float('inf')
        
        # Smoothing for stable visualization
        self.position_history = []
        self.max_history = 5
        
        # Initialize target position
        rospy.sleep(2.0)  # Wait for Gazebo to fully load
        self.reset_target()
        
        rospy.loginfo("Visual Training Interface initialized with stable control!")
        
    def joint_states_callback(self, msg):
        """Update joint positions and calculate end effector position"""
        if len(msg.name) >= 4:
            # Update joint positions
            for i, name in enumerate(self.joint_names):
                if name in msg.name:
                    idx = msg.name.index(name)
                    if idx < len(msg.position):
                        self.joint_positions[i] = msg.position[idx]
            
            # Get end effector position using TF from actual robot state
            ee_pos = self.get_end_effector_position()
            
            # Update end effector visual marker
            self.update_end_effector_marker(ee_pos)
            
            # Calculate distance to target
            distance = self.calculate_distance(ee_pos, self.target_position)
            
            # Publish training information
            self.publish_training_info(ee_pos, distance)
            
            # Check for success
            if distance < self.success_threshold:
                self.handle_success()
    
    def get_end_effector_position(self):
        """
        Get actual end effector position from TF - this ensures the blue marker 
        tracks the real robot end effector (link_4_1) position
        """
        try:
            # Get transform from base_link to link_4_1 (end effector)
            # Try different possible frame names for the end effector
            possible_frames = ['link_4_1', 'Link_4_1', 'link4', 'end_effector']
            transform = None
            
            for frame_name in possible_frames:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        'base_link',  # target frame  
                        frame_name,   # source frame (end effector)
                        rospy.Time(),  # get latest
                        rospy.Duration(0.1)  # timeout
                    )
                    break  # Success, stop trying
                except:
                    continue  # Try next frame name
            
            if transform is not None:
                # Extract position from transform
                x = transform.transform.translation.x
                y = transform.transform.translation.y 
                z = transform.transform.translation.z
                return [x, y, z]
            else:
                # No TF available, use fallback calculation
                return self.calculate_end_effector_fallback()
            
        except Exception as e:
            rospy.logwarn_throttle(10.0, f"Failed to get end effector position via TF: {e}")
            # Fallback to approximate calculation
            return self.calculate_end_effector_fallback()
    
    def calculate_end_effector_fallback(self):
        """Fallback method to estimate end effector position when TF is not available"""
        # Simple approximation based on joint angles
        q1, q2, q3, q4 = self.joint_positions
        
        # Rough estimate using basic trigonometry for a 4DOF arm
        # These are approximate link lengths in meters
        base_height = 0.033
        link1_length = 0.052
        link2_length = 0.063
        link3_length = 0.052
        
        # Simple forward kinematics approximation
        x = (link1_length + link2_length * np.cos(q2) + link3_length * np.cos(q2 + q3)) * np.cos(q1)
        y = (link1_length + link2_length * np.cos(q2) + link3_length * np.cos(q2 + q3)) * np.sin(q1)
        z = base_height + link2_length * np.sin(q2) + link3_length * np.sin(q2 + q3)
        
        return [x, y, z]
    
    def update_end_effector_marker(self, ee_pos):
        """Update the blue end effector sphere position in Gazebo with smoothing"""
        try:
            # Add position to history for smoothing
            self.position_history.append(ee_pos.copy())
            if len(self.position_history) > self.max_history:
                self.position_history.pop(0)
            
            # Calculate smoothed position (moving average)
            if len(self.position_history) > 1:
                smoothed_pos = np.mean(self.position_history, axis=0)
            else:
                smoothed_pos = ee_pos
            
            model_state = ModelState()
            model_state.model_name = "end_effector_sphere"
            model_state.pose.position.x = smoothed_pos[0]
            model_state.pose.position.y = smoothed_pos[1] 
            model_state.pose.position.z = smoothed_pos[2]
            model_state.pose.orientation.w = 1.0
            
            self.set_model_state(model_state)
        except Exception as e:
            rospy.logwarn(f"Failed to update end effector marker: {e}")
    
    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two 3D points"""
        return np.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))
    
    def publish_training_info(self, ee_pos, distance):
        """Publish training information for the RL agent"""
        # End effector position
        ee_msg = Point()
        ee_msg.x, ee_msg.y, ee_msg.z = ee_pos
        self.ee_pos_pub.publish(ee_msg)
        
        # Target position  
        target_msg = Point()
        target_msg.x, target_msg.y, target_msg.z = self.target_position
        self.target_pos_pub.publish(target_msg)
        
        # Distance to target
        distance_msg = Float64()
        distance_msg.data = distance
        self.distance_pub.publish(distance_msg)
        
        # Store for reward calculation
        self.last_distance = distance
    
    def reset_target_callback(self, msg):
        """Reset target to a new random position"""
        if msg.data:
            self.reset_target()
    
    def reset_target(self):
        """Move target sphere to a new random position within cylindrical workspace"""
        # Cylindrical workspace parameters (matching visual cylinder)
        cylinder_center_x = 0.10  # Cylinder center X
        cylinder_center_y = 0.0   # Cylinder center Y
        cylinder_radius = 0.18    # Maximum radius
        min_radius = 0.06         # Minimum radius (avoid robot base)
        z_min = 0.05              # Minimum height
        z_max = 0.20              # Maximum height
        
        # Generate random position within cylinder using polar coordinates
        # Sample radius with proper distribution (sqrt for uniform area distribution)
        r = random.uniform(min_radius**2, cylinder_radius**2)
        r = np.sqrt(r)
        
        # Sample angle uniformly
        theta = random.uniform(0, 2 * np.pi)
        
        # Convert to Cartesian coordinates
        x = cylinder_center_x + r * np.cos(theta)
        y = cylinder_center_y + r * np.sin(theta)
        z = random.uniform(z_min, z_max)
        
        self.target_position = [x, y, z]
        
        # Update target sphere position in Gazebo
        try:
            model_state = ModelState()
            model_state.model_name = "target_sphere"
            model_state.pose.position.x = x
            model_state.pose.position.y = y
            model_state.pose.position.z = z
            model_state.pose.orientation.w = 1.0
            
            self.set_model_state(model_state)
            
            rospy.loginfo(f"Target reset to: ({x:.3f}, {y:.3f}, {z:.3f})")
            
        except Exception as e:
            rospy.logerr(f"Failed to reset target: {e}")
    
    def handle_success(self):
        """Handle successful target reaching"""
        self.success_count += 1
        
        # Publish success
        success_msg = Bool()
        success_msg.data = True
        self.success_pub.publish(success_msg)
        
        # Log success
        rospy.loginfo(f"🎯 TARGET REACHED! Success {self.success_count}")
        
        # Reset target after short delay
        rospy.Timer(rospy.Duration(1.0), self.delayed_reset, oneshot=True)
    
    def delayed_reset(self, event):
        """Reset target after a delay"""
        self.reset_target()
        self.episode_count += 1
        
        # Publish episode info
        info_msg = String()
        success_rate = (self.success_count / max(1, self.episode_count)) * 100
        info_msg.data = f"Episode: {self.episode_count}, Successes: {self.success_count}, Success Rate: {success_rate:.1f}%"
        self.episode_info_pub.publish(info_msg)

if __name__ == '__main__':
    try:
        interface = VisualTrainingInterface()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass