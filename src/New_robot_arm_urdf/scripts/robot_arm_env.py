#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class RobotArmEnv:
    """
    ROS-Gazebo Environment for 4-DOF Robot Arm RL Training
    
    This class provides a gym-like interface for reinforcement learning
    with the robot arm in Gazebo simulation.
    """
    
    def __init__(self, headless=False, real_time=False):
        """
        Initialize the robot arm environment
        
        Args:
            headless (bool): Run Gazebo without GUI for faster training
            real_time (bool): Run simulation in real-time vs max speed
        """
        rospy.init_node('robot_arm_rl_env', anonymous=True)
        
        # Environment parameters
        self.n_actions = 4  # 4 joints
        self.n_observations = 12  # 4 joint positions + 4 joint velocities + 4 joint efforts
        self.max_episode_steps = 1000
        self.current_step = 0
        
        # Joint limits (radians) - adjust based on your robot
        self.joint_limits = {
            'Joint_1': [-np.pi, np.pi],
            'Joint_2': [-np.pi/2, np.pi/2],
            'Joint_3': [-np.pi/2, np.pi/2], 
            'Joint_4': [-np.pi/2, np.pi/2]
        }
        
        # Target position (end-effector goal)
        self.target_position = np.array([0.2, 0.0, 0.3])  # x, y, z
        self.position_tolerance = 0.05  # meters
        
        # ROS Publishers for joint control
        self.joint_publishers = {
            'Joint_1': rospy.Publisher('/New/Joint_1_position_controller/command', Float64, queue_size=1),
            'Joint_2': rospy.Publisher('/New/Joint_2_position_controller/command', Float64, queue_size=1),
            'Joint_3': rospy.Publisher('/New/Joint_3_position_controller/command', Float64, queue_size=1),
            'Joint_4': rospy.Publisher('/New/Joint_4_position_controller/command', Float64, queue_size=1)
        }
        
        # ROS Subscriber for joint states
        self.joint_state_sub = rospy.Subscriber('/New/joint_states', JointState, self.joint_state_callback)
        
        # Gazebo services
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/get_model_state')
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
        # TF2 for end-effector position
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # State variables
        self.joint_positions = np.zeros(4)
        self.joint_velocities = np.zeros(4) 
        self.joint_efforts = np.zeros(4)
        self.end_effector_pos = np.zeros(3)
        
        # Wait for initial joint state
        rospy.loginfo("Waiting for joint states...")
        rospy.wait_for_message('/New/joint_states', JointState)
        rospy.sleep(1.0)
        rospy.loginfo("Robot Arm RL Environment initialized!")
        
    def joint_state_callback(self, msg):
        """Callback for joint state updates"""
        # Map joint states to our joint order
        joint_names = ['Joint_1', 'Joint_2', 'Joint_3', 'Joint_4']
        
        for i, joint_name in enumerate(joint_names):
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                self.joint_positions[i] = msg.position[idx]
                self.joint_velocities[i] = msg.velocity[idx] if msg.velocity else 0.0
                self.joint_efforts[i] = msg.effort[idx] if msg.effort else 0.0
                
    def get_end_effector_position(self):
        """Get end-effector position using TF2"""
        try:
            # Get transform from base_link to end-effector (link_4_1)
            transform = self.tf_buffer.lookup_transform('base_link', 'link_4_1', rospy.Time(0))
            pos = transform.transform.translation
            return np.array([pos.x, pos.y, pos.z])
        except Exception as e:
            rospy.logwarn(f"Could not get end-effector position: {e}")
            return self.end_effector_pos  # Return last known position
            
    def reset(self):
        """
        Reset the environment to initial state
        
        Returns:
            observation (np.array): Initial observation
        """
        self.current_step = 0
        
        # Reset robot to home position
        home_position = [0.0, 0.0, 0.0, 0.0]  # All joints at zero
        self.set_joint_positions(home_position)
        
        # Wait for robot to settle
        rospy.sleep(1.0)
        
        # Update end-effector position
        self.end_effector_pos = self.get_end_effector_position()
        
        # Randomize target position for variety
        self.randomize_target()
        
        return self.get_observation()
        
    def step(self, action):
        """
        Execute one step in the environment
        
        Args:
            action (np.array): Joint position commands (4 values)
            
        Returns:
            observation (np.array): New observation
            reward (float): Reward for this step
            done (bool): Whether episode is finished
            info (dict): Additional information
        """
        self.current_step += 1
        
        # Clip actions to joint limits
        clipped_action = self.clip_actions(action)
        
        # Send commands to robot
        self.set_joint_positions(clipped_action)
        
        # Wait for one control step
        rospy.sleep(0.1)  # 10 Hz control frequency
        
        # Update end-effector position
        self.end_effector_pos = self.get_end_effector_position()
        
        # Calculate reward
        reward = self.calculate_reward()
        
        # Check if episode is done
        done = self.is_done()
        
        # Additional info
        info = {
            'end_effector_pos': self.end_effector_pos.copy(),
            'target_pos': self.target_position.copy(),
            'distance_to_target': np.linalg.norm(self.end_effector_pos - self.target_position),
            'step': self.current_step
        }
        
        return self.get_observation(), reward, done, info
        
    def get_observation(self):
        """
        Get current observation state
        
        Returns:
            observation (np.array): Current state observation
        """
        # Combine joint positions, velocities, and target position
        obs = np.concatenate([
            self.joint_positions,           # 4 values
            self.joint_velocities,          # 4 values  
            self.target_position,           # 3 values
            self.end_effector_pos          # 3 values
        ])
        return obs.astype(np.float32)
        
    def calculate_reward(self):
        """Calculate reward based on distance to target and other factors"""
        # Distance reward
        distance = np.linalg.norm(self.end_effector_pos - self.target_position)
        distance_reward = -distance  # Negative distance (closer = higher reward)
        
        # Success bonus
        success_bonus = 100.0 if distance < self.position_tolerance else 0.0
        
        # Joint velocity penalty (encourage smooth motion)
        velocity_penalty = -0.01 * np.sum(np.abs(self.joint_velocities))
        
        # Joint limit penalty
        limit_penalty = self.get_joint_limit_penalty()
        
        total_reward = distance_reward + success_bonus + velocity_penalty + limit_penalty
        
        return total_reward
        
    def is_done(self):
        """Check if episode should terminate"""
        # Success condition
        distance = np.linalg.norm(self.end_effector_pos - self.target_position)
        if distance < self.position_tolerance:
            rospy.loginfo(f"Target reached! Distance: {distance:.4f}")
            return True
            
        # Max steps reached
        if self.current_step >= self.max_episode_steps:
            return True
            
        # Joint limits violated severely
        if self.check_joint_limits_violation():
            rospy.logwarn("Joint limits violated!")
            return True
            
        return False
        
    def set_joint_positions(self, positions):
        """Send position commands to all joints"""
        joint_names = ['Joint_1', 'Joint_2', 'Joint_3', 'Joint_4']
        
        for i, joint_name in enumerate(joint_names):
            if i < len(positions):
                msg = Float64()
                msg.data = float(positions[i])
                self.joint_publishers[joint_name].publish(msg)
                
    def clip_actions(self, actions):
        """Clip actions to joint limits"""
        joint_names = ['Joint_1', 'Joint_2', 'Joint_3', 'Joint_4']
        clipped = []
        
        for i, joint_name in enumerate(joint_names):
            if i < len(actions):
                min_val, max_val = self.joint_limits[joint_name]
                clipped_val = np.clip(actions[i], min_val, max_val)
                clipped.append(clipped_val)
                
        return clipped
        
    def get_joint_limit_penalty(self):
        """Calculate penalty for being near joint limits"""
        penalty = 0.0
        joint_names = ['Joint_1', 'Joint_2', 'Joint_3', 'Joint_4']
        
        for i, joint_name in enumerate(joint_names):
            min_val, max_val = self.joint_limits[joint_name]
            pos = self.joint_positions[i]
            
            # Penalty increases as we approach limits
            range_size = max_val - min_val
            if pos < min_val + 0.1 * range_size or pos > max_val - 0.1 * range_size:
                penalty -= 1.0
                
        return penalty
        
    def check_joint_limits_violation(self):
        """Check if any joint has severely violated limits"""
        joint_names = ['Joint_1', 'Joint_2', 'Joint_3', 'Joint_4']
        
        for i, joint_name in enumerate(joint_names):
            min_val, max_val = self.joint_limits[joint_name] 
            pos = self.joint_positions[i]
            
            if pos < min_val - 0.1 or pos > max_val + 0.1:
                return True
                
        return False
        
    def randomize_target(self):
        """Randomize target position within reachable workspace"""
        # Define workspace bounds (adjust based on your robot's reach)
        x_range = [0.1, 0.3]
        y_range = [-0.2, 0.2] 
        z_range = [0.1, 0.4]
        
        self.target_position = np.array([
            np.random.uniform(*x_range),
            np.random.uniform(*y_range), 
            np.random.uniform(*z_range)
        ])
        
        rospy.loginfo(f"New target: {self.target_position}")
        
    def close(self):
        """Clean up resources"""
        rospy.loginfo("Closing Robot Arm Environment")
        rospy.signal_shutdown("Environment closed")


if __name__ == '__main__':
    # Test the environment
    env = RobotArmEnv()
    
    try:
        # Reset environment
        obs = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial observation: {obs}")
        
        # Run a few random actions
        for i in range(10):
            action = np.random.uniform(-0.1, 0.1, 4)  # Small random actions
            obs, reward, done, info = env.step(action)
            
            print(f"Step {i+1}:")
            print(f"  Action: {action}")
            print(f"  Reward: {reward:.4f}")
            print(f"  Distance to target: {info['distance_to_target']:.4f}")
            print(f"  Done: {done}")
            
            if done:
                print("Episode finished!")
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        env.close()