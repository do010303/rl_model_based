"""
Gazebo-integrated 4-DOF Robot Arm Environment for Reinforcement Learning
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty
import threading
import time
from typing import Dict, Any, Tuple, Optional
import math

class GazeboRobot4DOFEnv(gym.Env):
    """
    Gazebo-integrated 4-DOF Robot Arm Environment for goal-reaching tasks.
    
    Communicates with Gazebo simulation via ROS topics and services.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Configuration
        self.config = config or {}
        self.use_ros = self.config.get('use_ros', True)
        
        # Robot parameters (from real URDF)
        self.dof = 4
        self.link_lengths = np.array([0.033399, 0.053, 0.063, 0.053])
        self.joint_limits = np.array([
            [-np.pi, np.pi],        # Joint 1: Base rotation ±180°
            [-np.pi/2, np.pi/2],    # Joint 2: Shoulder ±90°
            [-np.pi/2, np.pi/2],    # Joint 3: Elbow ±90°
            [-np.pi/2, np.pi/2]     # Joint 4: Wrist ±90°
        ])
        
        # Environment parameters
        self.max_steps = self.config.get('max_steps', 200)
        self.success_distance = self.config.get('success_distance', 0.02)
        self.workspace_center = np.array([0.0, 0.0, 0.1])
        self.workspace_radius = self.config.get('workspace_radius', 0.15)
        
        # Reward parameters
        self.dense_reward = self.config.get('dense_reward', True)
        self.success_reward = self.config.get('success_reward', 100.0)
        self.distance_reward_scale = self.config.get('distance_reward_scale', -1.0)
        self.action_penalty_scale = self.config.get('action_penalty_scale', -0.01)
        
        # State variables
        self.joint_positions = np.zeros(self.dof)
        self.joint_velocities = np.zeros(self.dof)
        self.end_effector_pos = np.zeros(3)
        self.target_position = np.zeros(3)
        self.step_count = 0
        self.state_lock = threading.Lock()
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.dof,), dtype=np.float32
        )
        
        # Observation: [joint_pos(4), joint_vel(4), end_effector_pos(3), target_pos(3)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        
        # ROS initialization
        if self.use_ros:
            self._init_ros()
        
        # Initialize
        self.reset()
    
    def _init_ros(self):
        """Initialize ROS publishers, subscribers, and services"""
        try:
            # Check if ROS node exists, if not create one
            if not rospy.get_node_uri():
                rospy.init_node('gazebo_robot_env', anonymous=True)
            
            # Publishers for joint commands
            self.joint_cmd_pubs = []
            for i in range(self.dof):
                pub = rospy.Publisher(
                    f'/joint{i+1}_position_controller/command',
                    Float64,
                    queue_size=1
                )
                self.joint_cmd_pubs.append(pub)
            
            # Subscriber for joint states
            self.joint_state_sub = rospy.Subscriber(
                '/joint_states',
                JointState,
                self._joint_state_callback
            )
            
            # Service clients
            try:
                rospy.wait_for_service('/robot/reset', timeout=5.0)
                self.reset_robot_srv = rospy.ServiceProxy('/robot/reset', Empty)
            except rospy.ROSException:
                rospy.logwarn("Reset service not available, using manual reset")
                self.reset_robot_srv = None
            
            # Wait for initial joint state
            rospy.loginfo("Waiting for initial joint states...")
            timeout = time.time() + 10.0
            while time.time() < timeout and np.allclose(self.joint_positions, 0.0):
                rospy.sleep(0.1)
            
            rospy.loginfo("Gazebo Robot Environment ROS interface initialized")
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize ROS interface: {e}")
            self.use_ros = False
    
    def _joint_state_callback(self, msg):
        """Callback for joint state updates from Gazebo"""
        with self.state_lock:
            try:
                joint_names = ['Joint1', 'Joint2', 'Joint3', 'Joint4']
                for i, joint_name in enumerate(joint_names):
                    if joint_name in msg.name:
                        idx = msg.name.index(joint_name)
                        self.joint_positions[i] = msg.position[idx]
                        if len(msg.velocity) > idx:
                            self.joint_velocities[i] = msg.velocity[idx]
                
                # Update end-effector position
                self.end_effector_pos = self._forward_kinematics(self.joint_positions)
                
            except Exception as e:
                rospy.logwarn(f"Error in joint state callback: {e}")
    
    def _send_joint_commands(self, joint_positions):
        """Send joint position commands to Gazebo"""
        if not self.use_ros:
            return
        
        try:
            for i, pos in enumerate(joint_positions):
                if i < len(self.joint_cmd_pubs):
                    # Clamp to joint limits
                    clamped_pos = np.clip(pos, self.joint_limits[i, 0], self.joint_limits[i, 1])
                    self.joint_cmd_pubs[i].publish(Float64(clamped_pos))
        except Exception as e:
            rospy.logwarn(f"Error sending joint commands: {e}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset step counter
        self.step_count = 0
        
        # Reset to neutral pose
        reset_pose = np.array([0.0, 0.2, -0.3, 0.1])
        
        if self.use_ros:
            # Send reset commands to Gazebo
            self._send_joint_commands(reset_pose)
            
            # Wait for robot to reach reset position
            rospy.sleep(1.5)
            
            # Get actual joint positions from Gazebo
            with self.state_lock:
                self.joint_positions = self.joint_positions.copy()
                self.joint_velocities = self.joint_velocities.copy()
        else:
            # Fallback to internal simulation
            self.joint_positions = reset_pose
            self.joint_velocities = np.zeros(self.dof)
        
        # Calculate end-effector position
        self.end_effector_pos = self._forward_kinematics(self.joint_positions)
        
        # Generate random target position within workspace
        self._generate_target()
        
        # Initialize previous distance for progress tracking
        self.prev_distance = np.linalg.norm(self.end_effector_pos - self.target_position)
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        # Validate action
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            action = np.zeros(self.dof)
        
        action = np.clip(action, -1.0, 1.0)
        
        # Convert action to joint position commands
        # Scale action to reasonable joint position changes
        max_joint_change = 0.1  # radians per step
        joint_position_delta = action * max_joint_change
        
        # Calculate new target joint positions
        new_joint_positions = self.joint_positions + joint_position_delta
        
        # Enforce joint limits
        new_joint_positions = np.clip(
            new_joint_positions,
            self.joint_limits[:, 0],
            self.joint_limits[:, 1]
        )
        
        if self.use_ros:
            # Send commands to Gazebo
            self._send_joint_commands(new_joint_positions)
            
            # Wait for control step
            rospy.sleep(0.05)  # 20 Hz control rate
            
            # Get actual joint states from Gazebo
            with self.state_lock:
                self.joint_positions = self.joint_positions.copy()
                self.joint_velocities = self.joint_velocities.copy()
                self.end_effector_pos = self.end_effector_pos.copy()
        else:
            # Fallback to internal simulation
            self.joint_positions = new_joint_positions
            self.joint_velocities = joint_position_delta / 0.05  # Estimate velocities
            self.end_effector_pos = self._forward_kinematics(self.joint_positions)
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check termination conditions
        distance_to_target = np.linalg.norm(self.end_effector_pos - self.target_position)
        goal_reached = distance_to_target < self.success_distance
        
        self.step_count += 1
        max_steps_reached = self.step_count >= self.max_steps
        
        # Check for invalid states
        invalid_state = (
            np.any(np.isnan(self.joint_positions)) or
            np.any(np.isnan(self.end_effector_pos)) or
            np.isnan(reward) or np.isinf(reward)
        )
        
        terminated = goal_reached or invalid_state
        truncated = max_steps_reached
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info.update({
            'goal_reached': goal_reached,
            'distance_to_target': distance_to_target,
            'invalid_state': invalid_state
        })
        
        return obs, reward, terminated, truncated, info
    
    def _forward_kinematics(self, joint_angles):
        """Compute forward kinematics matching the real robot"""
        q1, q2, q3, q4 = joint_angles
        
        # Base rotation
        c1, s1 = np.cos(q1), np.sin(q1)
        
        # Calculate reach in the x-y plane
        x = (self.link_lengths[1] * np.cos(q2) + 
             self.link_lengths[2] * np.cos(q2 + q3) + 
             self.link_lengths[3] * np.cos(q2 + q3 + q4))
        
        # Apply base rotation
        ee_x = c1 * x
        ee_y = s1 * x
        ee_z = (self.link_lengths[0] + 
                self.link_lengths[1] * np.sin(q2) + 
                self.link_lengths[2] * np.sin(q2 + q3) + 
                self.link_lengths[3] * np.sin(q2 + q3 + q4))
        
        return np.array([ee_x, ee_y, ee_z])
    
    def _generate_target(self):
        """Generate a random target position within workspace"""
        # Generate target in spherical coordinates around workspace center
        radius = self.np_random.uniform(0.05, self.workspace_radius)
        theta = self.np_random.uniform(0, 2 * np.pi)  # Azimuth
        phi = self.np_random.uniform(np.pi/4, 3*np.pi/4)  # Elevation
        
        # Convert to Cartesian
        target_x = radius * np.sin(phi) * np.cos(theta)
        target_y = radius * np.sin(phi) * np.sin(theta)
        target_z = self.workspace_center[2] + radius * np.cos(phi)
        
        # Ensure target is reachable
        target_z = np.clip(target_z, 0.05, 0.25)
        
        self.target_position = np.array([target_x, target_y, target_z])
    
    def _calculate_reward(self, action):
        """Calculate reward for current state and action"""
        distance_to_target = np.linalg.norm(self.end_effector_pos - self.target_position)
        
        if self.dense_reward:
            # Dense reward based on distance
            distance_reward = self.distance_reward_scale * distance_to_target
            
            # Progress reward
            progress = self.prev_distance - distance_to_target
            progress_reward = 10.0 * progress
            
            # Action penalty
            action_penalty = self.action_penalty_scale * np.sum(np.square(action))
            
            # Success bonus
            success_bonus = self.success_reward if distance_to_target < self.success_distance else 0.0
            
            total_reward = distance_reward + progress_reward + action_penalty + success_bonus
            
            # Update previous distance
            self.prev_distance = distance_to_target
            
            return total_reward
        else:
            # Sparse reward
            return self.success_reward if distance_to_target < self.success_distance else -1.0
    
    def _get_observation(self):
        """Get current observation"""
        obs = np.concatenate([
            self.joint_positions,
            self.joint_velocities,
            self.end_effector_pos,
            self.target_position
        ])
        return obs.astype(np.float32)
    
    def _get_info(self):
        """Get additional information"""
        return {
            'joint_positions': self.joint_positions.copy(),
            'joint_velocities': self.joint_velocities.copy(),
            'end_effector_pos': self.end_effector_pos.copy(),
            'target_position': self.target_position.copy(),
            'step_count': self.step_count,
            'use_ros': self.use_ros
        }
    
    def render(self, mode='human'):
        """Render the environment (Gazebo provides visualization)"""
        if mode == 'human' and self.use_ros:
            # Gazebo already provides visualization
            pass
        return None
    
    def close(self):
        """Clean up ROS resources"""
        if self.use_ros and hasattr(self, 'joint_state_sub'):
            self.joint_state_sub.unregister()
        super().close()

# Backward compatibility alias
Robot4DOFEnv = GazeboRobot4DOFEnv