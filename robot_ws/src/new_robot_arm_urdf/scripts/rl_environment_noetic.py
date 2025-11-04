#!/usr/bin/env python3
"""
ROS Noetic RL Environment for 4DoF Robot Arm
=============================================

Adapted from the successful ROS2 robotic_arm_environment for ROS Noetic.
This is the main RL environment that connects the robot control with training algorithms.

Author: Adapted for ROS Noetic RL Training
Date: October 2025
"""

import os
import sys
import time
import rospy
import random
import numpy as np
import threading
from typing import Tuple, List, Dict, Any

# ROS Noetic imports
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# TF for end-effector position
import tf2_ros
from geometry_msgs.msg import TransformStamped

class RosNoeticRLEnvironment:
    """
    Main RL Environment for 4DoF Robot Arm in ROS Noetic
    Adapted from successful ROS2 robotic_arm_environment structure
    """
    
    def __init__(self, node_name='rl_environment_node'):
        """Initialize the RL environment"""
        
        # Initialize ROS node
        rospy.init_node(node_name, anonymous=True)
        rospy.loginfo("🚀 Initializing ROS Noetic RL Environment...")
        
        # Environment parameters
        self.max_episode_steps = 50
        self.current_step = 0
        self.episode_count = 0
        
        # State variables
        self.joint_positions = [0.0, 0.0, 0.0, 0.0]  # 4DoF
        self.joint_velocities = [0.0, 0.0, 0.0, 0.0]
        self.end_effector_pos = [0.0, 0.0, 0.0]
        self.target_pos = [0.0, 0.0, 0.0]
        
        # State lock for thread safety
        self.state_lock = threading.Lock()
        
        # Setup ROS interfaces
        self._setup_tf_listener()
        self._setup_action_client()
        self._setup_service_clients()
        self._setup_subscribers()
        
        # Wait for connections
        rospy.sleep(2.0)
        rospy.loginfo("✅ ROS Noetic RL Environment initialized!")
    
    def _setup_tf_listener(self):
        """Setup TF listener for end-effector position"""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
    
    def _setup_action_client(self):
        """Setup action client for robot control"""
        rospy.loginfo("Setting up trajectory action client...")
        self.trajectory_client = actionlib.SimpleActionClient(
            '/arm_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction
        )
        
        # Wait for action server
        rospy.loginfo("Waiting for arm controller action server...")
        if not self.trajectory_client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.logwarn("Arm controller action server not available!")
        else:
            rospy.loginfo("✅ Trajectory action client connected!")
    
    def _setup_service_clients(self):
        """Setup service clients for environment control"""
        rospy.loginfo("Setting up Gazebo service clients...")
        
        # Service for moving target sphere
        rospy.wait_for_service('/gazebo/set_model_state', timeout=10.0)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        rospy.loginfo("✅ Gazebo service clients connected!")
    
    def _setup_subscribers(self):
        """Setup ROS subscribers"""
        rospy.loginfo("Setting up ROS subscribers...")
        
        # Subscribe to joint states
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self._joint_callback)
        
        # Subscribe to model states (for target position)
        self.model_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self._model_callback)
        
        rospy.loginfo("✅ ROS subscribers initialized!")
    
    def _joint_callback(self, msg: JointState):
        """Callback to update joint positions and velocities"""
        with self.state_lock:
            try:
                # Map joint names to positions (handle reordering)
                joint_mapping = {}
                for i, name in enumerate(msg.name):
                    joint_mapping[name] = {
                        'position': msg.position[i],
                        'velocity': msg.velocity[i] if msg.velocity else 0.0
                    }
                
                # Update joint states for 4DoF robot
                joint_names = ['Joint_1', 'Joint_2', 'Joint_3', 'Joint_4']
                for i, joint_name in enumerate(joint_names):
                    if joint_name in joint_mapping:
                        self.joint_positions[i] = joint_mapping[joint_name]['position']
                        self.joint_velocities[i] = joint_mapping[joint_name]['velocity']
                
            except Exception as e:
                rospy.logwarn(f"Joint callback error: {e}")
    
    def _model_callback(self, msg: ModelStates):
        """Callback to update target sphere position"""
        with self.state_lock:
            try:
                # Find target sphere in model states
                if 'target_sphere' in msg.name:
                    idx = msg.name.index('target_sphere')
                    pose = msg.pose[idx]
                    self.target_pos = [
                        pose.position.x,
                        pose.position.y,
                        pose.position.z
                    ]
                elif 'my_sphere' in msg.name:  # Alternative sphere name
                    idx = msg.name.index('my_sphere')
                    pose = msg.pose[idx]
                    self.target_pos = [
                        pose.position.x,
                        pose.position.y,
                        pose.position.z
                    ]
            except Exception as e:
                rospy.logwarn(f"Model callback error: {e}")
    
    def _update_end_effector_position(self):
        """Update end-effector position from TF"""
        try:
            # Get transform from base_link to end_effector
            transform = self.tf_buffer.lookup_transform(
                'base_link', 'link_4_1', rospy.Time()
            )
            
            self.end_effector_pos = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ]
            
        except Exception as e:
            # rospy.logwarn_throttle(5.0, f"TF lookup failed: {e}")
            pass
    
    def get_state(self) -> np.ndarray:
        """
        Get current environment state
        
        Returns:
            State vector: [joint_pos(4), joint_vel(4), end_effector_pos(3), target_pos(3), success_flag(1)]
                         Total: 15 dimensions
        """
        with self.state_lock:
            # Update end-effector position
            self._update_end_effector_position()
            
            # Calculate success flag (within 5cm of target)
            if self.end_effector_pos and self.target_pos:
                distance = np.linalg.norm(
                    np.array(self.end_effector_pos) - np.array(self.target_pos)
                )
                success_flag = 1.0 if distance < 0.05 else 0.0
            else:
                success_flag = 0.0
            
            # Construct state vector
            state = (
                list(self.joint_positions) +     # 4 elements
                list(self.joint_velocities) +    # 4 elements  
                list(self.end_effector_pos) +    # 3 elements
                list(self.target_pos) +          # 3 elements
                [success_flag]                   # 1 element
            )
            
            return np.array(state, dtype=np.float32)
    
    def execute_action(self, action: np.ndarray) -> bool:
        """
        Execute action on robot
        
        Args:
            action: Joint position deltas [-1, 1] for each joint
            
        Returns:
            True if action executed successfully
        """
        try:
            # Clip actions to valid range
            action = np.clip(action, -1.0, 1.0)
            
            # Convert action deltas to target positions
            current_positions = self.joint_positions.copy()
            
            # Scale actions to reasonable joint movements (0.1 rad max per step)
            action_scale = 0.1
            target_positions = []
            
            # Joint limits (approximate for safety)
            joint_limits = [
                (-3.14, 3.14),   # Joint_1
                (-1.57, 1.57),   # Joint_2  
                (-1.57, 1.57),   # Joint_3
                (-1.57, 1.57)    # Joint_4
            ]
            
            for i, (current, delta) in enumerate(zip(current_positions, action)):
                new_pos = current + (delta * action_scale)
                # Apply joint limits
                min_limit, max_limit = joint_limits[i]
                new_pos = np.clip(new_pos, min_limit, max_limit)
                target_positions.append(new_pos)
            
            # Send trajectory goal
            goal = FollowJointTrajectoryGoal()
            goal.trajectory.header.stamp = rospy.Time.now()
            goal.trajectory.joint_names = ['Joint_1', 'Joint_2', 'Joint_3', 'Joint_4']
            
            # Create trajectory point
            point = JointTrajectoryPoint()
            point.positions = target_positions
            point.velocities = [0.0] * 4
            point.time_from_start = rospy.Duration(0.5)  # 0.5 second movement
            
            goal.trajectory.points = [point]
            
            # Send goal (non-blocking)
            self.trajectory_client.send_goal(goal)
            
            # Wait briefly for execution to start
            rospy.sleep(0.1)
            
            return True
            
        except Exception as e:
            rospy.logerr(f"Action execution failed: {e}")
            return False
    
    def get_reward(self) -> Tuple[float, bool]:
        """
        Calculate reward based on distance to target
        
        Returns:
            (reward, done): reward value and episode termination flag
        """
        if not self.end_effector_pos or not self.target_pos:
            return -1.0, False
        
        # Calculate distance to target
        distance = np.linalg.norm(
            np.array(self.end_effector_pos) - np.array(self.target_pos)
        )
        
        # Reward function: negative distance + success bonus
        reward = -distance
        
        # Success bonus (reached within 5cm)
        done = False
        if distance < 0.05:
            reward += 10.0
            done = True
            rospy.loginfo(f"🎯 Target reached! Distance: {distance:.4f}m")
        
        # Penalty for being very far (> 50cm)
        elif distance > 0.5:
            reward -= 5.0
        
        # Episode timeout
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            done = True
            rospy.loginfo(f"⏰ Episode timeout after {self.current_step} steps")
        
        return reward, done
    
    def reset_environment(self) -> np.ndarray:
        """
        Reset environment: move robot to home position and randomize target
        
        Returns:
            Initial state after reset
        """
        rospy.loginfo(f"🔄 Resetting environment (Episode {self.episode_count + 1})")
        
        # Reset step counter
        self.current_step = 0
        self.episode_count += 1
        
        # Move robot to home position
        self._move_robot_home()
        
        # Randomize target position
        self._reset_target_position()
        
        # Wait for state to stabilize
        rospy.sleep(1.0)
        
        rospy.loginfo("✅ Environment reset complete")
        return self.get_state()
    
    def _move_robot_home(self) -> bool:
        """Move robot to home position"""
        try:
            home_positions = [0.0, -0.5, 0.8, -0.3]  # Stable forward-leaning pose
            
            goal = FollowJointTrajectoryGoal()
            goal.trajectory.header.stamp = rospy.Time.now()
            goal.trajectory.joint_names = ['Joint_1', 'Joint_2', 'Joint_3', 'Joint_4']
            
            point = JointTrajectoryPoint()
            point.positions = home_positions
            point.velocities = [0.0] * 4
            point.time_from_start = rospy.Duration(2.0)  # Slow movement to home
            
            goal.trajectory.points = [point]
            
            # Send goal and wait for completion
            self.trajectory_client.send_goal_and_wait(goal, execute_timeout=rospy.Duration(5.0))
            
            rospy.loginfo("✅ Robot moved to home position")
            return True
            
        except Exception as e:
            rospy.logerr(f"Failed to move robot home: {e}")
            return False
    
    def _reset_target_position(self) -> bool:
        """Reset target sphere to random position in workspace"""
        try:
            # Define cylindrical workspace (safe reachable area for 4DoF arm)
            max_radius = 0.15  # 15cm radius
            min_radius = 0.05  # 5cm minimum distance
            min_height = 0.05  # 5cm above table
            max_height = 0.20  # 20cm max height
            
            # Generate random cylindrical coordinates
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(min_radius, max_radius)
            height = random.uniform(min_height, max_height)
            
            # Convert to Cartesian coordinates
            target_x = radius * np.cos(angle)
            target_y = radius * np.sin(angle)  
            target_z = height
            
            # Create service request
            req = SetModelStateRequest()
            req.model_state.model_name = 'target_sphere'  # or 'my_sphere'
            req.model_state.pose.position.x = target_x
            req.model_state.pose.position.y = target_y
            req.model_state.pose.position.z = target_z
            req.model_state.pose.orientation.w = 1.0
            
            # Send service request
            response = self.set_model_state(req)
            
            if response.success:
                rospy.loginfo(f"🎯 Target reset to: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})")
                return True
            else:
                rospy.logwarn("Failed to reset target position")
                return False
                
        except Exception as e:
            rospy.logerr(f"Target reset error: {e}")
            return False
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """Get action space information"""
        return {
            'shape': (4,),  # 4DoF actions
            'low': np.array([-1.0, -1.0, -1.0, -1.0]),
            'high': np.array([1.0, 1.0, 1.0, 1.0]),
            'dtype': np.float32
        }
    
    def get_observation_space_info(self) -> Dict[str, Any]:
        """Get observation space information"""
        return {
            'shape': (15,),  # 4+4+3+3+1 = 15 dimensional state
            'low': np.array([-np.inf] * 15),
            'high': np.array([np.inf] * 15),
            'dtype': np.float32
        }
    
    def shutdown(self):
        """Shutdown the environment"""
        rospy.loginfo("🛑 Shutting down RL environment")
        rospy.signal_shutdown("Environment shutdown requested")

# Factory function for easy instantiation
def create_rl_environment():
    """Create and return RL environment instance"""
    return RosNoeticRLEnvironment()

if __name__ == "__main__":
    """Test the RL environment"""
    try:
        env = create_rl_environment()
        
        rospy.loginfo("🧪 Testing RL environment...")
        
        # Test reset
        initial_state = env.reset_environment()
        rospy.loginfo(f"Initial state shape: {initial_state.shape}")
        
        # Test a few action steps
        for step in range(5):
            action = np.random.uniform(-1, 1, 4)  # Random 4DoF action
            rospy.loginfo(f"Step {step + 1}: Action = {action}")
            
            success = env.execute_action(action)
            rospy.sleep(0.5)  # Wait for execution
            
            state = env.get_state()
            reward, done = env.get_reward()
            
            rospy.loginfo(f"   State shape: {state.shape}")
            rospy.loginfo(f"   Reward: {reward:.3f}")
            rospy.loginfo(f"   Done: {done}")
            
            if done:
                break
        
        rospy.loginfo("✅ RL environment test completed!")
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted by user")
    except Exception as e:
        rospy.logerr(f"Test failed: {str(e)}")
    finally:
        if 'env' in locals():
            env.shutdown()