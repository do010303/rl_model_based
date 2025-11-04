#!/usr/bin/env python3
"""
Visual Target Reaching Environment for 4-DOF Robot Arm
=====================================================

This environment provides visual feedback where you can see:
- Red target sphere that moves to random positions
- Blue end effector sphere tracking robot tip
- Robot arm learning to reach targets in real-time
- Success feedback when target is reached
"""

import os
import sys
import gym
import rospy
import numpy as np
import subprocess
import time
import random
from gym import spaces
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Bool, String
from geometry_msgs.msg import Point
from gazebo_msgs.srv import ResetSimulation

class VisualTargetReachingEnv(gym.Env):
    def __init__(self, 
                 workspace_path="/home/ducanh/rl_model_based/robot_ws",
                 use_gui=True,
                 real_time=True,
                 auto_reset_target=True):
        """
        Visual Target Reaching Environment
        
        Args:
            workspace_path: Path to ROS workspace
            use_gui: Whether to show Gazebo GUI for visual training
            real_time: Whether to run in real-time (slower but visual)
            auto_reset_target: Whether to automatically reset target on success
        """
        super().__init__()
        
        self.workspace_path = workspace_path
        self.use_gui = use_gui
        self.real_time = real_time
        self.auto_reset_target = auto_reset_target
        
        # Initialize ROS if not already done
        if not rospy.core.is_initialized():
            rospy.init_node('visual_training_env', anonymous=True)
        
        # Action and observation spaces
        # Actions: joint velocities for 4 joints
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(4,),
            dtype=np.float32
        )
        
        # Observations: joint positions (4) + joint velocities (4) + target pos (3) + ee pos (3) + distance (1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),
            dtype=np.float32
        )
        
        # Joint configuration
    self.joint_names = ['Joint1', 'Joint2', 'Joint3', 'Joint4']
        self.joint_limits = [
            [-np.pi, np.pi],      # Joint 1: full rotation
            [-np.pi/2, np.pi/2],  # Joint 2: shoulder 
            [-np.pi/2, np.pi/2],  # Joint 3: elbow
            [-np.pi/2, np.pi/2]   # Joint 4: wrist
        ]
        
        # State variables
        self.joint_positions = np.zeros(4)
        self.joint_velocities = np.zeros(4) 
        self.target_position = np.array([0.3, 0.2, 0.4])
        self.ee_position = np.array([0.0, 0.0, 0.2])
        self.distance_to_target = 1.0
        
        # Training parameters
        self.max_episode_steps = 500
        self.current_step = 0
        self.success_threshold = 0.05  # 5cm
        self.success_reward = 100.0
        self.step_penalty = -0.1
        self.distance_reward_scale = -10.0
        
        # Episode tracking
        self.episode_count = 0
        self.total_successes = 0
        self.episode_reward = 0.0
        
        # ROS Publishers
        self.joint_pubs = {}
        for i, joint_name in enumerate(self.joint_names):
            pub_name = f'/joint{i+1}_position_controller/command'
            self.joint_pubs[joint_name] = rospy.Publisher(pub_name, Float64, queue_size=1)
        
        self.reset_target_pub = rospy.Publisher('/training/reset_target', Bool, queue_size=1)
        
        # ROS Subscribers
        self.joint_states_sub = rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        self.target_pos_sub = rospy.Subscriber('/training/target_position', Point, self.target_pos_callback)
        self.ee_pos_sub = rospy.Subscriber('/training/end_effector_position', Point, self.ee_pos_callback)
        self.distance_sub = rospy.Subscriber('/training/distance_to_target', Float64, self.distance_callback)
        self.success_sub = rospy.Subscriber('/training/success', Bool, self.success_callback)
        
        # State flags
        self.success_achieved = False
        self.last_update_time = time.time()
        
        # Wait for connections
        rospy.sleep(1.0)
        rospy.loginfo("Visual Target Reaching Environment initialized!")
        
    def joint_states_callback(self, msg):
        """Update joint state information"""
        if len(msg.name) >= 4 and len(msg.position) >= 4 and len(msg.velocity) >= 4:
            for i, joint_name in enumerate(self.joint_names):
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    self.joint_positions[i] = msg.position[idx]
                    self.joint_velocities[i] = msg.velocity[idx] if idx < len(msg.velocity) else 0.0
    
    def target_pos_callback(self, msg):
        """Update target position"""
        self.target_position = np.array([msg.x, msg.y, msg.z])
    
    def ee_pos_callback(self, msg):
        """Update end effector position"""
        self.ee_position = np.array([msg.x, msg.y, msg.z])
    
    def distance_callback(self, msg):
        """Update distance to target"""
        self.distance_to_target = msg.data
    
    def success_callback(self, msg):
        """Handle success signal"""
        if msg.data:
            self.success_achieved = True
            self.total_successes += 1
            rospy.loginfo(f"🎯 Success! Total: {self.total_successes}")
    
    def reset(self):
        """Reset the environment for a new episode"""
        self.episode_count += 1
        self.current_step = 0
        self.success_achieved = False
        self.episode_reward = 0.0
        
        # Reset joint positions to safe initial configuration
        initial_positions = [0.0, 0.2, -0.2, 0.1]  # Safe starting pose
        
        # Gradually move to initial position to prevent jumping
        current_pos = self.joint_positions.copy()
        steps = 20  # Number of interpolation steps
        
        for step in range(steps):
            alpha = (step + 1) / steps
            interpolated_pos = current_pos + alpha * (np.array(initial_positions) - current_pos)
            
            for i, joint_name in enumerate(self.joint_names):
                self.joint_pubs[joint_name].publish(Float64(interpolated_pos[i]))
            
            rospy.sleep(0.05)  # Small delay between steps
        
        # Final position set
        for i, joint_name in enumerate(self.joint_names):
            self.joint_pubs[joint_name].publish(Float64(initial_positions[i]))
        
        # Reset target position
        reset_msg = Bool()
        reset_msg.data = True
        self.reset_target_pub.publish(reset_msg)
        
        # Wait for reset to complete
        rospy.sleep(1.0)  # Longer wait for stability
        
        # Calculate success rate
        success_rate = (self.total_successes / max(1, self.episode_count)) * 100
        rospy.loginfo(f"Episode {self.episode_count} | Success Rate: {success_rate:.1f}%")
        
        return self.get_observation()
    
    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Apply action (smooth incremental control)
        action = np.clip(action, -1.0, 1.0)
        
        # Much smaller incremental movements for stability
        max_step = 0.02  # Reduced from 0.05 to 0.02 for smoother movement
        target_positions = self.joint_positions + action * max_step
        
        # Apply joint limits
        for i in range(4):
            target_positions[i] = np.clip(
                target_positions[i], 
                self.joint_limits[i][0], 
                self.joint_limits[i][1]
            )
        
        # Smooth interpolation to target positions (prevents jumping)
        smoothing_factor = 0.3  # How much to move toward target each step
        smooth_positions = self.joint_positions + (target_positions - self.joint_positions) * smoothing_factor
        
        # Send smooth commands to robot
        for i, joint_name in enumerate(self.joint_names):
            self.joint_pubs[joint_name].publish(Float64(smooth_positions[i]))
        
        # Longer wait for stable physics update
        if self.real_time:
            rospy.sleep(0.1)  # 10Hz for stable visual training
        else:
            rospy.sleep(0.05)  # 20Hz for fast but stable training
        
        # Calculate reward
        reward = self.calculate_reward()
        self.episode_reward += reward
        
        # Check termination conditions
        done = self.is_episode_done()
        
        # Create info dict
        info = {
            'episode': self.episode_count,
            'step': self.current_step,
            'distance_to_target': self.distance_to_target,
            'success': self.success_achieved,
            'episode_reward': self.episode_reward,
            'success_rate': (self.total_successes / max(1, self.episode_count)) * 100
        }
        
        return self.get_observation(), reward, done, info
    
    def calculate_reward(self):
        """Calculate reward based on distance to target and success"""
        reward = 0.0
        
        # Success reward
        if self.success_achieved:
            reward += self.success_reward
            rospy.loginfo(f"🎉 Success reward: {self.success_reward}")
        
        # Distance-based reward (negative distance encourages getting closer)
        distance_reward = self.distance_reward_scale * self.distance_to_target
        reward += distance_reward
        
        # Step penalty (encourages faster completion)
        reward += self.step_penalty
        
        # Bonus for being very close (within 10cm)
        if self.distance_to_target < 0.1:
            reward += 5.0
        
        # Penalty for being very far (beyond 50cm)
        if self.distance_to_target > 0.5:
            reward -= 5.0
        
        return reward
    
    def is_episode_done(self):
        """Check if episode should terminate"""
        # Success condition
        if self.success_achieved:
            return True
        
        # Maximum steps reached
        if self.current_step >= self.max_episode_steps:
            return True
        
        # Target too far away (robot got lost)
        if self.distance_to_target > 1.0:
            return True
        
        return False
    
    def get_observation(self):
        """Get current observation state"""
        obs = np.concatenate([
            self.joint_positions,           # 4 values: current joint angles
            self.joint_velocities,          # 4 values: current joint velocities  
            self.target_position,           # 3 values: target xyz position
            self.ee_position,               # 3 values: end effector xyz position
            [self.distance_to_target]       # 1 value: distance to target
        ])
        return obs.astype(np.float32)
    
    def render(self, mode='human'):
        """Render environment (Gazebo GUI handles this)"""
        if mode == 'human':
            # Visual rendering is handled by Gazebo GUI
            pass
        return None
    
    def close(self):
        """Clean up environment"""
        rospy.loginfo("Closing Visual Target Reaching Environment")
        
    def get_training_stats(self):
        """Get current training statistics"""
        success_rate = (self.total_successes / max(1, self.episode_count)) * 100
        return {
            'episodes': self.episode_count,
            'total_successes': self.total_successes, 
            'success_rate': success_rate,
            'current_distance': self.distance_to_target,
            'current_reward': self.episode_reward
        }

# Factory function for easy environment creation
def make_visual_env(use_gui=True, real_time=True):
    """Create a visual target reaching environment"""
    return VisualTargetReachingEnv(
        workspace_path="/home/ducanh/rl_model_based/robot_ws",
        use_gui=use_gui,
        real_time=real_time,
        auto_reset_target=True
    )