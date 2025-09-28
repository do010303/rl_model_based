"""
4-DOF Robot Arm Environment for Reinforcement Learning
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional, List
import math

class Robot4DOFEnv(gym.Env):
    """
    4-DOF Robot Arm Environment for goal-reaching tasks.
    
    The robot arm has 4 degrees of freedom and needs to reach target positions
    in 3D space. The environment supports both dense and sparse reward functions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Configuration
        self.config = config or {}
        
        # Robot parameters
        self.dof = 4
        self.link_lengths = np.array([0.3, 0.3, 0.25, 0.2])  # Link lengths in meters
        self.joint_limits = np.array([[-np.pi, np.pi]] * self.dof)
        
        # Environment parameters
        self.max_steps = self.config.get('max_steps', 200)
        self.success_distance = self.config.get('success_distance', 0.05)  # 5cm
        self.workspace_center = np.array([0.0, 0.0, 0.4])
        self.workspace_radius = self.config.get('workspace_radius', 0.8)
        
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
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.dof,), dtype=np.float32
        )
        
        # Observation: [joint_pos(4), joint_vel(4), end_effector_pos(3), target_pos(3)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        
        # Initialize
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset step counter
        self.step_count = 0
        
        # Initialize joint positions randomly
        self.joint_positions = self.np_random.uniform(
            low=-np.pi/2, high=np.pi/2, size=self.dof
        )
        self.joint_velocities = np.zeros(self.dof)
        
        # Calculate initial end-effector position
        self.end_effector_pos = self._forward_kinematics(self.joint_positions)
        
        # Generate random target position within workspace
        self._generate_target()
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        # Clip and scale action
        action = np.clip(action, -1.0, 1.0)
        
        # Apply action as joint velocity commands
        max_joint_velocity = 2.0  # rad/s
        joint_velocity_commands = action * max_joint_velocity
        
        # Update joint positions (simple integration)
        dt = 0.05  # 20 Hz control frequency
        self.joint_positions += joint_velocity_commands * dt
        
        # Enforce joint limits
        self.joint_positions = np.clip(
            self.joint_positions,
            self.joint_limits[:, 0],
            self.joint_limits[:, 1]
        )
        
        # Update joint velocities
        self.joint_velocities = joint_velocity_commands
        
        # Calculate new end-effector position
        self.end_effector_pos = self._forward_kinematics(self.joint_positions)
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if goal is reached
        distance_to_target = np.linalg.norm(self.end_effector_pos - self.target_position)
        goal_reached = distance_to_target < self.success_distance
        
        # Check termination conditions
        self.step_count += 1
        terminated = goal_reached
        truncated = self.step_count >= self.max_steps
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info['goal_reached'] = goal_reached
        info['distance_to_target'] = distance_to_target
        
        return obs, reward, terminated, truncated, info
    
    def _forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Calculate end-effector position using forward kinematics.
        
        Args:
            joint_angles: Joint angles in radians
            
        Returns:
            End-effector position in Cartesian coordinates
        """
        # Simple 4-DOF forward kinematics
        # Joint 0: Base rotation
        # Joint 1: Shoulder pitch
        # Joint 2: Elbow pitch  
        # Joint 3: Wrist pitch
        
        x = 0.0
        y = 0.0
        z = 0.0
        
        # Cumulative transformation
        theta_sum = 0.0
        
        for i, (angle, length) in enumerate(zip(joint_angles, self.link_lengths)):
            if i == 0:  # Base joint - rotation around Z axis
                # This affects the X-Y plane orientation
                base_angle = angle
            else:  # Pitch joints
                theta_sum += angle
                if i == 1:  # Shoulder
                    x += length * np.cos(theta_sum) * np.cos(base_angle)
                    y += length * np.cos(theta_sum) * np.sin(base_angle)
                    z += length * np.sin(theta_sum)
                else:  # Elbow and wrist
                    x += length * np.cos(theta_sum) * np.cos(base_angle)
                    y += length * np.cos(theta_sum) * np.sin(base_angle)
                    z += length * np.sin(theta_sum)
        
        return np.array([x, y, z + 0.1])  # +0.1 for base height
    
    def _generate_target(self):
        """Generate a random target position within the workspace."""
        # Generate target in cylindrical coordinates for better distribution
        radius = self.np_random.uniform(0.2, self.workspace_radius)
        angle = self.np_random.uniform(0, 2 * np.pi)
        height = self.np_random.uniform(0.1, 0.8)
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        
        self.target_position = np.array([x, y, z])
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward based on current state and action."""
        distance_to_target = np.linalg.norm(self.end_effector_pos - self.target_position)
        
        reward = 0.0
        
        if self.dense_reward:
            # Dense reward based on distance to target
            reward += self.distance_reward_scale * distance_to_target
            
            # Bonus for getting close
            if distance_to_target < 0.1:
                reward += 10.0
            if distance_to_target < 0.05:
                reward += 20.0
            
            # Small penalty for large actions (encourage smooth movement)
            action_penalty = self.action_penalty_scale * np.sum(np.square(action))
            reward += action_penalty
        
        # Success bonus
        if distance_to_target < self.success_distance:
            reward += self.success_reward
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.concatenate([
            self.joint_positions,           # 4 elements
            self.joint_velocities,          # 4 elements  
            self.end_effector_pos,          # 3 elements
            self.target_position            # 3 elements
        ])
        return obs.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information."""
        return {
            'joint_positions': self.joint_positions.copy(),
            'joint_velocities': self.joint_velocities.copy(),
            'end_effector_position': self.end_effector_pos.copy(),
            'target_position': self.target_position.copy(),
            'step_count': self.step_count,
            'max_steps': self.max_steps
        }
    
    def render(self, mode: str = 'rgb_array') -> Optional[np.ndarray]:
        """Render the environment."""
        if not hasattr(self, 'fig'):
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.clear()
        
        # Draw robot arm
        positions = self._get_link_positions()
        
        # Plot links
        for i in range(len(positions) - 1):
            self.ax.plot3D(
                [positions[i][0], positions[i+1][0]],
                [positions[i][1], positions[i+1][1]], 
                [positions[i][2], positions[i+1][2]],
                'b-', linewidth=3, marker='o', markersize=6
            )
        
        # Plot target
        self.ax.scatter(
            *self.target_position, 
            color='red', s=100, marker='*', label='Target'
        )
        
        # Plot end-effector
        self.ax.scatter(
            *self.end_effector_pos,
            color='green', s=80, marker='o', label='End Effector'
        )
        
        # Set labels and limits
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.legend()
        
        # Set equal aspect ratio
        max_range = 1.0
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([0, max_range])
        
        # Add title with current distance
        distance = np.linalg.norm(self.end_effector_pos - self.target_position)
        self.ax.set_title(f'4-DOF Robot Arm - Distance to Target: {distance:.3f}m')
        
        if mode == 'human':
            plt.pause(0.01)
            return None
        elif mode == 'rgb_array':
            self.fig.canvas.draw()
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return buf
    
    def _get_link_positions(self) -> List[np.ndarray]:
        """Get positions of all links for visualization."""
        positions = [np.array([0, 0, 0.1])]  # Base position
        
        # Calculate cumulative positions
        current_pos = positions[0].copy()
        theta_sum = 0.0
        base_angle = self.joint_positions[0]
        
        for i, (angle, length) in enumerate(zip(self.joint_positions, self.link_lengths)):
            if i == 0:  # Base joint
                continue
            else:  # Pitch joints
                theta_sum += angle
                dx = length * np.cos(theta_sum) * np.cos(base_angle)
                dy = length * np.cos(theta_sum) * np.sin(base_angle)
                dz = length * np.sin(theta_sum)
                
                current_pos = current_pos + np.array([dx, dy, dz])
                positions.append(current_pos.copy())
        
        return positions
    
    def close(self):
        """Close the environment."""
        if hasattr(self, 'fig'):
            plt.close(self.fig)
