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
        
        # Robot parameters (from real URDF)
        self.dof = 4
        # Real robot link lengths from URDF joint origins (in meters)
        self.link_lengths = np.array([0.033399, 0.053, 0.063, 0.053])  # Base, shoulder, elbow, wrist
        # Real robot joint limits from URDF (in radians)
        self.joint_limits = np.array([
            [-np.pi, np.pi],        # Joint 1: Base rotation ±180°
            [-np.pi/2, np.pi/2],    # Joint 2: Shoulder ±90°
            [-np.pi/2, np.pi/2],    # Joint 3: Elbow ±90°
            [-np.pi/2, np.pi/2]     # Joint 4: Wrist ±90°
        ])
        
        # Environment parameters (adjusted for real robot size)
        self.max_steps = self.config.get('max_steps', 200)
        self.success_distance = self.config.get('success_distance', 0.02)  # 2cm for realistic real robot precision
        self.workspace_center = np.array([0.0, 0.0, 0.1])  # Lower center for small robot
        self.workspace_radius = self.config.get('workspace_radius', 0.15)  # Much smaller workspace
        
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
        
        # Initialize previous distance for progress tracking
        self.prev_distance = np.linalg.norm(self.end_effector_pos - self.target_position)
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        # Check for NaN in input action first
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            print(f"⚠️ Invalid action in environment step: {action}")
            action = np.zeros(self.dof)  # Use zero action as fallback
            
        # Clip and scale action
        action = np.clip(action, -1.0, 1.0)
        
        # Apply action as joint velocity commands (from URDF limits)
        max_joint_velocity = 2.0  # rad/s (matches URDF velocity limit)
        joint_velocity_commands = action * max_joint_velocity
        
        # Update joint positions (simple integration)
        dt = 0.05  # 20 Hz control frequency
        new_joint_positions = self.joint_positions + joint_velocity_commands * dt
        
        # Check for NaN in new positions
        if np.any(np.isnan(new_joint_positions)) or np.any(np.isinf(new_joint_positions)):
            print(f"⚠️ Invalid joint positions after update: {new_joint_positions}")
            # Keep previous positions if new ones are invalid
            new_joint_positions = self.joint_positions
        
        # Enforce joint limits
        self.joint_positions = np.clip(
            new_joint_positions,
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
        
        # Check for invalid values and terminate episode if found
        nan_detected = (
            np.any(np.isnan(self.joint_positions)) or 
            np.any(np.isnan(self.end_effector_pos)) or 
            np.isnan(reward) or np.isinf(reward)
        )
        
        # Check termination conditions
        self.step_count += 1
        terminated = goal_reached or nan_detected
        truncated = self.step_count >= self.max_steps
        
        # If NaN detected, force reset
        if nan_detected:
            print(f"⚠️ NaN detected, terminating episode at step {self.step_count}")
            reward = -1000.0  # Heavy penalty for invalid state
        
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
        # Check for NaN or infinite joint angles
        if np.any(np.isnan(joint_angles)) or np.any(np.isinf(joint_angles)):
            print(f"⚠️ Invalid joint angles: {joint_angles}")
            return np.array([0.1, 0.0, 0.1])  # Return safe default position
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
        
        return np.array([x, y, z + 0.033])  # +0.033m for real base height
    
    def _generate_target(self):
        """Generate a reachable target position for precision training."""
        # Calculate realistic workspace based on real robot geometry
        # Link lengths: [0.033, 0.053, 0.063, 0.053] = 0.202m total reach
        max_reach = np.sum(self.link_lengths)  # ~0.202m (20.2cm)
        min_reach = abs(self.link_lengths[0] + self.link_lengths[1] - self.link_lengths[2] - self.link_lengths[3])  # ~0.023m
        
        # Generate targets within reachable workspace with safety margin
        max_safe_reach = max_reach * 0.85  # ~0.17m (17cm) with safety margin
        min_safe_reach = max(0.05, min_reach * 1.2)  # 5cm minimum safe distance
        
        # Generate in cylindrical coordinates within safe reachable space
        radius = self.np_random.uniform(min_safe_reach, max_safe_reach)
        angle = self.np_random.uniform(0, 2 * np.pi)
        
        # Height constraint: ensure target is reachable vertically
        # Given radius, calculate feasible height range
        horizontal_reach = radius
        remaining_reach = np.sqrt(max(0, max_safe_reach**2 - horizontal_reach**2))
        
        # Constrain height to achievable range (adjusted for real robot size)
        min_height = 0.05  # Above ground + small base
        max_height = min(0.25, 0.033 + remaining_reach)  # Base height + vertical reach
        
        # Ensure valid range (max_height >= min_height)
        if max_height <= min_height:
            max_height = min_height + 0.05  # Add small buffer
        
        height = self.np_random.uniform(min_height, max_height)
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)  
        z = height
        
        # Verify target is actually reachable by testing inverse kinematics
        test_target = np.array([x, y, z])
        if not self._is_target_reachable(test_target):
            # Fallback to a guaranteed reachable position
            self._generate_safe_target()
        else:
            self.target_position = test_target
    
    def _is_target_reachable(self, target: np.ndarray) -> bool:
        """Check if target position is within robot's reachable workspace."""
        distance_from_origin = np.linalg.norm(target[:2])  # X-Y distance
        total_distance = np.linalg.norm(target - np.array([0, 0, 0.033]))  # From real base
        
        # Check basic geometric constraints
        if total_distance > np.sum(self.link_lengths) * 0.95:  # Too far
            return False
        if total_distance < 0.2:  # Too close (collision with base)
            return False
        if target[2] < 0.033 or target[2] > 0.25:  # Real robot height limits
            return False
            
        return True
    
    def _generate_safe_target(self):
        """Generate a guaranteed reachable target for fallback."""
        # Simple safe target within known reachable space (real robot scale)
        radius = self.np_random.uniform(0.08, 0.15)  # Safe middle range for small robot
        angle = self.np_random.uniform(0, 2 * np.pi)
        height = self.np_random.uniform(0.08, 0.20)  # Safe height range for small robot
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        
        self.target_position = np.array([x, y, z])
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """
        Real robot precision-focused reward function (max reach: 20cm).
        
        Design principles for small robot:
        1. Gentle guidance from far distances (>10cm)
        2. Progressive rewards as we approach target
        3. High precision rewards for <2cm accuracy
        4. No catastrophic penalties - keep robot learning
        """
        # Check for NaN values in positions
        if np.any(np.isnan(self.end_effector_pos)) or np.any(np.isnan(self.target_position)):
            print("⚠️ NaN detected in positions, returning penalty")
            return -1000.0
            
        distance_to_target = np.linalg.norm(self.end_effector_pos - self.target_position)
        
        # Check for NaN or infinite distance
        if np.isnan(distance_to_target) or np.isinf(distance_to_target):
            print(f"⚠️ Invalid distance: {distance_to_target}, returning penalty")
            return -1000.0
        
        # PRECISION ZONES adapted for real robot (20cm max reach)
        EXCELLENT_ZONE = 0.005  # < 5mm: Perfect precision
        GOOD_ZONE = 0.01        # < 1cm: Excellent precision  
        ACCEPTABLE_ZONE = 0.02  # < 2cm: Good precision
        LEARNING_ZONE = 0.05    # < 5cm: Acceptable for training
        REACHABLE_ZONE = 0.15   # < 15cm: Within robot capability
        # > 15cm: Outside normal reach but not catastrophic
        
        reward = 0.0
        
        if self.dense_reward:
            # 1. DISTANCE-BASED REWARD (Scaled for small robot)
            if distance_to_target < EXCELLENT_ZONE:
                # PERFECT: Maximum reward for sub-5mm precision
                precision_reward = 500.0 + (EXCELLENT_ZONE - distance_to_target) * 10000.0
            elif distance_to_target < GOOD_ZONE:
                # EXCELLENT: High reward for sub-1cm precision
                precision_reward = 200.0 + (GOOD_ZONE - distance_to_target) * 5000.0
            elif distance_to_target < ACCEPTABLE_ZONE:
                # GOOD: Strong positive reward for sub-2cm precision
                precision_reward = 100.0 + (ACCEPTABLE_ZONE - distance_to_target) * 2000.0
            elif distance_to_target < LEARNING_ZONE:
                # LEARNING: Positive reward for sub-5cm (training phase)
                precision_reward = 50.0 + (LEARNING_ZONE - distance_to_target) * 500.0
            elif distance_to_target < REACHABLE_ZONE:
                # REACHABLE: Small positive reward - still learning
                precision_reward = 10.0 + (REACHABLE_ZONE - distance_to_target) * 100.0
            else:
                # FAR: Gentle guidance, no harsh penalties
                max_expected_distance = 0.25  # 25cm max expected
                normalized_distance = min(distance_to_target, max_expected_distance)
                precision_reward = -(normalized_distance - REACHABLE_ZONE) * 20.0
            
            reward += precision_reward
            
            # 2. IMPROVEMENT REWARD (Movement toward precision)
            if hasattr(self, 'prev_distance'):
                distance_change = self.prev_distance - distance_to_target
                
                # Scale improvement rewards based on current precision
                if distance_to_target < GOOD_ZONE:
                    improvement_reward = distance_change * 1000.0  # High reward for precision zone
                elif distance_to_target < LEARNING_ZONE:
                    improvement_reward = distance_change * 200.0   # Medium reward getting closer
                else:
                    improvement_reward = distance_change * 50.0    # Basic reward for general improvement
                
                reward += improvement_reward
        
        # Store distance for next step
        self.prev_distance = distance_to_target
        
        # 3. SUCCESS BONUSES (Clear achievement rewards)
        if distance_to_target < EXCELLENT_ZONE:
            reward += 1000.0  # Perfect precision bonus
        elif distance_to_target < GOOD_ZONE:
            reward += 300.0   # Excellent precision bonus
        elif distance_to_target < ACCEPTABLE_ZONE:
            reward += 100.0   # Good precision bonus
        elif distance_to_target < self.success_distance:
            reward += 50.0    # Basic success bonus
        
        # 4. ACTION PENALTY (Encourage smooth movements)
        action_penalty = np.sum(np.square(action)) * self.action_penalty_scale
        reward -= action_penalty
        
        # Final NaN check and clipping
        if np.isnan(reward) or np.isinf(reward):
            print(f"⚠️ Invalid reward: {reward}, returning penalty")
            return -1000.0
            
        # Clip reward to reasonable range to prevent explosion
        reward = np.clip(reward, -10000.0, 10000.0)
        
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
