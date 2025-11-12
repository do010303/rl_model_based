#!/usr/bin/env python3
"""
DDPG Training for 4DOF Gazebo Robot - Short Episodes
Integrates the working test_simple_movement.py timing with DDPG agent

Key Features:
- Episodes: Only 5 actions per episode (~5.75s total)
- Action timing: 1.15s per action (1s trajectory + 0.15s buffer)
- State space: 14D [4 joint angles, 4 velocities, 3 ee_pos, 3 target_pos]
-        # Info for debugging
        info = {
            'distance': distance,
            'min_distance': self.min_distance,
            'success': done
        }
        
        return reward, done, infopersistent /joint_states subscriber (instant reads)
- Optimized for ultra-rapid RL training in Gazebo
"""

import rospy
import numpy as np
import sys
import os
import time
import signal
from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/ducanh/rl_model_based')  # For agents module

from agents.ddpg_gazebo import DDPGAgentGazebo
from main_rl_environment_noetic import RLEnvironmentNoetic
from constrained_ik import constrained_ik, SURFACE_X
from fk_ik_utils import fk


# ============================================================================
# SIGNAL HANDLER FOR CLEAN EXIT
# ============================================================================

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n👋 Ctrl+C detected! Exiting training script. Goodbye!")
    rospy.signal_shutdown("User requested exit")
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)


# ============================================================================
# TRAINING HYPERPARAMETERS (optimized for fast Gazebo training)
# ============================================================================

# Episode settings
NUM_EPISODES = 500          # Total training episodes
MAX_STEPS_PER_EPISODE = 5   # Only 5 actions per episode (17.5s total)
LEARNING_STARTS = 50        # Start learning after this many episodes

# Training settings
OPT_STEPS_PER_EPISODE = 40  # Gradient updates per episode
SAVE_INTERVAL = 25          # Save models every N episodes
EVAL_INTERVAL = 10          # Evaluate (without noise) every N episodes

# Reward settings (SIMPLIFIED - matching reference robotic_arm_environment)
GOAL_THRESHOLD = 0.01       # 1cm threshold for success (increased precision from 5cm)
SUCCESS_REWARD = 10.0       # +10 when goal reached (same as reference)
STEP_REWARD = -1.0          # -1 for each step not at goal (same as reference)

# Action timing (ULTRA-FAST: 1s trajectory + 0.15s buffer = 1.15s total)
TRAJECTORY_TIME = 1.0       # Trajectory execution time (very fast!)
BUFFER_TIME = 0.15          # Buffer after trajectory
ACTION_WAIT_TIME = TRAJECTORY_TIME + BUFFER_TIME  # 1.15s total


# ============================================================================
# PERSISTENT /joint_states SUBSCRIBER (from test_simple_movement.py)
# ============================================================================

_joint_state_data = {'positions': None, 'velocities': None, 'timestamp': 0}
_joint_state_sub = None

def _joint_state_callback(msg):
    """Persistent callback for /joint_states - updates global data"""
    global _joint_state_data
    try:
        positions = []
        velocities = []
        for i in range(1, 5):
            joint_name = f'Joint{i}'
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                positions.append(msg.position[idx])
                velocities.append(msg.velocity[idx] if len(msg.velocity) > idx else 0.0)
        
        if len(positions) == 4:
            _joint_state_data['positions'] = np.array(positions)
            _joint_state_data['velocities'] = np.array(velocities)
            _joint_state_data['timestamp'] = time.time()
    except Exception as e:
        rospy.logerr_throttle(1.0, f"Error parsing /joint_states: {e}")

def init_joint_state_subscriber():
    """Initialize the persistent /joint_states subscriber"""
    global _joint_state_sub
    if _joint_state_sub is None:
        _joint_state_sub = rospy.Subscriber('/joint_states', JointState, _joint_state_callback, queue_size=1)
        rospy.sleep(0.2)  # Give subscriber time to connect
        rospy.loginfo("✅ Persistent /joint_states subscriber initialized")

def get_joint_positions_direct(timeout=0.5):
    """
    Get joint positions DIRECTLY from /joint_states using persistent subscriber.
    This is FAST because it just reads cached data from the callback.
    """
    global _joint_state_data
    
    start = time.time()
    while time.time() - start < timeout:
        if _joint_state_data['positions'] is not None:
            age = time.time() - _joint_state_data['timestamp']
            if age < 0.5:  # Data is fresh (less than 0.5s old)
                return _joint_state_data['positions'].copy(), _joint_state_data['velocities'].copy()
        rospy.sleep(0.01)
    
    # Timeout - use stale data if available
    if _joint_state_data['positions'] is not None:
        age = time.time() - _joint_state_data['timestamp']
        rospy.logwarn(f"Using stale joint_states data (age: {age:.2f}s)")
        return _joint_state_data['positions'].copy(), _joint_state_data['velocities'].copy()
    
    return None, None


# ============================================================================
# GAZEBO RL WRAPPER
# ============================================================================

class GazeboRLWrapper:
    """
    Wrapper for RLEnvironmentNoetic that provides Gym-like interface
    
    CONSTRAINED APPROACH:
    - Agent outputs (Y, Z) positions for end-effector (2D action space)
    - X is FIXED at 0.15m (drawing surface)
    - IK solver converts (X=0.15, Y, Z) → joint angles
    - Robot reaches targets on vertical plane
    """
    
    def __init__(self, env):
        self.env = env
        self.state_dim = 11  # [4 joints, 4 vels, 3 target_yz_relative]
        self.action_dim = 2  # Only Y and Z control!
        
        # Action limits (Y and Z positions on drawing surface)
        # Conservative workspace for 94% IK success rate
        self.y_limits = np.array([-0.10, 0.10])  # ±10cm (was ±14cm)
        self.z_limits = np.array([0.12, 0.18])   # 12-18cm (was 8-18cm, increased to prevent robot breaking at low Z)
        
        # Track episode stats
        self.episode_reward = 0
        self.episode_steps = 0
        self.min_distance = float('inf')
        
        rospy.loginfo("✅ GazeboRLWrapper initialized (CONSTRAINED IK MODE)")
        rospy.loginfo(f"   Surface X: {SURFACE_X}m (FIXED)")
        rospy.loginfo(f"   Action space: 2D (Y, Z only)")
        rospy.loginfo(f"   Y range: [{self.y_limits[0]}, {self.y_limits[1]}]")
        rospy.loginfo(f"   Z range: [{self.z_limits[0]}, {self.z_limits[1]}]")
    
    def get_state(self):
        """
        Get current state for CONSTRAINED task
        
        State vector (11D):
        - 4 joint angles
        - 4 joint velocities  
        - 3 relative target position (target_Y - ee_Y, target_Z - ee_Z, distance_YZ)
        
        Note: X is not in state because it's FIXED at surface position!
        
        Returns:
            11D numpy array
        """
        # Get joint positions and velocities (FAST with persistent subscriber)
        joint_positions, joint_velocities = get_joint_positions_direct(timeout=0.5)
        
        if joint_positions is None or joint_velocities is None:
            rospy.logwarn("⚠️ Could not get joint states, using zeros")
            joint_positions = np.zeros(4)
            joint_velocities = np.zeros(4)
        
        # Get end-effector and target positions from environment
        ee_pos = np.array(self.env.ee_position) if hasattr(self.env, 'ee_position') else np.zeros(3)
        target_pos = np.array(self.env.target_position) if hasattr(self.env, 'target_position') else np.zeros(3)
        
        # Extract ONLY Y and Z (X is fixed at surface!)
        ee_y, ee_z = ee_pos[1], ee_pos[2]
        target_y, target_z = target_pos[1], target_pos[2]
        
        # Relative target position in Y-Z plane
        delta_y = target_y - ee_y
        delta_z = target_z - ee_z
        distance_yz = np.sqrt(delta_y**2 + delta_z**2)
        
        # Combine into state vector
        state = np.concatenate([
            joint_positions,           # 4D
            joint_velocities,          # 4D
            [delta_y, delta_z, distance_yz]  # 3D (relative target in Y-Z)
        ])
        
        return state.astype(np.float32)
    
    def reset(self):
        """
        Reset environment and return initial state
        
        Returns:
            11D state vector
        """
        rospy.loginfo("🔄 Resetting environment...")
        
        # Try normal reset first
        success = self.env.reset_environment()
        
        if not success:
            rospy.logwarn("⚠️ Environment reset failed, retrying with longer wait...")
            rospy.sleep(2.0)
            success = self.env.reset_environment()
        
        # If still failing, robot might be broken - try resetting Gazebo physics
        if not success:
            rospy.logerr("❌ Environment reset failed twice!")
            rospy.logerr("🔧 Attempting to recover by resetting Gazebo physics...")
            try:
                from std_srvs.srv import Empty
                rospy.wait_for_service('/gazebo/reset_simulation', timeout=2.0)
                reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
                reset_simulation()
                rospy.loginfo("✅ Gazebo simulation reset")
                rospy.sleep(3.0)  # Wait for simulation to stabilize
                # Try environment reset again after Gazebo reset
                success = self.env.reset_environment()
            except Exception as e:
                rospy.logerr(f"❌ Failed to reset Gazebo: {e}")
                rospy.logerr("⚠️ Continuing with potentially broken state...")
        
        # Reset episode tracking
        self.episode_reward = 0
        self.episode_steps = 0
        self.min_distance = float('inf')
        
        # Wait for robot to settle
        rospy.sleep(1.0)
        
        # Get initial state
        state = self.get_state()
        
        # SAFETY: Check if state is valid
        if not np.isfinite(state).all():
            rospy.logerr(f"🛑 INVALID STATE after reset! Contains NaN/Inf: {state}")
            rospy.logerr("   Replacing with safe default state...")
            # Return safe default state
            state = np.zeros(11, dtype=np.float32)
        
        return state
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done, info)
        
        CONSTRAINED APPROACH:
        1. Agent outputs normalized (Y, Z) in range [-1, 1]
        2. Denormalize to actual (Y, Z) positions on surface
        3. Use IK to calculate joint angles for (X=0.15, Y, Z)
        4. Send joint angles to robot
        
        Args:
            action: 2D normalized action vector [-1, 1] for (Y, Z)
            
        Returns:
            next_state: 11D state vector
            reward: float
            done: bool
            info: dict with extra info
        """
        # SAFETY CHECK 1: Validate action for NaN/Inf
        if not np.isfinite(action).all():
            rospy.logerr(f"      🛑 INVALID ACTION! Contains NaN/Inf: {action}")
            rospy.logerr("      Agent output corrupted, likely due to previous robot break")
            rospy.logerr("      Resetting episode with large penalty...")
            next_state = self.get_state()
            reward = -100.0
            done = True
            # Get current distance for info
            ee_pos = np.array(self.env.ee_position)
            target_pos = np.array(self.env.target_position)
            distance = np.linalg.norm(ee_pos - target_pos)
            self.min_distance = min(self.min_distance, distance)
            info = {
                'error': 'invalid_action',
                'error_code': -1000,
                'distance': distance,
                'min_distance': self.min_distance,
                'success': False,
                'steps': self.episode_steps,
                'ee_position': ee_pos,
                'target_position': target_pos
            }
            self.episode_reward += reward
            return next_state, reward, done, info
        
        # Get state BEFORE action
        state_before = self.get_state()
        joints_before, vels_before = get_joint_positions_direct()
        
        # SAFETY CHECK 2: Validate joint state
        if joints_before is None or not np.isfinite(joints_before).all():
            rospy.logerr(f"      🛑 INVALID JOINT STATE! joints={joints_before}")
            rospy.logerr("      Robot state corrupted, ending episode...")
            next_state = self.get_state()
            reward = -100.0
            done = True
            ee_pos = np.array(self.env.ee_position)
            target_pos = np.array(self.env.target_position)
            distance = np.linalg.norm(ee_pos - target_pos)
            self.min_distance = min(self.min_distance, distance)
            info = {
                'error': 'invalid_joint_state',
                'error_code': -1001,
                'distance': distance,
                'min_distance': self.min_distance,
                'success': False,
                'steps': self.episode_steps,
                'ee_position': ee_pos,
                'target_position': target_pos
            }
            self.episode_reward += reward
            return next_state, reward, done, info
        
        # Get current EE and target positions
        ee_pos = np.array(self.env.ee_position)
        target_pos = np.array(self.env.target_position)
        
        # Denormalize action to actual (Y, Z) positions
        target_y = self.y_limits[0] + (action[0] + 1) * 0.5 * (self.y_limits[1] - self.y_limits[0])
        target_z = self.z_limits[0] + (action[1] + 1) * 0.5 * (self.z_limits[1] - self.z_limits[0])
        
        # Clip to ensure within limits
        target_y = np.clip(target_y, self.y_limits[0], self.y_limits[1])
        target_z = np.clip(target_z, self.z_limits[0], self.z_limits[1])
        
        # Log action details
        rospy.loginfo(f"      📝 Normalized action (Y,Z): {np.round(action, 3)}")
        rospy.loginfo(f"      📝 Target EE position: X={SURFACE_X:.3f}, Y={target_y:+.3f}, Z={target_z:.3f}")
        rospy.loginfo(f"      📍 BEFORE: ee={np.round(ee_pos, 4)}, joints={np.round(joints_before, 3)}")
        rospy.loginfo(f"      🎯 TARGET: {np.round(target_pos, 4)}")
        
        # Use IK to calculate joint angles
        rospy.loginfo(f"      🧮 Solving IK for (X={SURFACE_X}, Y={target_y:+.3f}, Z={target_z:.3f})...")
        
        # Use current joint positions as initial guess
        joint_positions, ik_success, ik_error, x_error = constrained_ik(
            target_y, target_z,
            initial_guess=joints_before
        )
        
        if not ik_success:
            rospy.logwarn(f"      ⚠️ IK failed! Error: {ik_error*1000:.1f}mm, X error: {x_error*1000:.1f}mm")
            rospy.logwarn(f"      Using best-effort solution anyway...")
        else:
            rospy.loginfo(f"      ✅ IK succeeded! Error: {ik_error*1000:.1f}mm")
        
        rospy.loginfo(f"      📝 Joint command (rad): {np.round(joint_positions, 3)}")
        rospy.loginfo(f"      📝 Joint command (deg): {np.round(np.degrees(joint_positions), 1)}")
        
        # Execute action (send trajectory)
        result = self.env.move_to_joint_positions(joint_positions)
        
        # SAFETY: Handle critical robot errors (same as before)
        if result['error_code'] == -999:
            rospy.logerr("      🛑 CRITICAL ERROR! Robot is broken (NaN detected). Resetting environment...")
            next_state = self.get_state()
            reward = -100.0
            done = True
            # Get current state for distance calculation
            joints_after, vels_after = get_joint_positions_direct()
            ee_pos_after = fk(joints_after)
            target_pos = self.env.target_position  # Property, not method!
            distance = np.linalg.norm(ee_pos_after - target_pos)
            # Update min_distance tracker
            self.min_distance = min(self.min_distance, distance)
            info = {
                'error': 'robot_broken', 
                'error_code': -999,
                'distance': distance,
                'min_distance': self.min_distance,  # Added!
                'success': False,  # Robot broke, not success
                'steps': self.episode_steps,
                'ee_position': ee_pos_after,
                'target_position': target_pos
            }
            self.episode_reward += reward
            return next_state, reward, done, info
        
        if result['error_code'] == -998:
            rospy.logwarn("      ⚠️ OVERREACH PREVENTED! Robot would collapse. Penalizing and continuing...")
            next_state = self.get_state()
            reward = -50.0
            done = False
            # Get current state for distance calculation
            joints_after, vels_after = get_joint_positions_direct()
            ee_pos_after = fk(joints_after)
            target_pos = self.env.target_position  # Property, not method!
            distance = np.linalg.norm(ee_pos_after - target_pos)
            # Update min_distance tracker
            self.min_distance = min(self.min_distance, distance)
            info = {
                'error': 'overreach_prevented', 
                'error_code': -998,
                'distance': distance,
                'min_distance': self.min_distance,  # Added!
                'success': False,
                'steps': self.episode_steps,
                'ee_position': ee_pos_after,
                'target_position': target_pos
            }
            self.episode_reward += reward
            return next_state, reward, done, info
        
        if result['error_code'] == -997:
            rospy.logwarn("      ⚠️ GROUND COLLISION PREVENTED! Penalizing and continuing...")
            next_state = self.get_state()
            reward = -30.0
            done = False
            # Get current state for distance calculation
            joints_after, vels_after = get_joint_positions_direct()
            ee_pos_after = fk(joints_after)
            target_pos = self.env.target_position  # Property, not method!
            distance = np.linalg.norm(ee_pos_after - target_pos)
            # Update min_distance tracker
            self.min_distance = min(self.min_distance, distance)
            info = {
                'error': 'ground_collision_prevented', 
                'error_code': -997,
                'distance': distance,
                'min_distance': self.min_distance,  # Added!
                'success': False,
                'steps': self.episode_steps,
                'ee_position': ee_pos_after,
                'target_position': target_pos
            }
            self.episode_reward += reward
            return next_state, reward, done, info
        
        if not result['success'] and result['error_code'] not in [-5]:
            rospy.logwarn(f"      ⚠️ Action failed with error code: {result['error_code']}")
        
        # Wait for trajectory completion
        rospy.logdebug(f"Waiting {ACTION_WAIT_TIME}s for trajectory completion...")
        rospy.sleep(ACTION_WAIT_TIME)
        
        # Get next state
        next_state = self.get_state()
        ee_after = np.array(self.env.ee_position)
        joints_after, vels_after = get_joint_positions_direct()
        
        # Calculate movement
        ee_movement = np.linalg.norm(ee_after - ee_pos)
        joint_movement = np.linalg.norm(joints_after - joints_before)
        
        rospy.loginfo(f"      📍 AFTER:  ee={np.round(ee_after, 4)}, joints={np.round(joints_after, 3)}")
        rospy.loginfo(f"      📏 EE moved: {ee_movement:.4f}m, Joints moved: {joint_movement:.4f}rad")
        
        # Verify X constraint
        x_deviation = abs(ee_after[0] - SURFACE_X)
        if x_deviation > 0.01:  # More than 1cm off surface
            rospy.logwarn(f"      ⚠️ X constraint violated! ee_X={ee_after[0]:.4f}, expected={SURFACE_X:.4f}, error={x_deviation*1000:.1f}mm")
        
        # Increment step counter BEFORE calculating reward
        self.episode_steps += 1
        
        # Calculate reward
        reward, done, info = self._calculate_reward(next_state)
        
        self.episode_reward += reward
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, state):
        """
        Calculate reward based on distance to target IN Y-Z PLANE
        (X is fixed at surface, so only Y-Z matter!)
        
        SIMPLIFIED APPROACH (matching reference robotic_arm_environment):
        - +10 if distance <= 5cm (goal reached)
        - -1 otherwise
        
        Args:
            state: 11D state vector
            
        Returns:
            reward: float
            done: bool (True if goal reached)
            info: dict
        """
        # Extract EE and target positions
        ee_pos = np.array(self.env.ee_position)
        target_pos = np.array(self.env.target_position)
        
        # Calculate FULL 3D distance (for goal checking)
        distance_3d = np.linalg.norm(ee_pos - target_pos)
        
        # Also calculate Y-Z distance (for debugging)
        distance_yz = np.sqrt((ee_pos[1] - target_pos[1])**2 + (ee_pos[2] - target_pos[2])**2)
        
        # Track minimum distance (for logging)
        self.min_distance = min(self.min_distance, distance_3d)
        
        # SIMPLE BINARY REWARD (like reference project)
        if distance_3d <= GOAL_THRESHOLD:
            reward = SUCCESS_REWARD  # +10
            done = True
            rospy.loginfo(f"🎉 GOAL REACHED! 3D distance: {distance_3d*100:.2f}cm, Y-Z distance: {distance_yz*100:.2f}cm")
        else:
            reward = STEP_REWARD  # -1
            done = False
        
        # Info for debugging
        info = {
            'distance': distance_3d,
            'distance_yz': distance_yz,
            'min_distance': self.min_distance,
            'success': done,
            'steps': self.episode_steps
        }
        
        return reward, done, info


# ============================================================================
# TRAINING LOOP
# ============================================================================

def manual_test_mode(env_wrapper):
    """Interactive manual testing mode - test Y-Z positions with constrained IK"""
    rospy.loginfo("=" * 70)
    rospy.loginfo("🎮 MANUAL TEST MODE (CONSTRAINED IK - CONSERVATIVE WORKSPACE)")
    rospy.loginfo("=" * 70)
    rospy.loginfo("📝 Commands:")
    rospy.loginfo("  - Enter Y Z positions (e.g., '0.05 0.13') to move end-effector")
    rospy.loginfo(f"  - X is FIXED at {SURFACE_X}m (drawing surface)")
    rospy.loginfo("  - Type 'reset' or 'r' to reset robot to home + move target")
    rospy.loginfo("  - Type 'clear' or 'c' to erase trajectory drawing")
    rospy.loginfo("  - Press Enter to exit manual mode")
    rospy.loginfo("=" * 70)
    rospy.loginfo(f"CONSERVATIVE ranges (94% success): Y∈[{env_wrapper.y_limits[0]:.2f}, {env_wrapper.y_limits[1]:.2f}]m, Z∈[{env_wrapper.z_limits[0]:.2f}, {env_wrapper.z_limits[1]:.2f}]m")
    rospy.loginfo("=" * 70)
    
    while True:
        try:
            print("\nEnter Y Z positions in meters (space-separated):")
            print(f"Example: 0.05 0.13  (will reach X={SURFACE_X}, Y=0.05, Z=0.13)")
            print("Or 'reset'/'r' to reset, 'clear'/'c' to erase drawing, or Enter to exit")
            
            user_input = input("Y Z: ").strip()
            
            if not user_input:
                rospy.loginfo("Exiting manual test mode...")
                break
            
            # Check for reset command
            if user_input.lower() in ['reset', 'r', 'restart']:
                print("🔄 Resetting environment (robot to home + new target position)...")
                env_wrapper.reset()
                state = env_wrapper.get_state()
                target_pos = np.array(env_wrapper.env.target_position)
                print(f"✅ Environment reset!")
                print(f"   Robot moved to home: [0°, 0°, 0°, 90°]")
                print(f"   New target position: {np.round(target_pos, 4)} m")
                continue
            
            # Check for clear command
            if user_input.lower() in ['clear', 'c', 'erase']:
                env_wrapper.env.clear_trajectory()
                traj_info = env_wrapper.env.get_trajectory_info()
                print(f"✅ Trajectory cleared! (Had {traj_info['num_points']} points)")
                continue
            
            # Parse Y Z positions
            pos_str = user_input.split()
            if len(pos_str) != 2:
                print("❌ Please enter exactly 2 values (Y and Z)!")
                continue
            
            target_y = float(pos_str[0])
            target_z = float(pos_str[1])
            
            # Validate ranges
            if not (env_wrapper.y_limits[0] <= target_y <= env_wrapper.y_limits[1]):
                print(f"❌ Y out of range! Must be in [{env_wrapper.y_limits[0]}, {env_wrapper.y_limits[1]}]")
                continue
            if not (env_wrapper.z_limits[0] <= target_z <= env_wrapper.z_limits[1]):
                print(f"❌ Z out of range! Must be in [{env_wrapper.z_limits[0]}, {env_wrapper.z_limits[1]}]")
                continue
            
            print(f"\n{'='*70}")
            print(f"🎯 Target EE position: X={SURFACE_X}, Y={target_y:+.3f}, Z={target_z:.3f}")
            print(f"{'='*70}")
            
            # Get state before
            state_before = env_wrapper.get_state()
            ee_pos_before = np.array(env_wrapper.env.ee_position)
            target_pos = np.array(env_wrapper.env.target_position)
            joints_before, vels_before = get_joint_positions_direct()
            dist_before = np.linalg.norm(ee_pos_before - target_pos)
            
            print(f"\nBEFORE ACTION:")
            print(f"  Current joints:   {np.round(joints_before, 3)} rad = {np.round(np.degrees(joints_before), 1)}°")
            print(f"  End-effector:     {np.round(ee_pos_before, 4)} m")
            print(f"  Target sphere:    {np.round(target_pos, 4)} m")
            print(f"  Distance to goal: {dist_before:.4f}m ({dist_before*100:.2f}cm)")
            
            # Solve IK
            print(f"\n⏳ Solving IK for (X={SURFACE_X}, Y={target_y:+.3f}, Z={target_z:.3f})...")
            target_joints, ik_success, ik_error, x_error = constrained_ik(
                target_y, target_z,
                initial_guess=joints_before
            )
            
            if not ik_success:
                print(f"⚠️  IK solution not perfect! Error: {ik_error*1000:.1f}mm, X error: {x_error*1000:.1f}mm")
                print(f"   Continuing with best-effort solution...")
            else:
                print(f"✅ IK solved! Error: {ik_error*1000:.1f}mm")
            
            print(f"\n🎯 Calculated joint angles:")
            print(f"   {target_joints} rad")
            print(f"   {np.round(np.degrees(target_joints), 1)}°")
            
            # Send action
            print(f"\n⏳ Executing action...")
            start_time = time.time()
            
            # Call environment's move_to_joint_positions directly
            result = env_wrapper.env.move_to_joint_positions(target_joints)
            
            if not result['success'] and result['error_code'] not in [-5]:
                print(f"⚠️  Trajectory command failed with error code: {result['error_code']}")
            
            # Wait for trajectory completion
            rospy.sleep(ACTION_WAIT_TIME)
            
            # Get state after
            state_after = env_wrapper.get_state()
            ee_pos_after = np.array(env_wrapper.env.ee_position)
            joints_after, vels_after = get_joint_positions_direct()
            dist_after = np.linalg.norm(ee_pos_after - target_pos)
            
            # Calculate movements
            ee_movement = np.linalg.norm(ee_pos_after - ee_pos_before)
            joint_movement = np.linalg.norm(joints_after - joints_before)
            
            # Calculate errors
            joint_errors = np.abs(joints_after - target_joints)
            max_joint_error = np.max(joint_errors)
            max_vel = np.max(np.abs(vels_after)) if vels_after is not None else 0.0
            
            # Check X constraint
            x_deviation = abs(ee_pos_after[0] - SURFACE_X)
            
            elapsed = time.time() - start_time
            
            # Print results
            print(f"\nAFTER ACTION:")
            print(f"  Final joints:     {np.round(joints_after, 3)} rad = {np.round(np.degrees(joints_after), 1)}°")
            print(f"  End-effector:     {np.round(ee_pos_after, 4)} m")
            print(f"  X constraint:     {'✅ SATISFIED' if x_deviation < 0.01 else f'⚠️ VIOLATED ({x_deviation*1000:.1f}mm off)'}")
            print(f"  Distance to goal: {dist_after:.4f}m ({dist_after*100:.2f}cm)")
            print(f"  EE moved:         {ee_movement:.4f}m ({ee_movement*100:.2f}cm)")
            print(f"  Joints moved:     {joint_movement:.4f}rad ({np.degrees(joint_movement):.1f}°)")
            
            print(f"\n{'='*70}")
            print(f"MOVEMENT VALIDATION:")
            print(f"{'='*70}")
            print(f"Target joints:    {target_joints}")
            print(f"Reached joints:   {joints_after}")
            print(f"Joint errors:     {np.round(joint_errors, 4)} rad = {np.round(np.degrees(joint_errors), 2)}°")
            print(f"Max joint error:  {max_joint_error:.4f} rad = {np.degrees(max_joint_error):.2f}°")
            print(f"Final velocities: {np.round(vels_after, 4)} rad/s")
            print(f"Max velocity:     {max_vel:.4f} rad/s")
            
            # Check tolerance (like test_simple_movement.py)
            tolerance = 0.1  # 0.1 radian = ~5.7 degrees
            position_ok = np.allclose(joints_after, target_joints, atol=tolerance)
            velocity_ok = max_vel < 0.05
            
            print(f"\n{'='*70}")
            print(f"RESULTS:")
            print(f"{'='*70}")
            print(f"Execution time:   {elapsed:.1f}s")
            print(f"Position reached: {'✅ YES' if position_ok else '❌ NO'} (tolerance: ±{tolerance} rad = ±{np.degrees(tolerance):.1f}°)")
            print(f"Robot stopped:    {'✅ YES' if velocity_ok else '⚠️  OSCILLATING'} (max vel: {max_vel:.4f} rad/s)")
            print(f"Distance improved: {dist_before - dist_after:.4f}m")
            
            # CHECK FOR GOAL REACHED (like RL training)
            # Note: Distance is to SPHERE CENTER (sphere radius = 1cm)
            if dist_after <= GOAL_THRESHOLD:
                print(f"\n🎉🎉🎉 GOAL REACHED! 🎉🎉🎉")
                print(f"    Distance to center: {dist_after*100:.2f}cm ≤ {GOAL_THRESHOLD*100:.0f}cm threshold")
                print(f"    Sphere radius: 1cm")
                print(f"    Reward would be: +{SUCCESS_REWARD}")
            else:
                print(f"\n❌ Goal not reached yet")
                print(f"    Distance to center: {dist_after*100:.2f}cm > {GOAL_THRESHOLD*100:.0f}cm threshold")
                print(f"    Sphere radius: 1cm")
                print(f"    Need to get closer by: {(dist_after - GOAL_THRESHOLD)*100:.2f}cm")
                print(f"    Reward would be: {STEP_REWARD}")
            
            # Show trajectory info
            traj_info = env_wrapper.env.get_trajectory_info()
            if traj_info['num_points'] > 1:
                print(f"🎨 Trajectory:    {traj_info['num_points']} points, {traj_info['length_cm']:.2f}cm total path")
            
            if position_ok and velocity_ok:
                print(f"\n✅ SUCCESS! Robot reached target and stopped!")
            elif position_ok:
                print(f"\n⚠️  PARTIAL SUCCESS: Position OK but robot still oscillating")
                print(f"   (This is acceptable for RL training)")
            else:
                print(f"\n❌ FAILED! Robot did not reach target position")
                print(f"   Max error: {np.degrees(max_joint_error):.2f}° (tolerance: ±{np.degrees(tolerance):.1f}°)")
            
            print(f"{'='*70}\n")
            
        except ValueError:
            print("❌ Invalid input! Please enter 4 numbers.")
        except KeyboardInterrupt:
            print("\nExiting manual test mode...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


def train_ddpg_gazebo():
    """Main training loop"""
    
    # Initialize ROS node
    rospy.init_node('ddpg_gazebo_training', anonymous=True)
    rospy.loginfo("🚀 Starting DDPG Training for Gazebo 4DOF Robot")
    rospy.loginfo("=" * 70)
    
    # Initialize persistent /joint_states subscriber (CRITICAL for speed)
    init_joint_state_subscriber()
    
    # Create environment
    rospy.loginfo("📦 Creating Gazebo RL environment...")
    base_env = RLEnvironmentNoetic(max_episode_steps=MAX_STEPS_PER_EPISODE)
    rospy.sleep(3.0)  # Wait for initialization
    
    # Wrap environment
    env = GazeboRLWrapper(base_env)
    
    # ========================================================================
    # INTERACTIVE MENU: Choose mode
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("🎮 TRAINING MENU")
    print("=" * 70)
    print("1. Manual Test Mode - Test joint angles manually")
    print("2. RL Training Mode - Train DDPG agent")
    print("=" * 70)
    print("💡 TIP: Press Ctrl+C anytime to exit")
    print("=" * 70)
    
    while True:
        choice = input("Choose mode (1 or 2): ").strip()
        if choice == '1':
            manual_test_mode(env)
            print("\nReturning to menu...")
            continue
        elif choice == '2':
            break
        else:
            print("❌ Invalid choice! Please enter 1 or 2.")
    
    # ========================================================================
    # RL TRAINING CONFIGURATION
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("⚙️  RL TRAINING CONFIGURATION")
    print("=" * 70)
    
    # Get number of episodes
    while True:
        try:
            episodes_input = input(f"Number of episodes (default {NUM_EPISODES}): ").strip()
            if not episodes_input:
                num_episodes = NUM_EPISODES
                break
            num_episodes = int(episodes_input)
            if num_episodes > 0:
                break
            print("❌ Please enter a positive number!")
        except ValueError:
            print("❌ Invalid input! Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Exiting training script. Goodbye!")
            rospy.signal_shutdown("User requested exit")
            sys.exit(0)
    
    # Get steps per episode
    while True:
        try:
            steps_input = input(f"Steps per episode (default {MAX_STEPS_PER_EPISODE}): ").strip()
            if not steps_input:
                max_steps = MAX_STEPS_PER_EPISODE
                break
            max_steps = int(steps_input)
            if max_steps > 0:
                break
            print("❌ Please enter a positive number!")
        except ValueError:
            print("❌ Invalid input! Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Exiting training script. Goodbye!")
            rospy.signal_shutdown("User requested exit")
            sys.exit(0)
    
    rospy.loginfo("=" * 70)
    rospy.loginfo(f"✅ Configuration:")
    rospy.loginfo(f"   Episodes: {num_episodes}")
    rospy.loginfo(f"   Steps per episode: {max_steps}")
    rospy.loginfo(f"   Estimated time: ~{num_episodes * max_steps * ACTION_WAIT_TIME / 60:.1f} minutes")
    rospy.loginfo("=" * 70)
    
    # Create DDPG agent (11D state, 2D action for Y-Z control)
    rospy.loginfo("🤖 Creating DDPG agent...")
    agent = DDPGAgentGazebo(state_dim=11, n_actions=2, max_action=1.0, min_action=-1.0)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'ddpg_gazebo')
    os.makedirs(checkpoint_dir, exist_ok=True)
    rospy.loginfo(f"📁 Checkpoint directory: {checkpoint_dir}")
    
    # Create training logs directory
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_logs')
    os.makedirs(logs_dir, exist_ok=True)
    rospy.loginfo(f"📁 Training logs directory: {logs_dir}")
    
    # Training tracking
    episode_rewards = []
    episode_distances = []
    episode_successes = []
    actor_losses = []
    critic_losses = []
    
    best_avg_reward = -float('inf')
    
    rospy.loginfo("=" * 70)
    rospy.loginfo(f"🎯 Training Configuration:")
    rospy.loginfo(f"   Episodes: {num_episodes}")
    rospy.loginfo(f"   Steps per episode: {max_steps}")
    rospy.loginfo(f"   Action time: {ACTION_WAIT_TIME}s")
    rospy.loginfo(f"   Episode time: ~{max_steps * ACTION_WAIT_TIME:.1f}s")
    rospy.loginfo(f"   Goal threshold: {GOAL_THRESHOLD}m ({GOAL_THRESHOLD*100:.1f}cm)")
    rospy.loginfo("=" * 70)
    
    # Training loop
    for episode in range(num_episodes):
        episode_start = time.time()
        
        # Reset environment
        state = env.reset()
        done = False
        
        rospy.loginfo(f"\n{'='*70}")
        rospy.loginfo(f"📍 Episode {episode+1}/{num_episodes}")
        
        # Episode loop
        step_count = 0
        while not done and step_count < max_steps:
            # Choose action (with exploration noise)
            evaluate = (episode % EVAL_INTERVAL == 0)  # Deterministic every N episodes
            action = agent.choose_action(state, evaluate=evaluate)
            
            # Execute action
            rospy.loginfo(f"   🎯 Step {step_count+1}/{max_steps}: Executing action...")
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            step_count += 1
            
            # Log step
            rospy.loginfo(f"      ✓ distance={info['distance']:.4f}m, reward={reward:.2f}, done={done}")
        
        # Episode finished
        episode_time = time.time() - episode_start
        
        # Learn from experience (multiple optimization steps)
        actor_loss_ep = None
        critic_loss_ep = None
        
        if episode >= LEARNING_STARTS:
            for _ in range(OPT_STEPS_PER_EPISODE):
                actor_loss, critic_loss = agent.learn()
                if actor_loss is not None:
                    actor_loss_ep = actor_loss
                    critic_loss_ep = critic_loss
        
        # Track statistics
        episode_rewards.append(env.episode_reward)
        episode_distances.append(info['min_distance'])
        episode_successes.append(1 if info['success'] else 0)
        
        if actor_loss_ep is not None:
            actor_losses.append(actor_loss_ep)
            critic_losses.append(critic_loss_ep)
        
        # Calculate running average
        avg_reward = np.mean(episode_rewards[-100:])
        success_rate = np.mean(episode_successes[-100:]) * 100
        
        # Log episode summary
        rospy.loginfo(f"{'='*70}")
        rospy.loginfo(f"📊 Episode {episode+1} Summary:")
        rospy.loginfo(f"   Total reward: {env.episode_reward:.2f}")
        rospy.loginfo(f"   Min distance: {info['min_distance']:.4f}m ({info['min_distance']*100:.2f}cm)")
        rospy.loginfo(f"   Success: {'✅ YES' if info['success'] else '❌ NO'}")
        rospy.loginfo(f"   Episode time: {episode_time:.1f}s")
        rospy.loginfo(f"   Avg reward (100): {avg_reward:.2f}")
        rospy.loginfo(f"   Success rate (100): {success_rate:.1f}%")
        
        # Show trajectory info
        traj_info = env.env.get_trajectory_info()
        if traj_info['num_points'] > 0:
            rospy.loginfo(f"   🎨 Trajectory: {traj_info['num_points']} points, {traj_info['length_cm']:.2f}cm total length")
        
        if actor_loss_ep is not None:
            rospy.loginfo(f"   Actor loss: {actor_loss_ep:.4f}")
            rospy.loginfo(f"   Critic loss: {critic_loss_ep:.4f}")
        
        # Clear trajectory drawing for next episode
        env.env.clear_trajectory()
        
        # Save best model
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save_models(episode=None)  # Save as best
            rospy.loginfo(f"   💾 New best model saved! (avg reward: {avg_reward:.2f})")
        
        # Periodic save
        if (episode + 1) % SAVE_INTERVAL == 0:
            agent.save_models(episode=episode+1)
            rospy.loginfo(f"   💾 Checkpoint saved at episode {episode+1}")
        
        rospy.loginfo(f"{'='*70}\n")
    
    # Training complete
    rospy.loginfo("=" * 70)
    rospy.loginfo("🎉 Training Complete!")
    rospy.loginfo(f"   Best avg reward: {best_avg_reward:.2f}")
    rospy.loginfo(f"   Final success rate: {success_rate:.1f}%")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'episode_rewards': episode_rewards,
        'episode_distances': episode_distances,
        'episode_successes': episode_successes,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'config': {
            'num_episodes': NUM_EPISODES,
            'max_steps': MAX_STEPS_PER_EPISODE,
            'action_time': ACTION_WAIT_TIME,
            'goal_threshold': GOAL_THRESHOLD
        }
    }
    
    results_file = os.path.join(logs_dir, f'training_results_{timestamp}.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    rospy.loginfo(f"   📁 Results saved to: {results_file}")
    rospy.loginfo("=" * 70)
    
    # Plot results
    plot_training_results(results, timestamp, logs_dir)
    
    return results


def plot_training_results(results, timestamp, logs_dir):
    """Plot training results with individual episode data and moving averages"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Episode rewards with moving average
    ax = axes[0, 0]
    episodes = np.arange(len(results['episode_rewards']))
    ax.plot(episodes, results['episode_rewards'], 'o-', alpha=0.3, markersize=3, label='Episode Reward')
    
    # Moving average for rewards
    window = min(20, len(results['episode_rewards']) // 2) if len(results['episode_rewards']) > 1 else 1
    if len(results['episode_rewards']) >= window and window > 1:
        moving_avg = np.convolve(results['episode_rewards'], 
                                np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(results['episode_rewards'])), 
               moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Average')
    
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Total Reward', fontsize=11)
    ax.set_title('Episode Rewards', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Minimum distance to goal
    ax = axes[0, 1]
    ax.plot(episodes, results['episode_distances'], 'o-', alpha=0.5, markersize=3, color='blue', label='Min Distance')
    ax.axhline(y=GOAL_THRESHOLD, color='r', linestyle='--', linewidth=2, label=f'Goal Threshold ({GOAL_THRESHOLD}m)')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Min Distance (m)', fontsize=11)
    ax.set_title('Minimum Distance to Goal', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Success/Fail for each episode with moving average
    ax = axes[1, 0]
    
    # Plot individual success (1) and fail (0) as scatter points
    successes = np.array(results['episode_successes'])
    success_episodes = episodes[successes == 1]
    fail_episodes = episodes[successes == 0]
    
    ax.scatter(success_episodes, np.ones_like(success_episodes), 
               color='green', marker='o', s=50, alpha=0.6, label='Success', zorder=3)
    ax.scatter(fail_episodes, np.zeros_like(fail_episodes), 
               color='red', marker='x', s=50, alpha=0.6, label='Fail', zorder=3)
    
    # Moving average success rate
    window = min(20, len(results['episode_successes']) // 2) if len(results['episode_successes']) > 1 else 1
    if len(results['episode_successes']) >= window and window > 1:
        success_rate = np.convolve(results['episode_successes'], 
                                   np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(results['episode_successes'])), 
               success_rate, 'r-', linewidth=2, label=f'{window}-Episode Average', zorder=2)
    
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Success (1) / Fail (0)', fontsize=11)
    ax.set_title('Episode Success/Fail with Moving Average', fontsize=12, fontweight='bold')
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Actor/Critic losses
    ax = axes[1, 1]
    if len(results['actor_losses']) > 0:
        ax.plot(results['actor_losses'], 'o-', alpha=0.5, markersize=3, label='Actor Loss')
        ax.plot(results['critic_losses'], 'o-', alpha=0.5, markersize=3, label='Critic Loss')
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Training Losses', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_file = os.path.join(logs_dir, f'training_plot_{timestamp}.png')
    plt.savefig(plot_file, dpi=150)
    rospy.loginfo(f"   📊 Training plot saved to: {plot_file}")
    
    plt.close()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        results = train_ddpg_gazebo()
        rospy.loginfo("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        rospy.loginfo("\n⚠️ Training interrupted by user")
        
    except Exception as e:
        rospy.logerr(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
