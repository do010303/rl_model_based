#!/usr/bin/env python3
"""
DDPG Training for 4DOF Gazebo Robot - Short Episodes
Integrates the working test_simple_movement.py timing with DDPG agent

Key Features:
- Episodes: Only 5 actions per episode (~5.75s total)
- Action timing: 1.15s per action (1s trajectory + 0.15s buffer)
- State space: 14D [4 joint angles, 4 velocities, 3 ee_pos, 3 target_pos]
- Fast persistent /joint_states subscriber (instant reads)
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

# Reward settings
GOAL_THRESHOLD = 0.02       # 2cm threshold for success (meters)
DISTANCE_WEIGHT = 10.0      # Weight for distance reward
SUCCESS_BONUS = 50.0        # Bonus for reaching goal
STEP_PENALTY = 0.1          # Small penalty per step

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
    Optimized for ultra-fast training with 1.15s action execution
    """
    
    def __init__(self, env):
        self.env = env
        self.state_dim = 14  # [4 joints, 4 vels, 3 ee_pos, 3 target_pos]
        self.action_dim = 4  # 4 joint positions
        
        # Joint limits (UPDATED: Joint1 ±90°, Joint2-3 ±90°, Joint4 0-180°)
        self.joint_limits_low = np.array([-np.pi/2, -np.pi/2, -np.pi/2, 0.0])
        self.joint_limits_high = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi])
        
        # Track episode stats
        self.episode_reward = 0
        self.episode_steps = 0
        self.min_distance = float('inf')
        
        rospy.loginfo("✅ GazeboRLWrapper initialized")
    
    def get_state(self):
        """
        Get current state: [4 joint angles, 4 velocities, 3 ee_pos, 3 target_pos]
        
        Returns:
            14D numpy array
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
        
        # Combine into state vector
        state = np.concatenate([
            joint_positions,      # 4D
            joint_velocities,     # 4D
            ee_pos,              # 3D
            target_pos           # 3D
        ])
        
        return state.astype(np.float32)
    
    def reset(self):
        """
        Reset environment and return initial state
        
        Returns:
            14D state vector
        """
        rospy.loginfo("🔄 Resetting environment...")
        success = self.env.reset_environment()
        
        if not success:
            rospy.logwarn("⚠️ Environment reset failed, retrying...")
            rospy.sleep(1.0)
            self.env.reset_environment()
        
        # Reset episode tracking
        self.episode_reward = 0
        self.episode_steps = 0
        self.min_distance = float('inf')
        
        # Wait for robot to settle
        rospy.sleep(1.0)
        
        # Get initial state
        state = self.get_state()
        
        return state
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done, info)
        
        Args:
            action: 4D normalized action vector [-1, 1]
            
        Returns:
            next_state: 14D state vector
            reward: float
            done: bool
            info: dict with extra info
        """
        # Get state BEFORE action
        state_before = self.get_state()
        ee_before = state_before[8:11]
        target = state_before[11:14]
        joints_before, vels_before = get_joint_positions_direct()
        
        # Denormalize action to joint positions
        joint_positions = self._denormalize_action(action)
        
        # Log action details
        rospy.loginfo(f"      📝 Normalized action: {np.round(action, 3)}")
        rospy.loginfo(f"      📝 Joint command (rad): {np.round(joint_positions, 3)}")
        rospy.loginfo(f"      📝 Joint command (deg): {np.round(np.degrees(joint_positions), 1)}")
        rospy.loginfo(f"      📍 BEFORE: ee={np.round(ee_before, 4)}, joints={np.round(joints_before, 3)}")
        rospy.loginfo(f"      🎯 TARGET: {np.round(target, 4)}")
        
        # Execute action (send trajectory)
        result = self.env.move_to_joint_positions(joint_positions)
        
        # SAFETY: Handle critical robot errors (NaN, broken state)
        if result['error_code'] == -999:
            rospy.logerr("      🛑 CRITICAL ERROR! Robot is broken. Resetting environment...")
            # Force episode to end
            next_state = self.get_state()
            reward = -100.0  # Large penalty for breaking the robot
            done = True
            info = {'error': 'robot_broken', 'error_code': -999}
            self.episode_reward += reward
            return next_state, reward, done, info
        
        if not result['success'] and result['error_code'] not in [-5]:
            rospy.logwarn(f"      ⚠️ Action failed with error code: {result['error_code']}")
        
        # Wait for trajectory completion (FAST: 3.5s total)
        rospy.logdebug(f"Waiting {ACTION_WAIT_TIME}s for trajectory completion...")
        rospy.sleep(ACTION_WAIT_TIME)
        
        # Get next state
        next_state = self.get_state()
        ee_after = next_state[8:11]
        joints_after, vels_after = get_joint_positions_direct()
        
        # Calculate movement
        ee_movement = np.linalg.norm(ee_after - ee_before)
        joint_movement = np.linalg.norm(joints_after - joints_before)
        
        rospy.loginfo(f"      📍 AFTER:  ee={np.round(ee_after, 4)}, joints={np.round(joints_after, 3)}")
        rospy.loginfo(f"      📏 EE moved: {ee_movement:.4f}m, Joints moved: {joint_movement:.4f}rad")
        
        # Increment step counter BEFORE calculating reward
        self.episode_steps += 1
        
        # Calculate reward
        reward, done, info = self._calculate_reward(next_state)
        
        self.episode_reward += reward
        
        return next_state, reward, done, info
    
    def _denormalize_action(self, action):
        """
        Convert normalized action [-1, 1] to actual joint positions
        
        Args:
            action: 4D vector in [-1, 1]
            
        Returns:
            4D joint positions in radians
        """
        # Linear mapping: [-1, 1] -> [joint_min, joint_max]
        joint_positions = self.joint_limits_low + \
                         (action + 1) * 0.5 * (self.joint_limits_high - self.joint_limits_low)
        
        # Clip to ensure within limits
        joint_positions = np.clip(joint_positions, self.joint_limits_low, self.joint_limits_high)
        
        return joint_positions
    
    def _calculate_reward(self, state):
        """
        Calculate reward based on distance to target
        
        Args:
            state: 14D state vector
            
        Returns:
            reward: float
            done: bool (ONLY True if goal reached, NOT on max steps)
            info: dict
        """
        # Extract positions from state
        ee_pos = state[8:11]      # Elements 8-10 are ee_pos
        target_pos = state[11:14]  # Elements 11-13 are target_pos
        
        # Calculate distance
        distance = np.linalg.norm(ee_pos - target_pos)
        
        # Track minimum distance
        self.min_distance = min(self.min_distance, distance)
        
        # Reward components
        distance_reward = -DISTANCE_WEIGHT * distance
        step_penalty = -STEP_PENALTY
        
        # Success bonus
        success = distance < GOAL_THRESHOLD
        success_bonus = SUCCESS_BONUS if success else 0.0
        
        # Total reward
        reward = distance_reward + step_penalty + success_bonus
        
        # Episode done ONLY if goal reached (not on max steps - that's handled in training loop)
        done = success
        
        info = {
            'distance': distance,
            'min_distance': self.min_distance,
            'success': success,
            'steps': self.episode_steps
        }
        
        return reward, done, info


# ============================================================================
# TRAINING LOOP
# ============================================================================

def manual_test_mode(env_wrapper):
    """Interactive manual testing mode - test joint angles before training"""
    rospy.loginfo("=" * 70)
    rospy.loginfo("🎮 MANUAL TEST MODE")
    rospy.loginfo("=" * 70)
    rospy.loginfo("📝 Commands:")
    rospy.loginfo("  - Enter joint angles (e.g., '0.1 0 0 0') to move robot")
    rospy.loginfo("  - Type 'clear' or 'c' to erase trajectory drawing")
    rospy.loginfo("  - Press Enter to exit manual mode")
    rospy.loginfo("=" * 70)
    
    while True:
        try:
            print("\nEnter 4 joint angles in radians (space-separated):")
            print("Example: 0.1 0 0 0")
            print("Or 'clear'/'c' to erase drawing, or Enter to exit")
            
            user_input = input("Joint angles: ").strip()
            
            if not user_input:
                rospy.loginfo("Exiting manual test mode...")
                break
            
            # Check for clear command
            if user_input.lower() in ['clear', 'c', 'erase', 'reset']:
                env_wrapper.env.clear_trajectory()
                traj_info = env_wrapper.env.get_trajectory_info()
                print(f"✅ Trajectory cleared! (Had {traj_info['num_points']} points)")
                continue
            
            # Parse joint angles
            angles_str = user_input.split()
            if len(angles_str) != 4:
                print("❌ Please enter exactly 4 values!")
                continue
            
            target_joints = np.array([float(a) for a in angles_str])
            
            print(f"\n{'='*70}")
            print(f"🎯 Target joint angles: {target_joints} rad")
            print(f"🎯 Target joint angles: {np.round(np.degrees(target_joints), 1)}°")
            print(f"{'='*70}")
            
            # Get state before
            state_before = env_wrapper.get_state()
            ee_before = state_before[8:11]
            target_pos = state_before[11:14]
            joints_before, vels_before = get_joint_positions_direct()
            dist_before = np.linalg.norm(ee_before - target_pos)
            
            print(f"\nBEFORE ACTION:")
            print(f"  Current joints:   {np.round(joints_before, 3)} rad = {np.round(np.degrees(joints_before), 1)}°")
            print(f"  End-effector:     {np.round(ee_before, 4)} m")
            print(f"  Target sphere:    {np.round(target_pos, 4)} m")
            print(f"  Distance to goal: {dist_before:.4f}m ({dist_before*100:.2f}cm)")
            
            # Send action DIRECTLY (no normalization - this is confusing!)
            print(f"\n⏳ Executing action...")
            start_time = time.time()
            
            # Call environment's move_to_joint_positions directly
            result = env_wrapper.env.move_to_joint_positions(target_joints)
            
            if not result['success'] and result['error_code'] not in [-5]:
                print(f"⚠️  Trajectory command failed with error code: {result['error_code']}")
            
            # Wait for trajectory completion (3.5s like test_simple_movement.py)
            rospy.sleep(ACTION_WAIT_TIME)
            
            # Get state after
            state_after = env_wrapper.get_state()
            ee_after = state_after[8:11]
            joints_after, vels_after = get_joint_positions_direct()
            dist_after = np.linalg.norm(ee_after - target_pos)
            
            # Calculate movements
            ee_movement = np.linalg.norm(ee_after - ee_before)
            joint_movement = np.linalg.norm(joints_after - joints_before)
            
            # Calculate errors
            joint_errors = np.abs(joints_after - target_joints)
            max_joint_error = np.max(joint_errors)
            max_vel = np.max(np.abs(vels_after)) if vels_after is not None else 0.0
            
            elapsed = time.time() - start_time
            
            # Print results
            print(f"\nAFTER ACTION:")
            print(f"  Final joints:     {np.round(joints_after, 3)} rad = {np.round(np.degrees(joints_after), 1)}°")
            print(f"  End-effector:     {np.round(ee_after, 4)} m")
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
    
    # Create DDPG agent
    rospy.loginfo("🤖 Creating DDPG agent...")
    agent = DDPGAgentGazebo(state_dim=14, n_actions=4, max_action=1.0, min_action=-1.0)
    
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
    """Plot training results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Episode rewards
    ax = axes[0, 0]
    ax.plot(results['episode_rewards'], alpha=0.3, label='Episode Reward')
    
    # Moving average
    window = 20
    if len(results['episode_rewards']) >= window:
        moving_avg = np.convolve(results['episode_rewards'], 
                                np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(results['episode_rewards'])), 
               moving_avg, 'r-', label=f'{window}-Episode Average')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Minimum distance to goal
    ax = axes[0, 1]
    ax.plot(results['episode_distances'], alpha=0.5)
    ax.axhline(y=GOAL_THRESHOLD, color='r', linestyle='--', label=f'Goal Threshold ({GOAL_THRESHOLD}m)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Min Distance (m)')
    ax.set_title('Minimum Distance to Goal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Success rate
    ax = axes[1, 0]
    window = 20
    if len(results['episode_successes']) >= window:
        success_rate = np.convolve(results['episode_successes'], 
                                   np.ones(window)/window, mode='valid') * 100
        ax.plot(range(window-1, len(results['episode_successes'])), 
               success_rate, 'g-')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'Success Rate ({window}-Episode Moving Average)')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Actor/Critic losses
    ax = axes[1, 1]
    if len(results['actor_losses']) > 0:
        ax.plot(results['actor_losses'], alpha=0.5, label='Actor Loss')
        ax.plot(results['critic_losses'], alpha=0.5, label='Critic Loss')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
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
