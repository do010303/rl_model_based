#!/usr/bin/env python3
"""
DDPG Training for 4DOF Gazebo Robot - Short Episodes
Integrates the working test_simple_movement.py timing with DDPG agent

Key Features:
- Episodes: Only 5 actions per episode (~17.5s total)
- Action timing: 3.5s per action (3s trajectory + 0.5s buffer)
- State space: 14D [4 joint angles, 4 velocities, 3 ee_pos, 3 target_pos]
- Fast persistent /joint_states subscriber (instant reads)
- Optimized for rapid RL training in Gazebo
"""

import rospy
import numpy as np
import sys
import os
import time
from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'robot_ws/src/new_robot_arm_urdf/scripts'))

from agents.ddpg_gazebo import DDPGAgentGazebo
from robot_ws.src.new_robot_arm_urdf.scripts.main_rl_environment_noetic import RLEnvironmentNoetic


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

# Action timing (matching test_simple_movement.py)
TRAJECTORY_TIME = 3.0       # Trajectory execution time
BUFFER_TIME = 0.5           # Buffer after trajectory
ACTION_WAIT_TIME = TRAJECTORY_TIME + BUFFER_TIME  # 3.5s total


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
    Optimized for fast training with 3.5s action execution
    """
    
    def __init__(self, env):
        self.env = env
        self.state_dim = 14  # [4 joints, 4 vels, 3 ee_pos, 3 target_pos]
        self.action_dim = 4  # 4 joint positions
        
        # Joint limits (from environment)
        self.joint_limits_low = np.array([-np.pi, -np.pi/2, -np.pi/2, -np.pi/2])
        self.joint_limits_high = np.array([np.pi, np.pi/2, np.pi/2, np.pi/2])
        
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
        # Denormalize action to joint positions
        joint_positions = self._denormalize_action(action)
        
        # Execute action (send trajectory)
        result = self.env.move_to_joint_positions(joint_positions)
        
        # Wait for trajectory completion (FAST: 3.5s total)
        rospy.sleep(ACTION_WAIT_TIME)
        
        # Get next state
        next_state = self.get_state()
        
        # Calculate reward
        reward, done, info = self._calculate_reward(next_state)
        
        self.episode_reward += reward
        self.episode_steps += 1
        
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
            done: bool
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
        
        # Episode done if goal reached or max steps
        done = success or (self.episode_steps >= MAX_STEPS_PER_EPISODE)
        
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
    
    # Create DDPG agent
    rospy.loginfo("🤖 Creating DDPG agent...")
    agent = DDPGAgentGazebo(state_dim=14, n_actions=4, max_action=1.0, min_action=-1.0)
    
    # Training tracking
    episode_rewards = []
    episode_distances = []
    episode_successes = []
    actor_losses = []
    critic_losses = []
    
    best_avg_reward = -float('inf')
    
    rospy.loginfo("=" * 70)
    rospy.loginfo(f"🎯 Training Configuration:")
    rospy.loginfo(f"   Episodes: {NUM_EPISODES}")
    rospy.loginfo(f"   Steps per episode: {MAX_STEPS_PER_EPISODE}")
    rospy.loginfo(f"   Action time: {ACTION_WAIT_TIME}s")
    rospy.loginfo(f"   Episode time: ~{MAX_STEPS_PER_EPISODE * ACTION_WAIT_TIME:.1f}s")
    rospy.loginfo(f"   Goal threshold: {GOAL_THRESHOLD}m ({GOAL_THRESHOLD*100:.1f}cm)")
    rospy.loginfo("=" * 70)
    
    # Training loop
    for episode in range(NUM_EPISODES):
        episode_start = time.time()
        
        # Reset environment
        state = env.reset()
        done = False
        
        rospy.loginfo(f"\n{'='*70}")
        rospy.loginfo(f"📍 Episode {episode+1}/{NUM_EPISODES}")
        
        # Episode loop
        while not done:
            # Choose action (with exploration noise)
            evaluate = (episode % EVAL_INTERVAL == 0)  # Deterministic every N episodes
            action = agent.choose_action(state, evaluate=evaluate)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Log step
            rospy.loginfo(f"   Step {info['steps']}/{MAX_STEPS_PER_EPISODE}: "
                         f"distance={info['distance']:.4f}m, reward={reward:.2f}")
        
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
        
        if actor_loss_ep is not None:
            rospy.loginfo(f"   Actor loss: {actor_loss_ep:.4f}")
            rospy.loginfo(f"   Critic loss: {critic_loss_ep:.4f}")
        
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
    
    results_file = f'training_results_{timestamp}.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    rospy.loginfo(f"   📁 Results saved to: {results_file}")
    rospy.loginfo("=" * 70)
    
    # Plot results
    plot_training_results(results, timestamp)
    
    return results


def plot_training_results(results, timestamp):
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
    plot_file = f'training_plot_{timestamp}.png'
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
