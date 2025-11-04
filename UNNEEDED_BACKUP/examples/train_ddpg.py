"""
Quick Start Training Script for 4-DOF Robot Arm
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.robot_4dof_env import Robot4DOFEnv
from agents.ddpg_agent import DDPGAgent
from replay_memory.replay_buffer import ReplayBuffer
from utils.her import HER
from mbpo_trainer import MBPOTrainer

def train_ddpg(episodes: int = 5, render: bool = False) -> Dict:
    """
    Train DDPG agent on 4-DOF robot arm task.
    
    Args:
        episodes: Number of training episodes
        render: Whether to render environment during training
        
    Returns:
        Training results dictionary
    """
    
    # Environment configuration
    env_config = {
        'max_steps': 200,
        'success_distance': 0.05,
        'dense_reward': True,
        'success_reward': 100.0
    }
    
    # Agent configuration  
    agent_config = {
        'lr_actor': 0.001,
        'lr_critic': 0.002,
        'gamma': 0.99,
        'tau': 0.005,
        'noise_std': 0.2,
        'noise_decay': 0.995,
        'hidden_dims': [256, 128]
    }
    
    # Initialize environment and agent
    env = Robot4DOFEnv(config=env_config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=agent_config
    )
    # Initialize replay buffer and HER
    replay_buffer = ReplayBuffer(capacity=100000)
    her = HER(replay_buffer=replay_buffer, k=4, strategy='future')
    
    # Training statistics
    episode_rewards = []
    success_rate_history = []
    distance_history = []
    
    print("Starting DDPG Training for 4-DOF Robot Arm")
    print(f"Episodes: {episodes}, Max Steps: {env_config['max_steps']}")
    print("-" * 60)
    
    for episode in range(episodes):
        # Reset environment and episode stats
        state, info = env.reset()
        episode_reward = 0.0
        episode_success = False
        episode_distances = []
        # Store episode trajectory for HER
        episode_states = []
        episode_actions = []
        episode_rewards_ep = []
        episode_next_states = []
        episode_dones = []
        for step in range(env_config['max_steps']):
            action = agent.act(state, add_noise=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Store transition
            episode_states.append(state.copy())
            episode_actions.append(action.copy())
            episode_rewards_ep.append(reward)
            episode_next_states.append(next_state.copy())
            episode_dones.append(done)
            # Update statistics
            episode_reward += reward
            episode_distances.append(info['distance_to_target'])
            if info.get('goal_reached', False):
                episode_success = True
            if render and episode % 50 == 0:
                env.render(mode='human')
            state = next_state
            if done:
                break
        # HER: add episode to buffer
        her.add_episode(episode_states, episode_actions, episode_rewards_ep, episode_next_states, episode_dones, info)
        # Agent update
        batch_size = 64
        if len(replay_buffer) > batch_size:
            for _ in range(min(step, 40)):
                batch = replay_buffer.sample(batch_size)
                agent.train_step(batch)
        # Update statistics
        episode_rewards.append(episode_reward)
        recent_episodes = min(episode + 1, 100)
        recent_successes = sum(1 for i in range(max(0, episode - 99), episode + 1) 
                              if i < len(success_rate_history) and success_rate_history[i])
        success_rate = recent_successes / recent_episodes if recent_episodes > 0 else 0
        success_rate_history.append(episode_success)
        distance_history.append(np.mean(episode_distances) if episode_distances else float('inf'))
        # Always print progress for every episode with buffer info
        buffer_size = len(replay_buffer)
        buffer_capacity = replay_buffer.capacity
        buffer_percent = (buffer_size / buffer_capacity) * 100
        
        # Buffer status indicator  
        if buffer_percent >= 90.0:
            buffer_status = "🧹 CLEANUP!"
        elif buffer_percent >= 85.0:
            buffer_status = "⚠️ NEAR CLEANUP"
        elif buffer_percent >= 75.0:
            buffer_status = "📈 FILLING"
        else:
            buffer_status = "📊 NORMAL"
        
        status_icon = "✅" if episode_success else "🔄"
        print(f"{status_icon} Episode {episode:3d} | "
              f"Reward: {episode_reward:6.1f} | "
              f"Success Rate: {success_rate:.2%} | "
              f"Avg Distance: {distance_history[-1]:.3f}m | "
              f"Steps: {step+1} | "
              f"Buffer: {buffer_size}/{buffer_capacity} ({buffer_percent:.1f}%) {buffer_status}")
    
    # Save trained model
    os.makedirs('checkpoints/ddpg', exist_ok=True)
    agent.save_model('checkpoints/ddpg/ddpg_4dof')
    print("\n Model saved to checkpoints/ddpg/ddpg_4dof")
    
    # Plot results
    plot_training_results(episode_rewards, success_rate_history, distance_history)
    
    # Calculate final statistics
    final_success_rate = np.mean(success_rate_history[-100:]) if len(success_rate_history) >= 100 else np.mean(success_rate_history)
    final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    
    results = {
        'episode_rewards': episode_rewards,
        'success_rate_history': success_rate_history,
        'distance_history': distance_history,
        'final_success_rate': final_success_rate,
        'final_avg_reward': final_avg_reward,
        'total_episodes': episodes
    }
    
    print("\n🎯 Training Complete!")
    print(f"Final Success Rate: {final_success_rate:.2%}")
    print(f"Final Average Reward: {final_avg_reward:.1f}")
    
    # Lưu replay buffer sau khi train
    os.makedirs('checkpoints/replay_buffers', exist_ok=True)
    replay_buffer.save('checkpoints/replay_buffers/replay_buffer.pkl')
    print("Replay buffer đã được lưu vào checkpoints/replay_buffers/replay_buffer.pkl")
    env.close()
    return results

def plot_training_results(rewards: List[float], success_rates: List[bool], distances: List[float]):
    """Plot training results."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # Success rate (rolling average)
    window = 50
    success_rate_avg = []
    for i in range(len(success_rates)):
        start_idx = max(0, i - window + 1)
        recent_successes = sum(success_rates[start_idx:i+1])
        recent_episodes = i - start_idx + 1
        success_rate_avg.append(recent_successes / recent_episodes)
    
    ax2.plot(success_rate_avg)
    ax2.set_title(f'Success Rate (Rolling Average, Window={window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate')
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    
    # Distance to target
    ax3.plot(distances)
    ax3.set_title('Average Distance to Target per Episode')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Distance (m)')
    ax3.grid(True)
    
    # Success histogram
    success_count = sum(success_rates)
    total_episodes = len(success_rates)
    ax4.bar(['Success', 'Failure'], [success_count, total_episodes - success_count], 
            color=['green', 'red'], alpha=0.7)
    ax4.set_title('Overall Success/Failure Distribution')
    ax4.set_ylabel('Episodes')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('logs/results', exist_ok=True)
    plt.savefig('logs/results/training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Training plots saved to logs/results/training_results.png")

if __name__ == "__main__":
    import argparse
    import time
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train RL agent on 4-DOF robot arm')
    parser.add_argument('--episodes', type=int, default=200, help='Number of episodes to train')
    parser.add_argument('--method', choices=['ddpg', 'mbpo'], default='mbpo', help='Training method to use')
    parser.add_argument('--render', action='store_true', help='Render environment during training')
    args = parser.parse_args()
    
    print(f"Starting training with method: {args.method}, episodes: {args.episodes}")
    
    if args.method == 'mbpo':
        env_config = {
            'max_steps': 200,
            'success_distance': 0.05,
            'dense_reward': True,
            'success_reward': 100.0
        }
        agent_config = {
            'lr_actor': 0.001,
            'lr_critic': 0.002,
            'gamma': 0.99,
            'tau': 0.005,
            'noise_std': 0.2,
            'noise_decay': 0.995,
            'hidden_dims': [256, 128]
        }
        start_time = time.time()
        trainer = MBPOTrainer(env_config, agent_config, buffer_capacity=100000, ensemble_size=1)
        results = trainer.run(episodes=args.episodes, max_steps=200, rollout_every=10)
        elapsed_time = time.time() - start_time
        
        # Additional MBPO-specific summary
        print(f"🕒 Total training time: {elapsed_time/60:.2f} minutes ({elapsed_time:.1f} seconds)")
        print(f"⚡ Time per episode: {elapsed_time/args.episodes:.1f} seconds/episode")
        print(f"🚀 MBPO Performance vs Pure DDPG: Enhanced sample efficiency with synthetic data!")
    else:  # DDPG method
        start_time = time.time()
        results = train_ddpg(episodes=args.episodes, render=args.render)
        elapsed_time = time.time() - start_time
        print(f"\n🎉 Training completed with {results['final_success_rate']:.1%} success rate!")
        print(f"Total training time: {elapsed_time/60:.2f} minutes ({elapsed_time:.1f} seconds)")
        print("\n===== POST-TRAINING EVALUATION =====")
        print(f"Total episodes: {results['total_episodes']}")
        print(f"Final success rate: {results['final_success_rate']:.2%}")
        print(f"Average reward (last 100 episodes): {results['final_avg_reward']:.2f}")
        print(f"Average reward (all episodes): {np.mean(results['episode_rewards']):.2f}")
        print(f"Average distance (last 100 episodes): {np.mean(results['distance_history'][-100:]) if len(results['distance_history'])>=100 else np.mean(results['distance_history']):.3f}")
        print(f"Average distance (all episodes): {np.mean(results['distance_history']):.3f}")
        print("====================================\n")
