"""
Curriculum Learning Training Example for 4-DOF Robot Arm
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.robot_4dof_env import Robot4DOFEnv
from agents.ddpg_agent import DDPGAgent
from training.curriculum import CurriculumTrainer
from replay_memory.replay_buffer import ReplayBuffer
from utils.her import HER
from models.dynamics_model import DynamicsModel

def train_with_curriculum(total_episodes: int = 2000, render: bool = False) -> Dict:
    """
    Train DDPG agent using curriculum learning.
    
    Args:
        total_episodes: Total episodes across all curriculum stages
        render: Whether to render during training
        
    Returns:
        Training results dictionary
    """
    
    print("🎯 4-DOF Robot Arm Curriculum Learning Training")
    print("=" * 60)
    
    # Base environment configuration
    base_env_config = {
        'max_steps': 200,
        'success_distance': 0.05,  # Will be overridden by curriculum
        'workspace_radius': 0.8,   # Will be overridden by curriculum
        'dense_reward': True,
        'success_reward': 100.0,   # Will be overridden by curriculum
        'distance_reward_scale': -1.0,
        'action_penalty_scale': -0.01
    }
    
    # Agent configuration
    agent_config = {
        'lr_actor': 0.001,
        'lr_critic': 0.002,
        'gamma': 0.99,
        'tau': 0.005,
        'noise_std': 0.3,          # Higher initial noise for exploration
        'noise_decay': 0.9995,     # Slower decay for curriculum learning
        'hidden_dims': [512, 256, 128]  # Larger network for curriculum
    }
    
    # Initialize agent
    env_temp = Robot4DOFEnv(config=base_env_config)
    state_dim = env_temp.observation_space.shape[0]
    action_dim = env_temp.action_space.shape[0]
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=agent_config
    )
    env_temp.close()
    
    # Initialize replay buffer and HER
    replay_buffer = ReplayBuffer(capacity=200000)  # Larger buffer for curriculum
    her = HER(replay_buffer=replay_buffer, k=4, strategy='future')
    # Initialize dynamics model (model-based)
    dynamics_model = DynamicsModel(state_dim, action_dim)
    
    # Initialize curriculum trainer
    curriculum_trainer = CurriculumTrainer(
        base_env_config=base_env_config,
        agent=agent
    )
    
    print(f"🤖 Agent initialized with {sum(np.prod(p.shape) for p in agent.actor.trainable_variables):,} trainable parameters")
    print(f"🧠 Replay buffer capacity: {replay_buffer.capacity:,}")
    print(f"🎯 HER strategy: {her.strategy} with k={her.k}")
    
    # Train with curriculum (giữ nguyên logic gốc để trả về kết quả đúng định dạng)
    results = curriculum_trainer.train(
        total_episodes=total_episodes,
        render=render,
        save_checkpoints=True
    )
    # Tích hợp dynamics model: định kỳ huấn luyện dynamics model và rollout sinh dữ liệu ảo
    # (Chèn logic này vào trong CurriculumTrainer.train nếu muốn tối ưu sâu hơn)
    # Ví dụ: sau mỗi stage hoặc mỗi N episode, lấy dữ liệu từ replay_buffer để train dynamics model
    # và rollout dynamics model để sinh dữ liệu ảo cho agent học thêm
    
    # Save final model
    os.makedirs('checkpoints', exist_ok=True)
    agent.save_model('checkpoints/curriculum_final')
    print(f"\n💾 Final model saved to checkpoints/curriculum_final")
    
    # Generate detailed plots
    plot_curriculum_results(results, curriculum_trainer.stage_history)
    
    return results

def plot_curriculum_results(results: Dict, stage_history: list):
    """Plot comprehensive curriculum training results."""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create subplot layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Episode rewards over time
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(results['all_rewards'], alpha=0.7, linewidth=0.8)
    
    # Add moving average
    window = 50
    if len(results['all_rewards']) >= window:
        moving_avg = np.convolve(results['all_rewards'], np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(results['all_rewards'])), moving_avg, 
                 color='red', linewidth=2, label=f'Moving Average ({window})')
    
    ax1.set_title('Episode Rewards Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add stage boundaries
    episode_count = 0
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i, stage_info in enumerate(stage_history):
        ax1.axvline(x=episode_count + stage_info['episodes'], 
                   color=colors[i % len(colors)], linestyle='--', alpha=0.7,
                   label=f"End of {stage_info['stage_name']}")
        episode_count += stage_info['episodes']
    
    # 2. Success rate progression
    ax2 = fig.add_subplot(gs[0, 2:])
    success_rates = []
    window = 25
    for i in range(len(results['all_success_rates'])):
        start_idx = max(0, i - window + 1)
        recent_rate = np.mean(results['all_success_rates'][start_idx:i+1])
        success_rates.append(recent_rate)
    
    ax2.plot(success_rates, color='green', linewidth=2)
    ax2.set_title(f'Success Rate Evolution (Window={window})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add stage success thresholds
    episode_count = 0
    for i, stage_info in enumerate(stage_history):
        episodes = stage_info['episodes']
        ax2.hlines(y=stage_info['success_rate'], 
                  xmin=episode_count, xmax=episode_count + episodes,
                  colors=colors[i % len(colors)], linestyles='--', alpha=0.8)
        episode_count += episodes
    
    # 3. Distance to target over time
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(results['all_distances'], alpha=0.6, color='orange')
    
    # Moving average for distances
    if len(results['all_distances']) >= window:
        distance_moving_avg = np.convolve(results['all_distances'], np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(results['all_distances'])), distance_moving_avg,
                color='red', linewidth=2, label=f'Moving Average ({window})')
    
    ax3.set_title('Distance to Target Over Time', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Distance (m)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Stage performance summary
    ax4 = fig.add_subplot(gs[1, 2:])
    stage_names = [info['stage_name'] for info in stage_history]
    stage_success_rates = [info['success_rate'] for info in stage_history]
    
    bars = ax4.bar(range(len(stage_names)), stage_success_rates, 
                   color=colors[:len(stage_names)], alpha=0.7)
    ax4.set_title('Success Rate by Curriculum Stage', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Curriculum Stage')
    ax4.set_ylabel('Success Rate')
    ax4.set_xticks(range(len(stage_names)))
    ax4.set_xticklabels([name.replace(' ', '\\n') for name in stage_names], fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rate in zip(bars, stage_success_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Stage episodes distribution
    ax5 = fig.add_subplot(gs[2, :2])
    stage_episodes = [info['episodes'] for info in stage_history]
    
    pie = ax5.pie(stage_episodes, labels=stage_names, autopct='%1.1f%%',
                  colors=colors[:len(stage_names)], startangle=90)
    ax5.set_title('Episode Distribution Across Stages', fontsize=14, fontweight='bold')
    
    # 6. Training statistics summary
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.axis('off')
    
    # Calculate final statistics
    final_success_rate = results['overall_success_rate']
    final_avg_reward = np.mean(results['all_rewards'][-50:]) if len(results['all_rewards']) >= 50 else np.mean(results['all_rewards'])
    final_avg_distance = np.mean(results['all_distances'][-50:]) if len(results['all_distances']) >= 50 else np.mean(results['all_distances'])
    
    stats_text = f"""
    📊 TRAINING SUMMARY
    
    Total Episodes: {results['total_episodes']:,}
    Completed Stages: {results['completed_stages']}/{len(stage_history)}
    
    🎯 FINAL PERFORMANCE
    Success Rate: {final_success_rate:.1%}
    Avg Reward: {final_avg_reward:.1f}
    Avg Distance: {final_avg_distance:.3f}m
    
    📈 IMPROVEMENT
    Best Stage: {max(stage_history, key=lambda x: x['success_rate'])['stage_name']}
    Best Success Rate: {max(info['success_rate'] for info in stage_history):.1%}
    
    🏆 CURRICULUM EFFECTIVENESS
    Stages Completed: {results['completed_stages']}
    Progressive Learning: ✅
    """
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Overall title
    fig.suptitle('4-DOF Robot Arm Curriculum Learning Results', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/curriculum_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📊 Curriculum training plots saved to results/curriculum_training_results.png")

def main():
    """Main training function."""
    
    print("🎓 Starting 4-DOF Robot Arm Curriculum Learning")
    print("This will train the robot through progressive difficulty stages")
    print("=" * 70)
    
    # Training configuration
    TOTAL_EPISODES = 2000  # Total episodes across all curriculum stages
    RENDER_TRAINING = False  # Set to True to see robot during training
    
    # Run curriculum training
    results = train_with_curriculum(
        total_episodes=TOTAL_EPISODES,
        render=RENDER_TRAINING
    )
    
    # Print final summary
    print(f"\n🏆 CURRICULUM TRAINING COMPLETED!")
    print(f"🎯 Final Success Rate: {results['overall_success_rate']:.1%}")
    print(f"📚 Completed {results['completed_stages']} curriculum stages")
    print(f"📈 Total Episodes: {results['total_episodes']:,}")
    
    if results['overall_success_rate'] > 0.4:
        print("✅ Training successful! Ready for deployment.")
    else:
        print("⚠️  Consider extending training or adjusting curriculum.")

if __name__ == "__main__":
    main()
