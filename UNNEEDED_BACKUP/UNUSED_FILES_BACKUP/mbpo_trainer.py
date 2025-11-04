import numpy as np
import os
import matplotlib.pyplot as plt
import json
from datetime import datetime
from models.dynamics_model import DynamicsModel
from agents.ddpg_agent import DDPGAgent
from replay_memory.replay_buffer import ReplayBuffer
from environments.robot_4dof_env import Robot4DOFEnv

class MBPOTrainer:
    def __init__(self, env_config, agent_config, buffer_capacity=100000, ensemble_size=1, buffer_path='checkpoints/replay_buffers/mbpo_replay_buffer.pkl'):
        self.env = Robot4DOFEnv(config=env_config)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.agent = DDPGAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=agent_config
        )
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.buffer_path = buffer_path
        # Load buffer nếu file tồn tại
        if os.path.exists(self.buffer_path):
            try:
                self.replay_buffer.load(self.buffer_path)
                print(f"Loaded MBPO replay buffer from {self.buffer_path} ({len(self.replay_buffer)} samples)")
            except Exception as e:
                print(f"Could not load MBPO replay buffer: {e}")
        # Ensemble dynamics model (for simplicity, 1 model, can extend to >1)
        self.dynamics_models = [DynamicsModel(self.state_dim, self.action_dim) for _ in range(ensemble_size)]

    def train_dynamics(self, epochs=5, batch_size=64):
        transitions = self.replay_buffer.get_all_transitions()
        if len(transitions) < batch_size:
            return
        states = np.array([t['state'] for t in transitions])
        actions = np.array([t['action'] for t in transitions])
        next_states = np.array([t['next_state'] for t in transitions])
        rewards = np.array([t['reward'] for t in transitions]).reshape(-1, 1)
        for model in self.dynamics_models:
            model.train(states, actions, next_states, rewards, epochs=epochs, batch_size=batch_size)

    def rollout_model(self, num_rollouts=10, rollout_length=5):
        transitions = self.replay_buffer.get_all_transitions()
        if len(transitions) == 0:
            return
        states = np.array([t['state'] for t in transitions])
        for _ in range(num_rollouts):
            idx = np.random.randint(0, len(states))
            state = states[idx]
            for _ in range(rollout_length):
                action = self.agent.act(state, add_noise=True)
                # Use first model for simplicity (can randomize for ensemble)
                next_state_pred, reward_pred = self.dynamics_models[0].predict(state, action)
                self.replay_buffer.add(state, action, reward_pred, next_state_pred, False)
                state = next_state_pred

    def collect_real_data(self, num_steps=200, render=False):
        """Collect real environment data for one episode."""
        state, info = self.env.reset()
        episode_reward = 0.0
        episode_success = False
        episode_distances = []
        
        for step in range(num_steps):
            action = self.agent.act(state, add_noise=True)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store real transition
            self.replay_buffer.add(state, action, reward, next_state, done)
            
            # Update episode statistics
            episode_reward += reward
            episode_distances.append(info.get('distance_to_target', 0.0))
            if info.get('goal_reached', False):
                episode_success = True
            
            if render:
                self.env.render()
                
            state = next_state
            if done:
                break
        
        avg_distance = np.mean(episode_distances) if episode_distances else 1.0
        return episode_reward, episode_success, avg_distance
    
    def agent_update(self, batch_size=64):
        """Single agent update step."""
        if len(self.replay_buffer) < batch_size:
            return
        batch = self.replay_buffer.sample(batch_size)
        self.agent.train_step(batch)

    def train_policy(self, batch_size=128, updates_per_step=40):
        if len(self.replay_buffer) < batch_size:
            return
        for _ in range(updates_per_step):
            batch = self.replay_buffer.sample(batch_size)
            self.agent.train_step(batch)

    def run(self, episodes=1000, max_steps=200, rollout_every=10):
        episode_rewards = []
        success_history = []
        distance_history = []
        print("Starting MBPO Training for 4-DOF Robot Arm")
        print(f"Episodes: {episodes}, Max Steps: {max_steps}")
        print("-" * 60)
        for episode in range(episodes):
            state, info = self.env.reset()
            episode_reward = 0.0
            episode_success = False
            episode_distances = []
            for step in range(max_steps):
                action = self.agent.act(state, add_noise=True)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.replay_buffer.add(state, action, reward, next_state, done)
                episode_reward += reward
                episode_distances.append(info.get('distance_to_target', 0.0))
                if info.get('goal_reached', False):
                    episode_success = True
                state = next_state
                if done:
                    break
            # Train policy
            self.train_policy()
            # Định kỳ train dynamics model và rollout
            if episode > 0 and episode % rollout_every == 0:
                self.train_dynamics()
                self.rollout_model()
            
            # Smart cleanup is triggered automatically at 90% buffer capacity
            
            # Statistics
            episode_rewards.append(episode_reward)
            success_history.append(episode_success)
            distance_history.append(np.mean(episode_distances) if episode_distances else float('inf'))
            # Always print progress for every episode with detailed buffer info
            buffer_size = len(self.replay_buffer)
            buffer_capacity = self.replay_buffer.capacity
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
                  f"Success Rate: {np.mean(success_history[-100:]):.2%} | "
                  f"Avg Distance: {distance_history[-1]:.3f}m | "
                  f"Steps: {step+1} | "
                  f"Buffer: {buffer_size}/{buffer_capacity} ({buffer_percent:.1f}%) {buffer_status}")
        # Kết thúc huấn luyện: đóng env, lưu buffer, in summary
        os.makedirs('checkpoints/mbpo', exist_ok=True)
        os.makedirs('checkpoints/replay_buffers', exist_ok=True)
        
        # Save MBPO model
        self.agent.save_model('checkpoints/mbpo/mbpo_4dof')
        
        # Save replay buffer
        self.replay_buffer.save(self.buffer_path)
        
        # Close environment
        self.env.close()
        
        # Calculate final statistics
        final_success_rate = np.mean(success_history[-100:]) if len(success_history) >= 100 else np.mean(success_history)
        final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        
        # Print comprehensive final results summary
        print(f"\n� MBPO Training completed with {final_success_rate:.1%} success rate!")
        print(f"Model saved to checkpoints/mbpo/mbpo_4dof")
        print(f"Buffer saved to {self.buffer_path}")
        
        print(f"\n===== POST-TRAINING EVALUATION =====")
        print(f"📈 Total episodes: {episodes}")
        print(f"🎯 Final success rate: {final_success_rate:.2%}")
        print(f"🏆 Average reward (last 100 episodes): {final_avg_reward:.2f}")
        print(f"📊 Average reward (all episodes): {np.mean(episode_rewards):.2f}")
        print(f"📏 Average distance (last 100 episodes): {np.mean(distance_history[-100:]) if len(distance_history)>=100 else np.mean(distance_history):.3f}m")
        print(f"📐 Average distance (all episodes): {np.mean(distance_history):.3f}m")
        print(f"🧠 Dynamics model trained: {len([i for i in range(episodes) if i > 0 and i % rollout_every == 0])} times")
        print(f"💾 Final buffer size: {len(self.replay_buffer)}/{self.replay_buffer.capacity} ({len(self.replay_buffer)/self.replay_buffer.capacity*100:.1f}%)")
        print("=====================================\n")
        
        # Save comprehensive training results automatically
        results = {
            'episode_rewards': episode_rewards,
            'success_history': success_history,
            'distance_history': distance_history,
            'final_success_rate': final_success_rate,
            'final_avg_reward': final_avg_reward,
            'total_episodes': episodes,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # Create results directory if it doesn't exist
        results_dir = "logs/results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results data
        timestamp = results['timestamp']
        results_file = f"{results_dir}/mbpo_training_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in results.items()}
            json.dump(json_results, f, indent=2)
        
        # Generate and save plots automatically
        self._save_training_plots(results, results_dir, timestamp)
        
        print(f"📊 Training results and plots saved to {results_dir}/")
        print(f"📈 Results file: mbpo_training_results_{timestamp}.json")
        print(f"🖼️ Plots file: mbpo_training_plots_{timestamp}.png")
        
        return results
    
    def _save_training_plots(self, results, results_dir, timestamp):
        """Generate and save comprehensive training plots automatically"""
        episode_rewards = results['episode_rewards']
        success_history = results['success_history']
        distance_history = results['distance_history']
        episodes = len(episode_rewards)
        
        # Create figure with 4 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'MBPO Training Results - {timestamp}', fontsize=16, fontweight='bold')
        
        # Plot 1: Episode Rewards with Moving Average
        ax1.plot(episode_rewards, alpha=0.6, color='skyblue', label='Episode Rewards')
        # Always show moving average, adjust window size based on available episodes
        if len(episode_rewards) >= 3:
            window_size = min(20, max(3, len(episode_rewards) // 3))
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(range(window_size-1, episodes), moving_avg, color='red', linewidth=2, 
                    label=f'Moving Average (Window={window_size})')
        ax1.set_title('MBPO Episode Rewards with Moving Average')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Success Rate (Rolling Average)
        ax2.plot(success_history, color='blue', linewidth=2, alpha=0.6, label='Episode Success')
        if len(success_history) >= 3:
            success_window = min(50, max(3, len(success_history) // 3))
            success_rolling = np.convolve(success_history, np.ones(success_window)/success_window, mode='valid')
            ax2.plot(range(success_window-1, episodes), success_rolling, color='darkblue', linewidth=2,
                    label=f'Rolling Average (Window={success_window})')
        ax2.set_title('MBPO Success Rate (Rolling Average)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.set_ylim(0, 1.0)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Distance to Target with Moving Average
        ax3.plot(distance_history, alpha=0.6, color='skyblue', label='Episode Distance')
        if len(distance_history) >= 3:
            dist_window = min(20, max(3, len(distance_history) // 3))
            dist_moving_avg = np.convolve(distance_history, np.ones(dist_window)/dist_window, mode='valid')
            ax3.plot(range(dist_window-1, episodes), dist_moving_avg, color='orange', linewidth=2, 
                    label=f'Moving Average (Window={dist_window})')
        ax3.set_title('MBPO Distance to Target with Moving Average')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Distance (m)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Overall Success/Failure Distribution
        success_count = np.sum(success_history)
        failure_count = episodes - success_count
        ax4.bar(['Success', 'Failure'], [success_count, failure_count], 
               color=['green', 'red'], alpha=0.7)
        ax4.set_title('MBPO Overall Success/Failure Distribution')
        ax4.set_ylabel('Episodes')
        
        # Add text annotations with final statistics
        success_rate = results['final_success_rate']
        avg_reward = results['final_avg_reward']
        ax4.text(0.5, 0.95, f'Final Success Rate: {success_rate:.2%}\nAvg Reward: {avg_reward:.1f}', 
                transform=ax4.transAxes, ha='center', va='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_file = f"{results_dir}/mbpo_training_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plots saved successfully to {plot_file}")
