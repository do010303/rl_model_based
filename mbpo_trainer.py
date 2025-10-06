import numpy as np
import os
from models.dynamics_model import DynamicsModel
from agents.ddpg_agent import DDPGAgent
from replay_memory.replay_buffer import ReplayBuffer
from environments.robot_4dof_env import Robot4DOFEnv

class MBPOTrainer:
    def __init__(self, env_config, agent_config, buffer_capacity=100000, ensemble_size=1, buffer_path='checkpoints/mbpo_replay_buffer.pkl'):
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
            # Statistics
            episode_rewards.append(episode_reward)
            success_history.append(episode_success)
            distance_history.append(np.mean(episode_distances) if episode_distances else float('inf'))
            # Always print progress for every episode
            status_icon = "✅" if episode_success else "🔄"
            print(f"{status_icon} Episode {episode:3d} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Success Rate: {np.mean(success_history[-100:]):.2%} | "
                  f"Avg Distance: {distance_history[-1]:.3f}m | "
                  f"Steps: {step+1}")
    # Kết thúc huấn luyện: đóng env, lưu buffer, in summary
    os.makedirs('checkpoints', exist_ok=True)
