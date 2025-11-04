#!/usr/bin/env python3
"""
DDPG Algorithm for 4DOF Robot Arm in ROS Noetic
Adapted from ROS2 robotic_arm_environment DDPG implementation

Deep Deterministic Policy Gradient (DDPG) for continuous control
- Adapted for 4DOF robot instead of 6DOF
- ROS Noetic integration with visual RL environment
- Continuous action space for joint control
- Experience replay and target networks
- Ornstein-Uhlenbeck noise for exploration

Author: Adapted from David Valencia's ROS2 DDPG implementation for 4DOF ROS Noetic
"""

import sys
import os
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Critic(nn.Module):
    """
    Critic network for DDPG - estimates Q-value for state-action pairs
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(Critic, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Network layers
        self.h_linear_1 = nn.Linear(in_features=self.input_size, out_features=self.hidden_size)
        self.h_linear_2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.h_linear_3 = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, state, action):
        """
        Forward pass: concatenate state and action, then pass through network
        
        Args:
            state: Robot state (end-effector + joints + target)
            action: Joint actions (4 joint positions)
        """
        x = torch.cat([state, action], 1)  # Concatenate state and action
        x = F.relu(self.h_linear_1(x))
        x = F.relu(self.h_linear_2(x))
        x = self.h_linear_3(x)  # No activation for Q-value output
        return x


class Actor(nn.Module):
    """
    Actor network for DDPG - generates continuous actions (joint positions)
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(Actor, self).__init__()

        # Network layers
        self.h_linear_1 = nn.Linear(input_size, hidden_size)
        self.h_linear_2 = nn.Linear(hidden_size, hidden_size)
        self.h_linear_3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        """
        Forward pass: generate continuous actions from state
        
        Args:
            state: Robot state (end-effector + joints + target)
            
        Returns:
            action: Continuous actions (joint positions) in range [-1, 1]
        """
        x = F.relu(self.h_linear_1(state))
        x = F.relu(self.h_linear_2(x))
        x = torch.tanh(self.h_linear_3(x))  # tanh for bounded output [-1, 1]
        return x


class OUNoise(object):
    """
    Ornstein-Uhlenbeck Process for action exploration
    Adds correlated noise to actions for better exploration in continuous spaces
    """

    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]  # 4 for 4DOF robot
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        """Reset noise state"""
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        """Evolve the noise state"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        """
        Add noise to action for exploration
        
        Args:
            action: Base action from actor network
            t: Time step for noise decay
            
        Returns:
            Noisy action clipped to valid range
        """
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class Memory:
    """
    Experience replay buffer for DDPG
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        """Add experience tuple to buffer"""
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """Sample random batch of experiences"""
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    """
    DDPG Agent for 4DOF Robot Arm Control
    """

    def __init__(self, state_dim: int, action_dim: int, action_space, 
                 hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3,
                 gamma=0.99, tau=1e-2, max_memory_size=50000):

        self.state_dim = state_dim      # 10 for 4DOF robot (3 end-eff + 4 joints + 3 target)
        self.action_dim = action_dim    # 4 for 4DOF robot
        self.action_space = action_space

        self.gamma = gamma  # Discount factor
        self.tau = tau      # Soft update parameter
        self.t_step = 0     # Counter for learning steps

        # Networks
        self.actor = Actor(self.state_dim, hidden_size, self.action_dim)
        self.critic = Critic(self.state_dim + self.action_dim, hidden_size, 1)

        # Target networks
        self.actor_target = Actor(self.state_dim, hidden_size, self.action_dim)
        self.critic_target = Critic(self.state_dim + self.action_dim, hidden_size, 1)

        # Initialize target networks as copies of main networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Experience replay memory
        self.memory = Memory(max_memory_size)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        # Noise for exploration
        self.noise = OUNoise(action_space)

        print(f"🤖 DDPG Agent initialized:")
        print(f"   State dimension: {self.state_dim}")
        print(f"   Action dimension: {self.action_dim}")
        print(f"   Hidden size: {hidden_size}")

    def get_action(self, state, add_noise=True, noise_scale=1.0):
        """
        Get action from actor network
        
        Args:
            state: Current environment state
            add_noise: Whether to add exploration noise
            noise_scale: Scale factor for noise
            
        Returns:
            Action scaled to joint limits
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        state_var = Variable(state_tensor)

        # Get action from actor network
        action = self.actor.forward(state_var)
        action = action.detach().numpy()[0]

        # Add noise for exploration during training
        if add_noise:
            action = self.noise.get_action(action, self.t_step) * noise_scale

        # Scale action from [-1, 1] to joint limits
        action_scaled = self._scale_action_to_joint_limits(action)
        
        return action_scaled

    def _scale_action_to_joint_limits(self, action):
        """Scale action from [-1, 1] to actual joint limits"""
        # action is in range [-1, 1], scale to joint limits
        low = self.action_space.low
        high = self.action_space.high
        
        # Scale from [-1, 1] to [low, high]
        action_scaled = low + (action + 1.0) * (high - low) / 2.0
        return np.clip(action_scaled, low, high)

    def step_training(self, batch_size=128):
        """
        Perform learning step if enough experiences collected
        """
        LEARN_EVERY_STEP = 100
        self.t_step += 1

        if self.t_step % LEARN_EVERY_STEP == 0:
            if len(self.memory) > batch_size:
                self.learn_step(batch_size)

    def learn_step(self, batch_size):
        """
        Update actor and critic networks using sampled experiences
        """
        # Sample batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(np.array(dones))

        # --- Update Critic ---
        # Get next actions from target actor
        next_actions = self.actor_target(next_states)
        
        # Compute target Q-values
        target_q_values = self.critic_target(next_states, next_actions)
        target_q_values = rewards + (self.gamma * target_q_values * ~dones)

        # Current Q-values
        current_q_values = self.critic(states, actions)

        # Critic loss
        critic_loss = F.mse_loss(current_q_values, target_q_values.detach())

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor ---
        # Actor loss (maximize Q-value)
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft update target networks ---
        self._soft_update_target_networks()

    def _soft_update_target_networks(self):
        """Soft update target networks"""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        self.memory.push(state, action, reward, next_state, done)

    def save_models(self, filepath_prefix: str):
        """Save actor and critic models"""
        torch.save(self.actor.state_dict(), f"{filepath_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{filepath_prefix}_critic.pth")
        torch.save(self.actor_target.state_dict(), f"{filepath_prefix}_actor_target.pth")
        torch.save(self.critic_target.state_dict(), f"{filepath_prefix}_critic_target.pth")
        print(f"💾 Models saved with prefix: {filepath_prefix}")

    def load_models(self, filepath_prefix: str):
        """Load actor and critic models"""
        try:
            self.actor.load_state_dict(torch.load(f"{filepath_prefix}_actor.pth"))
            self.critic.load_state_dict(torch.load(f"{filepath_prefix}_critic.pth"))
            self.actor_target.load_state_dict(torch.load(f"{filepath_prefix}_actor_target.pth"))
            self.critic_target.load_state_dict(torch.load(f"{filepath_prefix}_critic_target.pth"))
            print(f"📁 Models loaded with prefix: {filepath_prefix}")
        except FileNotFoundError as e:
            print(f"❌ Model files not found: {e}")

    def reset_noise(self):
        """Reset exploration noise"""
        self.noise.reset()


# Utility functions for training visualization and logging
class TrainingMetrics:
    """Track and visualize training metrics"""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.distances = []
        
    def add_episode(self, total_reward, episode_length, success, final_distance):
        """Add episode metrics"""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        self.success_rates.append(1.0 if success else 0.0)
        self.distances.append(final_distance)
    
    def get_recent_success_rate(self, window=100):
        """Calculate success rate over recent episodes"""
        if len(self.success_rates) < window:
            return np.mean(self.success_rates) if self.success_rates else 0.0
        return np.mean(self.success_rates[-window:])
    
    def plot_training_progress(self, save_path=None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        
        # Success rate (moving average)
        if len(self.success_rates) > 10:
            window = min(50, len(self.success_rates) // 2)
            success_ma = [np.mean(self.success_rates[max(0, i-window):i+1]) 
                         for i in range(len(self.success_rates))]
            axes[0, 1].plot(success_ma)
        axes[0, 1].set_title('Success Rate (Moving Average)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate')
        
        # Episode lengths
        axes[1, 0].plot(self.episode_lengths)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        
        # Final distances
        axes[1, 1].plot(self.distances)
        axes[1, 1].set_title('Final Distance to Target')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Distance (m)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"📊 Training plot saved: {save_path}")
        
        return fig
    
    def save_metrics(self, filepath):
        """Save metrics to file"""
        metrics_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths, 
            'success_rates': self.success_rates,
            'distances': self.distances
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(metrics_data, f)
        print(f"📈 Metrics saved: {filepath}")
    
    def load_metrics(self, filepath):
        """Load metrics from file"""
        try:
            with open(filepath, 'rb') as f:
                metrics_data = pickle.load(f)
            
            self.episode_rewards = metrics_data['episode_rewards']
            self.episode_lengths = metrics_data['episode_lengths']
            self.success_rates = metrics_data['success_rates'] 
            self.distances = metrics_data['distances']
            print(f"📈 Metrics loaded: {filepath}")
        except FileNotFoundError:
            print(f"❌ Metrics file not found: {filepath}")


if __name__ == "__main__":
    """
    Test DDPG implementation with dummy environment
    """
    print("🧪 Testing DDPG Agent for 4DOF Robot...")
    
    # Mock action space for testing
    class MockActionSpace:
        def __init__(self):
            self.shape = (4,)
            self.low = np.array([-3.14159, -1.57, -2.0, -3.14159])
            self.high = np.array([3.14159, 1.57, 2.0, 3.14159])
    
    # Initialize agent
    action_space = MockActionSpace()
    agent = DDPGAgent(state_dim=10, action_dim=4, action_space=action_space)
    
    # Test with random state
    test_state = np.random.randn(10)
    action = agent.get_action(test_state)
    
    print(f"✅ Test action generated: {action}")
    print(f"   Action within limits: {np.all(action >= action_space.low) and np.all(action <= action_space.high)}")
    
    # Test experience addition
    next_state = np.random.randn(10)
    agent.add_experience(test_state, action, -1.0, next_state, False)
    print(f"✅ Experience added to buffer (size: {len(agent.memory)})")
    
    print("🎉 DDPG Agent test completed!")