"""
Deep Deterministic Policy Gradient (DDPG) Agent
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, Optional
import os

from .base_agent import BaseAgent

class DDPGAgent(BaseAgent):
    """
    DDPG Agent implementation with configurable architecture.
    
    Features:
    - Actor-Critic architecture
    - Target networks with soft updates
    - Ornstein-Uhlenbeck noise for exploration
    - Experience replay integration
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(state_dim, action_dim, config)
        
        # Hyperparameters
        self.lr_actor = self.config.get('lr_actor', 0.001)
        self.lr_critic = self.config.get('lr_critic', 0.002)
        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.005)
        self.noise_std = self.config.get('noise_std', 0.2)
        self.noise_decay = self.config.get('noise_decay', 0.995)
        
        # Network architecture
        self.hidden_dims = self.config.get('hidden_dims', [256, 128])
        
        # Build networks
        self._build_networks()
        
        # Initialize target networks
        self._init_target_networks()
        
        # Noise process
        self.noise = OrnsteinUhlenbeckNoise(
            size=action_dim, 
            std=self.noise_std
        )
        
        # Training metrics
        self.actor_loss_history = []
        self.critic_loss_history = []
        
    def _build_networks(self):
        """Build actor and critic networks."""
        
        # Actor Network
        self.actor = self._create_actor()
        self.actor_target = self._create_actor()
        
        # Critic Network
        self.critic = self._create_critic()
        self.critic_target = self._create_critic()
        
        # Optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=self.lr_actor)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=self.lr_critic)
    
    def _create_actor(self):
        """Create actor network."""
        inputs = keras.layers.Input(shape=(self.state_dim,))
        x = inputs
        
        for dim in self.hidden_dims:
            x = keras.layers.Dense(dim, activation='relu')(x)
            x = keras.layers.BatchNormalization()(x)
        
        outputs = keras.layers.Dense(
            self.action_dim, 
            activation='tanh',
            kernel_initializer='random_uniform'
        )(x)
        
        # Scale outputs to action bounds
        outputs = outputs * np.pi  # Assuming joint limits [-π, π]
        
        model = keras.Model(inputs, outputs)
        return model
    
    def _create_critic(self):
        """Create critic network."""
        state_input = keras.layers.Input(shape=(self.state_dim,))
        action_input = keras.layers.Input(shape=(self.action_dim,))
        
        # State processing
        state_h = keras.layers.Dense(self.hidden_dims[0], activation='relu')(state_input)
        state_h = keras.layers.BatchNormalization()(state_h)
        
        # Action processing
        action_h = keras.layers.Dense(self.hidden_dims[0], activation='relu')(action_input)
        
        # Concatenate state and action
        concat = keras.layers.Concatenate()([state_h, action_h])
        
        # Additional hidden layers
        x = concat
        for dim in self.hidden_dims[1:]:
            x = keras.layers.Dense(dim, activation='relu')(x)
            x = keras.layers.BatchNormalization()(x)
        
        outputs = keras.layers.Dense(1)(x)
        
        model = keras.Model([state_input, action_input], outputs)
        return model
    
    def _init_target_networks(self):
        """Initialize target networks with main network weights."""
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())
    
    def act(self, state: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """
        Select action given state.
        
        Args:
            state: Current state
            add_noise: Whether to add exploration noise
            
        Returns:
            Selected action
        """
        state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        action = self.actor(state)[0].numpy()
        
        if add_noise:
            noise = self.noise.sample()
            action = np.clip(action + noise, -np.pi, np.pi)
        
        return action
    
    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Batch of experiences
            
        Returns:
            Training metrics
        """
        states = tf.convert_to_tensor(batch['states'], dtype=tf.float32)
        actions = tf.convert_to_tensor(batch['actions'], dtype=tf.float32)
        rewards = tf.convert_to_tensor(batch['rewards'], dtype=tf.float32)
        next_states = tf.convert_to_tensor(batch['next_states'], dtype=tf.float32)
        dones = tf.convert_to_tensor(batch['dones'], dtype=tf.float32)
        
        # Train critic
        critic_loss = self._train_critic(states, actions, rewards, next_states, dones)
        
        # Train actor
        actor_loss = self._train_actor(states)
        
        # Update target networks
        self._update_target_networks()
        
        # Update noise
        self.noise.std *= self.noise_decay
        
        # Update training step
        self.training_step += 1
        
        # Store losses
        self.actor_loss_history.append(float(actor_loss))
        self.critic_loss_history.append(float(critic_loss))
        
        return {
            'actor_loss': float(actor_loss),
            'critic_loss': float(critic_loss),
            'noise_std': self.noise.std
        }
    
    @tf.function
    def _train_critic(self, states, actions, rewards, next_states, dones):
        """Train critic network."""
        # Target Q values
        target_actions = self.actor_target(next_states)
        target_q = self.critic_target([next_states, target_actions])
        y = rewards + self.gamma * target_q * (1 - dones)
        
        with tf.GradientTape() as tape:
            q_values = self.critic([states, actions])
            critic_loss = tf.reduce_mean(tf.square(y - q_values))
        
        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))
        
        return critic_loss
    
    @tf.function
    def _train_actor(self, states):
        """Train actor network."""
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            q_values = self.critic([states, actions])
            actor_loss = -tf.reduce_mean(q_values)
        
        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
        
        return actor_loss
    
    def _update_target_networks(self):
        """Soft update target networks."""
        # Update actor target
        actor_weights = self.actor.get_weights()
        actor_target_weights = self.actor_target.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = (
                self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]
            )
        self.actor_target.set_weights(actor_target_weights)
        
        # Update critic target
        critic_weights = self.critic.get_weights()
        critic_target_weights = self.critic_target.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = (
                self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
            )
        self.critic_target.set_weights(critic_target_weights)
    
    def save_model(self, filepath: str) -> None:
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.actor.save_weights(f"{filepath}_actor.h5")
        self.critic.save_weights(f"{filepath}_critic.h5")
        
        # Save configuration
        config = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'config': self.config,
            'training_step': self.training_step
        }
        
        import json
        with open(f"{filepath}_config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_model(self, filepath: str) -> None:
        """Load model from disk."""
        self.actor.load_weights(f"{filepath}_actor.h5")
        self.critic.load_weights(f"{filepath}_critic.h5")
        
        # Load configuration
        import json
        try:
            with open(f"{filepath}_config.json", 'r') as f:
                saved_config = json.load(f)
                self.training_step = saved_config.get('training_step', 0)
        except FileNotFoundError:
            print(f"Warning: Config file {filepath}_config.json not found")


class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck noise process for continuous action exploration.
    """
    
    def __init__(self, size: int, std: float = 0.2, theta: float = 0.15, dt: float = 1e-2):
        self.size = size
        self.std = std
        self.theta = theta
        self.dt = dt
        self.reset()
    
    def reset(self):
        """Reset noise process."""
        self.state = np.zeros(self.size)
    
    def sample(self) -> np.ndarray:
        """Sample noise."""
        dx = self.theta * (-self.state) * self.dt + self.std * np.sqrt(self.dt) * np.random.randn(self.size)
        self.state += dx
        return self.state
