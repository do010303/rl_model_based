"""
DDPG Agent adapted for Gazebo 4DOF Robot Arm
Based on the backup RL implementation, optimized for fast Gazebo training
"""
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replay_memory.ReplayBuffer import ReplayBuffer
from utils.networks import ActorNetwork, CriticNetwork


## DDPG Algorithm Parameters (optimized for Gazebo)

# Learning rates
ALPHA = 1e-4  # actor learning rate
BETA = 1e-3   # critic learning rate

# RL parameters
GAMMA = 0.98      # discount factor
TAU = 0.001       # target networks soft update factor 
NOISE_FACTOR = 0.15  # exploration noise (reduced for faster convergence)

# Replay buffer
MAX_BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128


class DDPGAgentGazebo:
    """
    DDPG Agent optimized for 4DOF Gazebo robot with fast action execution
    
    Key differences from original:
    - Adapted for 14D state space (4 joint angles + 4 velocities + 3 ee_pos + 3 target_pos)
    - 4 continuous actions (joint positions)
    - Optimized for 3.5s action execution time
    - Designed for short episodes (5 actions = ~17.5s per episode)
    """
    
    def __init__(self, state_dim=14, n_actions=4, max_action=1.0, min_action=-1.0):
        """
        Initialize DDPG agent for Gazebo environment
        
        Args:
            state_dim: Dimension of state space (14 for our robot)
            n_actions: Number of actions (4 joints)
            max_action: Maximum action value (1.0 for normalized)
            min_action: Minimum action value (-1.0 for normalized)
        """
        self.gamma = GAMMA
        self.tau = TAU
        self.batch_size = BATCH_SIZE
        self.noise_factor = NOISE_FACTOR
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.max_action = max_action
        self.min_action = min_action
        
        # Replay buffer
        self.memory = ReplayBuffer(MAX_BUFFER_SIZE, state_dim, n_actions)
        
        # Initialize networks
        self._initialize_networks(n_actions)
        self.update_parameters(tau=1)  # Copy weights initially
        
        print(f"✅ DDPG Agent initialized:")
        print(f"   State dim: {state_dim}, Actions: {n_actions}")
        print(f"   Gamma: {self.gamma}, Tau: {self.tau}, Noise: {self.noise_factor}")
        print(f"   Buffer size: {MAX_BUFFER_SIZE}, Batch size: {self.batch_size}")
    
    def choose_action(self, state, evaluate=False):
        """
        Choose action based on actor network
        
        Args:
            state: Current state (14D vector)
            evaluate: If True, use deterministic policy (no noise)
            
        Returns:
            action: 4D action vector (normalized joint positions)
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actor(state)
        
        # Add exploration noise during training
        if not evaluate:
            noise = tf.random.normal(
                shape=[self.n_actions], 
                mean=0.0, 
                stddev=self.noise_factor
            )
            actions += noise
        
        # Clip actions to valid range
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        
        return actions[0].numpy()
    
    def remember(self, state, action, reward, new_state, done):
        """Store experience in replay buffer"""
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def learn(self):
        """
        Main DDPG learning process
        
        Returns:
            (actor_loss, critic_loss) or (None, None) if not enough samples
        """
        # Need enough samples to start learning
        if self.memory.counter < self.batch_size:
            return None, None
        
        # Sample batch from replay buffer
        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
        
        # Update Critic Network
        with tf.GradientTape() as tape:
            # Compute target Q-values
            target_actions = self.target_actor(new_states)
            target_q = tf.squeeze(self.target_critic(new_states, target_actions), 1)
            
            # Compute current Q-values
            current_q = tf.squeeze(self.critic(states, actions), 1)
            
            # Bellman equation
            target = rewards + self.gamma * target_q * (1 - dones)
            
            # MSE loss
            critic_loss = tf.keras.losses.MSE(target, current_q)
            critic_loss_value = critic_loss.numpy()
        
        # Apply gradients to critic
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(
            zip(critic_gradients, self.critic.trainable_variables)
        )
        
        # Update Actor Network
        with tf.GradientTape() as tape:
            # Compute actor loss (negative Q-value)
            new_actions = self.actor(states)
            actor_loss = -self.critic(states, new_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)
            actor_loss_value = actor_loss.numpy()
        
        # Apply gradients to actor
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(actor_gradients, self.actor.trainable_variables)
        )
        
        # Soft update target networks
        self.update_parameters()
        
        return actor_loss_value, critic_loss_value
    
    def update_parameters(self, tau=None):
        """
        Soft update target networks
        
        θ_target = τ * θ_local + (1 - τ) * θ_target
        """
        if tau is None:
            tau = self.tau
        
        # Update target actor
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(tau * weight + (1 - tau) * targets[i])
        self.target_actor.set_weights(weights)
        
        # Update target critic
        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(tau * weight + (1 - tau) * targets[i])
        self.target_critic.set_weights(weights)
    
    def save_models(self, episode=None):
        """Save actor and critic networks"""
        suffix = f"_ep{episode}" if episode is not None else ""
        
        print(f"💾 Saving models{suffix}...")
        self.actor.save_weights(self.actor.checkpoints_file.replace('.h5', f'{suffix}.h5'))
        self.critic.save_weights(self.critic.checkpoints_file.replace('.h5', f'{suffix}.h5'))
        self.target_actor.save_weights(self.target_actor.checkpoints_file.replace('.h5', f'{suffix}.h5'))
        self.target_critic.save_weights(self.target_critic.checkpoints_file.replace('.h5', f'{suffix}.h5'))
        print("✅ Models saved!")
    
    def load_models(self, episode=None):
        """Load actor and critic networks"""
        suffix = f"_ep{episode}" if episode is not None else ""
        
        print(f"📂 Loading models{suffix}...")
        try:
            self.actor.load_weights(self.actor.checkpoints_file.replace('.h5', f'{suffix}.h5'))
            self.critic.load_weights(self.critic.checkpoints_file.replace('.h5', f'{suffix}.h5'))
            self.target_actor.load_weights(self.target_actor.checkpoints_file.replace('.h5', f'{suffix}.h5'))
            self.target_critic.load_weights(self.target_critic.checkpoints_file.replace('.h5', f'{suffix}.h5'))
            print("✅ Models loaded!")
            return True
        except Exception as e:
            print(f"⚠️ Could not load models: {e}")
            return False
    
    def _initialize_networks(self, n_actions):
        """Initialize actor, critic and their target networks"""
        model_name = "ddpg_gazebo"
        
        # Main networks
        self.actor = ActorNetwork(n_actions, name="actor", model=model_name)
        self.critic = CriticNetwork(name="critic", model=model_name)
        
        # Target networks
        self.target_actor = ActorNetwork(n_actions, name="target_actor", model=model_name)
        self.target_critic = CriticNetwork(name="target_critic", model=model_name)
        
        # Compile with optimizers
        self.actor.compile(keras.optimizers.Adam(learning_rate=ALPHA))
        self.critic.compile(keras.optimizers.Adam(learning_rate=BETA))
        self.target_actor.compile(keras.optimizers.Adam(learning_rate=ALPHA))
        self.target_critic.compile(keras.optimizers.Adam(learning_rate=BETA))
