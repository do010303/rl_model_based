import numpy as np
import tensorflow as tf
from tensorflow import keras

class DynamicsModel:
    def __init__(self, state_dim, action_dim):
        inputs = keras.layers.Input(shape=(state_dim + action_dim,))
        x = keras.layers.Dense(256, activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        next_state = keras.layers.Dense(state_dim)(x)
        reward = keras.layers.Dense(1)(x)
        self.model = keras.Model(inputs=inputs, outputs=[next_state, reward])
        
        # Use optimizer with gradient clipping
        optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='mse')

    def train(self, states, actions, next_states, rewards, epochs=10, batch_size=64):
        # Filter out NaN values before training
        inputs = np.concatenate([states, actions], axis=1)
        targets = [next_states, rewards]
        
        # Check for NaN values
        valid_mask = (
            ~np.any(np.isnan(inputs), axis=1) & 
            ~np.any(np.isnan(next_states), axis=1) & 
            ~np.isnan(rewards.flatten())
        )
        
        if not np.any(valid_mask):
            print("⚠️ All training data contains NaN, skipping dynamics model training")
            return
        
        if np.sum(valid_mask) < len(inputs) * 0.5:
            print(f"⚠️ {np.sum(~valid_mask)} samples contain NaN, filtering them out")
        
        # Filter data
        clean_inputs = inputs[valid_mask]
        clean_next_states = next_states[valid_mask]
        clean_rewards = rewards[valid_mask]
        clean_targets = [clean_next_states, clean_rewards]
        
        if len(clean_inputs) > batch_size:
            self.model.fit(clean_inputs, clean_targets, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, state, action):
        # Check input validity
        if np.any(np.isnan(state)) or np.any(np.isnan(action)):
            print("⚠️ NaN detected in dynamics model input")
            return np.zeros_like(state), 0.0
            
        input_ = np.concatenate([state, action], axis=-1)[None]
        
        # Check for finite input values
        if not np.all(np.isfinite(input_)):
            print("⚠️ Invalid input to dynamics model")
            return np.zeros_like(state), 0.0
        
        try:
            next_state, reward = self.model.predict(input_, verbose=0)
            
            # Check for NaN in predictions
            if np.any(np.isnan(next_state)) or np.isnan(reward):
                print("⚠️ NaN prediction from dynamics model")
                return np.zeros_like(state), 0.0
                
            return next_state[0], reward[0, 0]
        except Exception as e:
            print(f"⚠️ Error in dynamics model prediction: {e}")
            return np.zeros_like(state), 0.0
