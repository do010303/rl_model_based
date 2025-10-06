import numpy as np
import tensorflow as tf
from tensorflow import keras

class DynamicsModel:
    def __init__(self, state_dim, action_dim):
        inputs = keras.layers.Input(shape=(state_dim + action_dim,))
        x = keras.layers.Dense(256, activation='relu')(inputs)
        x = keras.layers.Dense(256, activation='relu')(x)
        next_state = keras.layers.Dense(state_dim)(x)
        reward = keras.layers.Dense(1)(x)
        self.model = keras.Model(inputs=inputs, outputs=[next_state, reward])
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, states, actions, next_states, rewards, epochs=10, batch_size=64):
        inputs = np.concatenate([states, actions], axis=1)
        targets = [next_states, rewards]
        self.model.fit(inputs, targets, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, state, action):
        input_ = np.concatenate([state, action], axis=-1)[None]
        next_state, reward = self.model.predict(input_, verbose=0)
        return next_state[0], reward[0, 0]
