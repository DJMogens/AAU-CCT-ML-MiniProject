# agent.py

import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape  # (84, 84, 4)
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        
        # Hyperparameters
        self.gamma = 0.99    # Discount factor
        self.epsilon = 0.5   # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999995  # Decay per step
        self.learning_rate = 0.00025
        self.batch_size = 32
        self.train_start = 50000  # Start training after storing some samples
        self.update_target_freq = 10000  # Steps to update target network
        
        # Build the main and target neural networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Internal step counter
        self.step = 0
    
    def _build_model(self):
        """Builds a Convolutional Neural Network to approximate the Q-value function."""
        model = models.Sequential()
        model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_shape))
        model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='huber_loss', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Updates the target network with weights from the main network."""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Stores experiences in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Chooses an action based on an ε-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Trains the neural network using experiences sampled from memory."""
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        state_batch = np.array([sample[0] for sample in minibatch]).reshape(self.batch_size, *self.state_shape)
        action_batch = [sample[1] for sample in minibatch]
        reward_batch = [sample[2] for sample in minibatch]
        next_state_batch = np.array([sample[3] for sample in minibatch]).reshape(self.batch_size, *self.state_shape)
        done_batch = [sample[4] for sample in minibatch]
        
        target = self.model.predict(state_batch, verbose=0)
        target_next = self.target_model.predict(next_state_batch, verbose=0)
        
        for i in range(self.batch_size):
            if done_batch[i]:
                target[i][action_batch[i]] = reward_batch[i]
            else:
                target[i][action_batch[i]] = reward_batch[i] + self.gamma * np.amax(target_next[i])
        
        self.model.fit(state_batch, target, batch_size=self.batch_size, epochs=1, verbose=0)
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        if self.step % self.update_target_freq == 0:
            self.update_target_model()
    
    def load(self, name):
        """Loads a saved model."""
        self.model.load_weights(name)
    
    def save(self, name):
        """Saves the current model."""
        self.model.save_weights(name)
