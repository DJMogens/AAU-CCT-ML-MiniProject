# This file defines the DQNAgent class responsible for the agent's behavior and learning.

import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        
        # Hyperparameters
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.00025
        self.batch_size = 64
        self.train_start = 1000  # Start training after storing some samples
        
        # Build the neural network
        self.model = self._build_model()
    
    def _build_model(self):
        """Builds a Convolutional Neural Network to approximate Q-value function."""
        model = models.Sequential()
        model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_shape))
        model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Stores experiences in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Chooses an action based on ε-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Trains the neural network using experiences sampled from memory."""
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        state_batch = np.zeros((len(minibatch), *self.state_shape))
        next_state_batch = np.zeros((len(minibatch), *self.state_shape))
        action_batch, reward_batch, done_batch = [], [], []
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            state_batch[i] = state
            next_state_batch[i] = next_state
            action_batch.append(action)
            reward_batch.append(reward)
            done_batch.append(done)
        
        target = self.model.predict(state_batch, verbose=0)
        target_next = self.model.predict(next_state_batch, verbose=0)
        
        for i in range(len(minibatch)):
            if done_batch[i]:
                target[i][action_batch[i]] = reward_batch[i]
            else:
                target[i][action_batch[i]] = reward_batch[i] + self.gamma * np.amax(target_next[i])
        
        self.model.fit(state_batch, target, batch_size=self.batch_size, epochs=1, verbose=0)
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
