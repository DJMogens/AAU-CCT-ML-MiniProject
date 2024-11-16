# Frogger DQN with Gymnasium

This project implements a Deep Q-Network (DQN) agent to solve the Frogger game using the Gymnasium API.

## Requirements

- Python 3.x
- Gymnasium\[atari\]
- TensorFlow
- NumPy
- OpenCV
- Matplotlib

## Installation

## Speeding Up Training

This project uses vectorized environments to speed up training. The training script has been modified to use `AsyncVectorEnv` from Gymnasium to run multiple instances of the Frogger environment in parallel.
