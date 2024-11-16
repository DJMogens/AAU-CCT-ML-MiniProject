This project failed due to the contenuis nature of the game, which isn't suitable for DQN. The agent was unable to learn the game and failed to make any progress. The project was abandoned in favor of other projects.

# CarRacing DQN Agent

This project implements a Deep Q-Network (DQN) agent to play the [CarRacing-v3](https://www.gymlibrary.dev/environments/box2d/car_racing/) environment from OpenAI Gymnasium. The agent learns to navigate the car using reinforcement learning techniques.

## Overview

The agent uses a Convolutional Neural Network (CNN) to approximate the Q-value function, enabling it to make decisions based on visual input from the game. The state observations are preprocessed into grayscale images to simplify the input data and reduce computational complexity.

## Installation

```bash
pip install swig

pip install "gymnasium[box2d]"

pip install -r requirements.txt
```

## Usage

To train the agent, run the following command:

```bash
python train.py
```




