# train.py

import gymnasium as gym
import numpy as np
import os
from agent import DQNAgent
from preprocessing import preprocess_state
from utils import save_rewards, plot_rewards

# Create a custom action wrapper
class DiscreteActionWrapper(gym.ActionWrapper):
    ACTIONS = [
        np.array([0.0, 0.0, 0.0]),    # Do nothing
        np.array([-1.0, 0.0, 0.0]),   # Steer left
        np.array([1.0, 0.0, 0.0]),    # Steer right
        np.array([0.0, 1.0, 0.0]),    # Gas
        np.array([0.0, 0.0, 0.8])     # Brake
    ]
    
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        
    def action(self, action):
        return self.ACTIONS[action]

def train_agent(n_episodes=1000):
    env = gym.make("CarRacing-v3", render_mode="human")
    env = DiscreteActionWrapper(env)
    state_shape = (96, 96, 1)  # Grayscale image
    action_size = env.action_space.n
    output_dir = 'model_output/car_racing/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    agent = DQNAgent(state_shape, action_size)
    total_rewards = []

    for e in range(n_episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        total_reward = 0
        done = False
        time_step = 0

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            next_state = preprocess_state(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            time_step += 1

            if time_step % 5 == 0:
                agent.replay()

            if done:
                print(f"Episode: {e+1}/{n_episodes}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
                break

        total_rewards.append(total_reward)

        # Save the model every 50 episodes
        if (e + 1) % 50 == 0:
            agent.model.save(f"{output_dir}weights_{e+1}.hdf5")
            save_rewards(total_rewards)
            plot_rewards(total_rewards)

    # Save the final model
    agent.model.save(f"{output_dir}weights_final.hdf5")
    save_rewards(total_rewards)
    plot_rewards(total_rewards)

if __name__ == "__main__":
    train_agent()
