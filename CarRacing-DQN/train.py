# train.py

import gymnasium as gym
import numpy as np
import os
from agent import DQNAgent
from preprocessing import preprocess_state
from utils import save_rewards, plot_rewards

def train_agent(n_episodes=1000):
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=False
    )
    action_size = env.action_space.n  # Correctly get the number of discrete actions
    output_dir = 'model_output/car_racing/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    agent = DQNAgent(state_shape=(96, 96, 1), action_size=action_size)

    # Load existing model if available
    model_path = f"{output_dir}weights_latest.keras"
    if os.path.exists(model_path):
        agent.load(model_path)
        print("Loaded existing model.")

    total_rewards = []

    for e in range(n_episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        total_reward = 0
        done = False
        time_step = 0

        while not done:
            action_index = agent.act(state)  # Get action as integer
            next_state, reward, terminated, truncated, _ = env.step(action_index)  # Pass integer directly
            done = terminated or truncated
            total_reward += reward
            next_state = preprocess_state(next_state)
            agent.remember(state, action_index, reward, next_state, done)
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
            agent.model.save(f"{output_dir}weights_latest.keras")
            save_rewards(total_rewards)
            plot_rewards(total_rewards)

    # Save the final model
    agent.model.save(f"{output_dir}weights_final.keras")
    save_rewards(total_rewards)
    plot_rewards(total_rewards)

if __name__ == "__main__":
    train_agent()