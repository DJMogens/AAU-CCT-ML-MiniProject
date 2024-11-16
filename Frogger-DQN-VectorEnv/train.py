# train.py

import gymnasium as gym
import numpy as np
import os
from agent import DQNAgent
from preprocessing import stack_frames
from utils import save_rewards, plot_rewards
import ale_py
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import RecordVideo
from config import Config

# Register ALE environments
gym.register_envs(ale_py)

def make_env():
    return gym.make(Config.ENV_NAME, render_mode="rgb_array")

def train_agent(n_episodes=Config.N_EPISODES, n_envs=20):
    envs = AsyncVectorEnv([make_env for _ in range(n_envs)])
    action_size = envs.single_action_space.n
    state_shape = Config.STATE_SHAPE
    output_dir = Config.OUTPUT_DIR
    video_folder = Config.VIDEO_FOLDER
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    
    agent = DQNAgent(state_shape, action_size)
    if os.path.exists(f"{output_dir}weights_final.hdf5"):
        agent.load(f"{output_dir}weights_final.hdf5")
        print("Loaded saved model.")
    
    total_rewards = []
    stacked_frames = [[] for _ in range(n_envs)]
    
    for e in range(n_episodes):
        if e % Config.VIDEO_INTERVAL == 0:
            envs = AsyncVectorEnv([lambda: RecordVideo(make_env(), video_folder, episode_trigger=lambda x: x % Config.VIDEO_INTERVAL == 0, name_prefix=f"video-{e}") for _ in range(n_envs)])
        
        frames, _ = envs.reset()
        states = []
        for i in range(n_envs):
            state, stacked_frames[i] = stack_frames(stacked_frames[i], frames[i], is_new_episode=True)
            states.append(np.reshape(state, (1, *state_shape)))
        states = np.concatenate(states)
        total_reward = np.zeros(n_envs)
        done = [False] * n_envs
        
        while not all(done):
            agent.step += 1
            actions = [agent.act(np.expand_dims(state, axis=0)) for state in states]
            next_frames, rewards, terminations, truncations, infos = envs.step(actions)
            dones = terminations | truncations
            next_states = []
            for i in range(n_envs):
                if not done[i]:
                    next_state, stacked_frames[i] = stack_frames(stacked_frames[i], next_frames[i], is_new_episode=False)
                    next_states.append(np.reshape(next_state, (1, *state_shape)))
                else:
                    next_states.append(np.reshape(states[i], (1, *state_shape)))
            next_states = np.concatenate(next_states)
            total_reward += rewards
            
            for i in range(n_envs):
                if not done[i]:
                    agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
            states = next_states
            
            agent.replay()
            
            done = dones
        
        avg_reward = np.mean(total_reward)
        total_rewards.append(avg_reward)
        print(f"Episode: {e + 1}/{n_episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
        
        # Output success of each terminated environment
        for i in range(n_envs):
            if dones[i]:
                print(f"Environment {i} terminated with reward: {total_reward[i]}")
        
        # Save the model every 5 episodes
        if (e + 1) % 5 == 0:
            agent.save(f"{output_dir}weights_{e + 1}.hdf5")
            save_rewards(total_rewards)
            plot_rewards(total_rewards, filename=Config.REWARD_PLOT_FILE)
    
    # Save the final model
    agent.save(f"{output_dir}weights_final.hdf5")
    save_rewards(total_rewards)
    plot_rewards(total_rewards, filename=Config.REWARD_PLOT_FILE)
    envs.close()  # Ensure the environments are closed

if __name__ == "__main__":
    train_agent()
