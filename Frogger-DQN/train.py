# train.py

import gymnasium as gym
import numpy as np
import os
from agent import DQNAgent
from preprocessing import stack_frames
from utils import save_rewards, plot_rewards
import ale_py



# Register ALE environments
gym.register_envs(ale_py)

def train_agent(n_episodes=1000):
    env = gym.make("ALE/Frogger-v5", render_mode="human")  # Add render_mode
    action_size = env.action_space.n
    state_shape = (84, 84, 4)  # 4 frames stacked
    output_dir = 'model_output/frogger/'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    agent = DQNAgent(state_shape, action_size)
    total_rewards = []
    stacked_frames = []
    
    for e in range(n_episodes):
        frame, _ = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, frame, is_new_episode=True)
        state = np.reshape(state, (1, *state_shape))
        total_reward = 0
        done = False
        
        while not done:
            agent.step += 1
            action = agent.act(state)
            next_frame, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state, stacked_frames = stack_frames(stacked_frames, next_frame, is_new_episode=False)
            next_state = np.reshape(next_state, (1, *state_shape))
            total_reward += reward
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            agent.replay()
            
            if done:
                print(f"Episode: {e + 1}/{n_episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.4f}")
                break
        
        total_rewards.append(total_reward)
        
        # Save the model every 50 episodes
        if (e + 1) % 50 == 0:
            agent.save(f"{output_dir}weights_{e + 1}.hdf5")
            save_rewards(total_rewards)
            plot_rewards(total_rewards)
    
    # Save the final model
    agent.save(f"{output_dir}weights_final.hdf5")
    save_rewards(total_rewards)
    plot_rewards(total_rewards)
    env.close()  # Ensure the environment is closed

if __name__ == "__main__":
    train_agent()
