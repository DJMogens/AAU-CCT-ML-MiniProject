# utils.py

import matplotlib.pyplot as plt
import csv

def save_rewards(rewards, filename='training_rewards.csv'):
    """Saves the rewards to a CSV file."""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total Reward"])
        for i, reward in enumerate(rewards):
            writer.writerow([i + 1, reward])

def plot_rewards(rewards, filename='rewards_plot.png'):
    """Plots the rewards over episodes."""
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Agent Performance Over Time')
    plt.savefig(filename)
    plt.close()
