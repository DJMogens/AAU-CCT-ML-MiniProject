# config.py

class Config:
    ENV_NAME = "ALE/Frogger-v5"
    N_EPISODES = 1000
    STATE_SHAPE = (84, 84, 4)
    OUTPUT_DIR = 'model_output/frogger/'
    VIDEO_FOLDER = 'saved-video-folder/'
    VIDEO_INTERVAL = 10
    REWARD_PLOT_FILE = 'rewards.png'
