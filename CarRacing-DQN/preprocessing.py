### This file contains functions to preprocess the environment's observations.

import cv2
import numpy as np

def preprocess_state(state):
    """Converts RGB images to grayscale and normalizes."""
    gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (96, 96))
    normalized = resized / 255.0
    return normalized.reshape(1, 96, 96, 1)
