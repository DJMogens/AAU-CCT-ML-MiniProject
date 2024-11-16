# preprocessing.py

import cv2
import numpy as np

def preprocess_frame(frame):
    """Preprocesses a single frame."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    # Normalize pixel values
    normalized = resized / 255.0
    return normalized

def stack_frames(stacked_frames, frame, is_new_episode):
    """Stacks frames for temporal information."""
    processed_frame = preprocess_frame(frame)
    
    if is_new_episode:
        # Clear stack and fill with the same frame
        stacked_frames = [processed_frame for _ in range(4)]
    else:
        # Append frame and remove oldest
        stacked_frames.append(processed_frame)
        stacked_frames.pop(0)
    
    # Stack along the depth dimension
    stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames
