import gymnasium as gym
from stable_baselines3 import PPO
import sys
import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import datetime

sys.path.append(os.getcwd())
from phase3_maze.env import MicromouseMazeEnv
from common.visualization import record_video

def evaluate_and_record(model_path, output_dir, n_episodes=5):
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    env = MicromouseMazeEnv(render_mode="rgb_array", maze_size=(3, 3))
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Record video
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(output_dir, f"eval_run_{timestamp}.mp4")
    print(f"Recording video to {video_path}...")
    record_video(env, model, video_path, num_episodes=n_episodes, deterministic=True)
    
    env.close()

if __name__ == "__main__":
    model_path = "models/phase3_maze_3x3.zip"
    output_dir = "outputs/phase3_maze/video"
    evaluate_and_record(model_path, output_dir, n_episodes=5)
