import gymnasium as gym
from stable_baselines3 import PPO
import sys
import os
import datetime
sys.path.append(os.getcwd())
from phase2_slalom.env import MicromouseSlalomEnv
import numpy as np
import cv2

def main():
    # Create environment with rgb_array rendering
    env = MicromouseSlalomEnv(render_mode="rgb_array")
    
    model_path = "models/phase2_slalom.zip"
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    print("Recording video...")
    obs, _ = env.reset()
    
    frames = []
    
    # Run for one episode
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        frame = env.render()
        # Frame is RGB, convert to BGR for OpenCV
        if frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame_bgr)
            
        # Print mouse position for debugging
        if i % 10 == 0:
            mouse_pos = env.data.xpos[env.model.body('mouse').id]
            print(f"Step {i}: Mouse Pos: {mouse_pos}")

        if terminated or truncated:
            print(f"Episode finished at step {i}")
            break
            
    env.close()
    
    if len(frames) > 0:
        height, width, layers = frames[0].shape
        # Save as MP4
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/phase2_slalom/video_{timestamp}.mp4"
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        
        for frame in frames:
            out.write(frame)
            
        out.release()
        print(f"Video saved to {output_path}")
    else:
        print("No frames captured.")

if __name__ == "__main__":
    main()
