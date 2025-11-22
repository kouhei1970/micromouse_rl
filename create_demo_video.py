import gymnasium as gym
from stable_baselines3 import PPO
from micromouse_env import MicromouseEnv
import numpy as np
import cv2
import os

def main():
    # Create environment with rgb_array rendering
    env = MicromouseEnv(render_mode="rgb_array")
    
    model_path = "ppo_micromouse_cmd.zip"
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=env)
    
    print("Initializing video writer...")
    obs, _ = env.reset()
    
    # Get a sample frame to determine size
    first_frame = env.render()
    height, width, layers = first_frame.shape
    print(f"Frame size: {width}x{height}")
    
    # Define the codec and create VideoWriter
    # mp4v is usually safe for .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter('micromouse_demo.mp4', fourcc, 30.0, (width, height))
    
    print("Recording simulation (500 steps)...")
    
    collisions = 0
    
    for i in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        frame = env.render()
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add Info Text
        target_cmd = int(obs[4])
        target_str = ["Forward", "Backward", "Left", "Right", "Stop"][target_cmd]
        action_str = ["Forward", "Backward", "Left", "Right", "Stop"][action]
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0) # Green
        thickness = 1
        
        cv2.putText(frame_bgr, f"Step: {i}", (10, 20), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"Target: {target_str}", (10, 40), font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Change color for Action if it matches Target
        act_color = (0, 255, 0) if action == target_cmd else (0, 0, 255) # Green if match, Red if not
        cv2.putText(frame_bgr, f"Action: {action_str}", (10, 60), font, font_scale, act_color, thickness, cv2.LINE_AA)
        
        cv2.putText(frame_bgr, f"Reward: {reward:.2f}", (10, 80), font, font_scale, color, thickness, cv2.LINE_AA)
        
        if terminated:
            cv2.putText(frame_bgr, "COLLISION!", (width//2 - 50, height//2), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            collisions += 1
            
        out.write(frame_bgr)
        
        if terminated or truncated:
            obs, _ = env.reset()
            
    out.release()
    env.close()
    print(f"Video saved to micromouse_demo.mp4")
    print(f"Total collisions in video: {collisions}")

if __name__ == "__main__":
    main()
