import gymnasium as gym
from stable_baselines3 import PPO
from micromouse_env import MicromouseEnv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def visualize():
    model_name = "ppo_micromouse_open"
    model_path = f"{model_name}.zip"
    
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Please wait for training to finish.")
        return

    print(f"Loading model from {model_path}...")
    # Create environment with render_mode="rgb_array"
    env = MicromouseEnv(render_mode="rgb_array", xml_file="micromouse_open.xml")
    model = PPO.load(model_path, env=env)

    obs, _ = env.reset()
    
    # Data logging
    log_data = {
        "step": [],
        "target_lin": [],
        "actual_lin": [],
        "target_ang": [],
        "actual_ang": [],
        "command": []
    }
    
    frames = []
    
    print("Running visualization episode (1000 steps)...")
    for i in range(1000): # Run for 1000 steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Capture frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        # Log data
        log_data["step"].append(i)
        log_data["target_lin"].append(info["target_lin"])
        log_data["actual_lin"].append(info["linear_vel"])
        log_data["target_ang"].append(info["target_ang"])
        log_data["actual_ang"].append(info["angular_vel"])
        
        if terminated or truncated:
            obs, _ = env.reset()
            
    env.close()
    
    # 1. Save Video
    print("Saving video to agent_behavior.mp4...")
    if len(frames) > 0:
        height, width, layers = frames[0].shape
        # OpenCV expects BGR, but frames are RGB
        # Use 'mp4v' codec
        video = cv2.VideoWriter('agent_behavior.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(bgr_frame)
            
        video.release()
        print("Video saved.")
    else:
        print("No frames captured.")
        
    # 2. Plot Data
    print("Plotting data to tracking_performance.png...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    steps = log_data["step"]
    
    # Linear Velocity
    ax1.plot(steps, log_data["target_lin"], 'r--', label='Target Linear Vel')
    ax1.plot(steps, log_data["actual_lin"], 'b-', label='Actual Linear Vel')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Linear Velocity Tracking')
    ax1.legend()
    ax1.grid(True)
    
    # Angular Velocity
    ax2.plot(steps, log_data["target_ang"], 'r--', label='Target Angular Vel')
    ax2.plot(steps, log_data["actual_ang"], 'g-', label='Actual Angular Vel')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_xlabel('Step')
    ax2.set_title('Angular Velocity Tracking')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('tracking_performance.png')
    print("Plot saved.")

    # Calculate and print MSE
    target_lin = np.array(log_data["target_lin"])
    actual_lin = np.array(log_data["actual_lin"])
    target_ang = np.array(log_data["target_ang"])
    actual_ang = np.array(log_data["actual_ang"])

    mse_lin = np.mean((target_lin - actual_lin)**2)
    mse_ang = np.mean((target_ang - actual_ang)**2)

    print("-" * 30)
    print("Evaluation Metrics:")
    print(f"Linear Velocity MSE: {mse_lin:.4f}")
    print(f"Angular Velocity MSE: {mse_ang:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    visualize()