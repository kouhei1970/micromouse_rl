import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import cv2
from stable_baselines3.common.results_plotter import load_results, ts2xy

def record_video(env, model, output_path, num_episodes=1, deterministic=True, fps=30):
    """
    Records a video of the agent interacting with the environment.
    Also plots control target vs actual velocity if available in info.
    
    Args:
        env: The gymnasium environment (must be in rgb_array render mode).
        model: The trained Stable Baselines 3 model.
        output_path: Path to save the video (e.g., 'video.mp4').
        num_episodes: Number of episodes to record.
        deterministic: Whether to use deterministic actions.
        fps: Frames per second for the video.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    frames = []
    
    # Data storage for plotting
    history = {
        "target_lin": [], "actual_lin": [],
        "target_ang": [], "actual_ang": []
    }
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        step = 0
        total_reward = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Collect data if available
            if "target_lin" in info and "actual_lin" in info:
                history["target_lin"].append(info["target_lin"])
                history["actual_lin"].append(info["actual_lin"])
            if "target_ang" in info and "actual_ang" in info:
                history["target_ang"].append(info["target_ang"])
                history["actual_ang"].append(info["actual_ang"])
            
            frame = env.render()
            
            # Overlay Info using OpenCV
            frame_writable = frame.copy()
            text_color = (255, 255, 255)
            
            cv2.putText(frame_writable, f"Ep: {episode+1} Step: {step}", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            cv2.putText(frame_writable, f"Reward: {total_reward:.2f}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            
            if "dist_to_goal" in info:
                cv2.putText(frame_writable, f"Dist: {info['dist_to_goal']:.2f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            
            # Overlay Control Info if available
            if "target_lin" in info:
                cv2.putText(frame_writable, f"V_tgt: {info['target_lin']:.2f} V_act: {info['actual_lin']:.2f}", (10, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            if "target_ang" in info:
                cv2.putText(frame_writable, f"W_tgt: {info['target_ang']:.2f} W_act: {info['actual_ang']:.2f}", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            frames.append(frame_writable)
            
    print(f"Saving video to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    print("Video saved.")
    
    # Plot Control Performance if data exists
    if history["target_lin"]:
        plot_control_performance(history, output_path.replace(".mp4", "_control.png"))

def plot_control_performance(history, output_path):
    """
    Plots Target vs Actual velocity.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    steps = range(len(history["target_lin"]))
    
    # Linear Velocity
    ax1.plot(steps, history["target_lin"], label="Target Linear", linestyle='--', color='orange')
    ax1.plot(steps, history["actual_lin"], label="Actual Linear", color='blue')
    ax1.set_ylabel("Linear Velocity (m/s)")
    ax1.set_title("Control Performance: Linear Velocity")
    ax1.legend()
    ax1.grid(True)
    
    # Angular Velocity
    if history["target_ang"]:
        ax2.plot(steps, history["target_ang"], label="Target Angular", linestyle='--', color='orange')
        ax2.plot(steps, history["actual_ang"], label="Actual Angular", color='blue')
        ax2.set_ylabel("Angular Velocity (rad/s)")
        ax2.set_title("Control Performance: Angular Velocity")
        ax2.legend()
        ax2.grid(True)
        
    ax2.set_xlabel("Step")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Control performance plot saved to {output_path}")
    plt.close()

def plot_learning_curve(log_dir, output_path, window=100):
    """
    Plots the learning curve (Reward & Episode Length) from monitor files.
    Standard RL visualization: Rolling mean of reward and length.
    
    Args:
        log_dir: Directory containing *.monitor.csv files.
        output_path: Path to save the plot image.
        window: Rolling window size for smoothing.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load results using SB3 helper or manual pandas
    # Manual pandas is often more robust for custom plotting
    monitor_files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
    if not monitor_files:
        print(f"No monitor files found in {log_dir}")
        return

    data_frames = []
    for file in monitor_files:
        # Skip first 1 line (metadata)
        df = pd.read_csv(file, skiprows=1)
        data_frames.append(df)
    
    if not data_frames:
        print("No valid data found.")
        return
        
    # Concatenate if multiple files (e.g. multiple CPUs), but usually we sort by time
    # For single run, just one file usually.
    full_df = pd.concat(data_frames).sort_values(by='t')
    
    # Calculate rolling mean (at current position, using past window data)
    rolling_reward = full_df['r'].rolling(window=window).mean()
    rolling_length = full_df['l'].rolling(window=window).mean()

    # Create Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Reward Plot
    ax1.plot(full_df['l'].cumsum(), full_df['r'], 'o-', color='lightcoral', linewidth=1.5, markersize=3, alpha=0.6, label='Raw Reward')
    ax1.plot(full_df['l'].cumsum(), rolling_reward, '-', color='blue', linewidth=4, label=f'Rolling Mean ({window})')
    ax1.set_ylabel('Episode Reward', fontsize=12)
    ax1.set_title('Learning Curve: Reward', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Length Plot
    ax2.plot(full_df['l'].cumsum(), full_df['l'], 'o-', color='lightgreen', linewidth=1.5, markersize=3, alpha=0.6, label='Raw Length')
    ax2.plot(full_df['l'].cumsum(), rolling_length, '-', color='orange', linewidth=4, label=f'Rolling Mean ({window})')
    ax2.set_ylabel('Episode Length', fontsize=12)
    ax2.set_xlabel('Total Timesteps', fontsize=12)
    ax2.set_title('Learning Curve: Episode Length', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Learning curve saved to {output_path}")
    plt.close()
