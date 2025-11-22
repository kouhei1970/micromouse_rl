import sys
import os
import datetime
sys.path.append(os.getcwd())
from phase1_open.env import MicromouseEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
import cv2

def verify_control_performance(model=None, env=None, record_video=False):
    print("Verifying control performance...")
    
    # 1. Setup Environment if not provided
    if env is None:
        xml_path = "assets/micromouse_open.xml"
        if not os.path.exists(xml_path):
            print(f"Error: {xml_path} not found. Run from workspace root.")
            return
        # Use a longer max_steps to accommodate the 1000 step test
        # If recording video, we need rgb_array mode
        render_mode = "rgb_array" if record_video else None
        env = MicromouseEnv(render_mode=render_mode, xml_file=xml_path, max_steps=2000)
    else:
        # If env is provided but we want video, ensure render_mode is correct
        if record_video and env.unwrapped.render_mode != "rgb_array":
            print("Switching env render_mode to 'rgb_array' for video recording.")
            env.unwrapped.render_mode = "rgb_array"
    
    # 2. Load Model if not provided
    if model is None:
        # Try to load the main trained model first
        model_path = "models/phase1_open.zip"
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found. Trying test model...")
            model_path = "models/phase1_test_model.zip"
            
        if not os.path.exists(model_path):
            print("Error: No trained model found. Please train the model first.")
            return

        print(f"Loading model from {model_path}...")
        model = PPO.load(model_path, env=env)
    
    # 3. Run Verification Loop
    obs, _ = env.reset()
    
    target_lins = []
    actual_lins = []
    target_angs = []
    actual_angs = []
    
    # Video Writer Setup
    out = None
    if record_video:
        # Render first frame to get size
        first_frame = env.render()
        if first_frame is not None:
            height, width, layers = first_frame.shape
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"outputs/phase1_open/control_verification_{timestamp}.mp4"
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
            print(f"Recording video to {video_path}...")
        else:
            print("Warning: Render returned None. Video will not be recorded.")

    # Define target sequence (Change every 200 steps)
    # Total 1000 steps -> 5 phases
    # Phase 1 (0-199): Stop (0, 0)
    # Phase 2 (200-399): Straight Slow (0.5, 0)
    # Phase 3 (400-599): Straight Fast (1.0, 0)
    # Phase 4 (600-799): Turn (0, 3.14)
    # Phase 5 (800-999): Curve (0.5, 1.57)
    
    for i in range(1000):
        # Manually set targets based on step count
        if i < 200:
            tgt_lin, tgt_ang = 0.0, 0.0
            phase_name = "Stop"
        elif i < 400:
            tgt_lin, tgt_ang = 0.5, 0.0
            phase_name = "Straight Slow"
        elif i < 600:
            tgt_lin, tgt_ang = 1.0, 0.0
            phase_name = "Straight Fast"
        elif i < 800:
            tgt_lin, tgt_ang = 0.0, np.pi # Turn left
            phase_name = "Turn Left"
        else:
            tgt_lin, tgt_ang = 0.5, np.pi/2 # Curve
            phase_name = "Curve"
            
        # Override environment targets
        # Note: We need to access the unwrapped env if it's wrapped (e.g. Monitor)
        unwrapped_env = env.unwrapped
        unwrapped_env.target_linear_velocity = tgt_lin
        unwrapped_env.target_angular_velocity = tgt_ang
        
        # Also update the observation to reflect these new targets
        # The observation contains [..., tgt_lin, tgt_ang] at indices 7 and 8
        obs[7] = tgt_lin
        obs[8] = tgt_ang
        
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        
        # Record data
        # Obs: 4:LinV, 5:AngV
        actual_lins.append(obs[4])
        actual_angs.append(obs[5])
        target_lins.append(tgt_lin)
        target_angs.append(tgt_ang)
        
        if out is not None:
            frame = env.render()
            if frame is not None:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Add Info Text
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame_bgr, f"Step: {i} Phase: {phase_name}", (10, 30), font, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"Tgt Lin: {tgt_lin:.2f} Act: {obs[4]:.2f}", (10, 60), font, 0.6, (0, 255, 255), 1)
                cv2.putText(frame_bgr, f"Tgt Ang: {tgt_ang:.2f} Act: {obs[5]:.2f}", (10, 90), font, 0.6, (0, 255, 255), 1)
                
                out.write(frame_bgr)
        
        if terminated or truncated:
            obs, _ = env.reset()
            
    if out is not None:
        out.release()
        print("Video recording complete.")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(target_lins, label='Target', linestyle='--', color='orange', linewidth=2)
    ax1.plot(actual_lins, label='Actual', color='blue', alpha=0.7)
    ax1.set_title('Linear Velocity Tracking')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(target_angs, label='Target', linestyle='--', color='orange', linewidth=2)
    ax2.plot(actual_angs, label='Actual', color='blue', alpha=0.7)
    ax2.set_title('Angular Velocity Tracking')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Velocity (rad/s)')
    ax2.legend()
    ax2.grid(True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/phase1_open/control_performance_{timestamp}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Control performance plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    # When run as a script, generate video by default
    verify_control_performance(record_video=True)
