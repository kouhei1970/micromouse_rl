import sys
import os
import datetime
sys.path.append(os.getcwd())
from phase1_open.env import MicromouseEnv
from phase1_open.verify_control import verify_control_performance
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    print("Testing Phase 1 Training Pipeline...")
    
    # 1. Setup Environment
    # Ensure assets directory is correct relative to CWD
    xml_path = "assets/micromouse_open.xml"
    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} not found. Run from workspace root.")
        return

    env = MicromouseEnv(render_mode=None, xml_file=xml_path, max_steps=500)
    log_dir = "logs/phase1_test"
    os.makedirs(log_dir, exist_ok=True)
    # Monitor will create monitor.csv
    env = Monitor(env, filename=os.path.join(log_dir, "monitor"))
    
    # 2. Setup Model
    print("Initializing PPO model...")
    model = PPO("MlpPolicy", env, verbose=1)
    
    # 3. Run Short Training
    print("Running short training (5000 steps)...")
    model.learn(total_timesteps=5000)
    print("Training finished.")
    
    # 4. Save Model
    model_path = "models/phase1_test_model"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")
    
    # 5. Analyze Results (Visualization)
    print("Analyzing results...")
    try:
        # Read monitor.csv
        # stable_baselines3 monitor logs are CSVs with a header
        # The filename might have .monitor.csv appended or just .csv depending on version
        log_path = os.path.join(log_dir, "monitor.monitor.csv")
        if not os.path.exists(log_path):
             log_path = os.path.join(log_dir, "monitor.csv")
        
        if os.path.exists(log_path):
            # Skip first line (metadata)
            df = pd.read_csv(log_path, skiprows=1)
            
            plt.figure(figsize=(10, 5))
            plt.plot(df['r'], label='Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Reward (Test Run)')
            plt.legend()
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_plot = f"outputs/phase1_open/test_reward_plot_{timestamp}.png"
            os.makedirs(os.path.dirname(output_plot), exist_ok=True)
            plt.savefig(output_plot)
            print(f"Reward plot saved to {output_plot}")
            plt.close()
        else:
            print(f"Log file not found at {log_path}")
            
    except Exception as e:
        print(f"Visualization failed: {e}")

    # 6. Verify Control Performance
    # Use the model and env we just trained/used
    # Enable video recording for verification
    verify_control_performance(model, env, record_video=True)

if __name__ == "__main__":
    main()
