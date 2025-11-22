import gymnasium as gym
from stable_baselines3 import PPO
from micromouse_env import MicromouseEnv
import time
import numpy as np

def main():
    # Create environment with human rendering
    # render_mode="human" will open the MuJoCo viewer window
    env = MicromouseEnv(render_mode="human")
    
    # Load the trained model
    model_path = "ppo_micromouse_continuous"
    print(f"Loading model from {model_path}...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    obs, _ = env.reset()
    
    print("Running visualization...")
    print("The robot will switch commands randomly every 100 steps (approx 1 second).")
    print("Press Ctrl+C in the terminal to stop.")
    
    commands = ["Forward", "Backward", "Turn Left", "Turn Right", "Stop"]
    last_cmd = -1
    
    try:
        while True:
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Get current command
            current_cmd_idx = info.get("target_command", 0)
            
            if current_cmd_idx != last_cmd:
                cmd_name = commands[current_cmd_idx]
                print(f"Command: {cmd_name}")
                last_cmd = current_cmd_idx
            
            # Control frame rate (approx 60-100 FPS)
            time.sleep(0.01)
            
            if terminated or truncated:
                print("Resetting environment...")
                obs, _ = env.reset()
                last_cmd = -1
                
    except KeyboardInterrupt:
        print("\nVisualization stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
