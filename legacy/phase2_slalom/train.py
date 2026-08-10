import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import sys
import os
sys.path.append(os.getcwd())
from phase2_slalom.env import MicromouseSlalomEnv

def main():
    # Create environment
    env = MicromouseSlalomEnv(render_mode=None)
    
    model_name = "models/phase2_slalom"
    
    print("Creating NEW High-Level model for Slalom Task")
    # We use MlpPolicy. 
    # The input is 8-dim (Sensors + Goal Info).
    # The output is 2-dim (Target V, Target Omega).
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./outputs/phase2_slalom/logs/")
    
    print("Starting training for Slalom Task...")
    
    # Save checkpoints
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./outputs/phase2_slalom/checkpoints/', name_prefix='ppo_slalom')
    
    # Train
    # 200k steps at 10Hz = 20,000s.
    # At 200Hz, 200k steps = 1,000s.
    # To get similar experience time, we need 20x steps = 4,000,000.
    # Let's try 1,000,000 steps first (5,000s = 1.4 hours of sim time).
    model.learn(total_timesteps=1000000, callback=checkpoint_callback)
    
    print("Training complete.")
    
    # Save the model
    model.save(model_name)
    print(f"Model saved to {model_name}.zip")

if __name__ == "__main__":
    main()
