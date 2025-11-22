import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from micromouse_env import MicromouseEnv
import os
import numpy as np

def main():
    # Create environment with Open Maze
    env = MicromouseEnv(render_mode=None, xml_file="micromouse_open.xml")
    
    # Instantiate the agent
    model_name = "ppo_micromouse_open"
    model_path = f"{model_name}.zip"
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path} to continue training...")
        model = PPO.load(model_path, env=env, verbose=1)
    else:
        print("Creating NEW model for Open Field")
        model = PPO("MlpPolicy", env, verbose=1)
    
    print("Starting training in Open Field...")
    # Train for sufficient steps to master velocity tracking
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./logs/', name_prefix='ppo_micromouse_open')
    model.learn(total_timesteps=200000, callback=checkpoint_callback)
    print("Training complete.")
    
    # Save the model
    model.save(model_name)
    print(f"Model saved to {model_name}.zip")
    
    # Test the trained agent
    print("Testing trained agent...")
    obs, _ = env.reset()
    for i in range(200): 
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Action is now continuous [LeftV, RightV]
        left_v = action[0] * 3.0
        right_v = action[1] * 3.0
        
        target_lin = obs[7]
        target_ang = obs[8]
        
        # Observation: 0-3:Dist, 4:LinV, 5:AngV, 6:LatAcc, 7:TgtLin, 8:TgtAng
        lin_v = obs[4]
        ang_v = obs[5]
        
        print(f"Step {i}: TgtLin={target_lin:.2f}, TgtAng={target_ang:.2f}, L={left_v:.2f}, R={right_v:.2f}, LinV={lin_v:.2f}, AngV={ang_v:.2f}, Rew={reward:.2f}")
        
        if terminated or truncated:
            print("Episode finished")
            obs, _ = env.reset()

if __name__ == "__main__":
    main()
