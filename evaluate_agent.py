import gymnasium as gym
from stable_baselines3 import PPO
from micromouse_env import MicromouseEnv
import numpy as np

def main():
    env = MicromouseEnv(render_mode=None)
    
    model_path = "ppo_micromouse_cmd.zip"
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=env)
    
    print("Starting evaluation (1000 steps)...")
    obs, _ = env.reset()
    
    collisions = 0
    total_reward = 0
    steps = 0
    
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        target_cmd = int(obs[4])
        target_str = ["Forward", "Backward", "Left", "Right", "Stop"][target_cmd]
        action_str = ["Forward", "Backward", "Left", "Right", "Stop"][action]
        
        if i % 100 == 0:
            print(f"Step {i}: Target={target_str}, Action={action_str}, Reward={reward:.2f}, MinDist={np.min(obs[:4]):.4f}")
        
        if terminated:
            print(f"Collision at step {i}!")
            collisions += 1
            obs, _ = env.reset()
        elif truncated:
            obs, _ = env.reset()
            
    print(f"Evaluation complete.")
    print(f"Total Steps: {steps}")
    print(f"Collisions: {collisions}")
    print(f"Average Reward: {total_reward / steps:.4f}")

if __name__ == "__main__":
    main()
