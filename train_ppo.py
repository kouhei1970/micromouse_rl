import gymnasium as gym
from stable_baselines3 import PPO
from micromouse_env import MicromouseEnv
import os
import numpy as np

def main():
    # Create environment
    env = MicromouseEnv(render_mode=None)
    
    # Instantiate the agent
    model_path = "ppo_micromouse_continuous.zip"
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = PPO.load(model_path, env=env, verbose=1)
    else:
        print("Creating new model (Continuous Action Space)")
        model = PPO("MlpPolicy", env, verbose=1)
    
    print("Starting training...")
    # Train for a longer duration to allow learning continuous control
    model.learn(total_timesteps=1000000)
    print("Training complete.")
    
    # Save the model
    model.save("ppo_micromouse_continuous")
    print("Model saved to ppo_micromouse_continuous.zip")
    
    # Test the trained agent
    print("Testing trained agent...")
    obs, _ = env.reset()
    for i in range(100): # Run longer test
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Action is now continuous [LeftV, RightV]
        left_v = action[0] * 3.0
        right_v = action[1] * 3.0
        
        target_cmd = int(obs[-1])
        target_str = ["Forward", "Backward", "Left", "Right", "Stop"][target_cmd]
        
        print(f"Step {i}: Target={target_str}, L_V={left_v:.2f}, R_V={right_v:.2f}, Reward={reward:.4f}, MinDist={np.min(obs[:4]):.4f}")
        
        if terminated or truncated:
            print("Episode finished (Collision or Truncated)")
            obs, _ = env.reset()

if __name__ == "__main__":
    main()
