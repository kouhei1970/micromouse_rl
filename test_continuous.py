import gymnasium as gym
from stable_baselines3 import PPO
from micromouse_env import MicromouseEnv
import numpy as np
import time

def main():
    env = MicromouseEnv(render_mode=None)
    
    # --- Physics Check ---
    print("\n--- Physics Check (Full Forward 3V) ---")
    obs, _ = env.reset()
    for i in range(20):
        # Apply full forward voltage (Action 1.0 -> 3.0V)
        # We bypass the agent and send action directly
        action = np.array([1.0, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        lin_vel = obs[4]
        left_v = action[0] * 3.0
        right_v = action[1] * 3.0
        min_dist = np.min(obs[:4])
        
        print(f"Step {i}: L_V={left_v:.1f}, R_V={right_v:.1f}, LinVel={lin_vel:.4f}, MinDist={min_dist:.4f}")
        if terminated:
            print("Collision!")
            break
            
    print("--- End Physics Check ---\n")

    model_path = "ppo_micromouse_continuous.zip"
    model = PPO.load(model_path, env=env)
    
    obs, _ = env.reset()
    
    # Manual command sequence
    # 0: Forward, 4: Stop, 1: Backward, 2: Left, 3: Right
    commands = [0, 4, 1, 4, 2, 4, 3, 4]
    steps_per_command = 50
    
    print(f"{'Step':<5} | {'Cmd':<8} | {'L_Volts':<7} | {'R_Volts':<7} | {'LinVel':<7} | {'AngVel':<7} | {'Reward':<7} | {'MinDist':<7}")
    print("-" * 80)
    
    total_steps = 0
    for cmd in commands:
        # Force the environment's target command
        env.target_command = cmd
        env.steps_since_command_change = 0 # Prevent auto-change
        
        cmd_str = ["Fwd", "Back", "Left", "Right", "Stop"][cmd]
        
        for i in range(steps_per_command):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Extract info for printing
            left_v = action[0] * 3.0
            right_v = action[1] * 3.0
            lin_vel = obs[4]
            ang_vel = obs[5]
            min_dist = np.min(obs[:4])
            
            if i % 10 == 0:
                print(f"{total_steps:<5} | {cmd_str:<8} | {left_v:7.2f} | {right_v:7.2f} | {lin_vel:7.3f} | {ang_vel:7.3f} | {reward:7.3f} | {min_dist:7.3f}")
            
            total_steps += 1
            
            if terminated or truncated:
                print("!!! Collision or Truncated !!!")
                obs, _ = env.reset()
                # Restore command after reset if we want to persist testing
                env.target_command = cmd 
                env.steps_since_command_change = 0

if __name__ == "__main__":
    main()
