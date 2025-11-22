from micromouse_env import MicromouseEnv
from gymnasium.utils.env_checker import check_env
import numpy as np

def main():
    print("Initializing environment...")
    env = MicromouseEnv(render_mode=None)
    
    print("Checking environment compliance with Gym API...")
    # check_env will raise warnings/errors if the env is not compliant
    check_env(env)
    print("Environment check passed!")
    
    print("Testing simulation loop...")
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Obs={obs}, Reward={reward}")
        
    env.close()
    print("Test complete.")

if __name__ == "__main__":
    main()
