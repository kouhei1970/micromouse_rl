import gymnasium as gym
from micromouse_env import MicromouseEnv
import numpy as np
import time

def measure_max_velocity():
    # Create environment
    env = MicromouseEnv(render_mode=None, xml_file="micromouse_open.xml")
    obs, _ = env.reset()
    
    print("Measuring Maximum Linear Velocity...")
    max_lin_vel = 0.0
    
    # Run for 1000 steps (approx 10 seconds) with max forward input
    for i in range(1000):
        # Max forward: [1.0, 1.0]
        action = np.array([1.0, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_lin_vel = info["linear_vel"]
        if current_lin_vel > max_lin_vel:
            max_lin_vel = current_lin_vel
            
        if i % 100 == 0:
            print(f"Step {i}: Current Lin Vel = {current_lin_vel:.4f} m/s")
            
    print(f"Maximum Linear Velocity achieved: {max_lin_vel:.4f} m/s")
    
    # Reset for angular velocity measurement
    obs, _ = env.reset()
    print("\nMeasuring Maximum Angular Velocity...")
    max_ang_vel = 0.0
    
    # Run for 1000 steps with max rotation input
    for i in range(1000):
        # Max rotation (turn left): [-1.0, 1.0]
        action = np.array([-1.0, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_ang_vel = abs(info["angular_vel"])
        if current_ang_vel > max_ang_vel:
            max_ang_vel = current_ang_vel
            
        if i % 100 == 0:
            print(f"Step {i}: Current Ang Vel = {current_ang_vel:.4f} rad/s")
            
    print(f"Maximum Angular Velocity achieved: {max_ang_vel:.4f} rad/s")
    
    env.close()

if __name__ == "__main__":
    measure_max_velocity()
