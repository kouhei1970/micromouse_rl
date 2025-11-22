import gymnasium as gym
from stable_baselines3 import PPO
from micromouse_env import MicromouseEnv
import numpy as np
import time
import mujoco

def get_heading(q):
    # MuJoCo quaternion is [w, x, y, z]
    w, x, y, z = q
    # Yaw (z-axis rotation)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return yaw

def main():
    # Create environment
    env = MicromouseEnv(render_mode="human")
    
    # Load model
    model_path = "ppo_micromouse_continuous"
    print(f"Loading model from {model_path}...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    obs, _ = env.reset()
    
    # Maze Center Target (approximate)
    # Maze is 16x16 cells of 0.18m
    # Center is around (1.44, 1.44)
    target_x, target_y = 1.44, 1.44
    
    print("Starting Maze Solver...")
    print("Goal: Center of the maze.")
    
    try:
        while True:
            # 1. Get Robot State
            # qpos[0]=x, qpos[1]=y, qpos[3:7]=quat
            x, y = env.data.qpos[0], env.data.qpos[1]
            q = env.data.qpos[3:7]
            yaw = get_heading(q)
            
            # Discretize Heading (0:East, 1:North, 2:West, 3:South)
            # Yaw: 0=East, pi/2=North, pi/-pi=West, -pi/2=South
            # Normalize to 0..4
            heading_idx = int(round(yaw / (np.pi/2))) % 4
            
            # 2. Check Walls using Sensors
            # obs[:4] = [LF, LS, RF, RS]
            # Threshold for "Blocked" (e.g. < 0.15m)
            # Note: Sensors are relative to robot
            front_blocked = (obs[0] < 0.12) or (obs[2] < 0.12)
            left_blocked = (obs[1] < 0.12)
            right_blocked = (obs[3] < 0.12)
            
            # Map to Global Directions
            # Directions: 0:E, 1:N, 2:W, 3:S
            walls = [False, False, False, False]
            
            # Mapping relative to global
            # Front is heading_idx
            # Left is (heading_idx + 1) % 4
            # Right is (heading_idx - 1) % 4
            # Back is (heading_idx + 2) % 4 (Assume open unless we know otherwise, or treat as last resort)
            
            walls[heading_idx] = front_blocked
            walls[(heading_idx + 1) % 4] = left_blocked
            walls[(heading_idx - 1) % 4] = right_blocked
            
            # 3. Greedy Decision
            best_dir = -1
            min_dist = float('inf')
            
            # Check all 4 directions
            for d in range(4):
                # If wall detected, skip (unless it's 'Back', which we can't sense directly, so we assume open)
                # But wait, if we are blocked Front/Left/Right, Back is the ONLY option.
                # So we only skip if we KNOW there is a wall.
                # We know walls for Front, Left, Right. We don't know Back.
                
                is_back = (d == (heading_idx + 2) % 4)
                if not is_back and walls[d]:
                    continue
                
                # Predict position
                step = 0.18
                nx, ny = x, y
                if d == 0: nx += step
                elif d == 1: ny += step
                elif d == 2: nx -= step
                elif d == 3: ny -= step
                
                dist = np.sqrt((nx - target_x)**2 + (ny - target_y)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_dir = d
            
            # 4. Convert to Robot Command
            # 0:Forward, 1:Backward, 2:Left, 3:Right, 4:Stop
            cmd = 0 # Default Forward
            
            if best_dir == -1:
                # Stuck? Should not happen if Back is allowed.
                cmd = 4 # Stop
            else:
                diff = (best_dir - heading_idx) % 4
                if diff == 0: cmd = 0 # Forward
                elif diff == 1: cmd = 2 # Turn Left
                elif diff == 2: cmd = 1 # Backward (or Turn Left/Right)
                elif diff == 3: cmd = 3 # Turn Right
            
            # Override Environment Command
            env.target_command = cmd
            env.steps_since_command_change = 0 # Reset counter to prevent random switch
            
            # Update Observation with new command
            obs[-1] = float(cmd)
            
            # Predict Action
            action, _ = model.predict(obs, deterministic=True)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Debug Print (optional)
            # cmd_str = ["Fwd", "Back", "Left", "Right", "Stop"][cmd]
            # print(f"Pos:({x:.2f},{y:.2f}) Head:{heading_idx} Cmd:{cmd_str}")
            
            time.sleep(0.01)
            
            if terminated or truncated:
                print("Resetting...")
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\nSolver stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
