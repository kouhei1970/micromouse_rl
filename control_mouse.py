import mujoco
import mujoco.viewer
import time
import numpy as np
import random

def main():
    try:
        print("Loading model...")
        model = mujoco.MjModel.from_xml_path("micromouse_maze.xml")
        data = mujoco.MjData(model)
        print("Model loaded. Launching viewer...")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()
            
            # Random movement state
            last_action_time = 0
            action_duration = 0
            left_vel = 0.0
            right_vel = 0.0
            
            while viewer.is_running():
                step_start = time.time()
                elapsed = time.time() - start_time
                
                # Random Control Logic
                if elapsed - last_action_time > action_duration:
                    # Pick new action
                    # Equal probability for all actions
                    action_type = random.choices(
                        ['forward', 'backward', 'turn_left', 'turn_right', 'stop'],
                        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
                        k=1
                    )[0]
                    
                    if action_type == 'forward':
                        left_vel = 25.0
                        right_vel = 25.0
                        action_duration = random.uniform(1.0, 3.0)
                    elif action_type == 'backward':
                        left_vel = -20.0
                        right_vel = -20.0
                        action_duration = random.uniform(0.5, 1.5)
                    elif action_type == 'turn_left':
                        left_vel = -15.0
                        right_vel = 15.0
                        action_duration = random.uniform(0.3, 0.8)
                    elif action_type == 'turn_right':
                        left_vel = 15.0
                        right_vel = -15.0
                        action_duration = random.uniform(0.3, 0.8)
                    else: # stop
                        left_vel = 0.0
                        right_vel = 0.0
                        action_duration = random.uniform(0.5, 1.0)
                    
                    last_action_time = elapsed
                    print(f"Action: {action_type}, Duration: {action_duration:.2f}s")
                
                data.ctrl[0] = left_vel
                data.ctrl[1] = right_vel

                # Step simulation
                mujoco.mj_step(model, data)
                
                # Sync viewer
                viewer.sync()
                
                # Print position every 0.5 seconds
                if int(elapsed * 2) > int((elapsed - (time.time() - step_start)) * 2):
                    # qpos[0:3] is x, y, z of free joint
                    print(f"Time: {elapsed:.1f}s, Pos: {data.qpos[0]:.3f}, {data.qpos[1]:.3f}, {data.qpos[2]:.3f}")

                # Time keeping
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
