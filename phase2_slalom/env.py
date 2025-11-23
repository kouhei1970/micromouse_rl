import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os
from stable_baselines3 import PPO

class MicromouseSlalomEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, xml_file="assets/micromouse_slalom.xml", model_path="models/phase1_open.zip", max_steps=40000):
        self.render_mode = render_mode
        self.max_steps = max_steps
        
        # Load MuJoCo model
        if not os.path.exists(xml_file):
            raise FileNotFoundError(f"Model file {xml_file} not found. Please run phase2_slalom/generate_maze.py first.")
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        
        # Load Low-Level Controller (Phase 1)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Please ensure Phase 1 training is complete.")
        
        print(f"Loading Low-Level Controller from {model_path}...")
        self.low_level_model = PPO.load(model_path)
        print("Low-Level Controller loaded.")

        # High-Level Action Space: Target Velocity [v_ref, omega_ref]
        # v_ref: 0.0 to 0.5 m/s (Conservative speed for slalom)
        # omega_ref: -3.0 to 3.0 rad/s
        self.action_space = spaces.Box(
            low=np.array([0.0, -3.0]), 
            high=np.array([0.5, 3.0]), 
            dtype=np.float32
        )
        
        # High-Level Observation Space:
        # 0-3: Distance sensors (LF, LS, RF, RS)
        # 4: Linear Velocity
        # 5: Angular Velocity
        # 6: Goal Distance
        # 7: Goal Angle (relative to robot heading)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        self.viewer = None
        self.renderer = None
        
        # Goal Position (Center of cell 2,2)
        # Cell size 0.18m. (0,0) is at 0.09.
        # (2,2) is at 2*0.18 + 0.09 = 0.45
        self.goal_pos = np.array([0.45, 0.45]) 
        self.goal_threshold = 0.05 # 5cm radius
        
        self.current_step = 0
        self.last_distance = 0.0
        
        # Low-Level Control Frequency Settings
        # Phase 1 trained with 5 simulation steps per action (200Hz control)
        # High-Level Policy will run at 200Hz (5ms per step) to match Low-Level.
        # So we need 1 Low-Level step per 1 High-Level step.
        # 1 * 5ms = 5ms
        self.low_level_steps = 1
        
        # For rendering
        self.render_fps = 60

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Start at (0,0) facing North (+Y)
        # Path: (0,0) -> (0,1) -> (0,2) -> Turn Right -> (1,2) -> (2,2)
        self.data.qpos[0] = 0.09 # Center of (0,0)
        self.data.qpos[1] = 0.09
        
        # Orientation: Facing +Y (90 degrees)
        theta = np.pi / 2
        self.data.qpos[3] = np.cos(theta/2)
        self.data.qpos[4] = 0
        self.data.qpos[5] = 0
        self.data.qpos[6] = np.sin(theta/2)
        
        mujoco.mj_forward(self.model, self.data)
        
        # Initialize last distance
        self.last_distance = np.linalg.norm(self.data.qpos[0:2] - self.goal_pos)
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # Clip action to be safe
        target_v = np.clip(action[0], 0.0, 0.5)
        target_omega = np.clip(action[1], -3.0, 3.0)
        
        collision = False
        terminated = False
        truncated = False
        
        # --- Hierarchical Control Loop ---
        # Run Low-Level Controller for multiple steps
        for _ in range(self.low_level_steps):
            # 1. Construct observation for Low-Level Model
            # Phase 1 Obs: [LF, LS, RF, RS, LinV, AngV, LatAcc, TgtLin, TgtAng]
            
            data = self.data.sensordata
            distances = data[:4].copy()
            distances[distances < 0] = 0.15 # Max range replacement
            
            wheel_radius = 0.0135
            left_vel = self.data.qvel[6] * wheel_radius
            right_vel = self.data.qvel[7] * wheel_radius
            linear_velocity = (left_vel + right_vel) / 2.0
            
            angular_velocity = data[9] # Gyro Z
            lateral_accel = data[5]    # Accel Y
            
            low_level_obs = np.concatenate([
                distances,
                [linear_velocity],
                [angular_velocity],
                [lateral_accel],
                [target_v],
                [target_omega]
            ]).astype(np.float32)
            
            # 2. Predict motor action
            low_level_action, _ = self.low_level_model.predict(low_level_obs, deterministic=True)
            
            # 3. Apply to motors (Scale from [-1, 1] to [-3V, 3V])
            self.data.ctrl[0] = low_level_action[0] * 3.0
            self.data.ctrl[1] = low_level_action[1] * 3.0
            
            # 4. Step simulation (5 times to match Phase 1 training)
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)
            
            # 5. Check for collision immediately
            new_distances = self.data.sensordata[:4].copy()
            valid_distances = new_distances[new_distances >= 0]
            if len(valid_distances) > 0 and np.min(valid_distances) < 0.02:
                collision = True
                break
        
        # --- Calculate Reward & Check Termination ---
        
        # Current state
        current_pos = self.data.qpos[0:2]
        dist_to_goal = np.linalg.norm(current_pos - self.goal_pos)
            
        # Reward Components
        r_goal = 0.0
        r_collision = 0.0
        r_progress = 0.0
        r_time = -0.1
        
        # Goal Reached
        if dist_to_goal < self.goal_threshold:
            r_goal = 100.0
            terminated = True
            print("Goal Reached!")
            
        # Collision
        if collision:
            r_collision = -100.0
            terminated = True
            # print("Collision!")
            
        # Progress
        r_progress = 10.0 * (self.last_distance - dist_to_goal)
        self.last_distance = dist_to_goal
        
        reward = r_goal + r_collision + r_progress + r_time
        
        if self.current_step >= self.max_steps:
            truncated = True
            
        # Get new observation for High-Level
        obs = self._get_obs()
        
        info = {
            "dist_to_goal": dist_to_goal,
            "target_v": target_v,
            "target_omega": target_omega
        }
        
        if self.render_mode == "human":
            self._render_frame()
            
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        data = self.data.sensordata
        
        # 1. Distance Sensors
        distances = data[:4].copy()
        distances[distances < 0] = 0.15
        
        # 2. Velocities
        wheel_radius = 0.0135
        left_vel = self.data.qvel[6] * wheel_radius
        right_vel = self.data.qvel[7] * wheel_radius
        linear_velocity = (left_vel + right_vel) / 2.0
        angular_velocity = data[9]
        
        # 3. Goal Info
        current_pos = self.data.qpos[0:2]
        dist_to_goal = np.linalg.norm(self.goal_pos - current_pos)
        
        # Angle to goal
        # Vector to goal
        dx = self.goal_pos[0] - current_pos[0]
        dy = self.goal_pos[1] - current_pos[1]
        angle_to_goal = np.arctan2(dy, dx)
        
        # Robot heading (Yaw)
        # q = [w, x, y, z]
        q = self.data.qpos[3:7]
        # Yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
        # Since x=y=0 for 2D plane rotation usually, but let's be general
        w, x, y, z = q
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        
        # Relative angle
        rel_angle = angle_to_goal - yaw
        # Normalize to [-pi, pi]
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
        
        obs = np.concatenate([
            distances,
            [linear_velocity],
            [angular_velocity],
            [dist_to_goal],
            [rel_angle]
        ]).astype(np.float32)
        
        return obs

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=800, width=800)
            self.renderer.update_scene(self.data)
            return self.renderer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()
