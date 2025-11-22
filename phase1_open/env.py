import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os

class MicromouseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, xml_file="assets/micromouse_open.xml", max_steps=4000):
        self.render_mode = render_mode
        self.max_steps = max_steps
        
        # Load model
        model_path = xml_file
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Please run phase1_open/generate_maze.py first.")
            
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Action space: Continuous Motor Voltage Control
        # [Left Motor Voltage, Right Motor Voltage]
        # Range: -1.0 to 1.0 (Scaled to -3V to 3V in step)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation space: 
        # 0-3: Distance sensors (LF, LS, RF, RS)
        # 4: Linear Velocity (m/s)
        # 5: Angular Velocity (rad/s)
        # 6: Lateral Acceleration (m/s^2)
        # 7: Target Linear Velocity (m/s)
        # 8: Target Angular Velocity (rad/s)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        self.viewer = None
        self.renderer = None
        
        self.target_linear_velocity = 0.0
        self.target_angular_velocity = 0.0
        self.steps_since_target_change = 0
        self.last_time = 0.0
        
        # Integral error states
        self.lin_integral = 0.0
        self.ang_integral = 0.0
        
        # Initial target setting
        self._update_targets()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.lin_integral = 0.0
        self.ang_integral = 0.0
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset to center of the open field (0, 0)
        # qpos[0:2] is x, y position
        # qpos[3:7] is orientation quaternion
        self.data.qpos[0] = 0
        self.data.qpos[1] = 0
        
        # Randomize orientation (Yaw rotation around Z axis)
        # Quaternion for rotation around Z axis by angle theta:
        # w = cos(theta/2), x = 0, y = 0, z = sin(theta/2)
        theta = self.np_random.uniform(0, 2 * np.pi)
        self.data.qpos[3] = np.cos(theta / 2)
        self.data.qpos[4] = 0
        self.data.qpos[5] = 0
        self.data.qpos[6] = np.sin(theta / 2)
        
        # Do NOT reset targets or counter here to maintain continuity across resets
        # self._update_targets()
        # self.steps_since_target_change = 0
        
        self.last_time = self.data.time
        
        # Forward dynamics to update sensors
        mujoco.mj_forward(self.model, self.data)
        
        observation = self._get_obs()
        info = {}
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info

    def step(self, action):
        self.current_step += 1
        # Change target command periodically (every 200 steps = 2 seconds approx)
        self.steps_since_target_change += 1
        if self.steps_since_target_change > 200:
            self._update_targets()
            self.steps_since_target_change = 0
            
        # Map continuous action to motor voltages
        # Action is [-1, 1], Motor limit is [-3V, 3V]
        left_v = action[0] * 3.0
        right_v = action[1] * 3.0
        
        self.data.ctrl[:] = [left_v, right_v]
        
        # Step simulation
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            
        observation = self._get_obs()
        
        # Extract current state for reward calculation
        linear_vel = observation[4]
        angular_vel = observation[5]
        
        # Reward Function
        reward = 0.0
        terminated = False
        truncated = False
        
        # 1. Collision Penalty & Termination
        # If any sensor is very close (< 0.02m), it's a collision
        if np.any(observation[:4] < 0.02):
            reward = 0.0 # No penalty for collision, just reset
            terminated = True
        elif self.current_step >= self.max_steps:
            truncated = True
        else:
            # 2. Target Velocity Tracking Reward (Feedback Control Goal)
            
            # Calculate errors
            lin_error = self.target_linear_velocity - linear_vel
            ang_error = self.target_angular_velocity - angular_vel
            
            # Update integrals (dt = 0.005)
            self.lin_integral += lin_error * 0.005
            self.ang_integral += ang_error * 0.005
            
            # Reward with Integral Penalty (PI control style)
            # Penalize both immediate error and accumulated error
            reward = 1.0 - (2.0 * abs(lin_error) + 1.0 * abs(self.lin_integral) + 
                            1.0 * abs(ang_error) + 1.0 * abs(self.ang_integral))
            
            # 3. Wall Proximity Penalty (Keep distance)
            # Only if not terminated
            min_dist = np.min(observation[:4])
            if min_dist < 0.05:
                reward -= 0.5 * (0.05 - min_dist) / 0.05 # Proportional penalty
            
        info = {
            "target_lin": self.target_linear_velocity,
            "target_ang": self.target_angular_velocity,
            "linear_vel": linear_vel,
            "angular_vel": angular_vel
        }
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, truncated, info

    def _update_targets(self):
        # Reset integrals on target change
        self.lin_integral = 0.0
        self.ang_integral = 0.0
        
        # Randomly select a mode
        r = self.np_random.random()
        
        if r < 0.3:
            # Mode 1: Turn in place (Linear = 0)
            self.target_linear_velocity = 0.0
            self.target_angular_velocity = self.np_random.uniform(-np.pi, np.pi)
        elif r < 0.6:
            # Mode 2: Straight (Angular = 0)
            # Target range updated to [0.1, 1.0] m/s as requested for Phase 1
            self.target_linear_velocity = self.np_random.uniform(0.1, 1.0) 
            self.target_angular_velocity = 0.0
        elif r < 0.7:
            # Mode 3: Stop (Both = 0)
            self.target_linear_velocity = 0.0
            self.target_angular_velocity = 0.0
        else:
            # Mode 4: Curve (Both non-zero)
            self.target_linear_velocity = self.np_random.uniform(0.1, 1.0)
            self.target_angular_velocity = self.np_random.uniform(-np.pi, np.pi)

    def _get_obs(self):
        data = self.data.sensordata
        
        # 1. Distance Sensors
        distances = data[:4].copy()
        # Replace -1 (no hit) with max range (0.15)
        distances[distances < 0] = 0.15
        
        # 2. Linear Velocity
        wheel_radius = 0.0135
        left_vel = self.data.qvel[6] * wheel_radius
        right_vel = self.data.qvel[7] * wheel_radius
        linear_velocity = (left_vel + right_vel) / 2.0
        
        # 3. Angular Velocity (Gyro Z)
        angular_velocity = data[9]
        
        # 4. Lateral Acceleration (Accel Y)
        lateral_accel = data[5]
        
        obs = np.concatenate([
            distances,
            [linear_velocity],
            [angular_velocity],
            [lateral_accel],
            [self.target_linear_velocity],
            [self.target_angular_velocity]
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
                # Increase resolution to 800x800
                self.renderer = mujoco.Renderer(self.model, height=800, width=800)
            
            # Use tracking camera if available
            try:
                self.renderer.update_scene(self.data, camera="track")
            except KeyError:
                self.renderer.update_scene(self.data)
                
            return self.renderer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()
