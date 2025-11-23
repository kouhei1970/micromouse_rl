import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os
from stable_baselines3 import PPO
import sys

# Add workspace root to path
sys.path.append(os.getcwd())
from phase3_maze.maze_generator import RandomMazeGenerator

class MicromouseMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, maze_size=(7, 7), model_path="models/phase1_open.zip", max_steps=40000, maze_regen_interval=20, spawn_area_size=5):
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.maze_width, self.maze_height = maze_size

        # Spawn area configuration (inner region for robot start position)
        # For 7x7 maze with 5x5 spawn area: cells (1,1) to (5,5)
        self.spawn_area_size = spawn_area_size
        self.spawn_offset = (self.maze_width - self.spawn_area_size) // 2

        # Maze regeneration control
        self.maze_regen_interval = maze_regen_interval  # Regenerate maze every N episodes
        self.episode_count = 0

        # Maze Generator
        self.generator = RandomMazeGenerator(self.maze_width, self.maze_height)
        self.xml_file = f"assets/micromouse_random_{self.maze_width}x{self.maze_height}.xml"

        # Generate initial maze
        self.generator.generate_maze()
        self.generator.generate_mjcf(self.xml_file)
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(self.xml_file)
        self.data = mujoco.MjData(self.model)
        
        # Load Low-Level Controller (Phase 1)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Please ensure Phase 1 training is complete.")
        
        print(f"Loading Low-Level Controller from {model_path}...")
        self.low_level_model = PPO.load(model_path)
        print("Low-Level Controller loaded.")

        # Action Space: Target Velocity [v (m/s), omega (rad/s)]
        # Increased range as per user request
        self.action_space = spaces.Box(
            low=np.array([0.0, -5.0]),
            high=np.array([1.0, 5.0]),
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
        
        # Goal Position (Center of top-right cell)
        # (width-1, height-1)
        self.goal_pos = np.array([
            (self.maze_width - 1) * 0.18 + 0.09,
            (self.maze_height - 1) * 0.18 + 0.09
        ])
        self.goal_threshold = 0.05 # 5cm radius
        
        self.current_step = 0
        self.last_distance = 0.0
        
        # Low-Level Control Frequency Settings
        # High-Level Policy will run at 200Hz (5ms per step) to match Low-Level.
        self.low_level_steps = 1
        
        # For rendering
        self.render_fps = 60

    def _get_adjacent_cells(self, cell):
        """
        Get adjacent cells that are reachable (no wall between).
        Returns list of (x, y) tuples.
        """
        x, y = cell
        adjacent = []

        # Check all 4 directions
        # North: y+1
        if y + 1 < self.maze_height and self.generator.h_walls[x, y+1] == 0:
            adjacent.append((x, y+1))

        # South: y-1
        if y - 1 >= 0 and self.generator.h_walls[x, y] == 0:
            adjacent.append((x, y-1))

        # East: x+1
        if x + 1 < self.maze_width and self.generator.v_walls[x+1, y] == 0:
            adjacent.append((x+1, y))

        # West: x-1
        if x - 1 >= 0 and self.generator.v_walls[x, y] == 0:
            adjacent.append((x-1, y))

        return adjacent

    def _find_two_step_cells(self, start_cell):
        """
        Find all cells exactly 2 steps away from start_cell.
        Returns list of (goal_cell, intermediate_cell) tuples.
        """
        two_step_cells = []

        # Get 1-step neighbors
        one_step_neighbors = self._get_adjacent_cells(start_cell)

        # For each 1-step neighbor, get their neighbors
        for intermediate in one_step_neighbors:
            two_step_neighbors = self._get_adjacent_cells(intermediate)

            for goal in two_step_neighbors:
                # Exclude start cell and 1-step neighbors
                if goal != start_cell and goal not in one_step_neighbors:
                    two_step_cells.append((goal, intermediate))

        return two_step_cells

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_count += 1

        # Regenerate Maze periodically to save time
        # Regenerate every N episodes (e.g., every 20 episodes)
        if self.episode_count % self.maze_regen_interval == 1:
            self.generator.generate_maze()
            self.generator.generate_mjcf(self.xml_file)

            # Reload MuJoCo model
            self.model = mujoco.MjModel.from_xml_path(self.xml_file)
            self.data = mujoco.MjData(self.model)

            # Reset Viewer if exists
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None

        mujoco.mj_resetData(self.model, self.data)

        # Try to find a valid start cell with 2-step neighbors
        # Restrict to spawn area (inner 5x5 for 7x7 maze)
        spawn_cells = [
            (x, y)
            for x in range(self.spawn_offset, self.spawn_offset + self.spawn_area_size)
            for y in range(self.spawn_offset, self.spawn_offset + self.spawn_area_size)
        ]
        np.random.shuffle(spawn_cells)

        start_cell = None
        two_step_options = []

        for candidate_cell in spawn_cells:
            two_step_options = self._find_two_step_cells(candidate_cell)
            if len(two_step_options) > 0:
                start_cell = candidate_cell
                break

        # If no valid start cell found in spawn area, regenerate maze
        if start_cell is None:
            print("Warning: No valid 2-step paths in spawn area. Regenerating maze...")
            self.generator.generate_maze()
            self.generator.generate_mjcf(self.xml_file)
            self.model = mujoco.MjModel.from_xml_path(self.xml_file)
            self.data = mujoco.MjData(self.model)

            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None

            mujoco.mj_resetData(self.model, self.data)

            # Try again with new maze
            np.random.shuffle(spawn_cells)
            for candidate_cell in spawn_cells:
                two_step_options = self._find_two_step_cells(candidate_cell)
                if len(two_step_options) > 0:
                    start_cell = candidate_cell
                    break

            # If still no valid cell, use center cell as fallback
            if start_cell is None:
                print("Warning: Still no valid 2-step paths. Using center cell.")
                center = self.maze_width // 2
                start_cell = (center, center)

        # Randomly select one 2-step path
        if len(two_step_options) > 0:
            goal_cell, intermediate_cell = two_step_options[np.random.randint(len(two_step_options))]
        else:
            # Fallback: use adjacent cell if available
            one_step_neighbors = self._get_adjacent_cells(start_cell)
            if len(one_step_neighbors) > 0:
                goal_cell = one_step_neighbors[0]
                intermediate_cell = None
            else:
                goal_cell = start_cell
                intermediate_cell = None

        # Store for potential intermediate reward
        self.start_cell = start_cell
        self.intermediate_cell = intermediate_cell
        self.goal_cell = goal_cell

        start_x, start_y = start_cell

        # Set robot position at start cell center
        self.data.qpos[0] = start_x * 0.18 + 0.09
        self.data.qpos[1] = start_y * 0.18 + 0.09
        self.data.qpos[2] = 0.005 # Lift 5mm to prevent floor penetration

        # Random initial orientation: North, East, South, or West
        # 0° = East (+X), 90° = North (+Y), 180° = West (-X), 270° = South (-Y)
        orientations = [0, 90, 180, 270]
        angle_deg = np.random.choice(orientations)
        angle = np.radians(angle_deg)

        # Euler (0, 0, angle) -> Quaternion
        # q = [cos(a/2), 0, 0, sin(a/2)] for z-axis rotation
        self.data.qpos[3] = np.cos(angle/2)
        self.data.qpos[4] = 0
        self.data.qpos[5] = 0
        self.data.qpos[6] = np.sin(angle/2)

        # Set goal position at goal cell center
        self.goal_pos = np.array([
            goal_cell[0] * 0.18 + 0.09,
            goal_cell[1] * 0.18 + 0.09
        ])

        # Initial observation
        self.last_distance = np.linalg.norm(self.data.qpos[0:2] - self.goal_pos)

        # Track visited cells for Exploration Reward
        self.visited_cells = set()
        self.visited_cells.add(start_cell)

        # Track if intermediate cell has been reached
        self.intermediate_reached = False

        # Direction name for logging
        direction_names = {0: "East", 90: "North", 180: "West", 270: "South"}
        direction_name = direction_names[angle_deg]

        print(f"Episode {self.episode_count}: Start={start_cell} ({direction_name}), Goal={goal_cell}, Intermediate={intermediate_cell}")

        obs = self._get_obs()

        return obs, {}

    def step(self, action):
        # High-Level Action: Target Velocity [v_ref, omega_ref]
        # Action is in range [-1, 1] if using PPO with Box space, but we defined it as [0, 0.5] and [-3, 3].
        # SB3 PPO automatically clips actions to the defined action_space during training?
        # Actually, SB3 PPO outputs actions in the range of the action space if it's Box.
        # But if the policy is initialized with small weights, it might output near 0.
        
        target_v = float(action[0])
        target_omega = float(action[1])
        
        # Force minimum velocity to prevent getting stuck if target_v is too small but positive?
        # Or just rely on the agent to learn.
        # However, if the agent outputs 0.01, and friction is high, it might not move.
        # But we are using a Low-Level Controller which controls voltage.
        # If Low-Level Controller sees target 0.01, it tries to achieve 0.01.
        
        # Run Low-Level Controller
        # Low-Level takes [LF, LS, RF, RS, v_act, w_act, lat_acc, v_ref, w_ref]
        # And outputs [left_motor, right_motor]
        
        # Get Low-Level Observation
        ll_obs = self._get_low_level_obs(target_v, target_omega)
        
        # Predict Low-Level Action (Deterministic)
        ll_action, _ = self.low_level_model.predict(ll_obs, deterministic=True)
        
        # Apply to MuJoCo
        # Phase 1 Action Space was [-1, 1], mapped to [-3V, 3V]
        # We need to ensure the mapping is consistent with Phase 1 env.
        self.data.ctrl[0] = ll_action[0] * 3.0
        self.data.ctrl[1] = ll_action[1] * 3.0
        
        # Step Simulation
        # Phase 1 ran at 200Hz (5ms) with 1ms physics timestep.
        # So we step physics 5 times per control step.
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            
            # Check for collision (simple check: if robot overlaps with wall)
            # MuJoCo handles collision physics, but we want to detect it for reward.
            # We can check contact forces.
            if len(self.data.contact) > 0:
                # Check if any contact involves the robot chassis/wheels and a wall
                # This is a bit complex in MuJoCo raw.
                # Simplified: If we are stuck or very close to wall?
                # Better: Use sensor data. If distance < threshold?
                pass

        self.current_step += 1
        
        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        collision = False
        
        # Check collision using sensors (if too close)
        sensor_data = self.data.sensordata[:4]
        
        # Wall surface is at dist X. Sensor measures dist to surface.
        # If sensor reads < 0.01 (1cm), it's a crash.
        valid_sensor_data = sensor_data[sensor_data > 0]
        if len(valid_sensor_data) > 0 and np.min(valid_sensor_data) < 0.01:
             collision = True
        
        # Check Tilt (Tip-over)
        # q[3:7] is quaternion [w, x, y, z]
        q = self.data.qpos[3:7]
        # Calculate Z component of the local Z-axis in world coordinates
        # z_axis_z = 1 - 2(x^2 + y^2)
        z_axis_z = 1.0 - 2.0 * (q[1]*q[1] + q[2]*q[2])
        
        is_tipped = z_axis_z < 0.5 # If tilted more than ~60 degrees
        
        # --- Calculate Reward & Check Termination ---

        # Current state
        current_pos = self.data.qpos[0:2]
        dist_to_goal = np.linalg.norm(current_pos - self.goal_pos)

        # Reward Components (Improved Design)
        r_goal = 0.0
        r_goal_orientation = 0.0
        r_intermediate = 0.0
        r_collision = 0.0
        r_time = -1.0  # Stronger time penalty to encourage efficiency

        # Calculate current grid cell
        ix = int(current_pos[0] / 0.18)
        iy = int(current_pos[1] / 0.18)
        current_cell = (ix, iy)

        # Check if intermediate cell reached (only once)
        if (self.intermediate_cell is not None and
            not self.intermediate_reached and
            current_cell == self.intermediate_cell):
            r_intermediate = 100.0
            self.intermediate_reached = True
            print(f"Intermediate cell reached: {self.intermediate_cell}")

        # Goal Reached
        if dist_to_goal < self.goal_threshold:
            r_goal = 1000.0  # Large reward for reaching goal

            # Check orientation alignment (bonus for proper final pose)
            q = self.data.qpos[3:7]
            w, x, y, z = q
            yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            yaw_deg = np.degrees(yaw)

            # Check if aligned to cardinal direction (±15°)
            cardinal_angles = [0, 90, 180, -90]  # East, North, West, South
            is_aligned = False
            for cardinal in cardinal_angles:
                if abs(yaw_deg - cardinal) <= 15.0:
                    is_aligned = True
                    break

            if is_aligned:
                r_goal_orientation = 200.0  # Orientation bonus
                print(f"Goal Reached with proper orientation ({yaw_deg:.1f}°)!")
            else:
                print(f"Goal Reached but poor orientation ({yaw_deg:.1f}°)")

            terminated = True

        # Collision Penalty (discourage wall bumping)
        if collision:
            r_collision = -10.0
            # Do NOT terminate on simple collision

        # Tip-over (Hard Termination)
        if is_tipped:
            r_collision = -100.0  # 転倒は復帰不能なので大きなペナルティで終了
            terminated = True
            # print("Robot Tipped Over!")

        # REMOVED: Exploration Reward (防止報酬ハッキング)
        # 探索報酬を削除することで、その場で回転して新しいセルを訪れる戦略を防ぐ

        reward = r_goal + r_goal_orientation + r_intermediate + r_collision + r_time
        
        if self.current_step >= self.max_steps:
            truncated = True
            
        # Get new observation for High-Level
        obs = self._get_obs()
        
        # Calculate actual velocities for info
        wheel_radius = 0.0135
        left_vel = self.data.qvel[6] * wheel_radius
        right_vel = self.data.qvel[7] * wheel_radius
        actual_v = (left_vel + right_vel) / 2.0
        actual_omega = self.data.sensordata[9] # Gyro Z

        info = {
            "dist_to_goal": dist_to_goal,
            "target_lin": target_v,
            "target_ang": target_omega,
            "actual_lin": actual_v,
            "actual_ang": actual_omega
        }
        
        if self.render_mode == "human":
            self._render_frame()
            
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        data = self.data.sensordata
        
        # 1. Distance Sensors
        distances = data[:4].copy()
        distances[distances < 0] = 0.15 # Max range
        
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
        dx = self.goal_pos[0] - current_pos[0]
        dy = self.goal_pos[1] - current_pos[1]
        angle_to_goal = np.arctan2(dy, dx)
        
        # Robot heading (Yaw)
        q = self.data.qpos[3:7]
        w, x, y, z = q
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        
        # Relative angle
        rel_angle = angle_to_goal - yaw
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
        
        obs = np.concatenate([
            distances,
            [linear_velocity],
            [angular_velocity],
            [dist_to_goal],
            [rel_angle]
        ]).astype(np.float32)
        
        return obs

    def _get_low_level_obs(self, target_v, target_omega):
        # Construct observation for Low-Level Controller
        # [LF, LS, RF, RS, v_act, w_act, lat_acc, v_ref, w_ref]
        
        data = self.data.sensordata
        
        # HACK: Overwrite sensors to "Max Range" (0.15) for Low-Level Controller
        # The Low-Level Controller (Phase 1) was trained in an Open Field and might panic
        # if it sees walls too close (which is common in a Maze).
        # We want the Low-Level to just track velocity, ignoring walls.
        distances = np.full(4, 0.15, dtype=np.float32)
        
        wheel_radius = 0.0135
        left_vel = self.data.qvel[6] * wheel_radius
        right_vel = self.data.qvel[7] * wheel_radius
        linear_velocity = (left_vel + right_vel) / 2.0
        angular_velocity = data[9]
        lateral_accel = data[5]
        
        obs = np.concatenate([
            distances,
            [linear_velocity],
            [angular_velocity],
            [lateral_accel],
            [target_v],
            [target_omega]
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
