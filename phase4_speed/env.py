"""
Phase 4: Speed Learning Environment
センサーパターンから最適な速度を学習する環境
Phase3をベースに、速度最大化を目標とした報酬設計
"""
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os
from stable_baselines3 import PPO
import sys

# ワークスペースルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase3_maze.maze_generator import RandomMazeGenerator


class MicromouseSpeedEnv(gym.Env):
    """
    Phase4: 高速走行学習環境

    目標: センサーパターンから安全かつ高速な速度を学習
    - 直進路では加速
    - ターン前では減速
    - 衝突を避けながら最速でゴールに到達
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        maze_size=(7, 7),
        model_path="models/phase1_open.zip",
        max_steps=4000,
        maze_regen_interval=20,
        spawn_area_size=5
    ):
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.maze_width, self.maze_height = maze_size

        # スポーンエリア設定
        self.spawn_area_size = spawn_area_size
        self.spawn_offset = (self.maze_width - self.spawn_area_size) // 2

        # 迷路再生成の制御
        self.maze_regen_interval = maze_regen_interval
        self.episode_count = 0

        # 迷路ジェネレータ
        self.generator = RandomMazeGenerator(self.maze_width, self.maze_height)
        self.xml_file = f"assets/micromouse_random_{self.maze_width}x{self.maze_height}.xml"

        # 初期迷路を生成
        self.generator.generate_maze()
        self.generator.generate_mjcf(self.xml_file)

        # MuJoCoモデルをロード
        self.model = mujoco.MjModel.from_xml_path(self.xml_file)
        self.data = mujoco.MjData(self.model)

        # 低レベルコントローラ（Phase1）をロード
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file {model_path} not found. "
                "Please ensure Phase 1 training is complete."
            )

        print(f"Loading Low-Level Controller from {model_path}...")
        self.low_level_model = PPO.load(model_path)
        print("Low-Level Controller loaded.")

        # 行動空間: 目標速度 [v (m/s), omega (rad/s)]
        self.action_space = spaces.Box(
            low=np.array([0.0, -5.0]),
            high=np.array([1.0, 5.0]),
            dtype=np.float32
        )

        # 観測空間（10次元）:
        # 0-3: 距離センサー (LF, LS, RF, RS)
        # 4: 線速度
        # 5: 角速度
        # 6: ゴール距離
        # 7: ゴール角度
        # 8: 前方開放度（直進可能性の指標）
        # 9: 左右バランス（ターン方向の指標）
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        self.viewer = None
        self.renderer = None

        # ゴール位置（右上セルの中央）
        self.goal_pos = np.array([
            (self.maze_width - 1) * 0.18 + 0.09,
            (self.maze_height - 1) * 0.18 + 0.09
        ])
        self.goal_threshold = 0.05  # 5cm半径

        self.current_step = 0
        self.last_distance = 0.0

        # 前ステップの速度（スムーズネス報酬用）
        self.prev_linear_vel = 0.0
        self.prev_angular_vel = 0.0

        # 低レベル制御周波数設定
        self.low_level_steps = 1

        # レンダリング用
        self.render_fps = 60

    def _get_adjacent_cells(self, cell):
        """隣接する通行可能なセルを取得"""
        x, y = cell
        adjacent = []

        # 北: y+1
        if y + 1 < self.maze_height and self.generator.h_walls[x, y+1] == 0:
            adjacent.append((x, y+1))

        # 南: y-1
        if y - 1 >= 0 and self.generator.h_walls[x, y] == 0:
            adjacent.append((x, y-1))

        # 東: x+1
        if x + 1 < self.maze_width and self.generator.v_walls[x+1, y] == 0:
            adjacent.append((x+1, y))

        # 西: x-1
        if x - 1 >= 0 and self.generator.v_walls[x, y] == 0:
            adjacent.append((x-1, y))

        return adjacent

    def _find_two_step_cells(self, start_cell):
        """開始セルから2ステップ先のセルを探索"""
        two_step_cells = []

        one_step_neighbors = self._get_adjacent_cells(start_cell)

        for intermediate in one_step_neighbors:
            two_step_neighbors = self._get_adjacent_cells(intermediate)

            for goal in two_step_neighbors:
                if goal != start_cell and goal not in one_step_neighbors:
                    two_step_cells.append((goal, intermediate))

        return two_step_cells

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_count += 1

        # 前ステップの速度をリセット
        self.prev_linear_vel = 0.0
        self.prev_angular_vel = 0.0

        # 迷路を定期的に再生成
        if self.episode_count % self.maze_regen_interval == 1:
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

        # 2ステップ経路がある開始セルを探す
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

            np.random.shuffle(spawn_cells)
            for candidate_cell in spawn_cells:
                two_step_options = self._find_two_step_cells(candidate_cell)
                if len(two_step_options) > 0:
                    start_cell = candidate_cell
                    break

            if start_cell is None:
                center = self.maze_width // 2
                start_cell = (center, center)

        # ゴールセルをランダムに選択
        if len(two_step_options) > 0:
            goal_cell, intermediate_cell = two_step_options[
                np.random.randint(len(two_step_options))
            ]
        else:
            one_step_neighbors = self._get_adjacent_cells(start_cell)
            if len(one_step_neighbors) > 0:
                goal_cell = one_step_neighbors[0]
                intermediate_cell = None
            else:
                goal_cell = start_cell
                intermediate_cell = None

        self.start_cell = start_cell
        self.intermediate_cell = intermediate_cell
        self.goal_cell = goal_cell

        start_x, start_y = start_cell

        # ロボットの初期位置を設定
        self.data.qpos[0] = start_x * 0.18 + 0.09
        self.data.qpos[1] = start_y * 0.18 + 0.09
        self.data.qpos[2] = 0.005  # 床貫通防止のため5mm浮かせる

        # ランダムな初期向き
        orientations = [0, 90, 180, 270]
        angle_deg = np.random.choice(orientations)
        angle = np.radians(angle_deg)

        self.data.qpos[3] = np.cos(angle/2)
        self.data.qpos[4] = 0
        self.data.qpos[5] = 0
        self.data.qpos[6] = np.sin(angle/2)

        # ゴール位置を設定
        self.goal_pos = np.array([
            goal_cell[0] * 0.18 + 0.09,
            goal_cell[1] * 0.18 + 0.09
        ])

        self.last_distance = np.linalg.norm(self.data.qpos[0:2] - self.goal_pos)

        # 訪問セルの追跡
        self.visited_cells = set()
        self.visited_cells.add(start_cell)

        # 中間セル到達フラグ
        self.intermediate_reached = False

        direction_names = {0: "East", 90: "North", 180: "West", 270: "South"}
        direction_name = direction_names[angle_deg]

        print(f"Episode {self.episode_count}: Start={start_cell} ({direction_name}), "
              f"Goal={goal_cell}, Intermediate={intermediate_cell}")

        obs = self._get_obs()

        return obs, {}

    def step(self, action):
        # 高レベルアクション: 目標速度 [v_ref, omega_ref]
        target_v = float(action[0])
        target_omega = float(action[1])

        # 低レベル観測を構築
        ll_obs = self._get_low_level_obs(target_v, target_omega)

        # 低レベルコントローラで予測
        ll_action, _ = self.low_level_model.predict(ll_obs, deterministic=True)

        # MuJoCoに適用
        self.data.ctrl[0] = ll_action[0] * 3.0
        self.data.ctrl[1] = ll_action[1] * 3.0

        # シミュレーション実行
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1

        # === 終了条件のチェック ===
        terminated = False
        truncated = False
        collision = False

        # センサーで衝突チェック
        sensor_data = self.data.sensordata[:4]
        valid_sensor_data = sensor_data[sensor_data > 0]
        if len(valid_sensor_data) > 0 and np.min(valid_sensor_data) < 0.01:
            collision = True

        # 転倒チェック
        q = self.data.qpos[3:7]
        z_axis_z = 1.0 - 2.0 * (q[1]*q[1] + q[2]*q[2])
        is_tipped = z_axis_z < 0.5

        # === 報酬計算 ===
        current_pos = self.data.qpos[0:2]
        dist_to_goal = np.linalg.norm(current_pos - self.goal_pos)

        # センサー距離
        sensor_distances = self.data.sensordata[:4].copy()
        sensor_distances[sensor_distances < 0] = 0.15
        min_wall_dist = np.min(sensor_distances)

        # 現在の速度
        wheel_radius = 0.0135
        left_vel = self.data.qvel[6] * wheel_radius
        right_vel = self.data.qvel[7] * wheel_radius
        linear_vel = (left_vel + right_vel) / 2.0
        angular_vel = self.data.sensordata[9]

        # 報酬要素の初期化
        r_goal = 0.0
        r_intermediate = 0.0
        r_speed = 0.0
        r_time = -0.5  # 時間ペナルティ（強化）
        r_collision = 0.0
        r_smoothness = 0.0
        r_safety = 0.0
        r_straight = 0.0  # 直進ボーナス
        r_angular = 0.0   # 角速度ペナルティ

        # グリッドセルを計算
        ix = int(current_pos[0] / 0.18)
        iy = int(current_pos[1] / 0.18)
        current_cell = (ix, iy)

        # 中間セル到達チェック
        if (self.intermediate_cell is not None and
            not self.intermediate_reached and
            current_cell == self.intermediate_cell):
            r_intermediate = 100.0
            self.intermediate_reached = True
            print(f"Intermediate cell reached: {self.intermediate_cell}")

        # ゴール到達
        if dist_to_goal < self.goal_threshold:
            r_goal = 500.0
            print(f"Goal Reached! Steps: {self.current_step}")
            terminated = True

        # 速度報酬（メイン報酬）- 高速走行を促進
        if linear_vel > 0:
            r_speed = 3.0 * linear_vel  # 最大3.0/step (1.0m/sの場合)

        # 安全報酬（壁接近ペナルティ）
        if min_wall_dist < 0.02:  # 2cm以下: 危険
            r_safety = -5.0
        elif min_wall_dist < 0.04:  # 4cm以下: 警告
            r_safety = -2.0 * (0.04 - min_wall_dist) / 0.02

        # スムーズネス報酬（急な速度変化にペナルティ）- 穏やかな調整
        v_change = abs(linear_vel - self.prev_linear_vel)
        omega_change = abs(angular_vel - self.prev_angular_vel)
        r_smoothness = -0.3 * v_change - 0.3 * omega_change  # v2より緩やかなペナルティ

        # 角速度ペナルティ（不必要な旋回を抑制）- 穏やかに調整
        r_angular = -0.1 * abs(angular_vel)  # v2(-0.3)より緩やか

        # 直進ボーナス（前方が開いていて、角速度が小さい場合）
        front_openness = (sensor_distances[0] + sensor_distances[2]) / 2.0  # 前方センサー平均
        if front_openness > 0.08 and abs(angular_vel) < 1.0:  # 閾値を0.5→1.0に緩和
            r_straight = 0.3  # ボーナスも0.5→0.3に軽減

        # 前ステップの速度を保存
        self.prev_linear_vel = linear_vel
        self.prev_angular_vel = angular_vel

        # 衝突ペナルティ
        if collision:
            r_collision = -100.0
            terminated = True

        # 転倒ペナルティ
        if is_tipped:
            r_collision = -100.0
            terminated = True

        # 総報酬
        reward = (r_goal + r_intermediate + r_speed + r_time +
                  r_collision + r_smoothness + r_safety +
                  r_straight + r_angular)

        if self.current_step >= self.max_steps:
            truncated = True

        # 観測を取得
        obs = self._get_obs()

        info = {
            "dist_to_goal": dist_to_goal,
            "target_lin": target_v,
            "target_ang": target_omega,
            "actual_lin": linear_vel,
            "actual_ang": angular_vel,
            "r_speed": r_speed,
            "r_time": r_time,
            "r_safety": r_safety,
            "r_smoothness": r_smoothness,
            "r_straight": r_straight,
            "r_angular": r_angular
        }

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """観測を取得（10次元）"""
        data = self.data.sensordata

        # 1. 距離センサー
        distances = data[:4].copy()
        distances[distances < 0] = 0.15  # 最大距離

        # 2. 速度
        wheel_radius = 0.0135
        left_vel = self.data.qvel[6] * wheel_radius
        right_vel = self.data.qvel[7] * wheel_radius
        linear_velocity = (left_vel + right_vel) / 2.0
        angular_velocity = data[9]

        # 3. ゴール情報
        current_pos = self.data.qpos[0:2]
        dist_to_goal = np.linalg.norm(self.goal_pos - current_pos)

        dx = self.goal_pos[0] - current_pos[0]
        dy = self.goal_pos[1] - current_pos[1]
        angle_to_goal = np.arctan2(dy, dx)

        q = self.data.qpos[3:7]
        w, x, y, z = q
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

        rel_angle = angle_to_goal - yaw
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi

        # 4. 追加特徴量
        # 前方開放度: 前方センサーの平均（大きいほど直進可能）
        front_openness = (distances[0] + distances[2]) / 0.30  # 正規化

        # 左右バランス: サイドセンサーの差（ターン方向の指標）
        side_balance = (distances[1] - distances[3]) / 0.15  # 正規化

        obs = np.concatenate([
            distances,                # 4次元
            [linear_velocity],        # 1次元
            [angular_velocity],       # 1次元
            [dist_to_goal],           # 1次元
            [rel_angle],              # 1次元
            [front_openness],         # 1次元（追加）
            [side_balance]            # 1次元（追加）
        ]).astype(np.float32)

        return obs

    def _get_low_level_obs(self, target_v, target_omega):
        """低レベルコントローラ用の観測を構築"""
        data = self.data.sensordata

        # 低レベルコントローラには壁を「見せない」（最大距離を設定）
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
