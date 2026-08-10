"""
mouse/corridor_env.py
================
M1（廊下追従・exp_001_corridor）用 Gymnasium 環境。分岐のない自己回避パス
コース（mouse.corridor_gen）上で、壁に当たらず前進する方策を学習させる。

観測は (距離センサ本数 + 7) 次元（距離 ×n・ジャイロz・加速度xy・車輪角速度2・前ステップ行動2）で、
生値は MouseSim.observation()（凍結、14次元）から抽出・正規化する。
報酬はポテンシャルベース整形 r = γΦ(s')−Φ(s)−0.001（Φ=−残り経路長）+ゴール到達+1.0。
詳細は spec_m1_corridor.md §2 を参照。

--------------------------------------------------------------------------
2 つの動作モード（2026-08-10 指揮側追加指示: 固定200本プールへの過学習防止）
--------------------------------------------------------------------------
- mode="fixed"（既定）: course_dir 内の npz+xml（mouse.corridor_gen.save_course で
  事前生成済み）を対象に、reset ごとに1本ランダム選択する。course_seeds を渡すと
  候補をその集合に絞れる（評価器はコースseedを1本に固定するのに使う。§3 参照）。
  MjModel は LRU キャッシュ（max_cache）で保持し、毎 reset の XML パースを避ける。
  評価用コース（assets/corridor/eval、seed 3000-3019）はこのモードで使う。
- mode="generate": 毎エピソード mouse.corridor_gen.generate_course(seed) で
  コースを手続き的に新規生成し、一時ファイルに XML を書き出して MjModel を
  ロードした後、一時ファイルは即座に削除する（mouse/mjcf.py 自体は変更せず、
  build_maze_robot_xml の出力先を一時パスにするだけ）。
  seed = base_seed + このインスタンスの通算エピソード番号。course_dir/course_seeds/
  max_cache は無視される。訓練用途（本モード）は固定プールを持たないため、コース
  形状への過学習を避けられる一方、reset のたびに生成+XMLコンパイルのコストが乗る
  （train.py の smoke 計測で影響を報告する）。複数ワーカで使う場合は、コースseedが
  ワーカ間で重複しないよう base_seed をワーカごとにずらすのは呼び出し側の責務
  （例: base_seed = 2000 + worker_index * 1_000_000）。
"""
import math
import os
import tempfile
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from mouse.corridor_gen import (
    generate_course,
    initial_heading_deg,
    path_cell_centers,
    path_cumulative_lengths,
    remaining_path_length,
)
from mouse.mjcf import build_maze_robot_xml
from mouse.params import RobotParams
from mouse.sim import MouseSim

# 観測次元 = 距離センサ本数 n + ジャイロ z(1) + 加速度 xy(2) + 車輪角速度(2) + 前回行動(2)
# n はモデルから導出するため定数では持たない（計画書 r6: センサ 6→4 本）。
_OBS_EXTRA_DIM = 7

# 観測正規化定数（spec_m1_corridor.md §2。固定値、VecNormalize は使わない）
_DIST_SCALE = 0.3      # センサ cutoff [m]
_GYRO_SCALE = 10.0
_ACCEL_SCALE = 10.0
_WHEEL_SCALE = 300.0   # 無負荷上限 286.8 rad/s

_TIME_PENALTY = 0.001
_GOAL_BONUS = 1.0
_GOAL_RADIUS_M = 0.05

# 上限ステップ = ceil(3.72 * n_cells / control_dt)（L0実測1.24 s/セルの3倍）
_TIME_LIMIT_SEC_PER_CELL = 3.72

# reset の初期擾乱（進行方向に直交する向きへの位置擾乱・方位擾乱）
_LATERAL_PERTURB_M = 0.02
_HEADING_PERTURB_DEG = 10.0


class CorridorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, course_dir=None, course_seeds=None, max_cache=32, seed=None,
                 gamma: float = 0.995, mode: str = "fixed", base_seed: int = 2000):
        super().__init__()
        if mode not in ("fixed", "generate"):
            raise ValueError(f"mode は 'fixed' か 'generate' のみ対応: {mode!r}")
        if mode == "fixed" and course_dir is None:
            raise ValueError("mode='fixed' には course_dir が必須です")

        self.mode = mode
        self.gamma = float(gamma)
        self.params = RobotParams()
        self.max_cache = int(max_cache)
        # 距離センサ本数は仕様（RobotParams.sensors）から取る。生成される XML の
        # rangefinder 数と一致することは検証スイート S3 が保証している
        self._n_dist = len(self.params.sensors)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._n_dist + _OBS_EXTRA_DIM,), dtype=np.float32)

        # mode="fixed" 用状態
        self._sim_cache = {}
        self._cache_order = []
        self._courses = None
        self.course_dir = None

        # mode="generate" 用状態
        self.base_seed = int(base_seed)
        self._episode_count = 0

        if mode == "fixed":
            self.course_dir = Path(course_dir)
            self._courses = self._load_course_index(self.course_dir, course_seeds)
            if not self._courses:
                raise ValueError(f"コースが見つかりません: {self.course_dir}")

        self.sim = None
        self.course = None
        self._path_centers = None
        self._cum_lengths = None
        self._prev_potential = None
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._max_steps = 0

        if seed is not None:
            # 構築時に np_random をシードしておく。gym.Env の既定 reset() 実装を
            # 直接呼ぶことで seeding のみ行い、コース読込を伴う本クラスの reset()
            # 全体（mode="generate" ならエピソードカウンタも進めてしまう）は
            # ここでは実行しない。
            gym.Env.reset(self, seed=seed)

    # ------------------------------------------------------------------
    # コース読み込み（mode="fixed"）
    # ------------------------------------------------------------------
    def _load_course_index(self, course_dir: Path, course_seeds):
        npz_paths = sorted(course_dir.glob("corridor_*.npz"))
        allowed = set(int(s) for s in course_seeds) if course_seeds is not None else None
        courses = []
        for p in npz_paths:
            data = np.load(p)
            seed = int(data["seed"])
            if allowed is not None and seed not in allowed:
                continue
            xml_path = p.with_suffix(".xml")
            courses.append(dict(
                seed=seed, npz_path=p, xml_path=xml_path,
                v_walls=data["v_walls"], h_walls=data["h_walls"],
                path=[tuple(int(v) for v in c) for c in data["path"].tolist()],
                width=int(data["width"]), height=int(data["height"]),
                n_cells=int(data["n_cells"]), n_turns=int(data["n_turns"]),
            ))
        return courses

    def _get_sim(self, xml_path: str) -> MouseSim:
        key = str(xml_path)
        if key in self._sim_cache:
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return self._sim_cache[key]

        sim = MouseSim(key, params=self.params)
        self._sim_cache[key] = sim
        self._cache_order.append(key)
        if len(self._cache_order) > self.max_cache:
            oldest = self._cache_order.pop(0)
            del self._sim_cache[oldest]
        return sim

    # ------------------------------------------------------------------
    # コース手続き生成（mode="generate"）
    # ------------------------------------------------------------------
    def _generate_course_and_load(self, course_seed: int) -> dict:
        raw = generate_course(course_seed)
        path = raw["path"]
        cell_size = self.params.cell_size
        sx, sy = path[0]
        mouse_pos = f"{sx * cell_size + cell_size / 2} {sy * cell_size + cell_size / 2} 0.002"
        heading = initial_heading_deg(path)
        mouse_euler = f"0 0 {heading}"

        fd, tmp_path = tempfile.mkstemp(suffix=".xml", prefix=f"corridor_gen_{course_seed}_")
        os.close(fd)
        try:
            build_maze_robot_xml(
                raw["v_walls"], raw["h_walls"], tmp_path,
                model_name=f"corridor_gen_{course_seed}",
                mouse_pos=mouse_pos, mouse_euler=mouse_euler,
                center_goal=False, params=self.params,
            )
            self.sim = MouseSim(tmp_path, params=self.params)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return dict(
            seed=course_seed, npz_path=None, xml_path=None,
            v_walls=raw["v_walls"], h_walls=raw["h_walls"], path=path,
            width=raw["width"], height=raw["height"],
            n_cells=raw["n_cells"], n_turns=raw["n_turns"],
        )

    # ------------------------------------------------------------------
    # 初期姿勢（スタートセル中心 + 進行方向に直交する擾乱）
    # ------------------------------------------------------------------
    def _apply_initial_disturbance(self, start_cell, heading_deg: float,
                                    lateral_m: float, heading_offset_deg: float):
        cx, cy = start_cell
        cell_size = self.params.cell_size
        cx_m = cx * cell_size + cell_size / 2
        cy_m = cy * cell_size + cell_size / 2

        heading_rad = math.radians(heading_deg)
        # 進行方向に直交する単位ベクトル（進行方向を90°反時計回りしたもの）
        perp_x = -math.sin(heading_rad)
        perp_y = math.cos(heading_rad)

        x = cx_m + lateral_m * perp_x
        y = cy_m + lateral_m * perp_y
        new_heading_rad = math.radians(heading_deg + heading_offset_deg)

        sim = self.sim
        # MouseSim は root joint の qpos アドレスを private 属性でしかキャッシュして
        # いない（mouse/sim.py は凍結・変更禁止）ため、公開 API (mj_name2id) で
        # 独自に解決する（sim._root_joint_id 等の private 属性には触れない）。
        root_jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, 'root')
        qpos_adr = sim.model.jnt_qposadr[root_jid]
        sim.data.qpos[qpos_adr] = x
        sim.data.qpos[qpos_adr + 1] = y
        sim.data.qpos[qpos_adr + 3] = math.cos(new_heading_rad / 2.0)
        sim.data.qpos[qpos_adr + 4] = 0.0
        sim.data.qpos[qpos_adr + 5] = 0.0
        sim.data.qpos[qpos_adr + 6] = math.sin(new_heading_rad / 2.0)
        mujoco.mj_forward(sim.model, sim.data)

    # ------------------------------------------------------------------
    # 観測・情報
    # ------------------------------------------------------------------
    def _make_observation(self) -> np.ndarray:
        raw = self.sim.observation()
        # raw(6+n) = [距離 ×n(0:n), ax,ay,az(n:n+3), gx,gy,gz(n+3:n+6), wL,wR(末尾2)]
        # 距離センサ本数 n はモデルから導出される（計画書 r6 でセンサ 6→4 本に変更
        # されたため、切り出し位置を固定値で持たない）
        n = self._n_dist
        dist = raw[0:n] / _DIST_SCALE
        gyro_z = raw[n + 5] / _GYRO_SCALE
        accel_xy = raw[n:n + 2] / _ACCEL_SCALE
        wheels = raw[n + 6:n + 8] / _WHEEL_SCALE
        obs = np.concatenate([dist, [gyro_z], accel_xy, wheels, self._prev_action])
        return obs.astype(np.float32)

    def _make_info(self, remaining_m: float, collision: bool, goal: bool, sim_time: float) -> dict:
        total = self._cum_lengths[-1]
        progress_m = total - remaining_m
        mean_speed = (progress_m / sim_time) if sim_time > 0 else 0.0
        return {
            "course_seed": self.course["seed"],
            "n_cells": self.course["n_cells"],
            "progress_m": float(progress_m),
            "remaining_m": float(remaining_m),
            "collision": bool(collision),
            "goal": bool(goal),
            "sim_time": float(sim_time),
            "mean_speed": float(mean_speed),
        }

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.mode == "fixed":
            idx = int(self.np_random.integers(0, len(self._courses)))
            course = self._courses[idx]
            self.sim = self._get_sim(str(course["xml_path"]))
        else:
            course_seed = self.base_seed + self._episode_count
            self._episode_count += 1
            course = self._generate_course_and_load(course_seed)

        self.course = course
        path = course["path"]
        cell_size = self.params.cell_size
        self._path_centers = path_cell_centers(path, cell_size)
        self._cum_lengths = path_cumulative_lengths(self._path_centers)

        start_cell = path[0]
        heading_deg = initial_heading_deg(path)
        self.sim.full_reset(cell=start_cell, heading_deg=heading_deg)

        lateral = float(self.np_random.uniform(-_LATERAL_PERTURB_M, _LATERAL_PERTURB_M))
        heading_offset = float(self.np_random.uniform(-_HEADING_PERTURB_DEG, _HEADING_PERTURB_DEG))
        self._apply_initial_disturbance(start_cell, heading_deg, lateral, heading_offset)

        self._step_count = 0
        self._max_steps = math.ceil(
            _TIME_LIMIT_SEC_PER_CELL * course["n_cells"] / self.params.control_dt)
        self._prev_action = np.zeros(2, dtype=np.float32)

        x, y, _yaw = self.sim.privileged_pose()
        remaining0 = remaining_path_length(self._path_centers, self._cum_lengths, x, y)
        self._prev_potential = -remaining0

        obs = self._make_observation()
        info = self._make_info(remaining0, collision=False, goal=False, sim_time=self.sim.sim_time)
        return obs, info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        v_left = float(action[0]) * self.params.voltage_limit
        v_right = float(action[1]) * self.params.voltage_limit

        result = self.sim.step_control(v_left, v_right)
        self._step_count += 1

        x, y, _yaw = self.sim.privileged_pose()
        remaining_m = remaining_path_length(self._path_centers, self._cum_lengths, x, y)
        potential = -remaining_m

        goal_cell = self.course["path"][-1]
        cell_size = self.params.cell_size
        gx = goal_cell[0] * cell_size + cell_size / 2
        gy = goal_cell[1] * cell_size + cell_size / 2
        goal_reached = math.hypot(x - gx, y - gy) <= _GOAL_RADIUS_M

        # 「壁接触」は物理接触(collision)・転倒(tipped)のいずれも即終了扱いとする
        # （spec_m1_corridor.md §2「壁接触: collision または tipped」）。info["collision"]
        # は spec が列挙する info キーに tipped が無いため、両者をまとめて表現する
        # （corridor_eval.py の失敗様式分類が collision/timeout の2区分のため）。
        physical_fail = bool(result["collision"] or result["tipped"])

        reward = self.gamma * potential - self._prev_potential - _TIME_PENALTY
        if goal_reached:
            reward += _GOAL_BONUS
        self._prev_potential = potential

        terminated = bool(goal_reached or physical_fail)
        truncated = bool((not terminated) and self._step_count >= self._max_steps)

        self._prev_action = np.asarray(action, dtype=np.float32)
        obs = self._make_observation()
        info = self._make_info(remaining_m, physical_fail, goal_reached, result["sim_time"])

        return obs, float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        self._sim_cache.clear()
        self._cache_order.clear()
