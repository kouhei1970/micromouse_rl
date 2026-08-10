"""
mouse/maze6_env.py
================
M2（6x6 迷路の単走）用 Gymnasium 環境。

M1（`mouse/corridor_env.py`、分岐なしの 1 本道）との違い:
  - 分岐がある。方策は「どちらへ曲がるか」を決めなければならない
  - ゴールは中央 2x2 の広場。**距離センサの瞬時値では原理的に認識できない**
    （`research_notes/scripts/check_goal_recognizability.py` で確認済み: ゴール姿勢の
    97.5〜100% が非ゴール姿勢と ±5 mm 以内で一致する）。したがって方策は
    **オドメトリで自己位置を推定し、規約既知のゴール位置（中央）と照合する**必要がある

--------------------------------------------------------------------------
特権情報の線引き（2026-08-10 教授裁定）
--------------------------------------------------------------------------
- **方策に与えてよい**: 「ゴールは迷路の中央 2x2 にある」という**競技規約の知識**。
  実機のマウサーも規定からこれを知っている
- **方策に与えてはならない**: 「いま自分がどこにいるか」の**真値**。方策は
  **自前センサ（車輪角速度・ジャイロ）の積分だけ**から推定する。本環境の
  オドメトリ積分は sim.observation() の値のみを使い、privileged_pose() は
  報酬計算・終了判定・評価にしか使わない（報酬は学習時にしか計算されず、
  方策の入力ではないので入出力契約に触れない）
- **訪問済みビットは渡さない**（環境が真の位置から計算したものになるため）。
  方策が自分の推定位置から履歴を組み立てるのは可

観測（17 次元）:
  距離 4 (/0.3 m) ・距離の 1 階差分 4 (/0.05 m, クリップ)
  ・ジャイロ z (/10) ・加速度 xy (/10) ・車輪角速度 2 (/300) ・前ステップ行動 2
  ・**機体座標系で見たゴールへの推定相対位置 2**（自己位置の推定値から計算。/1.53 m）

報酬（exp_005 で確立した構成を踏襲。`experiments/m2_design.md` §2）:
  r = γΦ(s') − Φ(s) − 0.001,  Φ = D₀ − d(現在区画)
      + 1.0   ゴール到達
      − 1.0   衝突・転倒
      + 0.02  未訪問の区画へ初めて入ったとき（1 区画 1 回のみ）
  d はゴールまでの迷路距離 [m]（幅優先で計算。学習時の報酬にのみ使う）。
  D₀ はスタート区画の d（エピソード内で定数）なので Φ ≥ 0 になり、滞留は必ず損。

  実装前の検算（D₀ = 1.8 m・速度 0.96 m/s・上限 6000 ステップ）:
    最短でゴール +0.975 > 遠回りしてゴール +0.340 > 探索だけで時間切れ +0.160
    > 半分進んで衝突 −0.137 > その場に留まる −0.200
  訪問報酬なし（r_v=0）だと「遠回りしてゴール」が −0.020 とほぼゼロになり学習信号が
  弱すぎる。r_v=0.05 まで上げると探索（+0.700）がゴール（+0.975）に迫って危ない。
  **r_v=0.02** は 36 区画ぶんの総和 0.72 がゴールボーナス 1.0 を下回り、順序が保たれる。
"""
import math
import os
import tempfile
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from mouse.maze6_gen import (
    GOAL_CELLS, SIZE, generate_maze, initial_heading_deg, shortest_distances,
)
from mouse.mjcf import build_maze_robot_xml
from mouse.params import RobotParams
from mouse.sim import MouseSim

# 観測の正規化定数（M1 と揃える）
_DIST_SCALE = 0.3
_DIST_DIFF_SCALE = 0.05
_GYRO_SCALE = 10.0
_ACCEL_SCALE = 10.0
_WHEEL_SCALE = 300.0
# ゴールへの相対位置の正規化: 6x6 の対角長 6·0.18·√2 ≈ 1.53 m
_REL_SCALE = SIZE * 0.18 * math.sqrt(2.0)

_TIME_PENALTY = 0.001
_GOAL_BONUS = 1.0
_COLLISION_PENALTY = -1.0
_VISIT_BONUS = 0.02          # 未訪問区画への初回進入（1 区画 1 回のみ）
_TIME_LIMIT_STEPS = 6000     # 60 秒（単走。競技の持ち時間 420 秒とは別物）

# 予約 seed（研究計画書 §9-7 の三分割）。学習は 8000 以降を使う。
_RESERVED_MAZE_SEEDS = frozenset(range(6000, 6020)) | frozenset(range(7000, 7020))

_LATERAL_PERTURB_M = 0.02
_HEADING_PERTURB_DEG = 10.0


class Maze6Env(gym.Env):
    """6x6 迷路の単走環境。mode='loop'（M2-0）/ 'full'（M2-1）。"""

    metadata = {"render_modes": []}

    def __init__(self, maze_dir=None, maze_seeds=None, max_cache=8, seed=None,
                 gamma: float = 0.995, mode: str = "fixed", base_seed: int = 8000,
                 maze_mode: str = "loop", visit_bonus: float = _VISIT_BONUS,
                 collision_penalty: float = _COLLISION_PENALTY,
                 action_smooth_penalty: float = 0.0):
        super().__init__()
        if mode not in ("fixed", "generate"):
            raise ValueError(f"mode は 'fixed' か 'generate': {mode!r}")
        if mode == "fixed" and maze_dir is None:
            raise ValueError("mode='fixed' には maze_dir が必須です")

        self.mode = mode
        self.maze_mode = maze_mode
        self.gamma = float(gamma)
        self.params = RobotParams()
        self.max_cache = int(max_cache)
        self.visit_bonus = float(visit_bonus)
        self.collision_penalty = float(collision_penalty)
        self.action_smooth_penalty = float(action_smooth_penalty)
        self._n_dist = len(self.params.sensors)

        # 距離 n + 差分 n + ジャイロ1 + 加速度2 + 車輪2 + 前回行動2 + ゴール相対2
        n_obs = 2 * self._n_dist + 9
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(n_obs,), dtype=np.float32)

        self._sim_cache, self._cache_order = {}, []
        self.maze_dir = Path(maze_dir) if maze_dir is not None else None
        self._maze_seeds = list(maze_seeds) if maze_seeds is not None else None
        self.base_seed = int(base_seed)
        self._episode_count = 0

        self.sim = None
        self.maze = None
        self._dist_map = None          # 区画 → ゴールまでの歩数
        self._prev_potential = None
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_dist_raw = None
        self._visited = set()
        self._step_count = 0
        # オドメトリ（自前センサの積分のみ。真の位置は使わない）
        self._odo_x = self._odo_y = self._odo_yaw = 0.0

        if seed is not None:
            gym.Env.reset(self, seed=seed)

    # ------------------------------------------------------------------
    def _next_maze_seed(self) -> int:
        """学習用の迷路 seed（評価・検証に予約された帯は決定的に読み飛ばす）。"""
        while True:
            s = self.base_seed + self._episode_count
            self._episode_count += 1
            if s not in _RESERVED_MAZE_SEEDS:
                return s

    def _load_maze(self, maze_seed: int):
        m = generate_maze(maze_seed, mode=self.maze_mode)
        cs = self.params.cell_size
        sx, sy = m["start"]
        heading = initial_heading_deg(m["v_walls"], m["h_walls"], m["start"])
        fd, tmp = tempfile.mkstemp(suffix=".xml", prefix=f"maze6_{maze_seed}_")
        os.close(fd)
        try:
            build_maze_robot_xml(
                m["v_walls"], m["h_walls"], tmp, model_name=f"maze6_{maze_seed}",
                mouse_pos=f"{sx * cs + cs / 2} {sy * cs + cs / 2} 0.002",
                mouse_euler=f"0 0 {heading}", center_goal=False, params=self.params)
            self.sim = MouseSim(tmp, params=self.params)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return m, heading

    # ------------------------------------------------------------------
    def _cell_of(self, x: float, y: float):
        """真の位置から区画を求める（**報酬と終了判定にのみ使う**。方策へは渡さない）。"""
        cs = self.params.cell_size
        return (min(max(int(x / cs), 0), SIZE - 1), min(max(int(y / cs), 0), SIZE - 1))

    def _potential(self, cell) -> float:
        """Φ = D₀ − d(cell) [m]。d はゴールまでの迷路距離。"""
        d = self._dist_map.get(cell, -1)
        if d < 0:
            d = self._d_start          # 到達不能（起きない想定）は基準値で据え置く
        return (self._d_start - d) * self.params.cell_size

    def _update_odometry(self, raw):
        """自前センサだけから自己位置を積分する（実機のデッドレコニングと同じ）。

        真の位置（privileged_pose）は**使わない**。車輪の滑りやジャイロの誤差は
        そのまま推定誤差として乗る。
        """
        n = self._n_dist
        gyro_z = float(raw[n + 5])
        omega_l, omega_r = float(raw[n + 6]), float(raw[n + 7])
        v = self.params.wheel_radius * (omega_l + omega_r) / 2.0
        dt = self.params.control_dt
        self._odo_yaw += gyro_z * dt
        self._odo_x += v * math.cos(self._odo_yaw) * dt
        self._odo_y += v * math.sin(self._odo_yaw) * dt

    def _goal_relative(self):
        """機体座標系で見たゴール中心への**推定**相対位置 [m]。

        ゴール中心は「中央 2x2 の中心」＝規約既知。自己位置・方位は推定値を使う。
        """
        cs = self.params.cell_size
        gx = (SIZE / 2.0) * cs
        gy = (SIZE / 2.0) * cs
        dx, dy = gx - self._odo_x, gy - self._odo_y
        c, s = math.cos(-self._odo_yaw), math.sin(-self._odo_yaw)
        return (dx * c - dy * s, dx * s + dy * c)

    def _make_observation(self) -> np.ndarray:
        raw = self.sim.observation()
        n = self._n_dist
        dist_raw = np.asarray(raw[0:n], dtype=np.float64)
        dist = dist_raw / _DIST_SCALE
        if self._prev_dist_raw is None:
            diff = np.zeros(n, dtype=np.float64)
        else:
            diff = np.clip((dist_raw - self._prev_dist_raw) / _DIST_DIFF_SCALE, -1.0, 1.0)
        self._prev_dist_raw = dist_raw

        gyro_z = raw[n + 5] / _GYRO_SCALE
        accel_xy = np.asarray(raw[n:n + 2], dtype=np.float64) / _ACCEL_SCALE
        wheels = np.asarray(raw[n + 6:n + 8], dtype=np.float64) / _WHEEL_SCALE
        rel = np.asarray(self._goal_relative(), dtype=np.float64) / _REL_SCALE

        obs = np.concatenate([dist, diff, [gyro_z], accel_xy, wheels,
                              self._prev_action, rel])
        return obs.astype(np.float32)

    def _make_info(self, cell, collision, goal, sim_time) -> dict:
        x, y, _ = self.sim.privileged_pose()
        return {
            "maze_seed": self.maze["seed"],
            "cell": cell,
            "dist_to_goal": int(self._dist_map.get(cell, -1)),
            "d_start": int(self._d_start),
            "n_visited": len(self._visited),
            "collision": bool(collision),
            "goal": bool(goal),
            "sim_time": float(sim_time),
            # 自己位置推定の誤差（学習には使わない。評価・分析用）
            "odom_error_m": float(math.hypot(self._odo_x - x, self._odo_y - y)),
        }

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        maze_seed = (self._next_maze_seed() if self.mode == "generate"
                     else int(self.np_random.choice(self._maze_seeds)))
        self.maze, heading = self._load_maze(maze_seed)

        self._dist_map = shortest_distances(self.maze["v_walls"], self.maze["h_walls"])
        start = tuple(self.maze["start"])
        self._d_start = self._dist_map[start]

        self.sim.full_reset(cell=start, heading_deg=heading)
        cs = self.params.cell_size
        cx_m, cy_m = start[0] * cs + cs / 2, start[1] * cs + cs / 2
        hr = math.radians(heading)
        lateral = float(self.np_random.uniform(-_LATERAL_PERTURB_M, _LATERAL_PERTURB_M))
        dh = float(self.np_random.uniform(-_HEADING_PERTURB_DEG, _HEADING_PERTURB_DEG))
        x = cx_m + lateral * (-math.sin(hr))
        y = cy_m + lateral * math.cos(hr)
        nh = math.radians(heading + dh)
        root_jid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        qadr = self.sim.model.jnt_qposadr[root_jid]
        self.sim.data.qpos[qadr] = x
        self.sim.data.qpos[qadr + 1] = y
        self.sim.data.qpos[qadr + 3] = math.cos(nh / 2.0)
        self.sim.data.qpos[qadr + 4] = 0.0
        self.sim.data.qpos[qadr + 5] = 0.0
        self.sim.data.qpos[qadr + 6] = math.sin(nh / 2.0)
        mujoco.mj_forward(self.sim.model, self.sim.data)

        # オドメトリの初期値は**擾乱後の真の姿勢**にする。実機はスタート区画で機体を
        # 壁に押し当てて位置と向きを出してから走り出すので、始点では自分の姿勢を
        # 正確に知っている。初期値だけを与え、以後は自前センサの積分のみで進める
        # （＝実機のデッドレコニングと同じ構造。積分誤差は車輪の滑りとジャイロから
        # 自然に蓄積する）。
        # 初期値を「区画中心・規定方位」に固定すると、方位擾乱 ±10° がそのまま推定
        # 誤差になり、0.45 秒走っただけで 68 mm（区画の 38%）ずれる。これは実機の
        # 状況ではなく、単に初期条件を知らせていないだけの人工的な誤差である。
        tx, ty, tyaw = self.sim.privileged_pose()
        self._odo_x, self._odo_y, self._odo_yaw = tx, ty, tyaw

        self._visited = {start}
        self._step_count = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_dist_raw = None
        self._prev_potential = self._potential(start)

        obs = self._make_observation()
        return obs, self._make_info(start, False, False, self.sim.sim_time)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        result = self.sim.step_control(float(action[0]) * self.params.voltage_limit,
                                       float(action[1]) * self.params.voltage_limit)
        self._step_count += 1
        self._update_odometry(self.sim.observation())

        x, y, _yaw = self.sim.privileged_pose()
        cell = self._cell_of(x, y)
        goal_reached = cell in GOAL_CELLS
        physical_fail = bool(result["collision"] or result["tipped"])

        potential = self._potential(cell)
        reward = self.gamma * potential - self._prev_potential - _TIME_PENALTY
        if goal_reached:
            reward += _GOAL_BONUS
        elif physical_fail:
            reward += self.collision_penalty
        if cell not in self._visited:
            self._visited.add(cell)
            reward += self.visit_bonus
        if self.action_smooth_penalty != 0.0:
            d = action - self._prev_action
            reward -= self.action_smooth_penalty * float(np.dot(d, d))
        self._prev_potential = potential

        terminated = bool(goal_reached or physical_fail)
        truncated = bool((not terminated) and self._step_count >= _TIME_LIMIT_STEPS)
        self._prev_action = np.asarray(action, dtype=np.float32)

        obs = self._make_observation()
        info = self._make_info(cell, physical_fail, goal_reached, result["sim_time"])
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        self._sim_cache.clear()
        self._cache_order.clear()
