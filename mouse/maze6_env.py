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

  実装前の検算（D₀ = 1.8 m・速度 0.96 m/s・上限 6000 ステップ、**行動の高周波成分への
  罰（案3, k=8.7e-3）を含まない前提**の値であり、かつ割引前／割引後の区別が曖昧なまま
  書かれている。古い数値は消さず、以下に是正の注記を追記する。2026-08-11 是正）:
    最短でゴール +0.975 > 遠回りしてゴール +0.340 > 探索だけで時間切れ +0.160
    > 半分進んで衝突 −0.137 > その場に留まる −0.200
  訪問報酬なし（r_v=0）だと「遠回りしてゴール」が −0.020 とほぼゼロになり学習信号が
  弱すぎる。r_v=0.05 まで上げると探索（+0.700）がゴール（+0.975）に迫って危ない。
  **r_v=0.02** は訪問報酬の対象区画ぶんの総和がゴールボーナス 1.0 を下回るように選んだ
  （旧記載「36 区画ぶんの総和 0.72」は誤り。**スタート区画は reset() で _visited に
  入るので訪問報酬の対象は最大 35 区画 = 0.70**。2026-08-11 是正）。

  **これは罰なし（k=0）前提の設計時検算にすぎない。**実際に投入した
  k=8.7e-3・E‖a−ā‖²=0.459 込みの再計算は
  `experiments/exp_012_continuous_potential/design.md` §2 にある。是正後の値
  （k=8.7e-3・検証帯 D₀ 中央値 9.5 区画での割引後総収益）:
    ゴール +0.30〜+0.94 > 滞留 −0.200 / 衝突 −0.16〜−0.55 > 探索(6000歩) −0.802
    ＝ 望ましい順序「ゴール > 探索 > 滞留 > 衝突」は 20 面すべてで崩れている

continuous_potential=True（exp_012 で追加。既定 False）にすると、Φ を区画単位の
階段関数ではなく、開口部の中点を経由する折れ線へ真の位置を射影した連続量にする
（M1 `mouse/corridor_env.py` の Φ と同じ考え方）。動機・数式の導出・検算は
`experiments/exp_012_continuous_potential/design.md` を参照。既定 False では
挙動は変わらない（`Maze6Env._potential_stair()` が既存の階段版そのもの）。

geodesic_potential=True（exp_012 の条件 C。既定 False）にすると、Φ を自由空間の
測地距離場（reset 時に前計算した格子 Dijkstra）で決める。区画やその前後関係
（cell・prev_cell）を一切見ない位置だけの関数になる。優先順位は
geodesic > continuous > stair（`_potential()` の分岐）。動機・数式・検証項目は
`experiments/exp_012_continuous_potential/design.md`「条件 C: 自由空間の測地距離場」
を参照。既定 False では挙動は変わらない。
"""
import heapq
import math
import os
import tempfile
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from mouse.maze6_gen import (
    GOAL_CELLS, SIZE, cells_open, generate_maze, initial_heading_deg, shortest_distances,
)
from mouse.mjcf import WALL_THICKNESS, build_maze_robot_xml
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

# ==========================================================================
# 測地距離場（条件 C。exp_012 design.md「実装内容」節）
# ==========================================================================
# 格子解像度 h_g。区画寸法 cs = 0.18 m（RobotParams.cell_size の既定値。他の定数
# （_REL_SCALE 等）と同じくここでは直値で持つ）を _GEO_STEPS_PER_CELL 分割して
# 決める。181×181 = 32761 点（design.md の指定どおり）。
_GEO_STEPS_PER_CELL = 30
_GEO_GRID_N = SIZE * _GEO_STEPS_PER_CELL + 1       # 181
_GEO_GRID_H = 0.18 / _GEO_STEPS_PER_CELL           # 0.006 m

# 条件 C の測地距離場は「**配置空間**の測地距離」である（裁定 R32）。
# 点の測地（機体を点とみなす）は**実機の中心が辿れない経路**の距離を返すので、
# 用途に合わない代理量になる。機体中心は閉じた壁面から w_lat 以上離れる必要があり、
# 壁は境界の中心線を軸に厚み t_w を持つので、**壁の中心線からの離隔**は t_w/2 + w_lat。
#   ⚠️ ラベルに注意: 壁**面**からの離隔 = w_lat = 0.0400 m
#                    壁**中心線**からの離隔 = t_w/2 + w_lat = 0.0460 m ← 格子の判定に使うのはこちら
# w_lat はモデルのメッシュ頂点から導出した機体の真の最外側半幅（コーナー片 mein_body2..5。
# 車輪 0.0395 でも AABB 0.04141 でもない。ROBOT_SPEC §2.1・裁定 R25/R26）。
# tests/test_maze6_potential.py が実行時導出値との一致を検査する。
_ROBOT_LAT_HALF_WIDTH = 0.0400
_GEO_CLEARANCE = WALL_THICKNESS / 2 + _ROBOT_LAT_HALF_WIDTH      # 0.0460 m（壁中心線から）

_GEO_TOPOLOGY = None    # プロセス内で 1 度だけ計算する位相キャッシュ（迷路によらない）


def _geo_topology():
    """測地距離場グラフの位相（迷路によらない部分）を遅延計算してキャッシュする。"""
    global _GEO_TOPOLOGY
    if _GEO_TOPOLOGY is None:
        _GEO_TOPOLOGY = _build_geo_topology()
    return _GEO_TOPOLOGY


def _build_geo_topology():
    """測地距離場の格子グラフの位相（迷路に依存しない部分）を作る。

    [0, SIZE·cs] 四方を h_g 刻みの (N, N) 格子に区切り、8 近傍（軸 4・斜め 4）の
    辺を各格子点から方向 (1,0)・(0,1)・(1,1)・(1,-1) の 4 通りだけ張ることで
    重複なく列挙する（逆向きは迷路ごとの組み立て時に複製する）。

    区画境界は「floor(座標/cs) は区画の高い側に属す」という規約（浮動小数点を
    経由せず、格子インデックスの整数除算 i // _GEO_STEPS_PER_CELL だけで厳密に
    決まる）で判定し、各辺を 3 種に分類する:
      free   … 両端が同じ区画に属する → 壁の有無に関わらず常に開通
      axis   … ちょうど 1 つの座標だけ区画が変わる → その区画境界
               （v_walls か h_walls の該当 1 マス）の開通状況で決まる
      corner … 斜め辺の両端で x も y も区画が変わる。格子点が区画 4 隅の交点
               （区画境界を h_g 単位で割り切れる _GEO_STEPS_PER_CELL の倍数）に
               ちょうど一致する退化ケースで、壁の厚みを持たない前提の下では
               どちらの区画に属すとも一意に決まらない特異点である。標準的な
               格子探索の corner-cutting 対策にならい、2 本の迂回路（L 字）の
               どちらかが両方開通しているときに限り通す（迷路ごとに
               `mouse.maze6_gen.cells_open()` で判定。件数は自ずと少ない）

    戻り値は迷路に依存しない静的データ（座標・重み・分類・区画対・ゴール格子点）
    の辞書。`_compute_geodesic_field()` が迷路ごとの壁配列と組み合わせて使う。
    """
    N = _GEO_GRID_N
    ar = np.arange(N)
    cellx = np.minimum(ar // _GEO_STEPS_PER_CELL, SIZE - 1)     # 格子インデックス→区画インデックス
    II, JJ = np.meshgrid(ar, ar, indexing="ij")                 # II[i,j]=i, JJ[i,j]=j

    coords = np.empty((N * N, 2), dtype=np.float64)
    coords[:, 0] = (II * _GEO_GRID_H).ravel()
    coords[:, 1] = (JJ * _GEO_GRID_H).ravel()

    free_parts, axis_parts, corner_parts = [], [], []
    directions = ((1, 0, _GEO_GRID_H), (0, 1, _GEO_GRID_H),
                  (1, 1, _GEO_GRID_H * math.sqrt(2.0)), (1, -1, _GEO_GRID_H * math.sqrt(2.0)))
    for di, dj, w in directions:
        bi, bj = II + di, JJ + dj
        valid = (bi >= 0) & (bi < N) & (bj >= 0) & (bj < N)
        ai, aj, bi, bj = II[valid], JJ[valid], bi[valid], bj[valid]
        a_id = (ai * N + aj).astype(np.int64)
        b_id = (bi * N + bj).astype(np.int64)
        cax, cay = cellx[ai], cellx[aj]
        cbx, cby = cellx[bi], cellx[bj]
        same = (cax == cbx) & (cay == cby)
        axis_v = (~same) & (cay == cby)       # x だけ変わる（same=False で確定的に x!=cbx）
        axis_h = (~same) & (cax == cbx)       # y だけ変わる
        corner = (~same) & (~axis_v) & (~axis_h)

        free_parts.append((a_id[same], b_id[same], np.full(int(same.sum()), w)))

        m = axis_v
        axis_parts.append((a_id[m], b_id[m], np.full(int(m.sum()), w),
                            np.maximum(cax[m], cbx[m]), cay[m],
                            np.zeros(int(m.sum()), dtype=np.int8)))     # 0 = v_walls
        m = axis_h
        axis_parts.append((a_id[m], b_id[m], np.full(int(m.sum()), w),
                            cax[m], np.maximum(cay[m], cby[m]),
                            np.ones(int(m.sum()), dtype=np.int8)))      # 1 = h_walls

        m = corner
        corner_parts.append((a_id[m], b_id[m], np.full(int(m.sum()), w),
                              np.stack([cax[m], cay[m]], axis=1),
                              np.stack([cbx[m], cby[m]], axis=1)))

    free_src = np.concatenate([p[0] for p in free_parts])
    free_dst = np.concatenate([p[1] for p in free_parts])
    free_w = np.concatenate([p[2] for p in free_parts])

    axis_src = np.concatenate([p[0] for p in axis_parts])
    axis_dst = np.concatenate([p[1] for p in axis_parts])
    axis_w = np.concatenate([p[2] for p in axis_parts])
    axis_wx = np.concatenate([p[3] for p in axis_parts])
    axis_wy = np.concatenate([p[4] for p in axis_parts])
    axis_kind = np.concatenate([p[5] for p in axis_parts])

    corner_src = np.concatenate([p[0] for p in corner_parts])
    corner_dst = np.concatenate([p[1] for p in corner_parts])
    corner_w = np.concatenate([p[2] for p in corner_parts])
    corner_cellA = np.concatenate([p[3] for p in corner_parts], axis=0)
    corner_cellB = np.concatenate([p[4] for p in corner_parts], axis=0)

    goal_xs = sorted({c[0] for c in GOAL_CELLS})
    goal_ys = sorted({c[1] for c in GOAL_CELLS})
    goal_mask = np.isin(cellx[II], goal_xs) & np.isin(cellx[JJ], goal_ys)
    goal_nodes = (II * N + JJ)[goal_mask].astype(np.int64).ravel()

    return dict(
        N=N, coords=coords,
        free_src=free_src, free_dst=free_dst, free_w=free_w,
        axis_src=axis_src, axis_dst=axis_dst, axis_w=axis_w,
        axis_wx=axis_wx, axis_wy=axis_wy, axis_kind=axis_kind,
        corner_src=corner_src, corner_dst=corner_dst, corner_w=corner_w,
        corner_cellA=corner_cellA, corner_cellB=corner_cellB,
        goal_nodes=goal_nodes,
    )


class Maze6Env(gym.Env):
    """6x6 迷路の単走環境。mode='loop'（M2-0）/ 'full'（M2-1）。"""

    metadata = {"render_modes": []}

    def __init__(self, maze_dir=None, maze_seeds=None, max_cache=8, seed=None,
                 gamma: float = 0.995, mode: str = "fixed", base_seed: int = 8000,
                 maze_mode: str = "loop", visit_bonus: float = _VISIT_BONUS,
                 collision_penalty: float = _COLLISION_PENALTY,
                 action_smooth_penalty: float = 0.0,
                 action_highpass_penalty: float = 0.0,
                 action_highpass_alpha: float = 0.5,
                 continuous_potential: bool = False,
                 geodesic_potential: bool = False):
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
        # 案 3（M1 で検証済みの構成をそのまま M2 へ持ってくる）。行動の低域通過成分 ā を
        # 環境側で持ち、毎ステップ −k·‖a_t − ā_t‖² を元報酬へ加える。
        #     ā_t = α·ā_(t−1) + (1 − α)·a_t   （ā_(−1) = 0）
        # M1 の実測: α=0.5・k=8.7e-3 で符号反転 11.6〜14.8 回/s（仕様書由来の必達線
        # 15.4 を下回る）・gate 完走率 0.94〜0.97・1 区画 0.142 s（exp_005 比 20% 高速）・
        # 銅損 9.33 W → 1.01 W。詳細は experiments/exp_006_action_smoothness/card.md。
        # **M2 は分岐が増えて舵を切る場面が多いので、α の再評価は M2 の実測で行う**
        # （2026-08-11 教授裁定。いま決めるより M2 の舵の帯域を測ってからの方が良い）。
        self.action_highpass_penalty = float(action_highpass_penalty)
        if not 0.0 <= float(action_highpass_alpha) < 1.0:
            raise ValueError(
                f"action_highpass_alpha は 0 以上 1 未満: {action_highpass_alpha!r}")
        self.action_highpass_alpha = float(action_highpass_alpha)
        # Φ を連続化するか（exp_012）。既定 False では _potential_stair() のみを通り、
        # 挙動は本変更の前と bit 単位で同一。
        self.continuous_potential = bool(continuous_potential)
        # Φ を自由空間の測地距離場にするか（exp_012 条件 C）。geodesic > continuous
        # > stair の優先順位で _potential() が分岐する。既定 False では無関係。
        self.geodesic_potential = bool(geodesic_potential)
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
        self._cell = None              # 直近の区画（区画遷移の検出に使う）
        self._prev_cell = None         # c_prev: 直前に居た区画（連続 Φ 専用。reset で None）
        self._geo_field = None         # (N, N) 測地距離場（測地版 Φ 専用。reset で前計算）
        self._geo_start = None         # g(reset 直後の真の位置)（測地版 Φ のエピソード定数）
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._action_lowpass = np.zeros(2, dtype=np.float64)   # ā_(−1) = 0（案 3）
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

    def _potential_stair(self, cell) -> float:
        """Φ = D₀ − d(cell) [m]（区画単位の階段関数。continuous_potential=False の実装）。

        d はゴールまでの迷路距離。本メソッドは exp_012 以前の `_potential()` そのもの
        （リネームのみ、本体は変更していない）。
        """
        d = self._dist_map.get(cell, -1)
        if d < 0:
            d = self._d_start          # 到達不能（起きない想定）は基準値で据え置く
        return (self._d_start - d) * self.params.cell_size

    # ------------------------------------------------------------------
    # 連続版 Φ の幾何ヘルパ（exp_012）
    # ------------------------------------------------------------------
    def _cell_center(self, cell):
        """区画中心の座標 [m]（真の位置系）。"""
        cs = self.params.cell_size
        return (cell[0] * cs + cs / 2.0, cell[1] * cs + cs / 2.0)

    def _edge_midpoint(self, a, b):
        """隣接する区画 a, b の共有辺の中点 w(a, b) = 2 区画中心の中点 [m]。

        a, b の順序に依存しない（対称な平均なので浮動小数点でも bit 単位で一致する。
        w_in == w_out の判定に使う恒等性はこれに依存している）。
        """
        ax, ay = self._cell_center(a)
        bx, by = self._cell_center(b)
        return ((ax + bx) / 2.0, (ay + by) / 2.0)

    def _open_neighbors(self, cell):
        """区画 cell の開通済み隣接区画一覧（上下左右）。壁配列の規約は maze6_gen と同一。"""
        x, y = cell
        out = []
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            nx, ny = nb
            if (0 <= nx < SIZE and 0 <= ny < SIZE
                    and cells_open(self.maze["v_walls"], self.maze["h_walls"], cell, nb)):
                out.append(nb)
        return out

    def _descending_neighbors(self, cell):
        """cell の**降下隣接**一覧: 開通した隣接区画のうち d が cell よりちょうど 1 小さいもの。

        複数ありうる（迷路グラフは格子の部分グラフで二部グラフなので、開通した隣接の
        d は必ず ±1 だけ違う。d(cell) が 0 でなければ降下隣接は必ず 1 つ以上存在する）。
        """
        d0 = self._dist_map[cell]
        return [nb for nb in self._open_neighbors(cell)
                if self._dist_map.get(nb, -1) == d0 - 1]

    def _descending_neighbor(self, cell):
        """cell の降下隣接を 1 つだけ、(x, y) の辞書順最小で決定的に選ぶ。

        **`_potential_continuous` はもう本メソッドを使わない**（2026-08-13 改訂で
        tie-break そのものを廃止し、`_descending_neighbors` の全件について min を
        取る形へ置き換えたため）。削除はせず残す。単体テスト（(b-2) の「降下方向が
        曲がる構成」列挙）が引き続き参照する。
        """
        cands = self._descending_neighbors(cell)
        if not cands:
            raise AssertionError(f"降下隣接が見つからない: cell={cell} d={self._dist_map[cell]}")
        return min(cands)

    def _potential_continuous(self, cell, prev_cell, x: float, y: float) -> float:
        """Φ の連続版（exp_012。continuous_potential=True）。区画ごとの明示式 ＋ 全降下隣接の min。

        真の位置 (x, y) は**報酬計算にのみ使う**（方策へは渡さない）。cs = 区画寸法
        (params.cell_size = 0.18 m)、h = cs/2、C = cell の中心。降下隣接 n ごとに
        w_out = w(cell, n)（cell から n への開口部の中点）、
        w_in  = w(prev_cell, cell)（prev_cell が None のときは n ごとに w_out と同じ扱い）、
        a = (w_in−C)/|w_in−C|、b = (w_out−C)/|w_out−C|、
        s = (P−C)·a、t = (P−C)·b（P = (x, y)）として、区画内の残り弧長 ℓ_n を

          直線 (a = −b)      : ℓ_n = clamp(h − t, 0, 2h)
          折れ (a ⊥ b)       : ℓ_n = clamp(s, 0, h) + (h − clamp(t, −h, h))
          w_in == w_out      : ℓ_n = clamp(h − t, 0, h)    ← 後戻り・reset 直後

        残り距離は**全降下隣接についての最小値**（tie-break は廃止）:

          残り距離 = min_n [ ℓ_n(P) + d(n)·cs + cs/2 ] ,   Φ = d_start·cs − 残り距離

        d(n) = d(cell) − 1 は全降下隣接で等しいので、min は実質 ℓ_n だけに効く。
        d(cell) = 0（ゴール区画）のとき残り距離 = 0（従来どおり）。
        `self._dist_map.get(cell, -1) < 0`（到達不能・起きない想定）のときは d0 を
        d_start に据え置く（従来どおり）。

        --------------------------------------------------------------------
        検算（`experiments/exp_012_continuous_potential/design.md` §4「新しい定義
        （本改訂で採用する）」と同一）
        --------------------------------------------------------------------
        - **区画中心 P=C**: s=t=0 なので、直線・折れ・w_in==w_out のどの場合分けでも
          ℓ_n(C)=h。よって 残り距離 = min_n[h + d(n)·cs + h] = d(n)·cs + cs = d(cell)·cs
          → **階段版 `_potential_stair` と一致**（テスト (a)）
        - **全降下開口部 w(cell, n)**: cell 側では ℓ_n=0 かつ他の降下隣接は ℓ≥0 なので
          min = d(n)·cs + cs/2。n 側では進入点なので w_in(n)=w_out(cell) となり全ての
          ℓ=2h=cs、min = cs + (d(n)−1)·cs + cs/2 = d(n)·cs + cs/2 → **両側で一致**。
          tie-break が無いので**どの降下隣接の開口部を通っても**この一致が成立する
          （テスト (b-2)。旧実装は tie-break が選んだ 1 開口部でしか一致しなかった）
        - **Lipschitz 定数は厳密に √2**: 折れの内側（両方の clamp が効かない領域）で
          ∇ℓ_n = a−b、a⊥b かつ単位ベクトルなので |a−b|=√2。境界一致（区画をまたぐと
          Φ がちょうど cs 増える一方、境界間の直線距離は cs/√2 しかない）だけから
          L ≥ cs/(cs/√2) = √2 が導かれるので、これは**この制約下での最良値**。
          √2-Lipschitz な ℓ_n の min はやはり √2-Lipschitz（min を取っても定数は
          増えない）なので、Φ 全体も √2-Lipschitz。**|∇Φ|≤1 は境界一致と両立できない
          （下限 √2）ので要求しない**
        - **旧実装（最近接点射影・`corridor_gen.remaining_path_length` の再利用）が
          持っていた 2 つの欠陥は、この定義では構成上起きない**:
            D1（折れ区画の内側・角の二等分線での真の不連続）: 旧実装は「垂線距離が
              最小の線分」を位置ごとに選んでいたため、二等分線をまたぐ瞬間に選択が
              切り替わり弧長が飛んだ。本定義は**位置に依らず a・b の幾何関係だけで
              場合分けが決まる**明示式なので、線分の選択自体が無く消える
            D2（tie-break が選ばなかった降下開口部を通ると跳ぶ）: 旧実装は降下隣接が
              同点で複数あるとき 1 つを辞書順で選んでいたため、選ばれなかった開口部
              から出ると Φ が別経路の残り距離へ切り替わって跳んだ。本定義は
              **全降下隣接についての min** を毎回取るので tie-break そのものが無く、
              どの開口部を通っても連続に一致する
        --------------------------------------------------------------------
        """
        cs = self.params.cell_size
        h = cs / 2.0
        d0 = self._dist_map.get(cell, -1)
        if d0 < 0:
            d0 = self._d_start          # 到達不能（起きない想定）は基準値で据え置く
        if d0 == 0:
            remaining = 0.0             # ゴール区画: 残り距離は 0
        else:
            neighbors = self._descending_neighbors(cell)
            if not neighbors:
                raise AssertionError(f"降下隣接が見つからない: cell={cell} d={d0}")
            cx, cy = self._cell_center(cell)
            px, py = x - cx, y - cy
            remaining = None
            for n in neighbors:
                w_out = self._edge_midpoint(cell, n)
                w_in = w_out if prev_cell is None else self._edge_midpoint(prev_cell, cell)
                bx, by = w_out[0] - cx, w_out[1] - cy
                nrm_b = math.hypot(bx, by)
                bx, by = bx / nrm_b, by / nrm_b
                if w_in == w_out:
                    # 後戻り・reset 直後: w_in = w_out として扱う
                    t = px * bx + py * by
                    ell = min(max(h - t, 0.0), h)
                else:
                    ax, ay = w_in[0] - cx, w_in[1] - cy
                    nrm_a = math.hypot(ax, ay)
                    ax, ay = ax / nrm_a, ay / nrm_a
                    s = px * ax + py * ay
                    t = px * bx + py * by
                    if ax * bx + ay * by < 0.0:
                        ell = min(max(h - t, 0.0), 2.0 * h)              # 直線 (a = −b)
                    else:
                        ell = min(max(s, 0.0), h) + (h - min(max(t, -h), h))  # 折れ (a ⊥ b)
                d_n = d0 - 1                       # 全降下隣接で共通（= self._dist_map[n]）
                cand = ell + d_n * cs + cs / 2.0
                remaining = cand if remaining is None else min(remaining, cand)
        return self._d_start * cs - remaining

    # ------------------------------------------------------------------
    # 測地距離版 Φ のヘルパ（exp_012 条件 C）
    # ------------------------------------------------------------------
    def _compute_geodesic_field(self) -> np.ndarray:
        """自由空間の測地距離場を格子 Dijkstra で前計算する（条件 C。reset で 1 度だけ呼ぶ）。

        位相（迷路によらない格子グラフの形）は `_geo_topology()` がプロセス内で
        1 度だけ計算してキャッシュしたものを使い回し、本メソッドは**今の迷路**の
        壁配列（`self.maze["v_walls"]` / `["h_walls"]`）と組み合わせて開通判定を
        行う（vectorized。壁の開閉は迷路ごとに違うのでここでしか決まらない）。

        始点集合は**ゴール 2×2 区画の内部にある全格子点**（距離 0）。scipy が
        使えれば `scipy.sparse.csgraph.dijkstra` の方が速いが、本環境（.venv）には
        scipy が入っていないため確認の上 heapq で自前実装した
        （`experiments/exp_012_continuous_potential/design.md`「実装内容」節 1）。

        戻り値: (N, N) の float64 配列。[i, j] は座標 (i·h_g, j·h_g) [m] における
        測地距離 [m]（ゴールまで到達不能なら inf だが、迷路生成の不変条件
        「全区画へ到達可能」により実際には起きない）。
        """
        topo = _geo_topology()
        N = topo["N"]
        v_walls, h_walls = self.maze["v_walls"], self.maze["h_walls"]

        # axis 辺: 迷路ごとの開通状況を壁配列から引く（vectorized）。
        # v_walls/h_walls どちらの配列に対しても添字の範囲内に収まることを
        # _build_geo_topology() のコメントのとおり確認済みなので、両方引いてから
        # axis_kind で選ぶだけでよい。
        v_val = v_walls[topo["axis_wx"], topo["axis_wy"]]
        h_val = h_walls[topo["axis_wx"], topo["axis_wy"]]
        axis_open = np.where(topo["axis_kind"] == 0, v_val, h_val) == 0

        # corner 辺（退化ケース。件数は少ない）: 迂回路のどちらかが両方開通して
        # いれば通す。cells_open() をそのまま使い、本体の壁判定と規約を合わせる。
        n_corner = len(topo["corner_src"])
        corner_open = np.zeros(n_corner, dtype=bool)
        for k in range(n_corner):
            c0 = (int(topo["corner_cellA"][k, 0]), int(topo["corner_cellA"][k, 1]))
            c1 = (int(topo["corner_cellB"][k, 0]), int(topo["corner_cellB"][k, 1]))
            via1 = (c1[0], c0[1])
            via2 = (c0[0], c1[1])
            corner_open[k] = (
                (cells_open(v_walls, h_walls, c0, via1) and cells_open(v_walls, h_walls, via1, c1))
                or (cells_open(v_walls, h_walls, c0, via2) and cells_open(v_walls, h_walls, via2, c1)))

        src = np.concatenate([topo["free_src"], topo["axis_src"][axis_open],
                              topo["corner_src"][corner_open]])
        dst = np.concatenate([topo["free_dst"], topo["axis_dst"][axis_open],
                              topo["corner_dst"][corner_open]])
        w = np.concatenate([topo["free_w"], topo["axis_w"][axis_open],
                            topo["corner_w"][corner_open]])
        # 無向グラフなので逆向きを複製する（辺は方向 4 通りだけで重複なく列挙したため）。
        all_src = np.concatenate([src, dst])
        all_dst = np.concatenate([dst, src])
        all_w = np.concatenate([w, w])

        # CSR 風の隣接構造（送り元でソートして開始位置の累積を作る）。
        order = np.argsort(all_src, kind="stable")
        s_sorted = all_src[order]
        d_list = all_dst[order].tolist()          # heapq ループでは list 添字の方が速い
        w_list = all_w[order].tolist()
        n_nodes = N * N
        counts = np.bincount(s_sorted, minlength=n_nodes)
        offsets = np.zeros(n_nodes + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        offsets_list = offsets.tolist()

        # --- 配置空間の自由空間マスク（裁定 R32）--------------------------
        # 機体中心が閉じた区画境界（＝壁の中心線）から _GEO_CLEARANCE 以上
        # 離れている格子点だけを「機体が到達しうる」とみなす。
        allowed = self._geo_allowed_mask()
        allowed_list = allowed.ravel().tolist()

        dist = np.full(n_nodes, np.inf, dtype=np.float64)
        visited = bytearray(n_nodes)
        heap = []
        for node in topo["goal_nodes"].tolist():
            if allowed_list[node] and dist[node] != 0.0:
                dist[node] = 0.0
                heapq.heappush(heap, (0.0, node))
        # 第 1 段: **到達可能な格子点だけ**で Dijkstra を回す。ここで確定した値が
        # 配置空間の測地距離であり、以後変更しない。
        while heap:
            du, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = 1
            for k in range(offsets_list[u], offsets_list[u + 1]):
                v = d_list[k]
                if visited[v] or not allowed_list[v]:
                    continue
                nd = du + w_list[k]
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))

        bad = ~np.isfinite(dist) & allowed.ravel()
        if bad.any():
            raise AssertionError(
                f"配置空間の測地距離場に到達不能な格子点が {int(bad.sum())} 個ある"
                "（迷路生成の不変条件「全区画へ到達可能」と矛盾する。実装の欠陥の可能性）")

        # 第 2 段: 到達不能側（壁際の帯）へ値を延長する。**到達可能な点の値は
        # 変えない**（第 1 段で確定済み）ので、壁際の帯を通る近道は生じない。
        # 延長が要るのは、機体が到達しうる位置を囲む格子セルの隅が帯に入りうる
        # ためで、双線形補間が inf を掴まないようにするためだけの措置である。
        heap = [(float(dist[i]), int(i)) for i in np.flatnonzero(np.isfinite(dist))]
        heapq.heapify(heap)
        visited2 = bytearray(n_nodes)
        while heap:
            du, u = heapq.heappop(heap)
            if visited2[u]:
                continue
            visited2[u] = 1
            for k in range(offsets_list[u], offsets_list[u + 1]):
                v = d_list[k]
                if visited2[v] or allowed_list[v]:
                    continue                      # 到達可能な点は第 1 段の値を守る
                nd = du + w_list[k]
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        if not np.all(np.isfinite(dist)):
            n_bad = int(np.sum(~np.isfinite(dist)))
            raise AssertionError(
                f"延長後も値の付かない格子点が {n_bad} 個ある（実装の欠陥の可能性）")
        return dist.reshape(N, N)

    def _geo_allowed_mask(self) -> np.ndarray:
        """機体中心が到達しうる格子点の真偽マスク (N, N)（配置空間。裁定 R32）。

        格子点 (i, j) は、その属する区画の 4 つの境界のうち**閉じているもの**
        （壁があるか迷路の外周）すべてから `_GEO_CLEARANCE` 以上離れているときに
        到達可能とする。`_GEO_CLEARANCE` は**壁の中心線**からの離隔である
        （壁面からの離隔 w_lat = 0.0400 m ＋ 壁の半厚 t_w/2 = 0.006 m）。
        """
        N, S, H = _GEO_GRID_N, _GEO_STEPS_PER_CELL, _GEO_GRID_H
        cs = self.params.cell_size
        v_walls, h_walls = self.maze["v_walls"], self.maze["h_walls"]
        idx = np.arange(N)
        cell = np.minimum(idx // S, SIZE - 1)
        pos = idx * H
        lo = pos - cell * cs                 # 区画の低い側の境界からの距離
        hi = (cell + 1) * cs - pos           # 高い側の境界からの距離
        allowed = np.ones((N, N), dtype=bool)
        for cx in range(SIZE):
            xs = np.flatnonzero(cell == cx)
            for cy in range(SIZE):
                ys = np.flatnonzero(cell == cy)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                c = (cx, cy)
                blk_x_lo = not (cx > 0 and cells_open(v_walls, h_walls, c, (cx - 1, cy)))
                blk_x_hi = not (cx < SIZE - 1 and cells_open(v_walls, h_walls, c, (cx + 1, cy)))
                blk_y_lo = not (cy > 0 and cells_open(v_walls, h_walls, c, (cx, cy - 1)))
                blk_y_hi = not (cy < SIZE - 1 and cells_open(v_walls, h_walls, c, (cx, cy + 1)))
                ok_x = np.ones(len(xs), dtype=bool)
                ok_y = np.ones(len(ys), dtype=bool)
                if blk_x_lo:
                    ok_x &= lo[xs] >= _GEO_CLEARANCE
                if blk_x_hi:
                    ok_x &= hi[xs] >= _GEO_CLEARANCE
                if blk_y_lo:
                    ok_y &= lo[ys] >= _GEO_CLEARANCE
                if blk_y_hi:
                    ok_y &= hi[ys] >= _GEO_CLEARANCE
                allowed[np.ix_(xs, ys)] = ok_x[:, None] & ok_y[None, :]
        return allowed

    def _geodesic_value(self, x: float, y: float) -> float:
        """測地距離場の値 g(P) を**双線形補間**で取り出す（条件 C。裁定 R30）。

        P を囲む格子セルの 4 隅の値を双線形に混ぜる。**候補集合を持たない**ので
        P について構成上連続であり、これが本方式を選んだ理由である。

        --------------------------------------------------------------
        なぜ「下側包絡」をやめたか（欠陥 D3。design.md §4 の D3 節が経緯を持つ）
        --------------------------------------------------------------
        当初は下側包絡 g(P) = min_q ( field[q] + |P − q| ) を使っていた。この形は
        「1-Lipschitz 関数の min はやはり 1-Lipschitz」という論法で正当化していたが、
        **この論法は誤りである**: min を取る**候補集合が P に依存する**とき、
        候補が抜ける瞬間に値が跳ぶ。実測でも、機体が到達しうる領域に限り壁を跨ぐ
        点対を除いてもなお、刻み 6.0 / 1.5 / 0.375 mm に対し最大比が
        1.1781 → 2.3944 → 7.6289 と**発散**した（超過は 0.0011 → 0.0021 → 0.0025 m で
        有界 ＝ 加法的な跳び）。候補窓を 3×3 → 7×7 と広げても消えなかった。

        双線形補間に置き換えると同じ走査で

            刻み 6.000 mm → 最大比 1.0000（超過 0.000000 m）
            刻み 1.500 mm → 最大比 1.3107（超過 0.000659 m）
            刻み 0.375 mm → 最大比 1.3883（超過 0.000206 m）

        となり、**比は √2 = 1.4142 以下へ収束し超過は刻みとともに縮む**
        ＝ 真の不連続なし。√2 は**双線形補間の理論上界**（格子方向に
        1-Lipschitz な場を補間するときの勾配上界）であって実測当てはめではない。

        壁越しの参照も構造的に解消する: 双線形は P を囲む 4 点しか使わず、
        壁をまたぐ格子セルは**機体中心が到達できない領域**（閉じた壁面から
        t_w/2 + w_lat = 0.0455 m 以内）にしか現れないためである。
        """
        h = _GEO_GRID_H
        u, v = x / h, y / h
        i0 = min(max(int(math.floor(u)), 0), _GEO_GRID_N - 2)
        j0 = min(max(int(math.floor(v)), 0), _GEO_GRID_N - 2)
        a, b = u - i0, v - j0
        f = self._geo_field
        return float((1.0 - a) * (1.0 - b) * f[i0, j0]
                     + a * (1.0 - b) * f[i0 + 1, j0]
                     + (1.0 - a) * b * f[i0, j0 + 1]
                     + a * b * f[i0 + 1, j0 + 1])

    def _potential_geodesic(self, x: float, y: float) -> float:
        """Φ の測地距離版（exp_012 条件 C）。cell・prev_cell は使わない位置だけの関数。

        Φ(P) = g(reset 直後の真の位置) − g(P)。g(reset 直後の真の位置) は
        `self._geo_start` として reset() がエピソード定数として保存する
        （擾乱後の真の位置で決めるので Φ₀ = 0 が構成上成立する）。
        """
        return self._geo_start - self._geodesic_value(x, y)

    def _potential(self, cell, prev_cell=None, x: float = None, y: float = None) -> float:
        """Φ [m]。優先順位は geodesic > continuous > stair（この順に分岐する）。

        geodesic_potential も continuous_potential も False（既定）では x, y,
        prev_cell は無視され、_potential_stair(cell) のみで決まる
        （既存挙動を bit 単位で保つ）。
        """
        if self.geodesic_potential:
            return self._potential_geodesic(x, y)
        if not self.continuous_potential:
            return self._potential_stair(cell)
        return self._potential_continuous(cell, prev_cell, x, y)

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
        if self.geodesic_potential:
            self._geo_field = self._compute_geodesic_field()

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
        if self.geodesic_potential:
            # g(reset 直後の真の位置) をエピソード定数として保存する（擾乱後の
            # 真の位置で決めるので Φ₀ = 0 が構成上成立する。design.md §「Φ の定義」）。
            self._geo_start = self._geodesic_value(tx, ty)

        self._visited = {start}
        self._step_count = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._action_lowpass = np.zeros(2, dtype=np.float64)   # ā_(−1) = 0（案 3）
        self._prev_dist_raw = None
        self._cell = start
        self._prev_cell = None      # c_prev（連続 Φ 用）。reset 直後は「直前の区画」が無い
        self._prev_potential = self._potential(start, self._prev_cell, tx, ty)

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
        if cell != self._cell:
            # 区画が変わった瞬間の直前区画を c_prev として保持する（連続 Φ 専用）。
            # 同じ区画に留まっている間は c_prev を更新しない（課題の仕様どおり）。
            self._prev_cell = self._cell
            self._cell = cell
        goal_reached = cell in GOAL_CELLS
        physical_fail = bool(result["collision"] or result["tipped"])

        potential = self._potential(cell, self._prev_cell, x, y)
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
        # ā の更新は罰の有無によらず**無条件**（k=0 と k>0 で内部状態の進み方を揃える）。
        # ā は観測には入らない。corridor_env と同じ規約。
        alpha = self.action_highpass_alpha
        self._action_lowpass = alpha * self._action_lowpass + (1.0 - alpha) * action
        if self.action_highpass_penalty != 0.0:
            hp = action - self._action_lowpass
            reward -= self.action_highpass_penalty * float(np.dot(hp, hp))
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
