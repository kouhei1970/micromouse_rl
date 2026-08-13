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
    GOAL_CELLS, SIZE, cells_open, generate_maze, initial_heading_deg, shortest_distances,
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
                 action_smooth_penalty: float = 0.0,
                 action_highpass_penalty: float = 0.0,
                 action_highpass_alpha: float = 0.5,
                 continuous_potential: bool = False):
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

    def _potential(self, cell, prev_cell=None, x: float = None, y: float = None) -> float:
        """Φ [m]。continuous_potential に応じて階段版／連続版を切り替える。

        continuous_potential=False（既定）では x, y, prev_cell は無視され、
        _potential_stair(cell) のみで決まる（既存挙動を bit 単位で保つ）。
        """
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
