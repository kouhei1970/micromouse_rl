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

geodesic_rho_scale=True（exp_012 の条件 C'。既定 False。geodesic_potential=True の
ときだけ効く）にすると、条件 C の Φ を**面ごとの定数** ρ 倍して、1 エピソードの
総整形量を条件 E（= cs·D₀）に揃える。

    ρ = cs·D₀ / g(start)          （D₀ = スタート区画の d [区画数]、
    Φ_C'(P) = ρ·(g(start) − g(P))   g(start) = reset 直後の真の位置の測地距離 [m]）

動機（裁定 R24-1・准教授の投入前指摘）: 測地距離場は角を切るぶん総整形量が条件 E より
小さいので、C が悪い結果になったとき「連続性のせいか報酬量が小さいせいか」を分離
できない。C 対 C' が報酬量の効果、C' 対 E が連続性・錨の効果を切り分ける。
既定 False では ρ = 1 で条件 C と bit 単位で同じ。
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
# 用途に合わない代理量になる。
#
# 🔴 マスクの定義は**障害物表面からの距離**である（裁定 R34。2026-08-13 是正）。
#   許可 = 「物理モデルに**実際に生成されている**障害物（壁・柱）の**表面**から
#           w_lat 以上離れている」。機体は半径 w_lat の**円板**とみなす。
#   ⚠️ ラベルに注意: 壁**面**からの離隔 = w_lat = 0.0400 m ← 一次の書き方（表面距離）
#                    壁**中心線**からの離隔 = t_w/2 + w_lat = 0.0460 m
#                    **両者が等価なのは壁の直線部だけ**であり、**柱・壁の端では等価に
#                    ならない**（そこでは表面までの最短距離が斜めに測られるため）。
#   旧実装は中心線基準の軸ごと判定で、**柱を描かず・壁の端も丸めていなかった**ため、
#   許可点の 9.17% が実際には壁・柱から 0.0400 未満だった（准教授の指摘。最悪は柱の角で 0）。
# w_lat はモデルのメッシュ頂点から導出した機体の真の最外側半幅（コーナー片 mein_body2..5。
# 車輪 0.0395 でも AABB 0.04141 でもない。ROBOT_SPEC §2.1・裁定 R25/R26）。
# tests/test_maze6_potential.py が実行時導出値との一致を検査する。
_ROBOT_LAT_HALF_WIDTH = 0.0400
# 壁の**直線部**での等価な中心線基準の離隔（記録・照合用。判定には使わない）。
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
                 geodesic_potential: bool = False,
                 geodesic_rho_scale: bool = False,
                 episode_limit_steps: int = _TIME_LIMIT_STEPS,
                 collision_respawn: bool = False,
                 goal_rule_containment: bool = False):
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
        # 条件 C'（裁定 R24-1）: 条件 C の Φ を面ごとの定数 ρ 倍して総整形量を条件 E に
        # 揃える。geodesic_potential=True のときだけ効く（単独指定は設計の取り違えなので
        # 弾く）。既定 False では ρ = 1 ＝ 条件 C と bit 単位で同じ。
        self.geodesic_rho_scale = bool(geodesic_rho_scale)
        # エピソード上限 [歩]（環境 v2 の設計値は 2000。裁定 2026-08-14）。
        # 既定は従来の 6000 なので、**渡さなければ挙動は bit 単位で変わらない**。
        # 🔴 上限が決めるのは総試行回数ではなく**配分**である（准教授 AUDIT_022 指摘 2）:
        # 1 セグメント（開始→衝突）は実測 157 歩なので、200 万歩で作れるセグメントは
        # **どの上限でも約 12,700 本**。上限 6000 なら 333 面 × 38.2 回、
        # 上限 2000 なら 1,000 面 × 12.7 回になる。**リスポーン（v2）の狙いは
        # 「同じ面で立て直す経験」なので、後者の配分が (c) の効果の本体**である。
        if int(episode_limit_steps) <= 0:
            raise ValueError(f"episode_limit_steps は正の整数: {episode_limit_steps!r}")
        self.episode_limit_steps = int(episode_limit_steps)
        # 🔴 衝突リスポーン（環境 v2 の対処案 (c)。裁定 2026-08-14。既定 False で従来どおり）。
        # 衝突・転倒で **collision_penalty を払って開始点へ戻し、エピソードは継続**する。
        # 終わるのは**ゴール成立か上限到達のときだけ**になる。
        #
        # 機構: ポテンシャル整形の恒等式 Σγ^t(γΦ(s')−Φ(s)) = γ^T·Φ(s_T) − Φ(s_0) は
        # **終端の Φ だけで決まる**ので、開始点へ戻る遷移で γΦ(start) − Φ_c = −Φ_c が
        # 1 度入り、**稼いだ整形をポテンシャル自身が取り返す**。
        # **報酬契約には一切触れない没収**であり、**走行中の 1 歩ごとの信号は変わらない**。
        # 訪問済み `_visited` は**エピソード単位で維持**する（裁定済み）ので、
        # 「衝突 → 戻る → 同じ区画で稼ぐ」報酬ポンプは**構成上作れない**。
        self.collision_respawn = bool(collision_respawn)
        # 🔴 R49 規約終端（環境 v2 項目 1・ユーザ裁定 1 で承認確定。既定 False で従来どおり）。
        # ゴールの成立を**機体中心が区画に入ったか**（`goal_rate_env`）ではなく
        # **機体全体がゴール 2×2 に内包されたか**（競技規約 §2）で判定する。
        # 判定は `competition/evaluator.py` の実装をそのまま呼ぶ（`_goal_containment()`。
        # **独立の再記述を作らない**。R34 の作法）。
        # ⚠️ 両者は**別の規則**であり、規約側の方が厳しい（中心が入っても内包は成らない）。
        self.goal_rule_containment = bool(goal_rule_containment)
        if self.geodesic_rho_scale and not self.geodesic_potential:
            raise ValueError(
                "geodesic_rho_scale=True は geodesic_potential=True のときだけ有効です"
                "（条件 C' は条件 C の Φ を ρ 倍したもの。design.md「条件 C'」）")
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
        # 🔴 名前は「**適用した倍率**」を指す（R11 バッチ項目 6・§9-17「量を名指しする」）。
        # **その迷路の ρ ではない**: 条件 C では常に 1.0 が入る（同じ迷路の ρ が 1.4 でも）。
        # 面の ρ が要るときは info["geo_inv_rho"] の逆数を使う。
        self._geo_rho_applied = 1.0    # 条件 C' で適用する面ごとの定数（条件 C では 1.0）
        self._center_in_goal_step = None   # 中心が最初にゴール区画へ入った歩（ΔT の測定用）
        self._episode_seed = 0         # リスポーンの擾乱を決定的に導く種（v2）
        self._respawn_count = 0        # 同一エピソード内のリスポーン回数 k（v2）
        self._start_heading = 0.0      # 開始点の方位 [deg]（リスポーンで再利用）
        # 規約判定（機体全体の内包）の記録用（裁定 R42-4）。評価ハーネスの実装を
        # 呼ぶので、ここでは導出結果のキャッシュだけを持つ（初回のゴール時に埋まる）。
        self._goal_footprint = None
        self._goal_bounds = None
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._action_lowpass = np.zeros(2, dtype=np.float64)   # ā_(−1) = 0（案 3）
        self._prev_dist_raw = None
        self._visited = set()
        self._step_count = 0
        # 🔴 R51 系の記録列（exp_020・記録のみ。**乱数を一切消費しない**）
        self._d0_max = None            # カリキュラムの D₀ 上限（None = 無効 ＝ 既定）
        self._path_len_m = 0.0         # 走行距離の総和 [m]（R51-2「歩あたりの進行距離」の分子）
        self._prev_xy = None           # 直前の真位置（None = 次の差分を数えない ＝ リスポーン直後）
        self._visit_steps = []         # 各区画の初訪問の歩（R51-2。割引後の訪問の取り分に要る）
        self._cell_entries = 0         # 走行で区画境界を跨いだ回数（R51-2。再入場を含む）
        self._min_d_since_respawn = -1  # 最後のリスポーン以降の min D（R51-1）
        # オドメトリ（自前センサの積分のみ。真の位置は使わない）
        self._odo_x = self._odo_y = self._odo_yaw = 0.0

        if seed is not None:
            gym.Env.reset(self, seed=seed)

    # ------------------------------------------------------------------
    def set_d0_max(self, d0_max) -> None:
        """カリキュラムの $D_0$ 上限を設定する（exp_020）。

        **`None` で無効**（既定）。**無効のとき `_next_maze_seed()` のコード経路は
        カリキュラム導入前と完全に同一**である（`experiments/exp_020_distance_curriculum/card.md`
        §2-3「不活性の bit 一致」の前提）。
        """
        self._d0_max = None if d0_max is None else int(d0_max)

    def _next_maze_seed(self) -> int:
        """学習用の迷路 seed（評価・検証に予約された帯は決定的に読み飛ばす）。

        **カリキュラム（exp_020）**: `self._d0_max` が設定されているときは、
        **開始点からゴールまでの区画距離 $D_0$ がその上限を超える面を読み飛ばす**。

        🔴 **この分岐は `self.np_random` を一切消費しない**（`generate_maze` は seed から
        決定的に作られ、`shortest_distances` は純粋な BFS である）。したがって
        **上限が `None` のときはもちろん、棄却が起きたときも乱数の消費順序は変わらない。**
        🔴 **棄却の判定に重い `_load_maze()`（XML 生成 ＋ MuJoCo モデル構築）は呼ばない。**
        実測費用は 1 候補 0.6 ms（D0 <= 4 で平均 7.9 候補 ＝ 約 5 ms／エピソード）。
        """
        while True:
            s = self.base_seed + self._episode_count
            self._episode_count += 1
            if s in _RESERVED_MAZE_SEEDS:
                continue
            if self._d0_max is not None:
                m = generate_maze(s, mode=self.maze_mode)
                d0 = int(shortest_distances(m["v_walls"], m["h_walls"])[tuple(m["start"])])
                if d0 > self._d0_max:
                    continue
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
                # 🔴 2026-08-14 是正（環境 v2 項目 5）: ここは **center_goal=False** だった。
                # 当時は柱を落とす条件が 16x16 専用だったので 6x6 では**無意味な引数**であり、
                # False でも True でも同じ XML が出ていた。**mjcf 側を一般化した結果、
                # この引数が初めて効くようになった**ので True にする。
                # → ゴール 2x2 の中心の柱 `post_3_3` が生成されなくなり、
                #   `mouse/maze6_gen.py` 冒頭の規格（内壁も柱もない広場）と一致する。
                mouse_euler=f"0 0 {heading}", center_goal=True, params=self.params)
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

        # --- 配置空間の自由空間マスク（裁定 R32・R34）----------------------
        # 機体中心が障害物（壁・柱）の**表面**から w_lat 以上離れている格子点が
        # 「機体が居られる位置」である。
        # 🔴 **距離場を解く領域はそれを 1 格子セルぶん広げたもの**にする
        # （離隔 w_lat − h_g·√2）。理由は下の第 2 段のコメント（帯の値を双線形に
        # 掴ませないため。実測で最大比 60.8 → 1.39 の差になる）。
        allowed = self._geo_allowed_mask(
            clearance=_ROBOT_LAT_HALF_WIDTH - _GEO_GRID_H * math.sqrt(2.0))
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

        # 第 2 段: 禁止帯（機体中心が居られない格子点）へ値を書き込む。**到達可能な点の
        # 値は変えない**（第 1 段で確定済み）ので、帯を通る近道は生じない。
        # 帯の値が要るのは、機体が居られる位置を囲む格子セルの隅が帯に入りうるためで、
        # 双線形補間が inf を掴まないようにするための措置である。
        #
        # 🔴 **帯の値は「機体が居られる位置」の取り出しには一切使われない**
        # （裁定 R38 → 2026-08-13 の是正）。第 1 段を解く領域を **1 格子セルぶん広げた
        # マスク**（離隔 w_lat − h_g·√2）にしてあるので、機体が居られる点 P（離隔 ≥ w_lat）を
        # 囲む 4 隅は**必ず第 1 段で解けている**（|P − 隅| ≤ h_g·√2 なので隅の離隔 ≥
        # w_lat − h_g·√2）。したがってここの延長は「格子全体を有限にするための後始末」で
        # あり、双線形が掴むことはない。
        #
        # ⚠️ **McShane 型（壁を無視した格子距離での 1-Lipschitz 拡張）は実測で棄却した**:
        # 比が 60.8 → 409.8 へ**悪化**した。理由は、**場 g そのものが直線距離について
        # 1-Lipschitz ではない**こと（壁を挟む 2 点は空間的に近くても測地的に遠い）。
        # McShane の構成は台の上で 1-Lipschitz な関数にしか使えない。**「1-Lipschitz 関数の
        # min は 1-Lipschitz」を、前提を確かめずに適用したのが誤りだった**（min の候補集合が
        # P に依存する罠と同じ型 ＝ 前提の未確認）。
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
                    continue                      # 第 1 段で解けた点の値は守る
                nd = du + w_list[k]
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        if not np.all(np.isfinite(dist)):
            n_bad = int(np.sum(~np.isfinite(dist)))
            raise AssertionError(
                f"延長後も値の付かない格子点が {n_bad} 個ある（実装の欠陥の可能性）")
        return dist.reshape(N, N)

    def _geo_obstacle_boxes(self) -> np.ndarray:
        """障害物（壁・柱）の平面上の軸並行箱を**モデルの実体から**読み出す（裁定 R34）。

        戻り値: (M, 4) の配列。各行 (cx, cy, hx, hy) ＝ 中心 [m] と半幅 [m]。

        **独立の再記述を作らない**（R34 の明示条件）。`mouse/mjcf.py` が
        `build_maze_robot_xml()` で実際に生成した geom（`post_*` / `v_wall_*` /
        `h_wall_*`）を MuJoCo モデルから読む。文書の記述（例: maze6_gen.py 冒頭の
        「ゴール中央は柱もない広場」）は**入力に使わない** — 実際にはゴール 2x2 の
        中心に `post_3_3` が立っており、文書とモデルが食い違っている
        （design.md「既知の環境特性」。是正は R11 バッチ）。

        壁上端の赤い geom（`*_wall_top_*`）は平面上の占有が本体と同一なので、
        名前で除外せずそのまま含める（min を取るので重複は無害。**除外条件を
        書かない＝取りこぼしの余地を作らない**）。
        """
        model = self.sim.model
        boxes = []
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if not name or not (name.startswith("post_") or name.startswith("v_wall_")
                                or name.startswith("h_wall_")):
                continue
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                raise AssertionError(f"障害物 geom {name} が箱ではない: {model.geom_type[g]}")
            if model.geom_bodyid[g] != 0:
                raise AssertionError(f"障害物 geom {name} が worldbody 直下にない"
                                     "（geom_pos を世界座標として読めない）")
            q = model.geom_quat[g]
            if not (abs(q[0] - 1.0) < 1e-12 and abs(q[1]) < 1e-12
                    and abs(q[2]) < 1e-12 and abs(q[3]) < 1e-12):
                raise AssertionError(f"障害物 geom {name} が回転している: quat={q}"
                                     "（軸並行の前提が崩れる）")
            boxes.append((float(model.geom_pos[g][0]), float(model.geom_pos[g][1]),
                          float(model.geom_size[g][0]), float(model.geom_size[g][1])))
        if not boxes:
            raise AssertionError("障害物 geom が 1 つも見つからない（実装の欠陥）")
        return np.asarray(boxes, dtype=np.float64)

    def _geo_allowed_mask(self, clearance: float = None) -> np.ndarray:
        """機体中心が到達しうる格子点の真偽マスク (N, N)（配置空間。裁定 R32・R34）。

        **判定は「障害物表面からの距離 ≥ w_lat」**（機体は半径 w_lat の円板）。
        `clearance` を渡すとその離隔で判定する（距離場を解く領域を 1 格子セルぶん
        広げるために使う。既定 None は w_lat ＝ 機体が実際に居られる集合）。
        軸並行な箱への平面距離は
            d = |( max(|x−cx|−hx, 0), max(|y−cy|−hy, 0) )|
        で、箱の外側では正確な最短距離になる（角では斜めに測られる ＝ **丸めが自動で従う**）。

        ⚠️ **機体モデルの限界**（design.md「マスクの是正」に条文として登録）:
        半径 w_lat の円板は**向きに依存しない近似**である。実機は向きによって張り出しが
        変わり前端は 0.0500 m あるが、これは含めない（(x, y, θ) の 3 次元配置空間は
        整形ポテンシャルの用途には過剰、という判断ごと登録した）。
        """
        N, H = _GEO_GRID_N, _GEO_GRID_H
        pos = np.arange(N) * H
        X = pos[:, None] * np.ones((1, N))
        Y = np.ones((N, 1)) * pos[None, :]
        dmin = np.full((N, N), np.inf)
        for cx, cy, hx, hy in self._geo_obstacle_boxes():
            dx = np.maximum(np.abs(X - cx) - hx, 0.0)
            dy = np.maximum(np.abs(Y - cy) - hy, 0.0)
            np.minimum(dmin, np.hypot(dx, dy), out=dmin)
        c = _ROBOT_LAT_HALF_WIDTH if clearance is None else float(clearance)
        return dmin >= c

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
        壁をまたぐ格子セルは**機体中心が到達できない領域**（閉じた壁の**中心線**から
        t_w/2 + w_lat = 0.0460 m 以内）にしか現れないためである。
        ⚠️ ラベルに注意（裁定 2026-08-13-R33）: 同じ離隔でも、壁**面**から測れば
        w_lat = 0.0400 m、壁**中心線**から測れば t_w/2 + w_lat = 0.0460 m である。
        本実装のマスクは壁を厚みゼロの線（区画境界線 = 壁中心線）として扱うので、
        判定に使うのは後者（_GEO_CLEARANCE）。
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

        Φ(P) = ρ·( g(reset 直後の真の位置) − g(P) )。g(reset 直後の真の位置) は
        `self._geo_start` として reset() がエピソード定数として保存する
        （擾乱後の真の位置で決めるので Φ₀ = 0 が構成上成立する）。

        ρ は条件 C' で**実際に適用する**面ごとの定数（`self._geo_rho_applied`。
        条件 C では 1.0）。
        定数倍なので Φ₀ = 0・位置だけの関数・c_prev 非依存はいずれも保たれ、
        Lipschitz 定数と 1 歩の跳びだけが ρ 倍される。
        """
        return self._geo_rho_applied * (self._geo_start - self._geodesic_value(x, y))

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

    def _goal_containment(self, x: float, y: float, yaw: float) -> bool:
        """**機体全体**がゴール 2×2 の内側に完全に入っているか（競技規約の判定）。

        🔴 これは**記録専用**であり、終了条件には使わない（裁定 R42-4・R11 バッチ項目 8）。
        環境の終了条件は**機体中心**が区画に入ったか（`goal_rate_env`）のままである。
        **終了条件を変えると学習の力学が変わる**（＝介入になる）ので、ここでは並記だけを
        足す（§9-19 の重み退避と同じ理屈で不介入）。

        判定の規則と外形は**評価ハーネスの実装をそのまま呼ぶ**
        （`competition/evaluator.py` の `body_footprint` / `body_fully_inside` /
        `goal_region_bounds`）。**独立の再記述を作らない** — R34 で確立した作法。
        外形はモデルの AABB 8 隅から導出されるので、寸法のハードコードも無い。
        """
        if self._goal_footprint is None:
            # 遅延 import（評価ハーネスは重い。既定 False の経路では読み込まない）
            from competition.evaluator import (body_footprint, goal_region_bounds)
            bid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
            self._goal_footprint = body_footprint(self.sim.model, self.sim.data, bid)
            self._goal_bounds = goal_region_bounds(SIZE, SIZE, self.params.cell_size)
        from competition.evaluator import body_fully_inside
        return bool(body_fully_inside(x, y, yaw, self._goal_footprint, self._goal_bounds))

    def _make_info(self, cell, collision, goal, sim_time) -> dict:
        x, y, _ = self.sim.privileged_pose()
        # 🔴 R51 系の記録（exp_020・**記録のみ。乱数を消費しない・報酬に一切入らない**）
        # 走行距離: リスポーン直後は `_prev_xy = None` にしてあるので、
        # **開始点への瞬間移動を走行距離に数えない**。
        if self._prev_xy is not None:
            self._path_len_m += math.hypot(x - self._prev_xy[0], y - self._prev_xy[1])
        self._prev_xy = (x, y)
        # 最後のリスポーン以降の min D（`_respawn_to_start()` が −1 に戻す）
        _d_now = int(self._dist_map.get(cell, -1))
        if _d_now >= 0:
            self._min_d_since_respawn = (_d_now if self._min_d_since_respawn < 0
                                         else min(self._min_d_since_respawn, _d_now))
        extra = {}
        if goal:
            # 🔴 規約判定の並記（裁定 R42-4）。**環境の判定（機体中心）でゴールとされた
            # 走行について、競技規約の判定（機体全体の内包）でも成立するか**を記録する。
            # 両者は**別の規則**である（design.md「goal_rate は環境の判定の量」）。
            # ゴールでない歩では計算しない（毎歩やると無駄。値は False で自明）。
            _x, _y, _yaw = self.sim.privileged_pose()
            extra["goal_contained_rule"] = self._goal_containment(_x, _y, _yaw)
        if self.geodesic_potential:
            # 条件 C・C' の記録の義務（裁定 R24-1）: 学習迷路ごとの
            # 1/ρ = g(start)/(cs·D₀) を残す。分布が予想と食い違うこと自体が
            # 結果の解釈に効くため、C（ρ=1）でも C'（ρ≠1）でも同じ量を記録する。
            denom = self.params.cell_size * self._d_start
            # 🔴 2026-08-14（R11 項目 6）: 鍵名を `geo_rho` → `geo_rho_applied` へ改名。
            # **値も計算式も変えていない**ので、**exp_012 の C・C' のログに残る旧名
            # `geo_rho` はこの `geo_rho_applied` と同じ量**（＝ 実際に適用した倍率）である。
            extra["geo_rho_applied"] = float(self._geo_rho_applied)
            extra["geo_inv_rho"] = (float(self._geo_start / denom) if denom > 0
                                    else float("nan"))
            extra["geo_g_start_m"] = float(self._geo_start)
        return {
            **extra,
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
            # 🔴 R51 系（exp_020・記録のみ）
            "path_len_m": float(self._path_len_m),
            "visit_steps": list(self._visit_steps),
            "cell_entries": int(self._cell_entries),
            "min_d_since_respawn": int(self._min_d_since_respawn),
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
        self._start_heading = float(heading)
        # リスポーンの擾乱を決定的に導く種（v2）。**seed が渡されればそれを使う**ので、
        # 同じ seed の走行はリスポーンを含めて完全に再現できる。
        self._episode_seed = int(seed) if seed is not None else int(
            self.np_random.integers(0, 2 ** 31 - 1))
        self._respawn_count = 0
        self._center_in_goal_step = None
        # R51 系の記録の初期化（記録のみ。乱数は引かない）
        self._path_len_m = 0.0
        self._prev_xy = None            # reset 直後の 1 歩目は差分を数えない
        self._visit_steps = [0]         # 開始区画を 0 歩目の初訪問として数える
        self._cell_entries = 1          # 開始区画への「入場」を 1 と数える
        self._min_d_since_respawn = -1
        lateral_reset = float(self.np_random.uniform(-_LATERAL_PERTURB_M, _LATERAL_PERTURB_M))
        dh_reset = float(self.np_random.uniform(-_HEADING_PERTURB_DEG, _HEADING_PERTURB_DEG))
        if self.geodesic_potential:
            self._geo_field = self._compute_geodesic_field()

        self.sim.full_reset(cell=start, heading_deg=heading)
        self._place_at_start(start, heading, lateral=lateral_reset, heading_delta=dh_reset)

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
            # 条件 C' の ρ（面ごと・エピソードごとの定数。裁定 R24-1）。
            #     ρ = cs·D₀ / g(start)
            # 分子は条件 E での総整形量（Φ の全変化量 = D₀ 区画 × cs）、分母は条件 C
            # でのそれ。**ρ 倍は Φ 全体に掛かる定数倍なので Φ₀ = 0 は保たれる。**
            self._geo_rho_applied = 1.0
            if self.geodesic_rho_scale:
                if not self._geo_start > 0.0:
                    raise AssertionError(
                        f"g(start) = {self._geo_start!r} が正でないので ρ を決められない"
                        "（スタート区画がゴール 2x2 に一致している可能性。実装の欠陥）")
                self._geo_rho_applied = (cs * self._d_start) / self._geo_start

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

    def _place_at_start(self, start, heading_deg: float,
                        lateral: float, heading_delta: float) -> None:
        """開始点へ機体を置く（reset とリスポーンで**同じ手続き**を使う）。"""
        cs = self.params.cell_size
        cx_m, cy_m = start[0] * cs + cs / 2, start[1] * cs + cs / 2
        hr = math.radians(heading_deg)
        x = cx_m + lateral * (-math.sin(hr))
        y = cy_m + lateral * math.cos(hr)
        nh = math.radians(heading_deg + heading_delta)
        root_jid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        qadr = self.sim.model.jnt_qposadr[root_jid]
        self.sim.data.qpos[qadr] = x
        self.sim.data.qpos[qadr + 1] = y
        self.sim.data.qpos[qadr + 3] = math.cos(nh / 2.0)
        self.sim.data.qpos[qadr + 4] = 0.0
        self.sim.data.qpos[qadr + 5] = 0.0
        self.sim.data.qpos[qadr + 6] = math.sin(nh / 2.0)
        # 速度も落とす（衝突直後の速度を引き継がない ＝ 係員が置き直すのと同じ）
        self.sim.data.qvel[:] = 0.0
        mujoco.mj_forward(self.sim.model, self.sim.data)

    def _respawn_to_start(self) -> None:
        """衝突リスポーン（v2 の対処案 (c)）。開始点へ戻し、エピソードは継続する。

        🔴 **擾乱は `(episode_seed, k)` から決定的に導く**（k = そのエピソードでの
        リスポーン回数。`corridor_eval._trial_seed` と同じ流儀。准教授 AUDIT_022）。
        **環境の乱数列を消費しない**ので、**乱数の消費順序への依存が消え、
        同じ seed の走行はリスポーンを含めて bit 単位で再現できる。**

        **`_visited` は消さない**（裁定済み。報酬ポンプを構成上作れなくするため）。
        """
        k = self._respawn_count
        self._respawn_count += 1
        rng = np.random.default_rng((int(self._episode_seed), int(k)))
        lateral = float(rng.uniform(-_LATERAL_PERTURB_M, _LATERAL_PERTURB_M))
        dh = float(rng.uniform(-_HEADING_PERTURB_DEG, _HEADING_PERTURB_DEG))
        start = tuple(self.maze["start"])
        self._place_at_start(start, self._start_heading, lateral, dh)
        # R51 系（記録のみ）: 瞬間移動を走行距離に数えないため基準を捨て、
        # 「最後のリスポーン以降の min D」の窓を切り直す。
        self._prev_xy = None
        self._min_d_since_respawn = -1
        # 区画の状態を開始点へ戻す（c_prev は reset と同じく「直前の区画が無い」状態）
        self._cell = start
        self._prev_cell = None
        # オドメトリは reset と同じ規約（係員が置き直した後は自分の姿勢を知っている）
        tx, ty, tyaw = self.sim.privileged_pose()
        self._odo_x, self._odo_y, self._odo_yaw = tx, ty, tyaw
        self._prev_dist_raw = None      # センサ差分は 1 歩ぶん未定義にする（reset と同じ）
        self._center_in_goal_step = None   # ΔT の測定は次の到達で取り直す

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        result = self.sim.step_control(float(action[0]) * self.params.voltage_limit,
                                       float(action[1]) * self.params.voltage_limit)
        self._step_count += 1
        self._update_odometry(self.sim.observation())

        x, y, yaw = self.sim.privileged_pose()
        cell = self._cell_of(x, y)
        if cell != self._cell:
            # 区画が変わった瞬間の直前区画を c_prev として保持する（連続 Φ 専用）。
            # 同じ区画に留まっている間は c_prev を更新しない（課題の仕様どおり）。
            self._prev_cell = self._cell
            self._cell = cell
            # 🔴 R51-2（exp_020・記録のみ）: **走行で区画境界を跨いだ回数**。
            # **再入場を含む**（同じ区画へ戻ってきたら、もう 1 回と数える）。
            # ⚠️ **リスポーンの瞬間移動は数えない**（`_respawn_to_start()` はここを通らない）。
            # **走行の速さの代理量にするため、駆動による通過だけを数える**のが定義である。
            self._cell_entries += 1
        center_in_goal = cell in GOAL_CELLS
        if self.goal_rule_containment:
            # 規約終端（v2）: **機体全体の内包**で成立。中心が区画内でなければ
            # 内包はあり得ないので、そのときだけ判定を呼ぶ（計算を無駄にしない）。
            goal_reached = bool(center_in_goal and self._goal_containment(x, y, yaw))
        else:
            goal_reached = center_in_goal
        # 規約終端の遅れ ΔT を測るための記録（中心が最初に入った歩）
        if center_in_goal and self._center_in_goal_step is None:
            self._center_in_goal_step = self._step_count
        physical_fail = bool(result["collision"] or result["tipped"])

        # 訪問報酬は**衝突した区画（＝この歩で居た区画）**に対して判定する。
        # リスポーンより先に評価するので、v1 と v2 で判定対象は同じである。
        visit_gain = 0.0
        if cell not in self._visited:
            self._visited.add(cell)
            visit_gain = self.visit_bonus
            self._visit_steps.append(self._step_count)   # R51-2（記録のみ）

        # 🔴 衝突リスポーン（v2）: **整形を計算する前に**開始点へ戻す。
        # こうすると Φ(s') = Φ(start) = 0 になり、**稼いだ整形をポテンシャル自身が
        # 取り返す**（−Φ_c が 1 度だけ入る）。報酬契約には触れていない。
        respawned = False
        if physical_fail and self.collision_respawn and not goal_reached:
            self._respawn_to_start()
            respawned = True
            x, y, _yaw2 = self.sim.privileged_pose()
            cell = self._cell

        potential = self._potential(cell, self._prev_cell, x, y)
        reward = self.gamma * potential - self._prev_potential - _TIME_PENALTY
        if goal_reached:
            reward += _GOAL_BONUS
        elif physical_fail:
            reward += self.collision_penalty
        reward += visit_gain
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

        # v2（リスポーン）では**衝突で終わらない**。終わるのはゴール成立か上限だけ。
        terminated = bool(goal_reached or (physical_fail and not respawned))
        truncated = bool((not terminated) and self._step_count >= self.episode_limit_steps)
        self._prev_action = np.asarray(action, dtype=np.float32)

        obs = self._make_observation()
        info = self._make_info(cell, physical_fail, goal_reached, result["sim_time"])
        if self.collision_respawn:
            info["respawned"] = bool(respawned)
            info["n_respawn"] = int(self._respawn_count)
        if goal_reached and self._center_in_goal_step is not None:
            # 🔴 規約終端の遅れ ΔT（中心が入ってから内包が成るまでの歩数）。
            # 机上模型（外接円半径 ÷ 1 歩の距離 = 0.96 m/s で 6.8 歩）と突き合わせる量。
            info["delta_t_containment"] = int(self._step_count - self._center_in_goal_step)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        self._sim_cache.clear()
        self._cache_order.clear()
