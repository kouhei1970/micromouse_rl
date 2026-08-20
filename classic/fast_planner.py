"""classic/fast_planner.py — 最短走行(FAST/RETURN2)の速度プロファイル計画器

`research_notes/note_031_profile_planner_and_eta.md`（判定条文・段 3/段 4 の結果）
を受け、Phase.FAST・Phase.RETURN2 をコマンド方式（1コマンド発行→完了待ち→次）から
速度プロファイル追従（`classic/tracker.py` の `ProfileTracker`）へ差し替えるための
計画器である。マウスが**学習した地図**（`classic.maze_map.MazeMap`。真の壁ではない）
から最短経路を引き、`classic/ideal.py` と同じ手順で

    経路（区画列） → 幾何プリミティブ列（直線/円弧） → 曲率 kappa(s) → 最小時間の速度計画 v(s)

を作る。同じ計算を二度書かないため、経路選択・半径の幾何最適化・共有直線の配分
（比例/先取り、速い方を採用）は `classic.ideal.true_shortest_path`/`ideal_time_for_path`
にそのまま委ねる。

## 未知区画の扱い（悲観）

マウス自身の地図（`WallState.UNKNOWN`/`WALL`/`OPEN` の3値）は、UNKNOWN を「壁がある」
として扱う（`classic.flood.is_passable` の `FloodMode.PESSIMISTIC` と同じ判定。
`classic/explorer.py` が既に最短走行へ移る条件をこの判定で確かめているのと同じ考え方）。
到達できないなら（`true_shortest_path` が `classic.route.NoRouteError` を送出したら）
`None` を返す。

## 🔴 真の壁・真値姿勢は一切読まない

本モジュールが入力に取るのは `MazeMap`（マウス自身が書いた地図）だけである。
`maze_info["xml_path"]` の MJCF・評価器の `v_walls`/`h_walls`（真値）は一切参照しない
（`classic/` パッケージの規約。`classic/explorer.py` モジュール docstring と同じ根拠）。

## `classic/ideal.py` の内部関数の再利用について

`ideal_time_for_path()` は「経路全体の合計時間」（`IdealResult.total`/`by_kind`）は
返すが、追従に要る**弧長 s ごとの `v_ref`/`kappa_ref` の格子**（`classic.profile.
IdealTime.s_grid`/`v_grid`/`kappa_grid`）までは外へ出さない（内部でブロックごとに
`min_time()` を呼び、その結果を合算・破棄している）。そこで本モジュールは、
`ideal_time_for_path()` が確定させた答え（`IdealResult.turns` — 各ターンの半径・
旋回角・配分）を使って、同じ幾何ブロックを `classic.ideal._geometry_blocks`
（`classic/ideal.py` 自身が「`tests/test_ideal.py` の衝突検査と両方から使う共通部品」
と明記している、意図して外部から再利用可能にしてある内部ヘルパ）で再構成し、
ブロックごとに `min_time()` を呼び直して格子を取り出す。

これは「同じ計算をもう一度行う」のではない。半径探索（二分探索×障害物との厳密な
干渉判定）・共有直線の比例配分（緩和法）という**重い計算**は `ideal_time_for_path()`
の結果（`IdealResult.turns`）を再利用するだけで再実行しない。再実行するのは
`min_time()`（区分定数の曲率列に対する前進/後退パスのスイープ）1回だけであり、
これは `ideal.py` が内部で最初から呼んでいた計算を、`ideal.py` が返さない中間結果
（格子）を得るために呼び直しているだけである。

## 経路追従とその場旋回の交互実行（`FastPlan`）

`mode="slalom"` で `ideal_time_for_path()` を呼ぶ（曲がるところは円弧で通り、
区間の切れ目でも止まらない）。ただし `classic/ideal.py` の docstring
「半径が0になる場合の扱い」のとおり、経路の最初のターンや180°折返しでは
半径が幾何的に0まで削られ、その場旋回が混ざることがある。この場合
`IdealResult.turns` の該当ターンは `radius<=0.0` になり、`_geometry_blocks` が
そこでブロックを分割する。`FastPlan.steps` は経路追従区間（`PathBlock`。
`ProfileTracker.load_plan()` にそのまま渡せる `s_grid`/`v_ref`/`kappa_ref`/
`psi_start`）とその場旋回区間（`SpinSegment`。`ProfileTracker.load_spin_plan()`
にそのまま渡せる `delta_theta`/`psi_start`）を**実行順に**持つ（先頭は必ず
`PathBlock`か`SpinSegment`のどちらかで、両者が交互に現れる。強制その場旋回が
一度も無い経路なら `steps` は `PathBlock` 1 個だけになる）。

各ステップの `psi_start` は、幾何（`_geometry_blocks`/`classic.geometry.poses_along`）
から**計画時点で**厳密に計算した値であり、実行中の推測航法の実測値ではない
（`tests/test_tracker.py::test_spin_turn_needs_no_settling_wait` が引き継ぎ検査で
使っている作法と同じ: 計画は開ループで作り、実行側のフィードバック
（`ProfileTracker` の `kp_psi`）がずれを吸収する）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from classic.geometry import poses_along, to_profile_segments
from classic.ideal import _geometry_blocks, _turns_and_runs, ideal_time_for_path, true_shortest_path
from classic.maze_map import Direction, MazeMap, WallState
from classic.profile import min_time, spin_turn_time, vehicle_limits
from classic.route import NoRouteError
from mouse.params import RobotParams

__all__ = ["PathBlock", "SpinSegment", "FastPlan", "plan_fast_run"]

Cell = Tuple[int, int]


# ============================================================================
# 1. FastPlan の要素
# ============================================================================
@dataclass(frozen=True)
class PathBlock:
    """経路追従の1区間。`ProfileTracker.load_plan(s_grid, v_ref, kappa_ref,
    psi_start=...)` にそのまま渡せる形。"""

    s_grid: Tuple[float, ...]
    v_ref: Tuple[float, ...]
    kappa_ref: Tuple[float, ...]
    psi_start: float  # [rad]。区間の開始姿勢の向き（計画時点の幾何から厳密に計算）。
    t_plan: float      # この区間だけの理想時間 [s]（診断用）。


@dataclass(frozen=True)
class SpinSegment:
    """その場旋回の1区間。`ProfileTracker.load_spin_plan(delta_theta,
    psi_start=...)` にそのまま渡せる形。"""

    delta_theta: float  # [rad]。正=左/反時計回り（`classic.ideal.TurnPlan.delta_theta` と同符号）。
    psi_start: float     # [rad]。旋回を始める時点の向き。
    t_plan: float         # この区間だけの理想時間 [s]（`classic.profile.spin_turn_time`。診断用）。


FastPlanStep = Union[PathBlock, SpinSegment]


@dataclass(frozen=True)
class FastPlan:
    """`plan_fast_run()` の返り値。"""

    steps: Tuple[FastPlanStep, ...]  # 実行順（PathBlock/SpinSegmentが交互に現れる）
    t_plan: float                      # 総理想時間 [s]（全ステップの合計）
    cells: Tuple[Cell, ...]            # 経路の区画列（診断用。学習地図上の悲観最短経路）
    n_turns: int                       # ターン数（診断用）
    n_forced_spins: int                # そのうち強制的にその場旋回になった数（診断用）


# ============================================================================
# 2. マウス自身の地図 → 真偽値の壁配列（未知=壁。悲観）
# ============================================================================
def _bool_walls_from_maze(maze: MazeMap) -> Tuple[np.ndarray, np.ndarray]:
    """マウスが学習した地図から、`classic.ideal` の入口が要求する真偽値の
    壁配列を作る。未知(UNKNOWN)は壁として扱う（悲観。`classic.flood.is_passable`
    の `FloodMode.PESSIMISTIC` — OPEN以外はすべて通れない — と同じ判定）。
    """
    v = maze.v_walls != int(WallState.OPEN)
    h = maze.h_walls != int(WallState.OPEN)
    return v, h


# ============================================================================
# 3. 本体
# ============================================================================
def plan_fast_run(
    maze: MazeMap,
    start: Cell,
    goals: Sequence[Cell],
    start_heading: Direction,
    params: Optional[RobotParams] = None,
    clearance_margin_m: float = 0.005,
    allocation: str = "best",
    friction_use: float = 1.0,
) -> Optional[FastPlan]:
    """マウスが学習した地図から最短走行の速度プロファイル計画を作る。

    未知の区画は壁があるものとして扱う（悲観）。`start` から `goals` のいずれかへ
    既知の壁だけで到達できないなら `None` を返す（呼び出し側はコマンド方式へ
    落ちること — `classic/explorer.py` の安全弁）。

    `params` は追従制御が使う車両パラメータ（`classic.tracker.ProfileTracker` と
    同じ `RobotParams`）。省略すると既定値（`RobotParams()`）を使う。

    `clearance_margin_m`（既定 `0.005` = 5mm）は `classic.geometry.
    max_feasible_radius` の `margin` として `ideal_time_for_path()` へそのまま
    渡す、機体の掃引と壁・柱のあいだに残す設計上の余裕である。既定 5mm は
    実測の横ずれ（10〜20mm、`research_notes/note_031_profile_planner_and_eta.md`
    「北向き開始が頭打ちになる」節）に対して足りていない疑いがあり、本引数で
    掃引できるようにした（任務指示 2026-08-20）。上げるほど通れる半径は小さく
    なり `t_plan` は伸びる（`friction_use` を下げるのと同じ構造のトレードオフ）。

    🔴 `classic/ideal.py::ideal_time_for_path`（判定の分母 `T_ideal` の計算元。
    `experiments/exp_025_s4_slalom/ideal_table.json` の作成に使う独立した
    呼び出し）は本引数と無関係に、常に既定の `margin=0.005` で呼ばれる
    （`classic/ideal.py` 自体・その既定値は変更していない）。本引数が効くのは
    **本関数が自分の計画（`t_plan`）を作るために呼ぶ `ideal_time_for_path()`**
    だけであり、分母 `T_ideal` を動かすことは無い。

    `friction_use`（記号 `u`）は速度計画が使ってよい摩擦円の割合
    （`research_notes/note_031_profile_planner_and_eta.md` §「摩擦円の使用率は
    η の上限を決める」）。`vehicle_limits()` が返す `A_TR`（前後加速度限界）・
    `A_LAT`（横加速度限界）の**両方に同じ `u` を掛けてから** `min_time` へ渡す
    （`0 < u <= 1.0`。既定 `1.0` は摩擦円を 100% 使う従来どおりの計画で、
    数値まで完全に同一になる — `1.0` を掛けても浮動小数点は変化しないため）。
    `u` を下げると、同じ経路・同じ半径のまま巡航速度と加減速だけが控えめになり、
    実行時の推測航法の誤差に対する通路の余地（壁までの距離的余裕）が増える
    代わりに `t_plan` は伸びる（exp_026 の実測: maze_41001 で
    u=1.00→9.175s, u=0.90→9.671s, u=0.75→10.594s, u=0.60→11.845s。概ね
    `1/sqrt(u)` で伸びる）。

    🔴 `V_TOP`（モータが出せる最高速度）・`alpha_yaw_max`（その場旋回の角加速度
    限界）には `u` を掛けない。`u` は**タイヤと路面の摩擦円に対してどれだけ
    余裕を残すか**という意味の量（横滑り・制動距離の余地）であり、モータが
    出せる電圧・トルクの頭打ち（`V_TOP`・`alpha_yaw_max` の由来）とは別物
    だから。したがって `spin_turn_time()`（その場旋回。`alpha_yaw_max`・
    `V_TOP` にしか依存しない）の挙動は `u` の値に関わらず変わらない。

    🔴 `classic/ideal.py::ideal_time_for_path`（判定の分母 `T_ideal` の計算元）
    は本関数の内部で呼ぶが、`vehicle_limits()` を `u` 無し（既定 `1.0` 相当）で
    使う既存の呼び出しのままであり、本引数の影響を一切受けない
    （`classic/ideal.py` 自体は変更していない）。`u` が効くのは、本関数がその
    結果（半径）を再利用して**追従用の速度プロファイルだけを引き直す**
    `min_time()` の呼び出しだけである。
    """
    if not (0.0 < friction_use <= 1.0):
        raise ValueError(f"friction_use は (0.0, 1.0] の範囲で指定してください: {friction_use}")
    if not (clearance_margin_m > 0.0):
        raise ValueError(
            f"clearance_margin_m は正の値で指定してください: {clearance_margin_m}"
        )

    goals = list(goals)
    v_walls, h_walls = _bool_walls_from_maze(maze)

    try:
        cells = true_shortest_path(v_walls, h_walls, start, goals, start_heading)
    except NoRouteError:
        return None

    if len(cells) <= 1:
        # 既にゴール/スタートにいる（trivial）。ステップの無い計画を返す
        # （呼び出し側はこれを「即座に完了」として扱う。安全弁の対象ではない）。
        return FastPlan(steps=(), t_plan=0.0, cells=tuple(cells), n_turns=0, n_forced_spins=0)

    limits = vehicle_limits(params)
    if friction_use != 1.0:
        # A_TR・A_LAT にのみ u を掛ける（docstring 参照）。V_TOP・alpha_yaw_max は
        # vehicle_limits() の返り値のまま変えない。
        limits = replace(
            limits,
            A_TR=limits.A_TR * friction_use,
            A_LAT=limits.A_LAT * friction_use,
        )

    # 半径の幾何最適化・共有直線の配分（比例/先取り、速い方）は ideal.py に委ねる
    # （モジュール docstring「classic/ideal.py の内部関数の再利用について」参照）。
    result = ideal_time_for_path(
        cells, v_walls, h_walls, start_heading, mode="slalom",
        margin=clearance_margin_m, allocation=allocation,
    )

    # 確定済みの半径（result.turns）から、同じ幾何ブロックを再構成する
    # （半径探索・比例配分は再実行しない。_turns_and_runs は直線ランの長さ
    # `runs` を得るためだけに呼ぶ軽い計算）。
    _turns, runs = _turns_and_runs(cells, start_heading)
    geo_blocks, block_starts = _geometry_blocks(cells, start_heading, result.turns, runs)

    spin_turns = [tp for tp in result.turns if tp.radius <= 0.0]
    if len(geo_blocks) != len(spin_turns) + 1:
        # `_geometry_blocks` は radius<=0.0 のターンごとにブロックを1つ増やす
        # （モジュール docstring参照）。食い違えば ideal.py 側の前提が変わった
        # ことを意味するので、黙って進めず早期に知らせる。
        raise AssertionError(
            "幾何ブロック数がその場旋回の数と整合しません"
            f"(blocks={len(geo_blocks)}, forced_spins={len(spin_turns)})"
        )

    steps: List[FastPlanStep] = []
    t_plan = 0.0
    for i, block in enumerate(geo_blocks):
        psi_start = block_starts[i].theta
        if block:
            segs = to_profile_segments(block)
            it = min_time(segs, limits, v_start=0.0, v_end=0.0)
            steps.append(PathBlock(
                s_grid=tuple(it.s_grid), v_ref=tuple(it.v_grid),
                kappa_ref=tuple(it.kappa_grid), psi_start=psi_start, t_plan=it.total,
            ))
            t_plan += it.total
            end_theta = poses_along(block, block_starts[i])[-1].theta
        else:
            end_theta = psi_start
        if i < len(spin_turns):
            tp = spin_turns[i]
            st = spin_turn_time(tp.delta_theta, limits)
            steps.append(SpinSegment(delta_theta=tp.delta_theta, psi_start=end_theta, t_plan=st.time))
            t_plan += st.time

    n_turns = len(result.turns)
    n_forced_spins = len(spin_turns)
    return FastPlan(
        steps=tuple(steps), t_plan=t_plan, cells=tuple(cells),
        n_turns=n_turns, n_forced_spins=n_forced_spins,
    )
