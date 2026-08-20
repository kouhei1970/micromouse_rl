"""
tests/test_fast_planner.py
================
`classic/fast_planner.py`（`plan_fast_run`）の検査。

`classic/ideal.py` の再利用が正しくかみ合っていること（`FastPlan.t_plan` の
合計が `ideal_time_for_path()` の `total` と一致すること）と、その場旋回が
混ざる経路で `FastPlan.steps` が経路追従⇄その場旋回を正しい順序・向きで
交互に持つことを実測する。
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest

from classic.fast_planner import PathBlock, SpinSegment, plan_fast_run
from classic.ideal import ideal_time_for_path
from classic.maze_map import ALL_DIRECTIONS, Direction, MazeMap, direction_between
from classic.maze_map import WallState as MapWallState

Cell = Tuple[int, int]

MAZE_W, MAZE_H = 6, 6
KNOWN_PATH: List[Cell] = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
GOAL_CELLS: List[Cell] = [(2, 2), (2, 3), (3, 2), (3, 3)]


def _carved_path_maze_and_walls(width: int, height: int, path: List[Cell]):
    """`path` に沿った区画だけを開けた回廊（`tests/test_classic_fast_run.py` と
    同じ作法）。既知の地図（`MazeMap`）と、同じ経路の真偽値の壁配列を対で返す。"""
    maze = MazeMap(width, height)
    for x in range(width):
        for y in range(height):
            for d in ALL_DIRECTIONS:
                if maze.neighbor(x, y, d) is not None:
                    maze.set_wall(x, y, d, MapWallState.WALL)

    v_walls = np.ones((width + 1, height), dtype=bool)
    h_walls = np.ones((width, height + 1), dtype=bool)
    for a, b in zip(path[:-1], path[1:]):
        d = direction_between(a, b)
        maze.set_wall(a[0], a[1], d, MapWallState.OPEN)
        if d is Direction.E:
            v_walls[a[0] + 1, a[1]] = False
        elif d is Direction.W:
            v_walls[a[0], a[1]] = False
        elif d is Direction.N:
            h_walls[a[0], a[1] + 1] = False
        else:  # Direction.S
            h_walls[a[0], a[1]] = False
    return maze, v_walls, h_walls


@pytest.fixture(scope="module")
def carved_maze_and_walls():
    return _carved_path_maze_and_walls(MAZE_W, MAZE_H, KNOWN_PATH)


# ==========================================================================
# 1. t_plan の合計が ideal_time_for_path().total と一致すること
#    （半径探索・配分は ideal.py に委ね、min_time の再実行だけで格子を
#    取り出しているので、値そのものが変わってはいけない）
# ==========================================================================
def test_t_plan_matches_ideal_time_for_path_total(carved_maze_and_walls):
    maze, v_walls, h_walls = carved_maze_and_walls
    plan = plan_fast_run(maze, start=(0, 0), goals=GOAL_CELLS, start_heading=Direction.N)
    assert plan is not None

    result = ideal_time_for_path(plan.cells, v_walls, h_walls, Direction.N, mode="slalom")
    assert plan.t_plan == pytest.approx(result.total, abs=1e-9), (
        f"FastPlan.t_plan({plan.t_plan}) が ideal_time_for_path().total({result.total}) と一致しない"
    )
    assert plan.n_turns == result.n_turns
    assert plan.n_forced_spins == sum(1 for tp in result.turns if tp.radius <= 0.0)
    # 経路がこの迷路の90°ターン(半径のあるslalom)だけで構成されるなら
    # 強制その場旋回は無いはず(このハンドメイド迷路の構成上の前提)。
    assert plan.n_forced_spins == 0
    assert len(plan.steps) == 1
    assert isinstance(plan.steps[0], PathBlock)


# ==========================================================================
# 2. 到達不能（未知区画=壁として悲観に扱う）なら None を返す
# ==========================================================================
def test_returns_none_when_unreachable_pessimistically():
    maze = MazeMap(MAZE_W, MAZE_H)  # 全マス未知（外周だけWALL）
    plan = plan_fast_run(maze, start=(0, 0), goals=GOAL_CELLS, start_heading=Direction.N)
    assert plan is None


# ==========================================================================
# 3. 強制その場旋回（開始方位が最初の移動と180°逆）が先頭のSpinSegmentになる
# ==========================================================================
def test_forced_spin_at_start_becomes_a_leading_spin_segment(carved_maze_and_walls):
    maze, v_walls, h_walls = carved_maze_and_walls
    # 経路の最初の移動は北(N)。開始方位を南(S)にすると、手前の直線が
    # 長さ0になり180°の弧が作れないため、classic/ideal.py の docstring
    # 「半径が0になる場合の扱い」により強制その場旋回になる。
    plan = plan_fast_run(maze, start=(0, 0), goals=GOAL_CELLS, start_heading=Direction.S)
    assert plan is not None
    assert plan.n_forced_spins == 1
    assert len(plan.steps) == 2

    spin = plan.steps[0]
    path = plan.steps[1]
    assert isinstance(spin, SpinSegment)
    assert isinstance(path, PathBlock)
    assert abs(abs(spin.delta_theta) - math.pi) < 1e-9, "180度の旋回になっていない"
    assert spin.psi_start == pytest.approx(math.radians(-90.0), abs=1e-9)  # 南向き

    # 経路追従区間の開始向きは、旋回後の北向きと一致するはず
    # （2πの整数倍の差は許容: atan2で正規化して比較する）
    expected = math.radians(90.0)
    diff = math.atan2(math.sin(path.psi_start - expected), math.cos(path.psi_start - expected))
    assert abs(diff) < 1e-6

    # ideal_time_for_path と合計時間が一致すること（南向き開始で計算し直す）
    result = ideal_time_for_path(plan.cells, v_walls, h_walls, Direction.S, mode="slalom")
    assert plan.t_plan == pytest.approx(result.total, abs=1e-9)


# ==========================================================================
# 4. 既にゴールにいる(trivial)なら steps が空で t_plan=0.0
# ==========================================================================
def test_trivial_plan_when_already_at_goal(carved_maze_and_walls):
    maze, _v_walls, _h_walls = carved_maze_and_walls
    plan = plan_fast_run(maze, start=(2, 3), goals=[(2, 3)], start_heading=Direction.N)
    assert plan is not None
    assert plan.steps == ()
    assert plan.t_plan == 0.0
    assert plan.cells == ((2, 3),)


# ==========================================================================
# 5. friction_use（摩擦円の使用率 u）: 任務指示「1. 摩擦円の使用率 u を計画の
#    引数にする」の検査。u=1.0（既定）は従来と数値まで完全に一致し、u を
#    下げると経路・半径（cells/n_turns/n_forced_spins）は変えずに t_plan だけ
#    伸びること（単調性）を見る。
# ==========================================================================
def test_friction_use_default_matches_pre_existing_behavior(carved_maze_and_walls):
    """friction_use を明示的に1.0で渡しても、省略時と完全に同じ結果になる
    （既定値であることの直接確認）。"""
    maze, _v_walls, _h_walls = carved_maze_and_walls
    plan_default = plan_fast_run(maze, start=(0, 0), goals=GOAL_CELLS, start_heading=Direction.N)
    plan_explicit = plan_fast_run(
        maze, start=(0, 0), goals=GOAL_CELLS, start_heading=Direction.N, friction_use=1.0)
    assert plan_default is not None and plan_explicit is not None
    assert plan_explicit.t_plan == plan_default.t_plan
    assert plan_explicit.cells == plan_default.cells


def test_friction_use_lowers_speed_without_changing_the_route(carved_maze_and_walls):
    """u を 1.00→0.90→0.75→0.60 と下げるにつれ t_plan が単調に伸びること
    （note_031 段5の実測: maze_41001 で同じ傾向）。経路そのもの（cells・
    ターン数・強制その場旋回の数）は u に関わらず不変であること
    （u は速度計画だけを変え、半径・経路選択は ideal.py の結果をそのまま
    再利用するため）も併せて確認する。"""
    maze, _v_walls, _h_walls = carved_maze_and_walls
    levels = (1.00, 0.90, 0.75, 0.60)
    plans = [
        plan_fast_run(maze, start=(0, 0), goals=GOAL_CELLS, start_heading=Direction.N, friction_use=u)
        for u in levels
    ]
    assert all(p is not None for p in plans)

    t_plans = [p.t_plan for p in plans]
    assert t_plans == sorted(t_plans), f"u を下げても t_plan が単調に伸びなかった: {list(zip(levels, t_plans))}"
    assert t_plans[0] < t_plans[-1], "u=1.00とu=0.60でt_planに差が無い"

    for p in plans[1:]:
        assert p.cells == plans[0].cells
        assert p.n_turns == plans[0].n_turns
        assert p.n_forced_spins == plans[0].n_forced_spins


def test_friction_use_out_of_range_raises(carved_maze_and_walls):
    maze, _v_walls, _h_walls = carved_maze_and_walls
    for bad_u in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            plan_fast_run(maze, start=(0, 0), goals=GOAL_CELLS, start_heading=Direction.N, friction_use=bad_u)


# ==========================================================================
# 6. clearance_margin_m（幾何の掃引と壁・柱のあいだに残す余裕）: 任務指示
#    「余裕を実測に合わせる」§1「margin を計画の引数にする」の検査。
#    既定 0.005（5mm）は省略時と完全に同一（回帰検査）、大きくすると
#    t_plan は縮まない（単調に伸びる、または経路自体がその場旋回へ
#    降格して伸びる）ことを見る。半径そのものへの効き方は
#    tests/test_geometry.py::test_larger_margin_shrinks_the_max_feasible_radius
#    が `classic.geometry.max_feasible_radius` を直接見ている。
# ==========================================================================
def test_clearance_margin_m_default_matches_pre_existing_behavior(carved_maze_and_walls):
    """clearance_margin_m を明示的に0.005で渡しても、省略時と完全に同じ結果になる
    （既定値であることの直接確認。旧引数名 `margin` からの改名で既定挙動が
    変わっていないことの回帰検査でもある）。"""
    maze, _v_walls, _h_walls = carved_maze_and_walls
    plan_default = plan_fast_run(maze, start=(0, 0), goals=GOAL_CELLS, start_heading=Direction.N)
    plan_explicit = plan_fast_run(
        maze, start=(0, 0), goals=GOAL_CELLS, start_heading=Direction.N, clearance_margin_m=0.005)
    assert plan_default is not None and plan_explicit is not None
    assert plan_explicit.t_plan == plan_default.t_plan
    assert plan_explicit.cells == plan_default.cells
    assert plan_explicit.n_turns == plan_default.n_turns
    assert plan_explicit.n_forced_spins == plan_default.n_forced_spins


def test_clearance_margin_m_default_matches_ideal_time_for_path_default(carved_maze_and_walls):
    """既定の `clearance_margin_m=0.005` で作った計画の `t_plan` が、
    `ideal_time_for_path()` を既定の `margin`（同じ 0.005）で呼んだ結果の
    `total` と一致すること（`plan_fast_run` 側の改名が `classic/ideal.py` の
    既定値の意味を変えていないことの回帰検査）。"""
    maze, v_walls, h_walls = carved_maze_and_walls
    plan = plan_fast_run(maze, start=(0, 0), goals=GOAL_CELLS, start_heading=Direction.N)
    assert plan is not None

    result_explicit_margin = ideal_time_for_path(
        plan.cells, v_walls, h_walls, Direction.N, mode="slalom", margin=0.005)
    assert plan.t_plan == pytest.approx(result_explicit_margin.total, abs=1e-9)


def test_larger_clearance_margin_m_does_not_shorten_t_plan(carved_maze_and_walls):
    """`clearance_margin_m` を 5mm→30mm と上げるにつれ、`t_plan` が縮まない
    （半径が縮む、または弧がその場旋回へ降格することで速度計画が控えめになる）
    ことを見る。`friction_use` の単調性検査（検査5）と対になる、`margin` 側の
    「計画レベル」での効果確認（半径そのものへの効果は tests/test_geometry.py
    が直接見ている）。"""
    maze, _v_walls, _h_walls = carved_maze_and_walls
    levels_mm = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
    plans = [
        plan_fast_run(
            maze, start=(0, 0), goals=GOAL_CELLS, start_heading=Direction.N,
            clearance_margin_m=mm / 1000.0,
        )
        for mm in levels_mm
    ]
    assert all(p is not None for p in plans)

    t_plans = [p.t_plan for p in plans]
    print(f"\n[実測] clearance_margin_m[mm] -> t_plan[s]: {list(zip(levels_mm, t_plans))}")
    assert t_plans == sorted(t_plans), (
        f"clearance_margin_m を上げても t_plan が縮まなかった(想定外の増減): "
        f"{list(zip(levels_mm, t_plans))}"
    )
    assert t_plans[-1] > t_plans[0], "margin=5mmとmargin=30mmでt_planに差が無い"

    # 経路そのもの（区画列）は margin に関わらず不変（悲観歩数マップの経路選択は
    # 幾何の余裕を見ないため）。
    for p in plans[1:]:
        assert p.cells == plans[0].cells


def test_clearance_margin_m_non_positive_raises(carved_maze_and_walls):
    maze, _v_walls, _h_walls = carved_maze_and_walls
    for bad_margin in (0.0, -0.001):
        with pytest.raises(ValueError):
            plan_fast_run(
                maze, start=(0, 0), goals=GOAL_CELLS, start_heading=Direction.N,
                clearance_margin_m=bad_margin,
            )
