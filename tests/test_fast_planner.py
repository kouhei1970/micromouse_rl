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
