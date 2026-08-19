"""
tests/test_classic_route.py
================
classic/maze_map.py・classic/flood.py・classic/route.py（S0: 地図と経路。
純計算・走行なし）の単体テスト。pytest で実行する:

    .venv/bin/python -m pytest tests/test_classic_route.py -v

検証内容（research_notes/note_030_classical_rebuild_plan.md §3 S0 の出口条件）:
  1. 手で作った 4x4 迷路で、歩数マップと最短経路が手計算と一致すること
  2. 楽観／悲観で結果が変わる迷路を作り、両者が実際に違うことを確かめる
     （空振りしない検査。片方の呼び出しを忘れても片方だけでは通らない構成）
  3. ターン列への変換が、直進の連結とターン種別で正しいこと
  4. 同じ歩数の経路が複数あるときに、ターン数が少ない方が選ばれること
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

from classic.flood import FloodMode, compute_flood, is_reachable_known
from classic.maze_map import ALL_DIRECTIONS, Direction, MazeMap, WallState, opposite
from classic.route import (
    Command,
    CommandType,
    NoRouteError,
    path_to_commands,
    plan_route,
    shortest_path,
)


def fill_all_interior_walls(maze: MazeMap, state: WallState) -> None:
    """テスト用ヘルパ: 迷路内部の壁を全て state で埋める（外周は触らない）。

    「壁は隣接 2 区画の共有物」なので、片側から書き込めば十分だが、ここでは
    分かりやすさのため全区画・全方位を総当たりで書く（set_wall は同じ壁への
    重複書き込みを許容する）。
    """
    for x in range(maze.width):
        for y in range(maze.height):
            for d in ALL_DIRECTIONS:
                if maze.neighbor(x, y, d) is not None:
                    maze.set_wall(x, y, d, state)


def open_route(maze: MazeMap, cells) -> None:
    """cells で与えた区画列に沿って隣接壁を OPEN にする（両方向とも）。"""
    for a, b in zip(cells[:-1], cells[1:]):
        for d in ALL_DIRECTIONS:
            nb = maze.neighbor(a[0], a[1], d)
            if nb == b:
                maze.set_wall(a[0], a[1], d, WallState.OPEN)
                maze.set_wall(b[0], b[1], opposite(d), WallState.OPEN)
                break
        else:
            raise AssertionError(f"{a} と {b} は隣接していません（テストの迷路定義ミス）")


# ==========================================================================
# 1. 手計算との一致（4x4・全壁既知・経路が一意に定まる迷路）
# ==========================================================================
def _make_forced_corridor_maze() -> MazeMap:
    """(0,0) から北へ3区画、東へ3区画で (3,3) に至る一本道だけを開けた 4x4 迷路。
    他の壁は全て既知の WALL にする（経路が一意に定まるようにするため）。"""
    maze = MazeMap(width=4, height=4)
    fill_all_interior_walls(maze, WallState.WALL)
    open_route(maze, [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3)])
    return maze


def test_flood_matches_hand_calculation():
    maze = _make_forced_corridor_maze()
    goal = (3, 3)

    # 手計算: ゴールからの歩数は経路上を逆にたどるだけ。
    expected = {
        (3, 3): 0,
        (2, 3): 1,
        (1, 3): 2,
        (0, 3): 3,
        (0, 2): 4,
        (0, 1): 5,
        (0, 0): 6,
    }
    for mode in (FloodMode.OPTIMISTIC, FloodMode.PESSIMISTIC):
        dist = compute_flood(maze, [goal], mode)
        for cell, steps in expected.items():
            assert dist[cell] == steps, (mode, cell, dist[cell], steps)
        # 一本道以外に道が無い区画（例えば (1,0)）は、全周囲 WALL 既知なので
        # 楽観・悲観どちらでも今回は到達不能（周りを完全に壁で囲ってあるため）。
        assert dist[(1, 0)] == float("inf")


def test_shortest_path_matches_hand_calculation():
    maze = _make_forced_corridor_maze()
    goal = (3, 3)
    expected_path = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3)]

    for mode in (FloodMode.OPTIMISTIC, FloodMode.PESSIMISTIC):
        path = shortest_path(maze, start=(0, 0), goals=[goal], mode=mode)
        assert path == expected_path, (mode, path)


def test_shortest_path_unreachable_raises():
    maze = MazeMap(width=4, height=4)
    fill_all_interior_walls(maze, WallState.WALL)  # 全部壁 = start はどこにも行けない
    with pytest.raises(NoRouteError):
        shortest_path(maze, start=(0, 0), goals=[(3, 3)], mode=FloodMode.PESSIMISTIC)


# ==========================================================================
# 2. 楽観／悲観で結果が実際に変わることの確認（空振りしない検査）
# ==========================================================================
def _make_shortcut_vs_detour_maze() -> MazeMap:
    """(0,0)-(3,0) 間に 2 通りの道を仕込む:
      - 近道: y=0 の行を直進 3 区画（壁の状態を一切書き込まない = 未知のまま）
      - 迂回: y=0 -> y=3 -> x=3 -> y=0 と回る、既知の開いた道のり 9 区画
    近道以外の壁は全て既知の WALL にしておく（近道の 3 本の壁だけが未知）。
    """
    maze = MazeMap(width=4, height=4)
    fill_all_interior_walls(maze, WallState.WALL)

    detour = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 3), (2, 3), (3, 3),
        (3, 2), (3, 1), (3, 0),
    ]
    open_route(maze, detour)

    # 近道 (0,0)-(1,0)-(2,0)-(3,0) の壁は既知にしない（未知のまま残す）。
    # fill_all_interior_walls で WALL にされているので、明示的に UNKNOWN へ戻す。
    for a, b in [((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0))]:
        for d in ALL_DIRECTIONS:
            nb = maze.neighbor(a[0], a[1], d)
            if nb == b:
                maze.set_wall(a[0], a[1], d, WallState.UNKNOWN)
                maze.set_wall(b[0], b[1], opposite(d), WallState.UNKNOWN)
                break
    return maze


def test_optimistic_and_pessimistic_flood_actually_differ():
    maze = _make_shortcut_vs_detour_maze()
    start, goal = (0, 0), (3, 0)

    dist_opt = compute_flood(maze, [goal], FloodMode.OPTIMISTIC)
    dist_pes = compute_flood(maze, [goal], FloodMode.PESSIMISTIC)

    # 楽観: 未知の近道を通れるとみなすので 3 区画で着く。
    assert dist_opt[start] == 3
    # 悲観: 未知の近道は通れないとみなすので、既知の迂回路 9 区画を使う。
    assert dist_pes[start] == 9
    # 二つの値が実際に異なることを明示的に確認する（空振り防止）。
    assert dist_opt[start] != dist_pes[start]

    path_opt = shortest_path(maze, start, [goal], FloodMode.OPTIMISTIC)
    path_pes = shortest_path(maze, start, [goal], FloodMode.PESSIMISTIC)
    assert path_opt == [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert path_pes[0] == start and path_pes[-1] == goal
    assert len(path_pes) == 10  # 9 手 = 10 区画
    assert path_opt != path_pes


def test_is_reachable_known_requires_known_route():
    """既知の迂回路を消し、未知の近道しか無い迷路では、悲観モードでは
    到達可能と判定してはならない（既知だけで到達できるかの判定）。"""
    maze = MazeMap(width=4, height=4)
    fill_all_interior_walls(maze, WallState.WALL)  # 迂回路すら作らない: 全部壁
    start, goal = (0, 0), (3, 0)
    # 近道 3 本だけ未知のまま（WALL で埋めた後 UNKNOWN に戻す）。
    for a, b in [((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0))]:
        for d in ALL_DIRECTIONS:
            nb = maze.neighbor(a[0], a[1], d)
            if nb == b:
                maze.set_wall(a[0], a[1], d, WallState.UNKNOWN)
                maze.set_wall(b[0], b[1], opposite(d), WallState.UNKNOWN)
                break

    # 楽観なら未知を通って到達できる。
    dist_opt = compute_flood(maze, [goal], FloodMode.OPTIMISTIC)
    assert dist_opt[start] == 3

    # 悲観（= is_reachable_known）は、既知の壁だけでは到達できないと判定する。
    assert is_reachable_known(maze, start, [goal]) is False
    with pytest.raises(NoRouteError):
        shortest_path(maze, start, [goal], FloodMode.PESSIMISTIC)


# ==========================================================================
# 3. ターン列への変換（直進の連結とターン種別）
# ==========================================================================
def test_path_to_commands_straight_then_right_turn():
    path = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3)]
    commands = path_to_commands(path, start_heading=Direction.N)
    assert commands == [
        Command(CommandType.STRAIGHT, 3),
        Command(CommandType.TURN_RIGHT90),
        Command(CommandType.STRAIGHT, 3),
        Command(CommandType.GOAL_STOP),
    ]


def test_path_to_commands_left_turn_at_start():
    # 開始時の向き N に対し、最初の移動が西 (W) なので左ターンが先頭に立つ。
    path = [(1, 0), (0, 0)]
    commands = path_to_commands(path, start_heading=Direction.N)
    assert commands == [
        Command(CommandType.TURN_LEFT90),
        Command(CommandType.STRAIGHT, 1),
        Command(CommandType.GOAL_STOP),
    ]


def test_path_to_commands_180_turnaround():
    # 北へ1区画進んでから同じ区画へ戻る = 180度折返し。
    path = [(1, 1), (1, 2), (1, 1)]
    commands = path_to_commands(path, start_heading=Direction.N)
    assert commands == [
        Command(CommandType.STRAIGHT, 1),
        Command(CommandType.TURN_180),
        Command(CommandType.STRAIGHT, 1),
        Command(CommandType.GOAL_STOP),
    ]


def test_path_to_commands_start_is_goal():
    commands = path_to_commands([(2, 2)], start_heading=Direction.N)
    assert commands == [Command(CommandType.GOAL_STOP)]


def test_path_to_commands_rejects_nonadjacent_cells():
    with pytest.raises(ValueError):
        path_to_commands([(0, 0), (2, 2)], start_heading=Direction.N)


# ==========================================================================
# 4. 同歩数でターン数が少ない方が選ばれること
# ==========================================================================
def test_shortest_path_prefers_fewer_turns_on_tie():
    """(0,0) から (2,2) への最短距離は 4 区画（4手）で、実現方法は複数ある:
      - 直進2→ターン1回→直進2（L字。ターン数1） … 歩数4・ターン1
      - 東西南北を交互に切り替える経路（ジグザグ。ターン数3） … 歩数4・ターン3
    どちらも歩数マップ上の最短距離 4 を満たすが、ターン数が少ない L 字型が
    選ばれなければならない。
    """
    maze = MazeMap(width=4, height=4)
    fill_all_interior_walls(maze, WallState.OPEN)  # 内部を全部開けた迷路（全て等距離候補）

    start, goal = (0, 0), (2, 2)
    dist = compute_flood(maze, [goal], FloodMode.PESSIMISTIC)
    assert dist[start] == 4  # 手計算: |dx|+|dy| = 2+2

    path, commands = plan_route(maze, start, [goal], FloodMode.PESSIMISTIC, start_heading=Direction.N)
    assert len(path) == 5  # 4手 = 5区画
    assert path[0] == start and path[-1] == goal

    turn_types = {CommandType.TURN_RIGHT90, CommandType.TURN_LEFT90, CommandType.TURN_180}
    turns = [c for c in commands if c.type in turn_types]
    straights = [c for c in commands if c.type == CommandType.STRAIGHT]

    # ジグザグ（ターン3回）ではなく、L字型（ターン1回）が選ばれていること。
    assert len(turns) == 1, commands
    assert sum(c.cells for c in straights) == 4
    assert commands[-1] == Command(CommandType.GOAL_STOP)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
