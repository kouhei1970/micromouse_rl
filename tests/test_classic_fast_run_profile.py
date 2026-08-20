"""
tests/test_classic_fast_run_profile.py
================
`classic/explorer.py` の `fast_mode`（`research_notes/note_031_profile_planner_
and_eta.md`・任務指示「S3': 最短走行のプロファイル追従化」）の検査。

`tests/test_classic_fast_run.py` の作法（手で作った小さな回廊の迷路、
`_begin_fast_run` を直接呼んで Phase.FAST から始める、`MouseSim` を実際に回す）
を踏襲する。ここでは `classic/tracker.py`/`classic/fast_planner.py` の単体検査
（`tests/test_tracker.py`/`tests/test_fast_planner.py`）と重複しない、
**`classic/explorer.py` への配線そのもの**（`fast_mode`・`plan_id`・
RETURN2への遷移・安全弁）だけを検査する。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from mouse.mjcf import build_maze_robot_xml
from mouse.params import RobotParams
from mouse.sim import MouseSim

from classic.explorer import ClassicExplorer, Phase
from classic.maze_map import ALL_DIRECTIONS, Direction, MazeMap, direction_between
from classic.maze_map import WallState as MapWallState

Cell = Tuple[int, int]

MAZE_W, MAZE_H = 6, 6
KNOWN_PATH: List[Cell] = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
GOAL_CELLS: List[Cell] = [(2, 2), (2, 3), (3, 2), (3, 3)]


@pytest.fixture(scope="module")
def params() -> RobotParams:
    return RobotParams()


def _carved_path_maze_and_walls(width: int, height: int, path: List[Cell]):
    maze = MazeMap(width, height)
    for x in range(width):
        for y in range(height):
            for d in ALL_DIRECTIONS:
                if maze.neighbor(x, y, d) is not None:
                    maze.set_wall(x, y, d, MapWallState.WALL)

    v_walls = np.ones((width + 1, height), dtype=int)
    h_walls = np.ones((width, height + 1), dtype=int)
    for a, b in zip(path[:-1], path[1:]):
        d = direction_between(a, b)
        maze.set_wall(a[0], a[1], d, MapWallState.OPEN)
        if d is Direction.E:
            v_walls[a[0] + 1, a[1]] = 0
        elif d is Direction.W:
            v_walls[a[0], a[1]] = 0
        elif d is Direction.N:
            h_walls[a[0], a[1] + 1] = 0
        else:  # Direction.S
            h_walls[a[0], a[1]] = 0
    return maze, v_walls, h_walls


@pytest.fixture(scope="module")
def carved_maze_and_walls():
    return _carved_path_maze_and_walls(MAZE_W, MAZE_H, KNOWN_PATH)


def _fresh_sim(tmp_path: Path, params: RobotParams, v_walls, h_walls, label: str) -> MouseSim:
    xml_path = str(tmp_path / f"{label}.xml")
    build_maze_robot_xml(v_walls, h_walls, xml_path, model_name=label, params=params)
    sim = MouseSim(xml_path, params=params)
    sim.full_reset(cell=(0, 0), heading_deg=90.0)
    return sim


def _fresh_explorer_in_fast(sim: MouseSim, params: RobotParams, maze: MazeMap,
                             fast_mode: str = "command") -> ClassicExplorer:
    ex = ClassicExplorer(MAZE_W, MAZE_H, params=params, fast_mode=fast_mode)
    ex.maze = maze
    obs = sim.observation()
    ex._begin_fast_run(obs)
    ex._need_replan = False
    return ex


def _drive(sim: MouseSim, ex: ClassicExplorer, max_ticks: int,
           stop_when) -> List[str]:
    """`stop_when(ex, plan_id)` が True を返すまで駆動し、plan_id の列を返す。"""
    plan_ids: List[str] = []
    for _ in range(max_ticks):
        obs = sim.observation()
        vl, vr, plan_id = ex.tick(obs)
        plan_ids.append(plan_id)
        sim.step_control(vl, vr)
        if stop_when(ex, plan_id):
            return plan_ids
    raise AssertionError(f"{max_ticks} ティック以内に終了条件へ到達しなかった: 直近plan_id={plan_ids[-20:]}")


def _distinct(seq: List[str]) -> List[str]:
    out: List[str] = []
    for s in seq:
        if not out or out[-1] != s:
            out.append(s)
    return out


def _goal_stop_hold_done(ex: ClassicExplorer, plan_id: str) -> bool:
    return plan_id == "fast:goal_stop" and ex._goal_stop_ticks_left == 0


# ==========================================================================
# 1. fast_mode="command"（既定）は "profile" を一切含まない
#    （空振り側。fast_mode を追加してもコード経路が変わっていないことの傍証）
# ==========================================================================
def test_command_mode_never_emits_profile_plan_ids(tmp_path, params, carved_maze_and_walls):
    maze, v_walls, h_walls = carved_maze_and_walls
    sim = _fresh_sim(tmp_path, params, v_walls, h_walls, "cmd")
    ex = _fresh_explorer_in_fast(sim, params, maze, fast_mode="command")
    assert ex._profile_active is False

    plan_ids = _drive(sim, ex, max_ticks=8000, stop_when=_goal_stop_hold_done)
    distinct = _distinct(plan_ids)
    print(f"\n[実測] fast_mode=command の plan_id列(重複除去): {distinct}")

    assert all("profile" not in p for p in plan_ids), (
        "fast_mode=command なのに plan_id に 'profile' が混入した"
    )
    assert ex.cell in GOAL_CELLS


# ==========================================================================
# 2. fast_mode="profile" は経路追従の区間で "fast:profile" を使い、
#    コマンド方式より少ないティック数でゴールへ到達する（作動側）
# ==========================================================================
def test_profile_mode_reaches_goal_using_profile_plan_ids_and_is_faster(
        tmp_path, params, carved_maze_and_walls):
    maze, v_walls, h_walls = carved_maze_and_walls

    sim_cmd = _fresh_sim(tmp_path, params, v_walls, h_walls, "cmp_cmd")
    ex_cmd = _fresh_explorer_in_fast(sim_cmd, params, maze, fast_mode="command")
    ticks_cmd = len(_drive(sim_cmd, ex_cmd, max_ticks=8000, stop_when=_goal_stop_hold_done))

    sim_prof = _fresh_sim(tmp_path, params, v_walls, h_walls, "cmp_prof")
    ex_prof = _fresh_explorer_in_fast(sim_prof, params, maze, fast_mode="profile")
    assert ex_prof._profile_active is True
    plan_ids = _drive(sim_prof, ex_prof, max_ticks=8000, stop_when=_goal_stop_hold_done)
    distinct = _distinct(plan_ids)
    ticks_prof = len(plan_ids)

    print(f"\n[実測] plan_id列(重複除去): {distinct}")
    print(f"[実測] ティック数: command={ticks_cmd} profile={ticks_prof} "
          f"比(profile/command)={ticks_prof / ticks_cmd:.4f}")

    assert any(p == "fast:profile" for p in plan_ids), (
        "fast_mode=profile なのに 'fast:profile' が1件も無い(作動側の失敗)"
    )
    assert distinct[-1] == "fast:goal_stop"
    assert ex_prof.cell in GOAL_CELLS
    assert ticks_prof < ticks_cmd, (
        "profile追従がコマンド方式より遅い(整定待ちが消える設計上、"
        "速くなるはず — note_031 段4の実測と矛盾する)"
    )


# ==========================================================================
# 3. RETURN2 完了後、そのまま次の FAST(profile) へループする
# ==========================================================================
def test_profile_mode_loops_from_return2_back_to_fast(tmp_path, params, carved_maze_and_walls):
    maze, v_walls, h_walls = carved_maze_and_walls
    sim = _fresh_sim(tmp_path, params, v_walls, h_walls, "loop")
    ex = _fresh_explorer_in_fast(sim, params, maze, fast_mode="profile")

    # 最初のゴール停止ホールド完了まで
    plan_ids = _drive(sim, ex, max_ticks=8000, stop_when=_goal_stop_hold_done)
    assert ex.phase is Phase.FAST
    n_fast_before = sum(1 for p in plan_ids if p.startswith("fast:profile"))
    assert n_fast_before > 0

    # RETURN2 で 'return2:profile' が現れ、続いて次の FAST の
    # 'fast:profile' が再度現れるまで駆動する。
    def _second_fast_started(ex_: ClassicExplorer, plan_id: str) -> bool:
        return plan_id == "fast:profile"

    plan_ids2 = _drive(sim, ex, max_ticks=8000, stop_when=_second_fast_started)
    distinct2 = _distinct(plan_ids2)
    print(f"\n[実測] RETURN2以降のplan_id列(重複除去): {distinct2}")

    assert any(p.startswith("return2:profile") for p in plan_ids2), (
        "RETURN2 が profile 追従で実行された形跡が無い"
    )
    assert plan_ids2[-1] == "fast:profile"
    assert ex.phase is Phase.FAST
    # RETURN2 完了(_finish_profile_plan)でスタート区画へ合わせた直後に
    # 次のFASTを始めているので、この時点の自前カウンタはスタート区画のはず。
    assert ex.cell == ex.start_cell


# ==========================================================================
# 4. 安全弁: plan_fast_run() が計画できない(未知区画=壁の悲観判定で到達不能)
#    場合、fast_command_fallback が立ち、plan_id接頭辞に反映される
# ==========================================================================
def test_profile_mode_falls_back_and_marks_plan_id_when_unreachable(tmp_path, params):
    """全マス未知（外周のみ既知）の地図では、`plan_fast_run` も現行の
    `plan_route`（悲観歩数マップ）も同じ理由で到達不能と判定する
    （どちらも「未知=壁」という同じ悲観判定を使うため、判定そのものは
    一致する）。ここで見たいのは値の一致ではなく、
    (a) fast_mode=profile の安全弁(`_fast_command_fallback`)が実際に立つこと
    (b) それが plan_id の接頭辞(`fast_fallback:blocked`)に反映されること
    (c) 例外を投げずその場で安全に停止すること
    の3点である。"""
    maze = MazeMap(MAZE_W, MAZE_H)  # 全マス未知
    v_walls = np.ones((MAZE_W + 1, MAZE_H), dtype=int)
    h_walls = np.ones((MAZE_W, MAZE_H + 1), dtype=int)
    sim = _fresh_sim(tmp_path, params, v_walls, h_walls, "fallback")

    ex = ClassicExplorer(MAZE_W, MAZE_H, params=params, fast_mode="profile")
    ex.maze = maze
    obs = sim.observation()
    ex._begin_fast_run(obs)

    print(f"\n[実測] fallback={ex._fast_command_fallback} plan_id={ex._active_plan_id}")
    assert ex._fast_command_fallback is True, "到達不能なのに安全弁(フォールバック)が立っていない"
    assert ex._active_plan_id == "fast_fallback:blocked", (
        f"plan_id接頭辞にフォールバックが反映されていない(実際={ex._active_plan_id})"
    )
    assert ex._profile_active is False
