"""
tests/test_classic_fast_run.py
================
`classic/explorer.py`（Phase.FAST・Phase.RETURN2）と `classic/motion.py`
（`cells_completed`・`reanchor_heading`）が実装する **S3: 最短走行**
（`research_notes/note_030_classical_rebuild_plan.md` §3 S3）の検査。

`tests/test_classic_policy.py`・`tests/test_classic_localization.py` の作法を
そのまま踏襲する: **実際にシミュレータを動かして**実測し、[実測] タグ付き
print と、発火側・空振り側を対にした assert で確かめる。

構成（note_030 §3 S3 任務指示の T1〜T6 に対応）:
  T1. 手で作った小さな既知の迷路で、FAST が `classic.route.plan_route` と
      同じコマンド列を実行することの検査
  T2. `classic.checks.assert_same_callable` で、走行中に実際に呼ばれる
      経路計画が `classic.route.plan_route` であることを表明する（型 B）
  T3. `classic.checks.plan_adherence` で、最短走行の区間のティックが
      "fast:*" に乗っていた割合を実測して報告する（🔴 閾値は置かない）
  T4. 否定対照 N1/N2（`classic.checks.negative_control`）
  T5. 直線延伸（`extend_straights`）が効いていることの対の検査
  T6. 多区画直進中の区画ごと補正が実際に呼ばれていることの実測

これに加え、実装の過程で判明した設計上の要点（plan_route を
`start_heading=Direction.N` 固定で呼ぶ以上、実際の帰還後の向きが北でない
場合に必要になる「先頭への補正旋回」）についても 1 件、単体で検査する
（`test_fast_run_reconciles_actual_heading_to_north_before_executing_the_plan`）。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from mouse.mjcf import build_maze_robot_xml
from mouse.params import RobotParams
from mouse.sim import MouseSim

import classic.explorer as explorer_module
import classic.route as route_module
from classic.checks import assert_same_callable, negative_control, plan_adherence
from classic.explorer import ClassicExplorer, Phase
from classic.flood import FloodMode
from classic.maze_map import ALL_DIRECTIONS, Direction, MazeMap, direction_between
from classic.maze_map import WallState as MapWallState
from classic.route import Command, CommandType, plan_route

Cell = Tuple[int, int]


@pytest.fixture(scope="module")
def params() -> RobotParams:
    return RobotParams()


# ==========================================================================
# 手で作った小さな既知の迷路（note_030 §3 S3 検査の作法）
# ==========================================================================
# (0,0) スタートから中央 2x2 ゴールへの唯一の通路だけを開けた回廊。
# 経路は直進2区画→右90°→直進1区画→左90°→直進1区画→右90°→直進1区画→
# ゴール停止となり、90°右・90°左の両方のターン種別が出る（180°折返しは
# 出ない — `classic.route.shortest_path` が復元する経路は同一区画を
# 再訪しない単純経路であり、格子上の単純経路の途中に 180° 折返しは
# 構造上現れない。詳細は下の
# `test_fast_run_reconciles_actual_heading_to_north_before_executing_the_plan`
# のコメントを参照。180° 折返しは「実行時の向き補正」としてのみ現れる）。
MAZE_W, MAZE_H = 6, 6
KNOWN_PATH: List[Cell] = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
GOAL_CELLS: List[Cell] = [(2, 2), (2, 3), (3, 2), (3, 3)]  # width//2-1,width//2 x 同様の縦

_LABEL_FOR_TYPE = {
    CommandType.STRAIGHT: "fast:straight",
    CommandType.TURN_RIGHT90: "fast:turn_right",
    CommandType.TURN_LEFT90: "fast:turn_left",
    CommandType.TURN_180: "fast:turn_180",
    CommandType.GOAL_STOP: "fast:goal_stop",
}


def _carved_path_maze_and_walls(width: int, height: int, path: List[Cell]):
    """`path` に沿った区画だけを開けた回廊。既知の地図（`MazeMap`。壁は
    path の外側すべてを WALL として書き込み済み）と、同じ経路から矛盾なく
    作った物理シミュレータ用の壁配列（v_walls/h_walls）を対で返す。"""
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


def _fresh_sim(tmp_path: Path, params: RobotParams, v_walls, h_walls, label: str,
                heading_deg: float = 90.0) -> MouseSim:
    xml_path = str(tmp_path / f"{label}.xml")
    build_maze_robot_xml(v_walls, h_walls, xml_path, model_name=label, params=params)
    sim = MouseSim(xml_path, params=params)
    sim.full_reset(cell=(0, 0), heading_deg=heading_deg)
    return sim


def _fresh_explorer_in_fast(sim: MouseSim, params: RobotParams, maze: MazeMap,
                             extend_straights: bool = True,
                             localization_enabled: bool = True) -> ClassicExplorer:
    """探索・帰還を経由せず、既に完成した地図を直接注入して Phase.FAST から
    始める（`_begin_fast_run` を直接呼ぶ。RETURN 完了時と同じ呼び方）。"""
    ex = ClassicExplorer(MAZE_W, MAZE_H, params=params, extend_straights=extend_straights,
                          localization_enabled=localization_enabled)
    ex.maze = maze
    obs = sim.observation()
    ex._begin_fast_run(obs)
    ex._need_replan = False
    return ex


def _drive_to_goal_stop_hold_complete(sim: MouseSim, ex: ClassicExplorer, max_ticks: int = 8000) -> List[str]:
    """ゴール停止の 0.2 秒静止ホールドが完了するまで駆動し、ティックごとの
    plan_id の列を返す。"""
    plan_ids: List[str] = []
    for _ in range(max_ticks):
        obs = sim.observation()
        vl, vr, plan_id = ex.tick(obs)
        plan_ids.append(plan_id)
        sim.step_control(vl, vr)
        if plan_id == "fast:goal_stop" and ex._goal_stop_ticks_left == 0:
            break
    else:
        raise AssertionError(f"{max_ticks} ティック以内にゴール停止(静止ホールド完了)へ到達しなかった")
    return plan_ids


def _distinct_sequence(seq: List[str]) -> List[str]:
    out: List[str] = []
    for s in seq:
        if not out or out[-1] != s:
            out.append(s)
    return out


# ==========================================================================
# T1: FAST が plan_route と同じコマンド列を実行すること
# ==========================================================================
def test_fast_executes_the_same_command_sequence_as_plan_route(tmp_path, params, carved_maze_and_walls):
    maze, v_walls, h_walls = carved_maze_and_walls
    expected_path, expected_commands = plan_route(
        maze, start=(0, 0), goals=GOAL_CELLS, mode=FloodMode.PESSIMISTIC, start_heading=Direction.N)
    expected_labels = [_LABEL_FOR_TYPE[c.type] for c in expected_commands]
    expected_straight_cells = [c.cells for c in expected_commands if c.type == CommandType.STRAIGHT]

    sim = _fresh_sim(tmp_path, params, v_walls, h_walls, "t1")
    ex = ClassicExplorer(MAZE_W, MAZE_H, params=params)
    ex.maze = maze

    # start_forward に渡された区画数を記録する（STRAIGHT の cells が
    # 1コマンドに正しくまとまっていることの実測。_begin_fast_run 呼び出し
    # より前に差し替える必要がある）。
    recorded_n_cells: List[int] = []
    original_start_forward = ex.motion.start_forward

    def _wrapped_start_forward(n):
        recorded_n_cells.append(n)
        return original_start_forward(n)

    ex.motion.start_forward = _wrapped_start_forward

    obs = sim.observation()
    ex._begin_fast_run(obs)
    ex._need_replan = False

    plan_ids = _drive_to_goal_stop_hold_complete(sim, ex)
    got_labels = _distinct_sequence(plan_ids)

    print(f"\n[実測] plan_route の出力: {expected_commands}")
    print(f"[実測] FAST が実行した plan_id 列(重複除去): {got_labels}")
    print(f"[実測] start_forward に渡された区画数の列: {recorded_n_cells}")
    print(f"[実測] 総ティック数: {len(plan_ids)} 到達区画: {ex.cell}")

    assert got_labels == expected_labels, (
        "FAST が実行したコマンド種別の列が plan_route の出力と一致しない"
    )
    assert recorded_n_cells == expected_straight_cells, (
        "FAST が start_forward に渡した区画数の列が plan_route の STRAIGHT.cells と一致しない"
    )
    assert ex.cell == expected_path[-1], "最終到達区画が plan_route の経路の終点と一致しない"


def test_fast_run_reconciles_actual_heading_to_north_before_executing_the_plan(params, carved_maze_and_walls):
    """`_begin_fast_run` は `plan_route` を `start_heading=Direction.N` 固定で
    呼ぶ（教授裁定・note_030 §3 任務指示）。実際の帰還後の向き
    （`self.heading`）は、スタート区画の唯一の出入口の向きで決まり、北とは
    限らない（例: 北の隣接区画から入れば南向きで区画中心に着く）。

    `classic.route.shortest_path` が復元する経路は同一区画を再訪しない
    単純経路であり、ある区画へ方向 d で入ってすぐ方向 -d で出る（180°
    折返し）は直前の区画へ逆戻りすることになるため、単純経路の途中には
    構造上現れない（スタート区画 (0,0) 自体も境界の壁で南・西が塞がれて
    いるため、1手目が 180° になることも無い）。したがって
    `plan_route(..., start_heading=Direction.N)` の生の出力に 180° 折返しが
    含まれることは通常無い。180° 折返しが実際に必要になるのは、この
    「実際の向き→北」への実行時の補正旋回としてのみである。ここではその
    補正が正しく差し込まれる（かつ plan_route 自身の出力(commands そのもの)
    は変えない）ことを、東西南のいずれから始めても実測する。"""
    maze, _v_walls, _h_walls = carved_maze_and_walls
    _expected_path, expected_commands = plan_route(
        maze, start=(0, 0), goals=GOAL_CELLS, mode=FloodMode.PESSIMISTIC, start_heading=Direction.N)
    assert all(c.type != CommandType.TURN_180 for c in expected_commands), (
        "前提が崩れている: plan_route の生の出力に 180° 折返しが含まれている"
        "（上のコメントの前提が成り立たない迷路になっている）"
    )

    expected_first_label_by_rel = {1: "fast:turn_right", 2: "fast:turn_180", 3: "fast:turn_left"}
    for actual_heading in (Direction.E, Direction.S, Direction.W):
        ex = ClassicExplorer(MAZE_W, MAZE_H, params=params)
        ex.maze = maze
        ex.heading = actual_heading  # 帰還後の実際の向きを模す（北ではない）
        dummy_obs = np.zeros(len(params.sensors) + 8, dtype=np.float64)
        ex._begin_fast_run(dummy_obs)

        rel = (int(Direction.N) - int(actual_heading)) % 4
        expected_first_label = expected_first_label_by_rel[rel]
        print(f"\n[実測] 実際の向き={actual_heading.name} → 最初に発行されたコマンド={ex._active_plan_id}")

        assert ex._active_plan_id == expected_first_label, (
            f"向き{actual_heading.name}から FAST を始めても期待した補正旋回が発行されない"
        )
        assert ex._fast_cmd_index == 0, (
            "補正旋回のぶんだけ plan_route のコマンド列そのものを消費してしまっている"
            "（補正は実行系列だけへの追加であるべき）"
        )
        assert ex._fast_commands == expected_commands, (
            "補正の有無に関わらず、保持している plan_route の出力自体は変わらないはず"
        )


# ==========================================================================
# T2: 走行中に実際に呼ばれる経路計画が classic.route.plan_route であること
#     （型 B 再発防止）
# ==========================================================================
def test_fast_run_uses_classic_route_plan_route():
    """`classic.checks.assert_same_callable` で、最短経路の計画を担う
    `ClassicExplorer._begin_fast_run`・`_issue_next_fast_command` が
    `ClassicExplorer` 自身の実装（mixin 等の写しではない）であることを
    表明する（既存 `test_issue_forward_is_classicexplorers_own_implementation`
    等と同じ作法）。関数はクラスメソッドの MRO を持たないため、それだけでは
    「呼んでいる先が本物の plan_route か」までは確認できない。念のため
    `classic.explorer` が import している `plan_route` が `classic.route.plan_route`
    そのもの（同一オブジェクト）であることも直接確認する。"""
    assert_same_callable(ClassicExplorer, "_begin_fast_run", ClassicExplorer)
    assert_same_callable(ClassicExplorer, "_issue_next_fast_command", ClassicExplorer)

    is_same = explorer_module.plan_route is route_module.plan_route
    print(f"\n[実測] classic.explorer.plan_route is classic.route.plan_route: {is_same}")
    assert is_same, (
        "classic/explorer.py が import している plan_route が classic/route.py の"
        "実装と別物になっている（型 B: 処置の写しが2本に分かれている可能性）"
    )


# ==========================================================================
# T3: 最短走行区間のティックが "fast:*" に乗っていた割合を実測して報告する
#     （🔴 閾値は置かない。判定条文は教授の専管事項）
# ==========================================================================
def test_fast_run_plan_adherence_report(tmp_path, params, carved_maze_and_walls):
    maze, v_walls, h_walls = carved_maze_and_walls
    sim = _fresh_sim(tmp_path, params, v_walls, h_walls, "t3")
    ex = _fresh_explorer_in_fast(sim, params, maze)
    plan_ids = _drive_to_goal_stop_hold_complete(sim, ex)

    got = plan_adherence(plan_ids, intended={
        "fast:straight", "fast:turn_right", "fast:turn_left", "fast:turn_180", "fast:goal_stop",
    })
    print(f"\n[実測] plan_adherence（最短走行1回分。閾値は置かない・実測を報告するのみ）:\n{got.describe()}")


# ==========================================================================
# T4: 否定対照 N1/N2（真値を一切使っていないことの実測）
# ==========================================================================
N1N2_STEPS = 3000  # 数区画分の最短走行(直進+複数回の旋回)を確実に含む長さ


def _drive_fast_collect_voltages(tmp_path, params, v_walls, h_walls, maze, label, corrupt, n_steps):
    sim = _fresh_sim(tmp_path, params, v_walls, h_walls, label)
    if corrupt:
        sim.privileged_pose = lambda: (9999.0, -9999.0, 12.345)
        sim.privileged_velocity = lambda: (9999.0, -9999.0)
    ex = _fresh_explorer_in_fast(sim, params, maze)
    voltages = []
    for _ in range(n_steps):
        obs = sim.observation()
        vl, vr, _plan_id = ex.tick(obs)
        voltages.append((vl, vr))
        sim.step_control(vl, vr)
    return tuple(voltages)


def _drive_dummy_privileged_controller(sim, n_steps):
    """N2（対照）: 真値 (privileged_pose) を実際に読んで動く、真値へ露骨に
    依存する単純なダミー制御（tests/test_classic_motion.py と同じ作法）。"""
    voltages = []
    for _ in range(n_steps):
        x, _y, _yaw = sim.privileged_pose()
        vl = float(np.clip(x, -1.0, 1.0))
        vr = 0.0
        voltages.append((vl, vr))
        sim.step_control(vl, vr)
    return tuple(voltages)


def test_fast_run_does_not_use_privileged_information(tmp_path, params, carved_maze_and_walls):
    """N1: FAST 中の `ClassicExplorer` は、`sim.privileged_pose()/
        privileged_velocity()` の戻り値を意図的に壊しても電圧列が bit 一致する。
    N2: 真値を使うと分かっている簡単な対照（ダミー制御）に同じ壊し方を
        当てると、必ず電圧列が変わる（N1 が「壊し方が効いていないだけ」で
        通っていないことの確認）。"""
    maze, v_walls, h_walls = carved_maze_and_walls

    def run_under_test(broken: bool):
        return _drive_fast_collect_voltages(tmp_path, params, v_walls, h_walls, maze,
                                             f"t4_ut_{broken}", corrupt=broken, n_steps=N1N2_STEPS)

    def run_control(broken: bool):
        sim = _fresh_sim(tmp_path, params, v_walls, h_walls, f"t4_ctrl_{broken}")
        if broken:
            sim.privileged_pose = lambda: (9999.0, -9999.0, 12.345)
        return _drive_dummy_privileged_controller(sim, N1N2_STEPS)

    got = negative_control(run_under_test=run_under_test, run_control=run_control)
    print(f"\n[実測] 否定対照(FAST・privileged_pose/velocity): {got.verdict}")
    assert got.passed, got.verdict


# ==========================================================================
# T5: 直線延伸（extend_straights）が効いていることの対の検査
# ==========================================================================
def test_extending_straights_completes_faster_than_the_stop_and_go_control(tmp_path, params, carved_maze_and_walls):
    """発火側 = extend_straights=True の方が、空振り側 = False（STRAIGHT n を
    1 区画の直進 n 回として実行する対照）より所要ティックが少ないこと。
    経路・速度上限は完全に同一で、直線を伸ばす効果だけが異なる。"""
    maze, v_walls, h_walls = carved_maze_and_walls

    def _run(extend_straights: bool, label: str) -> int:
        sim = _fresh_sim(tmp_path, params, v_walls, h_walls, label)
        ex = _fresh_explorer_in_fast(sim, params, maze, extend_straights=extend_straights)
        plan_ids = _drive_to_goal_stop_hold_complete(sim, ex)
        return len(plan_ids)

    ticks_extended = _run(True, "t5_extend")
    ticks_control = _run(False, "t5_control")

    reduction_pct = (1 - ticks_extended / ticks_control) * 100
    print(f"\n[実測] 所要ティック: extend_straights=True {ticks_extended} / "
          f"False(対照・区画ごと停止) {ticks_control} （削減 {reduction_pct:.1f}%）")

    assert ticks_extended < ticks_control, (
        "直線を伸ばした方(extend_straights=True)が、伸ばさない対照(False)より"
        "所要ティックで速くならなかった"
    )


# ==========================================================================
# T6: 多区画直進中の区画ごと補正が実際に呼ばれていることの実測
# ==========================================================================
def test_multi_cell_forward_reanchors_localization_at_every_cell_boundary(tmp_path, params):
    """n 区画の直進 1 回につき `Localizer.lateral_bias_for_forward` が
    （コマンド発行時 1 回 ＋ 途中の区画中心 n-1 回）＝ n 回呼ばれることを、
    実際の呼び出し回数を数えて実測する（`Localizer.events` は前後位置補正
    (axis="forward") も混ざるうえ、横ずれ推定値がちょうど 0.0 のときは
    記録されない — bias!=0.0 のときだけ append する設計 — ので、事象記録
    ではなく呼び出し自体をラップして数える）。

    空振り側 = localization_enabled=False では、呼び出し自体は同じ回数
    起きる（呼び出し経路そのものは変わらない設計。classic/localization.py
    docstring）が、`Localizer.enabled=False` のため内部で即座に 0.0 を返し、
    `CellMotionController.reanchor_heading` は一度も呼ばれない
    （＝横位置補正が一切効かない）ことを対で確かめる。"""
    n_cells = 6
    width, height = 1, n_cells + 2
    v_walls = np.zeros((width + 1, height), dtype=int)
    v_walls[0, :] = 1
    v_walls[width, :] = 1
    h_walls = np.zeros((width, height + 1), dtype=int)
    h_walls[:, 0] = 1
    h_walls[:, height] = 1

    def _run(localization_enabled: bool, label: str):
        xml_path = str(tmp_path / f"{label}.xml")
        build_maze_robot_xml(v_walls, h_walls, xml_path, model_name=label, params=params)
        sim = MouseSim(xml_path, params=params)
        sim.full_reset(cell=(0, 1), heading_deg=90.0)

        ex = ClassicExplorer(width, height, params=params, localization_enabled=localization_enabled)
        ex.cell = (0, 1)
        ex.heading = Direction.N
        ex.phase = Phase.FAST
        ex._fast_commands = [Command(CommandType.STRAIGHT, n_cells), Command(CommandType.GOAL_STOP)]
        ex._fast_cmd_index = 0
        ex._need_replan = False

        bias_calls = [0]
        original_bias = ex.localizer.lateral_bias_for_forward

        def _counting_bias(*args, **kwargs):
            bias_calls[0] += 1
            return original_bias(*args, **kwargs)

        ex.localizer.lateral_bias_for_forward = _counting_bias

        reanchor_calls = [0]
        original_reanchor = ex.motion.reanchor_heading

        def _counting_reanchor(*args, **kwargs):
            reanchor_calls[0] += 1
            return original_reanchor(*args, **kwargs)

        ex.motion.reanchor_heading = _counting_reanchor

        obs = sim.observation()
        ex._issue_next_fast_command(obs)

        for _ in range(8000):
            obs = sim.observation()
            vl, vr, plan_id = ex.tick(obs)
            sim.step_control(vl, vr)
            if plan_id == "fast:goal_stop" and ex._goal_stop_ticks_left == 0:
                break
        else:
            raise AssertionError("収束しなかった")

        lateral_events = len([e for e in ex.localizer.events if e.axis == "lateral"])
        return bias_calls[0], reanchor_calls[0], lateral_events, ex.cell

    bias_on, reanchor_on, events_on, cell_on = _run(True, "t6_on")
    bias_off, reanchor_off, events_off, cell_off = _run(False, "t6_off")

    print(f"\n[実測] {n_cells}区画の直進1回:")
    print(f"  localization_enabled=True : lateral_bias_for_forward呼び出し={bias_on}回 "
          f"reanchor_heading呼び出し={reanchor_on}回 横位置補正イベント={events_on}件")
    print(f"  localization_enabled=False: lateral_bias_for_forward呼び出し={bias_off}回 "
          f"reanchor_heading呼び出し={reanchor_off}回 横位置補正イベント={events_off}件")

    assert cell_on == (0, 1 + n_cells) and cell_off == (0, 1 + n_cells), "到達区画がn区画ぶん進んでいない"

    assert bias_on == n_cells, (
        f"{n_cells}区画の直進1回で lateral_bias_for_forward が n 回"
        f"(発行時1回+途中n-1回)呼ばれるはずが {bias_on}回だった"
    )
    assert reanchor_on == n_cells - 1, (
        f"reanchor_heading が n-1 回(途中の区画中心)呼ばれるはずが {reanchor_on}回だった"
    )
    # 空振り側: 呼び出し自体の回数は変わらない(呼び出し経路そのものは
    # localization_enabled に依らず同一という設計)が、効果は一切出ない。
    assert bias_off == n_cells, "localization_enabled=False で呼び出し経路そのものが変わってしまっている"
    assert reanchor_off == 0, "localization_enabled=False でも reanchor_heading が呼ばれてしまった"
    assert events_off == 0, "localization_enabled=False でも横位置補正イベントが記録されてしまった"
