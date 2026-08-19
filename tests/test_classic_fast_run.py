"""
tests/test_classic_fast_run.py
================
`classic/explorer.py`（Phase.FAST・Phase.RETURN2）と `classic/motion.py`
（`cells_completed`・`reanchor_heading`）が実装する **S3: 最短走行**
（`research_notes/note_030_classical_rebuild_plan.md` §3 S3）の検査。

`tests/test_classic_policy.py`・`tests/test_classic_localization.py` の作法を
そのまま踏襲する: **実際にシミュレータを動かして**実測し、[実測] タグ付き
print と、作動側・空振り側を対にした assert で確かめる（用語は
docs/JA_ENGINEERING_TERMS.md §1 に従う — 是正5）。

構成（note_030 §3 S3 任務指示の T1〜T6 に対応）:
  T1. 手で作った小さな既知の迷路で、FAST が `classic.route.plan_route` と
      同じコマンド列を実行することの検査
  T2. `classic.checks.assert_same_callable` で、走行中に実際に呼ばれる
      経路計画が `classic.route.plan_route` であることを表明する（型 B）
  T3. `classic.checks.plan_adherence` で、最短走行の区間のティックが
      "fast:*" に乗っていた割合を実測して報告する（🔴 閾値は置かない）
  T4. 否定対照 N1〜N4（`classic.checks.negative_control`）
  T5. 直線延伸（`extend_straights`）が効いていることの対の検査
  T6. 多区画直進中の区画ごと補正が実際に呼ばれていることの実測

これに加え、実装の過程で判明した設計上の要点（plan_route を
`start_heading=Direction.N` 固定で呼ぶ以上、実際の帰還後の向きが北でない
場合に必要になる「先頭への補正旋回」）についても 1 件、単体で検査する
（`test_fast_run_reconciles_actual_heading_to_north_before_executing_the_plan`）。

2026-08-19 の検収で見つかった不足への是正（`experiments/exp_024_s3_fast_run/
PREREG.md` 追記1）に対応して、以下を追加する:
  是正2. 「最短走行中は地図を書き換えない」ことを、地図の**内容**ではなく
      `_update_map_from_sensing` の**呼び出しの配線**で検査する
      （作動側・空振り側を対に。内容の一致を根拠にすると、検査用の迷路が
      単純すぎて書いても内容が変わらない、という穴を再現してしまう）
  是正3. `ClassicExplorer._enter_route_blocked`（旧 `_enter_fast_blocked`。
      経路が引けない・実行できないときの安全弁）を実際に作動させる検査
      （作動側・空振り側を対に）
  是正4. 実際の迷路（`competition/mazes/design_turn_v1/maze_41001.npz`、
      対象6迷路中で最短歩数が最小の52）を `CompetitionEvaluator` で走らせ、
      最短走行が競技評価器の2段階ゴール判定を通って成立することを検査する
      （🔴 合否のしきい値は置かない。判定条文は PREREG.md の領分）
  是正6. `Phase.RETURN2`（最短走行の帰路）が `Phase.FAST` と同じ
      「plan_route のコマンド列を実行する」機構で走ること、その間も
      地図を書き換えないことを検査する（作動側・空振り側を対に）
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

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
from classic.policy import ClassicExplorerPolicy
from classic.route import Command, CommandType, plan_route
from competition.evaluator import CompetitionEvaluator

REPO_ROOT = Path(__file__).resolve().parent.parent
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
# 否定対照 N3/N4（PREREG.md §10。経路計画が実際に使われていることの実測）
# ==========================================================================
N3N4_STEPS = N1N2_STEPS  # 同じ長さ(数区画分の最短走行を確実に含む)を流用


def _collapsing_plan_route(*args, **kwargs):
    """N3 の壊し方: 本物の `classic.route.plan_route` を呼んだ後、返って
    きた `Command` 列のうち `CommandType.STRAIGHT` の `cells` をすべて 1 に
    潰す。`Command` は frozen dataclass なので、書き換えず新しい `Command`
    を作って積み直す。"""
    path, commands = route_module.plan_route(*args, **kwargs)
    collapsed = [
        Command(CommandType.STRAIGHT, 1) if c.type == CommandType.STRAIGHT else c
        for c in commands
    ]
    return path, collapsed


def _drive_fast_collect_voltages_with_route_patch(monkeypatch, tmp_path, params, v_walls, h_walls,
                                                    maze, label, collapse_straights, n_steps):
    """`classic.explorer.plan_route`（`classic.route.plan_route` ではない —
    下のコメント参照）を差し替えて FAST を駆動し、電圧列を返す。

    🔴 `classic/explorer.py` は `from classic.route import ... plan_route` で
    `plan_route` を自分のモジュール名前空間へ**束縛済み**である。実装前に
    実測で確かめた事実（`classic.route.plan_route` だけを差し替える誤りを
    避けるため必ず先に確認すること）:
      - `classic.route.plan_route` だけを差し替えても
        `classic.explorer.plan_route` は元のまま変わらない（束縛はモジュール
        読み込み時の1回きりで、以後は互いに独立な別オブジェクトになる）。
      - `classic.explorer.plan_route` を直接差し替えると、
        `_plan_route_or_block` 内のベア名前 `plan_route` の参照解決は
        呼び出しのたびに `classic.explorer` モジュールの名前空間を引くため、
        実際に差し替えた実装が呼ばれる（`ClassicExplorer._begin_fast_run` を
        直接呼んで実測・確認済み）。
    したがって「壊す」には `classic.route.plan_route` ではなく
    `classic.explorer.plan_route` を差し替える必要がある。"""
    patched = _collapsing_plan_route if collapse_straights else route_module.plan_route
    monkeypatch.setattr(explorer_module, "plan_route", patched)
    is_bound_to_route_module = explorer_module.plan_route is route_module.plan_route
    print(f"[実測] classic.explorer.plan_route 差し替え後"
          f"(collapse_straights={collapse_straights}): "
          f"classic.route.plan_route と同一か={is_bound_to_route_module}")
    if collapse_straights:
        assert not is_bound_to_route_module, (
            "collapse_straights=True で差し替えたのに classic.explorer.plan_route が"
            "本物のままになっている（壊し方が効いていない）"
        )

    sim = _fresh_sim(tmp_path, params, v_walls, h_walls, label)
    ex = _fresh_explorer_in_fast(sim, params, maze)
    voltages = []
    for _ in range(n_steps):
        obs = sim.observation()
        vl, vr, _plan_id = ex.tick(obs)
        voltages.append((vl, vr))
        sim.step_control(vl, vr)
    return tuple(voltages)


def test_fast_run_uses_the_planned_route(tmp_path, params, carved_maze_and_walls, monkeypatch):
    """PREREG.md §10 N3/N4。

    N3(壊す側): `classic.explorer.plan_route` を、本物の `plan_route` の
    戻り値のうち `CommandType.STRAIGHT` の `cells` をすべて 1 に潰す
    ラッパー(`_collapsing_plan_route`)へ差し替える。期待は「最短走行が
    変わる」こと（経路計画が実際に使われていることの証拠）。

    N4(空振り側): N3 の壊し方を適用せず、同じ構成(本物の plan_route)を
    2 回走らせる。期待は「変わらない」こと（bit 一致）。これはシミュレータ
    が決定的であることそのものの実測でもある — 一致しなければそれ自体が
    報告すべき重大な発見である（握りつぶさない）。

    `classic.checks.negative_control` の役割を N1/N2 とは逆向きに割り当てる:
      run_under_test = N4 側(壊し方を適用しない。常に本物の plan_route を
        2 回走らせて bit 一致を見る)
      run_control    = N3 側(broken=True のときだけ直進を 1 区画に潰す)
    この配線で `changed_when_broken` が N4 の「変わらない」判定
    (False であってほしい)に、`changed_in_control` が N3 の「変わる」判定
    (True であってほしい)にそのまま対応し、
    `passed = changed_in_control and not changed_when_broken` が
    「N3 も N4 も両方成立」を表す。対照(N3)側が動かなければ `verdict` は
    「判定保留」に倒れ、`passed` は偽になる（PREREG.md §10 の指示どおり）。"""
    maze, v_walls, h_walls = carved_maze_and_walls

    def run_under_test(_unused: bool) -> Tuple[Tuple[float, float], ...]:
        # N4: 壊し方を適用しない(常に本物の plan_route)。同じ構成を2回
        # 走らせて bit 一致することを見る。
        return _drive_fast_collect_voltages_with_route_patch(
            monkeypatch, tmp_path, params, v_walls, h_walls, maze,
            f"n4_{_unused}", collapse_straights=False, n_steps=N3N4_STEPS)

    def run_control(broken: bool) -> Tuple[Tuple[float, float], ...]:
        # N3: broken=True のときだけ直進の区画数を1に潰す。
        return _drive_fast_collect_voltages_with_route_patch(
            monkeypatch, tmp_path, params, v_walls, h_walls, maze,
            f"n3_{broken}", collapse_straights=broken, n_steps=N3N4_STEPS)

    got = negative_control(run_under_test=run_under_test, run_control=run_control)
    print(f"\n[実測] 否定対照(FAST・N3/N4・経路計画の直進潰し): {got.verdict}")
    print(f"[実測]   N4側(壊し方を適用しない・変わらないはず) "
          f"changed_when_broken={got.changed_when_broken}")
    print(f"[実測]   N3側(直進を1区画に潰す・変わるはず)     "
          f"changed_in_control={got.changed_in_control}")
    assert got.passed, got.verdict


# ==========================================================================
# T5: 直線延伸（extend_straights）が効いていることの対の検査
# ==========================================================================
def test_extending_straights_completes_faster_than_the_stop_and_go_control(tmp_path, params, carved_maze_and_walls):
    """作動側 = extend_straights=True の方が、空振り側 = False（STRAIGHT n を
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


# ==========================================================================
# 是正2: 「最短走行中は地図を書き換えない」を呼び出しの配線で検査する
# （内容の一致では検査しない — 検収で判明した穴の再発防止。
#  PREREG.md 追記1 の是正2 参照）
# ==========================================================================
def _count_calls(obj, method_name: str):
    """`obj.method_name` をラップして呼び出し回数を数える。呼び出し回数の
    リスト（1要素）と、元のメソッドへ委譲する差し替え関数を返す。"""
    calls = [0]
    original = getattr(obj, method_name)

    def _wrapped(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    setattr(obj, method_name, _wrapped)
    return calls


def test_fast_run_never_calls_update_map_from_sensing(tmp_path, params, carved_maze_and_walls):
    """是正2 作動側: Phase.FAST の走行中、`_update_map_from_sensing` が
    一度も呼ばれないこと。内容の一致（地図が変わっていないこと）ではなく、
    呼び出しそのものの回数を数える（検収で判明した穴 — 検査用の迷路が
    単純すぎて、書いても内容が変わらず検査が空振りしていた — の再発防止）。"""
    maze, v_walls, h_walls = carved_maze_and_walls
    sim = _fresh_sim(tmp_path, params, v_walls, h_walls, "t_corr2_fast")
    ex = _fresh_explorer_in_fast(sim, params, maze)
    calls = _count_calls(ex, "_update_map_from_sensing")

    plan_ids = _drive_to_goal_stop_hold_complete(sim, ex)

    print(f"\n[実測] Phase.FAST 走行中(総ティック{len(plan_ids)})の "
          f"_update_map_from_sensing 呼び出し回数: {calls[0]}")
    assert calls[0] == 0, (
        "Phase.FAST の走行中に _update_map_from_sensing が呼ばれた"
        "（最短走行中に地図を書き換えている）"
    )


def test_explore_run_always_calls_update_map_from_sensing(tmp_path, params, carved_maze_and_walls):
    """是正2 空振り側: Phase.EXPLORE の走行中は `_update_map_from_sensing`
    が必ず呼ばれること（上の作動側検査に判別力があることの確認 —
    「呼ばれない」が常に真になってしまう壊れた配線ではないことを示す）。"""
    _maze, v_walls, h_walls = carved_maze_and_walls
    sim = _fresh_sim(tmp_path, params, v_walls, h_walls, "t_corr2_explore")
    ex = ClassicExplorer(MAZE_W, MAZE_H, params=params)  # 白紙の地図・Phase.EXPLORE のまま
    calls = _count_calls(ex, "_update_map_from_sensing")

    ticks = 0
    for _ in range(2000):
        obs = sim.observation()
        vl, vr, _plan_id = ex.tick(obs)
        sim.step_control(vl, vr)
        ticks += 1
        if ex.phase is not Phase.EXPLORE:
            break

    print(f"\n[実測] Phase.EXPLORE 走行中(駆動{ticks}ティック)の "
          f"_update_map_from_sensing 呼び出し回数: {calls[0]}")
    assert calls[0] > 0, (
        "Phase.EXPLORE の走行中に _update_map_from_sensing が一度も呼ばれなかった"
        "（作動側検査に判別力が無い）"
    )


# ==========================================================================
# 是正3: `_enter_route_blocked`（旧 `_enter_fast_blocked`。最短走行の
# 安全弁）を実際に作動させる検査（作動側・空振り側の対）
# ==========================================================================
def _sealed_unreachable_maze(width: int, height: int) -> MazeMap:
    """全ての内部壁を WALL にした、どの区画からもどこへも動けない地図を
    新しく作って返す（悲観どころか楽観でも到達不能。到達不能検査用。
    `tests/test_classic_policy.py` の `_seal_maze` と同じ作法。

    🔴 module スコープの `carved_maze_and_walls` フィクスチャを直接
    書き換えると、以後の他の検査が壊れた地図を受け取ってしまう。
    毎回新しい `MazeMap` を作ることでこれを避ける。"""
    maze = MazeMap(width, height)
    for x in range(width):
        for y in range(height):
            for d in ALL_DIRECTIONS:
                if maze.neighbor(x, y, d) is not None:
                    maze.set_wall(x, y, d, MapWallState.WALL)
    return maze


def test_fast_blocked_safety_valve_fires_when_the_goal_route_becomes_unroutable(
        tmp_path, params, carved_maze_and_walls):
    """是正3 作動側: 最短走行の経路が引けない状況を人為的に作る
    （`_begin_fast_run` の直前に地図をゴールへ到達不能な状態へ差し替える。
    `classic.route.plan_route` が `NoRouteError` を投げる状況）。

    `plan_id` が "fast:blocked" になり、**例外が投げられず**、電圧が 0 の
    まま停止し続けることを実測する。"""
    _maze, v_walls, h_walls = carved_maze_and_walls
    sim = _fresh_sim(tmp_path, params, v_walls, h_walls, "t_corr3_blocked")
    ex = ClassicExplorer(MAZE_W, MAZE_H, params=params)
    ex.maze = _sealed_unreachable_maze(MAZE_W, MAZE_H)  # ゴールへ到達不能な地図

    obs = sim.observation()
    ex._begin_fast_run(obs)  # ここで NoRouteError が起きるが、外へは漏れない
    ex._need_replan = False

    print(f"\n[実測] fast:blocked 作動直後: plan_id={ex._active_plan_id!r} "
          f"blocked_reason={ex._blocked_reason!r}")
    assert ex._active_plan_id == "fast:blocked"
    assert ex._blocked_reason is not None

    voltages = []
    for _ in range(50):
        obs = sim.observation()
        vl, vr, plan_id = ex.tick(obs)
        voltages.append((vl, vr, plan_id))
        sim.step_control(vl, vr)

    print(f"[実測] 作動後 50 ティックの plan_id・電圧（先頭5件）: {voltages[:5]}")
    assert all(pid == "fast:blocked" for _, _, pid in voltages), (
        "fast:blocked のまま停止し続けなかった（意図せず次のコマンドへ進んでしまった）"
    )
    assert all(vl == 0.0 and vr == 0.0 for vl, vr, _ in voltages), (
        "fast:blocked 中に電圧が 0 でなかった"
    )


def test_fast_blocked_safety_valve_does_not_fire_when_the_goal_route_is_reachable(
        tmp_path, params, carved_maze_and_walls):
    """是正3 空振り側: 同じ手順でも地図が到達可能なまま（`carved_maze_and_walls`
    の既知の地図）なら "fast:blocked" にならないこと（上の作動側検査に
    判別力があることの確認）。"""
    maze, v_walls, h_walls = carved_maze_and_walls
    sim = _fresh_sim(tmp_path, params, v_walls, h_walls, "t_corr3_ok")
    ex = ClassicExplorer(MAZE_W, MAZE_H, params=params)
    ex.maze = maze  # 到達可能な既知の地図のまま

    obs = sim.observation()
    ex._begin_fast_run(obs)
    ex._need_replan = False

    print(f"\n[実測] 到達可能な地図での _begin_fast_run 直後: plan_id={ex._active_plan_id!r}")
    assert ex._active_plan_id != "fast:blocked"
    assert ex._blocked_reason is None


# ==========================================================================
# 是正4: 競技評価器 (CompetitionEvaluator) を通した最短走行の検査
# tests/test_classic_policy.py の module スコープ fixture の作法を踏襲する
# （5〜15分かかるため、検査ごとに走らせ直さず1回だけ走らせて使い回す）。
# ==========================================================================
EXP024_MAZE_DIR = REPO_ROOT / "competition" / "mazes" / "design_turn_v1"
# 対象6迷路（PREREG.md §3）のうち最短歩数 D0 が最小（52）の迷路。
EXP024_MAZE_ID = "41001"
EXP024_TIME_BUDGET_S = 420.0  # docs/RESEARCH_PLAN.md §2 の持ち時間そのもの
EXP024_MAX_RUNS = 5


@pytest.fixture(scope="module")
def exp024_evaluator() -> CompetitionEvaluator:
    return CompetitionEvaluator(
        maze_dir=str(EXP024_MAZE_DIR),
        time_budget=EXP024_TIME_BUDGET_S,
        max_runs=EXP024_MAX_RUNS,
    )


@pytest.fixture(scope="module")
def exp024_maze_41001_result(exp024_evaluator) -> Dict[str, object]:
    """maze_41001（対象6迷路中で最短歩数が最小の52）を実際に
    `CompetitionEvaluator` で走らせる。module スコープにして、以降の
    検査(a)(b)(c)が同じ走行結果を再利用できるようにする（5〜15分かかる）。"""
    policy = ClassicExplorerPolicy()
    npz_path = EXP024_MAZE_DIR / f"maze_{EXP024_MAZE_ID}.npz"
    result = exp024_evaluator.evaluate_maze(npz_path, policy)
    return {"result": result, "policy": policy}


def test_fast_run_through_the_competition_evaluator_does_not_lose_any_runs(
        exp024_maze_41001_result):
    """是正4: `tests/test_classic_fast_run.py` の検査は、これまで
    `CompetitionEvaluator` を一度も通しておらず、評価器の2段階ゴール判定
    （`body_fully_inside` / `goal_not_contained`）を素通りしていた。ここで
    実際の迷路（maze_41001, 最短歩数52）を評価器で走らせ、次を検査する:
      (a) runs[] に outcome == "goal_not_contained" が1件も無いこと
      (b) 走行開始時の段階が FAST だった走行が1本以上あり、その outcome が
          "goal" であること（最短走行が競技評価器の上で実際に成立すること）
      (c) 実測値（各走行の index/outcome/run_time、段階の記録）を印字する

    🔴 合否のしきい値（タイムの比など）は置かない。それは
    `experiments/exp_024_s3_fast_run/PREREG.md` に事前登録された判定条文の
    領分であり、ここでは「走行が失われていないこと」だけを見る。"""
    result = exp024_maze_41001_result["result"]
    policy: ClassicExplorerPolicy = exp024_maze_41001_result["policy"]
    runs = result["runs"]
    run_phases = policy.get_run_phases()
    phase_by_index = {p["run_index"]: p["phase"] for p in run_phases}

    lines = [
        f"  走行{r['index']}: 開始段階={phase_by_index.get(r['index'], '?')} "
        f"outcome={r['outcome']} run_time={r['run_time']}"
        for r in runs
    ]
    print(f"\n[実測] maze_{EXP024_MAZE_ID}（D0=52）の各走行 "
          f"(持ち時間{EXP024_TIME_BUDGET_S}s・最大{EXP024_MAX_RUNS}走行):\n"
          + "\n".join(lines))
    print(f"[実測] run_phases（走行開始時点の段階の列）: {run_phases}")
    print(f"[実測] 走行本数: {len(runs)}")

    goal_not_contained = [r for r in runs if r["outcome"] == "goal_not_contained"]
    print(f"[実測] (a) goal_not_contained の件数: {len(goal_not_contained)}")
    assert not goal_not_contained, (
        f"goal_not_contained が {len(goal_not_contained)} 件あった"
        f"（ゴール停止ホールドが不十分な可能性。是正1 参照）: {goal_not_contained}"
    )

    fast_started_runs = [r for r in runs if phase_by_index.get(r["index"]) == "FAST"]
    fast_goal_runs = [r for r in fast_started_runs if r["outcome"] == "goal"]
    print(f"[実測] (b) FASTで開始した走行: {[r['index'] for r in fast_started_runs]} "
          f"うちゴールしたもの: {[r['index'] for r in fast_goal_runs]}")
    assert fast_started_runs, (
        "FASTで開始した走行が1本も無かった（最短走行そのものが始まっていない）"
    )
    assert fast_goal_runs, (
        "FASTで開始した走行はあったが、1本もゴールしなかった"
    )


# ==========================================================================
# 是正6: Phase.RETURN2（最短走行の帰路）が Phase.FAST と同じ
# 「plan_route のコマンド列を実行する」機構で走ることの検査
# ==========================================================================
_TURN_SUFFIX_FOR_TYPE = {
    CommandType.STRAIGHT: "straight",
    CommandType.TURN_RIGHT90: "turn_right",
    CommandType.TURN_LEFT90: "turn_left",
    CommandType.TURN_180: "turn_180",
}  # GOAL_STOP は含めない — RETURN2 はこれを "return2:goal_stop" として
   # 発行せず、受け取った瞬間に次の最短走行(_begin_fast_run)へ移る
   # （モジュール docstring・classic/explorer.py の docstring 参照）。


def test_return2_executes_the_same_command_sequence_as_plan_route(
        tmp_path, params, carved_maze_and_walls):
    """是正6 作動側: Phase.RETURN2 が Phase.FAST と同じ「plan_route の
    コマンド列を実行する」機構で走ることを、T1（`test_fast_executes_the_
    same_command_sequence_as_plan_route`）と同型の検査で確かめる。

    RETURN2 開始時点（FAST のゴール停止ホールドが完了した瞬間）の
    `ex.cell`・`ex.heading` を使って独立に `plan_route` を呼び、その出力
    （末尾の GOAL_STOP を除く）と、実行時の plan_id 列（"return2:*" だけを
    取り出し重複除去したもの）・`start_forward` に渡された区画数の列を
    突き合わせる。"""
    maze, v_walls, h_walls = carved_maze_and_walls
    sim = _fresh_sim(tmp_path, params, v_walls, h_walls, "t_corr6_cmds")
    ex = _fresh_explorer_in_fast(sim, params, maze)

    recorded_n_cells_by_phase: List[Tuple[str, int]] = []
    original_start_forward = ex.motion.start_forward

    def _wrapped_start_forward(n):
        recorded_n_cells_by_phase.append((ex.phase.name, n))
        return original_start_forward(n)

    ex.motion.start_forward = _wrapped_start_forward

    # 1本目の最短走行(FAST)をゴール停止ホールド完了まで駆動する。この時点
    # (_begin_return2_run が呼ばれる直前)の cell・heading が RETURN2 の
    # plan_route 呼び出しに使われる値（classic/explorer.py の
    # _begin_return2_run の実装どおり）。
    _drive_to_goal_stop_hold_complete(sim, ex)
    return2_start_cell = ex.cell
    return2_start_heading = ex.heading

    expected_path, expected_commands = plan_route(
        maze, start=return2_start_cell, goals=[ex.start_cell],
        mode=FloodMode.PESSIMISTIC, start_heading=return2_start_heading,
    )
    expected_labels = [
        f"return2:{_TURN_SUFFIX_FOR_TYPE[c.type]}" for c in expected_commands
        if c.type != CommandType.GOAL_STOP
    ]
    expected_straight_cells = [c.cells for c in expected_commands if c.type == CommandType.STRAIGHT]

    # RETURN2 を実行し、次の最短走行(Phase.FAST)へ入るまで駆動を続ける。
    plan_ids_return2: List[str] = []
    for _ in range(8000):
        obs = sim.observation()
        vl, vr, plan_id = ex.tick(obs)
        plan_ids_return2.append(plan_id)
        sim.step_control(vl, vr)
        if ex.phase is Phase.FAST:
            break
    else:
        raise AssertionError("8000 ティック以内に RETURN2 が完了しなかった")

    got_labels = _distinct_sequence([p for p in plan_ids_return2 if p.startswith("return2:")])
    got_straight_cells = [n for phase_name, n in recorded_n_cells_by_phase if phase_name == "RETURN2"]

    print(f"\n[実測] RETURN2 開始時点: cell={return2_start_cell} heading={return2_start_heading}")
    print(f"[実測] plan_route(RETURN2相当)の出力: {expected_commands}")
    print(f"[実測] RETURN2 が実行した plan_id 列(重複除去): {got_labels}")
    print(f"[実測] RETURN2 中に start_forward に渡された区画数の列: {got_straight_cells}")
    print(f"[実測] RETURN2 完了(=次のFAST開始)時点の到達区画: {ex.cell}"
          f"（plan_route の経路終点: {expected_path[-1]}）")

    assert expected_path[-1] == ex.start_cell, (
        "テスト前提が崩れている（RETURN2 の目的地はスタート区画のはず）"
    )
    assert got_labels == expected_labels, (
        "Phase.RETURN2 が実行したコマンド種別の列が plan_route の出力と一致しない"
    )
    assert got_straight_cells == expected_straight_cells, (
        "Phase.RETURN2 が start_forward に渡した区画数の列が"
        " plan_route の STRAIGHT.cells と一致しない"
    )
    assert ex.cell == ex.start_cell, "RETURN2 完了時の到達区画がスタート区画でない"


def test_return2_never_calls_update_map_from_sensing(tmp_path, params, carved_maze_and_walls):
    """是正6 作動側（是正2と同じ配線の検査）: Phase.RETURN2 の走行中も
    `_update_map_from_sensing` が一度も呼ばれないこと（地図は探索完了
    時点で既に確定しており、帰路でも書き換えない設計。空振り側は
    `test_explore_run_always_calls_update_map_from_sensing` が兼ねる —
    同じ配線・同じメソッドを対象にした検査であるため）。"""
    maze, v_walls, h_walls = carved_maze_and_walls
    sim = _fresh_sim(tmp_path, params, v_walls, h_walls, "t_corr6_map")
    ex = _fresh_explorer_in_fast(sim, params, maze)
    calls = _count_calls(ex, "_update_map_from_sensing")

    _drive_to_goal_stop_hold_complete(sim, ex)
    calls_after_fast = calls[0]

    ticks = 0
    for _ in range(8000):
        obs = sim.observation()
        vl, vr, _plan_id = ex.tick(obs)
        sim.step_control(vl, vr)
        ticks += 1
        if ex.phase is Phase.FAST:
            break
    else:
        raise AssertionError("8000 ティック以内に RETURN2 が完了しなかった")

    print(f"\n[実測] _update_map_from_sensing 呼び出し回数: "
          f"FAST完了時点={calls_after_fast} RETURN2完了(駆動{ticks}ティック)時点={calls[0]}")
    assert calls_after_fast == 0, "Phase.FAST 走行中に _update_map_from_sensing が呼ばれた"
    assert calls[0] == 0, (
        "Phase.RETURN2 走行中に _update_map_from_sensing が呼ばれた"
        "（最短走行の帰路で地図を書き換えている）"
    )
