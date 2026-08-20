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
from classic.fast_planner import PathBlock
from classic.localization import (
    DEFAULT_LATERAL_CORRECTION_FRACTION, DEFAULT_MAX_FORWARD_CORRECTION_M, estimate_lateral_offset,
)
from classic.maze_map import ALL_DIRECTIONS, Direction, MazeMap, direction_between
from classic.maze_map import WallState as MapWallState
from classic.sensing import WallSensing
from classic.sensing import WallState as SenseWallState
from classic.tracker import TrackerGains

Cell = Tuple[int, int]

REPO_ROOT = Path(__file__).resolve().parent.parent

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


# `_apply_wall_correction` の前後位置補正の検査専用: 最初の直線副区間を長く
# 取った回廊（KNOWN_PATH は最初のターンまで2区画=0.36mしか無く、弧に食われる
# 分を差し引くと合成テストに必要な余地が不足する）。北へ6区画→東へ折れて
# ゴール中央2x2領域（8x8迷路の(3,3)-(4,4)）へ抜ける。
LONG_STRAIGHT_W, LONG_STRAIGHT_H = 8, 8
LONG_STRAIGHT_PATH: List[Cell] = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
    (1, 6), (2, 6), (3, 6), (3, 5), (3, 4),
]


@pytest.fixture(scope="module")
def long_straight_maze_and_walls():
    return _carved_path_maze_and_walls(LONG_STRAIGHT_W, LONG_STRAIGHT_H, LONG_STRAIGHT_PATH)


def _fresh_sim(tmp_path: Path, params: RobotParams, v_walls, h_walls, label: str) -> MouseSim:
    xml_path = str(tmp_path / f"{label}.xml")
    build_maze_robot_xml(v_walls, h_walls, xml_path, model_name=label, params=params)
    sim = MouseSim(xml_path, params=params)
    sim.full_reset(cell=(0, 0), heading_deg=90.0)
    return sim


def _fresh_explorer_in_fast(sim: MouseSim, params: RobotParams, maze: MazeMap,
                             fast_mode: str = "command", friction_use: float = 1.0,
                             clearance_margin_m: float = 0.005,
                             wall_correction: bool = False, wall_correction_mode: str = "blend",
                             width: int = MAZE_W, height: int = MAZE_H) -> ClassicExplorer:
    ex = ClassicExplorer(width, height, params=params, fast_mode=fast_mode,
                          friction_use=friction_use, clearance_margin_m=clearance_margin_m,
                          wall_correction=wall_correction,
                          wall_correction_mode=wall_correction_mode)
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
# 3b. 係員回収（handle_retrieval）をまたいでも、ProfileTracker・壁センサ位置
#     補正の内部状態は次の FAST(profile) 計画・追従へ影響しない
#     （exp_027 一次記録で観測された「1本目の FAST だけ他と違う／2本目以降は
#     互いに完全に同一」という型の診断結果を固定する回帰検査。
#     `experiments/exp_027_friction_sweep/diagnose_retrieval.py` の実測
#     （maze_41004・u=0.50・wall_correction=True）で確認したのは:
#       - `ProfileTracker`/壁センサ位置補正の内部状態（積分器・s・yaw_est・
#         弧⇔直線判定の基準点）は `_begin_profile_run` が呼ばれるたびに
#         必ず reset()/load_plan()/load_spin_plan() で上書きされ、直前の
#         走行の値には一切依存しない（＝「持ち越し」バグは無かった）。
#       - 実際に違ったのは `start_heading`（自然な RETURN 完了時は実際の
#         到達方位、係員回収後は `handle_retrieval` が強制する
#         `Direction.N`）で、これは真値姿勢の実測とも一致する**正当な**
#         差である（回収前後で機体の向きが本当に異なるため）。
#     本検査は前者（state が持ち越されないこと）だけを固定する。後者
#     （start_heading が回収前後で異なりうること）は正しい挙動なので
#     「回収前後で FAST の計画が同一になること」は検査しない。
# ==========================================================================
def test_retrieval_does_not_leak_tracker_or_wall_correction_state(
        tmp_path, params, carved_maze_and_walls):
    maze, v_walls, h_walls = carved_maze_and_walls

    # (A) 「汚れた」explorer: 一度 FAST(profile) を走らせて tracker・壁センサ
    #     位置補正の内部状態を実際に進めた後、係員回収してもう一度 FAST を
    #     組み直す（評価器と同じ順序: sim.reset_to_start() → policy.on_retrieval()
    #     相当の ex.handle_retrieval()）。
    sim_a = _fresh_sim(tmp_path, params, v_walls, h_walls, "leak_a")
    ex_a = ClassicExplorer(MAZE_W, MAZE_H, params=params, fast_mode="profile", wall_correction=True)
    ex_a.maze = maze
    obs = sim_a.observation()
    ex_a._begin_fast_run(obs)
    ex_a._need_replan = False
    for _ in range(80):  # tracker.s・yaw_est・積分器・wc_* を実際に汚す
        obs = sim_a.observation()
        vl, vr, _plan_id = ex_a.tick(obs)
        sim_a.step_control(vl, vr)
    dirty_s = ex_a._tracker.s
    assert dirty_s > 0.0, "前提が崩れた: tracker.s が実際に進んでいない(汚れていない)"

    sim_a.reset_to_start(cell=(0, 0), heading_deg=90.0)
    ex_a.handle_retrieval()
    obs = sim_a.observation()
    # `tick()` ではなく `_on_stationary()` を直接呼ぶ（`_on_stationary_route` →
    # `_begin_fast_run` まで到達させ、その先の `_tick_profile`（tracker.update()
    # をもう1回呼ぶ）は呼ばない。(B)の「素の」explorerも `_begin_fast_run` を
    # 直接呼ぶだけで tracker.update() を挟まないため、両者を対称に保つ)。
    ex_a._on_stationary(obs)
    assert ex_a._fast_plan is not None

    # (B) 「素の」explorer: 一度も走らせず、いきなり FAST を組む対照。
    sim_b = _fresh_sim(tmp_path, params, v_walls, h_walls, "leak_b")
    ex_b = ClassicExplorer(MAZE_W, MAZE_H, params=params, fast_mode="profile", wall_correction=True)
    ex_b.maze = maze
    obs_b = sim_b.observation()
    ex_b._begin_fast_run(obs_b)
    assert ex_b._fast_plan is not None

    print(f"\n[実測] 汚れたtracker(回収直前): s={dirty_s:.4f}m")
    print(f"[実測] 回収後の計画: t_plan={ex_a._fast_plan.t_plan:.4f} cells={len(ex_a._fast_plan.cells)} "
          f"n_turns={ex_a._fast_plan.n_turns} n_forced_spins={ex_a._fast_plan.n_forced_spins}")
    print(f"[実測] 素の計画    : t_plan={ex_b._fast_plan.t_plan:.4f} cells={len(ex_b._fast_plan.cells)} "
          f"n_turns={ex_b._fast_plan.n_turns} n_forced_spins={ex_b._fast_plan.n_forced_spins}")

    # FastPlan は「地図・start・goals・start_heading」だけで決まるはず
    # （汚れたtrackerの値に一切依存しない）。回収後はどちらも
    # start_heading=Direction.N になる(ex_bは構築直後のself.heading=Direction.Nと一致)。
    assert ex_a._fast_plan.t_plan == ex_b._fast_plan.t_plan
    assert ex_a._fast_plan.cells == ex_b._fast_plan.cells
    assert ex_a._fast_plan.n_turns == ex_b._fast_plan.n_turns
    assert ex_a._fast_plan.n_forced_spins == ex_b._fast_plan.n_forced_spins

    # ProfileTracker・壁センサ位置補正の内部状態も「汚れ」を引きずらない。
    assert abs(ex_a._tracker.s - dirty_s) > 1e-6, "tracker.s が汚れた値のまま(reset漏れ)"
    assert ex_a._tracker.s == ex_b._tracker.s
    assert ex_a._tracker.yaw_estimate == ex_b._tracker.yaw_estimate
    assert ex_a._tracker._integ_l == 0.0 == ex_b._tracker._integ_l
    assert ex_a._tracker._integ_r == 0.0 == ex_b._tracker._integ_r
    assert ex_a._tracker._e_lat == 0.0 == ex_b._tracker._e_lat
    assert ex_a._wc_prev_kappa_nonzero is None
    assert ex_b._wc_prev_kappa_nonzero is None
    assert ex_a._wc_fwd_ref_s is None and ex_b._wc_fwd_ref_s is None


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


# ==========================================================================
# 5. friction_use（摩擦円の使用率 u）が ClassicExplorer から plan_fast_run() へ
#    実際に届いていること（配線そのものの検査。値の妥当性は tests/
#    test_fast_planner.py::test_friction_use_lowers_speed_without_changing_
#    the_route が既に見ている）
# ==========================================================================
def test_friction_use_is_threaded_to_the_plan(tmp_path, params, carved_maze_and_walls):
    maze, v_walls, h_walls = carved_maze_and_walls
    sim_full = _fresh_sim(tmp_path, params, v_walls, h_walls, "fu_full")
    ex_full = _fresh_explorer_in_fast(sim_full, params, maze, fast_mode="profile", friction_use=1.0)

    sim_half = _fresh_sim(tmp_path, params, v_walls, h_walls, "fu_half")
    ex_half = _fresh_explorer_in_fast(sim_half, params, maze, fast_mode="profile", friction_use=0.5)

    assert ex_full._fast_plan is not None and ex_half._fast_plan is not None
    print(f"\n[実測] t_plan: u=1.0 -> {ex_full._fast_plan.t_plan:.4f}s  "
          f"u=0.5 -> {ex_half._fast_plan.t_plan:.4f}s")
    assert ex_half._fast_plan.t_plan > ex_full._fast_plan.t_plan, (
        "friction_use=0.5 が ClassicExplorer から plan_fast_run() へ届いていない疑い"
        "(t_planが伸びていない)"
    )
    assert ex_half._fast_plan.cells == ex_full._fast_plan.cells, "friction_use で経路自体が変わった(半径探索・経路選択は u の影響を受けないはず)"


# ==========================================================================
# 5.5. clearance_margin_m（幾何の掃引と壁・柱のあいだに残す余裕）が
#      ClassicExplorer から plan_fast_run() へ実際に届いていること
#      （配線そのものの検査。値の妥当性は tests/test_fast_planner.py::
#      test_larger_clearance_margin_m_does_not_shorten_t_plan、半径への
#      効き方は tests/test_geometry.py::
#      test_larger_margin_shrinks_the_max_feasible_radius が既に見ている）
# ==========================================================================
def test_clearance_margin_m_is_threaded_to_the_plan(tmp_path, params, carved_maze_and_walls):
    maze, v_walls, h_walls = carved_maze_and_walls
    sim_default = _fresh_sim(tmp_path, params, v_walls, h_walls, "cm_default")
    ex_default = _fresh_explorer_in_fast(sim_default, params, maze, fast_mode="profile")

    sim_wide = _fresh_sim(tmp_path, params, v_walls, h_walls, "cm_wide")
    ex_wide = _fresh_explorer_in_fast(
        sim_wide, params, maze, fast_mode="profile", clearance_margin_m=0.030)

    assert ex_default._fast_plan is not None and ex_wide._fast_plan is not None
    print(f"\n[実測] t_plan: clearance_margin_m=5mm -> {ex_default._fast_plan.t_plan:.4f}s  "
          f"clearance_margin_m=30mm -> {ex_wide._fast_plan.t_plan:.4f}s")
    assert ex_wide._fast_plan.t_plan >= ex_default._fast_plan.t_plan, (
        "clearance_margin_m=30mm が ClassicExplorer から plan_fast_run() へ届いていない疑い"
        "(t_planが縮まなかった/伸びなかった)"
    )
    assert ex_wide._fast_plan.t_plan > ex_default._fast_plan.t_plan, (
        "clearance_margin_m を5mmから30mmへ上げてもt_planに差が無い"
    )
    assert ex_wide._fast_plan.cells == ex_default._fast_plan.cells, (
        "clearance_margin_m で経路自体が変わった(悲観歩数マップの経路選択は幾何の余裕を見ないはず)"
    )


def test_clearance_margin_m_default_is_5mm(params):
    """既定値 0.005（5mm）＝従来と同一であることの直接確認（`ClassicExplorer`
    レベル。`classic.fast_planner.plan_fast_run` レベルは tests/
    test_fast_planner.py::test_clearance_margin_m_default_matches_pre_existing_behavior
    が見ている）。"""
    ex = ClassicExplorer(MAZE_W, MAZE_H, params=params, fast_mode="profile")
    assert ex.clearance_margin_m == pytest.approx(0.005)


# ==========================================================================
# 6. wall_correction（壁センサによる位置補正）
# ==========================================================================
def test_wall_correction_default_off_means_zero_kp_lat(params):
    """既定 False のとき、`ProfileTracker` は `kp_lat=0`（既定ゲイン）のまま
    構築される（exp_026 と完全に同一の実行経路であることの直接確認）。"""
    ex = ClassicExplorer(MAZE_W, MAZE_H, params=params, fast_mode="profile")
    assert ex._tracker is not None
    assert ex._tracker.gains.kp_lat == 0.0


def test_wall_correction_true_sets_the_derived_kp_lat(params):
    """wall_correction=True のとき、kp_lat が `_apply_wall_correction`
    docstring どおりの式（既存のチューニング済み定数 kp_psi・
    DEFAULT_LATERAL_CORRECTION_FRACTION から導出）で設定されること。"""
    ex = ClassicExplorer(MAZE_W, MAZE_H, params=params, fast_mode="profile", wall_correction=True)
    assert ex._tracker is not None
    expected = -TrackerGains().kp_psi * (DEFAULT_LATERAL_CORRECTION_FRACTION / params.cell_size)
    assert ex._tracker.gains.kp_lat == pytest.approx(expected)
    assert ex._tracker.gains.kp_lat != 0.0


def _wall_sensing(left: SenseWallState, right: SenseWallState,
                   left_dist: float = 0.06, right_dist: float = 0.06) -> WallSensing:
    return WallSensing(front=SenseWallState.AMBIGUOUS, left=left, right=right,
                        front_dist=999.0, left_dist=left_dist, right_dist=right_dist)


def test_apply_wall_correction_snaps_forward_drift_to_the_nearest_cell_boundary(
        tmp_path, params, monkeypatch, long_straight_maze_and_walls):
    """`_apply_wall_correction` の (b) 前後位置補正（境界イベントの間隔は
    cell_size の整数倍という自己較正）を、`sense_walls` を差し替えて直接検査
    する。壁の生値そのものは使わず、境界通過イベント（側方壁の確定状態の
    反転）だけを人為的に発生させる。`long_straight_maze_and_walls`（最初の
    直線副区間が6区画=1.08m。弧に食われる分を差し引いても本検査に十分な
    余地がある）を使う（`KNOWN_PATH` は最初のターンまで2区画しか無く不足する）。

    本検査は「補正した s が cell_size の整数倍へ厳密に一致する」ことを見る
    ものなので `wall_correction_mode="snap"`（1回で一気に当てる従来方式）を
    明示する（既定の "blend" は `wall_correction_alpha` 倍・上限つきでしか
    当てないため、この厳密一致は成り立たない）。
    """
    maze, v_walls, h_walls = long_straight_maze_and_walls
    sim = _fresh_sim(tmp_path, params, v_walls, h_walls, "wc_fwd")
    ex = _fresh_explorer_in_fast(sim, params, maze, fast_mode="profile", wall_correction=True,
                                  wall_correction_mode="snap",
                                  width=LONG_STRAIGHT_W, height=LONG_STRAIGHT_H)
    assert ex._fast_plan is not None
    step = ex._fast_plan.steps[0]
    assert isinstance(step, PathBlock), "経路の先頭がPathBlockであるという前提が崩れた"

    cell_size = params.cell_size
    dummy_obs = sim.observation()

    # s=0.02（先頭の直線副区間の内部。s=0は必ず区画中心=直線の開始点になる
    # ―― classic/ideal.py::_geometry_blocks の block_starts の構成による）
    # がkappa=0であることを前提として使う。以降 s3 まで、最初のターンが
    # 弧に食う最大量(_R_HI=0.40m)を差し引いても直線区間内に収まる設計
    # （最初の直線副区間の生の長さは 6*cell_size=1.08m）。
    s0 = 0.02
    assert ex._tracker._kappa_at(s0) == 0.0, "テストの前提(先頭付近が直線)が崩れた"

    calls = {"sensing": _wall_sensing(SenseWallState.WALL, SenseWallState.WALL)}
    monkeypatch.setattr("classic.explorer.sense_walls", lambda obs, p: calls["sensing"])

    # tick 1: 最初の観測（両側WALL）。エッジ判定の基準ができるだけで、
    # 補正はまだ起きない。
    ex._tracker._s = s0
    ex._apply_wall_correction(dummy_obs)
    assert ex._wc_fwd_ref_s is None, "最初の観測でいきなり基準点ができてしまった"

    # tick 2: 左側がCLEARへ反転(区画境界を通過したという合図)。1回目のエッジ
    # なので較正だけ(基準点=このときのs)で補正はしない。
    s1 = s0 + 0.01
    ex._tracker._s = s1
    calls["sensing"] = _wall_sensing(SenseWallState.CLEAR, SenseWallState.WALL)
    ex._apply_wall_correction(dummy_obs)
    assert ex._wc_fwd_ref_s == pytest.approx(s1)
    assert ex._tracker._s == pytest.approx(s1), "較正だけのはずが弧長を書き換えた"

    # tick 3: 左側がWALLへ戻る(次の区画境界)。真の間隔は 1*cell_size だが、
    # 推測航法が +15mm 行き過ぎたと仮定する(小さな誤差。DEFAULT_MAX_FORWARD_
    # CORRECTION_M=50mm以内)。cell_sizeの整数倍への引き寄せで補正されるはず。
    drift = 0.015
    s2_measured = s1 + cell_size + drift
    s2_expected = s1 + cell_size
    ex._tracker._s = s2_measured
    assert ex._tracker._kappa_at(s2_measured) == 0.0, "テストの前提(直線区間内)が崩れた"
    calls["sensing"] = _wall_sensing(SenseWallState.WALL, SenseWallState.WALL)
    ex._apply_wall_correction(dummy_obs)
    print(f"\n[実測] 前後位置補正: 測定s={s2_measured:.4f} 補正後s={ex._tracker._s:.4f} "
          f"期待s={s2_expected:.4f}")
    assert ex._tracker._s == pytest.approx(s2_expected, abs=1e-9), "cell_sizeの整数倍へ補正されなかった"
    assert ex._wc_fwd_ref_s == pytest.approx(s2_expected, abs=1e-9)

    # tick 4: 次の境界で今度は大きくズレている(60mm。50mmの保険を超える)。
    # 誤検出とみなし弧長は書き換えず、基準点だけ現在地点へ張り直すはず。
    big_drift = 0.06
    assert big_drift > DEFAULT_MAX_FORWARD_CORRECTION_M
    s3_measured = s2_expected + cell_size + big_drift
    assert ex._tracker._kappa_at(s3_measured) == 0.0, "テストの前提(直線区間内)が崩れた"
    ex._tracker._s = s3_measured
    calls["sensing"] = _wall_sensing(SenseWallState.CLEAR, SenseWallState.WALL)
    ex._apply_wall_correction(dummy_obs)
    assert ex._tracker._s == pytest.approx(s3_measured), "大きすぎるズレなのに弧長を書き換えてしまった"
    assert ex._wc_fwd_ref_s == pytest.approx(s3_measured), "誤検出時に基準点が現在地点へ張り直されなかった"


def test_apply_wall_correction_ignores_a_same_boundary_double_detection(
        tmp_path, params, monkeypatch, long_straight_maze_and_walls):
    """是正（2026-08-20・北向き開始が南向き開始よりずっと脆い問題の診断で発見）:

    `k = round((s - ref) / cell_size)` が 0 になる（＝新しく観測した境界が
    基準点と同じ区画境界を指す）のに、その間 `s` は既に進んでいるという
    状況は、区画境界が実在してもう一度起きたのではなく、側方センサの
    AMBIGUOUS帯（重なり帯。`classic/sensing.py` docstring 参照）を機体の
    横ずれでまたいだ**同じ境界の二重検出**でしかあり得ない
    （区画境界は cell_size ごとにしか無いため）。これを信用して `s` を
    基準点まで後退させると、正しく進んでいた推測航法を壊す。

    実測（`experiments/exp_027_friction_sweep/diagnose_start_heading.py`、
    maze_41004・u=0.50・北向き開始）: s≈0.92m 付近でこの二重検出が起き、
    -47.9mm の誤補正のあと衝突まで一度も回復しなかった（到達距離が経路の
    約9%で終わっていた）。本検査はこの二重検出そのものを人為的に再現し、
    `_apply_wall_correction` が `s`/基準点のどちらも壊さないことを直接見る。
    """
    maze, v_walls, h_walls = long_straight_maze_and_walls
    sim = _fresh_sim(tmp_path, params, v_walls, h_walls, "wc_fwd_k0")
    ex = _fresh_explorer_in_fast(sim, params, maze, fast_mode="profile", wall_correction=True,
                                  width=LONG_STRAIGHT_W, height=LONG_STRAIGHT_H)
    assert ex._fast_plan is not None
    step = ex._fast_plan.steps[0]
    assert isinstance(step, PathBlock), "経路の先頭がPathBlockであるという前提が崩れた"

    cell_size = params.cell_size
    dummy_obs = sim.observation()

    s0 = 0.02
    assert ex._tracker._kappa_at(s0) == 0.0, "テストの前提(先頭付近が直線)が崩れた"

    calls = {"sensing": _wall_sensing(SenseWallState.WALL, SenseWallState.WALL)}
    monkeypatch.setattr("classic.explorer.sense_walls", lambda obs, p: calls["sensing"])

    # tick 1: 最初の観測（両側WALL）。基準を作るだけ。
    ex._tracker._s = s0
    ex._apply_wall_correction(dummy_obs)
    assert ex._wc_fwd_ref_s is None

    # tick 2: 左側がCLEARへ反転(1回目のエッジ)。較正だけで補正はしない。
    s1 = s0 + 0.01
    ex._tracker._s = s1
    calls["sensing"] = _wall_sensing(SenseWallState.CLEAR, SenseWallState.WALL)
    ex._apply_wall_correction(dummy_obs)
    assert ex._wc_fwd_ref_s == pytest.approx(s1)
    assert ex._tracker._s == pytest.approx(s1)

    # tick 3: cell_size(=0.18m)の半分(0.09m)よりずっと近い距離(0.03m=30mm)
    # で右側がCLEARへ反転(2回目のエッジ)。round((s2-s1)/cell_size)==0 になる
    # ―― 区画境界がこんなに近くにもう1つあることは無いので、これは同じ境界の
    # 二重検出とみなし、`_tracker._s` は書き換えず、基準点だけ現在地点へ
    # 張り直すはず。
    gap = 0.03
    assert gap < cell_size / 2.0, "テストの前提(半区画未満の近さ)が崩れた"
    s2 = s1 + gap
    assert round((s2 - s1) / cell_size) == 0, "テストの前提(k=0になる近さ)が崩れた"
    assert ex._tracker._kappa_at(s2) == 0.0, "テストの前提(直線区間内)が崩れた"
    ex._tracker._s = s2
    calls["sensing"] = _wall_sensing(SenseWallState.CLEAR, SenseWallState.CLEAR)
    ex._apply_wall_correction(dummy_obs)
    print(f"\n[実測] 同一境界の二重検出: s1={s1:.4f} s2(測定)={s2:.4f} "
          f"補正後s={ex._tracker._s:.4f} 基準点={ex._wc_fwd_ref_s}")
    assert ex._tracker._s == pytest.approx(s2), (
        "k=0の二重検出なのに弧長を基準点まで後退させてしまった"
        "（北向き開始が壊れていた原因そのもの）"
    )
    assert ex._wc_fwd_ref_s == pytest.approx(s2), "基準点が現在地点へ張り直されなかった"


def test_apply_wall_correction_lateral_matches_localization_estimate(
        tmp_path, params, monkeypatch, carved_maze_and_walls):
    """(a) 横位置補正: `_apply_wall_correction` が `ProfileTracker._e_lat` へ
    渡す値は `classic.localization.estimate_lateral_offset` の推定値そのもの
    であること（直線区間）。弧区間では新規の推定はせず、直前の直線区間の
    推定値を保持する（0.0へは戻さない。是正 2026-08-20。理由・実測は
    `classic/explorer.py` の `DEFAULT_WALL_CORRECTION_LOOKAHEAD_M` 直下の
    調査コメント参照）ことも見る。"""
    maze, v_walls, h_walls = carved_maze_and_walls
    sim = _fresh_sim(tmp_path, params, v_walls, h_walls, "wc_lat")
    ex = _fresh_explorer_in_fast(sim, params, maze, fast_mode="profile", wall_correction=True)
    assert ex._fast_plan is not None
    step = ex._fast_plan.steps[0]
    assert isinstance(step, PathBlock)
    dummy_obs = sim.observation()

    # 直線区間: 左右の距離を非対称にして、非ゼロの横ずれ推定を作る。
    sensing = _wall_sensing(SenseWallState.WALL, SenseWallState.WALL, left_dist=0.05, right_dist=0.07)
    expected = estimate_lateral_offset(sensing, params)
    assert expected is not None and expected.offset_m != 0.0, "テスト用センシングが非ゼロの推定を作れていない"

    monkeypatch.setattr("classic.explorer.sense_walls", lambda obs, p: sensing)
    s0 = 0.02
    assert ex._tracker._kappa_at(s0) == 0.0
    ex._tracker._s = s0
    ex._apply_wall_correction(dummy_obs)
    print(f"\n[実測] 横位置補正: 推定={expected.offset_m * 1000:.3f}mm "
          f"tracker._e_lat={ex._tracker._e_lat * 1000:.3f}mm")
    assert ex._tracker._e_lat == pytest.approx(expected.offset_m)

    # 弧区間: 曲率が非ゼロの s を探し、(1) センシングが変わっても新規の推定を
    # 拾わないこと、(2) 直前の直線区間の推定値（0.0ではない）を保持し続ける
    # ことを確かめる（弧の途中は姿勢が斜めで前提が崩れるため新規推定はしない。
    # 0.0へ戻すのは「理由なく捨てる」動きだったので是正 2026-08-20）。
    arc_s = None
    for i, kap in enumerate(step.kappa_ref):
        if kap != 0.0:
            arc_s = 0.5 * (step.s_grid[i] + step.s_grid[i + 1])
            break
    assert arc_s is not None, "この経路に弧区間が無い(テストの前提が崩れた)"
    other_sensing = _wall_sensing(SenseWallState.WALL, SenseWallState.WALL, left_dist=0.07, right_dist=0.05)
    other_expected = estimate_lateral_offset(other_sensing, params)
    assert other_expected is not None and other_expected.offset_m != pytest.approx(expected.offset_m), (
        "弧区間用のセンシングが直線区間と異なる推定を作れていない(テストの前提が崩れた)"
    )
    monkeypatch.setattr("classic.explorer.sense_walls", lambda obs, p: other_sensing)
    ex._tracker._s = arc_s
    ex._apply_wall_correction(dummy_obs)
    print(f"[実測] 弧区間: 直線側の推定={expected.offset_m * 1000:.3f}mm "
          f"弧側のセンシングでの推定={other_expected.offset_m * 1000:.3f}mm "
          f"tracker._e_lat={ex._tracker._e_lat * 1000:.3f}mm")
    assert ex._tracker._e_lat == pytest.approx(expected.offset_m), (
        "弧区間で横位置補正の推定値が変わってしまった"
        "（新規推定を拾った、または理由なく0.0へ戻された）"
    )


# ==========================================================================
# 6b. wall_correction_mode="blend"（既定）: 前後位置補正を「張り替え」から
#     「混ぜ込み」に変える是正（任務指示 2026-08-20）。
# ==========================================================================
def test_kappa_nonzero_within_lookahead_detects_upcoming_curvature_only_within_range():
    """`_kappa_nonzero_within_lookahead`（純粋関数）の単体検査。`_apply_wall_
    correction` の "blend" モードが使う「曲率が立ち上がる直前は補正しない」
    判定そのもの（任務指示 2026-08-20 の 3）。"""
    from classic.explorer import _kappa_nonzero_within_lookahead

    # s=[0,0.5)->kappa0.0, [0.5,1.0)->kappa0.0, [1.0,1.5)->kappa3.0(弧)
    step = PathBlock(s_grid=(0.0, 0.5, 1.0, 1.5), v_ref=(1.0, 1.0, 1.0, 1.0),
                      kappa_ref=(0.0, 0.0, 3.0), psi_start=0.0, t_plan=1.0)

    # 現在地 s=0.2: 弧の開始(s=1.0)まで0.8m。lookaheadが届かなければFalse。
    assert _kappa_nonzero_within_lookahead(step, 0.2, 0.3) is False
    # lookaheadが弧の開始まで届けばTrue。
    assert _kappa_nonzero_within_lookahead(step, 0.2, 0.9) is True
    # ちょうど境界(0.8m先)ぴったりは「まだ届いていない」扱い(半開区間、
    # horizon=s+L 自身は含まない)。浮動小数点の境界一致に依存しない設計。
    assert _kappa_nonzero_within_lookahead(step, 0.2, 0.8) is False
    assert _kappa_nonzero_within_lookahead(step, 0.2, 0.8 + 1e-6) is True
    # L=0(見送り機能そのものを無効化する既定の使い方)は「ちょうど先頭に
    # 立っている」場合も含めて常にFalse(実運用では kappa_nonzero_now が
    # 真のときはこの関数を呼ばない前提なので、s が弧の開始点そのものに
    # 一致するケースは通常起きない)。
    assert _kappa_nonzero_within_lookahead(step, 1.0, 0.0) is False
    assert _kappa_nonzero_within_lookahead(step, 0.999, 0.0) is False
    # 弧を通り過ぎた後(s=1.6、ブロック範囲外)は先に何も無いのでFalse。
    assert _kappa_nonzero_within_lookahead(step, 1.6, 1.0) is False


def test_apply_wall_correction_blend_caps_a_single_large_correction(
        tmp_path, params, monkeypatch, long_straight_maze_and_walls):
    """作動側: `wall_correction_mode="blend"` は 1 回の検出が示す補正量が
    大きくても（DEFAULT_MAX_FORWARD_CORRECTION_M=50mm未満なので誤検出としては
    弾かれない大きさ）、`wall_correction_alpha`・`wall_correction_max_step_m`
    のうち厳しいほうで頭打ちになり、1回で全量を当てない（任務指示
    2026-08-20「1回あたりの上限を設ける」）。同じ入力を
    `wall_correction_mode="snap"`（従来）に与えると全量がそのまま1回で
    当たることと対比する——これが「1回でも誤ると走行が終わる」性質
    そのものであり、"blend" が抑えようとしている対象である。"""
    maze, v_walls, h_walls = long_straight_maze_and_walls
    cell_size = params.cell_size
    alpha = 0.8
    cap_m = 0.010
    drift = 0.030  # 30mm。50mm閾値未満(誤検出扱いされない)だが cap(10mm)は超える大きさ。
    assert alpha * drift > cap_m, "テストの前提(capが実際に効く大きさ)が崩れた"

    calls = {"sensing": _wall_sensing(SenseWallState.WALL, SenseWallState.WALL)}
    monkeypatch.setattr("classic.explorer.sense_walls", lambda obs, p: calls["sensing"])

    results = {}
    for mode, tag in [("blend", "cap_blend"), ("snap", "cap_snap")]:
        sim = _fresh_sim(tmp_path, params, v_walls, h_walls, tag)
        ex = _fresh_explorer_in_fast(sim, params, maze, fast_mode="profile", wall_correction=True,
                                      wall_correction_mode=mode,
                                      width=LONG_STRAIGHT_W, height=LONG_STRAIGHT_H)
        ex.wall_correction_alpha = alpha
        ex.wall_correction_max_step_m = cap_m
        ex.wall_correction_lookahead_m = 0.0  # このテストはL(曲率直前見送り)の対象外にする
        dummy_obs = sim.observation()

        s0 = 0.02
        assert ex._tracker._kappa_at(s0) == 0.0
        ex._tracker._s = s0
        ex._apply_wall_correction(dummy_obs)  # 最初の観測。基準はまだ無い。

        s1 = s0 + 0.01
        ex._tracker._s = s1
        calls["sensing"] = _wall_sensing(SenseWallState.CLEAR, SenseWallState.WALL)
        ex._apply_wall_correction(dummy_obs)  # 1回目のエッジ。基準点=s1。
        assert ex._wc_fwd_ref_s == pytest.approx(s1)

        s2_measured = s1 + cell_size + drift
        assert ex._tracker._kappa_at(s2_measured) == 0.0, "テストの前提(直線区間内)が崩れた"
        ex._tracker._s = s2_measured
        calls["sensing"] = _wall_sensing(SenseWallState.WALL, SenseWallState.WALL)
        ex._apply_wall_correction(dummy_obs)  # 2回目のエッジ。補正が発生する。

        applied = ex._tracker._s - s2_measured
        results[mode] = applied
        print(f"\n[実測] mode={mode}: 測定s={s2_measured:.4f} 適用後s={ex._tracker._s:.4f} "
              f"適用量={applied * 1000:+.2f}mm (真の補正量={-drift * 1000:+.2f}mm)")

    assert results["snap"] == pytest.approx(-drift, abs=1e-9), (
        "snapモードは全量(-30mm)を1回で当てるはず(従来どおりの回帰確認)"
    )
    assert results["blend"] != 0.0, "blendなのに補正そのものが機能していない"
    assert abs(results["blend"]) <= cap_m + 1e-9, (
        "blendなのに上限(10mm)を超えて当ててしまった(作動側の失敗)"
    )
    assert abs(results["blend"]) < abs(results["snap"]), (
        "blendがsnapと同じかそれ以上の量を1回で当ててしまった(作動側の失敗)"
    )


def test_apply_wall_correction_blend_applies_alpha_scaled_correction_when_under_cap(
        tmp_path, params, monkeypatch, long_straight_maze_and_walls):
    """空振り側: 上限（`wall_correction_max_step_m`）に掛からない普通サイズの
    ドリフトでは、"blend" は `wall_correction_alpha * correction` をそのまま
    当てる——上限・見送り（このテストは L=0 で無効化）が余計な歪みを加えず、
    「混ぜ込み」そのものは素直に効くことの確認（作動側の検査が上限機構だけを
    見ているのに対する対照）。"""
    maze, v_walls, h_walls = long_straight_maze_and_walls
    cell_size = params.cell_size
    alpha = 0.8
    cap_m = 0.030  # alpha*drift(=8mm)より十分大きく、cap は効かない。
    drift = 0.010  # 10mm。50mm閾値未満・cap未満の"普通サイズ"のドリフト。
    assert alpha * drift < cap_m, "テストの前提(capが効かない大きさ)が崩れた"

    sim = _fresh_sim(tmp_path, params, v_walls, h_walls, "blend_scaled")
    ex = _fresh_explorer_in_fast(sim, params, maze, fast_mode="profile", wall_correction=True,
                                  wall_correction_mode="blend",
                                  width=LONG_STRAIGHT_W, height=LONG_STRAIGHT_H)
    ex.wall_correction_alpha = alpha
    ex.wall_correction_max_step_m = cap_m
    ex.wall_correction_lookahead_m = 0.0
    dummy_obs = sim.observation()

    calls = {"sensing": _wall_sensing(SenseWallState.WALL, SenseWallState.WALL)}
    monkeypatch.setattr("classic.explorer.sense_walls", lambda obs, p: calls["sensing"])

    s0 = 0.02
    ex._tracker._s = s0
    ex._apply_wall_correction(dummy_obs)

    s1 = s0 + 0.01
    ex._tracker._s = s1
    calls["sensing"] = _wall_sensing(SenseWallState.CLEAR, SenseWallState.WALL)
    ex._apply_wall_correction(dummy_obs)

    s2_measured = s1 + cell_size + drift
    assert ex._tracker._kappa_at(s2_measured) == 0.0
    ex._tracker._s = s2_measured
    calls["sensing"] = _wall_sensing(SenseWallState.WALL, SenseWallState.WALL)
    ex._apply_wall_correction(dummy_obs)

    applied = ex._tracker._s - s2_measured
    expected = -alpha * drift
    print(f"\n[実測] 空振り側: 適用量={applied * 1000:+.3f}mm 期待値(alpha*correction)={expected * 1000:+.3f}mm")
    assert applied == pytest.approx(expected, abs=1e-9), (
        "上限に掛からない普通サイズの補正で alpha*correction からずれた"
        "(空振り側の失敗: 上限・見送りが余計な歪みを加えている)"
    )


def test_apply_wall_correction_blend_vetoes_correction_just_before_curvature_onset(
        tmp_path, params, monkeypatch, carved_maze_and_walls):
    """`wall_correction_lookahead_m`（L）: 現在の s から先 L 以内に曲率が
    非ゼロになる区間があるとき、blend は補正そのものを見送る（s・基準点の
    どちらも書き換えない）。`carved_maze_and_walls`（最初のターンまで2区画
    しかない）を使い、弧の直前まで s を進めた状態で検査する。"""
    maze, v_walls, h_walls = carved_maze_and_walls
    sim = _fresh_sim(tmp_path, params, v_walls, h_walls, "blend_veto")
    ex = _fresh_explorer_in_fast(sim, params, maze, fast_mode="profile", wall_correction=True,
                                  wall_correction_mode="blend")
    assert ex._fast_plan is not None
    step = ex._fast_plan.steps[0]
    assert isinstance(step, PathBlock)

    # 最初に曲率が非ゼロになる格子点(弧の開始)を探す。
    arc_start = None
    for i, kap in enumerate(step.kappa_ref):
        if kap != 0.0:
            arc_start = step.s_grid[i]
            break
    assert arc_start is not None, "この経路に弧区間が無い(テストの前提が崩れた)"

    ex.wall_correction_alpha = 0.8
    ex.wall_correction_max_step_m = 0.010
    ex.wall_correction_lookahead_m = 0.05  # 弧の開始まで5cm以内なら見送る

    calls = {"sensing": _wall_sensing(SenseWallState.WALL, SenseWallState.WALL)}
    monkeypatch.setattr("classic.explorer.sense_walls", lambda obs, p: calls["sensing"])

    cell_size = params.cell_size

    # 基準点を作る(弧よりずっと手前)。
    s_ref = max(arc_start - 3 * cell_size, 0.02)
    assert ex._tracker._kappa_at(s_ref) == 0.0
    ex._tracker._s = s_ref
    ex._apply_wall_correction(dummy_obs := sim.observation())
    calls["sensing"] = _wall_sensing(SenseWallState.CLEAR, SenseWallState.WALL)
    ex._apply_wall_correction(dummy_obs)
    assert ex._wc_fwd_ref_s == pytest.approx(s_ref)

    # (A) 弧の開始まで L=0.05 より近い地点でエッジ発生 -> 見送られるはず。
    s_near = arc_start - 0.02  # 弧開始まで20mm(<L=50mm)
    assert s_near > s_ref, "テストの前提(基準点より先)が崩れた"
    assert ex._tracker._kappa_at(s_near) == 0.0, "テストの前提(直線区間内)が崩れた"
    ex._tracker._s = s_near + 0.003  # 実測ドリフトを少し足す(補正が起きるならわかるように)
    calls["sensing"] = _wall_sensing(SenseWallState.WALL, SenseWallState.WALL)
    ref_before = ex._wc_fwd_ref_s
    s_before = ex._tracker._s
    ex._apply_wall_correction(dummy_obs)
    print(f"\n[実測] 弧の開始まで{(arc_start - s_near) * 1000:.1f}mm(L={ex.wall_correction_lookahead_m*1000:.0f}mm): "
          f"s {s_before:.4f}->{ex._tracker._s:.4f}  ref {ref_before}->{ex._wc_fwd_ref_s}")
    assert ex._tracker._s == pytest.approx(s_before), "曲率立ち上がり直前なのに補正してしまった(検査項目3の失敗)"
    assert ex._wc_fwd_ref_s == ref_before, "曲率立ち上がり直前なのに基準点を書き換えてしまった"


# ==========================================================================
# 7. 北向き開始が南向き開始よりずっと脆い問題の再発防止
#    （2026-08-20・診断: `experiments/exp_027_friction_sweep/
#    diagnose_start_heading.py`。原因はテスト6の k=0 二重検出。ここでは
#    その症状そのもの――実際の迷路で北向き開始が異常に早く衝突すること――
#    を直接見る。EXPLORE/RETURN は使わず、迷路の真の壁をそのまま「既知の
#    地図」として与えて Phase.FAST を直接始める（このファイルの既存の
#    作法どおり。EXPLORE を挟むと数分かかり `-q` での前景完走に向かない）。
# ==========================================================================
NORTH_VS_SOUTH_MAZE_NPZ = REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "maze_41004.npz"


@pytest.fixture(scope="module")
def maze_41004_known_map_and_xml(tmp_path_factory):
    """maze_41004 の真の壁をそのまま『既知の地図』にした `MazeMap` と、
    対応する MJCF の XML パス。EXPLORE を一切使わないための近道
    （`experiments/exp_027_friction_sweep/judge.py::compute_t_plan_from_saved_map`
    と同じ「`MazeMap.v_walls`/`h_walls` へ直接代入する」作法）。"""
    data = np.load(NORTH_VS_SOUTH_MAZE_NPZ)
    v_walls_bool = data["v_walls"]
    h_walls_bool = data["h_walls"]
    width = int(data["width"]) if "width" in data else int(v_walls_bool.shape[0] - 1)
    height = int(data["height"]) if "height" in data else int(h_walls_bool.shape[1])

    maze = MazeMap(width, height)
    # 競技 npz の規約(1=壁あり・0=通行可、`competition/evaluator.py`
    # モジュールコメント参照)を MazeMap.WallState(WALL=1・OPEN=2)へ変換する。
    maze.v_walls[:, :] = np.where(v_walls_bool == 1, int(MapWallState.WALL), int(MapWallState.OPEN))
    maze.h_walls[:, :] = np.where(h_walls_bool == 1, int(MapWallState.WALL), int(MapWallState.OPEN))

    xml_dir = tmp_path_factory.mktemp("maze_41004_known_map")
    xml_path = str(xml_dir / "maze_41004.xml")
    build_maze_robot_xml(v_walls_bool, h_walls_bool, xml_path, model_name="maze_41004_known_map",
                          params=RobotParams())
    return maze, xml_path, width, height


def _drive_fast_run_to_completion(sim: MouseSim, ex: ClassicExplorer, max_ticks: int = 3000):
    """衝突またはゴール停止ホールドに達するまで駆動し、
    (到達ティック数, 衝突したか, 最終弧長[m未満・ProfileTracker.s]) を返す。"""
    n = 0
    collided = False
    for _ in range(max_ticks):
        obs = sim.observation()
        vl, vr, plan_id = ex.tick(obs)
        result = sim.step_control(vl, vr)
        n += 1
        if result["collision"]:
            collided = True
            break
        if plan_id == "fast:goal_stop":
            break
    s_final = ex._tracker.s if ex._tracker is not None else None
    return n, collided, s_final


def _begin_known_map_fast_run(maze_41004_known_map_and_xml, params: RobotParams,
                               heading_deg: float, ex_heading: Direction,
                               wall_correction_mode: str = "blend") -> Tuple[MouseSim, ClassicExplorer]:
    maze, xml_path, width, height = maze_41004_known_map_and_xml
    sim = MouseSim(xml_path, params=params)
    sim.full_reset(cell=(0, 0), heading_deg=heading_deg)
    ex = ClassicExplorer(width, height, params=params, fast_mode="profile",
                          friction_use=0.50, wall_correction=True,
                          wall_correction_mode=wall_correction_mode)
    ex.maze = maze
    ex.heading = ex_heading  # 実際の物理姿勢(heading_deg)と揃える(_begin_fast_run docstring参照)
    obs = sim.observation()
    ex._begin_fast_run(obs)
    ex._need_replan = False
    return sim, ex


def test_north_start_is_not_catastrophically_more_fragile_than_south_start(
        maze_41004_known_map_and_xml, params):
    """北向き開始（先頭が直線=その場旋回を挟まない）が、南向き開始（先頭が
    その場旋回）よりずっと早く衝突する、という exp_027 で見つかった脆さの
    直接検査。u=0.50・wall_correction=True・maze_41004（実在する迷路）で、
    北向き開始と南向き開始それぞれで Phase.FAST を直接始め、衝突するまでの
    到達ティック数を比べる。既定の `wall_correction_mode="blend"` で検査する
    （引数を明示していないので `_begin_known_map_fast_run` の既定＝実運用の
    既定と同じ経路）。

    是正前(k=0二重検出バグが残っていた状態)の実測: 北=186ティックで衝突・
    南=890ティックで衝突（比 0.21 — 北が南の1/5も走れない）。
    k=0是正後(snapのみ)の実測: 北=592ティック・南=868ティック（比 0.68）。
    blend 是正後の実測: 北=592〜593ティック・南=865ティック（比 0.68。
    誤差の範囲でほぼ同値）。
    弧突入時に横位置補正の推定値を0へ捨てるのをやめた是正後（2026-08-20。
    `classic/explorer.py` の `DEFAULT_WALL_CORRECTION_LOOKAHEAD_M` 直下の
    調査コメント参照）の実測: 北=595ティック（不変）・南=1073ティック
    （868→1073・大幅に改善。比は上がるが下限0.5は変えない）。
    しきい値は「是正前(186ティック・比0.21)」を確実に落とし、それ以降の
    どの状態にも十分な余裕を残す位置に置く。

    🔴 実測で判明した想定外（そのまま記録する）: 北向き開始の衝突地点
    （s≈4.77〜4.79m。半径0.178mの弧の途中）は `wall_correction=False`
    （壁センサ補正を全く使わない対照）でも同じ地点で起こる。**計画そのものは
    北向きと南向きで完全に同一**（`PathBlock` がビット一致。異なるのは
    南向きが先頭に実在の180°その場旋回を挟むことだけ）ことも確認済みで、
    北向き開始の頭打ちは計画の違いではない。原因は「その弧のマージン
    （設計5mm）と弧の間の横方向フィードバック欠如が生む、初期条件への
    鋭敏な依存性（カオス的）」までは特定したが、`friction_use`・`margin`・
    先頭へのその場旋回付加のいずれを動かしても再現性のある形では直せて
    いない（詳細・実測は `DEFAULT_WALL_CORRECTION_LOOKAHEAD_M` 直下の
    コメント参照）。このため `wall_correction_mode` をどう変えても
    北向き開始の到達ティック数はほぼ変わらない。
    """
    sim_n, ex_n = _begin_known_map_fast_run(
        maze_41004_known_map_and_xml, params, heading_deg=90.0, ex_heading=Direction.N)
    n_ticks_north, collided_north, s_north = _drive_fast_run_to_completion(sim_n, ex_n)

    sim_s, ex_s = _begin_known_map_fast_run(
        maze_41004_known_map_and_xml, params, heading_deg=-90.0, ex_heading=Direction.S)
    n_ticks_south, collided_south, s_south = _drive_fast_run_to_completion(sim_s, ex_s)

    ratio = n_ticks_north / n_ticks_south if n_ticks_south else float("inf")
    print(f"\n[実測] 北向き開始: {n_ticks_north}ティック 衝突={collided_north} s={s_north}")
    print(f"[実測] 南向き開始: {n_ticks_south}ティック 衝突={collided_south} s={s_south}")
    print(f"[実測] 到達ティック数の比(北/南): {ratio:.3f}")

    assert n_ticks_north >= 400, (
        f"北向き開始が{n_ticks_north}ティックしか走れなかった"
        "（是正前の水準=186ティックに近い。k=0の二重検出デバウンスの再発を疑う）"
    )
    assert ratio >= 0.5, (
        f"北向き開始が南向き開始に比べて著しく脆い(比={ratio:.3f})。"
        "「北向き開始でも南向き開始と同程度に走れる」という要件が崩れている"
    )


def test_wall_correction_mode_snap_matches_pre_blend_baseline_exactly(
        maze_41004_known_map_and_xml, params):
    """回帰検査: `wall_correction_mode="snap"` は "blend" と横位置補正
    （part (a)。`wall_correction_mode` に関わらず共通のコード経路）を共有する
    ため、決定論的な MuJoCo シミュレーションは常に厳密に同じティック数で
    衝突するはず。この値が変わったら、`wall_correction_mode` に関わらない
    共通コード（横位置補正・弧/直線判定・前後位置補正の基準点管理）が
    意図せず変わったことを疑う。

    比較対象の値は、弧突入時に横位置補正の推定値を0へ捨てるのをやめた
    是正（2026-08-20。`classic/explorer.py` の
    `DEFAULT_WALL_CORRECTION_LOOKAHEAD_M` 直下の調査コメント参照）を
    反映した実測（北=595ティック・南=617ティック）。この是正より前は
    北=592ティック・南=868ティックだった
    （`test_north_start_is_not_catastrophically_more_fragile_than_south_start`
    docstring 参照）。
    """
    sim_n, ex_n = _begin_known_map_fast_run(
        maze_41004_known_map_and_xml, params, heading_deg=90.0, ex_heading=Direction.N,
        wall_correction_mode="snap")
    n_ticks_north, collided_north, _s_north = _drive_fast_run_to_completion(sim_n, ex_n)

    sim_s, ex_s = _begin_known_map_fast_run(
        maze_41004_known_map_and_xml, params, heading_deg=-90.0, ex_heading=Direction.S,
        wall_correction_mode="snap")
    n_ticks_south, collided_south, _s_south = _drive_fast_run_to_completion(sim_s, ex_s)

    print(f"\n[実測] snap: 北={n_ticks_north}ティック(衝突={collided_north}) "
          f"南={n_ticks_south}ティック(衝突={collided_south})")

    assert collided_north and collided_south
    assert n_ticks_north == 595, (
        f"wall_correction_mode='snap' の北向き開始が想定と一致しない(実測={n_ticks_north})。"
        "wall_correction_mode に関わらない共通コード（横位置補正等）が変わった疑い(回帰)"
    )
    assert n_ticks_south == 617, (
        f"wall_correction_mode='snap' の南向き開始が想定と一致しない(実測={n_ticks_south})。"
        "wall_correction_mode に関わらない共通コード（横位置補正等）が変わった疑い(回帰)"
    )


def test_north_and_south_start_plan_the_identical_path_block(maze_41004_known_map_and_xml, params):
    """北向き開始が頭打ちになる原因の切り分け検査その1（2026-08-20・
    `DEFAULT_WALL_CORRECTION_LOOKAHEAD_M` 直下の調査コメント参照）:
    **計画（`PathBlock`）そのものは北向きと南向きで完全に同一**であり、
    頭打ちの原因が計画の違い（半径配分・曲率列）ではないことの回帰検査。

    北向き開始の計画は `steps=(PathBlock,)`（先頭がその場旋回なしで
    そのまま経路追従）、南向き開始の計画は `steps=(SpinSegment, PathBlock)`
    （先頭に実在の180°その場旋回。開始方位が経路の必要方位と逆向きのため）
    になるが、**その場旋回の後に続く `PathBlock` は北向き開始の唯一の
    `PathBlock` とビット一致する**（`s_grid`/`kappa_ref`/`v_ref` すべて）。
    「61対62ターン」の差はこの1回の強制その場旋回の有無だけで、経路の
    形状そのものは同一である。
    """
    from classic.explorer import _goal_cells
    from classic.fast_planner import plan_fast_run

    maze, _xml_path, width, height = maze_41004_known_map_and_xml
    goals = _goal_cells(width, height)

    plan_north = plan_fast_run(maze, start=(0, 0), goals=goals, start_heading=Direction.N,
                                params=params, friction_use=0.50)
    plan_south = plan_fast_run(maze, start=(0, 0), goals=goals, start_heading=Direction.S,
                                params=params, friction_use=0.50)

    assert [type(s).__name__ for s in plan_north.steps] == ["PathBlock"], (
        "北向き開始の計画の先頭がその場旋回なしの前提が崩れている"
    )
    assert [type(s).__name__ for s in plan_south.steps] == ["SpinSegment", "PathBlock"], (
        "南向き開始の計画の先頭がその場旋回1回の前提が崩れている"
    )

    block_north = plan_north.steps[0]
    block_south = plan_south.steps[1]
    assert block_north.s_grid == block_south.s_grid, "s_grid が北向き/南向きで一致しない"
    assert block_north.kappa_ref == block_south.kappa_ref, "kappa_ref が北向き/南向きで一致しない"
    assert block_north.v_ref == block_south.v_ref, "v_ref が北向き/南向きで一致しない"


def test_settling_spin_does_not_rescue_north_start(maze_41004_known_map_and_xml, params):
    """北向き開始が頭打ちになる原因の切り分け検査その2（2026-08-20・
    委譲指示で最優先とされた実験の回帰版）: 北向き開始の計画の先頭へ
    人為的にその場旋回を挟んでも到達弧長（`s_final`。到達ティック数
    ではない — その場旋回自体は距離を進まないぶんティック数を底上げする
    ので、比較には弧長を使う）は伸びないことの回帰検査。

    「先頭にその場旋回が無いから発進直後の追従が不安定で、それが頭打ちの
    原因」という仮説が正しいなら、その場旋回を挟むことで発進直後に
    フィードバック制御が収束する時間ができ、到達弧長が伸びるはずである。
    実測（`_apply_wall_correction` の是正後・maze_41004・u=0.50）:
        素のまま:      s_final=4.7877m
        +360度spin:   s_final=4.7377m（わずかに悪化・誤差の範囲）
        +720度spin:   s_final=0.0708m（壊滅的に悪化）
        +1080度spin:  s_final=0.0205m（壊滅的に悪化）
    360度（1回転して同じ向きに戻る、最も穏当な「発進直後の追従に時間を
    与えるだけ」の操作）でさえ到達弧長を改善しない。この仮説は否定される
    （詳細は `DEFAULT_WALL_CORRECTION_LOOKAHEAD_M` 直下の調査コメント参照）。
    このテストは「その場旋回を足せば直る」という誤った修正が将来再導入
    されないための回帰検査。
    """
    import math
    from dataclasses import replace as dc_replace

    from classic.fast_planner import SpinSegment
    from classic.profile import spin_turn_time, vehicle_limits

    maze, xml_path, width, height = maze_41004_known_map_and_xml

    sim_baseline, ex_baseline = _begin_known_map_fast_run(
        maze_41004_known_map_and_xml, params, heading_deg=90.0, ex_heading=Direction.N)
    _n_baseline, collided_baseline, s_baseline = _drive_fast_run_to_completion(sim_baseline, ex_baseline)
    assert collided_baseline

    sim_spin, ex_spin = _begin_known_map_fast_run(
        maze_41004_known_map_and_xml, params, heading_deg=90.0, ex_heading=Direction.N)
    plan = ex_spin._fast_plan
    limits = vehicle_limits(params)
    delta = 2.0 * math.pi  # 360度: 元の向きへ戻るので計画の幾何自体は変えない
    st = spin_turn_time(delta, limits)
    spin = SpinSegment(delta_theta=delta, psi_start=plan.steps[0].psi_start, t_plan=st.time)
    new_plan = dc_replace(plan, steps=(spin,) + tuple(plan.steps), t_plan=plan.t_plan + st.time)
    ex_spin._fast_plan = new_plan
    ex_spin._fast_plan_step_idx = 0
    ex_spin._tracker.reset(heading_deg=math.degrees(new_plan.steps[0].psi_start))
    ex_spin._load_current_profile_step()
    _n_spin, collided_spin, s_spin = _drive_fast_run_to_completion(sim_spin, ex_spin)
    assert collided_spin

    print(f"\n[実測] 北向き開始 素のまま: s_final={s_baseline:.4f}m")
    print(f"[実測] 北向き開始 先頭に360度spin: s_final={s_spin:.4f}m")

    assert s_spin < s_baseline * 1.05, (
        f"先頭に360度のその場旋回を挟んだら到達弧長が明確に伸びた"
        f"（素={s_baseline:.4f}m・spin後={s_spin:.4f}m）。"
        "「その場旋回が頭打ちを直す」という仮説が実は正しかった可能性がある。"
        "本テストの前提（実測では伸びない）が崩れているので、まず再現するか確認すること"
    )
