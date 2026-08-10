"""
tests/test_baseline_straightrun.py
================
competition/baseline_straightrun.py（StraightRunPolicy, L0-b）の単体・end-to-end テスト。

pytest は使わない plain Python スクリプト（tests/test_baseline.py と同じ流儀）。
実行方法（リポジトリルートで）:
    .venv/bin/python tests/test_baseline_straightrun.py

L0-a（tests/test_baseline.py）のテストのうち経路決定に関わるもの（flood-fill、
タイブレーク、TURN、単一区画の FORWARD）は L0-a と完全に同一であることの
確認を兼ねて再掲する。L0-b 固有の追加項目:
  - 直進区間の先読み _lookahead_straight_cells 単体（既知の直進回廊を
    走り抜けること、未知壁で保守的に打ち切ること、経路が曲がる手前で
    打ち切ること）
  - 速度上限の物理的根拠（壁検出→停止, v<=sqrt(2*a_forward*d_detect)）が
    sim.params から正しく導出されクリップされること
  - 評価迷路 end-to-end: **validation 帯（seed 4000-4019）** の 2〜3 面で
    completion を確認する。test 帯（seed 1000-1019、研究計画書 §9-7）は
    マイルストーン判定専用のため、実装の動作確認には使わない。

いずれかのテストで例外/assert が起きても他のテストは継続実行し、最後に全テストの
実測値をまとめて表として print する。
"""
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from competition.baseline_straightrun import StraightRunPolicy, _HEADING_RAD, _wrap_pi  # noqa: E402
from competition.evaluator import CompetitionEvaluator  # noqa: E402
from mouse.mjcf import build_maze_robot_xml  # noqa: E402
from mouse.params import RobotParams  # noqa: E402
from mouse.sim import MouseSim  # noqa: E402

# 研究計画書 §9-7 のデータ三分割規律: 検証用 = seed 4000-4019。
# gate判定（マイルストーン）専用の test帯（seed 1000-1019）は実装調整に使わない。
VALIDATION_MAZE_DIR = os.path.join(REPO_ROOT, "competition", "mazes", "validation")
VALIDATION_SEEDS = (4000, 4001, 4002)
OPEN_FLOOR_XML = os.path.join(REPO_ROOT, "assets", "mouse_v2.xml")

RESULTS = []  # list[dict(name, expected, actual, passed, note)]


def record(name, expected, actual, passed, note=""):
    RESULTS.append(dict(name=name, expected=expected, actual=actual, passed=bool(passed), note=note))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: expected={expected}, actual={actual} {note}")


# ==========================================================================
# テスト1: flood-fill 単体（L0-a と同一ロジックであることの確認。
# tests/test_baseline.py test1 と同じ手書き4x4迷路・同じ期待値）
# ==========================================================================
def _build_hand_maze_4x4():
    width, height = 4, 4
    v_walls = np.zeros((width + 1, height), dtype=int)
    h_walls = np.zeros((width, height + 1), dtype=int)
    v_walls[0, :] = 1
    v_walls[width, :] = 1
    h_walls[:, 0] = 1
    h_walls[:, height] = 1

    v_walls[1, 0] = 1   # スタート東壁
    h_walls[0, 2] = 1   # (0,1)-(0,2) 間を塞ぐ（迂回強制）
    v_walls[3, 1] = 1   # (2,1)-(3,1) 間を塞ぐ（行き止まり）
    return v_walls, h_walls


def test1_floodfill_hand_maze():
    print("\n=== テスト1: flood-fill単体（手書き4x4、L0-aと同一ロジックの確認） ===")
    v_walls, h_walls = _build_hand_maze_4x4()

    pol = StraightRunPolicy()
    pol.on_maze_start({"width": 4, "height": 4})
    pol._true_v = v_walls
    pol._true_h = h_walls
    for cx in range(4):
        for cy in range(4):
            pol._do_sense(cx, cy)

    dist = pol._flood_fill(pol._target_cells())

    expected = {
        (1, 1): 0, (1, 2): 0, (2, 1): 0, (2, 2): 0,
        (0, 1): 1, (1, 0): 1, (0, 2): 1, (1, 3): 1, (2, 0): 1, (3, 2): 1, (2, 3): 1,
        (0, 0): 2, (0, 3): 2, (3, 0): 2, (3, 3): 2, (3, 1): 2,
    }
    all_ok = True
    for cell, exp_d in sorted(expected.items()):
        actual_d = dist.get(cell)
        ok = (actual_d == exp_d)
        all_ok = all_ok and ok
        record(f"floodfill_dist{cell}", exp_d, actual_d, ok)

    ok_count = (len(dist) == 16)
    all_ok = all_ok and ok_count
    record("floodfill_all_cells_reachable", 16, len(dist), ok_count, "全16セルが到達可能")

    return all_ok


def test1b_floodfill_unknown_optimistic():
    print("\n=== テスト1b: 未知壁(-1)の楽観的扱い単体（1x3廊下、L0-aと同一ロジック） ===")
    pol = StraightRunPolicy()
    pol.width, pol.height = 3, 1

    all_ok = True

    pol.v_walls_known = np.array([[1], [-1], [0], [1]], dtype=int).reshape(4, 1)
    pol.h_walls_known = np.array([[1, 1], [1, 1], [1, 1]], dtype=int)
    dist_a = pol._flood_fill([(2, 0)])
    ok_a = (dist_a.get((0, 0)) == 2) and (dist_a.get((1, 0)) == 1)
    all_ok = all_ok and ok_a
    record("floodfill_unknown_treated_as_open", {(0, 0): 2, (1, 0): 1},
           {(0, 0): dist_a.get((0, 0)), (1, 0): dist_a.get((1, 0))}, ok_a,
           "v[1,0]=-1(未知)を壁なしとみなし(0,0)も到達可能になること")

    pol.v_walls_known = np.array([[1], [1], [0], [1]], dtype=int).reshape(4, 1)
    dist_b = pol._flood_fill([(2, 0)])
    ok_b = ((0, 0) not in dist_b) and (dist_b.get((1, 0)) == 1)
    all_ok = all_ok and ok_b
    record("floodfill_known_wall_blocks", "((0,0) not reachable)",
           f"(0,0) in dist={(0, 0) in dist_b}", ok_b,
           "v[1,0]=1(既知の壁)は塞ぎ(0,0)は到達不能のままであること（対照実験）")

    return all_ok


# ==========================================================================
# テスト2: タイブレーク単体（L0-a と同一ロジック）
# ==========================================================================
def test2_tiebreak_order():
    print("\n=== テスト2: タイブレーク単体（直進優先→時計回り優先、L0-aと同一ロジック） ===")
    all_ok = True

    cases = [
        ("N", ["N", "E", "S", "W"]),
        ("E", ["E", "S", "W", "N"]),
        ("S", ["S", "W", "N", "E"]),
        ("W", ["W", "N", "E", "S"]),
    ]
    for prev_dir, expected_order in cases:
        actual_order = StraightRunPolicy._tiebreak_order(prev_dir)
        ok = (actual_order == expected_order)
        all_ok = all_ok and ok
        record(f"tiebreak_order(prev={prev_dir})", expected_order, actual_order, ok)

    for heading_dir, expected_dir in [("N", "N"), ("E", "E")]:
        pol2 = StraightRunPolicy()
        pol2.width, pol2.height = 4, 4
        pol2.v_walls_known = np.zeros((5, 4), dtype=int)
        pol2.v_walls_known[0, :] = 1
        pol2.v_walls_known[4, :] = 1
        pol2.h_walls_known = np.zeros((4, 5), dtype=int)
        pol2.h_walls_known[:, 0] = 1
        pol2.h_walls_known[:, 4] = 1
        pol2.target_mode = "to_goal"
        pol2._heading_dir = heading_dir
        pol2._do_plan(0.09, 0.09)  # セル(0,0)中心
        ok = (pol2._planned_dir == expected_dir)
        all_ok = all_ok and ok
        record(f"tiebreak_via_do_plan(heading={heading_dir})", expected_dir,
               pol2._planned_dir, ok, "同距離の東/北候補から直進優先で選ばれること")

    return all_ok


# ==========================================================================
# テスト3: TURN単体（平原XML、90度/180度旋回。L0-aと同一ロジック）
# ==========================================================================
def test3_turn():
    print("\n=== テスト3: TURN単体（平原XML、90度/180度旋回、L0-aと同一ロジック） ===")
    if not os.path.exists(OPEN_FLOOR_XML):
        print(f"  [SKIP] {OPEN_FLOOR_XML} が見つかりません")
        record("turn_e2e", "実行", "SKIP(平原XMLなし)", False)
        return False

    params = RobotParams()
    all_ok = True

    for target_dir, label in [("E", "90度(北->東)"), ("S", "90度(北->南)"), ("W", "180度(北->西)")]:
        sim = MouseSim(OPEN_FLOOR_XML, params=params)
        sim.full_reset(cell=(0, 0), heading_deg=90.0)
        pol = StraightRunPolicy()
        pol.bind_sim(sim)
        # L0-bはTURN完了後に_start_straight_run(先読み)を呼ぶため、L0-aと違い
        # 壁地図(v_walls_known/h_walls_known)が初期化されている必要がある
        # （中身は本テストの主旨=TURNのPD収束には無関係なので、平原相当の
        # 適当な迷路サイズで初期化するだけでよい）。
        pol.on_maze_start({"width": 4, "height": 4})
        pol._heading_dir = "N"
        pol._planned_dir = target_dir
        pol._planned_next_cell = (0, 1)  # ダミー(TURN完了後のFORWARD遷移で使うが本テストでは未到達)
        pol._enter_turn(target_dir)

        max_steps = int(5.0 / params.control_dt)
        t_done = None
        for _ in range(max_steps):
            obs = sim.observation()
            _x, _y, yaw = sim.privileged_pose()
            vl, vr = pol._do_turn(obs, yaw)
            res = sim.step_control(vl, vr)
            if pol._state != "TURN":
                t_done = res["sim_time"]
                break

        _x, _y, yaw = sim.privileged_pose()
        err_deg = math.degrees(_wrap_pi(_HEADING_RAD[target_dir] - yaw))
        ok_converge = (t_done is not None)
        ok_err = ok_converge and (abs(err_deg) < 2.0)
        all_ok = all_ok and ok_converge and ok_err
        record(f"turn_{label}_converged", True, ok_converge, ok_converge,
               f"t_done={t_done}")
        record(f"turn_{label}_final_err_deg", "<2.0", f"{err_deg:.3f}", ok_err,
               f"所要時間={t_done:.3f}s" if t_done else "")

    return all_ok


# ==========================================================================
# テスト4: FORWARD単体（廊下XML、n=1区画。姿勢維持ロジックはL0-aと同一）
# ==========================================================================
def _build_corridor_xml(out_path, width=1, height=5, params=None):
    params = params or RobotParams()
    v_walls = np.zeros((width + 1, height), dtype=int)
    h_walls = np.zeros((width, height + 1), dtype=int)
    v_walls[0, :] = 1
    v_walls[width, :] = 1
    h_walls[:, 0] = 1
    h_walls[:, height] = 1
    build_maze_robot_xml(v_walls, h_walls, out_path, params=params)
    return v_walls, h_walls


def test4_forward_single_cell():
    print("\n=== テスト4: FORWARD単体（廊下XML、n=1区画） ===")
    params = RobotParams()
    tmp_dir = tempfile.mkdtemp(prefix="test_baseline_straightrun_corridor_")
    xml_path = os.path.join(tmp_dir, "corridor.xml")
    width, height = 1, 5
    v_walls, h_walls = _build_corridor_xml(xml_path, width, height, params)

    sim = MouseSim(xml_path, params=params)
    sim.full_reset(cell=(0, 0), heading_deg=90.0)
    pol = StraightRunPolicy()
    pol.bind_sim(sim)
    pol.on_maze_start({"width": width, "height": height})
    pol.bind_maze(v_walls, h_walls)
    pol._heading_dir = "N"
    x0, y0, _yaw0 = sim.privileged_pose()
    pol._enter_forward(x0, y0, target_cell=(0, 1))  # n=1: 次の1区画だけ

    max_steps = int(5.0 / params.control_dt)
    t_done = None
    for _ in range(max_steps):
        obs = sim.observation()
        x, y, yaw = sim.privileged_pose()
        vl, vr = pol._do_forward(obs, x, y, yaw)
        res = sim.step_control(vl, vr)
        if pol._state != "FORWARD":
            t_done = res["sim_time"]
            break

    x, y, _yaw = sim.privileged_pose()
    target_x, target_y = 0.09, 0.27  # セル(0,1)中心
    err_m = math.hypot(x - target_x, y - target_y)

    ok_converge = (t_done is not None)
    ok_err = ok_converge and (err_m < 0.01)
    record("forward_n1_converged", True, ok_converge, ok_converge, f"t_done={t_done}")
    record("forward_n1_stop_pos_err_m", "<0.01", f"{err_m:.5f}", ok_err,
           f"所要時間={t_done:.3f}s" if t_done else "")

    return ok_converge and ok_err


def test4b_forward_multi_cell():
    print("\n=== テスト4b: FORWARD単体（廊下XML、n=3区画を区画境界で減速せず走り抜け） ===")
    params = RobotParams()
    tmp_dir = tempfile.mkdtemp(prefix="test_baseline_straightrun_corridor3_")
    xml_path = os.path.join(tmp_dir, "corridor3.xml")
    width, height = 1, 6
    v_walls, h_walls = _build_corridor_xml(xml_path, width, height, params)

    sim = MouseSim(xml_path, params=params)
    sim.full_reset(cell=(0, 0), heading_deg=90.0)
    pol = StraightRunPolicy()
    pol.bind_sim(sim)
    pol.on_maze_start({"width": width, "height": height})
    pol.bind_maze(v_walls, h_walls)
    pol._heading_dir = "N"
    x0, y0, _yaw0 = sim.privileged_pose()
    pol._enter_forward(x0, y0, target_cell=(0, 3))  # n=3区画先まで一括

    max_steps = int(8.0 / params.control_dt)
    t_done = None
    max_speed = 0.0
    speed_near_cell1_boundary = None  # 区画境界(y=0.18)通過時の速度: 0に落ちていないことを確認
    for _ in range(max_steps):
        obs = sim.observation()
        x, y, yaw = sim.privileged_pose()
        v_fwd, _omega = sim.privileged_velocity()
        max_speed = max(max_speed, abs(v_fwd))
        if speed_near_cell1_boundary is None and y >= 0.18:
            speed_near_cell1_boundary = abs(v_fwd)
        vl, vr = pol._do_forward(obs, x, y, yaw)
        res = sim.step_control(vl, vr)
        if pol._state != "FORWARD":
            t_done = res["sim_time"]
            break

    x, y, _yaw = sim.privileged_pose()
    target_x, target_y = 0.09, 3 * 0.18 + 0.09  # セル(0,3)中心
    err_m = math.hypot(x - target_x, y - target_y)

    ok_converge = (t_done is not None)
    ok_err = ok_converge and (err_m < 0.01)
    # 区画境界(1区画目/2区画目の境目)通過時、L0-aのように0まで減速していないこと
    # （目安: forward_done_speedしきい値より明らかに速い）
    ok_no_stop = (speed_near_cell1_boundary is not None
                  and speed_near_cell1_boundary > 5 * pol.forward_done_speed)
    record("forward_n3_converged", True, ok_converge, ok_converge, f"t_done={t_done}")
    record("forward_n3_stop_pos_err_m", "<0.01", f"{err_m:.5f}", ok_err,
           f"所要時間={t_done:.3f}s" if t_done else "")
    record("forward_n3_no_stop_at_boundary",
           f">{5 * pol.forward_done_speed:.3f}m/s",
           f"{speed_near_cell1_boundary}", ok_no_stop,
           "区画境界を停止せず通過すること（L0-aとの差分の直接確認）")
    # 速度指令(setpoint)はv_maxに厳密にクリップされるが、実測値(privileged_velocity)は
    # 車輪速度制御の追従遅れでわずかにオーバーシュートしうるため、5%の許容を見る。
    speed_tol = 1.05 * pol.v_max
    record("forward_n3_max_speed_m_s", f"<={speed_tol:.3f}(v_max={pol.v_max:.3f}+5%)",
           f"{max_speed:.3f}", max_speed <= speed_tol,
           "速度上限を大きく超えないこと（指令値はv_maxに厳密クリップ、実測はわずかな追従遅れを許容）")

    return ok_converge and ok_err and ok_no_stop


# ==========================================================================
# テスト5: 直進区間の先読み _lookahead_straight_cells 単体（L0-b固有・§7の核心）
# ==========================================================================
def test5_lookahead_known_corridor():
    print("\n=== テスト5: 先読み単体（既知の直進回廊をゴールまで走り抜ける） ===")
    pol = StraightRunPolicy()
    pol.width, pol.height = 5, 1
    pol.v_walls_known = np.array([[1, 0, 0, 0, 0, 1]], dtype=int).reshape(6, 1)
    pol.h_walls_known = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=int).T

    dist_field = {(0, 0): 4, (1, 0): 3, (2, 0): 2, (3, 0): 1, (4, 0): 0}
    n = pol._lookahead_straight_cells((1, 0), "E", dist_field)
    ok = (n == 4)
    record("lookahead_full_known_corridor", 4, n, ok,
           "既知壁のみの直進回廊(0,0)->(4,0)を1回の直進区間で走り抜けること")
    return ok


def test5b_lookahead_unknown_wall_truncates():
    print("\n=== テスト5b: 先読み単体（未知壁で保守的に打ち切り） ===")
    pol = StraightRunPolicy()
    pol.width, pol.height = 5, 1
    # (2,0)-(3,0) 間の壁を未知(-1)にする（他は既知・開放）
    pol.v_walls_known = np.array([[1, 0, 0, -1, 0, 1]], dtype=int).reshape(6, 1)
    pol.h_walls_known = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=int).T

    dist_field = {(0, 0): 4, (1, 0): 3, (2, 0): 2, (3, 0): 1, (4, 0): 0}
    n = pol._lookahead_straight_cells((1, 0), "E", dist_field)
    # (1,0)->(2,0)までは既知壁で拡張できるが、(2,0)-(3,0)間が未知のため
    # そこで打ち切り(未知は壁ありと保守的に扱う)。経路決定自体(_select_direction)は
    # 楽観的規則のままなので、実際の移動先(_do_plan)としては(3,0)方向が
    # 選ばれ続ける点は変わらない(本テストは先読み=速度計画の打ち切りのみ確認)。
    ok = (n == 2)
    record("lookahead_unknown_wall_truncates", 2, n, ok,
           "未知壁(-1)の手前で先読みを打ち切ること(壁ありと保守的に扱う)")
    return ok


def test5c_lookahead_turn_ahead_truncates():
    print("\n=== テスト5c: 先読み単体（この先で曲がる場合は打ち切り） ===")
    pol = StraightRunPolicy()
    pol.width, pol.height = 3, 3
    pol.v_walls_known = np.full((4, 3), -1, dtype=int)
    pol.v_walls_known[0, :] = 1
    pol.v_walls_known[3, :] = 1
    pol.v_walls_known[2, 1] = 0   # (1,1)-(2,1) 東: 開放（直進候補）
    pol.h_walls_known = np.full((3, 4), -1, dtype=int)
    pol.h_walls_known[:, 0] = 1
    pol.h_walls_known[:, 3] = 1
    pol.h_walls_known[1, 2] = 0   # (1,1)-(1,2) 北: 開放（ゴールへの近道）
    pol.h_walls_known[1, 1] = 1   # (1,0)-(1,1) 南: 閉鎖

    # (1,1) では北(1,2)がゴール(距離0)で東(2,1)より近い→flood-fillは直進(E)ではなく
    # 北への旋回を選ぶはず。先読みはnext_cell=(1,1)自体を含む(n=1)が、そこから
    # 先へは拡張しない。
    dist_field = {(1, 1): 2, (1, 2): 0, (2, 1): 10, (0, 1): 10}
    n = pol._lookahead_straight_cells((1, 1), "E", dist_field)
    ok = (n == 1)
    record("lookahead_turn_ahead_truncates", 1, n, ok,
           "次区画で曲がる(北への近道がある)場合はそこで先読みを打ち切ること")
    return ok


# ==========================================================================
# テスト6: 速度上限の物理的根拠（壁検出→停止）が sim.params から
# ハードコードなしで導出・クリップされること（教授裁定 2026-08-10）
# ==========================================================================
class _FakeSim:
    """bind_sim() は sim.params しか参照しないため、MuJoCo実体を持たない
    軽量スタブで十分。"""
    def __init__(self, params):
        self.params = params


def test6_velocity_cap_derivation():
    print("\n=== テスト6: 速度上限の物理的根拠（v<=sqrt(2*a*d_detect)）の導出 ===")
    params = RobotParams()
    all_ok = True

    # ケースA: 要求v_maxが物理上限を明らかに超える場合、クリップされること
    pol = StraightRunPolicy(v_max=5.0, a_forward=3.4, detect_margin=0.6)
    pol.bind_sim(_FakeSim(params))

    d_detect_raw = min(params.sensor_cutoff, params.cell_size)
    d_detect_expected = 0.6 * d_detect_raw
    v_cap_expected = math.sqrt(2.0 * 3.4 * d_detect_expected)

    ok_d = abs(pol._d_detect - d_detect_expected) < 1e-9
    ok_cap = abs(pol._v_max_physical_cap - v_cap_expected) < 1e-9
    ok_clip = abs(pol.v_max - v_cap_expected) < 1e-9
    all_ok = all_ok and ok_d and ok_cap and ok_clip
    record("d_detect", f"{d_detect_expected:.4f}", f"{pol._d_detect:.4f}", ok_d,
           f"min(sensor_cutoff={params.sensor_cutoff}, cell_size={params.cell_size})*0.6")
    record("v_max_physical_cap", f"{v_cap_expected:.4f}", f"{pol._v_max_physical_cap:.4f}", ok_cap,
           "sqrt(2*a_forward*d_detect)")
    record("v_max_clipped_when_requested_too_high", f"{v_cap_expected:.4f}", f"{pol.v_max:.4f}",
           ok_clip, "要求v_max=5.0 m/sは物理上限を超えるためクリップされること")

    # ケースB: 要求v_maxが物理上限未満なら、そのまま維持されること
    pol2 = StraightRunPolicy(v_max=0.05, a_forward=3.4, detect_margin=0.6)
    pol2.bind_sim(_FakeSim(params))
    ok_keep = abs(pol2.v_max - 0.05) < 1e-9
    all_ok = all_ok and ok_keep
    record("v_max_kept_when_requested_below_cap", 0.05, pol2.v_max, ok_keep,
           "要求v_max=0.05 m/sは物理上限未満なのでクリップされないこと")

    return all_ok


# ==========================================================================
# テスト7: 評価迷路 end-to-end（validation帯 seed 4000, 4001, 4002。
# 研究計画書 §9-7: test帯 seed 1000-1019 はマイルストーン判定専用のため
# 実装調整には使わない）
# ==========================================================================
def test7_evaluator_e2e_validation():
    print("\n=== テスト7: 評価迷路end-to-end（validation帯 seed 4000-4002） ===")
    all_ok = True

    for seed in VALIDATION_SEEDS:
        npz_path = os.path.join(VALIDATION_MAZE_DIR, f"maze_{seed}.npz")
        if not os.path.exists(npz_path):
            print(f"  [SKIP] {npz_path} が見つかりません")
            record(f"eval_seed{seed}_e2e", "実行", "SKIP(迷路なし)", False)
            all_ok = False
            continue

        evaluator = CompetitionEvaluator(maze_dir=VALIDATION_MAZE_DIR)
        pol = StraightRunPolicy()
        t0 = time.time()
        result = evaluator.evaluate_maze(npz_path, pol)
        wall_clock = time.time() - t0

        runs = result["runs"]
        print(f"  seed={seed}: wall_clock={wall_clock:.2f}s, n_runs={len(runs)}, "
              f"best_time={result['best_time']}, success={result['success']}")
        for r in runs:
            print(f"    run#{r['index']}: outcome={r['outcome']} run_time={r['run_time']:.3f} "
                  f"visited_cells={r['visited_cells']} path_efficiency={r['path_efficiency']}")
        for inc in result["incidents"]:
            print(f"    incident: t={inc['t']:.2f} kind={inc['kind']} pos={inc['pos']}")

        ok_success = result["success"] is True
        n_collisions = sum(1 for r in runs if r["outcome"] == "collision")
        n_collisions += sum(1 for i in result["incidents"] if i["kind"] == "collision")
        ok_no_collision = (n_collisions == 0)

        all_ok = all_ok and ok_success and ok_no_collision
        record(f"eval_seed{seed}_success", True, ok_success, ok_success,
               f"best_time={result['best_time']}")
        record(f"eval_seed{seed}_no_collision", 0, n_collisions, ok_no_collision)

    return all_ok


# ==========================================================================
# メイン
# ==========================================================================
def main():
    print("competition/baseline_straightrun.py (L0-b) テストスイート")
    print("=" * 70)

    test_fns = [
        test1_floodfill_hand_maze,
        test1b_floodfill_unknown_optimistic,
        test2_tiebreak_order,
        test3_turn,
        test4_forward_single_cell,
        test4b_forward_multi_cell,
        test5_lookahead_known_corridor,
        test5b_lookahead_unknown_wall_truncates,
        test5c_lookahead_turn_ahead_truncates,
        test6_velocity_cap_derivation,
        test7_evaluator_e2e_validation,
    ]

    overall_ok = []
    t_suite_start = time.time()
    for fn in test_fns:
        try:
            ok = fn()
        except AssertionError as e:
            print(f"  [ERROR] {fn.__name__}: assertion failed: {e}")
            ok = False
        except Exception as e:  # noqa: BLE001 - テスト継続のため広く捕捉
            print(f"  [ERROR] {fn.__name__}: 例外発生: {e!r}")
            import traceback
            traceback.print_exc()
            ok = False
        overall_ok.append(ok)
    t_suite = time.time() - t_suite_start

    print("\n" + "=" * 70)
    print("全テスト実測値まとめ")
    print("=" * 70)
    print(f"{'項目':45s} {'期待値':>28s} {'実測値':>28s} {'判定':<5s}")
    print("-" * 70)
    for r in RESULTS:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['name']:45s} {str(r['expected']):>28s} {str(r['actual']):>28s} "
              f"{status:<5s}  {r['note']}")

    n_pass = sum(1 for r in RESULTS if r["passed"])
    n_total = len(RESULTS)
    print("-" * 70)
    print(f"合計: {n_pass}/{n_total} 項目 PASS")
    print(f"テスト関数レベル: {sum(1 for x in overall_ok if x)}/{len(overall_ok)} 個が全項目PASS")
    print(f"テストスイート全体の実行時間: {t_suite:.2f}s")

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
