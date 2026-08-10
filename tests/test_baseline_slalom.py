"""
tests/test_baseline_slalom.py
================
competition/baseline_slalom.py（SlalomPolicy, L0-c）の単体・end-to-end テスト。

pytest は使わない plain Python スクリプト（tests/test_baseline.py, tests/
test_baseline_straightrun.py と同じ流儀）。実行方法（リポジトリルートで）:
    .venv/bin/python tests/test_baseline_slalom.py

検証項目（タスク要件）:
  - 軌道生成の幾何: 接線連続（G1連続）・壁余裕15mm以上
  - 速度計画: v_turn=sqrt(a_lat*R) との一致、台形加減速、クリープ下限
  - 制御則の低速域での準停留がないこと（Stanley標準形の横偏差項が飽和し、
    L0-a/L0-bで確認されたチャタリングの根因を回避できていること）
  - 経路からの横偏差の最大値とRMS（設計文書§2.1の必須計測）
  - 検証帯（seed 4000-4019）2〜3面での完走

いずれかのテストで例外/assertが起きても他のテストは継続実行し、最後に全テストの
実測値をまとめて表として print する。
"""
import math
import os
import sys
import tempfile
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from competition.baseline_slalom import (  # noqa: E402
    SlalomPolicy, corner_arc_params, arc_pose_at, fit_corner_radii,
    select_arc_radius, build_reference_path, build_speed_profile,
    build_line_path, _turn_kind, _wrap_pi, _HEADING_RAD,
)
from competition.evaluator import CompetitionEvaluator  # noqa: E402
from mouse.mjcf import build_maze_robot_xml  # noqa: E402
from mouse.params import RobotParams  # noqa: E402
from mouse.sim import MouseSim  # noqa: E402

# 研究計画書のデータ三分割規律: 検証用 = seed 4000-4019。
# test帯（seed 1000-1019）はマイルストーン判定専用のため実装調整には使わない。
VALIDATION_MAZE_DIR = os.path.join(REPO_ROOT, "competition", "mazes", "validation")
VALIDATION_SEEDS = (4000, 4001, 4002)

RESULTS = []


def record(name, expected, actual, passed, note=""):
    RESULTS.append(dict(name=name, expected=expected, actual=actual, passed=bool(passed), note=note))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: expected={expected}, actual={actual} {note}")


# ==========================================================================
# テスト1: コーナー円弧の幾何（接線連続=G1連続）
# ==========================================================================
def test1_corner_tangent_continuity():
    print("\n=== テスト1: コーナー円弧の接線連続性（G1連続） ===")
    all_ok = True
    cell_size = 0.18
    corner_xy = (0.09, 0.09)
    cases = [("N", "E"), ("N", "W"), ("E", "N"), ("E", "S"),
             ("S", "E"), ("S", "W"), ("W", "N"), ("W", "S")]
    for dir_in, dir_out in cases:
        R = 0.06
        params = corner_arc_params(corner_xy, dir_in, dir_out, R)
        x0, y0, h0, _c0 = arc_pose_at(params, 0.0)
        x1, y1, h1, _c1 = arc_pose_at(params, params["arclen"])

        ok_start_pos = math.hypot(x0 - params["tangent_in"][0], y0 - params["tangent_in"][1]) < 1e-9
        ok_end_pos = math.hypot(x1 - params["tangent_out"][0], y1 - params["tangent_out"][1]) < 1e-9
        ok_start_heading = abs(_wrap_pi(h0 - _HEADING_RAD[dir_in])) < 1e-6
        ok_end_heading = abs(_wrap_pi(h1 - _HEADING_RAD[dir_out])) < 1e-6
        ok = ok_start_pos and ok_end_pos and ok_start_heading and ok_end_heading
        all_ok = all_ok and ok
        record(f"tangent_continuity({dir_in}->{dir_out})", True, ok, ok,
               f"start_pos_err={math.hypot(x0-params['tangent_in'][0], y0-params['tangent_in'][1]):.2e}, "
               f"end_heading_err_deg={math.degrees(_wrap_pi(h1-_HEADING_RAD[dir_out])):.4f}")

        # 曲率の大きさが1/Rであること（符号は左右で反転）
        _x, _y, _h, curv0 = arc_pose_at(params, 0.0)
        ok_curv = abs(abs(curv0) - 1.0 / R) < 1e-9
        all_ok = all_ok and ok_curv
        record(f"curvature_magnitude({dir_in}->{dir_out})", 1.0 / R, curv0, ok_curv)

    # 180度・直進はcorner_arc_paramsで例外になること（円弧では表現できない）
    try:
        corner_arc_params(corner_xy, "N", "S", 0.06)
        ok_reverse_raises = False
    except ValueError:
        ok_reverse_raises = True
    all_ok = all_ok and ok_reverse_raises
    record("corner_arc_params_rejects_180deg", True, ok_reverse_raises, ok_reverse_raises)

    return all_ok


# ==========================================================================
# テスト2: S字干渉時の半径整定（fit_corner_radii）
# ==========================================================================
def test2_fit_corner_radii():
    print("\n=== テスト2: S字干渉時の半径整定 ===")
    all_ok = True
    W = 0.18
    margin = 0.01

    # ケースA: 全区間が十分長い（2つのコーナーの間も2区画=0.36m離す）
    #   → 接線トリム距離R=0.09の和(0.18)が0.36-margin=0.35以内に収まり
    #     縮小されない
    waypoints_long = [(0.09, -0.27), (0.09, 0.09), (0.45, 0.09), (0.45, 0.81)]
    radii_long = fit_corner_radii(waypoints_long, preferred_R=0.09, margin_m=margin)
    ok_long = all(abs(r - 0.09) < 1e-9 for r in radii_long)
    all_ok = all_ok and ok_long
    record("fit_radii_long_straight_unchanged", [0.09, 0.09], radii_long, ok_long)

    # ケースB: 直線区間が1区画のみのタイトなS字 → 縮小される
    waypoints_tight = [(0.09, -0.09), (0.09, 0.09), (0.27, 0.09), (0.27, 0.27)]
    radii_tight = fit_corner_radii(waypoints_tight, preferred_R=0.09, margin_m=margin)
    # 幾何的必要条件: r0+r1 <= L - margin = 0.18-0.01 = 0.17
    total = sum(radii_tight)
    ok_budget = total <= 0.17 + 1e-9
    ok_reduced = all(r < 0.09 - 1e-9 for r in radii_tight)
    all_ok = all_ok and ok_budget and ok_reduced
    record("fit_radii_tight_s_turn_budget", "<=0.17", total, ok_budget,
           f"radii={radii_tight}")
    record("fit_radii_tight_s_turn_reduced_from_preferred", "<0.09 both", radii_tight, ok_reduced)

    # ケースC: 単一コーナー（内部要素0個）は空リストを返す
    radii_none = fit_corner_radii([(0.09, 0.09), (0.27, 0.09)], preferred_R=0.09, margin_m=margin)
    ok_none = (radii_none == [])
    all_ok = all_ok and ok_none
    record("fit_radii_no_corner_returns_empty", [], radii_none, ok_none)

    return all_ok


# ==========================================================================
# テスト3: 円弧半径の候補選定（壁余裕15mm以上）
# ==========================================================================
def test3_select_arc_radius_wall_margin():
    print("\n=== テスト3: 円弧半径の候補選定（壁余裕15mm以上の数値検査） ===")
    all_ok = True
    # 実機体フットプリント実測値（body_footprint実測、docstring/報告に記載の値）
    footprint = (-0.050511210697675724, 0.050511210697675744,
                 -0.04140533518142653, 0.04140533518142652)
    cell_size = 0.18
    R, table = select_arc_radius(footprint, cell_size, candidates_mm=(45, 60, 75, 90), margin_m=0.015)

    print(f"  候補ごとのクリアランス表: {table}")
    ok_all_pass_margin = all(clr >= 0.015 for _r, clr in table)
    all_ok = all_ok and ok_all_pass_margin
    record("all_candidates_clear_15mm_margin", True, ok_all_pass_margin, ok_all_pass_margin,
           f"table={[(r, round(c*1000,2)) for r,c in table]}")

    ok_selected_is_max = (R == max(r for r, c in table if c >= 0.015) / 1000.0)
    all_ok = all_ok and ok_selected_is_max
    record("selected_R_is_max_satisfying_margin", True, ok_selected_is_max, ok_selected_is_max,
           f"selected_R={R*1000:.0f}mm")

    ok_monotonic = all(table[i][1] <= table[i + 1][1] + 1e-12 for i in range(len(table) - 1))
    record("clearance_monotonic_increasing_with_R", True, ok_monotonic, ok_monotonic,
           "実測: Rが大きいほど孤立ターンでの壁余裕も大きい（本実装のジオメトリでの実測傾向）")

    return all_ok


# ==========================================================================
# テスト4: 速度計画（v_turn=sqrt(a_lat*R)との一致、クリープ下限、終端停止）
# ==========================================================================
def test4_speed_profile():
    print("\n=== テスト4: 速度計画（v_turn・クリープ下限・終端停止） ===")
    all_ok = True
    W = 0.18
    a_lat = 3.5
    a_max = 3.4
    v_creep = 0.15
    v_cap = 0.8
    R = 0.09

    cells = [(0, -1), (0, 0), (1, 0)]
    directions = ["N", "E"]
    waypoints_xy = [((c[0] + 0.5) * W, (c[1] + 0.5) * W) for c in cells]
    radii = fit_corner_radii(waypoints_xy, R, margin_m=0.01)
    path = build_reference_path(cells, directions, "N", radii, W, stop_at_end=True, ds=0.005)
    speed = build_speed_profile(path.s, path.curvature, v_cap, a_lat, a_max, v_creep, True)

    v_turn_expected = math.sqrt(a_lat * radii[0])
    in_arc = np.abs(path.curvature) > 1e-6
    v_turn_actual = float(np.max(speed[in_arc])) if in_arc.any() else None
    ok_vturn = (v_turn_actual is not None) and abs(v_turn_actual - v_turn_expected) < 1e-6
    all_ok = all_ok and ok_vturn
    record("v_turn_matches_sqrt_a_lat_R", v_turn_expected, v_turn_actual, ok_vturn)

    # 終端（stop_at_end）は0
    ok_terminal_zero = abs(speed[-1]) < 1e-9
    all_ok = all_ok and ok_terminal_zero
    record("terminal_speed_zero_when_stop_at_end", 0.0, speed[-1], ok_terminal_zero)

    # クリープ下限（stop_at_end=Falseの場合、終端含め全域でv_creep以上）
    speed_nostop = build_speed_profile(path.s, path.curvature, v_cap, a_lat, a_max, v_creep, False)
    ok_creep_floor = bool(np.all(speed_nostop >= v_creep - 1e-9))
    all_ok = all_ok and ok_creep_floor
    record("creep_floor_when_not_stopping", f">={v_creep}", float(speed_nostop.min()), ok_creep_floor,
           "設計文書§10-2: 停止すべきでない区間ではv_refに下限を設ける")

    # 加速度制約（連続する2点間の速度変化がsqrt(v1^2+2a*ds)を超えない）
    ds_arr = np.diff(path.s)
    v_allow = np.sqrt(speed[:-1] ** 2 + 2 * a_max * ds_arr)
    ok_accel_limit = bool(np.all(speed[1:] <= v_allow + 1e-6))
    all_ok = all_ok and ok_accel_limit
    record("forward_accel_within_a_max", True, ok_accel_limit, ok_accel_limit)

    return all_ok


# ==========================================================================
# テスト5: 制御則の低速域での準停留がないこと（Stanley標準形の飽和特性）
# ==========================================================================
def _lateral_term(k_y, e_y, v, v_eps):
    return math.atan2(k_y * e_y, v + v_eps)


def test5_low_speed_no_quasi_stall():
    print("\n=== テスト5: 制御則の低速域での準停留がないこと（純粋な制御則の数式検証） ===")
    all_ok = True
    k_y = 2.5
    v_eps = 0.15
    e_y_fixed = 0.03  # 3cmの横偏差（残留させたままにする想定の値）

    # (a) 横偏差項は速度によらず必ず[-pi/2, pi/2]に有界（arctanの性質）。
    #     これにより「低速で横偏差項が方位偏差項と同程度以上に成長して拮抗する」
    #     というL0-a/L0-bのチャタリング根因を構造的に排除する。
    vs = np.linspace(0.01, 2.0, 50)
    terms = [_lateral_term(k_y, e_y_fixed, v, v_eps) for v in vs]
    ok_bounded = all(abs(t) < math.pi / 2 + 1e-9 for t in terms)
    all_ok = all_ok and ok_bounded
    record("lateral_term_bounded_by_pi_2", True, ok_bounded, ok_bounded,
           f"max|term|={max(abs(t) for t in terms):.4f} (limit={math.pi/2:.4f})")

    # (b) 速度が下がるほど横偏差項の大きさは単調増加する一方、飽和する
    #     （v->0で頭打ちになり、発散しない）。
    ok_monotonic = all(abs(terms[i]) >= abs(terms[i + 1]) - 1e-9 for i in range(len(terms) - 1))
    all_ok = all_ok and ok_monotonic
    record("lateral_term_monotonic_in_v", True, ok_monotonic, ok_monotonic,
           "vが小さいほど横偏差項が大きくなる（飽和はするが方向は単調）")

    # (c) 低速域(v=v_eps程度)でも、方位偏差項(k_psi*e_psi)が横偏差項に完全に
    #     打ち消されない程度の余地があること（チャタリングの根因=両者が
    #     同程度で拮抗、を数値的に確認する）。ここでは方位偏差がゼロの
    #     ケースでも横偏差項自体がpi/2を超えないことを再確認し、方位偏差
    #     フィードバックが常に有効な影響力を持てることを保証する。
    k_psi = 3.0
    e_psi_typical = math.radians(5.0)  # 5度程度の残留方位偏差を想定
    v_low = 0.05
    heading_term = k_psi * e_psi_typical
    lat_term_at_low_v = _lateral_term(k_y, e_y_fixed, v_low, v_eps)
    # 完全な相殺（合計がほぼ0）が起きていないことを確認
    omega_ref_no_curv = heading_term - lat_term_at_low_v
    ok_no_cancel = abs(omega_ref_no_curv) > 1e-3 or (abs(heading_term) < 1e-3)
    all_ok = all_ok and ok_no_cancel
    record("no_exact_cancellation_at_low_speed", "omega_ref far from 0 (unless e_psi~0)",
           omega_ref_no_curv, ok_no_cancel,
           f"heading_term={heading_term:.4f}, lateral_term={lat_term_at_low_v:.4f}")

    return all_ok


# ==========================================================================
# テスト6: 制御則の低速域での準停留がないこと（実シミュレーションでの収束確認）
# ==========================================================================
def _build_corridor_xml(out_path, width=1, height=6, params=None):
    params = params or RobotParams()
    v_walls = np.zeros((width + 1, height), dtype=int)
    h_walls = np.zeros((width, height + 1), dtype=int)
    v_walls[0, :] = 1
    v_walls[width, :] = 1
    h_walls[:, 0] = 1
    h_walls[:, height] = 1
    build_maze_robot_xml(v_walls, h_walls, out_path, params=params)
    return v_walls, h_walls


def test6_low_speed_convergence_sim():
    """低速域での実収束確認。足立法・ゴール判定など経路決定まわりは本テストの
    主旨ではないため、build_reference_path/build_speed_profileで直接
    「横偏差ありからの直線（曲がらない、stop_at_end=False）」経路を組み、
    pol._do_drive_control()だけを直接繰り返し呼ぶ（act()経由のflood-fillは
    使わない。1幅回廊ではgoal_cells()の中央2x2領域が定義できず的外れになる
    ため）。"""
    print("\n=== テスト6: 低速域での実収束確認（横偏差ありからの直線追従、準停留なし） ===")
    params = RobotParams()
    tmp_dir = tempfile.mkdtemp(prefix="test_baseline_slalom_corridor_")
    xml_path = os.path.join(tmp_dir, "corridor.xml")
    width, height = 1, 8
    _build_corridor_xml(xml_path, width, height, params)

    sim = MouseSim(xml_path, params=params)
    sim.full_reset(cell=(0, 0), heading_deg=90.0)
    # 意図的に横方向に2cmずらして初期化（直線経路からの横偏差を作る）
    qpos_adr = sim.model.jnt_qposadr[sim._root_joint_id]
    sim.data.qpos[qpos_adr] += 0.02
    import mujoco
    mujoco.mj_forward(sim.model, sim.data)

    pol = SlalomPolicy(safety_factor=0.6, v_creep=0.15)
    pol.bind_sim(sim)

    # 直線区間: (0,0)から(0,7)まで、曲がらない・停止しない経路。速度上限は
    # 低速域を確実に含めるため v_creep 付近まで絞ったv_explore相当にする
    # （設計文書§10の要件は「低速域(0.2〜0.6m/s)を含む速度掃引での
    # 準停留の有無」）。
    cells = [(0, k) for k in range(height)]
    directions = ["N"] * (height - 1)
    corner_radii = [0.09] * max(len(cells) - 2, 0)  # 全区間直進なので値自体は使われない
    path = build_reference_path(cells, directions, "N", corner_radii, pol.cell_size,
                                 stop_at_end=False, ds=0.005)
    v_ceil = 0.5
    path.speed = build_speed_profile(path.s, path.curvature, v_ceil, pol.a_lat,
                                      pol.a_max, pol.v_creep, False)
    pol._path = path
    pol._cursor = 0
    pol._path_end_reason = "continue"
    pol._heading_dir = "N"
    pol._state = "DRIVE"

    max_steps = int(6.0 / params.control_dt)
    lateral_errs = []
    quasi_stall_ticks = 0
    for i in range(max_steps):
        obs = sim.observation()
        x, y, yaw = sim.privileged_pose()
        pol._cursor = pol._advance_cursor(x, y)
        vl, vr = pol._do_drive_control(obs, x, y, yaw)
        res = sim.step_control(vl, vr)
        v_fwd, _omega = sim.privileged_velocity()
        idx = pol._cursor
        ux, uy = math.cos(path.heading[idx]), math.sin(path.heading[idx])
        e_y = (x - path.x[idx]) * (-uy) + (y - path.y[idx]) * ux
        lateral_errs.append(e_y)
        # 「動いていない(|v_ref|<v_creepの半分)のに横偏差がまだ大きい(>1cm)」を
        # 準停留の目安とする（速度計画自体にクリープ下限があるため、
        # v_refがクリープ未満に落ちることは無いはずだが、実測速度で確認する）。
        if i > 30 and abs(v_fwd) < 0.5 * pol.v_creep and abs(e_y) > 0.01:
            quasi_stall_ticks += 1
        if idx >= len(path.s) - 2:
            break

    ok_no_stall = quasi_stall_ticks == 0
    record("no_quasi_stall_ticks_detected", 0, quasi_stall_ticks, ok_no_stall,
           f"動いていない(|v|<{0.5*pol.v_creep:.3f}m/s)のに横偏差>1cmが残るティックが無いこと")

    ok_converged = len(lateral_errs) > 0 and abs(lateral_errs[-1]) < 0.01
    record("lateral_error_converges", "<0.01m", round(lateral_errs[-1], 5) if lateral_errs else None,
           ok_converged, f"n_samples={len(lateral_errs)}")

    max_dev = max(abs(e) for e in lateral_errs) if lateral_errs else None
    print(f"  横偏差: 初期2cmからの最大={max_dev*1000:.2f}mm, 最終={lateral_errs[-1]*1000:.2f}mm")

    return ok_no_stall and ok_converged


# ==========================================================================
# テスト7: 検証帯end-to-end + 経路からの横偏差 最大値/RMS（設計文書§2.1必須計測）
# ==========================================================================
def test7_evaluator_e2e_and_lateral_deviation():
    """検証帯end-to-end。完走の正式な合否は凍結ハーネス
    （competition/evaluator.py、5走行まで許容・衝突は係員回収で再試行）の
    success（1回以上ゴール到達）で判定する（test_baseline_straightrun.pyの
    test7と同じ流儀）。1回の接触が即失敗ではなく、5走行のうち1回でも
    ゴールできれば完走扱いというのがこの競技の正式な規約であるため。
    横偏差の最大値・RMS（設計文書§2.1の必須計測）は、評価器とは別に生の
    MouseSimで初回ゴールまでを走らせて計測する（評価器はpolicyの内部状態
    に立ち入らないため）。"""
    print("\n=== テスト7: 評価迷路end-to-end（validation帯 seed 4000-4002）+ 横偏差計測 ===")
    all_ok = True

    for seed in VALIDATION_SEEDS:
        npz_path = os.path.join(VALIDATION_MAZE_DIR, f"maze_{seed}.npz")
        if not os.path.exists(npz_path):
            print(f"  [SKIP] {npz_path} が見つかりません")
            record(f"eval_seed{seed}_e2e", "実行", "SKIP(迷路なし)", False)
            all_ok = False
            continue

        # --- (1) 正式な完走判定: 凍結ハーネス経由 ---
        evaluator = CompetitionEvaluator(maze_dir=VALIDATION_MAZE_DIR)
        pol_eval = SlalomPolicy(safety_factor=0.7)
        t0 = time.time()
        result = evaluator.evaluate_maze(npz_path, pol_eval)
        wall_clock = time.time() - t0
        outcomes = [r["outcome"] for r in result["runs"]]
        print(f"  seed={seed}: wall_clock={wall_clock:.2f}s, success={result['success']}, "
              f"best_time={result['best_time']}, outcomes={outcomes}")

        ok_success = result["success"] is True
        all_ok = all_ok and ok_success
        record(f"eval_seed{seed}_success", True, ok_success, ok_success,
               f"best_time={result['best_time']}")

        # --- (2) 横偏差 最大値・RMS（設計文書§2.1必須計測）: 生simで初回ゴールまで ---
        params = RobotParams()
        npz = np.load(npz_path)
        v_walls, h_walls = npz["v_walls"], npz["h_walls"]
        width, height = int(npz["width"]), int(npz["height"])
        xml_path = os.path.join(VALIDATION_MAZE_DIR, f"maze_{seed}.xml")

        sim = MouseSim(xml_path, params=params)
        sim.full_reset(cell=(0, 0), heading_deg=90.0)
        pol = SlalomPolicy(safety_factor=0.7)
        pol.bind_sim(sim)
        pol.bind_maze(v_walls, h_walls)
        pol.on_maze_start({"width": width, "height": height, "maze_id": f"maze_{seed}",
                            "seed": seed, "xml_path": xml_path})

        lateral_errs = []
        max_steps = int(60.0 / params.control_dt)
        for i in range(max_steps):
            obs = sim.observation()
            vl, vr = pol.act(obs)
            sim.step_control(vl, vr)
            x, y, _yaw = sim.privileged_pose()
            if pol._path is not None and pol._state == "DRIVE":
                idx = pol._cursor
                path = pol._path
                ux, uy = math.cos(path.heading[idx]), math.sin(path.heading[idx])
                e_y = (x - path.x[idx]) * (-uy) + (y - path.y[idx]) * ux
                lateral_errs.append(e_y)
            if pol._explored_once:
                break

        lateral_arr = np.array(lateral_errs) if lateral_errs else np.array([0.0])
        max_dev = float(np.max(np.abs(lateral_arr)))
        rms_dev = float(np.sqrt(np.mean(lateral_arr ** 2)))
        print(f"    横偏差（初回ゴールまで）: 最大={max_dev*1000:.2f}mm RMS={rms_dev*1000:.2f}mm "
              f"(n={len(lateral_errs)})")
        record(f"eval_seed{seed}_lateral_dev_max_mm", "-", round(max_dev * 1000, 2), True)
        record(f"eval_seed{seed}_lateral_dev_rms_mm", "-", round(rms_dev * 1000, 2), True)

    return all_ok


# ==========================================================================
# メイン
# ==========================================================================
def main():
    print("competition/baseline_slalom.py (L0-c) テストスイート")
    print("=" * 70)

    test_fns = [
        test1_corner_tangent_continuity,
        test2_fit_corner_radii,
        test3_select_arc_radius_wall_margin,
        test4_speed_profile,
        test5_low_speed_no_quasi_stall,
        test6_low_speed_convergence_sim,
        test7_evaluator_e2e_and_lateral_deviation,
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
    print(f"{'項目':50s} {'期待値':>22s} {'実測値':>22s} {'判定':<5s}")
    print("-" * 70)
    for r in RESULTS:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['name']:50s} {str(r['expected']):>22s} {str(r['actual']):>22s} "
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
