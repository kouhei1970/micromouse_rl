"""
tests/test_baseline.py
================
competition/baseline_classical.py（AdachiPolicy, M0-T3）の単体・end-to-end テスト。

pytest は使わない plain Python スクリプト（tests/test_evaluator.py, tests/test_mouse_v2.py
と同じ流儀）。実行方法（リポジトリルートで）:
    .venv/bin/python tests/test_baseline.py

spec_baseline.md §3 で指定された 4 項目 + 追加の健全性チェック:
  1. flood-fill 単体: 手書き 4x4 迷路（ループ・行き止まりを含む）で距離場の期待値一致。
     未知壁 (-1) を楽観的に「壁なし」とみなす規則の単体確認も含む。
  2. タイブレーク単体: 直進優先→前回進行方向から時計回り優先の決定的規則を確認。
  3. TURN 単体: 平原 XML（assets/mouse_v2.xml）で 90°/180° 旋回 → 誤差 < 2°、所要時間を print。
  4. FORWARD 単体: 廊下 XML（内壁なしの細長迷路）で 1 セル直進 → 停止位置誤差 < 0.01 m、
     所要時間を print。
  5. 評価迷路 end-to-end（seed 1000 と 1001 の 2 面。指揮側から指定された範囲）: evaluator
     を time_budget=300 のまま実行 → 完走すること、走行タイム・訪問セル数を print。

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

from competition.baseline_classical import AdachiPolicy, _HEADING_RAD, _wrap_pi  # noqa: E402
from competition.evaluator import CompetitionEvaluator  # noqa: E402
from mouse.mjcf import build_maze_robot_xml  # noqa: E402
from mouse.params import RobotParams  # noqa: E402
from mouse.sim import MouseSim  # noqa: E402

EVAL_MAZE_DIR = os.path.join(REPO_ROOT, "competition", "mazes", "eval")
OPEN_FLOOR_XML = os.path.join(REPO_ROOT, "assets", "mouse_v2.xml")

RESULTS = []  # list[dict(name, expected, actual, passed, note)]


def record(name, expected, actual, passed, note=""):
    RESULTS.append(dict(name=name, expected=expected, actual=actual, passed=bool(passed), note=note))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: expected={expected}, actual={actual} {note}")


# ==========================================================================
# テスト1: flood-fill 単体（手書き4x4、ループ+行き止まりを含む）
# ==========================================================================
def _build_hand_maze_4x4():
    """4x4 の手書き迷路（既知壁のみ、-1 なし）を返す。
    ゴール4セル(1,1),(1,2),(2,1),(2,2) は内壁を全開放（ループを形成）。
    (0,1)-(0,2) 間を壁で塞ぎ迂回を強制。(2,1)-(3,1) 間を壁で塞ぎ行き止まりを作る。
    手計算した期待距離（本ファイル冒頭のコメント/報告書参照）:
      距離0: (1,1)(1,2)(2,1)(2,2)
      距離1: (0,1)(1,0)(0,2)(1,3)(2,0)(3,2)(2,3)
      距離2: (0,0)(0,3)(3,0)(3,3)(3,1)
    """
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
    print("\n=== テスト1: flood-fill単体（手書き4x4、ループ+行き止まり） ===")
    v_walls, h_walls = _build_hand_maze_4x4()

    pol = AdachiPolicy()
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


# ==========================================================================
# テスト1b: 未知壁(-1)の楽観的扱い単体
# ==========================================================================
def test1b_floodfill_unknown_optimistic():
    print("\n=== テスト1b: 未知壁(-1)の楽観的扱い単体（1x3廊下） ===")
    pol = AdachiPolicy()
    pol.width, pol.height = 3, 1

    all_ok = True

    # ケースA: v[1,0]=-1(未知), v[2,0]=0(既知・開放) → 楽観的に(0,0)も距離2で到達可能
    pol.v_walls_known = np.array([[1], [-1], [0], [1]], dtype=int).reshape(4, 1)
    pol.h_walls_known = np.array([[1, 1], [1, 1], [1, 1]], dtype=int)
    dist_a = pol._flood_fill([(2, 0)])
    ok_a = (dist_a.get((0, 0)) == 2) and (dist_a.get((1, 0)) == 1)
    all_ok = all_ok and ok_a
    record("floodfill_unknown_treated_as_open", {(0, 0): 2, (1, 0): 1},
           {(0, 0): dist_a.get((0, 0)), (1, 0): dist_a.get((1, 0))}, ok_a,
           "v[1,0]=-1(未知)を壁なしとみなし(0,0)も到達可能になること")

    # ケースB: v[1,0]=1(既知・壁あり) → (0,0)は到達不能（h方向の迂回路も無い1x3廊下のため）
    pol.v_walls_known = np.array([[1], [1], [0], [1]], dtype=int).reshape(4, 1)
    dist_b = pol._flood_fill([(2, 0)])
    ok_b = ((0, 0) not in dist_b) and (dist_b.get((1, 0)) == 1)
    all_ok = all_ok and ok_b
    record("floodfill_known_wall_blocks", "((0,0) not reachable)",
           f"(0,0) in dist={(0, 0) in dist_b}", ok_b,
           "v[1,0]=1(既知の壁)は塞ぎ(0,0)は到達不能のままであること（対照実験）")

    return all_ok


# ==========================================================================
# テスト2: タイブレーク単体（直進優先→時計回り優先）
# ==========================================================================
def test2_tiebreak_order():
    print("\n=== テスト2: タイブレーク単体（直進優先→時計回り優先） ===")
    all_ok = True

    cases = [
        ("N", ["N", "E", "S", "W"]),
        ("E", ["E", "S", "W", "N"]),
        ("S", ["S", "W", "N", "E"]),
        ("W", ["W", "N", "E", "S"]),
    ]
    for prev_dir, expected_order in cases:
        actual_order = AdachiPolicy._tiebreak_order(prev_dir)
        ok = (actual_order == expected_order)
        all_ok = all_ok and ok
        record(f"tiebreak_order(prev={prev_dir})", expected_order, actual_order, ok)

    # 実際に _do_plan で、複数候補が同距離の場合に選ばれる方向を確認する。
    # 内壁の無い4x4（スタート区画規定を適用しない生の全開放マップ）なら、
    # (0,0)からは東(1,0)・北(0,1)がどちらも flood-fill 距離1で同着になる
    # （ゴール4セル(1,1)(1,2)(2,1)(2,2)からのBFS。テスト1の全開放パターンで
    # 実測済みの距離場と同じ）。heading_dir="N"(直進候補)なら北を、
    # heading_dir="E"(直進候補)なら東を、直進優先規則で選ぶはず。
    for heading_dir, expected_dir in [("N", "N"), ("E", "E")]:
        pol2 = AdachiPolicy()
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
# テスト3: TURN単体（平原XML、90度/180度旋回）
# ==========================================================================
def test3_turn():
    print("\n=== テスト3: TURN単体（平原XML、90度/180度旋回） ===")
    if not os.path.exists(OPEN_FLOOR_XML):
        print(f"  [SKIP] {OPEN_FLOOR_XML} が見つかりません")
        record("turn_e2e", "実行", "SKIP(平原XMLなし)", False)
        return False

    params = RobotParams()
    all_ok = True

    for target_dir, label in [("E", "90度(北->東)"), ("S", "90度(北->南)"), ("W", "180度(北->西)")]:
        sim = MouseSim(OPEN_FLOOR_XML, params=params)
        sim.full_reset(cell=(0, 0), heading_deg=90.0)
        pol = AdachiPolicy()
        pol.bind_sim(sim)
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
# テスト4: FORWARD単体（廊下XML、1セル直進）
# ==========================================================================
def _build_corridor_xml(out_path, width=1, height=5, params=None):
    """内壁の無い width x height の廊下 XML を生成する。"""
    params = params or RobotParams()
    v_walls = np.zeros((width + 1, height), dtype=int)
    h_walls = np.zeros((width, height + 1), dtype=int)
    v_walls[0, :] = 1
    v_walls[width, :] = 1
    h_walls[:, 0] = 1
    h_walls[:, height] = 1
    build_maze_robot_xml(v_walls, h_walls, out_path, params=params)
    return v_walls, h_walls


def test4_forward():
    print("\n=== テスト4: FORWARD単体（廊下XML、1セル直進） ===")
    params = RobotParams()
    tmp_dir = tempfile.mkdtemp(prefix="test_baseline_corridor_")
    xml_path = os.path.join(tmp_dir, "corridor.xml")
    width, height = 1, 5
    v_walls, h_walls = _build_corridor_xml(xml_path, width, height, params)

    sim = MouseSim(xml_path, params=params)
    sim.full_reset(cell=(0, 0), heading_deg=90.0)
    pol = AdachiPolicy()
    pol.bind_sim(sim)
    pol.on_maze_start({"width": width, "height": height})
    pol.bind_maze(v_walls, h_walls)
    pol._heading_dir = "N"
    pol._planned_next_cell = (0, 1)
    x0, y0, _yaw0 = sim.privileged_pose()
    pol._enter_forward(x0, y0)

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
    record("forward_converged", True, ok_converge, ok_converge, f"t_done={t_done}")
    record("forward_stop_pos_err_m", "<0.01", f"{err_m:.5f}", ok_err,
           f"所要時間={t_done:.3f}s" if t_done else "")

    return ok_converge and ok_err


# ==========================================================================
# テスト5: 評価迷路 end-to-end（seed 1000, 1001）
# ==========================================================================
def test5_evaluator_e2e():
    print("\n=== テスト5: 評価迷路end-to-end（seed 1000, 1001） ===")
    all_ok = True

    for seed in (1000, 1001):
        npz_path = os.path.join(EVAL_MAZE_DIR, f"maze_{seed}.npz")
        if not os.path.exists(npz_path):
            print(f"  [SKIP] {npz_path} が見つかりません")
            record(f"eval_seed{seed}_e2e", "実行", "SKIP(迷路なし)", False)
            all_ok = False
            continue

        evaluator = CompetitionEvaluator(time_budget=300.0, max_runs=5)
        pol = AdachiPolicy()
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
    print("competition/baseline_classical.py (M0-T3) テストスイート")
    print("=" * 70)

    test_fns = [
        test1_floodfill_hand_maze,
        test1b_floodfill_unknown_optimistic,
        test2_tiebreak_order,
        test3_turn,
        test4_forward,
        test5_evaluator_e2e,
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
