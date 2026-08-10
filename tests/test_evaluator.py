"""
tests/test_evaluator.py
================
competition/evaluator.py（CompetitionEvaluator, M0-T1b）の end-to-end / 単体テスト。

pytest は使わない plain Python スクリプト（tests/test_mouse_v2.py と同じ流儀）。
実行方法（リポジトリルートで）:
    .venv/bin/python tests/test_evaluator.py

spec_evaluator.md §4 で指定された 4 項目 + 「ゴール到達方策」による追加の
end-to-end 確認（計 5 項目）:
  1. 静止方策（常に 0V）: 走行数 0、DNF、持ち時間満了で終了。incidents 記録。
  2. 直進突撃方策（常に +1.5V,+1.5V）: 5 走行を消化して評価終了。runs 配列の整合性
     （t_start<t_end、outcome=collision、走行番号の連番）を確認。
  3. セル判定単体: スタート領域/ゴール領域の境界値テスト。
  4. スタック検出単体: 位置履歴リングバッファのロジックをオフラインで検証。
  5. （追加）ゴール到達方策: 特権情報（bind_sim/bind_maze）を使う自作の
     ウェイポイント追従方策で、自作の小さな迷路（2x6, 直進のみで到達可能）
     をゴールまで走らせ、outcome="goal" → FREE 復帰 → best_time/success の
     計算までを一気通貫で確認する。1・2 は「失敗系」のみを検証するため、
     spec の 4 項目だけでは "goal" 分岐（RUN_ACTIVE 正常終了・経路効率計算等）
     が一度もテストされない穴を埋める。

いずれかのテストで例外/assert が起きても他のテストは継続実行し、
最後に全テストの結果表と実測値を print する。
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

from competition.evaluator import (  # noqa: E402
    CompetitionEvaluator,
    PositionRingBuffer,
    bfs_shortest_path,
    goal_cells,
    in_goal_region,
    in_start_region,
)
from competition.policy_interface import MousePolicy  # noqa: E402

EVAL_MAZE_DIR = os.path.join(REPO_ROOT, "competition", "mazes", "eval")
EVAL_MAZE_SAMPLE = os.path.join(EVAL_MAZE_DIR, "maze_1000.npz")

RESULTS = []  # list[dict(name, expected, actual, passed, note)]


def record(name, expected, actual, passed, note=""):
    RESULTS.append(dict(name=name, expected=expected, actual=actual, passed=bool(passed), note=note))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: expected={expected}, actual={actual} {note}")


# ==========================================================================
# ダミー方策
# ==========================================================================
class StaticPolicy(MousePolicy):
    """常に 0V（静止）。スタート区画から一度も出発しない。"""
    name = "static_zero_volt"

    def act(self, obs):
        return 0.0, 0.0


class DashPolicy(MousePolicy):
    """常に +1.5V, +1.5V（直進突撃）。壁に向かって走り続け、必ず衝突する。"""
    name = "dash_forward"

    def act(self, obs):
        return 1.5, 1.5


class WaypointCheatPolicy(MousePolicy):
    """特権情報（bind_sim/bind_maze）を使い、真の壁情報から BFS 最短経路を求め、
    そのセル中心をウェイポイントとして辿る「ズル」方策。

    テスト専用実装であり、production 用の古典ベースラインではない。
    ゲイン（kp_turn/turn_limit）は意図的に控えめに抑えてあり、方向転換を
    ほぼ必要としない直線的な迷路でのみ安定動作する（本テストの迷路は
    スタートの初期向き(北)のままゴールへ直進できる形に作ってある）。
    大きな方位誤差からの復帰は保証しない — この方策の目的はあくまで
    CompetitionEvaluator の「ゴール到達」分岐を一気通貫で駆動することにある。
    """
    name = "waypoint_cheat"
    requires_privileged = True

    def __init__(self, kp_turn=0.5, turn_limit=0.4, v_base=1.2, arrive_eps=0.04):
        self.kp_turn = kp_turn
        self.turn_limit = turn_limit
        self.v_base = v_base
        self.arrive_eps = arrive_eps
        self._sim = None
        self._v_walls = None
        self._h_walls = None
        self._waypoints = []
        self._wp_idx = 0

    def bind_sim(self, sim):
        self._sim = sim

    def bind_maze(self, v_walls, h_walls):
        self._v_walls = v_walls
        self._h_walls = h_walls

    def on_maze_start(self, maze_info):
        width, height = maze_info["width"], maze_info["height"]
        targets = goal_cells(width, height)
        path = bfs_shortest_path(self._v_walls, self._h_walls, width, height, (0, 0), set(targets))
        cell_size = 0.18
        self._waypoints = [
            (cx * cell_size + cell_size / 2, cy * cell_size + cell_size / 2)
            for (cx, cy) in path[1:]  # 出発セル(0,0)自身は除く
        ]
        self._wp_idx = 0

    def on_run_start(self, run_index):
        self._wp_idx = 0

    def on_run_end(self, outcome):
        pass

    def on_retrieval(self):
        self._wp_idx = 0

    def act(self, obs):
        if self._sim is None or self._wp_idx >= len(self._waypoints):
            return 0.0, 0.0
        x, y, yaw = self._sim.privileged_pose()
        tx, ty = self._waypoints[self._wp_idx]
        dx, dy = tx - x, ty - y
        if math.hypot(dx, dy) < self.arrive_eps:
            self._wp_idx += 1
            if self._wp_idx >= len(self._waypoints):
                return 0.0, 0.0
            tx, ty = self._waypoints[self._wp_idx]
            dx, dy = tx - x, ty - y
        desired = math.atan2(dy, dx)
        err = math.atan2(math.sin(desired - yaw), math.cos(desired - yaw))
        turn = max(-self.turn_limit, min(self.turn_limit, self.kp_turn * err))
        vl = max(-3.0, min(3.0, self.v_base + turn))
        vr = max(-3.0, min(3.0, self.v_base - turn))
        return vl, vr


def _make_open_room_maze_npz(width, height, seed, out_dir):
    """内部壁が一切無い（外周のみ）W×H の迷路 npz を tmp ディレクトリに保存する。
    テスト用の簡易迷路生成（spec_evaluator.md §4 が許容する自作の迷路配列）。"""
    v_walls = np.zeros((width + 1, height), dtype=int)
    h_walls = np.zeros((width, height + 1), dtype=int)
    v_walls[0, :] = 1
    v_walls[width, :] = 1
    h_walls[:, 0] = 1
    h_walls[:, height] = 1
    npz_path = Path(out_dir) / f"maze_{seed}.npz"
    np.savez(npz_path, v_walls=v_walls, h_walls=h_walls, seed=seed, width=width, height=height)
    return npz_path


# ==========================================================================
# テスト1: 静止方策
# ==========================================================================
def test1_static_policy():
    print("\n=== テスト1: 静止方策（常に0V） ===")
    if not os.path.exists(EVAL_MAZE_SAMPLE):
        print(f"  [SKIP] 評価迷路 {EVAL_MAZE_SAMPLE} が見つかりません")
        record("static_policy_e2e", "実行", "SKIP(迷路なし)", False)
        return False

    evaluator = CompetitionEvaluator(time_budget=30.0, max_runs=5)
    t0 = time.time()
    result = evaluator.evaluate_maze(EVAL_MAZE_SAMPLE, StaticPolicy())
    wall_clock = time.time() - t0

    n_runs = len(result["runs"])
    n_incidents = len(result["incidents"])
    print(f"  wall_clock={wall_clock:.3f}s, n_runs={n_runs}, n_incidents={n_incidents}, "
          f"best_time={result['best_time']}, success={result['success']}")
    for inc in result["incidents"]:
        print(f"    incident: t={inc['t']:.2f} kind={inc['kind']} pos={inc['pos']}")

    ok_runs = (n_runs == 0)
    ok_dnf = (result["best_time"] is None and result["success"] is False)
    ok_incidents = (n_incidents >= 1) and all(i["kind"] == "stuck" for i in result["incidents"])

    record("static_n_runs", 0, n_runs, ok_runs, "走行を一度も消費しないこと")
    record("static_dnf", "best_time=None, success=False", f"{result['best_time']}, {result['success']}", ok_dnf)
    record("static_incidents_recorded", ">=1件のstuck incident", f"{n_incidents}件: "
           f"{[i['kind'] for i in result['incidents']]}", ok_incidents,
           "20秒静止 → スタック回収がincidentとして記録されること")

    return ok_runs and ok_dnf and ok_incidents


# ==========================================================================
# テスト2: 直進突撃方策
# ==========================================================================
def test2_dash_policy():
    print("\n=== テスト2: 直進突撃方策（常に+1.5V,+1.5V） ===")
    if not os.path.exists(EVAL_MAZE_SAMPLE):
        print(f"  [SKIP] 評価迷路 {EVAL_MAZE_SAMPLE} が見つかりません")
        record("dash_policy_e2e", "実行", "SKIP(迷路なし)", False)
        return False

    evaluator = CompetitionEvaluator(time_budget=30.0, max_runs=5)
    t0 = time.time()
    result = evaluator.evaluate_maze(EVAL_MAZE_SAMPLE, DashPolicy())
    wall_clock = time.time() - t0

    runs = result["runs"]
    print(f"  wall_clock={wall_clock:.3f}s, n_runs={len(runs)}")
    for r in runs:
        print(f"    run#{r['index']}: t_start={r['t_start']:.3f} t_end={r['t_end']:.3f} "
              f"run_time={r['run_time']:.3f} outcome={r['outcome']} "
              f"collision_pos={r['collision_pos']}")

    ok_count = (len(runs) == 5)
    ok_outcome = all(r["outcome"] == "collision" for r in runs)
    ok_order = all(r["t_start"] < r["t_end"] for r in runs)
    ok_index = [r["index"] for r in runs] == list(range(1, len(runs) + 1))
    ok_monotonic_t = all(runs[i]["t_end"] <= runs[i + 1]["t_start"] for i in range(len(runs) - 1))
    ok_collision_pos = all(r["collision_pos"] is not None for r in runs)

    record("dash_run_count", 5, len(runs), ok_count, "max_runsまで走行を消化して評価終了")
    record("dash_all_collision", True, ok_outcome, ok_outcome, "全走行がoutcome=collisionで失敗")
    record("dash_t_start_lt_t_end", True, ok_order, ok_order, "各走行でt_start<t_end")
    record("dash_run_index_sequence", "1,2,3,4,5", str([r['index'] for r in runs]), ok_index)
    record("dash_t_monotonic_across_runs", True, ok_monotonic_t, ok_monotonic_t,
           "走行間でtが単調増加（係員回収を挟んで時間は逆行しない）")
    record("dash_collision_pos_recorded", True, ok_collision_pos, ok_collision_pos)

    return ok_count and ok_outcome and ok_order and ok_index and ok_monotonic_t and ok_collision_pos


# ==========================================================================
# テスト3: セル判定単体（境界値）
# ==========================================================================
def test3_region_boundaries():
    print("\n=== テスト3: セル判定単体（境界値） ===")
    cell_size = 0.18
    all_ok = True

    start_cases = [
        ((0.0, 0.0), True, "原点(下限含む)"),
        ((0.09, 0.09), True, "区画中心"),
        ((cell_size - 1e-6, 0.09), True, "上限直前(x)"),
        ((0.09, cell_size - 1e-6), True, "上限直前(y)"),
        ((cell_size, 0.09), False, "上限ちょうど(x)は区画外(半開区間)"),
        ((0.09, cell_size), False, "上限ちょうど(y)は区画外(半開区間)"),
        ((-1e-6, 0.09), False, "下限未満(x)"),
        ((0.09, -1e-6), False, "下限未満(y)"),
    ]
    for (x, y), expected, note in start_cases:
        actual = in_start_region(x, y, cell_size)
        ok = (actual == expected)
        all_ok = all_ok and ok
        record(f"in_start_region({x:.6f},{y:.6f})", expected, actual, ok, note)

    # 16x16 迷路のゴール領域: [1.26,1.62)x[1.26,1.62)（spec_evaluator.md §1 記載値と一致）
    goal_cases = [
        ((1.26, 1.26), True, "下限ちょうど(両軸)は区画内"),
        ((1.44, 1.44), True, "中央柱付近(中心)"),
        ((1.62 - 1e-6, 1.62 - 1e-6), True, "上限直前(両軸)"),
        ((1.62, 1.44), False, "上限ちょうど(x)は区画外"),
        ((1.44, 1.62), False, "上限ちょうど(y)は区画外"),
        ((1.26 - 1e-6, 1.44), False, "下限未満(x)"),
        ((1.44, 1.26 - 1e-6), False, "下限未満(y)"),
        ((0.09, 0.09), False, "スタート区画はゴールでない"),
    ]
    for (x, y), expected, note in goal_cases:
        actual = in_goal_region(x, y, 16, 16, cell_size)
        ok = (actual == expected)
        all_ok = all_ok and ok
        record(f"in_goal_region({x:.6f},{y:.6f})", expected, actual, ok, note)

    # 数値検算: goal_region_bounds が spec の値 (1.26, 1.62) と一致することも確認
    from competition.evaluator import goal_region_bounds
    bounds = goal_region_bounds(16, 16, cell_size)
    expected_bounds = (1.26, 1.62, 1.26, 1.62)
    ok_bounds = all(abs(a - b) < 1e-9 for a, b in zip(bounds, expected_bounds))
    all_ok = all_ok and ok_bounds
    record("goal_region_bounds(16,16)", expected_bounds, bounds, ok_bounds,
           "spec_evaluator.md §1 の [1.26,1.62)x[1.26,1.62) と一致")

    return all_ok


# ==========================================================================
# テスト4: スタック検出単体（位置履歴リングバッファ）
# ==========================================================================
def test4_stuck_detection():
    print("\n=== テスト4: スタック検出単体（位置履歴リングバッファ） ===")
    all_ok = True
    dt = 0.01  # 100Hz

    def sample_times(n):
        """インデックスから時刻列を生成（浮動小数点の累積誤差を避けるため
        t = round(i*dt, 6) で毎回インデックスから計算し、ループでの
        逐次加算による丸め誤差の蓄積を避ける）。"""
        return [round(i * dt, 6) for i in range(n)]

    # (a) 走行開始から20秒未満は判定しない（静止していてもFalse）
    # 20秒 = 2000サンプル分。19.99s時点(1999サンプル目、インデックス0-1998)ではまだ20秒未満。
    ring = PositionRingBuffer(window_s=20.0)
    times_a = sample_times(1999)  # 0.00 〜 19.98s
    for t in times_a:
        ring.push(t, 0.5, 0.5)
    t_check_a = times_a[-1]
    is_stuck, disp = ring.check_stuck(t_check_a, 0.5, 0.5, segment_start_t=0.0)
    ok_a = (is_stuck is False)
    all_ok = all_ok and ok_a
    record("stuck_gate_before_20s", False, is_stuck, ok_a, f"t={t_check_a:.2f}s (<20s), 静止していても非発火")

    # (b) ちょうど20秒経過・静止 → 発火（変位 < 5cm）
    ring2 = PositionRingBuffer(window_s=20.0)
    times_b = sample_times(2001)  # 0.00 〜 20.00s
    for t in times_b:
        ring2.push(t, 0.5, 0.5)
    t_check_b = times_b[-1]  # 20.00
    is_stuck2, disp2 = ring2.check_stuck(t_check_b, 0.5, 0.5, segment_start_t=0.0)
    ok_b = (is_stuck2 is True) and (disp2 is not None) and (disp2 < 0.05)
    all_ok = all_ok and ok_b
    record("stuck_fires_at_20s_stationary", True, is_stuck2, ok_b,
           f"t={t_check_b:.2f}s, disp={disp2}")

    # (c) 20秒経過・大きく移動（変位 >= 5cm）→ 非発火
    ring3 = PositionRingBuffer(window_s=20.0)
    times_c = sample_times(2001)
    for t in times_c:
        # 20秒かけて (0,0) から (0.10, 0) へ直線移動（変位0.10m > 5cm）
        frac = min(t / 20.0, 1.0)
        ring3.push(t, 0.10 * frac, 0.0)
    t_check_c = times_c[-1]
    is_stuck3, disp3 = ring3.check_stuck(t_check_c, 0.10, 0.0, segment_start_t=0.0)
    ok_c = (is_stuck3 is False) and (disp3 is not None) and (disp3 >= 0.05)
    all_ok = all_ok and ok_c
    record("stuck_not_fire_when_moving", False, is_stuck3, ok_c, f"disp={disp3} (>=0.05m)")

    # (d) 区間基準時刻(segment_start_t)のリセット: 係員回収直後を模して
    #     segment_start_t を t=10.0 にリセットすると、その後10秒(合計20秒未満)は非発火
    ring4 = PositionRingBuffer(window_s=20.0)
    times_d = sample_times(2001)  # 0..20s、途中静止のまま
    for t in times_d:
        ring4.push(t, 0.5, 0.5)
    t_check_d = times_d[-1]
    is_stuck4, _ = ring4.check_stuck(t_check_d, 0.5, 0.5, segment_start_t=10.0)  # 区間は t=10 開始 → 経過10秒
    ok_d = (is_stuck4 is False)
    all_ok = all_ok and ok_d
    record("stuck_gate_resets_with_segment_start_t", False, is_stuck4, ok_d,
           f"segment_start_t=10.0, t={t_check_d:.2f}s → 区間経過はまだ10秒")

    # (e) 境界値: ちょうど5cm(閾値)は「発火しない」(< 判定、以上は非スタック)
    ring5 = PositionRingBuffer(window_s=20.0)
    times_e = sample_times(2001)
    for t in times_e:
        frac = min(t / 20.0, 1.0)
        ring5.push(t, 0.05 * frac, 0.0)  # 20秒かけてちょうど0.05m移動
    t_check_e = times_e[-1]
    is_stuck5, disp5 = ring5.check_stuck(t_check_e, 0.05, 0.0, segment_start_t=0.0)
    ok_e = (is_stuck5 is False)
    all_ok = all_ok and ok_e
    record("stuck_threshold_boundary_5cm", False, is_stuck5, ok_e,
           f"disp={disp5} (ちょうど閾値5cmは非スタック: disp<0.05のみ発火)")

    return all_ok


# ==========================================================================
# テスト5（追加）: ゴール到達方策の end-to-end
# ==========================================================================
def test5_goal_reaching_policy():
    print("\n=== テスト5（追加）: ゴール到達方策（自作2x6迷路, 直進のみ） ===")
    tmp_dir = tempfile.mkdtemp(prefix="test_evaluator_goal_")
    npz_path = _make_open_room_maze_npz(width=2, height=6, seed=9997, out_dir=tmp_dir)

    evaluator = CompetitionEvaluator(time_budget=15.0, max_runs=5)
    t0 = time.time()
    result = evaluator.evaluate_maze(npz_path, WaypointCheatPolicy())
    wall_clock = time.time() - t0

    runs = result["runs"]
    print(f"  wall_clock={wall_clock:.3f}s, n_runs={len(runs)}")
    for r in runs:
        print(f"    run#{r['index']}: outcome={r['outcome']} run_time={r['run_time']:.3f} "
              f"path_length_m={r['path_length_m']:.4f} max_progress_cells={r['max_progress_cells']} "
              f"path_efficiency={r['path_efficiency']}")

    ok_reached = (len(runs) >= 1) and (runs[0]["outcome"] == "goal")
    ok_best_time = (result["best_time"] is not None) and (result["best_time"] > 0)
    ok_success = result["success"] is True
    ok_progress = ok_reached and (runs[0]["max_progress_cells"] == 2)  # start_to_goal_dist=2
    ok_efficiency = ok_reached and (runs[0]["path_efficiency"] is not None) and (runs[0]["path_efficiency"] > 0)
    ok_xml_cached = os.path.exists(str(npz_path.with_suffix(".xml")))

    record("goal_run_outcome", "goal", runs[0]["outcome"] if runs else "(no runs)", ok_reached)
    record("goal_best_time_positive", True, ok_best_time, ok_best_time)
    record("goal_success_flag", True, ok_success, ok_success)
    record("goal_max_progress_cells", 2, runs[0]["max_progress_cells"] if runs else None, ok_progress,
           "BFS最短距離(start->goal)と一致")
    record("goal_path_efficiency_computed", ">0", runs[0]["path_efficiency"] if runs else None, ok_efficiency)
    record("xml_auto_generated_and_cached", True, ok_xml_cached, ok_xml_cached,
           "npzから.xmlが自動生成・キャッシュされること(憲章の派生物規約)")

    return ok_reached and ok_best_time and ok_success and ok_progress and ok_efficiency and ok_xml_cached


# ==========================================================================
# メイン
# ==========================================================================
def main():
    print("competition/evaluator.py (M0-T1b) テストスイート")
    print("=" * 70)

    test_fns = [
        test1_static_policy,
        test2_dash_policy,
        test3_region_boundaries,
        test4_stuck_detection,
        test5_goal_reaching_policy,
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
