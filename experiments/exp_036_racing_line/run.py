"""
experiments/exp_036_racing_line/run.py
================
exp_036（`classic/racing_line.py`。柱間グラフの折れ線の角を丸めた走行ライン）の
測定スクリプト。`PREREG.md` §2 の主判定量 τ・§3 の合格条件・§4 の否定対照・
§7 の副次の記録を、対象の迷路すべてで測る。**走行はしない。純粋な計算だけ。**

対象: `competition/mazes/design_v4/`（調整用迷路。seed 41000以降。評価用に
予約された seed 1000〜40999 には入らない — 念のため明示的に検査する）。

比較対象（分母）: 「従来のターン種別の列」の理想時間。`classic/route.py`の
`shortest_path`（区画列）を`classic/fast_planner.py::plan_fast_run`に通した
`FastPlan.t_plan`（内部で`classic.profile.min_time`を呼ぶ。同じ`vehicle_limits()`・
同じ`clearance_margin_m`既定値0.005を使う — 丸め方の違いだけを見るため）。
**同じ発進**にするため、区画列の最初の1手の向きを`start_heading`として渡す
（柱間グラフ側の折れ線は出発時の向きという概念を持たない — 最初の一歩の方向を
向いている前提で作られている。区画グラフ側もそれに合わせて公平にする）。

保存するもの（`common/output_manager.py`の慣例。archive/に時刻つきで保存し、
finalize()でlatest/を更新する）:
  - raw_records.json: 迷路ごとの一次記録（κ(s)格子・理想時間・余裕の最小値・
    否定対照の実測値）。anchor_check.pyがこれだけを読んでτの中央値・余裕の
    最小値を数え直す。
  - metrics.json（OutputManager.save_metrics経由）: 集計（τの中央値など）
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from classic.flood import FloodMode
from classic.gap_graph import (
    GapPath,
    NoGapRouteError,
    build_gap_graph,
    shortest_path as gap_shortest_path,
    standard_goal_cells,
)
from classic.fast_planner import plan_fast_run
from classic.maze_map import Direction, MazeMap, WallState, direction_between
from classic.profile import Segment, min_time, vehicle_limits
from classic.racing_line import (
    RacingLineError,
    RacingLineOverlapError,
    build_racing_line,
    diagonal_length_m,
    evaluate_clearance,
    find_max_feasible_racing_line,
    max_kappa_jump,
    to_segments,
)
from classic.route import shortest_path as cell_shortest_path
from common.output_manager import OutputManager

Cell = Tuple[int, int]

DESIGN_V4_DIR = os.path.join(REPO_ROOT, "competition", "mazes", "design_v4")

# 評価用に予約された最終評価用の迷路（seed 1000〜40999。docs/RESEARCH_PLAN.md
# §2・§9-7が正）。design_v4は41000以降（調整用迷路）なのでこの範囲には入らない。
# 念のため明示的に検査する（`experiments/exp_032_post_gap_graph/run.py`と同じ作法）。
RESERVED_SEED_RANGE = (1000, 40999)

MODE = FloodMode.PESSIMISTIC  # 対象は全壁既知なので楽観/悲観の違いは出ない

MARGIN_M = 0.005  # `classic.racing_line`・`classic.fast_planner`の既定margin_mと揃える


# ==========================================================================
# 迷路の読み込み（`tests/test_gap_graph.py::_maze_from_truth_walls`と同じ複製）
# ==========================================================================
def _maze_from_truth_walls(v_walls: np.ndarray, h_walls: np.ndarray) -> MazeMap:
    width = v_walls.shape[0] - 1
    height = v_walls.shape[1]
    maze = MazeMap(width, height)
    maze.v_walls[:, :] = np.where(v_walls != 0, int(WallState.WALL), int(WallState.OPEN))
    maze.h_walls[:, :] = np.where(h_walls != 0, int(WallState.WALL), int(WallState.OPEN))
    return maze


def _select_design_v4() -> List[Dict]:
    manifest = json.loads(open(os.path.join(DESIGN_V4_DIR, "manifest.json"), encoding="utf-8").read())
    tasks = []
    for entry in manifest["mazes"]:
        seed = int(entry["seed"])
        if RESERVED_SEED_RANGE[0] <= seed <= RESERVED_SEED_RANGE[1]:
            raise RuntimeError(
                f"評価用に予約されたseed範囲{RESERVED_SEED_RANGE}に含まれるseed {seed} を"
                "測定に使おうとした（禁止）。"
            )
        tasks.append({"maze_id": f"maze_{seed}", "path": os.path.join(DESIGN_V4_DIR, f"maze_{seed}.npz")})
    return tasks


def _load_maze_npz(path: str) -> MazeMap:
    d = np.load(path)
    return _maze_from_truth_walls(d["v_walls"], d["h_walls"])


# ==========================================================================
# 比較対象（分母）: 従来のターン種別の列（route.py + fast_planner.py）
# ==========================================================================
def _baseline_ideal_time(maze: MazeMap, start: Cell, goals: Sequence[Cell]) -> Dict:
    """`classic/route.py`の区画列を`classic/fast_planner.py::plan_fast_run`に
    通した理想時間（内部でclassic.profile.min_timeを呼ぶ）。「同じ発進」に
    するため、区画列の最初の1手の向きをstart_headingとして渡す（モジュール
    docstring参照）。"""
    cells = cell_shortest_path(maze, start, list(goals), MODE)
    first_dir = direction_between(cells[0], cells[1]) if len(cells) > 1 else Direction.N
    plan = plan_fast_run(maze, start, list(goals), start_heading=first_dir, clearance_margin_m=MARGIN_M)
    if plan is None:
        raise RuntimeError("plan_fast_runが到達不能を返した（design_v4は全壁既知のはず）")
    return {
        "t_plan": plan.t_plan, "n_turns": plan.n_turns,
        "n_forced_spins": plan.n_forced_spins, "start_heading": first_dir.name,
    }


# ==========================================================================
# 否定対照（PREREG §4）
# ==========================================================================
def _n1_unrounded_time(path: GapPath, limits) -> float:
    """N1: 丸めを行わず折れ線のまま`min_time()`に通す。角では曲率が未定義
    （無限大）になるので、各線分を独立に「停止→停止」で解く（角で必ず止まる
    = `classic.ideal._ideal_spin`の直線ランと同じ扱い。その場旋回の時間は
    含めない — それでも遅くなることを示せれば十分で、含めればさらに遅くなる
    ので下駄を履かせていない）。"""
    xy = path.xy_m.astype(np.float64)
    seg_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    total = 0.0
    for length in seg_lengths:
        length_f = float(length)
        if length_f <= 1e-9:
            continue
        seg = Segment(length=length_f, curvature=0.0, kind="straight")
        it = min_time([seg], limits, v_start=0.0, v_end=0.0)
        total += it.total
    return total


def _n2_max_r_ignoring_margin(path: GapPath, ds: float, r_lo: float = 0.02, r_hi: float = 0.30,
                               iters: int = 30) -> Optional[float]:
    """N2: 余裕（壁との干渉）の判定を外し、消費長の予算だけを守って角を
    深く丸められる最大のRを探す（`classic.racing_line.build_racing_line`は
    壁を見ないので、そのまま二分探索するだけでよい）。"""
    def overlap_ok(R: float) -> bool:
        try:
            build_racing_line(path, R, ds=ds)
            return True
        except RacingLineOverlapError:
            return False

    if not overlap_ok(r_lo):
        return None
    if overlap_ok(r_hi):
        return r_hi
    lo, hi = r_lo, r_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if overlap_ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


# ==========================================================================
# 1迷路の測定
# ==========================================================================
def measure_one(task: Dict, limits) -> Dict:
    maze = _load_maze_npz(task["path"])
    start: Cell = (0, 0)
    goals: List[Cell] = standard_goal_cells(maze.width, maze.height)

    record: Dict = {"maze_id": task["maze_id"], "width": maze.width, "height": maze.height}

    # --- 柱間グラフの折れ線（斜め許可） ---
    graph_diag = build_gap_graph(maze, MODE, allow_diagonal=True)
    path_diag = gap_shortest_path(graph_diag, maze, start, goals)
    record["gap_path_distance_m"] = path_diag.distance_m

    # --- 丸めの深さRの探索 → 走行ラインの理想時間（主判定量τの分子） ---
    t0 = time.time()
    line = find_max_feasible_racing_line(path_diag, maze, margin_m=MARGIN_M)
    search_time_s = time.time() - t0
    min_clearance_m, worst_idx = evaluate_clearance(line, maze)
    jump = max_kappa_jump(line)

    racing_segs = to_segments(line)
    it_racing = min_time(racing_segs, limits, v_start=0.0, v_end=0.0)

    record["racing_line"] = {
        "R_m": line.R_m,
        "search_time_s": search_time_s,
        "n_cells": len(line.kappa_grid),
        "corner_count": line.corner_count,
        "s_grid": line.s_grid,
        "kappa_grid": line.kappa_grid,
        "kind_grid": line.kind_grid,
        "ideal_time_s": it_racing.total,
        "by_kind_s": {k: float(v) for k, v in it_racing.by_kind.items()},
        "min_clearance_m": min_clearance_m,
        "worst_idx": worst_idx,
        "max_kappa_jump": jump,
        "kappa_max_abs": line.kappa_max_abs(),
        "total_length_m": line.total_length_m,
        "original_length_m": line.original_length_m,
        "diagonal_length_m": diagonal_length_m(line),
    }

    # --- 比較対象（分母） ---
    baseline = _baseline_ideal_time(maze, start, goals)
    record["baseline"] = baseline

    tau = it_racing.total / baseline["t_plan"] if baseline["t_plan"] > 0 else None
    record["tau"] = tau

    # --- 否定対照 ---
    n1_time = _n1_unrounded_time(path_diag, limits)
    record["n1_unrounded_time_s"] = n1_time
    record["n1_fires"] = bool(n1_time > it_racing.total)  # 理想時間が悪化するはず

    r_n2 = _n2_max_r_ignoring_margin(path_diag, ds=0.002)
    if r_n2 is not None:
        line_n2 = build_racing_line(path_diag, r_n2, ds=0.002)
        clearance_n2, _ = evaluate_clearance(line_n2, maze)
    else:
        clearance_n2 = None
    record["n2_max_r_ignoring_margin_m"] = r_n2
    record["n2_clearance_at_that_r_m"] = clearance_n2
    record["n2_fires"] = bool(clearance_n2 is not None and clearance_n2 < 0.0)

    graph_nodiag = build_gap_graph(maze, MODE, allow_diagonal=False)
    path_nodiag = gap_shortest_path(graph_nodiag, maze, start, goals)
    line_nodiag = find_max_feasible_racing_line(path_nodiag, maze, margin_m=MARGIN_M)
    segs_nodiag = to_segments(line_nodiag)
    it_nodiag = min_time(segs_nodiag, limits, v_start=0.0, v_end=0.0)
    tau_nodiag = it_nodiag.total / baseline["t_plan"] if baseline["t_plan"] > 0 else None
    record["n3_tau_nodiag"] = tau_nodiag
    record["n3_fires"] = bool(
        tau is not None and tau_nodiag is not None and abs(tau_nodiag - 1.0) < abs(tau - 1.0)
    )

    return record


# ==========================================================================
# 集計
# ==========================================================================
def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    vs = [v for v in values if v is not None]
    if not vs:
        return None
    return float(np.median(vs))


def main() -> None:
    tasks = _select_design_v4()
    limits = vehicle_limits()

    records: List[Dict] = []
    t_start = time.time()
    for i, task in enumerate(tasks):
        t0 = time.time()
        rec = measure_one(task, limits)
        records.append(rec)
        print(
            f"[{i+1}/{len(tasks)}] {task['maze_id']}: tau={rec['tau']:.4f} "
            f"min_clear={rec['racing_line']['min_clearance_m']:.5f} "
            f"jump={rec['racing_line']['max_kappa_jump']:.4f} "
            f"({time.time()-t0:.1f}s)",
            flush=True,
        )
    total_elapsed = time.time() - t_start

    om = OutputManager("exp_036_racing_line")
    raw_path = om.get_path("raw_records.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({"design_v4": records}, f, ensure_ascii=False, indent=2)

    taus = [r["tau"] for r in records]
    min_clears = [r["racing_line"]["min_clearance_m"] for r in records]
    max_jumps = [r["racing_line"]["max_kappa_jump"] for r in records]
    n1_times = [r["n1_unrounded_time_s"] for r in records]
    racing_times = [r["racing_line"]["ideal_time_s"] for r in records]
    n2_clearances = [r["n2_clearance_at_that_r_m"] for r in records]
    n3_taus = [r["n3_tau_nodiag"] for r in records]

    summary = {
        "n_mazes": len(records),
        "tau_median": _median(taus),
        "tau_min": float(np.min(taus)) if taus else None,
        "tau_max": float(np.max(taus)) if taus else None,
        "min_clearance_overall_m": float(np.min(min_clears)) if min_clears else None,
        "max_kappa_jump_overall": float(np.max(max_jumps)) if max_jumps else None,
        "n_clearance_below_margin": sum(1 for c in min_clears if c < MARGIN_M - 1e-9),
        "negative_controls": {
            "N1_median_unrounded_time_s": _median(n1_times),
            "N1_median_racing_time_s": _median(racing_times),
            "N1_n_fired": sum(1 for r in records if r["n1_fires"]),
            "N2_median_r_ignoring_margin_m": _median([r["n2_max_r_ignoring_margin_m"] for r in records]),
            "N2_median_clearance_at_that_r_m": _median(n2_clearances),
            "N2_n_fired": sum(1 for r in records if r["n2_fires"]),
            "N3_tau_nodiag_median": _median(n3_taus),
            "N3_tau_diag_median": _median(taus),
            "N3_n_fired": sum(1 for r in records if r["n3_fires"]),
        },
        "secondary_records": {
            "diagonal_length_fraction_median": _median([
                r["racing_line"]["diagonal_length_m"] / r["racing_line"]["total_length_m"]
                for r in records if r["racing_line"]["total_length_m"] > 0
            ]),
            "corner_count_median": _median([float(r["racing_line"]["corner_count"]) for r in records]),
            "path_shortening_m_median": _median([
                r["racing_line"]["original_length_m"] - r["racing_line"]["total_length_m"] for r in records
            ]),
            "kappa_max_abs_overall": float(np.max([r["racing_line"]["kappa_max_abs"] for r in records])),
            "search_time_s_median": _median([r["racing_line"]["search_time_s"] for r in records]),
        },
        "total_elapsed_s": total_elapsed,
    }

    om.save_metrics({"summary": {
        "tau_median": summary["tau_median"],
        "min_clearance_overall_m": summary["min_clearance_overall_m"],
        "max_kappa_jump_overall": summary["max_kappa_jump_overall"],
    }}, phase_specific=summary)

    om.finalize(
        summary=(
            f"exp_036: tau中央値={summary['tau_median']:.4f} / "
            f"余裕の最小値(全迷路)={summary['min_clearance_overall_m']:.5f}m / "
            f"曲率の跳びの最大値={summary['max_kappa_jump_overall']:.4f} / "
            f"margin未満の迷路数={summary['n_clearance_below_margin']}"
        )
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
