"""
experiments/exp_032_post_gap_graph/run.py
================
exp_032（`classic/gap_graph.py`。柱間の中点をノードにした最短経路）の測定
スクリプト。`PREREG.md` §2 の主判定量 ρ（斜めあり÷斜めなし）と §7 の副次の
記録を、対象の迷路すべてで測る。**走行はしない。純粋な計算だけ。**

対象（PREREG §2。2群は混ぜずに別々に報告する）:
  - 実大会迷路 `competition/mazes/contest_historical/`（102迷路。競技区分
    ごとにさらに分ける。区分は迷路名の末尾から復元する — 下記 _classify_contest_category）
  - 古典の設計帯（調整用迷路 seed 41000〜）`competition/mazes/design_v4/`（20迷路）

出発・ゴール: npz が明示的に持っていればそれを使う（contest_historical は
start_x/y・goals_x/y を持つ）。無ければ `classic/gap_graph.standard_goal_cells`
の中央2x2規約（design_v4はこちら）。

各迷路で PREREG §3 の等価性の錨（斜め禁止時の柱間グラフ距離が既存
`classic/flood.py` の区画グラフ距離と一致すること）を必ず確かめ、一次記録に
残す。合格条件を満たさなかった迷路があれば、その数を summary に残す
（測定を止めずに記録だけする — どのみち ρ の計算自体は続行できる）。

保存するもの（`common/output_manager.py` の慣例。archive/ に時刻つきで保存し、
finalize() で latest/ を更新する）:
  - raw_records.json: 迷路ごとの生の値（一次記録。anchor_check.py がこれだけを
    読んで ρ の中央値を独立に数え直す）
  - metrics.json（OutputManager.save_metrics 経由）: 群・競技区分ごとの集計
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from classic.flood import FloodMode, compute_flood, distance_at
from classic.gap_graph import (
    CELL_PITCH_M,
    GapPath,
    NoGapRouteError,
    build_gap_graph,
    polyline_length_m,
    shortest_path,
    standard_goal_cells,
)
from classic.maze_map import MazeMap, WallState
from common.output_manager import OutputManager

Cell = Tuple[int, int]

CONTEST_DIR = os.path.join(REPO_ROOT, "competition", "mazes", "contest_historical")
DESIGN_V4_DIR = os.path.join(REPO_ROOT, "competition", "mazes", "design_v4")

# 評価用に予約された最終評価用の迷路（seed 1000〜40999。docs/RESEARCH_PLAN.md
# §2・§9-7 が正）。design_v4 は 41000 以降（調整用迷路）なのでこの範囲には
# 入らない。念のため明示的に検査する（cf. `experiments/exp_030_map_quality/run_exp030.py`）。
RESERVED_SEED_RANGE = (1000, 40999)

MODE = FloodMode.PESSIMISTIC  # 対象は全壁既知なので楽観/悲観は同値になるはずだが明示する


# ==========================================================================
# 迷路の読み込み
# ==========================================================================
def _maze_from_truth_walls(v_walls: np.ndarray, h_walls: np.ndarray) -> MazeMap:
    """npz の真の壁配列（0=壁なし・非0=壁あり）から全壁既知の MazeMap を作る
    （`experiments/exp_030_map_quality/judge.py::load_true_map` と同じ変換）。"""
    width = v_walls.shape[0] - 1
    height = v_walls.shape[1]
    maze = MazeMap(width, height)
    maze.v_walls[:, :] = np.where(v_walls != 0, int(WallState.WALL), int(WallState.OPEN))
    maze.h_walls[:, :] = np.where(h_walls != 0, int(WallState.WALL), int(WallState.OPEN))
    return maze


def _load_maze_npz(path: str) -> Dict:
    d = np.load(path, allow_pickle=True)
    maze = _maze_from_truth_walls(d["v_walls"], d["h_walls"])
    if "start_x" in d.files:
        start = (int(d["start_x"]), int(d["start_y"]))
    else:
        start = (0, 0)
    if "goals_x" in d.files:
        goals = list(zip((int(x) for x in d["goals_x"]), (int(y) for y in d["goals_y"])))
    else:
        goals = standard_goal_cells(maze.width, maze.height)
    seed = int(d["seed"]) if "seed" in d.files else None
    return {"maze": maze, "start": start, "goals": goals, "seed": seed}


def _classify_contest_category(maze_name: str) -> str:
    """迷路名の末尾から競技区分を復元する（manifest.json の naming 規約:
    「系列_回_年_クラス_段階。exp=エキスパート／frsh=フレッシュマン／
    fin=決勝／pre=予選」）。frsh を最優先で見る（例: 2011/2012年の
    "..._frsh_fin" はフレッシュマンの決勝で、フレッシュマンに数える）。
    この判定で manifest.metrics の内訳（決勝20・予選18・フレッシュマン18・
    残り46は無区分）に一致することを確認済み。"""
    if "frsh" in maze_name:
        return "フレッシュマン"
    if "_fin" in maze_name or maze_name.endswith("fin"):
        return "決勝"
    if "_pre" in maze_name or maze_name.endswith("pre"):
        return "予選"
    return "無区分"


def _select_contest_historical() -> List[Dict]:
    manifest = json.loads(open(os.path.join(CONTEST_DIR, "manifest.json"), encoding="utf-8").read())
    tasks = []
    for name in manifest["mazes"]:
        tasks.append({
            "maze_id": name,
            "path": os.path.join(CONTEST_DIR, f"{name}.npz"),
            "category": _classify_contest_category(name),
        })
    return tasks


def _select_design_v4() -> List[Dict]:
    manifest = json.loads(open(os.path.join(DESIGN_V4_DIR, "manifest.json"), encoding="utf-8").read())
    tasks = []
    for entry in manifest["mazes"]:
        seed = int(entry["seed"])
        if RESERVED_SEED_RANGE[0] <= seed <= RESERVED_SEED_RANGE[1]:
            raise RuntimeError(
                f"評価用に予約された seed 範囲 {RESERVED_SEED_RANGE} に含まれる "
                f"seed {seed} を測定に使おうとした（禁止）。"
            )
        tasks.append({
            "maze_id": f"maze_{seed}",
            "path": os.path.join(DESIGN_V4_DIR, f"maze_{seed}.npz"),
            "category": None,
        })
    return tasks


# ==========================================================================
# 副次の記録（PREREG §7）
# ==========================================================================
def _diagonal_run_lengths_m(path: GapPath) -> List[float]:
    """斜めの区間の連続長（各連続区間の合計距離 [m]）の一覧。"""
    seg_len = np.linalg.norm(np.diff(path.xy_m.astype(np.float64), axis=0), axis=1)
    runs: List[float] = []
    cur = 0.0
    for is_diag, length in zip(path.is_diagonal, seg_len):
        if is_diag:
            cur += float(length)
        else:
            if cur > 0.0:
                runs.append(cur)
            cur = 0.0
    if cur > 0.0:
        runs.append(cur)
    return runs


def _diagonal_fraction(path: GapPath) -> float:
    seg_len = np.linalg.norm(np.diff(path.xy_m.astype(np.float64), axis=0), axis=1)
    total = float(np.sum(seg_len))
    if total <= 0.0:
        return 0.0
    diag_total = float(np.sum(seg_len[np.array(path.is_diagonal, dtype=bool)]))
    return diag_total / total


# ==========================================================================
# 1迷路の測定
# ==========================================================================
def measure_one(task: Dict) -> Dict:
    loaded = _load_maze_npz(task["path"])
    maze: MazeMap = loaded["maze"]
    start: Cell = loaded["start"]
    goals: List[Cell] = loaded["goals"]

    record: Dict = {
        "maze_id": task["maze_id"],
        "category": task.get("category"),
        "width": maze.width,
        "height": maze.height,
        "start": list(start),
        "goals": [list(g) for g in goals],
    }

    # --- 等価性の錨（PREREG §3）: 斜め禁止時、既存の区画グラフと一致するか ---
    dist = compute_flood(maze, goals, MODE)
    steps = distance_at(dist, start)
    d_cell_m = None if steps is None else float(steps) * CELL_PITCH_M
    record["cell_graph_distance_m"] = d_cell_m

    graph_nodiag = build_gap_graph(maze, MODE, allow_diagonal=False)
    graph_diag = build_gap_graph(maze, MODE, allow_diagonal=True)

    try:
        path_nodiag = shortest_path(graph_nodiag, maze, start, goals)
    except NoGapRouteError:
        path_nodiag = None
    try:
        path_diag = shortest_path(graph_diag, maze, start, goals)
    except NoGapRouteError:
        path_diag = None

    d_nodiag = None if path_nodiag is None else path_nodiag.distance_m
    d_diag = None if path_diag is None else path_diag.distance_m
    record["gap_nodiag_distance_m"] = d_nodiag
    record["gap_diag_distance_m"] = d_diag

    anchor_ok = (d_cell_m is None) == (d_nodiag is None)
    if anchor_ok and d_cell_m is not None:
        anchor_ok = abs(d_cell_m - d_nodiag) < 1e-3
    record["anchor_ok"] = bool(anchor_ok)

    # 折れ線と距離の食い違いが無いかも一次記録に残す（辺の重みと幾何の整合）
    if path_nodiag is not None:
        record["polyline_len_nodiag_m"] = polyline_length_m(path_nodiag.xy_m)
    if path_diag is not None:
        record["polyline_len_diag_m"] = polyline_length_m(path_diag.xy_m)

    # --- 主判定量 ρ ---
    rho = None
    if d_nodiag is not None and d_nodiag > 0 and d_diag is not None:
        rho = d_diag / d_nodiag
    record["rho"] = rho

    # --- 副次の記録（PREREG §7） ---
    if path_diag is not None:
        record["diagonal_fraction"] = _diagonal_fraction(path_diag)
        record["diagonal_run_lengths_m"] = _diagonal_run_lengths_m(path_diag)
    else:
        record["diagonal_fraction"] = None
        record["diagonal_run_lengths_m"] = []

    path_changed = None
    if path_nodiag is not None and path_diag is not None:
        path_changed = path_nodiag.node_ids != path_diag.node_ids
    record["path_changed"] = path_changed

    return record


# ==========================================================================
# 集計
# ==========================================================================
def _median(values: Sequence[float]) -> Optional[float]:
    vs = [v for v in values if v is not None]
    if not vs:
        return None
    return float(np.median(vs))


def _summarize_group(records: List[Dict], group_name: str) -> Dict:
    rhos = [r["rho"] for r in records if r["rho"] is not None]
    n_anchor_fail = sum(1 for r in records if not r["anchor_ok"])
    n_path_changed = sum(1 for r in records if r.get("path_changed"))
    diag_fracs = [r["diagonal_fraction"] for r in records if r.get("diagonal_fraction") is not None]
    all_run_lengths: List[float] = []
    for r in records:
        all_run_lengths.extend(r.get("diagonal_run_lengths_m", []))

    summary = {
        "group": group_name,
        "n_mazes": len(records),
        "n_rho_defined": len(rhos),
        "rho_median": _median(rhos),
        "rho_min": float(np.min(rhos)) if rhos else None,
        "rho_max": float(np.max(rhos)) if rhos else None,
        "n_anchor_fail": n_anchor_fail,
        "anchor_fail_maze_ids": [r["maze_id"] for r in records if not r["anchor_ok"]],
        "n_path_changed": n_path_changed,
        "diagonal_fraction_median": _median(diag_fracs),
        "diagonal_run_length_m": {
            "n_runs": len(all_run_lengths),
            "median": _median(all_run_lengths),
            "mean": float(np.mean(all_run_lengths)) if all_run_lengths else None,
            "min": float(np.min(all_run_lengths)) if all_run_lengths else None,
            "max": float(np.max(all_run_lengths)) if all_run_lengths else None,
        },
    }
    return summary


def main() -> None:
    contest_tasks = _select_contest_historical()
    design_tasks = _select_design_v4()

    contest_records = [measure_one(t) for t in contest_tasks]
    design_records = [measure_one(t) for t in design_tasks]

    om = OutputManager("exp_032_post_gap_graph")

    raw_records = {
        "contest_historical": contest_records,
        "design_v4": design_records,
    }
    raw_path = om.get_path("raw_records.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_records, f, ensure_ascii=False, indent=2)

    # 実大会迷路は競技区分ごとにも分ける（PREREG §2）
    categories = ["決勝", "予選", "フレッシュマン", "無区分"]
    contest_by_category = {
        cat: _summarize_group([r for r in contest_records if r["category"] == cat], f"contest_historical:{cat}")
        for cat in categories
    }

    summary = {
        "mode": MODE.value,
        "design_v4": _summarize_group(design_records, "design_v4"),
        "contest_historical_all": _summarize_group(contest_records, "contest_historical:全体"),
        "contest_historical_by_category": contest_by_category,
    }

    om.save_metrics(
        {"summary": {
            "design_v4_rho_median": summary["design_v4"]["rho_median"],
            "contest_historical_rho_median": summary["contest_historical_all"]["rho_median"],
            "n_anchor_fail_total": summary["design_v4"]["n_anchor_fail"] + summary["contest_historical_all"]["n_anchor_fail"],
        }},
        phase_specific=summary,
    )

    n_anchor_fail_total = summary["design_v4"]["n_anchor_fail"] + summary["contest_historical_all"]["n_anchor_fail"]
    om.finalize(
        summary=(
            f"exp_032: design_v4 rho中央値={summary['design_v4']['rho_median']:.4f} / "
            f"contest_historical rho中央値={summary['contest_historical_all']['rho_median']:.4f} / "
            f"等価性の錨の不一致={n_anchor_fail_total}件"
        )
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
