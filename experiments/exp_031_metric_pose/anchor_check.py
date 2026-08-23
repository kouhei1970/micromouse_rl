"""
experiments/exp_031_metric_pose/anchor_check.py
================
exp_031 の錨の独立再計算（PREREG.md §5、`docs/RESEARCH_PLAN.md` §12-9）。

入力は走行の**一次記録**（`experiments/exp_031_metric_pose/run.py` が
`outputs/exp_031_metric_pose/raw/maze_<seed>_track.npz` へ保存した、
ティックごとの推定姿勢・真値姿勢・計画識別子の生ログ）だけ。

🔴 本ファイルは `run.py` の集計出力（`outputs/exp_031_metric_pose/maze_*.json`・
`summary.json`）を一切読まない。`run.py` の関数も一切 import しない
（`poses_to_cells`・`explore_window_length`・`phase_prefix` などは、
下でゼロから書き直している。同じ関数を import して使い回すと、その関数に
バグがあった場合に「集計側の誤りを集計側でもう一度確認しているだけ」に
なってしまい、独立再計算にならないため）。

$r_{cell}$ を自分で数え直し、`run.py` が出した値と**一致するかどうかを
確かめる作業自体は、本ファイルの外（両者の JSON を読み比べる別の手順）で
行う（`run.py` の集計出力を読まないという制約と両立させるため）。
本ファイルは自分の再計算結果だけを出力する。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "outputs" / "exp_031_metric_pose" / "raw"
OUT_PATH = REPO_ROOT / "outputs" / "exp_031_metric_pose" / "anchor_check_results.json"


def _cell_of(x: float, y: float, cell_size: float) -> tuple[int, int]:
    """floor(x/cell_size), floor(y/cell_size)。境界は下側を含む。
    `run.py` の `poses_to_cells` とは独立に、ここでゼロから書く。"""
    import math
    return (int(math.floor(x / cell_size)), int(math.floor(y / cell_size)))


def _find_explore_window_end(plan_ids: list[str]) -> int:
    """「探索走行」の終わり = 計画識別子の接頭辞(":"より前)が初めて "fast" に
    なったティックの直前まで。1度もならなければ全長。
    `run.py` の `explore_window_length`/`phase_prefix` とは独立に、
    ここでゼロから書く（同じ実装を import しない）。"""
    for i, pid in enumerate(plan_ids):
        prefix = pid.split(":")[0]
        if prefix == "fast":
            return i
    return len(plan_ids)


def recompute_r_cell_for_file(npz_path: Path) -> dict:
    data = np.load(npz_path)
    seed = int(data["seed"])
    cell_size = float(data["cell_size"])
    est_x = data["est_x"]
    est_y = data["est_y"]
    true_x = data["true_x"]
    true_y = data["true_y"]
    plan_ids = [str(s) for s in data["plan_id"]]

    n = len(plan_ids)
    if not (len(est_x) == len(est_y) == len(true_x) == len(true_y) == n):
        raise RuntimeError(
            f"{npz_path}: 配列長が一致しない(plan_id={n}, est_x={len(est_x)}, "
            f"est_y={len(est_y)}, true_x={len(true_x)}, true_y={len(true_y)})"
        )

    window = _find_explore_window_end(plan_ids)
    if window == 0:
        raise RuntimeError(f"{npz_path}: 探索走行の長さが0ティックだった")

    n_mismatch = 0
    for i in range(window):
        ecx, ecy = _cell_of(float(est_x[i]), float(est_y[i]), cell_size)
        tcx, tcy = _cell_of(float(true_x[i]), float(true_y[i]), cell_size)
        if ecx != tcx or ecy != tcy:
            n_mismatch += 1

    r_cell = n_mismatch / window

    return {
        "seed": seed,
        "npz_path": str(npz_path),
        "n_ticks_total": n,
        "n_ticks_explore_run": window,
        "n_mismatch_ticks": n_mismatch,
        "r_cell": r_cell,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args(argv)

    files = sorted(args.raw_dir.glob("maze_*_track.npz"),
                    key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise RuntimeError(f"{args.raw_dir} に一次記録(maze_*_track.npz)が見つからない")

    results = []
    for f in files:
        r = recompute_r_cell_for_file(f)
        results.append(r)
        print(f"[anchor_check] seed={r['seed']:>5} r_cell={r['r_cell']:.6f} "
              f"(mismatch={r['n_mismatch_ticks']}/{r['n_ticks_explore_run']}, "
              f"total_ticks={r['n_ticks_total']})")

    r_cells = [r["r_cell"] for r in results]
    median_r_cell = float(np.median(r_cells))
    print(f"\n[anchor_check] median r_cell (再計算) = {median_r_cell}")

    out = {
        "n_files": len(results),
        "per_maze": results,
        "median_r_cell": median_r_cell,
        "note": "run.py の集計出力(maze_*.json / summary.json)は一切読んでいない。"
                "入力は outputs/exp_031_metric_pose/raw/maze_*_track.npz(一次記録)のみ。",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print(f"[anchor_check] 保存: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
