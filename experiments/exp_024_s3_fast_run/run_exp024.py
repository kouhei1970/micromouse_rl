"""
experiments/exp_024_s3_fast_run/run_exp024.py
================
exp_024（段階 S3「最短走行」の完了判定）の一次記録を測る測定スクリプト。

🔴 本スクリプトは合否を一切判定しない。判定は
`experiments/exp_024_s3_fast_run/recompute_anchors.py` と教授セッションが行う
（`PREREG.md` を見ること。条文は本スクリプトにもここにも写さない）。

対象迷路（PREREG §3）: `competition/mazes/design_turn_v1/manifest.json` から
**最短歩数 D0 が小さい順に 6 迷路（同点は seed の小さい順）**を、実行のたびに
`select_target_mazes()` で選び直す（seed をハードコードしない）。

条件（PREREG §4）は 2 つ:
  - extended: `ClassicExplorerPolicy(extend_straights=True)`（本命の構成）
  - percell : `ClassicExplorerPolicy(extend_straights=False)`（直線延伸だけを外した対照）

1 (迷路, 条件) の組ごとに `CompetitionEvaluator(time_budget=420.0, max_runs=5)`
の `evaluate_maze()` を 1 回走らせ、返った辞書へ `plan_ids`（`get_plan_ids()`）
と `run_phases`（`get_run_phases()`）を足して
`outputs/exp_024_s3/<condition>/maze_<seed>.json` に保存する。

6 迷路 × 2 条件 = 12 本を multiprocessing で並列に走らせる（既定）。
`--serial` で逐次実行にも切り替えられる。親プロセスは前景で全部の完了を待つ
（バックグラウンドへは投げない）。

使い方:
    .venv/bin/python experiments/exp_024_s3_fast_run/run_exp024.py            # 並列実行
    .venv/bin/python experiments/exp_024_s3_fast_run/run_exp024.py --serial   # 逐次実行
    .venv/bin/python experiments/exp_024_s3_fast_run/run_exp024.py --dry-run  # 対象 6 迷路の
        seed・D0 を印字して終わる（測定は 1 本も走らせない）
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MANIFEST_PATH = REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "manifest.json"
OUT_ROOT = REPO_ROOT / "outputs" / "exp_024_s3"

N_TARGET_MAZES = 6
TIME_BUDGET_S = 420.0
MAX_RUNS = 5

# 条件キー → ClassicExplorerPolicy(extend_straights=...) の値（PREREG §4）。
CONDITIONS: List[tuple] = [("extended", True), ("percell", False)]

# 評価用に予約された seed 範囲（docs/RESEARCH_PLAN.md §2・§9-7 を基準とする）。
# ここでは「選んでしまっていないか」を機械的に確かめる安全弁としてのみ使う
# （判定条文ではない）。
RESERVED_SEED_RANGE = (1000, 40999)


# ==========================================================================
# 対象迷路の選定（PREREG §3。結果に依存しない・seed を直書きしない）
# ==========================================================================
def select_target_mazes(manifest_path: Path = MANIFEST_PATH, n: int = N_TARGET_MAZES) -> List[Dict]:
    """`manifest.json` から D0 昇順（同点は seed 昇順）で先頭 n 件を選ぶ。

    Returns:
        [{"seed": int, "d0": int}, ...] （D0 昇順・同点は seed 昇順）
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mazes = manifest["mazes"]
    ordered = sorted(mazes, key=lambda m: (int(m["d0"]), int(m["seed"])))
    selected = [{"seed": int(m["seed"]), "d0": int(m["d0"])} for m in ordered[:n]]

    lo, hi = RESERVED_SEED_RANGE
    for m in selected:
        if lo <= m["seed"] <= hi:
            raise RuntimeError(
                f"評価用に予約された seed 範囲 [{lo}, {hi}] に含まれる seed {m['seed']} を"
                f"選んでしまった（manifest.json の異常、または本スクリプトの誤り。測定禁止）。"
            )
    return selected


# ==========================================================================
# 1 (迷路, 条件) の測定（子プロセスで実行される）
# ==========================================================================
def _run_one(task: Dict) -> Dict:
    """1 (迷路, 条件) の測定を行う。multiprocessing の子プロセスから呼ばれる
    ことを想定し、重い依存（mujoco 経由の competition/classic）はここで初めて
    import する（`--dry-run` を軽く保つため、モジュール先頭では import しない）。
    """
    from classic.policy import ClassicExplorerPolicy
    from competition.evaluator import CompetitionEvaluator

    seed = task["seed"]
    d0 = task["d0"]
    condition = task["condition"]
    extend_straights = task["extend_straights"]
    maze_dir = Path(task["maze_dir"])
    out_path = Path(task["out_path"])

    npz_path = maze_dir / f"maze_{seed}.npz"

    t0 = time.time()
    evaluator = CompetitionEvaluator(
        maze_dir=str(maze_dir),
        time_budget=TIME_BUDGET_S,
        max_runs=MAX_RUNS,
    )
    policy = ClassicExplorerPolicy(extend_straights=extend_straights)
    result = evaluator.evaluate_maze(npz_path, policy)

    # PREREG §4: 一次記録は評価器の結果 dict そのままに、走行の一次記録
    # （集約値ではない）である plan_ids・run_phases を足したもの。
    result["plan_ids"] = policy.get_plan_ids()
    result["run_phases"] = policy.get_run_phases()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    elapsed_s = time.time() - t0
    outcomes = [r["outcome"] for r in result["runs"]]
    return {
        "seed": seed, "d0": d0, "condition": condition,
        "outcomes": outcomes, "elapsed_s": elapsed_s,
        "out_path": str(out_path),
    }


def _build_tasks(target_mazes: List[Dict], maze_dir: Path) -> List[Dict]:
    tasks = []
    for m in target_mazes:
        for condition, extend_straights in CONDITIONS:
            out_path = OUT_ROOT / condition / f"maze_{m['seed']}.json"
            tasks.append({
                "seed": m["seed"], "d0": m["d0"],
                "condition": condition, "extend_straights": extend_straights,
                "maze_dir": str(maze_dir), "out_path": str(out_path),
            })
    return tasks


def _print_progress(done_idx: int, total: int, r: Dict) -> None:
    print(
        f"[{done_idx}/{total}] maze={r['seed']:>5} (D0={r['d0']:>3}) "
        f"condition={r['condition']:<8} outcomes={r['outcomes']} "
        f"elapsed={r['elapsed_s']:.1f}s -> {r['out_path']}",
        flush=True,
    )


# ==========================================================================
# メイン
# ==========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--serial", action="store_true",
                         help="逐次実行に切り替える（既定は multiprocessing による並列実行）")
    parser.add_argument("--workers", type=int, default=None,
                         help="並列数（既定: min(6, os.cpu_count()-2)）")
    parser.add_argument("--dry-run", action="store_true",
                         help="対象 6 迷路の seed・D0 を印字して終わる（測定は 1 本も走らせない）")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH,
                         help="manifest.json のパス（既定: design_turn_v1）")
    args = parser.parse_args(argv)

    maze_dir = args.manifest.parent
    target_mazes = select_target_mazes(args.manifest, N_TARGET_MAZES)

    print(f"対象 {len(target_mazes)} 迷路（D0 昇順・同点は seed 昇順、"
          f"manifest: {args.manifest}）:")
    for i, m in enumerate(target_mazes, 1):
        print(f"  {i}. seed={m['seed']} D0={m['d0']}")

    if args.dry_run:
        print("\n--dry-run: 測定は実行しない（PREREG §3 の対象確認のみ）。")
        return 0

    tasks = _build_tasks(target_mazes, maze_dir)
    total = len(tasks)
    print(f"\n{total} 本（{len(target_mazes)} 迷路 × {len(CONDITIONS)} 条件）を"
          f"{'逐次' if args.serial else '並列'}実行する。")

    wall_clock_start = time.time()
    results: List[Dict] = []

    if args.serial:
        for i, task in enumerate(tasks, 1):
            r = _run_one(task)
            results.append(r)
            _print_progress(i, total, r)
    else:
        n_workers = args.workers if args.workers is not None else max(1, min(6, (os.cpu_count() or 2) - 2))
        n_workers = min(n_workers, total)
        print(f"並列数: {n_workers}")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_run_one, tasks), 1):
                results.append(r)
                _print_progress(i, total, r)

    total_wall_clock = time.time() - wall_clock_start
    print(f"\n完了: {total} 本 / 実時間 {total_wall_clock:.1f}s")
    print("🔴 合否の判定はここでは行わない。"
          "experiments/exp_024_s3_fast_run/recompute_anchors.py で行うこと。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
