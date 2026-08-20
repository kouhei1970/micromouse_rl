"""
experiments/exp_026_profile_run/run_exp026.py
================
exp_026（最短走行のプロファイル追従化と η の実測）の一次記録を測る測定スクリプト。

🔴 本スクリプトは合否を一切判定しない。判定は
`experiments/exp_026_profile_run/judge.py` が行う（`PREREG.md` を見ること。
判定条文そのものは `research_notes/note_031_profile_planner_and_eta.md` を参照）。

`experiments/exp_025_s4_slalom/run_exp025.py` の構造（対象迷路の選定・
multiprocessing による並列実行・`--serial`/`--dry-run`）を踏襲するが、次を変える:

  - 条件: `ClassicExplorerPolicy(fast_mode=...)`。`"command"`（対照）/
    `"profile"`（作動側）の2値（PREREG §3）。
  - `profile` 条件のみ、`evaluate_maze()` 完了後に方策が学習した地図
    （`policy._explorer.maze.v_walls`/`h_walls`。真値ではない）を一次記録へ
    書き足す（PREREG §5。`judge.py` が `t_plan` を独立に再計算するのに使う）。

使い方:
    .venv/bin/python experiments/exp_026_profile_run/run_exp026.py            # 並列実行
    .venv/bin/python experiments/exp_026_profile_run/run_exp026.py --serial   # 逐次実行
    .venv/bin/python experiments/exp_026_profile_run/run_exp026.py --dry-run  # 対象10迷路の
        seed一覧の印字だけで終わる（測定は1本も走らせない）

🔴 測定は前景で完走させること（バックグラウンドへ投げてターンを終えると
プロセスが落ちる）。
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
OUT_ROOT = REPO_ROOT / "outputs" / "exp_026"

TIME_BUDGET_S = 1500.0
MAX_RUNS = 5

# 条件キー → ClassicExplorerPolicy(fast_mode=...) の値（PREREG §3）。
CONDITIONS: List[str] = ["command", "profile"]

# 評価用に予約された seed 範囲（docs/RESEARCH_PLAN.md §2・§9-7 が正）。
# ここでは「選んでしまっていないか」を機械的に確かめる安全弁としてのみ使う
# （判定条文ではない。exp_024/exp_025 の同名チェックと同じ作法）。
RESERVED_SEED_RANGE = (1000, 40999)


# ==========================================================================
# 対象迷路の選定（PREREG §2。結果に依存しない・seedを直書きしない）
# ==========================================================================
def select_target_mazes(manifest_path: Path = MANIFEST_PATH) -> List[Dict]:
    """`manifest.json` の全迷路を seed 昇順で返す（PREREG §2 は design_turn_v1
    の全10迷路を対象とする）。

    Returns:
        [{"seed": int, "d0": int}, ...] （seed昇順）
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mazes = manifest["mazes"]
    ordered = sorted(mazes, key=lambda m: int(m["seed"]))
    selected = [{"seed": int(m["seed"]), "d0": int(m["d0"])} for m in ordered]

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
    fast_mode = task["fast_mode"]
    maze_dir = Path(task["maze_dir"])
    out_path = Path(task["out_path"])

    npz_path = maze_dir / f"maze_{seed}.npz"

    t0 = time.time()
    evaluator = CompetitionEvaluator(
        maze_dir=str(maze_dir),
        time_budget=TIME_BUDGET_S,
        max_runs=MAX_RUNS,
    )
    policy = ClassicExplorerPolicy(fast_mode=fast_mode)
    result = evaluator.evaluate_maze(npz_path, policy)

    # PREREG §4: 一次記録は評価器の結果 dict そのままに、走行の一次記録
    # （集約値ではない）である plan_ids・run_phases を足したもの。
    result["plan_ids"] = policy.get_plan_ids()
    result["run_phases"] = policy.get_run_phases()

    # PREREG §5: profile 条件のみ、方策が学習した地図（真値ではない）を
    # 一次記録へ書き足す（judge.py が t_plan を独立に再計算するのに使う）。
    # 🔴 読み取り専用の後処理（act()/tick() の電圧計算には一切関与しない）。
    if fast_mode == "profile" and getattr(policy, "_explorer", None) is not None:
        maze = policy._explorer.maze
        result["maze_v_walls"] = maze.v_walls.tolist()
        result["maze_h_walls"] = maze.h_walls.tolist()

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
        for condition in CONDITIONS:
            out_path = OUT_ROOT / condition / f"maze_{m['seed']}.json"
            tasks.append({
                "seed": m["seed"], "d0": m["d0"],
                "condition": condition, "fast_mode": condition,
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
                         help="対象10迷路のseed一覧の印字だけで終わる（測定は1本も走らせない）")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH,
                         help="manifest.jsonのパス（既定: design_turn_v1）")
    parser.add_argument("--only-seed", type=int, default=None,
                         help="指定したseedの迷路だけを測定する（否定対照N3/N4用）")
    args = parser.parse_args(argv)

    maze_dir = args.manifest.parent
    target_mazes = select_target_mazes(args.manifest)
    if args.only_seed is not None:
        target_mazes = [m for m in target_mazes if m["seed"] == args.only_seed]
        if not target_mazes:
            raise SystemExit(f"--only-seed={args.only_seed} は対象10迷路に含まれない")

    print(f"対象 {len(target_mazes)} 迷路（seed昇順、manifest: {args.manifest}）:")
    for i, m in enumerate(target_mazes, 1):
        print(f"  {i}. seed={m['seed']} D0={m['d0']}")

    if args.dry_run:
        print("\n--dry-run: 測定は実行しない（対象迷路の確認のみ）。")
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
          "experiments/exp_026_profile_run/judge.py で行うこと。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
