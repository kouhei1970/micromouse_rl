"""
experiments/exp_030_map_quality/run_exp030.py
================
exp_030（探索後の地図がどれだけ正しいかを測る）の一次記録を測る測定スクリプト。

背景（`research_notes/note_033_exploration_first.md`）: 工程を組み直し、第 1 工程を
「探索の信頼度を上げる」に定めた。出口条件は 10 迷路すべてで「致命的な誤り = 0」。
本スクリプトはその出発点として、現状の探索がどれだけ正しい地図を作れているかを
そのまま測る（探索の実装は一切変えない）。

🔴 本スクリプトは `ClassicExplorerPolicy`（`requires_privileged = False`）だけを
動かす。真の壁配列（npz）は一切読まない。真の地図を使った判定は
`judge.py` の中だけで行う（`note_033` §「地図の正しさ」）。

条件（任務指示どおり）:
  - 対象: `design_turn_v1` の10迷路
  - `CompetitionEvaluator(time_budget=1500.0, max_runs=5)`
  - `ClassicExplorerPolicy(fast_mode="command")`（既定のコマンド方式。探索は現状のまま）

保存するもの（迷路ごと1 JSON）:
  - result: 評価器の結果 dict（runs/incidents/best_time など）
  - v_walls_known / h_walls_known: 探索走行終了時点で方策が持っていた地図
    （`WallState`: 0=未知, 1=壁あり, 2=壁なし。`classic/maze_map.py` の値規約）。
    最短走行(FAST/RETURN2)は地図を書き換えない設計（`classic/explorer.py`
    モジュール docstring 「S3: 最短走行」）なので、評価終了時点の値は
    「探索完了時点の地図」と同一である。
  - explore_time_s: 最初にゴールへ到達した走行(outcome=="goal")の t_end
    （絶対シミュレーション時刻 [s]）。1本もゴールしなければ None。
  - run_phases: 走行が始まった瞬間の段階の列（診断用）。

🔴 測定は前景で完走させること（バックグラウンドへ投げてターンを終えると
プロセスが落ちる。exp_028/exp_029 までに5回起きている事故）。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MAZE_DIR = REPO_ROOT / "competition" / "mazes" / "design_turn_v1"
MANIFEST_PATH = MAZE_DIR / "manifest.json"
OUT_ROOT = REPO_ROOT / "outputs" / "exp_030_map_quality"

TIME_BUDGET_S = 1500.0
MAX_RUNS = 5
FAST_MODE = "command"  # 任務指示: 探索は現状のまま（既定のコマンド方式）

# 評価用に予約された seed 範囲（docs/RESEARCH_PLAN.md §2・§9-7 が正）。
RESERVED_SEED_RANGE = (1000, 40999)


# ==========================================================================
# 対象迷路の選定（exp_029 と同じ作法。結果に依存しない・seedを直書きしない）
# ==========================================================================
def select_target_mazes(manifest_path: Path = MANIFEST_PATH) -> List[Dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ordered = sorted(manifest["mazes"], key=lambda m: int(m["seed"]))
    selected = [{"seed": int(m["seed"]), "d0": int(m["d0"])} for m in ordered]

    lo, hi = RESERVED_SEED_RANGE
    for m in selected:
        if lo <= m["seed"] <= hi:
            raise RuntimeError(
                f"評価用に予約された seed 範囲 [{lo}, {hi}] に含まれる seed {m['seed']} を"
                f"選んでしまった（測定禁止）。"
            )
    return selected


def _make_policy():
    from classic.policy import ClassicExplorerPolicy
    return ClassicExplorerPolicy(fast_mode=FAST_MODE)


def _explore_time_s(result: Dict) -> Optional[float]:
    """最初にゴールへ到達した走行の t_end（絶対シミュレーション時刻）。
    1本もゴールしなければ None（=持ち時間内に探索が終わらなかった）。"""
    for run in result["runs"]:
        if run["outcome"] == "goal":
            return float(run["t_end"])
    return None


# ==========================================================================
# 1 迷路の測定
# ==========================================================================
def measure_one(task: Dict) -> Dict:
    from competition.evaluator import CompetitionEvaluator

    seed = task["seed"]
    maze_dir = Path(task["maze_dir"])
    out_path = Path(task["out_path"])

    ev = CompetitionEvaluator(maze_dir=str(maze_dir), time_budget=TIME_BUDGET_S, max_runs=MAX_RUNS)
    policy = _make_policy()

    t_wall0 = time.time()
    result = ev.evaluate_maze(maze_dir / f"maze_{seed}.npz", policy)
    wall_clock_s = time.time() - t_wall0

    record = {
        "seed": seed,
        "time_budget": TIME_BUDGET_S,
        "max_runs": MAX_RUNS,
        "fast_mode": FAST_MODE,
        "wall_clock_s": wall_clock_s,
        "result": result,
        "explore_time_s": _explore_time_s(result),
        "run_phases": policy.get_run_phases(),
        "v_walls_known": policy.v_walls_known.tolist(),
        "h_walls_known": policy.h_walls_known.tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return record


def _build_tasks(target_mazes: List[Dict], maze_dir: Path) -> List[Dict]:
    tasks = []
    for m in target_mazes:
        out_path = OUT_ROOT / f"maze_{m['seed']}.json"
        tasks.append({"seed": m["seed"], "maze_dir": str(maze_dir), "out_path": str(out_path)})
    return tasks


def _print_progress(i: int, total: int, r: Dict, t0: float) -> None:
    runs = r["result"]["runs"]
    n_goal = sum(1 for x in runs if x["outcome"] == "goal")
    best = r["result"]["best_time"]
    et = r["explore_time_s"]
    print(f"[{i}/{total}] seed={r['seed']:>5} n_runs={len(runs):>2} n_goal={n_goal} "
          f"explore_time={f'{et:.2f}s' if et is not None else '-':>9} "
          f"best_time={f'{best:.3f}s' if best else '-':>9} "
          f"wall_clock={r['wall_clock_s']:6.1f}s "
          f"elapsed={time.time()-t0:7.1f}s", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                         help="対象条件の一覧の印字だけで終わる（測定は1本も走らせない）")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--only-seed", type=int, default=None)
    parser.add_argument("--serial", action="store_true",
                         help="逐次実行に切り替える（既定は multiprocessing による並列実行）")
    parser.add_argument("--workers", type=int, default=None,
                         help="並列数（既定: min(10, os.cpu_count()-2)）")
    args = parser.parse_args(argv)

    maze_dir = args.manifest.parent
    target_mazes = select_target_mazes(args.manifest)
    if args.only_seed is not None:
        target_mazes = [m for m in target_mazes if m["seed"] == args.only_seed]

    print(f"対象 {len(target_mazes)} 迷路（manifest: {args.manifest}）")
    print(f"持ち時間={TIME_BUDGET_S}s max_runs={MAX_RUNS} fast_mode={FAST_MODE}")

    if args.dry_run:
        print("\n--dry-run: 測定は実行しない。")
        return 0

    tasks = _build_tasks(target_mazes, maze_dir)
    total = len(tasks)
    mode = "逐次" if args.serial else "並列"
    print(f"\n{total} 迷路を{mode}・前景で実行する（最大{MAX_RUNS}走行/迷路）。")

    t0 = time.time()
    if args.serial:
        for i, task in enumerate(tasks, 1):
            r = measure_one(task)
            _print_progress(i, total, r, t0)
    else:
        n_workers = args.workers if args.workers is not None else max(1, min(10, (os.cpu_count() or 2) - 2))
        n_workers = min(n_workers, total)
        print(f"並列数: {n_workers}")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(measure_one, tasks), 1):
                _print_progress(i, total, r, t0)

    print(f"\n完了: {total} 迷路 / 実時間 {time.time()-t0:.1f}s")
    print("🔴 地図の正しさの判定はここでは行わない。"
          "experiments/exp_030_map_quality/judge.py で行うこと。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
