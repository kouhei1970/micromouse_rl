"""
experiments/exp_029_full_protocol/run_exp029.py
================
exp_029（競技の全手順を通してηを測る）の一次記録を測る測定スクリプト。

背景（任務指示 2026-08-20/21）: exp_028（`research_notes/note_031` §exp_028）は
真の壁をそのまま「既知の地図」として最短走行だけを直接動かしており、
`η_map`（地図と経路選択の質）を含まない。本実験は `competition/evaluator.py`
の `CompetitionEvaluator` を通し、探索→帰還→最短走行の全手順で走らせて
その差（η_map）を測る。

条件（exp_028 で初めて完走が出た組み合わせをそのまま使う）:
  - 対象: `design_turn_v1` の10迷路
  - 余裕 `clearance_margin_m` 25mm・摩擦円の使用率 `u` 0.30・壁センサ補正あり
  - `fast_mode="profile"`（速度プロファイル追従。exp_028 と同一）
  - 持ち時間: 420秒（競技規約どおり）・1500秒（探索・帰還の遅さで420秒だと
    最短走行の標本が枯れる懸念があるため）の2通り
  - `max_runs = 5`

方策は `classic/policy.py::ClassicExplorerPolicy`（`requires_privileged=False`。
評価器は bind_sim/bind_maze を呼ばない＝センサ観測だけで走る）をそのまま使う。
`RecordingPolicy` はその薄いサブクラスで、`on_run_start` の瞬間に
Phase.FAST の速度プロファイル計画（`ex._fast_plan.t_plan` ==
マウス自身が学習した地図から `classic/fast_planner.py::plan_fast_run` が
計算した理論値）を run_index ごとに記録するだけであり、`act()`（電圧計算）
には一切触れない。地図は EXPLORE/RETURN 中にしか更新されない
（`classic/explorer.py` モジュール docstring）ので、FAST 突入後は
何回ループしても同じ地図から同じ t_plan が出るはずである
（judge.py はこれを検算しない。複数回記録された場合は
T_measured を出した run_index の値をそのまま使う）。

🔴 測定は前景で完走させること（バックグラウンドへ投げてターンを終えると
プロセスが落ちる。exp_028 までに5回起きている事故）。
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
OUT_ROOT = REPO_ROOT / "outputs" / "exp_029"

# 任務指示: 1500秒を先に測る（探索・帰還の遅さで420秒だと最短走行の標本が
# 枯れる懸念があるため）。
TIME_BUDGETS_S: List[float] = [1500.0, 420.0]
MAX_RUNS = 5
# exp_028（`research_notes/note_031` §exp_028）で初めて完走が出た組み合わせ。
FRICTION_USE = 0.30
CLEARANCE_MARGIN_M = 0.025
WALL_CORRECTION = True
FAST_MODE = "profile"

# 評価用に予約された seed 範囲（docs/RESEARCH_PLAN.md §2・§9-7 が正）。
RESERVED_SEED_RANGE = (1000, 40999)


def _tb_dirname(time_budget: float) -> str:
    return f"tb_{int(round(time_budget)):04d}s"


# ==========================================================================
# 対象迷路の選定（exp_028 と同じ作法。結果に依存しない・seedを直書きしない）
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


# ==========================================================================
# 方策: ClassicExplorerPolicy + FAST計画(t_plan)のrun_indexごとの記録
# ==========================================================================
def _make_policy():
    from classic.explorer import Phase
    from classic.policy import ClassicExplorerPolicy

    class RecordingPolicy(ClassicExplorerPolicy):
        """`on_run_start` の瞬間、Phase.FAST に入っていれば `ex._fast_plan.t_plan`
        （学習した地図から計算された速度プロファイル計画の理論時間）を
        run_index をキーに記録する。既存の `on_run_start`/`act` の呼び出しは
        そのまま素通しするだけで、電圧計算・状態機械には一切触れない。"""

        def __init__(self, *a, **kw) -> None:
            super().__init__(*a, **kw)
            self.fast_plan_by_run: Dict[int, float] = {}

        def on_run_start(self, run_index: int) -> None:
            super().on_run_start(run_index)
            ex = self._explorer
            if ex is not None and ex.phase is Phase.FAST and ex._fast_plan is not None:
                self.fast_plan_by_run[int(run_index)] = float(ex._fast_plan.t_plan)

    return RecordingPolicy(
        fast_mode=FAST_MODE,
        friction_use=FRICTION_USE,
        clearance_margin_m=CLEARANCE_MARGIN_M,
        wall_correction=WALL_CORRECTION,
    )


# ==========================================================================
# 1 (迷路, 持ち時間) の測定（競技の全手順: 探索→帰還→最短走行、最大5走行）
# ==========================================================================
def measure_one(task: Dict) -> Dict:
    from competition.evaluator import CompetitionEvaluator

    seed = task["seed"]
    time_budget = task["time_budget"]
    maze_dir = Path(task["maze_dir"])
    out_path = Path(task["out_path"])

    ev = CompetitionEvaluator(maze_dir=str(maze_dir), time_budget=time_budget, max_runs=MAX_RUNS)
    policy = _make_policy()

    t_wall0 = time.time()
    result = ev.evaluate_maze(maze_dir / f"maze_{seed}.npz", policy)
    wall_clock_s = time.time() - t_wall0

    record = {
        "seed": seed,
        "time_budget": time_budget,
        "max_runs": MAX_RUNS,
        "friction_use": FRICTION_USE,
        "clearance_margin_m": CLEARANCE_MARGIN_M,
        "wall_correction": WALL_CORRECTION,
        "fast_mode": FAST_MODE,
        "wall_clock_s": wall_clock_s,
        "result": result,
        "fast_plan_by_run": policy.fast_plan_by_run,
        "run_phases": policy.get_run_phases(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return record


def _build_tasks(target_mazes: List[Dict], maze_dir: Path,
                  time_budgets: List[float]) -> List[Dict]:
    tasks = []
    for tb in time_budgets:
        for m in target_mazes:
            out_path = OUT_ROOT / _tb_dirname(tb) / f"maze_{m['seed']}.json"
            tasks.append({
                "seed": m["seed"], "time_budget": tb,
                "maze_dir": str(maze_dir), "out_path": str(out_path),
            })
    return tasks


def _print_progress(i: int, total: int, r: Dict, t0: float) -> None:
    runs = r["result"]["runs"]
    n_goal = sum(1 for x in runs if x["outcome"] == "goal")
    best = r["result"]["best_time"]
    print(f"[{i}/{total}] seed={r['seed']:>5} tb={r['time_budget']:>6.0f}s "
          f"n_runs={len(runs):>2} n_goal={n_goal} "
          f"best_time={f'{best:.3f}s' if best else '-':>9} "
          f"wall_clock={r['wall_clock_s']:6.1f}s "
          f"elapsed={time.time()-t0:7.1f}s", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                         help="対象条件の一覧の印字だけで終わる（測定は1本も走らせない）")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--only-seed", type=int, default=None)
    parser.add_argument("--only-time-budget", type=float, default=None,
                         help="指定した持ち時間[s]の水準だけを測定する（分割実行用）")
    parser.add_argument("--serial", action="store_true",
                         help="逐次実行に切り替える（既定は multiprocessing による並列実行）")
    parser.add_argument("--workers", type=int, default=None,
                         help="並列数（既定: min(20, os.cpu_count()-2)）")
    args = parser.parse_args(argv)

    maze_dir = args.manifest.parent
    target_mazes = select_target_mazes(args.manifest)
    if args.only_seed is not None:
        target_mazes = [m for m in target_mazes if m["seed"] == args.only_seed]

    time_budgets = TIME_BUDGETS_S if args.only_time_budget is None else [args.only_time_budget]

    print(f"対象 {len(target_mazes)} 迷路（manifest: {args.manifest}）")
    print(f"持ち時間水準[s]: {time_budgets}")
    print(f"max_runs={MAX_RUNS} friction_use={FRICTION_USE} "
          f"clearance_margin_m={CLEARANCE_MARGIN_M} wall_correction={WALL_CORRECTION} "
          f"fast_mode={FAST_MODE}")

    if args.dry_run:
        print("\n--dry-run: 測定は実行しない。")
        return 0

    tasks = _build_tasks(target_mazes, maze_dir, time_budgets)
    total = len(tasks)
    mode = "逐次" if args.serial else "並列"
    print(f"\n{total} 条件（{len(target_mazes)}迷路 × {len(time_budgets)}持ち時間水準）を"
          f"{mode}・前景で実行する（探索→帰還→最短走行の全手順、迷路ごと最大{MAX_RUNS}走行）。")

    t0 = time.time()
    if args.serial:
        for i, task in enumerate(tasks, 1):
            r = measure_one(task)
            _print_progress(i, total, r, t0)
    else:
        n_workers = args.workers if args.workers is not None else max(1, min(20, (os.cpu_count() or 2) - 2))
        n_workers = min(n_workers, total)
        print(f"並列数: {n_workers}")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(measure_one, tasks), 1):
                _print_progress(i, total, r, t0)

    print(f"\n完了: {total} 条件 / 実時間 {time.time()-t0:.1f}s")
    print("🔴 η の判定はここでは行わない。"
          "experiments/exp_029_full_protocol/judge.py で行うこと。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
