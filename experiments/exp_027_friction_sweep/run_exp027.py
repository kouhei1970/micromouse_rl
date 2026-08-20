"""
experiments/exp_027_friction_sweep/run_exp027.py
================
exp_027（摩擦円の使用率 u の掃引と、壊れずに走り切れる上限の測定）の
一次記録を測る測定スクリプト。

🔴 本スクリプトは合否を一切判定しない。判定は
`experiments/exp_027_friction_sweep/judge.py` が行う（`PREREG.md` を見ること。
判定条文そのものは `research_notes/note_031_profile_planner_and_eta.md` を参照）。

`experiments/exp_026_profile_run/run_exp026.py` の構造（対象迷路の選定・
multiprocessing による並列実行・`--serial`/`--dry-run`）を踏襲するが、次を変える:

  - 条件: `ClassicExplorerPolicy(fast_mode="profile", friction_use=u,
    wall_correction=wc)`。`u ∈ {0.50,0.60,0.70,0.80,0.90,1.00}`（6水準）×
    `wc ∈ {False, True}`（2条件）＝12条件×10迷路＝120走行（PREREG §3）。
  - 一次記録には常に（`wc` に関わらず）方策が学習した地図
    （`policy._explorer.maze.v_walls`/`h_walls`。真値ではない）と、その走行が
    使った `friction_use`/`wall_correction` を書き足す（PREREG §4・§5。
    `judge.py` が同じ `u` で `t_plan` を独立に再計算するのに使う）。

🔴 測定は前景で完走させること（バックグラウンドへ投げてターンを終えると
プロセスが落ちる）。120走行は1回のコマンドで回すには長くなりうるので、
`--only-u`/`--only-wall-correction` で条件ごとに分割し、複数回に分けて
逐次・前景で実行できるようにしてある。

使い方:
    # 対象10迷路×12条件の一覧の確認だけ（測定は1本も走らせない）
    .venv/bin/python experiments/exp_027_friction_sweep/run_exp027.py --dry-run

    # u=1.00 の2条件(wc off/on)×10迷路=20走行だけを実行
    .venv/bin/python experiments/exp_027_friction_sweep/run_exp027.py --only-u 1.00

    # wall_correction=False の6水準×10迷路=60走行だけを実行
    .venv/bin/python experiments/exp_027_friction_sweep/run_exp027.py --only-wall-correction off

    # 全120走行を一度に実行（並列・時間がかかる）
    .venv/bin/python experiments/exp_027_friction_sweep/run_exp027.py
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
OUT_ROOT = REPO_ROOT / "outputs" / "exp_027"

TIME_BUDGET_S = 1500.0
MAX_RUNS = 5

# PREREG §3: 摩擦円の使用率 u の6水準・壁センサ補正の2条件。
U_LEVELS: List[float] = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
WALL_CORRECTION_LEVELS: List[bool] = [False, True]

# 評価用に予約された seed 範囲（docs/RESEARCH_PLAN.md §2・§9-7 が正）。
# ここでは「選んでしまっていないか」を機械的に確かめる安全弁としてのみ使う
# （判定条文ではない。exp_024/exp_025/exp_026 の同名チェックと同じ作法）。
RESERVED_SEED_RANGE = (1000, 40999)


def _u_dirname(u: float) -> str:
    return f"u_{u:.2f}"


def _wc_dirname(wc: bool) -> str:
    return "wc_on" if wc else "wc_off"


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
# 1 (迷路, u, wall_correction) の測定（子プロセスで実行される）
# ==========================================================================
def _run_one(task: Dict) -> Dict:
    """1 (迷路, u, wall_correction) の測定を行う。multiprocessing の子プロセス
    から呼ばれることを想定し、重い依存（mujoco 経由の competition/classic）は
    ここで初めて import する（`--dry-run` を軽く保つため、モジュール先頭では
    import しない）。
    """
    from classic.policy import ClassicExplorerPolicy
    from competition.evaluator import CompetitionEvaluator

    seed = task["seed"]
    d0 = task["d0"]
    u = task["u"]
    wc = task["wall_correction"]
    maze_dir = Path(task["maze_dir"])
    out_path = Path(task["out_path"])

    npz_path = maze_dir / f"maze_{seed}.npz"

    t0 = time.time()
    evaluator = CompetitionEvaluator(
        maze_dir=str(maze_dir),
        time_budget=TIME_BUDGET_S,
        max_runs=MAX_RUNS,
    )
    policy = ClassicExplorerPolicy(fast_mode="profile", friction_use=u, wall_correction=wc)
    result = evaluator.evaluate_maze(npz_path, policy)

    # PREREG §4: 一次記録は評価器の結果 dict そのままに、走行の一次記録
    # （集約値ではない）である plan_ids・run_phases・条件値2つを足したもの。
    result["plan_ids"] = policy.get_plan_ids()
    result["run_phases"] = policy.get_run_phases()
    result["friction_use"] = u
    result["wall_correction"] = wc

    # PREREG §5: 常に（wcに関わらず）方策が学習した地図（真値ではない）を
    # 一次記録へ書き足す（judge.py が同じuでt_planを独立に再計算するのに使う）。
    # 🔴 読み取り専用の後処理（act()/tick() の電圧計算には一切関与しない）。
    if getattr(policy, "_explorer", None) is not None:
        maze = policy._explorer.maze
        result["maze_v_walls"] = maze.v_walls.tolist()
        result["maze_h_walls"] = maze.h_walls.tolist()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    elapsed_s = time.time() - t0
    outcomes = [r["outcome"] for r in result["runs"]]
    return {
        "seed": seed, "d0": d0, "u": u, "wall_correction": wc,
        "outcomes": outcomes, "elapsed_s": elapsed_s,
        "out_path": str(out_path),
    }


def _build_tasks(target_mazes: List[Dict], maze_dir: Path,
                  u_levels: List[float], wc_levels: List[bool]) -> List[Dict]:
    tasks = []
    for u in u_levels:
        for wc in wc_levels:
            for m in target_mazes:
                out_path = OUT_ROOT / _u_dirname(u) / _wc_dirname(wc) / f"maze_{m['seed']}.json"
                tasks.append({
                    "seed": m["seed"], "d0": m["d0"],
                    "u": u, "wall_correction": wc,
                    "maze_dir": str(maze_dir), "out_path": str(out_path),
                })
    return tasks


def _print_progress(done_idx: int, total: int, r: Dict) -> None:
    print(
        f"[{done_idx}/{total}] maze={r['seed']:>5} (D0={r['d0']:>3}) "
        f"u={r['u']:.2f} wc={'on ' if r['wall_correction'] else 'off'} "
        f"outcomes={r['outcomes']} elapsed={r['elapsed_s']:.1f}s -> {r['out_path']}",
        flush=True,
    )


def _parse_wc_arg(value: str) -> bool:
    v = value.strip().lower()
    if v in ("on", "true", "1", "yes"):
        return True
    if v in ("off", "false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"on/off のいずれかで指定してください: {value!r}")


# ==========================================================================
# メイン
# ==========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--serial", action="store_true",
                         help="逐次実行に切り替える（既定は multiprocessing による並列実行）")
    parser.add_argument("--workers", type=int, default=None,
                         help="並列数（既定: min(16, os.cpu_count()-2)）")
    parser.add_argument("--dry-run", action="store_true",
                         help="対象10迷路×12条件の一覧の印字だけで終わる（測定は1本も走らせない）")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH,
                         help="manifest.jsonのパス（既定: design_turn_v1）")
    parser.add_argument("--only-seed", type=int, default=None,
                         help="指定したseedの迷路だけを測定する")
    parser.add_argument("--only-u", type=float, default=None,
                         help="指定した u の水準だけを測定する（他の水準はスキップ。"
                              "1回のコマンド実行を短く保つための分割実行用）")
    parser.add_argument("--only-wall-correction", type=_parse_wc_arg, default=None,
                         help="on/off のいずれかで、指定した wall_correction 条件だけを測定する")
    args = parser.parse_args(argv)

    maze_dir = args.manifest.parent
    target_mazes = select_target_mazes(args.manifest)
    if args.only_seed is not None:
        target_mazes = [m for m in target_mazes if m["seed"] == args.only_seed]
        if not target_mazes:
            raise SystemExit(f"--only-seed={args.only_seed} は対象10迷路に含まれない")

    u_levels = U_LEVELS if args.only_u is None else [args.only_u]
    if args.only_u is not None and args.only_u not in U_LEVELS:
        print(f"[警告] --only-u={args.only_u} はPREREGの6水準{U_LEVELS}に含まれない"
              "（それでも測定は行う。掃引外の値を明示的に指定した場合の用途）。")
    wc_levels = WALL_CORRECTION_LEVELS if args.only_wall_correction is None else [args.only_wall_correction]

    print(f"対象 {len(target_mazes)} 迷路（seed昇順、manifest: {args.manifest}）:")
    for i, m in enumerate(target_mazes, 1):
        print(f"  {i}. seed={m['seed']} D0={m['d0']}")
    print(f"u水準: {u_levels}")
    print(f"wall_correction条件: {wc_levels}")

    if args.dry_run:
        print("\n--dry-run: 測定は実行しない（対象条件の確認のみ）。")
        return 0

    tasks = _build_tasks(target_mazes, maze_dir, u_levels, wc_levels)
    total = len(tasks)
    print(f"\n{total} 本（{len(target_mazes)} 迷路 × {len(u_levels)} u水準 × "
          f"{len(wc_levels)} wc条件）を{'逐次' if args.serial else '並列'}実行する。")

    wall_clock_start = time.time()
    results: List[Dict] = []

    if args.serial:
        for i, task in enumerate(tasks, 1):
            r = _run_one(task)
            results.append(r)
            _print_progress(i, total, r)
    else:
        n_workers = args.workers if args.workers is not None else max(1, min(16, (os.cpu_count() or 2) - 2))
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
          "experiments/exp_027_friction_sweep/judge.py で行うこと。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
