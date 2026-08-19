"""
experiments/exp_025_s4_slalom/run_exp025.py
================
exp_025（段階 S4「スラローム」の完了判定）の一次記録を測る測定スクリプト。

🔴 本スクリプトは合否を一切判定しない。判定は
`experiments/exp_025_s4_slalom/recompute_anchors.py` と教授セッションが行う
（`PREREG.md` を見ること。条文は本スクリプトにもここにも写さない）。

`experiments/exp_024_s3_fast_run/run_exp024.py` の構造を踏襲するが、次を変える
（PREREG §3・§4・追記 1）:

  - 対象迷路: `competition/mazes/design_turn_v1/manifest.json` の **全 10 迷路**を
    seed 昇順で読む（exp_024 は D0 昇順で上位 6 迷路のみだった）。
  - 条件: `ClassicExplorerPolicy(slalom=..., extend_straights=True)`。
    `extend_straights` は両条件とも True で固定する（本実験が動かすのは
    `slalom` だけ。PREREG §2「1 実験 1 変更」）。
  - 持ち時間: `CompetitionEvaluator(time_budget=1500.0, max_runs=5)`
    （PREREG 追記 1。競技の 420 秒ではなく、探索・帰還が持ち時間切れで
    最短走行の標本を失わないよう広く取る）。

投入前の自己検査（PREREG 追記 1 の🔴・§9 A3）を、測定を 1 本でも投入する前に
**必ず**実行する。どちらか一方でも落ちたら測定を始めない（`SystemExit`）:

  1. `ClassicExplorerPolicy` とその依存（`classic/explorer.py`）が
     `time_budget`（`budget`/`time_left`/`remaining_time` も含む）を読んで
     いないこと。追記 1 の「持ち時間を 1500 秒に広げても探索・帰還の挙動は
     変わらない」という前提を機械的に検査する。
  2. `experiments/exp_025_s4_slalom/geometry_anchor.py`（錨 A3・幾何余裕）を
     サブプロセスで実行し、終了コードが 0 であること。

使い方:
    .venv/bin/python experiments/exp_025_s4_slalom/run_exp025.py            # 並列実行
    .venv/bin/python experiments/exp_025_s4_slalom/run_exp025.py --serial   # 逐次実行
    .venv/bin/python experiments/exp_025_s4_slalom/run_exp025.py --dry-run  # 対象 10 迷路の
        seed 一覧と、投入前の自己検査 2 件だけを実行して終わる（測定は 1 本も走らせない）

🔴 測定は前景で完走させること（バックグラウンドへ投げてターンを終えると
プロセスが落ちる）。
"""
from __future__ import annotations

import argparse
import inspect
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MANIFEST_PATH = REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "manifest.json"
OUT_ROOT = REPO_ROOT / "outputs" / "exp_025_s4"
GEOMETRY_ANCHOR_PATH = Path(__file__).resolve().parent / "geometry_anchor.py"

TIME_BUDGET_S = 1500.0
MAX_RUNS = 5

# 条件キー → ClassicExplorerPolicy(slalom=...) の値（PREREG §3 CONDITIONS）。
# extend_straights は両条件とも True で固定する（本実験が動かすのは slalom だけ）。
CONDITIONS: List[tuple] = [("slalom", True), ("control", False)]
EXTEND_STRAIGHTS = True

# 評価用に予約された seed 範囲（docs/RESEARCH_PLAN.md §2・§9-7 が正）。
# ここでは「選んでしまっていないか」を機械的に確かめる安全弁としてのみ使う
# （判定条文ではない。exp_024 run_exp024.py の同名チェックと同じ作法）。
RESERVED_SEED_RANGE = (1000, 40999)

# 自己検査 1（PREREG 追記 1 の🔴）で禁止する語。
# `ClassicExplorerPolicy`/`ClassicExplorer` が持ち時間を読んでいたら、
# 追記 1 の「1500 秒に広げても探索・帰還はティック単位で同一」という前提が
# 崩れる。
FORBIDDEN_TOKENS = ("time_budget", "budget", "time_left", "remaining_time")


# ==========================================================================
# 対象迷路の選定（PREREG §3。結果に依存しない・seed を直書きしない）
# ==========================================================================
def select_target_mazes(manifest_path: Path = MANIFEST_PATH) -> List[Dict]:
    """`manifest.json` の全迷路を seed 昇順で返す（PREREG §3 は design_turn_v1
    の全 10 迷路を対象とする。exp_024 のような上位 n 件の絞り込みはしない）。

    Returns:
        [{"seed": int, "d0": int}, ...] （seed 昇順）
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
# 投入前の自己検査 1: time_budget を読んでいないこと（PREREG 追記 1 の🔴）
# ==========================================================================
def check_time_budget_not_read() -> None:
    """`classic/policy.py`・`classic/explorer.py` の全文（`inspect.getsource`）に
    `FORBIDDEN_TOKENS` のいずれかが現れないことを検査する。

    現れたら追記 1 の前提（持ち時間を 1500 秒へ広げても探索・帰還の挙動は
    変わらない）が崩れているので、`SystemExit` で理由を印字して測定を止める
    （黙って続けない）。
    """
    import classic.explorer
    import classic.policy

    modules = {"classic/policy.py": classic.policy, "classic/explorer.py": classic.explorer}
    violations: List[str] = []
    for label, module in modules.items():
        src = inspect.getsource(module)
        for token in FORBIDDEN_TOKENS:
            if token in src:
                violations.append(f"{label} に禁止語 '{token}' が見つかった")

    if violations:
        raise SystemExit(
            "🔴 投入前の自己検査 1 に失敗した（PREREG 追記 1 の前提が崩れている）:\n  "
            + "\n  ".join(violations)
            + "\n`ClassicExplorerPolicy`/`ClassicExplorer` が持ち時間を読んでいる可能性がある。"
              "この状態で time_budget=1500.0 に変えると探索・帰還の挙動が変わりうるため、"
              "測定を開始しない。"
        )
    print("[自己検査 1/2] time_budget 系の語を classic/policy.py・classic/explorer.py で検出せず: OK")


# ==========================================================================
# 投入前の自己検査 2: 錨 A3（幾何余裕）をサブプロセスで実行し終了コード 0 を確認
# ==========================================================================
def check_geometry_anchor() -> None:
    """`geometry_anchor.py` をサブプロセスで実行し、終了コードが 0 であることを
    確認する。0 でなければ `SystemExit` で理由を印字して測定を止める。"""
    result = subprocess.run(
        [sys.executable, str(GEOMETRY_ANCHOR_PATH)],
        capture_output=True, text=True,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        raise SystemExit(
            f"🔴 投入前の自己検査 2 に失敗した（錨 A3・幾何余裕、終了コード {result.returncode}）:\n"
            f"{result.stderr}\n"
            "幾何の余裕が負（投入不可）である可能性がある。測定を開始しない。"
        )
    print("[自己検査 2/2] geometry_anchor.py 終了コード 0: OK")


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
    slalom = task["slalom"]
    maze_dir = Path(task["maze_dir"])
    out_path = Path(task["out_path"])

    npz_path = maze_dir / f"maze_{seed}.npz"

    t0 = time.time()
    evaluator = CompetitionEvaluator(
        maze_dir=str(maze_dir),
        time_budget=TIME_BUDGET_S,
        max_runs=MAX_RUNS,
    )
    # 🔴 slalom=... が TypeError になる場合、それは classic/ 側の実装待ちである
    # ことを意味する（本スクリプトの誤りではない）。ここで例外を飲み込んで
    # 回避策（例: extend_straights のみで代替する等）を入れてはならない。
    policy = ClassicExplorerPolicy(slalom=slalom, extend_straights=EXTEND_STRAIGHTS)
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
        for condition, slalom in CONDITIONS:
            out_path = OUT_ROOT / condition / f"maze_{m['seed']}.json"
            tasks.append({
                "seed": m["seed"], "d0": m["d0"],
                "condition": condition, "slalom": slalom,
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
                         help="対象 10 迷路の seed・D0 の印字と投入前の自己検査 2 件だけを実行して"
                              "終わる（測定は 1 本も走らせない）")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH,
                         help="manifest.json のパス（既定: design_turn_v1）")
    args = parser.parse_args(argv)

    maze_dir = args.manifest.parent
    target_mazes = select_target_mazes(args.manifest)

    print(f"対象 {len(target_mazes)} 迷路（seed 昇順、manifest: {args.manifest}）:")
    for i, m in enumerate(target_mazes, 1):
        print(f"  {i}. seed={m['seed']} D0={m['d0']}")

    # 投入前の自己検査（PREREG 追記 1 の🔴・§9 A3）。--dry-run でも実行する
    # （手順の確認に自己検査そのものを含む）。どちらか一方でも落ちたら
    # 測定を始めない。
    print()
    check_time_budget_not_read()
    check_geometry_anchor()

    if args.dry_run:
        print("\n--dry-run: 測定は実行しない（対象迷路の確認と自己検査 2 件のみ）。")
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
          "experiments/exp_025_s4_slalom/recompute_anchors.py で行うこと。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
