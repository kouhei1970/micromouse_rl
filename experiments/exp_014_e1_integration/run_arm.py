#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_013 の 1 腕を走らせ、走行ごとの指標を**規約どおりに**保存する。

**なぜ評価器 CLI をそのまま使わないのか**

凍結評価ハーネスが走行ごとに残すのは
`run_time / path_length_m / visited_cells / path_efficiency` までで、
**研究計画書 §2 が要求する 2 つが取れない**:

  - **(e') 初回最短走行の経路効率** ＝ 通過**区画数** ÷ $D_{true}$。
    評価器の `path_efficiency` は**距離 (m) で正規化**したもので、§2 は
    **「距離で正規化してはならない」**と明記している（旋回で内側を回るぶん、
    完璧な走行でも 1.00 にならず、経路の良し悪しではなく旋回の多さを測ってしまう）
  - **`n_turns`**（裁定 R4 の正本定義＝**区画列の進行方向変化**。180° = 2）

**そこで凍結ハーネスには一切触れず、方策を包んで軌跡を記録する。**
記録は読むだけで方策の出力に影響しない（同一条件の再走行で run_time が
exp_007 と絶対差 0.0000 s で一致することを 2026-08-11 に確認済み）。

保存するもの（§9-3「走行ごとの指標を全件保存」）:
  - 評価器の出力 JSON（そのまま。out-dir 直下）
  - **走行ごと**の (歩数, 旋回数, 実走距離, 平均速度, (e') の分子) — `runs_detail.json`
  - **軌跡**（走行境界つき）— `traj/<maze>.npz`

使い方:
    .venv/bin/python experiments/exp_014_e1_integration/run_arm.py \
        --arm l0a_e1 --policy competition.baseline_classical:AdachiPolicy
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.seed_bands import (assert_seeds_allowed,  # noqa: E402
                                describe_seeds)
from competition.evaluator import CompetitionEvaluator  # noqa: E402
from competition.explore_cost import true_shortest  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

OUT_ROOT = REPO_ROOT / "outputs" / "exp_014_e1_integration"


class PoseProbe:
    """方策を包み、制御周期ごとに (時刻, x, y, ヨー角) を記録する（読むだけ）。"""

    def __init__(self, inner):
        self._inner = inner
        self._sim = None
        self.rec = []

    name = property(lambda self: getattr(self._inner, "name", "unnamed"))
    requires_privileged = property(lambda self: getattr(self._inner, "requires_privileged", False))

    def bind_sim(self, sim):
        self._sim = sim
        return self._inner.bind_sim(sim)

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def act(self, obs):
        out = self._inner.act(obs)
        if self._sim is not None:
            x, y, yaw = self._sim.privileged_pose()
            self.rec.append((self._sim.sim_time, x, y, yaw))
        return out


def cells_and_turns(seg, cell):
    """【裁定 R4 の正本定義】区画列から (歩数, 旋回数)。180° 反転は 2。"""
    seq = []
    for _, x, y, _ in seg:
        c = (int(x // cell), int(y // cell))
        if not seq or seq[-1] != c:
            seq.append(c)
    if len(seq) < 2:
        return max(len(seq) - 1, 0), 0
    order = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}
    dirs = [(int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))) for a, b in zip(seq, seq[1:])]
    turns = 0
    for d1, d2 in zip(dirs, dirs[1:]):
        if d1 in order and d2 in order:
            k1, k2 = order[d1], order[d2]
            turns += min((k2 - k1) % 4, (k1 - k2) % 4)
    return len(seq) - 1, turns


def load_policy(spec):
    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls)()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--maze-dir", default="competition/mazes/eval")
    ap.add_argument("--out-root", default=None, help="既定は outputs/exp_014_e1_integration")
    # --- 帯の安全弁（裁定 R40 条件 4。`common/seed_bands.py` の共通ヘルパ） ---
    ap.add_argument("--purpose", default="validate", choices=("train", "validate", "gate"),
                    help="既定 validate。**評価帯を使うには gate と --reason が要る**")
    ap.add_argument("--reason", default=None,
                    help="--purpose gate のときに必須（何の判定でこの帯を使うのか）")
    args = ap.parse_args()

    cell = RobotParams().cell_size
    out_dir = (Path(args.out_root) if args.out_root else OUT_ROOT) / args.arm
    traj_dir = out_dir / "traj"
    traj_dir.mkdir(parents=True, exist_ok=True)
    mazes = sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))
    # **帯を明示し、許されない帯なら走らせない**（裁定 R40 条件 4）。
    # ⚠️ **2026-08-14 時点の既知の制約**: `common/seed_bands.py` は competition の
    # 候補プール [1000,40999] を一律 'pool' と分類し、**どの purpose も 'pool' を
    # 許さない**。したがって**凍結帯の gate 走行（本トラックの最終 1 回）は
    # `--purpose gate --reason ...` を付けても現状では拒否される**。
    # 教授へ報告済み（採用 20 面を 'pool' と別の帯に分類するか、gate に 'pool' を
    # 許すかの裁定待ち）。**回避策は書かない** — 安全弁を骨抜きにしないため。
    # 規律を文書ではなく実装で保証する（§9-7）。凍結帯は
    # `--purpose gate --reason "…"` を明示しない限り拒否される。
    seeds = [int(m.stem.split("_")[1]) for m in mazes]
    print(describe_seeds(seeds, "competition"), flush=True)
    assert_seeds_allowed(seeds, namespace="competition",
                         purpose=args.purpose, reason=args.reason)
    print(f"[{args.arm}] {args.policy} / {len(mazes)} 面 / {args.maze_dir}"
          f" / purpose={args.purpose}", flush=True)

    detail = []
    for m in mazes:
        z = np.load(m)
        d_true = int(true_shortest(z["v_walls"], z["h_walls"]))
        probe = PoseProbe(load_policy(args.policy))
        ev = CompetitionEvaluator(maze_dir=args.maze_dir, out_dir=str(out_dir))
        r = ev.evaluate_maze(m, probe)

        rec = np.asarray(probe.rec, dtype=np.float64)
        np.savez_compressed(
            traj_dir / f"{r['maze_id']}.npz",
            t=rec[:, 0], x=rec[:, 1].astype(np.float32), y=rec[:, 2].astype(np.float32),
            yaw=rec[:, 3],
            run_index=np.array([q["index"] for q in r["runs"]], dtype=np.int32),
            run_t_start=np.array([q["t_start"] for q in r["runs"]], dtype=np.float64),
            run_t_end=np.array([q["t_end"] for q in r["runs"]], dtype=np.float64),
            run_outcome=np.array([q["outcome"] for q in r["runs"]]))

        for run in r["runs"]:
            seg = [s for s in probe.rec
                   if run["t_start"] - 1e-9 <= s[0] <= run["t_end"] + 1e-9]
            n_cells, n_turns = cells_and_turns(seg, cell)
            dist = float(run["path_length_m"])
            t = float(run["run_time"]) if run["run_time"] else None
            detail.append(dict(
                maze=r["maze_id"], d_true=d_true, run=run["index"],
                outcome=run["outcome"], run_time=t,
                # (e') の分子。**区画数**であって距離ではない（§2）
                n_cells=n_cells,
                # 裁定 R4 の正本定義
                n_turns=n_turns,
                path_length_m=dist,
                mean_speed=(dist / t) if (t and t > 0) else None,
                visited_cells=run["visited_cells"]))
        goal_runs = [q for q in r["runs"] if q["outcome"] == "goal"]
        print(f"  {r['maze_id']} D={d_true:3d} 走行 {len(r['runs'])} 本"
              f"（ゴール {len(goal_runs)}）best={r['best_time']}", flush=True)

    p = out_dir / "runs_detail.json"
    json.dump(dict(arm=args.arm, policy=args.policy, maze_dir=args.maze_dir,
                   cell_size=cell, n_turns_definition="R4: 区画列の進行方向変化 (180°=2)",
                   runs=detail), open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n走行ごと記録: {p}（{len(detail)} 走行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
