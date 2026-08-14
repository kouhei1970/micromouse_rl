#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**載せ替えカードの測定**（`card_016diag_switch.md` 段取り 3・5）。

**1 プロセス = 1 条件**にしてある（対照と処理を別に回せるようにするため）。

--------------------------------------------------------------------------
何を記録するか（**§5 の報告項目を後から全部作れるだけの生データ**）
--------------------------------------------------------------------------
| 記録 | 何のために |
|---|---|
| 評価器の走行ごとの出力 | **(a)(b)(c)(d)** |
| **`n_cells`（計時窓の移動回数）** | **(e′) 経路効率**（裁定 R14 で +1 する） |
| **制御周期ごとの (t, x, y, ヨー角, 速度, 角速度, 参照の曲率, 参照の方位)** | **P3（区間ごとの時間の割合）・P4（円弧の平均速度）** |

**⚠️ 計装は読むだけである**（方策を包んで記録するのみ。電圧も軌跡も変えない）。
**参照の曲率と方位は、方策が保持している経路をカーソル位置で読むだけ**で、
**走行後に区間（直進／斜め／円弧）を復元できる**:

- **円弧** = 参照の曲率が 0 でない
- **斜め** = 曲率が 0 かつ**参照の方位が 45° の奇数倍**
- **直進** = 曲率が 0 かつ**参照の方位が 90° の倍数**

**走行中には何も判定しない**（016-B から続く作法。計装が挙動を変えようがない形）。

--------------------------------------------------------------------------
定義は書き直さない（裁定 R23）
--------------------------------------------------------------------------
**`exp_013/run_arm.py` の `cells_and_turns`（裁定 R4 の定義）と
凍結ハーネスの `maze_kpi` を `importlib` で読み込んでそのまま呼ぶ。**
**`exp_013` 側は 1 行も変更しない。**

使い方:
    # 対照（斜めなしの新既定）
    .venv/bin/python -u experiments/exp_016_diagonal/run_016diag_switch.py \
        --arm control --policy competition.baseline_slalom_e1t_tr_f0b_cal:SlalomE1TTRF0bCalPolicy
    # 処理（載せ替え版）
    .venv/bin/python -u experiments/exp_016_diagonal/run_016diag_switch.py \
        --arm diag --policy competition.baseline_slalom_diag_cal:SlalomDiagCalPolicy
"""
import argparse
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "experiments" / "exp_016_diagonal",
          REPO_ROOT / "experiments" / "exp_015_time_optimal_route"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.seed_bands import assert_seeds_allowed, describe_seeds  # noqa: E402
from competition.evaluator import CompetitionEvaluator, maze_kpi  # noqa: E402
from competition.explore_cost import true_shortest  # noqa: E402
from geometry import git_rev  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

ARM_RUNNER = REPO_ROOT / "experiments" / "exp_013_band_v4_reeval" / "run_arm.py"


def load_exp013():
    """`exp_013/run_arm.py` を読み込む（`cells_and_turns` を借りる・R23）。"""
    spec = importlib.util.spec_from_file_location("exp013_run_arm", ARM_RUNNER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp013_run_arm"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_policy(spec: str):
    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls)()


class Probe:
    """方策を包み、制御周期ごとに記録する（**読むだけ**）。"""

    def __init__(self, inner):
        self._inner = inner
        self._sim = None
        self.rec = []

    name = property(lambda self: getattr(self._inner, "name", "unnamed"))
    requires_privileged = property(
        lambda self: getattr(self._inner, "requires_privileged", False))

    def bind_sim(self, sim):
        self._sim = sim
        return self._inner.bind_sim(sim)

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def act(self, obs):
        out = self._inner.act(obs)
        if self._sim is not None:
            x, y, yaw = self._sim.privileged_pose()
            v, w = self._sim.privileged_velocity()
            path = getattr(self._inner, "_path", None)
            i = int(getattr(self._inner, "_cursor", 0))
            if path is not None and 0 <= i < len(path.x):
                kap = float(path.curvature[i])
                hd = float(path.heading[i])
            else:
                kap, hd = float("nan"), float("nan")
            self.rec.append((self._sim.sim_time, x, y, yaw, v, w, kap, hd))
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, help="条件の名前（出力ディレクトリになる）")
    ap.add_argument("--policy", required=True, help="方策（module:Class）")
    ap.add_argument("--maze-dir", default="competition/mazes/design_v4")
    ap.add_argument("--out-root", default=str(REPO_ROOT / "outputs" / "exp_016_diagonal"
                                               / "016diag_switch"))
    ap.add_argument("--gate-reason", default=None,
                    help="**確保済みの評価用迷路を使う場合のみ**。裁定 R40 の合言葉")
    args = ap.parse_args()

    purpose = "gate" if args.gate_reason else "validate"
    mazes = sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))
    seeds = [int(q.stem.split("_")[1]) for q in mazes]
    print(describe_seeds(seeds, "competition"))
    assert_seeds_allowed(seeds, namespace="competition", purpose=purpose,
                         reason=args.gate_reason)

    exp013 = load_exp013()
    cell = RobotParams().cell_size
    out_dir = Path(args.out_root) / args.arm
    traj_dir = out_dir / "traj"
    traj_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{args.arm}] {args.policy} / {len(mazes)} 迷路 / {args.maze_dir}\n", flush=True)

    detail = []
    for m in mazes:
        z = np.load(m)
        d_true = int(true_shortest(z["v_walls"], z["h_walls"]))
        probe = Probe(load_policy(args.policy))
        ev = CompetitionEvaluator(maze_dir=args.maze_dir, out_dir=str(out_dir))
        r = ev.evaluate_maze(m, probe)
        kpi = maze_kpi(r["runs"])

        rec = np.asarray(probe.rec, dtype=np.float64)
        np.savez_compressed(
            traj_dir / f"{r['maze_id']}.npz",
            t=rec[:, 0], x=rec[:, 1].astype(np.float32), y=rec[:, 2].astype(np.float32),
            yaw=rec[:, 3], v=rec[:, 4].astype(np.float32), w=rec[:, 5].astype(np.float32),
            ref_curvature=rec[:, 6].astype(np.float32), ref_heading=rec[:, 7],
            run_index=np.array([q["index"] for q in r["runs"]], dtype=np.int32),
            run_t_start=np.array([q["t_start"] for q in r["runs"]], dtype=np.float64),
            run_t_end=np.array([q["t_end"] for q in r["runs"]], dtype=np.float64),
            run_outcome=np.array([q["outcome"] for q in r["runs"]]))

        for run in r["runs"]:
            seg = [s[:4] for s in probe.rec
                   if run["t_start"] - 1e-9 <= s[0] <= run["t_end"] + 1e-9]
            n_cells, n_turns = exp013.cells_and_turns(seg, cell)
            t = float(run["run_time"]) if run["run_time"] else None
            detail.append(dict(
                maze=r["maze_id"], d_true=d_true, run=run["index"],
                outcome=run["outcome"], run_time=t,
                n_cells=n_cells, n_turns=n_turns,
                path_length_m=float(run["path_length_m"]),
                visited_cells=run["visited_cells"]))
        # 斜め経路を実際に引いた回数（載せ替え版だけが持つ。報告用）
        n_diag = getattr(probe._inner, "n_diag_plans", None)
        print(f"  {r['maze_id']} D={d_true:3d} 走行 {len(r['runs'])} 本"
              f" (d)最速={r.get('best_time')}"
              + (f" 斜め経路 {n_diag} 回" if n_diag is not None else ""), flush=True)
        detail[-1]["n_diag_plans"] = n_diag

    p = out_dir / "runs_detail.json"
    json.dump(dict(arm=args.arm, policy=args.policy, maze_dir=args.maze_dir,
                   git_rev=git_rev(), cell_size=cell,
                   n_turns_definition="R4: 区画列の進行方向変化 (180°=2)",
                   gate_reason=args.gate_reason, runs=detail),
              open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n走行ごと記録: {p}（{len(detail)} 走行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
