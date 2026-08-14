#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""016-cal 本体 — **安全率 1 点ぶんの校正走行**（カード `card_016cal.md`）。

**1 プロセス = 1 安全率**にしてある（**8 点を並行に回せる**ようにするため）。

--------------------------------------------------------------------------
測るもの（REQ_001 §2-4 の表を埋める）
--------------------------------------------------------------------------
- **完走**（**カード §5-2 の操作的定義**）:
  **(a) ゴール到達が真 かつ その迷路の全走行で outcome ∈ {goal, timeout}**
  （＝ **collision / tipover / stuck が 0 回**）
- 最速タイム（(b) の走行のうち最速。**指標の定義は凍結ハーネスの `maze_kpi` をそのまま呼ぶ**）
- **横偏差 最大 / RMS**（**方策を包んで記録するだけ**。電圧も軌跡も変えない）
- 失敗様式（迷路 ID つき）

--------------------------------------------------------------------------
⚠️ 制御には一切触れない
--------------------------------------------------------------------------
**安全率はコンストラクタ引数**なので、**方策ファイルを 1 行も変更せずに振れる**。
`_do_drive_control` を包んでいるのは**記録のためだけ**で、**親を呼んでその返り値を
そのまま返す**（横偏差は親と同じ式で再計算する）。

使い方:
    .venv/bin/python experiments/exp_016_diagonal/run_016cal.py --safety 0.70
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.seed_bands import (assert_seeds_allowed,  # noqa: E402
                                describe_seeds)
from competition.baseline_slalom_e1t_tr_f0b import SlalomE1TTRF0bPolicy  # noqa: E402
from competition.evaluator import CompetitionEvaluator, maze_kpi  # noqa: E402

# 走行が「壊れていない」とみなす outcome（カード §5-2）
OK_OUTCOMES = {"goal", "timeout"}


class ProbedCalPolicy(SlalomE1TTRF0bPolicy):
    """**新既定（F0＋F0-b 系）そのまま**。横偏差を記録するだけ。

    `_do_drive_control` は**親を呼んで返り値をそのまま返す**ので、
    **電圧も軌跡も 1 ビットも変わらない**。横偏差は**親と同じ式**で再計算している
    （親は `e_y = (x−p_x)(−sin ψ) + (y−p_y) cos ψ` を使う）。
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._ey = []

    def _do_drive_control(self, obs, x, y, yaw):
        out = super()._do_drive_control(obs, x, y, yaw)
        path, i = self._path, self._cursor
        if path is not None:
            psi = float(path.heading[i])
            self._ey.append((x - float(path.x[i])) * (-math.sin(psi))
                            + (y - float(path.y[i])) * math.cos(psi))
        return out


def wilson_lower(k: int, n: int, z: float = 1.645) -> float:
    """完走率の**片側 95% 下限**（Wilson）。REQ_001 §2-3 の判定に使う。

    **全数成功（k = n）でも 1.0 にはならない**のが要点である
    （n=20 で 0.861・n=60 で 0.951）。
    """
    if n == 0:
        return float("nan")
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safety", type=float, required=True)
    ap.add_argument("--maze-dir", default="competition/mazes/cal_v4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mazes = sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))
    seeds = [int(p.stem.split("_")[1]) for p in mazes]
    print(describe_seeds(seeds, "competition"))
    assert_seeds_allowed(seeds, namespace="competition", purpose="validate")

    out = Path(args.out or (REPO_ROOT / "outputs" / "exp_016_diagonal" / "016cal"
                            / f"sf{args.safety:g}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"安全率 {args.safety:g}／{len(mazes)} 迷路／{args.maze_dir}\n", flush=True)

    rows = []
    for m in mazes:
        pol = ProbedCalPolicy(safety_factor=args.safety)
        ev = CompetitionEvaluator(maze_dir=args.maze_dir,
                                  out_dir=str(out.parent / f"sf{args.safety:g}_traj"))
        r = ev.evaluate_maze(m, pol)
        kpi = maze_kpi(r["runs"])
        outs = [q["outcome"] for q in r["runs"]]
        broken = [o for o in outs if o not in OK_OUTCOMES]
        ey = np.abs(np.asarray(pol._ey, dtype=float)) if pol._ey else np.array([0.0])
        rows.append(dict(
            maze=r["maze_id"], n_runs=len(outs), outcomes=outs,
            # **完走の操作的定義**（カード §5-2）
            completed=bool(kpi["goal_reached"] and not broken),
            goal_reached=bool(kpi["goal_reached"]),
            n_broken=len(broken), broken_kinds=sorted(set(broken)),
            fast_time=kpi["fast_time"], explore_time=kpi["explore_time"],
            fast_run_done=bool(kpi["fast_run_done"]),
            e_y_max_m=float(ey.max()), e_y_rms_m=float(np.sqrt(np.mean(ey ** 2))),
            n_ticks=int(ey.size)))
        c = rows[-1]
        print(f"  {c['maze']}: {'完走' if c['completed'] else '**未完走**'}"
              f" 走行 {c['n_runs']} 本 {outs}"
              f" 最速 {c['fast_time'] if c['fast_time'] else '-'}"
              f" e_y最大 {c['e_y_max_m']*1000:.2f} mm", flush=True)

    n = len(rows)
    k = sum(1 for c in rows if c["completed"])
    fast = [c["fast_time"] for c in rows if c["fast_time"]]
    eymax = [c["e_y_max_m"] for c in rows]
    eyrms = [c["e_y_rms_m"] for c in rows]
    summ = dict(safety_factor=args.safety, n=n, n_completed=k,
                completion_rate=k / n if n else float("nan"),
                completion_lower95=wilson_lower(k, n),
                fast_time_median=(float(np.median(fast)) if fast else None),
                n_fast=len(fast),
                e_y_max_median_m=float(np.median(eymax)),
                e_y_max_max_m=float(np.max(eymax)),
                e_y_rms_median_m=float(np.median(eyrms)),
                failed_faces=[c["maze"] for c in rows if not c["completed"]],
                failure_kinds=sorted({x for c in rows for x in c["broken_kinds"]}))
    print(f"\n【安全率 {args.safety:g}】完走 {k}/{n} = {summ['completion_rate']*100:.1f}%"
          f"／**片側 95% 下限 {summ['completion_lower95']:.3f}**")
    print(f"  最速タイム 中央値 {summ['fast_time_median']}（n={summ['n_fast']}）")
    print(f"  横偏差 最大の中央値 {summ['e_y_max_median_m']*1000:.2f} mm"
          f"／最悪 {summ['e_y_max_max_m']*1000:.2f} mm"
          f"／RMS 中央値 {summ['e_y_rms_median_m']*1000:.2f} mm")
    if summ["failed_faces"]:
        print(f"  未完走の迷路: {summ['failed_faces']}／様式 {summ['failure_kinds']}")

    json.dump(dict(summary=summ, rows=rows, maze_dir=args.maze_dir),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
