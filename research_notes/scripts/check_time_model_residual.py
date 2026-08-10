#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""時間モデルの「差」に対する残差を測る — 予測利得が意味のある量かの判定。

**背景（教授指摘 2026-08-11）**: `check_distance_vs_time_optimal.py` は
  時間 [s] = a × 移動区画数 + b × 旋回回数
を 20 面の最速走行タイムに当てはめ（R²=0.9878、残差 RMS 2.24 s）、その上で
時間最適経路の利得を最大 1.81 s と見積もった。私（学生A）は「利得がモデル自身の
残差より小さいので測定になっていない」と書いたが、**これは過大な自己批判の可能性がある**。

理由: 残差 2.24 s は**迷路をまたいだ絶対時間**の予測誤差である。しかし比較して
いるのは**同一迷路上の 2 経路の差**なので、迷路ごとの系統的なずれ（スタート・
ゴールの幾何、外周の形）は差を取ると打ち消える。したがって
  ΔT = a·ΔN + b·ΔK
の予測誤差は 2.24 s より小さいはずである。**どれだけ小さいかは測らないと分からない。**

**測り方**: 同一迷路の**探索走行**と**最速走行**は、区画数も旋回回数も違い、
実測の時間差も分かっている。この 2 本を対にして
  モデルが予測する時間差  vs  実測の時間差
を突き合わせ、**差に対する残差**を出す。

**注意（先に置く）**: 探索走行には未知区画での速度制限や区画ごとの判断の待ち時間が
入る可能性があり、最速走行と同じ時間モデルに乗らないかもしれない。
**乗らなかった場合、それ自体が発見**（探索走行は最短走行と別のコスト構造を持つ）
なので、残差の構造を報告する。

軌跡は評価器に記録項目を足さず、方策を包んで取る（凍結ハーネスは触らない）。

使い方:
    .venv/bin/python research_notes/scripts/check_time_model_residual.py [--n 20]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition.baseline_classical import AdachiPolicy  # noqa: E402
from competition.evaluator import CompetitionEvaluator  # noqa: E402
from mouse.params import RobotParams  # noqa: E402


class TrajProbe:
    """方策を包み、制御周期ごとに (時刻, 位置) を記録する。"""

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
            x, y, _ = self._sim.privileged_pose()
            self.rec.append((self._sim.sim_time, x, y))
        return out


def path_stats(rec, t0, t1, cell):
    """時刻 [t0, t1] の軌跡から、通過区画数と 90 度旋回回数を数える。

    区画列は連続重複を除いて作る。旋回回数は進行方向（区画の移動方向）が
    変わった回数。180 度反転は 2 回として数える（`check_distance_vs_time_optimal`
    の Dijkstra と同じ規約）。
    """
    seq = []
    for t, x, y in rec:
        if not (t0 - 1e-9 <= t <= t1 + 1e-9):
            continue
        c = (int(x // cell), int(y // cell))
        if not seq or seq[-1] != c:
            seq.append(c)
    if len(seq) < 2:
        return len(seq) - 1, 0
    dirs = []
    for a, b in zip(seq, seq[1:]):
        dirs.append((int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))))
    order = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}
    turns = 0
    for d1, d2 in zip(dirs, dirs[1:]):
        if d1 not in order or d2 not in order:
            continue
        k1, k2 = order[d1], order[d2]
        turns += min((k2 - k1) % 4, (k1 - k2) % 4)
    return len(seq) - 1, turns


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--maze-dir", default="competition/mazes/eval")
    args = ap.parse_args()

    params = RobotParams()
    cell = params.cell_size
    mazes = sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"))[:args.n]

    pairs = []
    for m in mazes:
        probe = TrajProbe(AdachiPolicy())
        # max_runs=2 で「探索走行 + 最初の最速走行」だけを走らせる（時間短縮）
        ev = CompetitionEvaluator(maze_dir=args.maze_dir, max_runs=2,
                                  out_dir=str(REPO_ROOT / "outputs" / "time_model_residual"))
        r = ev.evaluate_maze(m, probe)
        goal_runs = [x for x in r["runs"] if x["outcome"] == "goal"]
        if len(goal_runs) < 2:
            print(f"  {m.stem}: 最速走行が成立せず除外")
            continue
        a, b = goal_runs[0], goal_runs[1]
        na, ka = path_stats(probe.rec, a["t_start"], a["t_end"], cell)
        nb, kb = path_stats(probe.rec, b["t_start"], b["t_end"], cell)
        pairs.append(dict(maze=r["maze_id"],
                          n_exp=na, k_exp=ka, t_exp=a["run_time"],
                          n_fast=nb, k_fast=kb, t_fast=b["run_time"]))
        print(f"  {r['maze_id']}: 探索 {na}区画/{ka}旋回 {a['run_time']:.2f}s → "
              f"最速 {nb}区画/{kb}旋回 {b['run_time']:.2f}s", flush=True)

    if len(pairs) < 5:
        print("対が足りません")
        return 1

    # --- 絶対値での当てはめ（最速走行のみ。前スクリプトと同じ土俵）
    A = np.array([[p["n_fast"], p["k_fast"]] for p in pairs], dtype=float)
    y = np.array([p["t_fast"] for p in pairs], dtype=float)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a_, b_ = float(coef[0]), float(coef[1])
    res_abs = y - A @ coef
    print(f"\n【絶対値の当てはめ（最速走行 n={len(pairs)}）】")
    print(f"  時間 = {a_:.4f}×区画 + {b_:.4f}×旋回   残差 RMS {np.sqrt(np.mean(res_abs**2)):.3f} s"
          f"（最大 {np.max(np.abs(res_abs)):.2f} s）")

    # --- 差での検証: 同一迷路の 探索 − 最速
    dN = np.array([p["n_exp"] - p["n_fast"] for p in pairs], dtype=float)
    dK = np.array([p["k_exp"] - p["k_fast"] for p in pairs], dtype=float)
    dT = np.array([p["t_exp"] - p["t_fast"] for p in pairs], dtype=float)
    pred = a_ * dN + b_ * dK
    res_d = dT - pred
    print(f"\n【差での検証（同一迷路の 探索 − 最速、n={len(pairs)}）】")
    print(f"{'面':<12}{'Δ区画':>7}{'Δ旋回':>7}{'予測Δt':>10}{'実測Δt':>10}{'残差':>9}")
    for p, pr, ac, rr in zip(pairs, pred, dT, res_d):
        print(f"{p['maze']:<12}{p['n_exp']-p['n_fast']:>7}{p['k_exp']-p['k_fast']:>7}"
              f"{pr:>10.2f}{ac:>10.2f}{rr:>9.2f}")
    rms_d = float(np.sqrt(np.mean(res_d ** 2)))
    print(f"\n  差の残差: RMS {rms_d:.3f} s、最大 {np.max(np.abs(res_d)):.2f} s、"
          f"平均 {np.mean(res_d):+.3f} s（偏りの有無）")
    print(f"  絶対値の残差 RMS {np.sqrt(np.mean(res_abs**2)):.3f} s に対し "
          f"{rms_d/np.sqrt(np.mean(res_abs**2)):.2f} 倍")
    print("\n  判定: 時間最適経路の予測利得（最大 1.81 s）に対して")
    if rms_d < 1.81 / 3:
        print(f"    差の残差 {rms_d:.2f} s は利得の 1/3 未満 → **利得は意味のある量**")
    elif rms_d < 1.81:
        print(f"    差の残差 {rms_d:.2f} s は利得より小さいが同程度 → **弱い証拠**")
    else:
        print(f"    差の残差 {rms_d:.2f} s は利得以上 → **利得は測定できない**")

    out = REPO_ROOT / "research_notes" / "data" / "time_model_residual.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(pairs=pairs, a=a_, b=b_,
                   rms_abs=float(np.sqrt(np.mean(res_abs ** 2))), rms_diff=rms_d),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n数値 JSON: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
