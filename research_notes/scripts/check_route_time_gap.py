#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候補経路は「時間で競っている」か — 歩数と折れ数の対で見る。

**起点はユーザ（マイクロマウスの実務家）の言葉**:

  > 途中から枝分かれしたとしても 2 本という味方です。大会ではよく
  > **北回り、南回り、それぞれ何歩、何折れ**みたいに迷路に説明がついてますね

**「何歩・何折れ」と書き分けるのは、歩数だけでは経路を特徴づけられないから**である。
そして我々は既に、その理由を実測している:

    時間 [s] = 0.7216 × 移動区画数 + 0.8667 × 旋回回数   （残差 RMS 0.167 s）
    → **旋回 1 回のコストは移動 1.20 区画ぶん**

**旋回が歩数とほぼ同じ重みで時間を食う。**だから「62 歩 14 折れ」と「66 歩 8 折れ」は、
**歩数では前者が短く、時間では後者が速い**ということが起こる。

## 測るもの

`competition/macro_routes.py` が返す巨視的に異なる候補経路それぞれについて:

  - **歩数**（移動区画数）と**折れ数**（進行方向が変わった回数。180 度は 2 回）
  - 時間モデルによる推定所要時間
  - **最良と次善の時間差**（相対値）

**狙い**: 「候補が 2 本ある」だけでは足りない。**2 本の時間が競っているか**が本質のはず。

  - 時間差が**大きい**（例 30%）→ 探索しなくても片方が明らかに良い。**選択の問題にならない**
  - 時間差が**小さい**（例 5%）→ **探索して初めてどちらが速いか分かる**

我々の生成器は経路長（$D$ と $R$）で選抜しているので、候補経路が「長さで明確に劣る」
形になっている恐れがある。その場合、**本数を合わせても選択の難しさは再現できていない**。

## 併せて測る: 大会実迷路での「距離最適 vs 時間最適」

以前 v2 帯で測ったとき、時間最適経路の利得は **2/20 面・最大 2.05%** と小さかった。
**効果が小さかったのは、我々の迷路に「歩数と折れ数が逆向きの候補」がほとんど
無かったからかもしれない。**大会実迷路で同じ測定をすれば判別できる。

時間モデルの係数は**ハードコードしない**。`research_notes/data/time_model_residual.json`
（L0-a の実測軌跡から最小二乗で当てはめたもの）を読む。

使い方:
    .venv/bin/python -u research_notes/scripts/check_route_time_gap.py
"""
import argparse
import glob
import heapq
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from competition.explore_cost import DIRS, GOAL_CELLS, true_shortest, true_wall  # noqa: E402
from competition.macro_routes import contest_files, load, macro_routes  # noqa: E402

N = 16
TIME_MODEL = REPO_ROOT / "research_notes" / "data" / "time_model_residual.json"
DELTA, THETA = 4, 3


def load_time_model():
    """時間モデルの係数を実測から読む（ハードコードしない）。"""
    j = json.load(open(TIME_MODEL, encoding="utf-8"))
    return float(j["a"]), float(j["b"]), float(j["rms_diff"])


def cells_turns(path):
    """経路（セル列）の (歩数, 折れ数)。180 度反転は 2 回として数える。"""
    if len(path) < 2:
        return 0, 0
    order = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}
    ds = [order[(int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))]
          for a, b in zip(path, path[1:])]
    turns = sum(min((d2 - d1) % 4, (d1 - d2) % 4) for d1, d2 in zip(ds, ds[1:]))
    return len(path) - 1, turns


def dijkstra_time(v, h, start, goals, turn_cost):
    """状態 (区画, 進行方向) 上で「移動 1 + 旋回 turn_cost」を最小化する。"""
    best, pq = {}, []
    for d in range(4):
        heapq.heappush(pq, (0.0, start, d, 0, 0))
    goals = set(tuple(g) for g in goals)
    while pq:
        cost, cell, d, nmove, nturn = heapq.heappop(pq)
        if cell in goals:
            return cost, nmove, nturn
        if best.get((cell, d), float("inf")) <= cost:
            continue
        best[(cell, d)] = cost
        for nd in range(4):
            dx, dy = DIRS[nd]
            nxt = (cell[0] + dx, cell[1] + dy)
            if not (0 <= nxt[0] < N and 0 <= nxt[1] < N):
                continue
            if true_wall(v, h, cell[0], cell[1], nd):
                continue
            t = min((nd - d) % 4, (d - nd) % 4)
            heapq.heappush(pq, (cost + 1.0 + turn_cost * t, nxt, nd, nmove + 1, nturn + t))
    return float("inf"), -1, -1


def analyse(f, a, b):
    v, h, start, goals = load(f)
    routes = macro_routes(v, h, start, goals, delta=DELTA, theta=THETA)
    stats = []
    for p in routes:
        c, t = cells_turns(p)
        stats.append(dict(cells=c, turns=t, time=a * c + b * t))
    stats.sort(key=lambda s: s["time"])
    gap = ((stats[1]["time"] - stats[0]["time"]) / stats[0]["time"]) if len(stats) >= 2 else None
    # 距離最適 vs 時間最適（候補集合に依らない、迷路そのものの性質）
    tc = b / a
    _, nd, kd = dijkstra_time(v, h, start, goals, 1e-6)
    _, nt, kt = dijkstra_time(v, h, start, goals, tc)
    t_dist, t_time = a * nd + b * kd, a * nt + b * kt
    return dict(name=Path(f).stem, K=len(routes), routes=stats, gap=gap,
                d_true=int(true_shortest(v, h, start, goals)),
                dist_opt=(nd, kd, t_dist), time_opt=(nt, kt, t_time),
                opt_gain=(t_dist - t_time) / t_dist)


def summarize(label, rows, rms):
    K = np.array([r["K"] for r in rows])
    gaps = np.array([r["gap"] for r in rows if r["gap"] is not None])
    gains = np.array([r["opt_gain"] for r in rows])
    print(f"\n【{label}】n={len(rows)}")
    print(f"  候補経路 K: 中央値 {np.median(K):.1f}／K>=2 {np.mean(K >= 2) * 100:.0f}%"
          f"／最大 {K.max()}")
    if gaps.size:
        print(f"  **最良と次善の時間差**: 中央値 {np.median(gaps) * 100:.1f}%"
              f"（四分位 {np.percentile(gaps, 25) * 100:.1f}〜{np.percentile(gaps, 75) * 100:.1f}%"
              f"、範囲 {gaps.min() * 100:.1f}〜{gaps.max() * 100:.1f}%）  n={gaps.size}")
        for th in (0.05, 0.10, 0.20):
            print(f"    時間差 <= {th * 100:.0f}% の面: "
                  f"{np.mean(gaps <= th) * 100:.0f}%（探索して初めて分かる水準）")
    else:
        print("  最良と次善の時間差: 候補が 2 本以上ある面が無い")
    print(f"  距離最適 vs 時間最適: 改善する面 {int((gains > 1e-9).sum())}/{len(gains)}"
          f"、短縮率 中央値 {np.median(gains) * 100:.2f}%、最大 {gains.max() * 100:.2f}%")
    # 歩数と折れ数が逆向きの候補（歩数は多いが折れ数は少ない）の面数
    n_tradeoff = 0
    for r in rows:
        s = r["routes"]
        if len(s) < 2:
            continue
        if any((x["cells"] > s[0]["cells"] and x["turns"] < s[0]["turns"])
               or (x["cells"] < s[0]["cells"] and x["turns"] > s[0]["turns"])
               for x in s[1:]):
            n_tradeoff += 1
    print(f"  **歩数と折れ数が逆向きの候補**を持つ面: {n_tradeoff}/{len(rows)}"
          f"（{n_tradeoff / len(rows) * 100:.0f}%）← 『何歩・何折れ』の書き分けが要る面")
    return dict(label=label, n=len(rows), K_med=float(np.median(K)),
                gap_med=float(np.median(gaps)) if gaps.size else None,
                gap_p25=float(np.percentile(gaps, 25)) if gaps.size else None,
                gap_p75=float(np.percentile(gaps, 75)) if gaps.size else None,
                n_gap=int(gaps.size), n_tradeoff=n_tradeoff,
                opt_gain_med=float(np.median(gains)), opt_gain_max=float(gains.max()),
                n_opt_gain=int((gains > 1e-9).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="*", default=["competition/mazes/eval",
                                                  "competition/mazes/eval_v2_low_detour",
                                                  "competition/mazes/eval_v2_short"])
    args = ap.parse_args()
    a, b, rms = load_time_model()
    print(f"時間モデル（実測から。ハードコードしない）: 時間 = {a:.4f}×歩数 + {b:.4f}×折れ数")
    print(f"  → 旋回 1 回 = 移動 {b / a:.3f} 区画ぶん／差の残差 RMS {rms:.3f} s")
    print(f"  候補経路の定義: Δ={DELTA}, θ={THETA}")

    out = []
    groups = [("大会実迷路 窓[45,110]内", contest_files(True)),
              ("大会実迷路 42 面全体", contest_files(False))]
    for d in args.dirs:
        groups.append((Path(d).name, sorted(glob.glob(str(REPO_ROOT / d / "maze_*.npz")))))
    detail = {}
    for lab, files in groups:
        rows = [analyse(f, a, b) for f in files]
        detail[lab] = rows
        out.append(summarize(lab, rows, rms))

    # 「何歩・何折れ」の実例（大会実迷路で候補が 2 本以上あり時間差が小さい面）
    print("\n【実例】大会実迷路で候補が競っている面（時間差の小さい順に 5 面）")
    print(f"{'面':<28}{'候補':>5}  {'各候補の (歩数, 折れ数) → 推定時間 [s]':<48}{'時間差':>8}")
    ex = [r for r in detail["大会実迷路 窓[45,110]内"] if r["gap"] is not None]
    ex.sort(key=lambda r: r["gap"])
    for r in ex[:5]:
        desc = " ／ ".join(f"({s['cells']},{s['turns']})→{s['time']:.1f}" for s in r["routes"][:3])
        print(f"{r['name']:<28}{r['K']:>5}  {desc:<48}{r['gap'] * 100:>7.1f}%")

    p = REPO_ROOT / "research_notes" / "data" / "route_time_gap.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(a=a, b=b, delta=DELTA, theta=THETA, summary=out), open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n数値 JSON: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
