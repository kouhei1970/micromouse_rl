#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多軸の受理条件の設計材料 — 現行の生成方式でどこまで到達できるかを測る。

**背景**: 評価帯 v2 は経路長の軸では大会実迷路に一致したが、**探索の遠回りという
軸では是正前より実態から遠ざかった**（経路比 是正前 1.216 → 是正後 1.016、
大会 1.566）。教授から「多軸の受理条件を設計せよ。実装はまだするな」との指示。

**設計の前に確かめるべき最重要の点**: そもそも**現行の生成方式で目標の経路比に
到達できるのか**。到達できないなら、どんな窓を置いても受理率がゼロになるだけで、
**生成手順そのものを変えるしかない**。窓の設計より先にこれを測る。

測る軸（すべて迷路の壁配列だけから決まる。物理シミュレーション不要）:
- $D_0$: 真の最短距離 [区画]（BFS 1 回）
- **経路比 R**: 足立法で初回にゴールへ着くまでの移動区画数 ÷ $D_0$
  （`verification/maze_exploration_cost.py` の `explore_first_run` を利用。
  准教授が古典アルゴリズムの定義から独立に書いたもの。物理版と順序が一致する
  ことが確認済み: 1.013/1.205/1.509 対 1.016/1.216/1.566）
- $N_\Delta/D_0$: 「遠回り $\Delta$ 以内で通りうる区画」の数（BFS 2 回）
- $\beta$: 独立閉路数

**生成器は改造していない。**本スクリプトの中で手順を再現し、
「経路保護のあり／なし」「除去枚数」「$D_0$ 窓のあり／なし」を振るだけ。
評価帯・検証帯の seed は使わず、設計専用の seed 帯 31000〜 を使う。

使い方:
    .venv/bin/python -u research_notes/scripts/design_multiaxis_windows.py [--n 20]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "competition" / "reference_mazes", REPO_ROOT / "verification"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from competition import maze_gen_v2 as G  # noqa: E402
from compare_generated_vs_contest import bfs_from, d_true, load_contest, load_generated  # noqa: E402
from maze_exploration_cost import explore_first_run  # noqa: E402  （准教授の実装を利用）

GOALS = [(7, 7), (7, 8), (8, 7), (8, 8)]
SEED_BASE = 31000        # 設計専用。評価帯 1000-1019・検証帯 4000-4019 は使わない


def gen_once(seed, protect, target, window, window_on="d0", floor_ratio=1.0):
    """現行 generate_maze と同一手順。経路保護・除去枚数・窓を切り替える。

    window_on="d0"    … 窓を除去**前**の D0 に掛ける（現行の実装）
    window_on="final" … 窓を**最終**の D に掛ける（保護なしだと除去で D が縮むため、
                        現行の掛け方では「窓に入れたのに短い迷路」が出てしまう）
    floor_ratio       … 経路保護の床。1.0 で「1 区画も縮めない」、<1 で緩める
    """
    rng = random.Random(seed)
    v = np.ones((17, 16), dtype=int)
    h = np.ones((16, 17), dtype=int)
    G._spanning_tree(rng, v, h)
    for e in G.GOAL_INNER:
        G._set(v, h, e, 0)
    for e in G.RING_EDGES:
        G._set(v, h, e, 1)
    gateway = rng.choice(G.RING_EDGES)
    G._set(v, h, gateway, 0)
    for e in G.FORCED_OPEN:
        G._set(v, h, e, 0)
    v[0, 0] = 1; h[0, 0] = 1; v[1, 0] = 1; h[0, 1] = 0
    protected_open = set(G.FORCED_OPEN) | {gateway} | set(G.GOAL_INNER) | {("h", 0, 1)}
    protected_wall = (set(G.RING_EDGES) - {gateway}) | {("v", 1, 0)}

    d0 = G.shortest_distance_to_goal(v, h)
    if d0 < 0:
        return None
    if window is not None and window_on == "d0" and not (window[0] <= d0 <= window[1]):
        return None
    floor = int(np.ceil(floor_ratio * d0))

    internal = ([("v", x, y) for x in range(1, 16) for y in range(16)]
                + [("h", x, y) for x in range(16) for y in range(1, 16)])
    rng.shuffle(internal)
    opened = 0
    for e in internal:
        if opened >= target:
            break
        if e in protected_wall or G._get(v, h, e) == 0:
            continue
        G._set(v, h, e, 0)
        k, x, y = e
        posts = ((x, y), (x, y + 1)) if k == "v" else ((x, y), (x + 1, y))
        if any(p != G.CENTER_POST and not any(G._get(v, h, pe) == 1 for pe in G.post_walls(*p))
               for p in posts):
            G._set(v, h, e, 1)
            continue
        if protect and G.shortest_distance_to_goal(v, h) < floor:
            G._set(v, h, e, 1)
        else:
            opened += 1

    for (px, py) in G.isolated_posts(v, h):
        cands = [e for e in G.post_walls(px, py) if e not in protected_open]
        if not cands:
            return None
        G._set(v, h, rng.choice(cands), 1)
    if sum(1 for e in G.RING_EDGES if G._get(v, h, e) == 0) != 1:
        return None
    if not G.all_cells_reachable(v, h):
        return None
    if G.isolated_posts(v, h):
        return None
    if any(G._get(v, h, e) == 1 for e in G.post_walls(*G.CENTER_POST)):
        return None
    if G.wall_follow_reaches_goal(v, h, "left") or G.wall_follow_reaches_goal(v, h, "right"):
        return None
    if window is not None and window_on == "final":
        d_fin = G.shortest_distance_to_goal(v, h)
        if not (window[0] <= d_fin <= window[1]):
            return None
    return v, h


def metrics(v, h, start=(0, 0), goals=None):
    goals = goals or GOALS
    D = d_true(v, h, start, goals)
    ds = bfs_from(v, h, [start])
    dg = bfs_from(v, h, list(goals))
    steps = explore_first_run({"v": v, "h": h, "start": start, "goals": list(goals)})
    open_edges = int((v[1:16, :] == 0).sum() + (h[:, 1:16] == 0).sum())
    return dict(
        D=D, R=(steps / D) if (steps and D) else float("nan"),
        N2=int(np.sum((ds >= 0) & (dg >= 0) & (ds + dg <= D + 2))) / D,
        N8=int(np.sum((ds >= 0) & (dg >= 0) & (ds + dg <= D + 8))) / D,
        beta=open_edges - 256 + 1)


def summarize(rows, label, used, elapsed):
    if not rows:
        print(f"{label:<38} 面数 0（消費 {used} seed）")
        return None
    q = lambda k, p: float(np.percentile([r[k] for r in rows], p))  # noqa: E731
    m = lambda k: float(np.median([r[k] for r in rows]))            # noqa: E731
    print(f"{label:<38}{len(rows):>4}{m('D'):>8.0f}{m('R'):>8.3f}"
          f"{q('R', 25):>7.3f}〜{q('R', 75):<7.3f}{m('N2'):>7.2f}{m('N8'):>7.2f}"
          f"{m('beta'):>6.0f}{used:>8}{elapsed:>8.0f}s")
    return dict(label=label, n=len(rows), D=m("D"), R=m("R"),
                R_p25=q("R", 25), R_p75=q("R", 75),
                N2=m("N2"), N8=m("N8"), beta=m("beta"), seeds_used=used)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="各条件で作る面数")
    ap.add_argument("--max-seeds", type=int, default=1500)
    args = ap.parse_args()

    # 目標値（大会実迷路。窓内 33 面 / 中央ゴール∩窓内 18 面の両方を出す）
    print("=== 目標: 大会実迷路 ===")
    print(f"{'条件':<38}{'n':>4}{'D_0':>8}{'経路比R':>8}{'R 四分位':>15}"
          f"{'N2/D0':>7}{'N8/D0':>7}{'β':>6}{'消費seed':>8}{'所要':>9}")
    con = []
    for f in sorted((REPO_ROOT / "competition" / "reference_mazes" / "contest").glob("*.npz")):
        v, h, s, g, _ = load_contest(f)
        r = metrics(np.array(v), np.array(h), s, g)
        r["_in"] = 45 <= r["D"] <= 110
        con.append(r)
    summarize([r for r in con if r["_in"]], "大会実迷路 窓[45,110]内", 0, 0)
    summarize(con, "大会実迷路 42 面 全体", 0, 0)

    print("\n=== 現行帯・是正前帯（保存済み npz から） ===")
    for d, lab in (("eval", "現行 eval 20 面"), ("eval_v2_short", "是正前 eval 20 面")):
        rows = []
        for f in sorted((REPO_ROOT / "competition" / "mazes" / d).glob("maze_*.npz")):
            v, h, s, g, _ = load_generated(f)
            rows.append(metrics(np.array(v), np.array(h), s, g))
        summarize(rows, lab, 0, 0)

    print("\n=== 生成方式の到達範囲（設計専用 seed 31000〜。生成器は無改造） ===")
    CONFIGS = [
        ("保護あり・除去15・D0窓（現行）", True, 15, (45, 110), "d0", 1.0),
        ("保護あり・除去30・D0窓", True, 30, (45, 110), "d0", 1.0),
        ("保護なし・除去15・D0窓", False, 15, (45, 110), "d0", 1.0),
        ("保護なし・除去30・D0窓（是正前+窓）", False, 30, (45, 110), "d0", 1.0),
        ("保護なし・除去30・窓なし（是正前相当）", False, 30, None, "d0", 1.0),
        ("保護なし・除去0・D0窓（完全迷路）", False, 0, (45, 110), "d0", 1.0),
        ("★保護なし・除去15・**最終D窓**", False, 15, (45, 110), "final", 1.0),
        ("★保護なし・除去30・**最終D窓**", False, 30, (45, 110), "final", 1.0),
        ("★保護なし・除去8・**最終D窓**", False, 8, (45, 110), "final", 1.0),
        ("★床0.85・除去15・**最終D窓**", True, 15, (45, 110), "final", 0.85),
        ("★床0.70・除去15・**最終D窓**", True, 15, (45, 110), "final", 0.70),
        ("★床0.50・除去20・**最終D窓**", True, 20, (45, 110), "final", 0.50),
    ]
    out = []
    for lab, protect, target, window, won, fr in CONFIGS:
        t0 = time.time()
        rows, seed, used = [], SEED_BASE, 0
        while len(rows) < args.n and used < args.max_seeds:
            r = gen_once(seed, protect, target, window, won, fr)
            used += 1
            seed += 1
            if r is not None:
                rows.append(metrics(r[0], r[1]))
        out.append(summarize(rows, lab, used, time.time() - t0))

    p = REPO_ROOT / "research_notes" / "data" / "multiaxis_design.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(contest=[{k: v for k, v in r.items() if not k.startswith("_")} for r in con],
                   configs=[o for o in out if o]),
              open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n数値 JSON: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
