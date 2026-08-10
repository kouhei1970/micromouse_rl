#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""診断: 経路保護の「厳しさ」を緩めると橋の割合が大会水準へ近づくか。

現行実装は「最短距離を 1 区画も縮めない壁だけ開ける」= 床 floor = D0（厳格）。
これを floor = ratio * D0 に緩めると、最短経路をまたぐ弦のうち「縮み幅が小さいもの」
が通るようになる。ratio を振って、橋の割合・D・β のトレードオフを測る。

**リポジトリの生成器は一切変更しない。**本スクリプトの中だけで手順を再現する
（competition/reference_mazes/prototype/proto2.py と同じ流儀）。
評価帯 1000-1019・検証帯 4000-4019 は使わず、診断専用 seed 帯 22000〜 を使う。
"""
import random
import sys

import numpy as np

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, os.path.dirname(_HERE))

from competition import maze_gen_v2 as G
from compare_generated_vs_contest import analyse

GOALS = [(7, 7), (7, 8), (8, 7), (8, 8)]
N_FACES = 20
LO, HI = 45, 110


def gen(seed, ratio, target, max_attempts=400):
    """現行 generate_maze と同一手順。判定の床だけ floor = ratio*D0 に置き換える。"""
    rng = random.Random(seed)
    for attempt in range(1, max_attempts + 1):
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
        if d0 < 0 or not (LO <= d0 <= HI):
            continue
        floor = int(np.ceil(ratio * d0))

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
            if G.shortest_distance_to_goal(v, h) < floor:
                G._set(v, h, e, 1)
            else:
                opened += 1

        ok = True
        for (px, py) in G.isolated_posts(v, h):
            cands = [e for e in G.post_walls(px, py) if e not in protected_open]
            if not cands:
                ok = False
                break
            G._set(v, h, rng.choice(cands), 1)
        if not ok:
            continue
        if sum(1 for e in G.RING_EDGES if G._get(v, h, e) == 0) != 1:
            continue
        if not G.all_cells_reachable(v, h):
            continue
        if G.isolated_posts(v, h):
            continue
        if any(G._get(v, h, e) == 1 for e in G.post_walls(*G.CENTER_POST)):
            continue
        if G.wall_follow_reaches_goal(v, h, "left") or G.wall_follow_reaches_goal(v, h, "right"):
            continue
        d = G.shortest_distance_to_goal(v, h)
        if not (LO <= d <= HI):          # 緩めると D が窓を割りうるので最終形でも確認
            continue
        return v, h, dict(attempts=attempt, d0=d0, d=d, opened=opened)
    return None


print(f"{'床 ratio':>9} {'D0 中央値':>10} {'D 中央値':>10} {'D 範囲':>12} {'β 中央値':>9} "
      f"{'橋の割合 中央値':>16} {'四分位':>16} {'最短路本数':>11} {'消費seed':>9}")
print("-" * 116)
for ratio in (1.00, 0.98, 0.95, 0.90, 0.85, 0.75):
    res, infos, seed, used = [], [], 22000, 0
    while len(res) < N_FACES and used < 400:
        r = gen(seed, ratio, 15, max_attempts=1)
        used += 1
        seed += 1
        if r:
            v, h, info = r
            res.append(analyse(v, h, (0, 0), GOALS, f"s{seed}"))
            infos.append(info)
    if len(res) < N_FACES:
        print(f"{ratio:>9.2f}  面数不足 {len(res)}/{N_FACES}（消費 {used} seed）")
        continue
    fr = np.array([x["canon_frac_inf"] for x in res])
    D = [x["d_true"] for x in res]
    print(f"{ratio:>9.2f} {np.median([i['d0'] for i in infos]):>10.0f} "
          f"{np.median(D):>10.0f} {min(D):>5.0f}〜{max(D):<6.0f} "
          f"{np.median([x['beta'] for x in res]):>9.0f} "
          f"{np.median(fr) * 100:>15.1f}% "
          f"{np.percentile(fr, 25) * 100:>6.1f}〜{np.percentile(fr, 75) * 100:<8.1f} "
          f"{np.median([x['n_shortest_paths'] for x in res]):>11.0f} {used:>9}", flush=True)

print("\n参考: 大会実迷路・窓内 33 面 → D 中央値 63（47〜105）、β 中央値 21、"
      "橋の割合 中央値 10.0%、最短経路本数 中央値 6")
