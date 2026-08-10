#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""診断: 経路保護型除去の枚数を振ったとき、最短経路上の「橋」の割合が下がるか。

生成器は一切改造しない。既存の引数 extra_open_target を振るだけの測定である。
評価帯 1000-1019・検証帯 4000-4019 は使わず、診断専用の seed 帯 21000〜 を使う。
"""
import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np
from competition.maze_gen_v2 import generate_maze
from compare_generated_vs_contest import analyse

GOALS = [(7, 7), (7, 8), (8, 7), (8, 8)]
N_FACES = 20

print(f"{'除去目標':>8} {'β 中央値':>10} {'D 中央値':>10} {'橋の割合 中央値':>16} "
      f"{'橋の割合 四分位':>18} {'実除去 中央値':>13} {'試行 合計':>10}")
print("-" * 96)
for target in (15, 25, 40, 60, 80):
    res, attempts, opened = [], 0, []
    seed = 21000
    while len(res) < N_FACES:
        try:
            v, h, info = generate_maze(seed, extra_open_target=target)
        except RuntimeError:
            seed += 1
            continue
        r = analyse(np.array(v), np.array(h), (0, 0), GOALS, f"s{seed}")
        res.append(r)
        attempts += info["attempts"]
        opened.append(info["extra_opened"])
        seed += 1
    fr = np.array([r["canon_frac_inf"] for r in res])
    print(f"{target:>8} {np.median([r['beta'] for r in res]):>10.0f} "
          f"{np.median([r['d_true'] for r in res]):>10.0f} "
          f"{np.median(fr) * 100:>15.1f}% "
          f"{np.percentile(fr, 25) * 100:>8.1f}〜{np.percentile(fr, 75) * 100:<8.1f} "
          f"{np.median(opened):>13.0f} {attempts:>10}", flush=True)

print("\n参考: 大会実迷路・窓内 33 面 → β 中央値 21、D 中央値 63、橋の割合 中央値 10.0%")
