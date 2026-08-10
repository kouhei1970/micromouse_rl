#!/usr/bin/env python3
"""現行評価迷路20面(既知の正解値: D_true中央値20, beta 33-35, 迂回率中央値1.43)に
   compute_stats.py と同一ロジックを適用し、自作コードのバグ検証を行う。
   読み取り専用: リポジトリのファイルは一切変更しない(出力は scratch のみ)。"""
import glob
import statistics as st
import numpy as np
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-kouhei-tmp-github-micromouse-rl/3ea48c6c-9f45-41ca-8aed-d1c591c0688d/scratchpad/contest_mazes")
from compute_stats import (bfs_from_start, count_shortest_paths, connected_components,
                            cell_degree, wall_follow_reaches_goal, goal_gateway_count,
                            start_open_directions)

EVAL_DIR = "/Users/kouhei/tmp/github/micromouse_rl/competition/mazes/eval"
GOAL = {(7, 7), (7, 8), (8, 7), (8, 8)}
START = (0, 0)

files = sorted(glob.glob(EVAL_DIR + "/maze_*.npz"))
print(f"n files = {len(files)}")

d_trues, detours, betas = [], [], []
for fp in files:
    d = np.load(fp)
    v, h = d["v_walls"], d["h_walls"]
    dist, order = bfs_from_start(v, h, START)
    d_true, n_paths = count_shortest_paths(v, h, START, GOAL, dist, order)
    manhattan = 14
    detour = d_true / manhattan
    n_comp, open_edges = connected_components(v, h)
    beta = open_edges - 256 + n_comp
    d_trues.append(d_true)
    detours.append(detour)
    betas.append(beta)

print(f"D_true: median={st.median(d_trues):.0f} range={min(d_trues)}-{max(d_trues)}")
print(f"detour: median={st.median(detours):.3f} range={min(detours):.3f}-{max(detours):.3f}")
print(f"beta:   median={st.median(betas):.0f} range={min(betas)}-{max(betas)}")
print("既知の実測値(前フェーズ報告): D_true 15-26(中央値20) / beta 33-35 / 迂回率中央値1.43")
