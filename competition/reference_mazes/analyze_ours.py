#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
現行の評価迷路(competition/mazes/{eval,validation,eval_v1_nonconforming})の
静的指標を計算する。リポジトリ内のファイルは読み取り専用でアクセスする。

npz形式:
  v_walls: shape (17,16), int  v_walls[x,y] = 区画(x-1,y)と(x,y)の間の縦壁 (1=壁, 0=通行可)
            x=0 は左外周壁, x=16 は右外周壁
  h_walls: shape (16,17), int  h_walls[x,y] = 区画(x,y-1)と(x,y)の間の横壁 (1=壁, 0=通行可)
            y=0 は下外周壁, y=16 は上外周壁

区画座標: x=0..15 (列), y=0..15 (行)。スタート=(0,0)。ゴール中央2x2=(7,7),(8,7),(7,8),(8,8)。
"""
import csv
import glob
import json
import os
import statistics
from collections import deque

import numpy as np

REPO = "/Users/kouhei/tmp/github/micromouse_rl"
# 2026-08-11 リポジトリ収容にあたり、揮発する一時領域の絶対パスから
# 本ファイル基準の相対パスへ変更した（唯一の改変点）。
OUTDIR = os.path.dirname(os.path.abspath(__file__))

W = H = 16
GOAL_CELLS = [(7, 7), (8, 7), (7, 8), (8, 8)]
START = (0, 0)
MANHATTAN_REF = 14  # |7-0|+|7-0|

DIRS = {
    "N": (0, 1),
    "E": (1, 0),
    "S": (0, -1),
    "W": (-1, 0),
}
DIR_ORDER = ["N", "E", "S", "W"]  # index i -> clockwise turns


def open_east(v_walls, x, y):
    """区画(x,y) <-> (x+1,y) の間(東側)が通行可か"""
    if x < 0 or x >= W - 1:
        return False
    return v_walls[x + 1, y] == 0


def open_north(h_walls, x, y):
    """区画(x,y) <-> (x,y+1) の間(北側)が通行可か"""
    if y < 0 or y >= H - 1:
        return False
    return h_walls[x, y + 1] == 0


def build_adjacency(v_walls, h_walls):
    """cell -> list[cell] の隣接リスト(通行可のみ)を作る"""
    adj = {(x, y): [] for x in range(W) for y in range(H)}
    edges = 0
    for x in range(W):
        for y in range(H):
            if open_east(v_walls, x, y):
                adj[(x, y)].append((x + 1, y))
                adj[(x + 1, y)].append((x, y))
                edges += 1
            if open_north(h_walls, x, y):
                adj[(x, y)].append((x, y + 1))
                adj[(x, y + 1)].append((x, y))
                edges += 1
    return adj, edges


def connected_components(adj):
    visited = set()
    comps = []
    for cell in adj:
        if cell in visited:
            continue
        comp = set()
        dq = deque([cell])
        visited.add(cell)
        while dq:
            c = dq.popleft()
            comp.add(c)
            for n in adj[c]:
                if n not in visited:
                    visited.add(n)
                    dq.append(n)
        comps.append(comp)
    return comps


def bfs_dist_and_paths(adj, start):
    """start からの最短距離と最短路本数(DAGカウント)を返す"""
    dist = {start: 0}
    npaths = {start: 1}
    dq = deque([start])
    order = [start]
    while dq:
        c = dq.popleft()
        for n in adj[c]:
            if n not in dist:
                dist[n] = dist[c] + 1
                npaths[n] = 0
                dq.append(n)
                order.append(n)
    # 2周目: 距離昇順に前駆から経路数を積算 (BFS順序=距離昇順なのでそのまま使える)
    npaths = {k: 0 for k in dist}
    npaths[start] = 1
    for c in order:
        if c == start:
            continue
        total = 0
        for n in adj[c]:
            if dist.get(n, None) == dist[c] - 1:
                total += npaths[n]
        npaths[c] = total
    return dist, npaths


def goal_gateway_count(v_walls, h_walls):
    """ゴール2x2の外側との開口数(入口数)"""
    goal_set = set(GOAL_CELLS)
    count = 0
    for x in range(W):
        for y in range(H):
            if open_east(v_walls, x, y):
                a, b = (x, y), (x + 1, y)
                if (a in goal_set) != (b in goal_set):
                    count += 1
            if open_north(h_walls, x, y):
                a, b = (x, y), (x, y + 1)
                if (a in goal_set) != (b in goal_set):
                    count += 1
    return count


def wall_follower_reaches(adj, v_walls, h_walls, hand):
    """壁づたい走行(左手/右手)でスタートからゴールに到達できるか

    hand: 'left' or 'right'
    優先順位:
      left  -> 左, 直進, 右, 後退
      right -> 右, 直進, 左, 後退
    状態(位置,向き)が再訪されたら無限ループとみなし到達不可(False)。
    """
    def can_move(pos, d):
        x, y = pos
        if d == "N":
            return open_north(h_walls, x, y)
        if d == "S":
            return open_north(h_walls, x, y - 1) if y - 1 >= 0 else False
        if d == "E":
            return open_east(v_walls, x, y)
        if d == "W":
            return open_east(v_walls, x - 1, y) if x - 1 >= 0 else False
        raise ValueError(d)

    def turn(d, steps):
        i = DIR_ORDER.index(d)
        return DIR_ORDER[(i + steps) % 4]

    pos = START
    facing = "N"  # スタート区画は北開放(h[0,1]=0)と規定されている
    visited_states = set()
    max_steps = 4 * W * H + 10

    if pos in GOAL_CELLS:
        return True

    for _ in range(max_steps):
        state = (pos, facing)
        if state in visited_states:
            return False
        visited_states.add(state)

        if hand == "left":
            order = [turn(facing, -1), facing, turn(facing, +1), turn(facing, +2)]
        else:
            order = [turn(facing, +1), facing, turn(facing, -1), turn(facing, +2)]

        moved = False
        for d in order:
            if can_move(pos, d):
                dx, dy = DIRS[d]
                pos = (pos[0] + dx, pos[1] + dy)
                facing = d
                moved = True
                break
        if not moved:
            # 完全に閉じ込められている(理論上起きないはず)
            return False
        if pos in GOAL_CELLS:
            return True
    return False


def degree_distribution(adj):
    deg = {}
    for c, nbrs in adj.items():
        d = len(nbrs)
        deg[d] = deg.get(d, 0) + 1
    return deg


def analyze_one(npz_path):
    d = np.load(npz_path)
    v_walls = d["v_walls"]
    h_walls = d["h_walls"]
    seed = int(d["seed"]) if "seed" in d else None

    adj, edges = build_adjacency(v_walls, h_walls)
    comps = connected_components(adj)
    n_components = len(comps)
    all_reachable = (n_components == 1) and (len(comps[0]) == W * H)

    V = W * H
    beta = edges - V + n_components  # 独立閉路数(サイクロマティック数)

    dist, npaths = bfs_dist_and_paths(adj, START)
    goal_dists = {g: dist.get(g, None) for g in GOAL_CELLS}
    reachable_goal_dists = [gd for gd in goal_dists.values() if gd is not None]
    if reachable_goal_dists:
        d_true = min(reachable_goal_dists)
        n_shortest_paths = sum(
            npaths.get(g, 0) for g in GOAL_CELLS if goal_dists[g] == d_true
        )
    else:
        d_true = None
        n_shortest_paths = 0

    detour_ratio = (d_true / MANHATTAN_REF) if d_true is not None else None

    deg_dist = degree_distribution(adj)
    dead_ends = deg_dist.get(1, 0)

    n_gateways = goal_gateway_count(v_walls, h_walls)

    left_ok = wall_follower_reaches(adj, v_walls, h_walls, "left")
    right_ok = wall_follower_reaches(adj, v_walls, h_walls, "right")

    row = {
        "file": os.path.basename(npz_path),
        "seed": seed,
        "D_true": d_true,
        "detour_ratio": round(detour_ratio, 4) if detour_ratio is not None else None,
        "beta_independent_cycles": beta,
        "open_edges": edges,
        "dead_ends": dead_ends,
        "deg1": deg_dist.get(1, 0),
        "deg2": deg_dist.get(2, 0),
        "deg3": deg_dist.get(3, 0),
        "deg4": deg_dist.get(4, 0),
        "deg0": deg_dist.get(0, 0),
        "n_shortest_paths": n_shortest_paths,
        "goal_gateways": n_gateways,
        "left_hand_reaches_goal": left_ok,
        "right_hand_reaches_goal": right_ok,
        "wall_follow_reaches_goal": bool(left_ok or right_ok),
        "n_components": n_components,
        "all_cells_reachable": all_reachable,
        "goal_reachable": d_true is not None,
    }
    return row


def summarize(rows, label):
    lines = []
    lines.append(f"=== {label} (n={len(rows)}) ===")
    if not rows:
        lines.append("(データなし)")
        return "\n".join(lines)

    def col(name):
        return [r[name] for r in rows if r[name] is not None]

    def fmt_stats(name, unit=""):
        vals = col(name)
        if not vals:
            return f"{name}: データなし"
        med = statistics.median(vals)
        return f"{name}: 中央値={med}{unit}, 範囲=[{min(vals)}, {max(vals)}]{unit}"

    for name in ["D_true", "detour_ratio", "beta_independent_cycles", "dead_ends",
                 "n_shortest_paths", "goal_gateways", "open_edges"]:
        lines.append(fmt_stats(name))

    n_all_reach = sum(1 for r in rows if r["all_cells_reachable"])
    n_goal_reach = sum(1 for r in rows if r["goal_reachable"])
    n_left = sum(1 for r in rows if r["left_hand_reaches_goal"])
    n_right = sum(1 for r in rows if r["right_hand_reaches_goal"])
    n_wallfollow = sum(1 for r in rows if r["wall_follow_reaches_goal"])
    lines.append(f"全区画到達可能: {n_all_reach}/{len(rows)}")
    lines.append(f"ゴール到達可能(スタートから): {n_goal_reach}/{len(rows)}")
    lines.append(f"左手法でゴール到達: {n_left}/{len(rows)}")
    lines.append(f"右手法でゴール到達: {n_right}/{len(rows)}")
    lines.append(f"壁づたい(左手 or 右手)でゴール到達: {n_wallfollow}/{len(rows)}")

    return "\n".join(lines)


def main():
    datasets = [
        ("eval", os.path.join(REPO, "competition/mazes/eval")),
        ("validation", os.path.join(REPO, "competition/mazes/validation")),
        ("eval_v1_nonconforming", os.path.join(REPO, "competition/mazes/eval_v1_nonconforming")),
    ]

    all_rows = []
    per_dataset_rows = {}
    for label, dirpath in datasets:
        if not os.path.isdir(dirpath):
            print(f"[skip] {dirpath} は存在しない")
            continue
        npz_files = sorted(glob.glob(os.path.join(dirpath, "maze_1*.npz")) +
                            glob.glob(os.path.join(dirpath, "maze_4*.npz")))
        # maze_10*.npz や maze_4*.npz の指定に合わせて広めに拾う
        if not npz_files:
            npz_files = sorted(glob.glob(os.path.join(dirpath, "*.npz")))
        rows = []
        for f in npz_files:
            row = analyze_one(f)
            row["dataset"] = label
            rows.append(row)
            all_rows.append(row)
        per_dataset_rows[label] = rows
        print(f"[{label}] {len(rows)} 面 読み込み: {dirpath}")

    # CSV書き出し
    csv_path = os.path.join(OUTDIR, "ours_stats.csv")
    fieldnames = ["dataset", "file", "seed", "D_true", "detour_ratio",
                  "beta_independent_cycles", "open_edges", "dead_ends",
                  "deg0", "deg1", "deg2", "deg3", "deg4",
                  "n_shortest_paths", "goal_gateways",
                  "left_hand_reaches_goal", "right_hand_reaches_goal",
                  "wall_follow_reaches_goal", "n_components",
                  "all_cells_reachable", "goal_reachable"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    print(f"\nCSV書き出し: {csv_path} ({len(all_rows)} 行)")

    print()
    for label, _ in datasets:
        rows = per_dataset_rows.get(label, [])
        print(summarize(rows, label))
        print()

    # rule_audit.json とのクロスチェック (信頼せず、独自計算結果と突き合わせるだけ)
    print("=== rule_audit.json とのクロスチェック ===")
    for label, dirpath in datasets:
        audit_path = os.path.join(dirpath, "rule_audit.json")
        if not os.path.isfile(audit_path):
            continue
        with open(audit_path, encoding="utf-8") as f:
            audit = json.load(f)
        audit_by_seed = {a["seed"]: a for a in audit}
        rows = per_dataset_rows.get(label, [])
        mismatches = []
        for r in rows:
            a = audit_by_seed.get(r["seed"])
            if a is None:
                continue
            checks = [
                ("cycles", "beta_independent_cycles"),
                ("open_edges", "open_edges"),
                ("goal_gateways", "goal_gateways"),
                ("left_hand_reaches", "left_hand_reaches_goal"),
                ("right_hand_reaches", "right_hand_reaches_goal"),
                ("goal_reachable", "goal_reachable"),
            ]
            for ak, rk in checks:
                if ak in a and a[ak] != r[rk]:
                    mismatches.append((label, r["seed"], ak, a[ak], rk, r[rk]))
        if mismatches:
            print(f"[{label}] 不一致 {len(mismatches)} 件:")
            for m in mismatches:
                print("  ", m)
        else:
            print(f"[{label}] rule_audit.json と全項目一致 ({len(rows)} 面)")


if __name__ == "__main__":
    main()
