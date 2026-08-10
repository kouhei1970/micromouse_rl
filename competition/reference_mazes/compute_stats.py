#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大会迷路(16x16, 42面)の静的指標を計算する。

配列規約は npz 形式に準拠:
  v_walls[x,y] shape(17,16): 区画(x-1,y)と(x,y)の間の縦壁 (1=壁, 0=通行可)
  h_walls[x,y] shape(16,17): 区画(x,y-1)と(x,y)の間の横壁 (1=壁, 0=通行可)
既存の /Users/kouhei/tmp/github/micromouse_rl/competition/audit_maze_rules.py
(現行20面の評価迷路の規定準拠検査スクリプト) と同じ壁配列規約・同じ規定判定
ロジックを踏襲しつつ、ゴール位置が迷路ごとに異なる(中央固定でない)大会迷路
向けに一般化する。リポジトリ内ファイルは読むのみで一切変更しない。
"""
from __future__ import annotations
import csv
import glob
import json
import os
from collections import deque

import numpy as np

CONTEST_DIR = os.path.dirname(os.path.abspath(__file__)) + "/contest"
OUT_CSV = os.path.dirname(os.path.abspath(__file__)) + "/contest_stats.csv"

SIZE = 16
V = SIZE * SIZE  # 256


def _open(v, h, a, b):
    """セル a,b (隣接前提) の間が開通しているか。"""
    (ax, ay), (bx, by) = a, b
    if ax == bx:
        return h[ax, max(ay, by)] == 0
    return v[max(ax, bx), ay] == 0


def cell_degree(v, h, x, y):
    """セル(x,y)の開通辺数 (0-4)。境界セルも v/h 配列の境界壁(通常1)で自動処理される。"""
    d = 0
    if v[x, y] == 0:       # 西
        d += 1
    if v[x + 1, y] == 0:   # 東
        d += 1
    if h[x, y] == 0:       # 南
        d += 1
    if h[x, y + 1] == 0:   # 北
        d += 1
    return d


def bfs_from_start(v, h, start):
    """スタートからの BFS。 (dist dict, order list) を返す。"""
    dist = {start: 0}
    order = [start]
    dq = deque([start])
    while dq:
        c = dq.popleft()
        cx, cy = c
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            n = (cx + dx, cy + dy)
            if 0 <= n[0] < SIZE and 0 <= n[1] < SIZE and n not in dist and _open(v, h, c, n):
                dist[n] = dist[c] + 1
                order.append(n)
                dq.append(n)
    return dist, order


def count_shortest_paths(v, h, start, goal_cells, dist, order):
    """BFS距離DAG上で最短経路本数を数える(標準的なBFSカウント法)。"""
    cnt = {start: 1}
    for c in order:
        if c == start:
            continue
        cx, cy = c
        total = 0
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            p = (cx - dx, cy - dy)
            if p in dist and dist[p] == dist[c] - 1 and _open(v, h, p, c):
                total += cnt.get(p, 0)
        cnt[c] = total
    d_true = min((dist[g] for g in goal_cells if g in dist), default=None)
    if d_true is None:
        return None, 0
    n_paths = sum(cnt.get(g, 0) for g in goal_cells if g in dist and dist[g] == d_true)
    return d_true, n_paths


def connected_components(v, h):
    """全256区画を対象に開通グラフの連結成分数 C と開通辺数 E を返す。"""
    seen = set()
    n_comp = 0
    for sx in range(SIZE):
        for sy in range(SIZE):
            s = (sx, sy)
            if s in seen:
                continue
            n_comp += 1
            dq = deque([s])
            seen.add(s)
            while dq:
                c = dq.popleft()
                cx, cy = c
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    n = (cx + dx, cy + dy)
                    if 0 <= n[0] < SIZE and 0 <= n[1] < SIZE and n not in seen and _open(v, h, c, n):
                        seen.add(n)
                        dq.append(n)
    open_edges = int((v[1:16, :] == 0).sum() + (h[:, 1:16] == 0).sum())
    return n_comp, open_edges


def goal_gateway_count(v, h, goal_cells):
    """ゴール2x2の外周8辺のうち開いている数。既存 audit_maze_rules.py と同一ロジック
    (ゴール集合のみパラメータ化)。"""
    n = 0
    for (cx, cy) in sorted(goal_cells):
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in goal_cells:
                continue
            if not (0 <= nx < SIZE and 0 <= ny < SIZE):
                continue  # 外周壁(通常ゴールは中央付近なので基本発生しない)
            if _open(v, h, (cx, cy), (nx, ny)):
                n += 1
    return n


def goal_center_post(goal_cells):
    """ゴール2x2に共通して接する格子点(中央柱位置)。 (gx,gy)を左下セルとして(gx+1,gy+1)。"""
    xs = sorted(set(c[0] for c in goal_cells))
    ys = sorted(set(c[1] for c in goal_cells))
    assert len(xs) == 2 and len(ys) == 2, f"ゴールが2x2でない: {goal_cells}"
    return xs[1], ys[1]


def goal_interior_walls(v, h, goal_cells):
    """ゴール4区画の内部にある壁の数(規定では0=通行可)。"""
    xs = sorted(set(c[0] for c in goal_cells))
    ys = sorted(set(c[1] for c in goal_cells))
    gx0, gx1 = xs
    gy0, gy1 = ys
    return (int(v[gx1, gy0] == 1) + int(v[gx1, gy1] == 1)
            + int(h[gx0, gy1] == 1) + int(h[gx1, gy1] == 1))


def isolated_posts(v, h, exclude=None):
    """どの壁とも接していない格子点の数(除外リスト指定可)。"""
    exclude = exclude or set()
    n = 0
    for px in range(SIZE + 1):
        for py in range(SIZE + 1):
            if (px, py) in exclude:
                continue
            attached = False
            if py < SIZE and v[px, py] == 1:
                attached = True
            if py > 0 and v[px, py - 1] == 1:
                attached = True
            if px < SIZE and h[px, py] == 1:
                attached = True
            if px > 0 and h[px - 1, py] == 1:
                attached = True
            if not attached:
                n += 1
    return n


def center_post_attached_walls(v, h, center_post):
    px, py = center_post
    return (int(v[px, py] == 1) + int(v[px, py - 1] == 1)
            + int(h[px, py] == 1) + int(h[px - 1, py] == 1))


def outer_walls_complete(v, h):
    return bool(v[0, :].all() and v[SIZE, :].all() and h[:, 0].all() and h[:, SIZE].all())


def start_open_directions(v, h, start):
    """スタート区画の開口方向のリスト(0=北,1=東,2=南,3=西)と壁枚数を返す。"""
    sx, sy = start
    opens = []
    if h[sx, sy + 1] == 0:
        opens.append(0)
    if v[sx + 1, sy] == 0:
        opens.append(1)
    if h[sx, sy] == 0:
        opens.append(2)
    if v[sx, sy] == 0:
        opens.append(3)
    walls = 4 - len(opens)
    return opens, walls


def wall_follow_reaches_goal(v, h, start, goal_cells, init_head, hand="left"):
    """左手法/右手法でゴールに到達できるか(既存 audit_maze_rules.py と同一アルゴリズム。
    初期向きのみ実データから決定した init_head を使う、パラメータ化)。"""
    d_vec = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
    order = [-1, 0, 1, 2] if hand == "left" else [1, 0, -1, 2]

    def can_go(cell, d):
        cx, cy = cell
        dx, dy = d_vec[d]
        nx, ny = cx + dx, cy + dy
        if not (0 <= nx < SIZE and 0 <= ny < SIZE):
            return None
        return (nx, ny) if _open(v, h, cell, (nx, ny)) else None

    cell, head = start, init_head
    seen = set()
    for _ in range(100000):
        if cell in goal_cells:
            return True
        state = (cell, head)
        if state in seen:
            return False
        seen.add(state)
        moved = False
        for turn in order:
            nd = (head + turn) % 4
            nxt = can_go(cell, nd)
            if nxt is not None:
                cell, head = nxt, nd
                moved = True
                break
        if not moved:
            return False
    return False


def analyze_one(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    v = d["v_walls"]
    h = d["h_walls"]
    start = (int(d["start_x"]), int(d["start_y"]))
    goal_cells = set(zip((int(x) for x in d["goals_x"]), (int(y) for y in d["goals_y"])))
    source = str(d["source_file"]) if "source_file" in d else os.path.basename(npz_path)

    dist, order = bfs_from_start(v, h, start)
    d_true, n_paths = count_shortest_paths(v, h, start, goal_cells, dist, order)
    manhattan = min(abs(gx - start[0]) + abs(gy - start[1]) for gx, gy in goal_cells)
    detour = (d_true / manhattan) if (d_true is not None and manhattan > 0) else None
    detour_fixed14 = (d_true / 14) if d_true is not None else None

    n_reach_from_start = len(dist)
    all_reachable = (n_reach_from_start == V)

    n_comp, open_edges = connected_components(v, h)
    beta = open_edges - V + n_comp

    deg = [cell_degree(v, h, x, y) for x in range(SIZE) for y in range(SIZE)]
    deg_hist = {k: sum(1 for x in deg if x == k) for k in range(5)}
    dead_ends = deg_hist[1]

    gw = goal_gateway_count(v, h, goal_cells)
    center_post = goal_center_post(goal_cells)
    giw = goal_interior_walls(v, h, goal_cells)
    cpw = center_post_attached_walls(v, h, center_post)
    iso = isolated_posts(v, h, exclude={center_post})
    outer_ok = outer_walls_complete(v, h)
    opens, start_walls = start_open_directions(v, h, start)
    # 初期向き: 北開口があれば北、なければ実際に開いている方向のうち規約順(北>東>南>西)で採用
    if 0 in opens:
        init_head = 0
    elif opens:
        init_head = opens[0]
    else:
        init_head = 0  # 開口なし(規定違反)、便宜上北

    left_reach = wall_follow_reaches_goal(v, h, start, goal_cells, init_head, "left")
    right_reach = wall_follow_reaches_goal(v, h, start, goal_cells, init_head, "right")

    # 規定準拠判定 (6項目)
    rule_gateway1 = (gw == 1)
    rule_wallfollow_fails = (not left_reach) and (not right_reach)
    rule_multi_path = (beta > 0)
    rule_no_isolated_post = (iso == 0)
    rule_outer_complete = outer_ok
    rule_start3walls = (start_walls == 3)
    n_rules_ok = sum([rule_gateway1, rule_wallfollow_fails, rule_multi_path,
                       rule_no_isolated_post, rule_outer_complete, rule_start3walls])

    return dict(
        source=source,
        start=start,
        goal_cells=sorted(goal_cells),
        n_comp_total=n_comp,
        d_true=d_true,
        manhattan=manhattan,
        detour=detour,
        detour_fixed14=detour_fixed14,
        beta=beta,
        open_edges=open_edges,
        n_paths=n_paths,
        deg1=deg_hist[1], deg2=deg_hist[2], deg3=deg_hist[3], deg4=deg_hist[4], deg0=deg_hist[0],
        dead_ends=dead_ends,
        goal_gateways=gw,
        goal_interior_walls=giw,
        center_post_walls=cpw,
        isolated_posts=iso,
        outer_ok=outer_ok,
        start_walls=start_walls,
        start_open_dirs=opens,
        init_head_used=init_head,
        left_reach=left_reach,
        right_reach=right_reach,
        all_reachable_from_start=all_reachable,
        n_reach_from_start=n_reach_from_start,
        rule_gateway1=rule_gateway1,
        rule_wallfollow_fails=rule_wallfollow_fails,
        rule_multi_path=rule_multi_path,
        rule_no_isolated_post=rule_no_isolated_post,
        rule_outer_complete=rule_outer_complete,
        rule_start3walls=rule_start3walls,
        n_rules_ok=n_rules_ok,
    )


def main():
    files = sorted(glob.glob(os.path.join(CONTEST_DIR, "contest_*.npz")))
    assert files, f"npzファイルが見つからない: {CONTEST_DIR}"
    rows = [analyze_one(f) for f in files]

    fieldnames = [
        "source", "start", "d_true", "manhattan", "detour", "detour_fixed14",
        "beta", "open_edges", "n_comp_total", "n_paths",
        "deg0", "deg1", "deg2", "deg3", "deg4", "dead_ends",
        "goal_gateways", "goal_interior_walls", "center_post_walls", "isolated_posts",
        "outer_ok", "start_walls", "start_open_dirs", "init_head_used",
        "left_reach", "right_reach", "all_reachable_from_start", "n_reach_from_start",
        "rule_gateway1", "rule_wallfollow_fails", "rule_multi_path", "rule_no_isolated_post",
        "rule_outer_complete", "rule_start3walls", "n_rules_ok",
        "goal_cells",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = dict(r)
            row["start_open_dirs"] = ",".join(str(x) for x in r["start_open_dirs"])
            row["goal_cells"] = ";".join(f"{x}-{y}" for x, y in r["goal_cells"])
            w.writerow(row)

    print(f"書き出し: {OUT_CSV} ({len(rows)}行)")

    # ---- 集計をターミナルにも出す ----
    import statistics as st
    n = len(rows)
    print(f"\n=== 大会迷路 {n} 面 静的指標 集計 ===\n")

    def summarize(key, label, fmt="{:.2f}"):
        vals = [r[key] for r in rows if r[key] is not None]
        print(f"{label}: 中央値={fmt.format(st.median(vals))}  "
              f"範囲={fmt.format(min(vals))}〜{fmt.format(max(vals))}  (n={len(vals)}/{n})")

    summarize("d_true", "真の最短距離 D_true", "{:.0f}")
    summarize("detour", "迂回率(自迷路のゴール位置基準)", "{:.3f}")
    summarize("detour_fixed14", "迂回率(固定分母14基準・参考)", "{:.3f}")
    summarize("beta", "独立閉路数 β", "{:.0f}")
    summarize("dead_ends", "行き止まり数(次数1)", "{:.0f}")
    summarize("n_paths", "最短経路の本数", "{:.0f}")
    summarize("goal_gateways", "ゴール入口数", "{:.0f}")

    print(f"\n壁づたい(左手法)でゴール到達: {sum(1 for r in rows if r['left_reach'])}/{n}")
    print(f"壁づたい(右手法)でゴール到達: {sum(1 for r in rows if r['right_reach'])}/{n}")
    print(f"全区画到達可能: {sum(1 for r in rows if r['all_reachable_from_start'])}/{n}")
    print(f"連結成分数=1 (全体が1つに繋がっている): {sum(1 for r in rows if r['n_comp_total'] == 1)}/{n}")

    print("\n--- 規定準拠 (6項目) ---")
    print(f"1. ゴール入口1箇所: {sum(1 for r in rows if r['rule_gateway1'])}/{n}")
    print(f"2. 壁づたいで到達不可(両手法とも失敗): {sum(1 for r in rows if r['rule_wallfollow_fails'])}/{n}")
    print(f"3. 複数経路(β>0): {sum(1 for r in rows if r['rule_multi_path'])}/{n}")
    print(f"4. 孤立柱なし(中央除く): {sum(1 for r in rows if r['rule_no_isolated_post'])}/{n}")
    print(f"5. 外周壁完備: {sum(1 for r in rows if r['rule_outer_complete'])}/{n}")
    print(f"6. スタート3方向壁: {sum(1 for r in rows if r['rule_start3walls'])}/{n}")
    print(f"全6項目適合: {sum(1 for r in rows if r['n_rules_ok'] == 6)}/{n}")

    print("\n--- ゴール内壁0枚(参考): "
          f"{sum(1 for r in rows if r['goal_interior_walls'] == 0)}/{n}")
    print("--- ゴール中央柱に壁が0本接続(NTF規定・参考): "
          f"{sum(1 for r in rows if r['center_post_walls'] == 0)}/{n}")

    # 非適合の詳細を列挙
    print("\n--- 非適合明細 ---")
    for r in rows:
        bad = []
        if not r["rule_gateway1"]:
            bad.append(f"入口{r['goal_gateways']}箇所")
        if not r["rule_wallfollow_fails"]:
            wf = []
            if r["left_reach"]:
                wf.append("左手法到達")
            if r["right_reach"]:
                wf.append("右手法到達")
            bad.append("壁づたいで解ける(" + ",".join(wf) + ")")
        if not r["rule_multi_path"]:
            bad.append("単一経路(β=0)")
        if not r["rule_no_isolated_post"]:
            bad.append(f"孤立柱{r['isolated_posts']}箇所")
        if not r["rule_outer_complete"]:
            bad.append("外周壁欠落")
        if not r["rule_start3walls"]:
            bad.append(f"スタート壁{r['start_walls']}枚")
        if bad:
            print(f"  {r['source']}: " + " / ".join(bad))
    if all(r["n_rules_ok"] == 6 for r in rows):
        print("  (非適合面なし)")

    # JSON詳細も残す(デバッグ用)
    out_json = os.path.join(os.path.dirname(OUT_CSV), "contest_stats_detail.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n詳細JSON: {out_json}")


if __name__ == "__main__":
    main()
