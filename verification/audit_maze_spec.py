#!/usr/bin/env python3
"""迷路仕様の独立監査 — docs/RESEARCH_PLAN.md §2「迷路の規格」の各条件を npz から直接検査する。

作成: 2026-08-11 准教授セッション（独立検証担当）
`competition/` 配下の生成器・検査器の実装は参照せず、§2 の条文だけから実装した。

壁配列の規約（npz の shape から推定し、外周壁がすべて 1 であることで検証した）:
  v_walls[x, y] : x=0..16, y=0..15 — 区画 (x-1,y) と (x,y) の間の縦壁
  h_walls[x, y] : x=0..15, y=0..16 — 区画 (x,y-1) と (x,y) の間の横壁
  1 = 壁あり, 0 = 壁なし
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
N = 16


def load(p: Path):
    d = np.load(p, allow_pickle=True)
    m = {"v": d["v_walls"], "h": d["h_walls"]}
    if "start_x" in d.files:
        m["start"] = (int(d["start_x"]), int(d["start_y"]))
        m["goals"] = list(zip(d["goals_x"].tolist(), d["goals_y"].tolist()))
        m["start_inferred"] = False
    else:
        # 生成迷路の npz はスタート・ゴールを持たない。§2 の規定から復元する:
        # ゴール = 中央 2x2、スタート = 3 方向が壁になっている四隅。
        m["goals"] = [(7, 7), (8, 7), (7, 8), (8, 8)]
        corners = [(0, 0), (N - 1, 0), (0, N - 1), (N - 1, N - 1)]
        cand = [c for c in corners if len(open_dirs(m, *c)) == 1]
        m["start"] = cand[0] if len(cand) == 1 else (0, 0)
        m["start_inferred"] = True
        m["start_candidates"] = len(cand)
    return m


def open_dirs(m, x, y):
    """区画 (x,y) から出られる方向。(dx,dy) のリスト。"""
    out = []
    if not m["h"][x, y + 1]:
        out.append((0, 1))    # 北
    if not m["h"][x, y]:
        out.append((0, -1))   # 南
    if not m["v"][x + 1, y]:
        out.append((1, 0))    # 東
    if not m["v"][x, y]:
        out.append((-1, 0))   # 西
    return out


def bfs_dist(m, targets):
    """targets からの区画距離（到達不能は -1）。"""
    dist = -np.ones((N, N), dtype=int)
    q = deque()
    for (gx, gy) in targets:
        dist[gx, gy] = 0
        q.append((gx, gy))
    while q:
        x, y = q.popleft()
        for dx, dy in open_dirs(m, x, y):
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and dist[nx, ny] < 0:
                dist[nx, ny] = dist[x, y] + 1
                q.append((nx, ny))
    return dist


def wall_follow_reaches_goal(m, hand: str) -> bool:
    """壁づたい走行（左手法 / 右手法）がゴールに到達するか。

    スタート区画で北向きに開始し、手を壁に当てて進む。
    最大 4·16·16 ステップで打ち切り（(区画, 向き) の状態数の上限）。
    """
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]   # 北, 東, 南, 西
    goals = set(m["goals"])
    x, y = m["start"]
    d = 0
    seen = set()
    for _ in range(4 * N * N + 10):
        if (x, y) in goals:
            return True
        if (x, y, d) in seen:
            return False        # 周期に入った ＝ 到達しない
        seen.add((x, y, d))
        # 手のある側を優先して曲がる
        turn = -1 if hand == "left" else 1
        order = [(d + turn) % 4, d, (d - turn) % 4, (d + 2) % 4]
        for nd in order:
            dx, dy = dirs[nd]
            if (dx, dy) in open_dirs(m, x, y):
                d = nd
                x, y = x + dx, y + dy
                break
    return False


def audit(p: Path) -> dict:
    m = load(p)
    v, h = m["v"], m["h"]
    goals = set(m["goals"])
    r: dict = {"maze": p.stem, "start_inferred": m.get("start_inferred", False),
               "start_candidates": m.get("start_candidates")}

    # --- 外周壁がすべて存在する（配列規約の検証も兼ねる）
    r["outer_walls_ok"] = bool(v[0, :].all() and v[N, :].all() and h[:, 0].all() and h[:, N].all())

    # --- ゴール 4 区画: 中央 2x2 か / 内部に壁がないか
    gx = sorted({g[0] for g in goals})
    gy = sorted({g[1] for g in goals})
    r["goal_is_center2x2"] = (len(goals) == 4 and gx == [7, 8] and gy == [7, 8])
    inner = []
    if r["goal_is_center2x2"]:
        inner = [v[8, 7], v[8, 8], h[7, 8], h[8, 8]]
    r["goal_interior_wall_free"] = bool(sum(int(a) for a in inner) == 0) if inner else None

    # --- ゴールの入口数（2x2 の外周のうち開いている辺の数）
    gates = 0
    for (cx, cy) in goals:
        for dx, dy in open_dirs(m, cx, cy):
            nb = (cx + dx, cy + dy)
            if nb not in goals:
                gates += 1
    r["goal_gateways"] = gates
    r["goal_single_gateway"] = (gates == 1)

    # --- スタート区画: 四隅か / 3 方向が壁か
    sx, sy = m["start"]
    r["start_is_corner"] = (sx in (0, N - 1) and sy in (0, N - 1))
    r["start_walls"] = 4 - len(open_dirs(m, sx, sy))
    r["start_three_walls"] = (r["start_walls"] == 3)

    # --- 柱（格子点）: 中央の 1 点を除き、すべてに最低 1 枚の壁が接する
    lone = []
    for i in range(N + 1):
        for j in range(N + 1):
            touch = 0
            if j - 1 >= 0 and i <= N and v[i, j - 1]:
                touch += 1
            if j < N and v[i, j]:
                touch += 1
            if i - 1 >= 0 and j <= N and h[i - 1, j]:
                touch += 1
            if i < N and h[i, j]:
                touch += 1
            if touch == 0:
                lone.append((i, j))
    r["wall_free_lattice_points"] = lone
    r["lattice_ok"] = (lone == [(8, 8)])

    # --- 真の最短距離 D_true
    dist = bfs_dist(m, m["goals"])
    d0 = int(dist[sx, sy])
    r["d_true"] = d0
    r["reachable"] = d0 >= 0
    r["d_true_in_window_45_110"] = (45 <= d0 <= 110) if d0 >= 0 else False

    # --- 壁づたい走行で到達できてしまわないか
    r["wallfollow_left_reaches"] = wall_follow_reaches_goal(m, "left")
    r["wallfollow_right_reaches"] = wall_follow_reaches_goal(m, "right")
    r["wallfollow_safe"] = not (r["wallfollow_left_reaches"] or r["wallfollow_right_reaches"])

    # --- 複数経路（ループ）を持つか: 連結成分で 辺数 > 頂点数-1 なら閉路あり
    ncells = N * N
    edges = 0
    for x in range(N):
        for y in range(N):
            if not v[x + 1, y] and x + 1 < N:
                edges += 1
            if not h[x, y + 1] and y + 1 < N:
                edges += 1
    r["cells"] = ncells
    r["edges"] = edges
    r["cycles"] = edges - (ncells - 1)     # 全区画が連結である前提での独立閉路数
    r["has_loops"] = r["cycles"] > 0
    r["all_cells_reachable"] = bool((dist >= 0).all())

    r["conforms_all"] = all([
        r["outer_walls_ok"], r["goal_is_center2x2"], bool(r["goal_interior_wall_free"]),
        r["goal_single_gateway"], r["start_is_corner"], r["start_three_walls"],
        r["lattice_ok"], r["wallfollow_safe"], r["has_loops"], r["d_true_in_window_45_110"],
    ])
    return r


def main() -> None:
    bands = sys.argv[1:] or ["contest_reference", "eval", "validation"]
    all_rows: dict[str, list[dict]] = {}
    for band in bands:
        d = REPO / "competition" / "mazes" / band
        if not d.exists():
            print(f"[skip] {d} が無い")
            continue
        rows = [audit(p) for p in sorted(d.glob("*.npz"))]
        all_rows[band] = rows
        print(f"\n===== {band}  (n={len(rows)}) =====")
        hdr = (f"{'maze':<24}{'D0':>5}{'窓':>4}{'入口':>5}{'柱':>4}{'壁づたい':>9}"
               f"{'ループ':>7}{'ゴール内':>9}{'開始3壁':>8}{'総合':>6}")
        print(hdr)
        for r in rows:
            print(f"{r['maze'].replace('maze_',''):<24}{r['d_true']:>5}"
                  f"{'OK' if r['d_true_in_window_45_110'] else 'NG':>4}"
                  f"{r['goal_gateways']:>5}"
                  f"{'OK' if r['lattice_ok'] else 'NG':>4}"
                  f"{('安全' if r['wallfollow_safe'] else '到達可'):>9}"
                  f"{('OK' if r['has_loops'] else 'NG'):>7}"
                  f"{('OK' if r['goal_interior_wall_free'] else 'NG'):>9}"
                  f"{('OK' if r['start_three_walls'] else 'NG'):>8}"
                  f"{('OK' if r['conforms_all'] else 'NG'):>6}")
        # 条件ごとの違反件数
        keys = ["goal_is_center2x2", "goal_interior_wall_free", "goal_single_gateway",
                "start_is_corner", "start_three_walls", "lattice_ok", "wallfollow_safe",
                "has_loops", "d_true_in_window_45_110", "outer_walls_ok", "all_cells_reachable"]
        print("  -- 条件別の違反面数 --")
        for k in keys:
            bad = [r["maze"] for r in rows if not r.get(k)]
            if bad:
                print(f"     {k:<28} 違反 {len(bad):2d}/{len(rows)}: "
                      f"{', '.join(x.replace('maze_','') for x in bad)}")
            else:
                print(f"     {k:<28} 違反  0/{len(rows)}")

    outp = REPO / "verification" / "out" / "maze_spec_audit.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(all_rows, ensure_ascii=False, indent=1, default=str))
    print(f"\n書き出し: {outp}")


if __name__ == "__main__":
    main()
