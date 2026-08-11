"""迷路が要求する「探索の遠回り」を、物理シミュレーション抜きで測る
Maze exploration cost: how far an optimistic flood-fill explorer must detour.

2026-08-11 新設。評価迷路の受理条件に**経路比 R** を加えるために用意する
（`competition/maze_gen_v3.py` から呼ばれる）。

## 何を測るか

**経路比 R = （未知の迷路を足立法で初めてゴールへ走ったときの移動区画数）/ D_0**

$D_0$ は壁が完全既知のときの最短距離。R = 1.0 は「初回探索がそのまま真の最短経路
だった」＝**探索が何も試されていない**ことを意味する。R が大きいほど、楽観的に
計画する探索器が遠回りを強いられる。

## なぜこの指標が要るのか

評価迷路 v2 は経路長（$D_0$）を大会実迷路に合わせたが、**探索の遠回りという軸では
是正前より実態から遠ざかった**（R 中央値: 是正前 1.216 → 是正後 1.016、大会 1.566）。
是正後の帯は **20 面中 10 面で初回探索がそのまま真の最短経路**であり、
**探索の質を測る土俵になっていない**（`research_notes/note_010_*.md`）。

## アルゴリズム（足立法・楽観地図）

古典アルゴリズムの定義から実装する:

1. 機体は現在区画に来た時点でその区画の 4 辺の壁を観測する（隣接区画側の同じ辺も既知に）
2. **未知の壁は「通れる」とみなす**（楽観地図）。楽観地図上でゴールからの距離を
   毎歩計算し直し、距離が最小になる隣接区画へ 1 区画進む
3. ゴール区画に入った時点で終了。移動した区画数を返す

**同点処理**（距離が同じ隣接区画が複数ある場合）で経路長は変わる。2 通り用意する:

- `"straight"`: 直進 > 右折 > 左折 > 後退（実機の定石）
- `"compass"`: 北 > 東 > 南 > 西（機首方位に依存しない固定順）

**受理条件には `"straight"` を使う**（実機の定石であり、機首方位を持つ実装と対応する）。
`"compass"` は**同点処理の癖に迷路を合わせてしまっていないか**を確かめる感度確認用
（教授指示・追加条件 B）。

## 独立実装との照合

准教授が独立に書いた `verification/maze_exploration_cost.py` と**同じアルゴリズム**を
別実装したもの。**依存はしない**（評価迷路は凍結物であり、生成が他セッションの
管理下にあるファイルに依存してはならないため）。両者が一致することは
`tests/test_explore_cost.py` で全数確認する。

使い方:
    from competition.explore_cost import detour_ratio
    R = detour_ratio(v_walls, h_walls)
"""
from collections import deque

import numpy as np

W = H = 16
# 北, 東, 南, 西
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
GOAL_CELLS = ((7, 7), (7, 8), (8, 7), (8, 8))
TIEBREAKS = ("straight", "compass")


def true_wall(v_walls, h_walls, x, y, d):
    """区画 (x,y) の方向 d 側に壁があるか（真の迷路）。1=壁、0=開通。"""
    if d == 0:
        return int(h_walls[x, y + 1])
    if d == 1:
        return int(v_walls[x + 1, y])
    if d == 2:
        return int(h_walls[x, y])
    return int(v_walls[x, y])


def _flood(known, is_wall, goals):
    """楽観地図（未知の壁は通れるとみなす）でゴールからの距離を計算する。"""
    dist = [[10 ** 6] * H for _ in range(W)]
    dq = deque()
    for gx, gy in goals:
        dist[gx][gy] = 0
        dq.append((gx, gy))
    while dq:
        x, y = dq.popleft()
        for d, (dx, dy) in enumerate(DIRS):
            if known[x][y][d] and is_wall[x][y][d]:
                continue          # 既知の壁は通れない。未知は通れるとみなす
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if dist[nx][ny] > dist[x][y] + 1:
                dist[nx][ny] = dist[x][y] + 1
                dq.append((nx, ny))
    return dist


def first_run_path(v_walls, h_walls, start=(0, 0), goals=GOAL_CELLS,
                   tiebreak="straight", max_steps=None):
    """未知の迷路を足立法で初めてゴールへ走ったときの**通過区画列**。

    返り値はスタートを含むセル座標のリスト（移動区画数はその長さ − 1）。
    到達できなければ None。図示にも使うので経路そのものを返す。
    """
    return _first_run(v_walls, h_walls, start, goals, tiebreak, max_steps)


def first_run_cells(v_walls, h_walls, start=(0, 0), goals=GOAL_CELLS,
                    tiebreak="straight", max_steps=None):
    """未知の迷路を足立法で初めてゴールへ走ったときの移動区画数。

    到達できない（打ち切りに達した／進める隣接区画が無い）場合は None。
    """
    path = _first_run(v_walls, h_walls, start, goals, tiebreak, max_steps)
    return None if path is None else len(path) - 1


def _first_run(v_walls, h_walls, start, goals, tiebreak, max_steps):
    if tiebreak not in TIEBREAKS:
        raise ValueError(f"tiebreak は {TIEBREAKS} のいずれか: {tiebreak!r}")
    goals = tuple(tuple(g) for g in goals)
    known = [[[False] * 4 for _ in range(H)] for _ in range(W)]
    is_wall = [[[False] * 4 for _ in range(H)] for _ in range(W)]

    def observe(x, y):
        for d, (dx, dy) in enumerate(DIRS):
            w = bool(true_wall(v_walls, h_walls, x, y, d))
            known[x][y][d] = True
            is_wall[x][y][d] = w
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H:
                known[nx][ny][(d + 2) % 4] = True
                is_wall[nx][ny][(d + 2) % 4] = w

    x, y = start
    heading = 0                      # 北向きで出発（スタート区画は 3 方向が壁）
    path = [(x, y)]
    limit = max_steps if max_steps is not None else 16 * W * H
    for _ in range(limit):
        if (x, y) in goals:
            return path
        observe(x, y)
        dist = _flood(known, is_wall, goals)
        order = ([heading, (heading + 1) % 4, (heading + 3) % 4, (heading + 2) % 4]
                 if tiebreak == "straight" else (0, 1, 2, 3))
        best, best_dir = None, None
        for d in order:
            if known[x][y][d] and is_wall[x][y][d]:
                continue
            dx, dy = DIRS[d]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if best is None or dist[nx][ny] < best:
                best, best_dir = dist[nx][ny], d
        if best_dir is None:
            return None
        heading = best_dir
        x, y = x + DIRS[best_dir][0], y + DIRS[best_dir][1]
        path.append((x, y))
    return None


def true_shortest(v_walls, h_walls, start=(0, 0), goals=GOAL_CELLS):
    """壁が完全既知のときの最短距離 D_0 [区画]。到達不能なら -1。"""
    goals = set(tuple(g) for g in goals)
    dist = {tuple(start): 0}
    dq = deque([tuple(start)])
    while dq:
        c = dq.popleft()
        if c in goals:
            return dist[c]
        for d, (dx, dy) in enumerate(DIRS):
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < W and 0 <= n[1] < H) or n in dist:
                continue
            if true_wall(v_walls, h_walls, c[0], c[1], d):
                continue
            dist[n] = dist[c] + 1
            dq.append(n)
    return -1


def true_shortest_path(v_walls, h_walls, start=(0, 0), goals=GOAL_CELLS):
    """壁が完全既知のときの最短経路（セル列）。図示用。到達不能なら None。

    複数の最短経路がある場合の選び方は決定的にする（近傍順 北→東→南→西 で
    最初に見つかった親をたどる）。
    """
    goals = set(tuple(g) for g in goals)
    prev = {tuple(start): None}
    dq = deque([tuple(start)])
    end = None
    while dq:
        c = dq.popleft()
        if c in goals:
            end = c
            break
        for d, (dx, dy) in enumerate(DIRS):
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < W and 0 <= n[1] < H) or n in prev:
                continue
            if true_wall(v_walls, h_walls, c[0], c[1], d):
                continue
            prev[n] = c
            dq.append(n)
    if end is None:
        return None
    path = [end]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return path[::-1]


def detour_ratio(v_walls, h_walls, start=(0, 0), goals=GOAL_CELLS, tiebreak="straight"):
    """経路比 R = 初回探索の移動区画数 / D_0。測れない場合は NaN。"""
    d0 = true_shortest(v_walls, h_walls, start, goals)
    if d0 <= 0:
        return float("nan")
    steps = first_run_cells(v_walls, h_walls, start, goals, tiebreak)
    return float("nan") if steps is None else steps / d0


def all_metrics(v_walls, h_walls, start=(0, 0), goals=GOAL_CELLS):
    """受理検査で使う量をまとめて返す（両方の同点処理での R を含む）。"""
    d0 = true_shortest(v_walls, h_walls, start, goals)
    out = {"d_true": int(d0)}
    for tb in TIEBREAKS:
        s = first_run_cells(v_walls, h_walls, start, goals, tb)
        out[f"cells_{tb}"] = None if s is None else int(s)
        out[f"R_{tb}"] = float("nan") if (s is None or d0 <= 0) else s / d0
    return out


def n_delta(v_walls, h_walls, deltas=(0, 2, 4, 6, 8), start=(0, 0), goals=GOAL_CELLS):
    """N_Δ = |{c : d(S,c) + d(c,G) <= D_0 + Δ}|（事後検査用。BFS 2 回）。"""
    def bfs(sources):
        dist = -np.ones((W, H), dtype=int)
        dq = deque()
        for c in sources:
            dist[c] = 0
            dq.append(tuple(c))
        while dq:
            x, y = dq.popleft()
            for d, (dx, dy) in enumerate(DIRS):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < W and 0 <= ny < H) or dist[nx, ny] >= 0:
                    continue
                if true_wall(v_walls, h_walls, x, y, d):
                    continue
                dist[nx, ny] = dist[x, y] + 1
                dq.append((nx, ny))
        return dist

    d0 = true_shortest(v_walls, h_walls, start, goals)
    ds, dg = bfs([tuple(start)]), bfs([tuple(g) for g in goals])
    ok = (ds >= 0) & (dg >= 0)
    return {f"N{dl}": int(np.sum(ok & (ds + dg <= d0 + dl))) for dl in deltas}
