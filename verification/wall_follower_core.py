#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
壁伝い法と位相の判定の中核（**迷路の壁の表現に依存しない部分**）

准教授セッション（8 代目）・2026-08-14・`AUDIT_040` の一部

このファイルは **迷路生成器の壁の形式を見る前に書いた**。
壁の情報は `blocked(cell, direction) -> bool` という 1 つの述語だけで受け取り、
配列のインデックス規約には一切触れない。**アルゴリズムの独立性を、
相手の実装だけでなく迷路の表現からも切り離すため**である。

The wall-following and topology routines below take walls only through a
`blocked(cell, dir)` predicate, so they are independent of the maze array layout.
"""

# 向き: 0=東(+x) 1=北(+y) 2=西(-x) 3=南(-y)
# directions: 0=East(+x) 1=North(+y) 2=West(-x) 3=South(-y)
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
DIR_NAMES = ["東", "北", "西", "南"]


def wall_follow(start, start_dir, goal_cells, blocked, max_steps=1_000_000, hand="left"):
    """壁伝い法を格子上で決定的に走らせる。

    Parameters
    ----------
    start : (int, int)          出発区画
    start_dir : int             出発時の向き（0..3）
    goal_cells : set            ゴール区画の集合
    blocked : callable          blocked((x, y), d) -> True なら区画 (x,y) から向き d へ進めない
    max_steps : int             歩数の上限（打ち切り用）
    hand : "left" | "right"     どちらの手を壁につけるか

    Returns
    -------
    dict:
      reached   : ゴールへ到達したか
      reason    : "goal" / "loop"（巡回の検出） / "step_limit"（歩数の上限）
      steps     : 実行した歩数
      visited   : 訪れた区画の集合
      n_states  : 訪れた (区画, 向き) の状態数

    壁伝いの規則（左手法の場合）:
      毎歩、**左 → 正面 → 右 → 後ろ** の順に進める向きを探し、最初に進める向きへ 1 区画進む。
      これは「左手を壁につけたまま歩く」の標準的な格子上の定式化である。
      右手法は左右を入れ替える（右 → 正面 → 左 → 後ろ）。

    **決定的なので、(区画, 向き) の状態が再訪された時点で以後は同じ軌道の繰り返しになる。**
    したがって巡回を検出したら「決して届かない」が確定する（カード §1-2 と同じ論拠）。
    """
    if hand == "left":
        turn_order = (1, 0, -1, 2)   # 左, 正面, 右, 後ろ（向きの加算量）
    elif hand == "right":
        turn_order = (-1, 0, 1, 2)   # 右, 正面, 左, 後ろ
    else:
        raise ValueError(f"hand は 'left' か 'right': {hand!r}")

    cell, d = tuple(start), int(start_dir)
    visited = {cell}
    seen_states = {(cell, d)}

    if cell in goal_cells:
        return dict(reached=True, reason="goal", steps=0,
                    visited=visited, n_states=len(seen_states))

    for step in range(1, max_steps + 1):
        for turn in turn_order:
            nd = (d + turn) % 4
            if not blocked(cell, nd):
                d = nd
                dx, dy = DIRS[nd]
                cell = (cell[0] + dx, cell[1] + dy)
                break
        else:
            # 四方すべて壁（孤立した区画）。動けないので確定で未到達
            return dict(reached=False, reason="loop", steps=step - 1,
                        visited=visited, n_states=len(seen_states))

        visited.add(cell)
        if cell in goal_cells:
            return dict(reached=True, reason="goal", steps=step,
                        visited=visited, n_states=len(seen_states))

        state = (cell, d)
        if state in seen_states:
            # 決定的な遷移なので、以後は同じ軌道を無限に繰り返す = 決して届かない
            return dict(reached=False, reason="loop", steps=step,
                        visited=visited, n_states=len(seen_states))
        seen_states.add(state)

    return dict(reached=False, reason="step_limit", steps=max_steps,
                visited=visited, n_states=len(seen_states))


def wall_segments(width, height, blocked):
    """迷路の壁を「線分」の集合として取り出す。

    区画の格子とは別に、**壁の格子点（柱の位置）**を節点として壁の連結성を見るための前処理。
    格子点は (0..width, 0..height) の (width+1) × (height+1) 個。

    縦の壁: 区画 (x, y) の東側の壁 = 格子点 (x+1, y) と (x+1, y+1) を結ぶ線分
    横の壁: 区画 (x, y) の北側の壁 = 格子点 (x, y+1) と (x+1, y+1) を結ぶ線分

    外周の壁も含めて返す（外周は必ず存在する前提を置かず、blocked から読む）。
    """
    segs = set()
    for x in range(width):
        for y in range(height):
            if blocked((x, y), 0):   # 東
                segs.add(((x + 1, y), (x + 1, y + 1)))
            if blocked((x, y), 1):   # 北
                segs.add(((x, y + 1), (x + 1, y + 1)))
            if blocked((x, y), 2):   # 西
                segs.add(((x, y), (x, y + 1)))
            if blocked((x, y), 3):   # 南
                segs.add(((x, y), (x + 1, y)))
    return segs


def wall_components(width, height, blocked):
    """壁の線分を、格子点を共有するかどうかで連結成分に分ける。

    Returns
    -------
    (comp_of_point, n_components)
      comp_of_point : {格子点: 成分番号}

    ⚠️ **定義の選択（カードは明記していない）**: 2 本の壁が**格子点（柱）を共有していれば
    繋がっている**とみなす。柱そのものが立っているかどうかは問わない。
    これは「壁を辿る手が乗り移れるか」に対応する自然な定義だが、**唯一の定義ではない**。
    別の定義（柱が立っている場合のみ繋がる 等）で数が変わるかは W-11 で調べる。
    """
    segs = wall_segments(width, height, blocked)
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for p, q in segs:
        union(p, q)

    comp_of_point = {p: find(p) for p in parent}
    roots = {}
    out = {}
    for p, r in comp_of_point.items():
        out[p] = roots.setdefault(r, len(roots))
    return out, len(roots)


def goal_walls_connected_to_outer(width, height, goal_cells, blocked):
    """「ゴールを囲む壁」が「外周の壁」と同じ連結成分にあるか。

    ゴールを囲む壁 = ゴール区画のいずれかに接している壁の線分。
    外周の壁 = 迷路の外枠をなす線分。

    Returns
    -------
    (connected: bool, n_goal_wall_segments: int)

    ゴール区画に接する壁が 1 本も無い場合は「囲まれていない」ので connected=True を返す
    （壁で隔てられていない ＝ 外から入れる）。
    """
    comp_of_point, _ = wall_components(width, height, blocked)

    # 外周の壁が属する成分（外周は連結なので 1 つの成分になるはず）
    outer_points = set()
    for x in range(width + 1):
        outer_points.add((x, 0))
        outer_points.add((x, height))
    for y in range(height + 1):
        outer_points.add((0, y))
        outer_points.add((width, y))
    outer_comps = {comp_of_point[p] for p in outer_points if p in comp_of_point}

    goal_segs = set()
    for (gx, gy) in goal_cells:
        for d in range(4):
            if blocked((gx, gy), d):
                if d == 0:
                    goal_segs.add(((gx + 1, gy), (gx + 1, gy + 1)))
                elif d == 1:
                    goal_segs.add(((gx, gy + 1), (gx + 1, gy + 1)))
                elif d == 2:
                    goal_segs.add(((gx, gy), (gx, gy + 1)))
                else:
                    goal_segs.add(((gx, gy), (gx + 1, gy)))

    if not goal_segs:
        return True, 0

    goal_comps = {comp_of_point[p] for seg in goal_segs for p in seg if p in comp_of_point}
    return bool(goal_comps & outer_comps), len(goal_segs)
