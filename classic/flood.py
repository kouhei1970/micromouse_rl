"""
classic/flood.py
================
歩数マップ（足立法）。ゴール区画群から幅優先探索で各区画までの歩数を配る。

実機の定石として、**未知の壁の扱いを 2 通り使い分ける**（note_030 §3 S0）:

- 楽観 (OPTIMISTIC): 未知の壁は「通れる」とみなす。探索走行で使う
  （知らない区画にも足を延ばして情報を集めるため）。
- 悲観 (PESSIMISTIC): 未知の壁は「通れない」とみなす。最短走行で使う。
  地図がまだ不完全でも、**既知の壁だけで確実にたどり着けるか**を判定できる
  （= 悲観の歩数マップで到達不能なら、まだ地図が足りない）。
"""
from collections import deque
from enum import Enum
from typing import Iterable, Optional, Tuple

import numpy as np

from classic.maze_map import ALL_DIRECTIONS, MazeMap, WallState

Cell = Tuple[int, int]

# 歩数マップの「未到達」を表す値。np.inf ではなく整数の歩数マップとして
# 扱いたい場面が多いので、有限の巨大値ではなく float の inf を使う
# （比較・BFS 距離の格納に numpy 配列を使うため float 配列で統一する）。
UNREACHABLE = float("inf")


class FloodMode(str, Enum):
    """未知の壁をどちらに倒すか。"""

    OPTIMISTIC = "optimistic"  # 未知 = 通れる（探索走行）
    PESSIMISTIC = "pessimistic"  # 未知 = 通れない（最短走行）


def is_passable(maze: MazeMap, x: int, y: int, direction, mode: FloodMode) -> bool:
    """区画 (x,y) から direction 方向へ実際に移動できるか（モード込み）。

    route.py の経路復元でも同じ判定基準を使うため公開関数にしてある
    （歩数マップの計算と経路上のエッジ判定で基準がずれると、存在しない
    移動を経路に含めてしまう恐れがある）。
    """
    state = maze.get_wall(x, y, direction)
    if state == WallState.OPEN:
        return True
    if state == WallState.WALL:
        return False
    # UNKNOWN
    return mode == FloodMode.OPTIMISTIC


def compute_flood(
    maze: MazeMap,
    goals: Iterable[Cell],
    mode: FloodMode,
) -> np.ndarray:
    """ゴール区画群から各区画までの歩数マップを幅優先探索で計算する。

    Args:
        maze: 既知の壁地図。
        goals: ゴール区画（複数可）の座標のイテラブル。
        mode: 未知の壁の扱い（楽観／悲観）。

    Returns:
        形状 (width, height) の float 配列。dist[x,y] は区画 (x,y) から
        最も近いゴール区画までの歩数。到達不能な区画は np.inf。
        ゴール区画自身は 0。
    """
    dist = np.full((maze.width, maze.height), UNREACHABLE, dtype=np.float64)
    q: deque = deque()
    for gx, gy in goals:
        if not maze.in_bounds(gx, gy):
            raise ValueError(f"ゴール区画 ({gx},{gy}) が迷路の範囲外です")
        if dist[gx, gy] != UNREACHABLE:
            continue
        dist[gx, gy] = 0.0
        q.append((gx, gy))

    while q:
        x, y = q.popleft()
        d = dist[x, y]
        for direction in ALL_DIRECTIONS:
            nb = maze.neighbor(x, y, direction)
            if nb is None:
                continue
            nx, ny = nb
            if dist[nx, ny] != UNREACHABLE:
                continue
            # (x,y) から direction 方向への移動可否は、(x,y) 側の壁で判定
            # すれば十分（壁は共有物なので nx,ny 側から見ても同じ値になる）。
            if not is_passable(maze, x, y, direction, mode):
                continue
            dist[nx, ny] = d + 1.0
            q.append((nx, ny))

    return dist


def is_reachable_known(maze: MazeMap, start: Cell, goals: Iterable[Cell]) -> bool:
    """既知の壁だけで start からゴール群のいずれかへ確実に到達できるか。

    悲観モードの歩数マップで start が到達可能（有限歩数）かどうかで判定する。
    最短走行に移ってよいかどうかの判定に使う（note_030 §3 S0）。
    """
    dist = compute_flood(maze, goals, FloodMode.PESSIMISTIC)
    sx, sy = start
    if not maze.in_bounds(sx, sy):
        raise ValueError(f"開始区画 ({sx},{sy}) が迷路の範囲外です")
    return bool(dist[sx, sy] != UNREACHABLE)


def distance_at(dist: np.ndarray, cell: Cell) -> Optional[float]:
    """歩数マップから区画の歩数を取り出す。到達不能なら None。"""
    x, y = cell
    v = dist[x, y]
    return None if v == UNREACHABLE else float(v)
