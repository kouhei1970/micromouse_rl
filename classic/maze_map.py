"""
classic/maze_map.py
================
探索走行の途中で機体自身が持つ「既知の壁地図」。

実機のマウスは走行中に得た壁センサの情報を少しずつ書き足していく。まだ調べて
いない壁は「未知」であり、それを「無い（開いている）」と勝手に決め付けては
実機の再現にならない。本モジュールは各壁について

    未知 (UNKNOWN) ／ 壁あり (WALL) ／ 壁なし (OPEN)

の 3 値を持つ地図を提供する。

壁は隣接する 2 区画の**共有物**である（実機でも区画 (x,y) の東壁は区画
(x+1,y) の西壁と同じもの）。区画ごとに壁の状態を別々に持つ実装は、
「片側だけ更新して反対側が未知のまま」という不整合を作り得るため採らない。
本モジュールは壁そのものを 1 枚の配列に持ち、区画からは常にその配列を
参照する。

壁の格納方法は `competition/evaluator.py` の v_walls / h_walls と同じ添字
規約に合わせてある（該当箇所のコメント参照）:
    v_walls[x, y]: 区画 (x-1,y) と (x,y) の間の縦壁。形状 (width+1, height)。
                   x=0, x=width が外周。
    h_walls[x, y]: 区画 (x,y-1) と (x,y) の間の横壁。形状 (width, height+1)。
                   y=0, y=height が外周。
値は WallState（0=未知, 1=壁あり, 2=壁なし）。evaluator 側の 0/1 表現（0=壁
なし, 非0=壁あり）とは値の意味が異なるので混同しないこと。
"""
from enum import IntEnum
from typing import Dict, Optional, Tuple

import numpy as np


class WallState(IntEnum):
    """1 枚の壁がとりうる 3 状態。"""

    UNKNOWN = 0
    WALL = 1
    OPEN = 2


class Direction(IntEnum):
    """区画から見た 4 方位。反時計回りに E→N→W→S の順で並べてある
    （工学系の流儀・2026-08-23 ユーザ決定。ヨー角は「0=東・反時計回りが正」
    （docs/COORDINATE_SYSTEM.md §7）であり、+1 で左回り90°、-1 で右回り90°、
    +2 で180°折返しになるように選んだ順序。route.py のターン種別の判定は
    この順序に依存する）。"""

    E = 0
    N = 1
    W = 2
    S = 3


# 各方位に対応する (dx, dy)。座標系は x=東方向、y=北方向。
_DIR_DELTA: Dict[Direction, Tuple[int, int]] = {
    Direction.E: (1, 0),
    Direction.N: (0, 1),
    Direction.W: (-1, 0),
    Direction.S: (0, -1),
}

# 探索順（N→E→S→W）。route.py の shortest_path 等はタイブレークをこの列挙
# 順に依存するため、Direction の数値付け替え（2026-08-23）後もこの並びは
# 変えていない（named member による指定なので、数値が変わっても走査順は
# 変わらない）。
ALL_DIRECTIONS: Tuple[Direction, ...] = (Direction.N, Direction.E, Direction.S, Direction.W)


def opposite(direction: Direction) -> Direction:
    """反対方位（N<->S, E<->W）を返す。"""
    return Direction((direction + 2) % 4)


def direction_between(a: Tuple[int, int], b: Tuple[int, int]) -> Direction:
    """隣接する区画 a→b の移動方向を返す。隣接していなければ ValueError。"""
    delta = (b[0] - a[0], b[1] - a[1])
    for direction, d in _DIR_DELTA.items():
        if d == delta:
            return direction
    raise ValueError(f"{a} -> {b} は隣接区画ではありません（1 区画分の移動のみ許容）")


class MazeMap:
    """既知の壁地図（3 値・共有壁）。

    Attributes:
        width: 迷路の幅（区画数）。
        height: 迷路の高さ（区画数）。
        v_walls: 縦壁の状態。形状 (width+1, height)。
        h_walls: 横壁の状態。形状 (width, height+1)。
    """

    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width, height は正の整数である必要があります")
        self.width = int(width)
        self.height = int(height)

        # 内部は全て未知で初期化し、外周だけ既知の壁として書き込む。
        self.v_walls = np.full((self.width + 1, self.height), WallState.UNKNOWN, dtype=np.int8)
        self.h_walls = np.full((self.width, self.height + 1), WallState.UNKNOWN, dtype=np.int8)

        self.v_walls[0, :] = WallState.WALL
        self.v_walls[self.width, :] = WallState.WALL
        self.h_walls[:, 0] = WallState.WALL
        self.h_walls[:, self.height] = WallState.WALL

    # ------------------------------------------------------------------
    # 座標のユーティリティ
    # ------------------------------------------------------------------
    def in_bounds(self, x: int, y: int) -> bool:
        """区画 (x,y) が迷路の範囲内かどうか。"""
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbor(self, x: int, y: int, direction: Direction) -> Optional[Tuple[int, int]]:
        """区画 (x,y) から direction 方向に隣接する区画の座標を返す。
        迷路の外に出る場合は None。"""
        dx, dy = _DIR_DELTA[direction]
        nx, ny = x + dx, y + dy
        if not self.in_bounds(nx, ny):
            return None
        return nx, ny

    def _wall_index(self, x: int, y: int, direction: Direction) -> Tuple[np.ndarray, Tuple[int, int]]:
        """(x,y) の direction 側の壁が格納されている配列と添字を返す。

        壁は隣接 2 区画の共有物なので、区画+方位から必ず一意の格納先が
        決まる（例えば区画(x,y)の東壁と区画(x+1,y)の西壁は同じ添字を返す）。
        """
        if not self.in_bounds(x, y):
            raise ValueError(f"区画 ({x},{y}) は迷路の範囲外です")
        if direction is Direction.E:
            return self.v_walls, (x + 1, y)
        if direction is Direction.W:
            return self.v_walls, (x, y)
        if direction is Direction.N:
            return self.h_walls, (x, y + 1)
        if direction is Direction.S:
            return self.h_walls, (x, y)
        raise ValueError(f"未知の方位: {direction!r}")

    # ------------------------------------------------------------------
    # 読み書き API
    # ------------------------------------------------------------------
    def get_wall(self, x: int, y: int, direction: Direction) -> WallState:
        """区画 (x,y) の direction 側の壁状態を返す。"""
        arr, idx = self._wall_index(x, y, direction)
        return WallState(int(arr[idx]))

    def set_wall(self, x: int, y: int, direction: Direction, state: WallState) -> None:
        """区画 (x,y) の direction 側の壁状態を書き込む。

        壁は共有物なので、隣接区画側から見ても同じ状態が読めるようになる
        （片側だけ更新される不整合は構造上起こらない）。
        """
        arr, idx = self._wall_index(x, y, direction)
        arr[idx] = WallState(state)

    def walls_around(self, x: int, y: int) -> Dict[Direction, WallState]:
        """区画 (x,y) から見た 4 方位すべての壁状態を返す。"""
        return {d: self.get_wall(x, y, d) for d in ALL_DIRECTIONS}

    def is_open(self, x: int, y: int, direction: Direction) -> bool:
        """壁が「無い（開いている）」と既知であるかどうか。未知は False。"""
        return self.get_wall(x, y, direction) == WallState.OPEN

    def is_wall(self, x: int, y: int, direction: Direction) -> bool:
        """壁が「有る」と既知であるかどうか。未知は False。"""
        return self.get_wall(x, y, direction) == WallState.WALL

    def is_unknown(self, x: int, y: int, direction: Direction) -> bool:
        """壁がまだ未知かどうか。"""
        return self.get_wall(x, y, direction) == WallState.UNKNOWN

    # ------------------------------------------------------------------
    # 全既知フラグ（探索走行の完了判定などに使う）
    # ------------------------------------------------------------------
    def all_known(self) -> bool:
        """迷路全体の壁が（外周も含めて）すべて既知かどうか。"""
        return bool(np.all(self.v_walls != WallState.UNKNOWN)) and bool(
            np.all(self.h_walls != WallState.UNKNOWN)
        )
