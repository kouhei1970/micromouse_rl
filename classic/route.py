"""
classic/route.py
================
歩数マップから最短経路（区画の列）を求め、実機の語彙に合わせた
**ターン種別の列**へ変換する。

実機の定石（note_030 §3 S0）:
- 同じ歩数の経路が複数あるときは、**ターン数が少ない方**を選ぶ（速いから）。
- 連続する直進は「直進 N 区画」にまとめる（実機は直線を伸ばして加速する）。

ターン種別の語彙は次の 5 つに固定する（docs/JA_ENGINEERING_TERMS.md の
競技用語に合わせる）:
    直進 N 区画 / 90° 右 / 90° 左 / 180° 折返し（その場旋回） / ゴール停止
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from classic.flood import FloodMode, compute_flood, is_passable
from classic.maze_map import ALL_DIRECTIONS, Direction, MazeMap, direction_between

Cell = Tuple[int, int]


class NoRouteError(Exception):
    """歩数マップ上で start からゴールへ到達できない場合に送出する。"""


# ==========================================================================
# 区画の列としての最短経路（歩数マップに基づく。ターン数最小のタイブレーク付き）
# ==========================================================================
def shortest_path(
    maze: MazeMap,
    start: Cell,
    goals: Sequence[Cell],
    mode: FloodMode,
) -> List[Cell]:
    """start からゴール群のいずれかまでの最短経路（区画の列）を返す。

    最短歩数の経路が複数存在する場合は、**方向転換の回数が最も少ない経路**を
    選ぶ（note_030 §3 S0 のターン数タイブレーク）。到達不能なら NoRouteError。

    始点の直前の向き（機体が最初にどちらを向いているか）はここでは考慮しない
    （経路そのものはどの向きから来ても同じであるべき）。最初の 1 手を含めた
    ターン数は `route_to_commands` の `start_heading` で扱う。
    """
    goals = list(goals)
    dist = compute_flood(maze, goals, mode)
    sx, sy = start
    if not maze.in_bounds(sx, sy):
        raise ValueError(f"開始区画 ({sx},{sy}) が迷路の範囲外です")
    total_steps = dist[sx, sy]
    if total_steps == float("inf"):
        raise NoRouteError(f"{start} からゴール {goals} へ到達できません（mode={mode}）")
    total_steps = int(total_steps)
    goal_set = set(goals)

    if total_steps == 0:
        return [start]

    # レベル k = start からの手数。各状態は (区画, その区画に着いた移動方向)。
    # 開始区画のみ移動方向が定義されない（None）。
    # 値は (ここまでの最小ターン回数, 1 つ前のレベルでの状態キー)。
    State = Tuple[Cell, Optional[Direction]]
    level: Dict[State, Tuple[int, Optional[State]]] = {(start, None): (0, None)}
    levels: List[Dict[State, Tuple[int, Optional[State]]]] = [level]

    for k in range(total_steps):
        next_level: Dict[State, Tuple[int, Optional[State]]] = {}
        target_dist = total_steps - k - 1
        for (cell, in_dir), (turns, _parent) in level.items():
            x, y = cell
            for direction in ALL_DIRECTIONS:
                nb = maze.neighbor(x, y, direction)
                if nb is None:
                    continue
                nx, ny = nb
                if dist[nx, ny] != target_dist:
                    continue
                if not is_passable(maze, x, y, direction, mode):
                    continue
                turn_cost = 0 if (in_dir is None or direction == in_dir) else 1
                new_turns = turns + turn_cost
                key = (nb, direction)
                prev = next_level.get(key)
                if prev is None or prev[0] > new_turns:
                    next_level[key] = (new_turns, (cell, in_dir))
        if not next_level:
            # dist 配列と実際の到達可能性が食い違うことは無いはずだが、
            # 防御的に検査しておく。
            raise NoRouteError(
                f"歩数マップ上は到達可能({start}->{goals})だが経路復元に失敗しました"
            )
        levels.append(next_level)
        level = next_level

    # 最終レベル（ゴール区画）の中から総ターン数最小の状態を選ぶ。
    final_level = levels[total_steps]
    best_key: Optional[State] = None
    best_turns = None
    for (cell, in_dir), (turns, _parent) in final_level.items():
        if cell not in goal_set:
            continue
        if best_turns is None or turns < best_turns:
            best_turns = turns
            best_key = (cell, in_dir)
    if best_key is None:
        raise NoRouteError(f"{start} からゴール {goals} への経路復元に失敗しました")

    # 逆順にバックトレースして区画列を復元する。
    path_rev: List[Cell] = []
    key: Optional[State] = best_key
    for k in range(total_steps, 0, -1):
        cell, _in_dir = key
        path_rev.append(cell)
        _turns, parent = levels[k][key]
        key = parent
    path_rev.append(start)
    path_rev.reverse()
    return path_rev


# ==========================================================================
# ターン種別の列への変換
# ==========================================================================
class CommandType(str, Enum):
    """実機の語彙に合わせたターン種別。"""

    STRAIGHT = "straight"  # 直進 N 区画
    TURN_RIGHT90 = "turn_right90"  # 90° 右
    TURN_LEFT90 = "turn_left90"  # 90° 左
    TURN_180 = "turn_180"  # 180° 折返し（その場旋回）
    GOAL_STOP = "goal_stop"  # ゴール停止


@dataclass(frozen=True)
class Command:
    """1 つの動作。STRAIGHT のときだけ cells が意味を持つ（区画数）。"""

    type: CommandType
    cells: int = 0

    def __repr__(self) -> str:  # pragma: no cover - 表示用
        if self.type == CommandType.STRAIGHT:
            return f"直進{self.cells}区画"
        label = {
            CommandType.TURN_RIGHT90: "90°右",
            CommandType.TURN_LEFT90: "90°左",
            CommandType.TURN_180: "180°折返し",
            CommandType.GOAL_STOP: "ゴール停止",
        }[self.type]
        return label


def _turn_type(from_dir: Direction, to_dir: Direction) -> CommandType:
    """from_dir から to_dir への向き変更をターン種別に変換する。

    Direction は E=0,N=1,W=2,S=3 の反時計回り順で定義してある
    （classic/maze_map.py 参照。2026-08-23 工学系の流儀へ付け替え）ので、
    差分の mod 4 でそのまま判定できる（+1=左90, +2=180, +3(-1)=右90）。
    """
    rel = (int(to_dir) - int(from_dir)) % 4
    if rel == 0:
        raise ValueError("同一方向はターンではありません（直進として扱うべき）")
    if rel == 1:
        return CommandType.TURN_LEFT90
    if rel == 2:
        return CommandType.TURN_180
    return CommandType.TURN_RIGHT90  # rel == 3


def path_to_commands(
    path: Sequence[Cell],
    start_heading: Direction = Direction.N,
) -> List[Command]:
    """区画列をターン種別の列へ変換する。

    - 連続する同方向の移動は「直進 N 区画」1 個にまとめる。
    - 方向が変わるところに 90°右／90°左／180°折返しを挿入する。
    - 最後に必ず「ゴール停止」を付ける。
    - path が区画 1 個だけ（start が既にゴール）の場合は「ゴール停止」のみ。

    Args:
        path: `shortest_path` が返す区画列（隣接区画が連続している前提）。
        start_heading: 経路をたどり始める時点で機体が向いている方位。
            実機の慣例に合わせてデフォルトは N（スタート区画から迷路の
            奥へ向かう向き）。
    """
    if len(path) == 0:
        raise ValueError("path が空です")
    if len(path) == 1:
        return [Command(CommandType.GOAL_STOP)]

    commands: List[Command] = []
    heading = start_heading
    run_len = 0  # 現在の直進ランの区画数

    for a, b in zip(path[:-1], path[1:]):
        d = direction_between(a, b)
        if d != heading:
            if run_len > 0:
                commands.append(Command(CommandType.STRAIGHT, run_len))
                run_len = 0
            commands.append(Command(_turn_type(heading, d)))
            heading = d
        run_len += 1

    if run_len > 0:
        commands.append(Command(CommandType.STRAIGHT, run_len))
    commands.append(Command(CommandType.GOAL_STOP))
    return commands


def plan_route(
    maze: MazeMap,
    start: Cell,
    goals: Sequence[Cell],
    mode: FloodMode,
    start_heading: Direction = Direction.N,
) -> Tuple[List[Cell], List[Command]]:
    """`shortest_path` + `path_to_commands` をまとめて呼ぶ便利関数。"""
    path = shortest_path(maze, start, goals, mode)
    commands = path_to_commands(path, start_heading=start_heading)
    return path, commands
