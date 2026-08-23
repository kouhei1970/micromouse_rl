#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`mazes/INDEX.tsv` を生成する（属性 ＋ 指標を迷路 1 面につき 1 行）。

生成の実装は `build_index_tsv()` の 1 か所だけに置く。`tests/test_maze_db.py`
の索引同期検査もこの関数を呼ぶので、生成方法が 2 箇所に分かれてずれることが
構造的に起きない（note_036 §2-6 の「索引の同期」検査）。

指標は最短距離 D0（壁が完全既知のときの区画数）だけを載せる。
`competition/explore_cost.py` は W=H=16 決め打ちで汎用でないため使わず、
ここでは width/height をそのまま使う素朴な BFS を独立に書いた
（`competition/evaluator.py` は mujoco を読み込むため軽量な用途には重すぎる）。

実行方法:
    .venv/bin/python research_notes/scripts/build_maze_index.py
"""
import sys
from collections import deque
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.maze_db import MazeDB, MazeRecord  # noqa: E402

INDEX_PATH = ROOT / "mazes" / "INDEX.tsv"

INDEX_COLUMNS = (
    "id",
    "kind",
    "source_type",
    "confidence",
    "series",
    "edition",
    "year",
    "class",
    "stage",
    "width",
    "height",
    "start_x",
    "start_y",
    "start_heading",
    "goal",
    "D0",
    "content_sha256",
)


def shortest_distance(
    v_walls: np.ndarray,
    h_walls: np.ndarray,
    width: int,
    height: int,
    start: Tuple[int, int],
    goals: Iterable[Tuple[int, int]],
) -> int:
    """壁が完全既知のときの最短距離 [区画]。到達不能なら -1。

    v_walls/h_walls は `competition/evaluator.py` と同じ 0/1 の約束
    （0=壁なし・1=壁あり）。0/1 判定だけを使うので `MazeDB.walls()` の
    戻り値をそのまま渡せる。
    """
    goal_set = set(tuple(g) for g in goals)
    start = tuple(start)
    if start in goal_set:
        return 0
    dist = {start: 0}
    dq = deque([start])
    while dq:
        x, y = dq.popleft()
        d = dist[(x, y)]
        neighbors = []
        if x + 1 <= width and v_walls[x + 1, y] == 0:
            neighbors.append((x + 1, y))
        if x - 1 >= 0 and v_walls[x, y] == 0:
            neighbors.append((x - 1, y))
        if y + 1 <= height and h_walls[x, y + 1] == 0:
            neighbors.append((x, y + 1))
        if y - 1 >= 0 and h_walls[x, y] == 0:
            neighbors.append((x, y - 1))
        for n in neighbors:
            nx, ny = n
            if not (0 <= nx < width and 0 <= ny < height) or n in dist:
                continue
            dist[n] = d + 1
            if n in goal_set:
                return d + 1
            dq.append(n)
    return -1


def _field(value) -> str:
    return "" if value is None else str(value)


def _row(rec: MazeRecord, db: MazeDB) -> str:
    # 未知壁（'.'）を含む面（kerikun11 の Cheese 系）が段3で加わったため、D0 の計算は
    # 未知=壁とみなす悲観側で丸める（保守的な最短距離。楽観側との差は個別に見ればよい）。
    v, h = db.walls(rec, unknown="wall")
    d0 = shortest_distance(v, h, rec.width, rec.height, rec.start, rec.goal)
    goal_str = ";".join(f"{x}-{y}" for x, y in sorted(rec.goal, key=lambda p: (p[1], p[0])))
    fields = [
        rec.id,
        _field(rec.kind),
        rec.source_type,
        rec.confidence,
        rec.series,
        _field(rec.edition),
        _field(rec.year),
        _field(rec.maze_class),
        _field(rec.stage),
        str(rec.width),
        str(rec.height),
        str(rec.start[0]),
        str(rec.start[1]),
        rec.start_heading,
        goal_str,
        str(d0),
        rec.content_sha256,
    ]
    return "\t".join(fields)


def build_index_tsv(db: MazeDB) -> str:
    """`db` が保持する全迷路の索引を TSV 文字列として返す（id 昇順）。"""
    lines = ["\t".join(INDEX_COLUMNS)]
    for rec in db.query():  # 属性を指定しない query() は全件を id 昇順で返す
        lines.append(_row(rec, db))
    return "\n".join(lines) + "\n"


def main() -> None:
    db = MazeDB()
    text = build_index_tsv(db)
    INDEX_PATH.write_text(text, encoding="utf-8")
    print(f"書き出し: {INDEX_PATH}（{len(db)} 面）")


if __name__ == "__main__":
    main()
