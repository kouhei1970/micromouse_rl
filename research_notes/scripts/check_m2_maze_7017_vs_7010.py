"""
research_notes/scripts/check_m2_maze_7017_vs_7010.py
=====================================================
**「なぜ 7017 だけか」**を調べる（裁定 R41 の追加指示。准教授の面の同定を受けて）。

exp_012 条件 E の seed1 が検証帯で記録した非ゼロのゴール率 3 点は、**すべて面 7017**
だった。7017 と、**同じ $D_0$=4 でありながら一度もゴールしていない 7010** を、
迷路の幾何だけで比較する（学習の話は入れない。**先に「面の違い」を尽くす**）。

比較する量（すべて迷路の壁配列から決まる決定的な量。推定・近似はしない）:
  - スタート区画・スタートの向き・$D_0$
  - 最短経路の区画列と**曲がりの回数**（進行方向が変わる回数）
  - 最短経路上の**分岐**（開いた隣接が 3 つ以上ある区画の数）
  - 最短経路の**最長直進**（連続して同じ向きに進む区画数）
  - スタート直後に**進める向きの数**（袋小路からの向き直しが要るか）
  - 迷路全体の開通辺数（面の込み具合）

使い方:
    .venv/bin/python research_notes/scripts/check_m2_maze_7017_vs_7010.py
    .venv/bin/python research_notes/scripts/check_m2_maze_7017_vs_7010.py --seeds 7017 7010 7000
"""
import argparse
import sys
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mouse.maze6_gen import (  # noqa: E402
    GOAL_CELLS, SIZE, cells_open, generate_maze, shortest_distances,
)

MAZE_MODE = "loop"
DIRS = {(1, 0): "→", (-1, 0): "←", (0, 1): "↑", (0, -1): "↓"}


def neighbors(m, c):
    out = []
    for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n = (c[0] + d[0], c[1] + d[1])
        if 0 <= n[0] < SIZE and 0 <= n[1] < SIZE and cells_open(m["v_walls"], m["h_walls"], c, n):
            out.append(n)
    return out


def shortest_path(m, dist, start):
    """d が 1 ずつ減る経路を 1 本取る（同点は座標の辞書順で決める＝決定的）。"""
    path = [start]
    c = start
    while dist[c] > 0:
        cands = [n for n in neighbors(m, c) if dist[n] == dist[c] - 1]
        c = min(cands)
        path.append(c)
    return path


def path_stats(path):
    dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    turns = sum(1 for a, b in zip(dirs, dirs[1:]) if a != b)
    runs, cur = [], 1
    for a, b in zip(dirs, dirs[1:]):
        if a == b:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return dirs, turns, max(runs)


def render(m, path=None):
    """迷路を ASCII で描く（+---+ 形式）。path 上の区画は * を置く。"""
    v, h = m["v_walls"], m["h_walls"]
    lines = []
    for y in range(SIZE - 1, -1, -1):
        top = ""
        for x in range(SIZE):
            top += "+" + ("---" if h[x, y + 1] else "   ")
        lines.append(top + "+")
        mid = ""
        for x in range(SIZE):
            mid += ("|" if v[x, y] else " ")
            c = (x, y)
            if c == tuple(m["start"]):
                mid += " S "
            elif c in GOAL_CELLS:
                mid += " G "
            elif path and c in path:
                mid += " * "
            else:
                mid += "   "
        lines.append(mid + ("|" if v[SIZE, y] else " "))
    bottom = ""
    for x in range(SIZE):
        bottom += "+" + ("---" if h[x, 0] else "   ")
    lines.append(bottom + "+")
    return "\n".join(lines)


def analyze(seed: int) -> dict:
    m = generate_maze(seed, mode=MAZE_MODE)
    dist = shortest_distances(m["v_walls"], m["h_walls"])
    start = tuple(m["start"])
    d0 = dist[start]
    path = shortest_path(m, dist, start)
    dirs, turns, longest = path_stats(path)
    n_branch = sum(1 for c in path if len(neighbors(m, c)) >= 3)
    n_open_edges = sum(len(neighbors(m, (x, y))) for x in range(SIZE) for y in range(SIZE)) // 2
    return dict(
        seed=seed, start=start, d0=int(d0), path=path,
        dir_str="".join(DIRS[d] for d in dirs), turns=turns, longest_run=longest,
        n_branch_on_path=n_branch, start_degree=len(neighbors(m, start)),
        n_open_edges=n_open_edges, gateway=tuple(m["gateway"]),
        render=render(m, path=set(path)),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="面 7017 と 7010 の幾何を比べる")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7017, 7010])
    args = ap.parse_args(argv)

    res = [analyze(s) for s in args.seeds]
    print("=" * 78)
    print("面の幾何の比較（壁配列から決まる決定的な量のみ。学習の話は入れない）")
    print("=" * 78)
    hdr = f"{'量':<28}" + "".join(f"{r['seed']:>12}" for r in res)
    print(hdr)
    rows = [
        ("スタート区画", lambda r: str(r["start"])),
        ("D₀（区画数）", lambda r: str(r["d0"])),
        ("最短経路の曲がりの回数", lambda r: str(r["turns"])),
        ("最短経路の最長直進[区画]", lambda r: str(r["longest_run"])),
        ("経路上の分岐区画数(次数≥3)", lambda r: str(r["n_branch_on_path"])),
        ("スタート区画の次数", lambda r: str(r["start_degree"])),
        ("開通辺の総数", lambda r: str(r["n_open_edges"])),
        ("ゴール入口(gateway)", lambda r: str(r["gateway"])),
        ("最短経路の向き列", lambda r: r["dir_str"]),
    ]
    for label, f in rows:
        print(f"{label:<28}" + "".join(f"{f(r):>12}" for r in res))
    for r in res:
        print(f"\n--- 面 {r['seed']}（S=スタート・G=ゴール・*=最短経路）---")
        print(r["render"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
