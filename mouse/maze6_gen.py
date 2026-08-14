"""
mouse/maze6_gen.py
================
M2（未知 6x6 迷路の単走）用の迷路生成器。

規格（2026-08-10 教授裁定）:
  - 6x6 区画、区画サイズは RobotParams.cell_size（180 mm）
  - **ゴールは中央 2x2**（(2,2),(3,2),(2,3),(3,3)）。**内壁も柱もない広場**にする
    （実競技 16x16 の中央 2x2 と同じ構造。M3 へそのまま伸ばせるようにするため）
  - **ゴールの入口は 1 箇所のみ**（IEEE R2 SAC 規定 4.3 と同じ考え方）
  - スタートは四隅のいずれか・3 方向が壁（同 4.3）
  - 外周は全て壁
  - 生成は seed のみから決定的（リジェクトサンプリングの試行回数も seed から一意）

2 つの難度（研究計画書の M2 を段階に刻む。`experiments/m2_design.md` §4）:
  - mode="loop"（M2-0）: **行き止まりがない**。スタートとゴール以外の全区画で開通数 ≥ 2。
    「壁に当たらず進む」だけで必ずゴールへ着けるので、失敗したら原因が
    **ゴール認識だけ**に絞られる
  - mode="full"（M2-1）: 行き止まりあり。本番

壁配列の規約は mouse/corridor_gen.py・competition/evaluator.py と同一:
  v_walls[x,y] shape (W+1,H): セル(x-1,y)-(x,y) 間の縦壁
  h_walls[x,y] shape (W,H+1): セル(x,y-1)-(x,y) 間の横壁
  1 = 壁あり、0 = 開通
"""
import argparse
import json
import random
from collections import deque
from pathlib import Path

import numpy as np

from mouse.mjcf import build_maze_robot_xml
from mouse.params import RobotParams

SIZE = 6
GOAL_CELLS = frozenset({(2, 2), (3, 2), (2, 3), (3, 3)})
# ゴール 2x2 の内部の辺（ここを開けると中央の柱も孤立する ＝ 広場になる）
GOAL_INNER_EDGES = (("v", 3, 2), ("v", 3, 3), ("h", 2, 3), ("h", 3, 3))
# ゴール 2x2 の外周 8 辺（このうち 1 つだけを入口として開ける）
GOAL_RING_EDGES = (("v", 2, 2), ("v", 2, 3), ("v", 4, 2), ("v", 4, 3),
                   ("h", 2, 2), ("h", 3, 2), ("h", 2, 4), ("h", 3, 4))
CORNERS = ((0, 0), (SIZE - 1, 0), (0, SIZE - 1), (SIZE - 1, SIZE - 1))

_MAX_ATTEMPTS = 400


# ==========================================================================
# 壁配列の基本操作
# ==========================================================================
def _new_walls():
    return (np.ones((SIZE + 1, SIZE), dtype=int), np.ones((SIZE, SIZE + 1), dtype=int))


def _get(v, h, e):
    kind, x, y = e
    return v[x, y] if kind == "v" else h[x, y]


def _set(v, h, e, val):
    kind, x, y = e
    if kind == "v":
        v[x, y] = val
    else:
        h[x, y] = val


def _edge_between(a, b):
    """隣接する 2 区画の間の辺を返す。"""
    (ax, ay), (bx, by) = a, b
    if bx - ax == 1 and by == ay:
        return ("v", bx, ay)
    if ax - bx == 1 and by == ay:
        return ("v", ax, ay)
    if by - ay == 1 and bx == ax:
        return ("h", ax, by)
    if ay - by == 1 and bx == ax:
        return ("h", ax, ay)
    raise AssertionError(f"隣接していない: {a} {b}")


def _neighbors(cell):
    x, y = cell
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < SIZE and 0 <= ny < SIZE:
            yield (nx, ny)


def open_degree(v, h, cell):
    """区画の開通方向数。"""
    return sum(1 for nb in _neighbors(cell) if _get(v, h, _edge_between(cell, nb)) == 0)


def cells_open(v, h, a, b):
    return _get(v, h, _edge_between(a, b)) == 0


def reachable_set(v, h, start=(0, 0)):
    seen = {start}
    q = deque([start])
    while q:
        c = q.popleft()
        for nb in _neighbors(c):
            if nb not in seen and cells_open(v, h, c, nb):
                seen.add(nb)
                q.append(nb)
    return seen


def all_cells_reachable(v, h, start=(0, 0)):
    return len(reachable_set(v, h, start)) == SIZE * SIZE


# ==========================================================================
# 生成
# ==========================================================================
def _spanning_tree(rng, v, h):
    """DFS で全域木を掘る（この時点では行き止まりだらけ）。"""
    start = (0, 0)
    visited = {start}
    stack = [start]
    while stack:
        c = stack[-1]
        cand = [nb for nb in _neighbors(c) if nb not in visited]
        if not cand:
            stack.pop()
            continue
        nb = cand[rng.randrange(len(cand))]
        _set(v, h, _edge_between(c, nb), 0)
        visited.add(nb)
        stack.append(nb)


def _carve_goal(rng, v, h):
    """ゴール 2x2 を広場にし、入口を 1 箇所だけ開ける。"""
    for e in GOAL_INNER_EDGES:
        _set(v, h, e, 0)
    for e in GOAL_RING_EDGES:
        _set(v, h, e, 1)
    gateway = GOAL_RING_EDGES[rng.randrange(len(GOAL_RING_EDGES))]
    _set(v, h, gateway, 0)
    return gateway


def _isolate_start(rng, v, h, start):
    """スタート区画の 3 方向を壁にする（開通は 1 方向のみ）。"""
    nbs = list(_neighbors(start))
    open_nbs = [nb for nb in nbs if cells_open(v, h, start, nb)]
    if not open_nbs:
        keep = nbs[rng.randrange(len(nbs))]
        _set(v, h, _edge_between(start, keep), 0)
        open_nbs = [keep]
    keep = open_nbs[rng.randrange(len(open_nbs))]
    for nb in nbs:
        if nb != keep:
            _set(v, h, _edge_between(start, nb), 1)


def _remove_dead_ends(rng, v, h, start, gateway):
    """行き止まり（開通数 1）を無くす。スタートとゴール区画は対象外。

    開通数 1 の区画に対し、閉じている隣接辺を 1 つ開ける。ゴールの外周は
    入口を 1 箇所に保つため触らない。
    """
    protected = set(GOAL_RING_EDGES) - {gateway}
    for _ in range(SIZE * SIZE * 4):
        dead = [c for c in _all_cells()
                if c != start and c not in GOAL_CELLS and open_degree(v, h, c) <= 1]
        if not dead:
            return True
        c = dead[rng.randrange(len(dead))]
        closed = [nb for nb in _neighbors(c)
                  if _get(v, h, _edge_between(c, nb)) == 1
                  and _edge_between(c, nb) not in protected
                  and nb != start]
        if not closed:
            return False
        nb = closed[rng.randrange(len(closed))]
        _set(v, h, _edge_between(c, nb), 0)
    return False


def _all_cells():
    return [(x, y) for x in range(SIZE) for y in range(SIZE)]


def _reconnect(rng, v, h, start, protected):
    """到達不能な区画が出たら、到達済みとの境界の壁を開けて繋ぎ直す。

    ゴールを広場にして入口を 1 箇所へ絞ると、元の全域木でゴールを経由していた
    経路が切れて孤立区画が生まれる（mode="full" の 4/20 がこれで失敗した）。
    protected（ゴール外周の入口以外＋スタートの閉じた 3 辺）は開けない。
    """
    for _ in range(SIZE * SIZE * 4):
        seen = reachable_set(v, h, start)
        if len(seen) == SIZE * SIZE:
            return True
        cands = []
        for c in _all_cells():
            if c in seen:
                continue
            for nb in _neighbors(c):
                if nb in seen:
                    e = _edge_between(c, nb)
                    if e not in protected and _get(v, h, e) == 1:
                        cands.append(e)
        if not cands:
            return False
        _set(v, h, cands[rng.randrange(len(cands))], 0)
    return False


def _protected_edges(start, gateway):
    """開けてはいけない辺（ゴール外周の入口以外＋スタート周りの壁）。"""
    prot = set(GOAL_RING_EDGES) - {gateway}
    for nb in _neighbors(start):
        prot.add(_edge_between(start, nb))
    return prot


def _validate(v, h, start, mode):
    """生成物が規格を満たすかを検査する（満たさなければ False で再試行）。"""
    if not (np.all(v[0, :] == 1) and np.all(v[SIZE, :] == 1)
            and np.all(h[:, 0] == 1) and np.all(h[:, SIZE] == 1)):
        return False                                  # 外周壁
    if any(_get(v, h, e) != 0 for e in GOAL_INNER_EDGES):
        return False                                  # ゴールは広場
    if sum(1 for e in GOAL_RING_EDGES if _get(v, h, e) == 0) != 1:
        return False                                  # 入口は 1 箇所
    if open_degree(v, h, start) != 1:
        return False                                  # スタートは 3 方向が壁
    if not all_cells_reachable(v, h, start):
        return False                                  # 全区画へ到達できる
    if mode == "loop":
        for c in _all_cells():
            if c == start or c in GOAL_CELLS:
                continue
            if open_degree(v, h, c) < 2:
                return False                          # 行き止まりなし
    return True


def generate_maze(seed: int, mode: str = "loop") -> dict:
    """seed から決定的に 6x6 迷路を 1 枚生成する。

    mode="loop": 行き止まりなし（M2-0）／ mode="full": 行き止まりあり（M2-1）
    """
    if mode not in ("loop", "full"):
        raise ValueError(f"mode は 'loop' か 'full': {mode!r}")
    rng = random.Random(seed)

    for _attempt in range(_MAX_ATTEMPTS):
        v, h = _new_walls()
        _spanning_tree(rng, v, h)
        gateway = _carve_goal(rng, v, h)
        start = CORNERS[rng.randrange(len(CORNERS))]
        _isolate_start(rng, v, h, start)
        protected = _protected_edges(start, gateway)
        # ゴールを広場にした際に切れた区画を繋ぎ直す
        if not _reconnect(rng, v, h, start, protected):
            continue
        if mode == "loop":
            if not _remove_dead_ends(rng, v, h, start, gateway):
                continue
            if not _reconnect(rng, v, h, start, protected):
                continue
        if not _validate(v, h, start, mode):
            continue
        return dict(v_walls=v, h_walls=h, start=start, goal_cells=sorted(GOAL_CELLS),
                    gateway=gateway, width=SIZE, height=SIZE, seed=seed, mode=mode)

    raise RuntimeError(f"generate_maze(seed={seed}, mode={mode}): 生成に失敗しました")


def shortest_distances(v, h, targets=GOAL_CELLS):
    """各区画からゴールまでの最短歩数（幅優先）。到達不能は -1。

    報酬のポテンシャル整形に使う（教授裁定 §5-C: 学習時の報酬に迷路地図を
    使うのは可。方策の入力ではないため入出力契約に触れない）。
    """
    dist = {c: -1 for c in _all_cells()}
    q = deque()
    for t in targets:
        dist[t] = 0
        q.append(t)
    while q:
        c = q.popleft()
        for nb in _neighbors(c):
            if dist[nb] == -1 and cells_open(v, h, c, nb):
                dist[nb] = dist[c] + 1
                q.append(nb)
    return dist


def initial_heading_deg(v, h, start) -> float:
    """スタート区画で唯一開いている方向の方位 [deg]（0=+x, 90=+y, 180=-x, 270=-y）。"""
    for nb in _neighbors(start):
        if cells_open(v, h, start, nb):
            dx, dy = nb[0] - start[0], nb[1] - start[1]
            return {(1, 0): 0.0, (0, 1): 90.0, (-1, 0): 180.0, (0, -1): 270.0}[(dx, dy)]
    raise AssertionError("スタートが完全に閉じている")


def save_maze(maze: dict, out_dir, params: RobotParams = None):
    """npz + XML として保存する。戻り値 (npz_path, xml_path)。"""
    params = params or RobotParams()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = maze["seed"]
    stem = f"maze6_{seed}"
    npz_path, xml_path = out_dir / f"{stem}.npz", out_dir / f"{stem}.xml"

    np.savez(npz_path, v_walls=maze["v_walls"], h_walls=maze["h_walls"],
             start=np.array(maze["start"], dtype=np.int64),
             goal_cells=np.array(maze["goal_cells"], dtype=np.int64),
             width=maze["width"], height=maze["height"], seed=seed)

    cs = params.cell_size
    sx, sy = maze["start"]
    heading = initial_heading_deg(maze["v_walls"], maze["h_walls"], maze["start"])
    build_maze_robot_xml(
        maze["v_walls"], maze["h_walls"], str(xml_path), model_name=f"maze6_{seed}",
        mouse_pos=f"{sx * cs + cs / 2} {sy * cs + cs / 2} 0.002",
        # 🔴 2026-08-14 是正（環境 v2 項目 5）: ここも `center_goal=False` だった。
        # `mouse/maze6_env.py` の呼び出し側と同じ理由（当時は柱を落とす条件が 16x16 専用で
        # 6x6 では無意味な引数だった）。**mjcf を一般化した結果この引数が効くようになった**
        # ので True にする。**環境（実走）と事前生成 XML の仕様を一致させる**ため。
        mouse_euler=f"0 0 {heading}", center_goal=True, params=params)
    return npz_path, xml_path


# ==========================================================================
# CLI
# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="M2 6x6 迷路生成器")
    # 研究計画書 §9-7 のデータ三分割。廊下（3000-3019 / 5000-5019）や 16x16
    # （1000-1019）と衝突しない帯を割り当てる。
    ap.add_argument("--split", choices=["train", "validation", "eval"], required=True)
    ap.add_argument("--seeds", type=str, required=True, help="例: 6000-6019")
    ap.add_argument("--mode", choices=["loop", "full"], default="loop")
    ap.add_argument("--out-root", type=str, default="assets/maze6")
    args = ap.parse_args(argv)

    start_s, _, end_s = args.seeds.partition("-")
    seeds = list(range(int(start_s), int(end_s) + 1))
    out_dir = Path(args.out_root) / args.mode / args.split

    records = []
    for seed in seeds:
        m = generate_maze(seed, mode=args.mode)
        npz_path, xml_path = save_maze(m, out_dir)
        d = shortest_distances(m["v_walls"], m["h_walls"])
        records.append(dict(seed=seed, start=list(m["start"]), gateway=list(m["gateway"]),
                            dist_from_start=d[m["start"]], npz=str(npz_path), xml=str(xml_path)))
        print(f"[maze6_gen] {args.mode}/{args.split} seed={seed} start={m['start']} "
              f"入口={m['gateway']} 最短={d[m['start']]} 歩")

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(dict(split=args.split, mode=args.mode, seeds=seeds,
                       generator="mouse/maze6_gen.py: generate_maze",
                       mazes=records), f, indent=2, ensure_ascii=False)
    print(f"[maze6_gen] saved {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
