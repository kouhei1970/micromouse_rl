#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**紙の上で**「対照はどの経路を引き、TR はどの経路を引くか」を再現する。

exp_015 の集計と考察に使う道具。物理シミュレーションを回さずに、迷路の壁だけから

  - **対照（L0-c+E1）が引く経路** = 足立法の距離場（BFS）を、方策と同じ
    決定的タイブレーク（直進優先 → 時計回り）で降りた経路
  - **TR が引く経路** = `competition/route_planner.py` の時間の場を降りた経路

を求め、時間モデルで採点する。**地図は全既知**（最短走行の時点では E1 により
最短経路が確定しているので、経路選択に効く範囲では既知地図＝真の地図）。

## なぜ必要か（この道具が答える問い）

「歩数最短 ≠ 時間最短の面はどれか」を**真の最短経路どうし**で比べるのは
**発動の判定として不十分**である。対照が実際に引くのは「距離最適な経路の中で
タイブレークが選んだ 1 本」であって、**距離最適の中で折れが最小の 1 本とは限らない**。
そのため

  - 距離最適 vs 時間最適（経路の**集合**どうし）… 差が出るのは maze_42134 のみ（1/20）
  - **対照が実際に引く経路** vs 時間最適 … 差が出るのは **5/20**

と、判定が変わる。**方策を差し替えたときに実際に何が変わるかを見たいのだから、
後者が正しい比較である。**

なお本モジュールの対照側の再現は**実測と突き合わせて検証してある**:
設計帯 20 面すべてで、`outputs/exp_014_design_check/l0c_e1` の最速走行の
移動区画数と一致した（折れ数は計時窓の切り取りのぶん実測が 1 少ない。
裁定 R14 の窓の張り方によるもので、経路そのものの食い違いではない）。

使い方:
    .venv/bin/python experiments/exp_015_time_optimal_route/route_model.py \\
        --maze-dir competition/mazes/design_v4
"""
import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition.baseline_slalom_e1_tr import load_time_model  # noqa: E402
from competition.route_planner import (DELTA, DIRS, StraightGridModel,  # noqa: E402
                                        TIE_EPS, turn_count, value_field)

_CW_NEXT = {"N": "E", "E": "S", "S": "W", "W": "N"}
START_HEADING = "N"          # スタート区画は北開放（`maze_gen.py` 手順 4）


def connects_true(v_walls, h_walls):
    """真の壁配列から通行可否を返す（方策の `_connects_known` と同じ引数の並び）。"""
    v, h = np.asarray(v_walls, int), np.asarray(h_walls, int)

    def f(x, y, nx, ny):
        if (nx, ny) == (x + 1, y):
            return int(v[x + 1, y]) != 1
        if (nx, ny) == (x - 1, y):
            return int(v[x, y]) != 1
        if (nx, ny) == (x, y + 1):
            return int(h[x, y + 1]) != 1
        if (nx, ny) == (x, y - 1):
            return int(h[x, y]) != 1
        raise ValueError(f"({x},{y}) と ({nx},{ny}) は隣接していません")
    return f


def tiebreak_order(prev_dir):
    """方策の `_tiebreak_order` と同一（直進優先 → 時計回り）。"""
    order = [prev_dir]
    d = prev_dir
    for _ in range(3):
        d = _CW_NEXT[d]
        order.append(d)
    return order


def bfs_field(targets, width, height, connects):
    """方策の `_flood_fill` と同一の距離場（多始点 BFS）。"""
    dist = {tuple(c): 0 for c in targets}
    q = deque(dist)
    while q:
        x, y = q.popleft()
        for d in DIRS:
            dx, dy = DELTA[d]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height) or (nx, ny) in dist:
                continue
            if connects(x, y, nx, ny):
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))
    return dist


def _descend(start, heading, width, height, connects, pick):
    """場を 1 手ずつ降りる。`pick(cell, d_in)` が (方向, 到達したか) を返す。"""
    cell, d_in = tuple(start), heading
    path, moves, turns = [cell], 0, 0
    for _ in range(width * height * 4):
        chosen, done = pick(cell, d_in)
        if done:
            return dict(path=path, moves=moves, turns=turns, reached=True)
        if chosen is None:
            return dict(path=path, moves=moves, turns=turns, reached=False)
        turns += turn_count(d_in, chosen)
        moves += 1
        cell = (cell[0] + DELTA[chosen][0], cell[1] + DELTA[chosen][1])
        d_in = chosen
        path.append(cell)
    return dict(path=path, moves=moves, turns=turns, reached=False)


def control_route(v_walls, h_walls, start, goals, width=16, height=16):
    """**対照（L0-c+E1）が引く経路**: 距離場 + 直進優先→時計回りタイブレーク。"""
    connects = connects_true(v_walls, h_walls)
    dist = bfs_field(goals, width, height, connects)

    def pick(cell, d_in):
        if dist.get(cell) == 0:
            return None, True
        cands = []
        for d in DIRS:
            n = (cell[0] + DELTA[d][0], cell[1] + DELTA[d][1])
            if not (0 <= n[0] < width and 0 <= n[1] < height):
                continue
            if not connects(cell[0], cell[1], n[0], n[1]):
                continue
            if n in dist:
                cands.append((d, dist[n]))
        if not cands:
            return None, False
        m = min(x[1] for x in cands)
        best = {d for d, nd in cands if nd == m}
        return next(d for d in tiebreak_order(d_in) if d in best), False

    return _descend(start, START_HEADING, width, height, connects, pick)


def tr_route(v_walls, h_walls, start, goals, model, width=16, height=16):
    """**TR が引く経路**: 時間の場 + 同じタイブレーク（`SlalomE1TRPolicy` と同一規則）。"""
    connects = connects_true(v_walls, h_walls)
    field = value_field(list(goals), width, height, connects, model)

    def pick(cell, d_in):
        if field.get(cell) == 0:
            return None, True
        cands = []
        for d_out, _n, st, w in model.successors(cell, d_in, width, height, connects):
            v = field.states.get(st)
            if v is not None:
                cands.append((d_out, w + v))
        if not cands:
            return None, False
        m = min(c for _, c in cands)
        best = {d for d, c in cands if c <= m + TIE_EPS}
        return next(d for d in tiebreak_order(d_in) if d in best), False

    return _descend(start, START_HEADING, width, height, connects, pick)


def load_maze(path):
    z = np.load(path)
    v, h = np.array(z["v_walls"], int), np.array(z["h_walls"], int)
    if "start_x" in z.files:
        start = (int(z["start_x"]), int(z["start_y"]))
        goals = tuple((int(a), int(b)) for a, b in zip(z["goals_x"], z["goals_y"]))
    else:
        start, goals = (0, 0), ((7, 7), (7, 8), (8, 7), (8, 8))
    return v, h, start, goals


def analyse_dir(maze_dir, a=None, b=None):
    """迷路ディレクトリの全面について、対照と TR の経路を紙の上で比べる。"""
    if a is None or b is None:
        a, b = load_time_model()
    model = StraightGridModel(a, b)
    rows = []
    for f in sorted((REPO_ROOT / maze_dir).glob("maze_*.npz"),
                    key=lambda p: int(p.stem.split("_")[1])):
        v, h, start, goals = load_maze(f)
        gset = set(map(tuple, goals))
        c = control_route(v, h, start, gset)
        t = tr_route(v, h, start, gset, model)
        t_c = a * c["moves"] + b * c["turns"]
        t_t = a * t["moves"] + b * t["turns"]
        rows.append(dict(maze=f.stem,
                         ctrl_moves=c["moves"], ctrl_turns=c["turns"], ctrl_time=t_c,
                         tr_moves=t["moves"], tr_turns=t["turns"], tr_time=t_t,
                         route_differs=(c["path"] != t["path"]),
                         gain_s=t_c - t_t,
                         gain_pct=(t_c - t_t) / t_c * 100.0 if t_c > 0 else 0.0,
                         ctrl_reached=c["reached"], tr_reached=t["reached"]))
    return dict(maze_dir=str(maze_dir), a=a, b=b, rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maze-dir", default="competition/mazes/design_v4")
    ap.add_argument("--out", default=None, help="JSON の書き出し先（省略時は書かない）")
    args = ap.parse_args()

    res = analyse_dir(args.maze_dir)
    a, b = res["a"], res["b"]
    print(f"時間モデル（実測の回帰。ハードコードしない）: t = {a:.4f}×歩数 + {b:.4f}×折れ数"
          f"  → 旋回 1 回 = 移動 {b / a:.3f} 区画ぶん")
    print(f"迷路: {args.maze_dir}\n")
    print(f"{'面':<14}{'対照 (歩,折)':>16}{'対照 t':>9}"
          f"{'TR (歩,折)':>16}{'TR t':>9}{'短縮 [s]':>10}{'短縮 %':>8}  発動")
    n_fire = 0
    for r in res["rows"]:
        fire = r["gain_s"] > 1e-9
        n_fire += fire
        ctrl = "({}, {})".format(r["ctrl_moves"], r["ctrl_turns"])
        tr = "({}, {})".format(r["tr_moves"], r["tr_turns"])
        print(f"{r['maze']:<14}{ctrl:>16}{r['ctrl_time']:>9.3f}"
              f"{tr:>16}{r['tr_time']:>9.3f}"
              f"{r['gain_s']:>10.3f}{r['gain_pct']:>8.2f}  {'*' if fire else ''}")
    g = np.array([r["gain_pct"] for r in res["rows"]])
    fired = g[g > 1e-9]
    print(f"\n発動（対照の経路より時間モデル上で速い面）: {n_fire}/{len(res['rows'])}")
    if fired.size:
        print(f"  発動面の短縮率: 中央値 {np.median(fired):.2f}%／最大 {fired.max():.2f}%")
    print(f"  全面の短縮率 中央値: {np.median(g):.2f}%")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
        print(f"\n書き出し: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
