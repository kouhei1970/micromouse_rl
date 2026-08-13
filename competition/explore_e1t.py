#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探索戦略 **E1T** — 確定判定を「歩数」ではなく「時間」で行う。

2026-08-13 新設（exp_018・裁定 R36-2）。**`competition/explore_e1.py`（E1）は
変更しない** — exp_014・exp_015 の対照が再現できなくなるため。

--------------------------------------------------------------------------
なぜ要るのか（exp_015 が見つけた欠陥）
--------------------------------------------------------------------------
exp_015 で経路選択を「時間最短」に替えた（TR）ところ、**最速タイムは 7/20 面で
改善した一方、初回最短走行が 6 面で +5.99〜+14.83 s 遅くなった**（(e) 最大 2.016）。

**機構**: **E1 が確定させているのは「歩数最短が真に最短であること」であって、
時間最短の経路の壁ではない。**未知壁は楽観的に「通れる」とみなすので、
**未探索の区画は「壁が無い＝まっすぐ走れて折れが少ない」ように見える**。
時間モデルは折れに費用を置くので **TR は未探索の領域へ引き寄せられる**。
そこへ行って壁を見つけ、引き返す。**確定の基準と経路の基準が食い違っている。**

--------------------------------------------------------------------------
理論 — E1 の確定定理は費用モデルに依らない
--------------------------------------------------------------------------
`explore_e1.py` 冒頭の議論は、**加法的で非負の費用ならそのまま成り立つ**。
楽観地図（未知壁 = 通行可）の上の最小費用を C_opt、真の迷路の上の最小費用を
C_true とすると:

1. 楽観地図の辺集合は真の迷路の辺集合の**上位集合**。真の迷路のどの経路も
   楽観地図に同じ費用で存在するので **C_opt <= C_true**
2. 楽観地図上の最小費用経路が**確定済みの辺だけ**で構成されていれば、その経路は
   真の迷路にも同じ費用で実在するので **C_true <= C_opt**
3. よって **C_opt = C_true**

したがって

    「現在の楽観**時間**最短経路上に未知壁が 1 つも残っていない」瞬間に、
    それが真の時間最短経路であると確定できる。

**歩数の版との違いは、経路が区画の上ではなく状態（区画 × 進行方向）の上で
決まる点だけ**である。判定に使う 2 つの場も状態の上に張る。

--------------------------------------------------------------------------
判定の実装
--------------------------------------------------------------------------
前向きの場 D_f（出発状態 → 各状態）と後ろ向きの場 D_b（各状態 → ゴール）を
`competition/route_planner.py` で張り、T = D_b(出発状態) とする。
状態 s から s' への遷移が**未知壁をまたぎ**、かつ

    D_f(s) + c(s -> s') + D_b(s') = T

を満たすとき、その未知壁は「**もし開いていたら時間最短経路に使われうる**」
= relevant である。**relevant が空になった時点で時間最短経路が確定する。**

**出発時の向きを含める**（スタート区画は北開放。`route_model.py` と同じ規約）。
歩数の版は無向 BFS なので向きを持たないが、旋回に費用がある以上、
**どちらを向いて出発するかで最適経路が変わりうる**。

本モジュールは壁地図（-1=未知 / 0=壁なし / 1=壁あり）に対する純粋関数として
実装する（`explore_e1.py` と同じ流儀）。
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition.explore_e1 import UNKNOWN, wall_state  # noqa: E402
from competition.route_planner import (DIRS, TIE_EPS, forward_field,  # noqa: E402
                                        value_field)

# スタート区画は北開放（`competition/maze_gen.py` 手順 4）。出発時の向きは北。
START_HEADING = "N"


def optimistic_connects(v_known, h_known):
    """未知壁を「通れる」とみなす通行判定（`explore_e1.passable` と同じ規約）。

    方策の `_connects_known` と同じ引数の並びにしてある。
    """
    def f(x, y, nx, ny):
        return wall_state(v_known, h_known, x, y, nx, ny) != 1
    return f


def relevant_unknown_walls_time(v_known, h_known, width, height, start, goals,
                                model, start_heading: str = START_HEADING):
    """「もし開いていたら**時間**最短経路に使われうる未知壁」の一覧。

    返り値: [((x, y), (nx, ny)), ...]（未知壁を隔てて隣接する区画対。重複は除く）。
    空リストなら**現在の楽観時間最短経路が真の時間最短経路として確定**している。
    """
    connects = optimistic_connects(v_known, h_known)
    start_state = (tuple(start), start_heading)

    db = value_field(list(goals), width, height, connects, model)
    total = db.states.get(start_state)
    if total is None:
        return []              # ゴールへ到達不能（想定外）。確定扱いにしない

    df = forward_field([start_state], width, height, connects, model)

    seen, out = set(), []
    for (cell, d_in), fwd in df.items():
        for d_out, ncell, nxt, w in model.successors(cell, d_in, width, height, connects):
            if wall_state(v_known, h_known, cell[0], cell[1], ncell[0], ncell[1]) != UNKNOWN:
                continue        # 既知の辺は確認不要
            back = db.states.get(nxt)
            if back is None:
                continue
            if abs(fwd + w + back - total) > TIE_EPS:
                continue        # この辺を使う時間最短経路は無い
            key = (cell, ncell) if cell < ncell else (ncell, cell)
            if key in seen:
                continue
            seen.add(key)
            out.append((cell, ncell))
    return out


def is_time_shortest_confirmed(v_known, h_known, width, height, start, goals,
                               model, start_heading: str = START_HEADING) -> bool:
    """現在の楽観**時間**最短経路が、真の時間最短経路として確定しているか。"""
    return not relevant_unknown_walls_time(v_known, h_known, width, height,
                                            start, goals, model, start_heading)


def explore_targets_time(v_known, h_known, width, height, start, goals,
                         model, start_heading: str = START_HEADING):
    """追加探索で向かうべき区画の集合（**時間**の基準）。

    relevant な未知壁に隣接する区画。ここへ行って壁を観測すれば、その未知壁が
    確定して relevant が減る（`explore_e1.explore_targets` と同じ考え方）。
    """
    rel = relevant_unknown_walls_time(v_known, h_known, width, height,
                                      start, goals, model, start_heading)
    targets = set()
    for a, b in rel:
        targets.add(a)
        targets.add(b)
    return targets
