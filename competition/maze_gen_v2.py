"""
規定準拠 評価迷路生成器 v2（ループあり・壁づたい走行不可）
Rule-conforming evaluation maze generator (loops, wall-follower-proof).

2026-08-10、ユーザ判断により旧生成器（competition/maze_gen.py）を置換する。
旧版は seed 1000-1019 の全 20 面で以下の規定違反があった（検査: audit_maze_rules.py）:
  - ゴール入口が 1 箇所なのは 1/20 面のみ（中央値 4・最大 7）
  - **左手法・右手法の壁づたい走行で 20/20 面がゴールに到達できた**

準拠する規定と出典:
- IEEE R2 SAC 2020 規定 4.3: ゴール区画の入口は 1 箇所のみ。スタート区画は四隅の
  いずれかで 3 方向が壁
- IEEE R2 SAC 2020 規定 4.5: 複数経路は許容され想定されるべき。ゴールは
  **壁づたい走行では発見できない位置**に置かれる
  https://attend.ieee.org/r2sac-2020/wp-content/uploads/sites/175/2020/01/MicroMouse_Rules_2020.pdf
- NTF クラシック規定 注意 9: 終点の 4 区画内には壁も柱も存在しない
- NTF クラシック規定 2-4: 終点の中央を除く全格子点に最低 1 枚の壁が接する／外周壁は完備

■ 壁づたい走行を成立させない仕組み（本生成器の核心）
壁づたい走行の機体は「最初に触れた壁の連結成分」の縁だけをなぞる。したがって
**ゴールを囲む 7 枚の壁（入口 1 箇所を除く）が、外周壁とつながらない独立した島**に
なっていれば、スタート（外周壁に接する）から出た機体は原理的にゴールへ入れない。
実装では、ゴール外周の 8 つの格子点に接する壁のうち「ゴールを囲むリング以外」を
すべて強制的に開放（forced-open）し、以後の壁追加でもそこを塞がない。
構造で保証したうえで、最後に実際の左手法・右手法の走行シミュレーションでも検証する
（構造の意図と実挙動の両方を確認する = 研究計画書 §9 の検証方針）。

■ 独立閉路数（複数経路）の設計
実競技の迷路は完全迷路（閉路 0）ではない。本生成器は全域木（開通 255 辺）に対して
内部壁をさらに除去して閉路を作る。除去の上限は「終点中央を除く全格子点に壁が
最低 1 枚接する」規定が自動的に与える — この規定は 2x2 セルが全開放される「広場」の
発生を禁じるのと等価であり、通路構造を保ったまま最大限の複数経路が得られる。
既定の目標除去数 EXTRA_OPEN_TARGET はこの制約下で到達可能な水準に設定している。

使い方:
    .venv/bin/python -m competition.maze_gen_v2 --seeds 1000-1019
"""
import argparse
import json
import os
import random
import sys
from collections import deque

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

W = H = 16
GOAL_CELLS = frozenset({(7, 7), (7, 8), (8, 7), (8, 8)})
CENTER_POST = (8, 8)

# ゴール 2x2 の外周 8 辺（('v'|'h', x, y) 形式）
RING_EDGES = (
    ("h", 7, 7), ("h", 8, 7),      # 南
    ("h", 7, 9), ("h", 8, 9),      # 北
    ("v", 7, 7), ("v", 7, 8),      # 西
    ("v", 9, 7), ("v", 9, 8),      # 東
)
# ゴール 2x2 の内壁 4 枚（常に開放。NTF 注意 9）
GOAL_INNER = (("v", 8, 7), ("v", 8, 8), ("h", 7, 8), ("h", 8, 8))
# ゴール外周の格子点（中央を除く 8 点）
RING_POSTS = ((7, 7), (8, 7), (9, 7), (7, 8), (9, 8), (7, 9), (8, 9), (9, 9))

# 全域木に追加で開ける内部壁の目標数（＝閉路数の目安）。
# 設計根拠（seed 1000-1003 で実測したセルの分岐次数分布）:
#   目標  閉路  平均次数  行止り/通路・角/T字/十字
#      0     4     2.02    29 / 194 / 31 /  1   ← 完全迷路。複数経路がなく IEEE 4.5 に反する
#     15    18     2.14    22 / 178 / 51 /  3
#     30    34     2.25    17 / 161 / 71 /  5   ← 採用
#     60    64     2.49     8 / 125 /111 / 10   ← 分岐が通路を上回り「広場」的で実競技と乖離
# 30 を採用する理由: 独立閉路 34 で複数経路が十分にある一方、通路・角のセルが
# 63%（161/256）と多数を占めて通路構造を保ち、探索を非自明にする行き止まりも
# 17 箇所残る。実競技の迷路は「通路主体・分岐と行き止まりが適度にある」構造であり、
# この水準が最も近い。上限は「終点中央を除く全格子点に壁が最低 1 枚」の規定
# （＝ 2x2 の広場を禁じる）が自動的に与える。
EXTRA_OPEN_TARGET = 30
MAX_ATTEMPTS = 400       # 1 seed あたりの再試行上限（リジェクトサンプリング）


# ==========================================================================
# 壁配列の基本操作
# ==========================================================================
def _get(v, h, e):
    k, x, y = e
    return int(v[x, y] if k == "v" else h[x, y])


def _set(v, h, e, val):
    k, x, y = e
    if k == "v":
        v[x, y] = val
    else:
        h[x, y] = val


def post_walls(px, py):
    """格子点 (px,py) に接しうる壁の一覧（盤外は含めない）。"""
    out = []
    if py < H: out.append(("v", px, py))
    if py > 0: out.append(("v", px, py - 1))
    if px < W: out.append(("h", px, py))
    if px > 0: out.append(("h", px - 1, py))
    return out


def cells_open(v, h, a, b):
    (ax, ay), (bx, by) = a, b
    if ax == bx:
        return h[ax, max(ay, by)] == 0
    return v[max(ax, bx), ay] == 0


def all_cells_reachable(v, h):
    seen = {(0, 0)}
    dq = deque([(0, 0)])
    while dq:
        cx, cy = dq.popleft()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            n = (cx + dx, cy + dy)
            if 0 <= n[0] < W and 0 <= n[1] < H and n not in seen and cells_open(v, h, (cx, cy), n):
                seen.add(n)
                dq.append(n)
    return len(seen) == W * H


def isolated_posts(v, h):
    """壁が 1 枚も接していない格子点（ゴール中央を除く）の一覧。"""
    out = []
    for px in range(W + 1):
        for py in range(H + 1):
            if (px, py) == CENTER_POST:
                continue
            if not any(_get(v, h, e) == 1 for e in post_walls(px, py)):
                out.append((px, py))
    return out


def wall_follow_reaches_goal(v, h, hand="left"):
    """左手法/右手法（壁づたい走行）でゴールに到達するか。スタート (0,0)・北向き。"""
    d_vec = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
    order = [-1, 0, 1, 2] if hand == "left" else [1, 0, -1, 2]
    cell, head = (0, 0), 0
    seen = set()
    for _ in range(200000):
        if cell in GOAL_CELLS:
            return True
        st = (cell, head)
        if st in seen:
            return False
        seen.add(st)
        for turn in order:
            nd = (head + turn) % 4
            dx, dy = d_vec[nd]
            nxt = (cell[0] + dx, cell[1] + dy)
            if 0 <= nxt[0] < W and 0 <= nxt[1] < H and cells_open(v, h, cell, nxt):
                cell, head = nxt, nd
                break
        else:
            return False
    return False


def independent_cycles(v, h):
    open_edges = int((v[1:W, :] == 0).sum() + (h[:, 1:H] == 0).sum())
    return open_edges - W * H + 1, open_edges


# ==========================================================================
# 生成本体
# ==========================================================================
def _spanning_tree(rng, v, h):
    """DFS（再帰的バックトラッカ）で全域木を作る（開通 255 辺）。"""
    visited = np.zeros((W, H), bool)
    stack = [(0, 0)]
    visited[0, 0] = True
    while stack:
        cx, cy = stack[-1]
        nbrs = []
        for d, (dx, dy) in (("N", (0, 1)), ("E", (1, 0)), ("S", (0, -1)), ("W", (-1, 0))):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < W and 0 <= ny < H and not visited[nx, ny]:
                nbrs.append((d, nx, ny))
        if not nbrs:
            stack.pop()
            continue
        d, nx, ny = rng.choice(nbrs)
        if d == "N": h[cx, cy + 1] = 0
        elif d == "S": h[cx, cy] = 0
        elif d == "E": v[cx + 1, cy] = 0
        else: v[cx, cy] = 0
        visited[nx, ny] = True
        stack.append((nx, ny))


def _forced_open_edges():
    """ゴールリングを外周壁から切り離すために強制開放する壁の集合。

    ゴール外周の 8 格子点に接する壁のうち、リング自身の 8 辺（と内壁 4 枚）を
    除いたものが対象。ここを開けておくことでリングは独立した島になる。
    """
    ring = set(RING_EDGES) | set(GOAL_INNER)
    forced = set()
    for (px, py) in RING_POSTS:
        for e in post_walls(px, py):
            if e not in ring:
                forced.add(e)
    return forced


FORCED_OPEN = _forced_open_edges()


def generate_maze(seed, extra_open_target=EXTRA_OPEN_TARGET, max_attempts=MAX_ATTEMPTS):
    """規定準拠の 16x16 迷路を生成する。

    返り値: (v_walls, h_walls, info)。info には試行回数・閉路数等を含む。
    受け入れ条件を満たすまで同一 seed 内で内部乱数を進めて再試行する
    （リジェクトサンプリング）。
    """
    rng = random.Random(seed)
    for attempt in range(1, max_attempts + 1):
        v = np.ones((W + 1, H), dtype=int)
        h = np.ones((W, H + 1), dtype=int)

        # 1. 全域木
        _spanning_tree(rng, v, h)

        # 2. ゴール: 内壁を開放し、外周 8 辺をいったん全て壁にする
        for e in GOAL_INNER:
            _set(v, h, e, 0)
        for e in RING_EDGES:
            _set(v, h, e, 1)

        # 3. 入口をちょうど 1 箇所開ける（IEEE 4.3）
        gateway = rng.choice(RING_EDGES)
        _set(v, h, gateway, 0)

        # 4. リングを島にする（外周壁と連結させない）
        for e in FORCED_OPEN:
            _set(v, h, e, 0)

        # 5. スタート区画 (0,0): 3 方向壁・北のみ開口（IEEE 4.3）
        v[0, 0] = 1; h[0, 0] = 1; v[1, 0] = 1; h[0, 1] = 0

        protected_open = set(FORCED_OPEN) | {gateway} | set(GOAL_INNER) | {("h", 0, 1)}
        protected_wall = (set(RING_EDGES) - {gateway}) | {("v", 1, 0)}

        # 6. 閉路を作る: 内部壁をランダムに除去（格子点の孤立を招くものは除く）
        internal = [("v", x, y) for x in range(1, W) for y in range(H)] + \
                   [("h", x, y) for x in range(W) for y in range(1, H)]
        rng.shuffle(internal)
        opened = 0
        for e in internal:
            if opened >= extra_open_target:
                break
            if e in protected_wall or _get(v, h, e) == 0:
                continue
            _set(v, h, e, 0)
            # 「終点中央を除く全格子点に壁が最低 1 枚」= 2x2 の広場を作らない規定
            k, x, y = e
            posts = ((x, y), (x, y + 1)) if k == "v" else ((x, y), (x + 1, y))
            if any(p != CENTER_POST and not any(_get(v, h, pe) == 1 for pe in post_walls(*p))
                   for p in posts):
                _set(v, h, e, 1)   # 取り消し
            else:
                opened += 1

        # 7. 孤立格子点の修復（強制開放の副作用等）。protected_open は塞がない
        ok_repair = True
        for (px, py) in isolated_posts(v, h):
            cands = [e for e in post_walls(px, py) if e not in protected_open]
            if not cands:
                ok_repair = False
                break
            _set(v, h, rng.choice(cands), 1)
        if not ok_repair:
            continue

        # 8. 受け入れ条件の検査（構造の意図を実挙動で確認する）
        gateways = sum(1 for e in RING_EDGES if _get(v, h, e) == 0)
        if gateways != 1:
            continue
        if not all_cells_reachable(v, h):
            continue
        if isolated_posts(v, h):
            continue
        if any(_get(v, h, e) == 1 for e in post_walls(*CENTER_POST)):
            continue
        if wall_follow_reaches_goal(v, h, "left") or wall_follow_reaches_goal(v, h, "right"):
            continue
        cycles, open_edges = independent_cycles(v, h)
        info = dict(seed=seed, attempts=attempt, cycles=int(cycles),
                    open_edges=int(open_edges), gateway=list(gateway),
                    extra_opened=int(opened))
        return v, h, info

    raise RuntimeError(f"seed={seed}: {max_attempts} 回試行しても受け入れ条件を満たせませんでした")


def main():
    ap = argparse.ArgumentParser(description="規定準拠 評価迷路の生成（v2）")
    ap.add_argument("--seeds", default="1000-1019")
    ap.add_argument("--out-dir", default="competition/mazes/eval")
    ap.add_argument("--extra-open", type=int, default=EXTRA_OPEN_TARGET)
    args = ap.parse_args()

    a, _, b = args.seeds.partition("-")
    seeds = range(int(a), int(b) + 1) if b else [int(a)]

    from mouse.mjcf import build_maze_robot_xml
    from mouse.params import RobotParams
    params = RobotParams()
    os.makedirs(args.out_dir, exist_ok=True)

    infos = []
    for s in seeds:
        v, h, info = generate_maze(s, extra_open_target=args.extra_open)
        npz = os.path.join(args.out_dir, f"maze_{s}.npz")
        np.savez(npz, v_walls=v, h_walls=h, seed=s, width=W, height=H)
        build_maze_robot_xml(v, h, npz[:-4] + ".xml", model_name=f"maze_{s}", params=params)
        infos.append(info)
        print(f"[maze_gen_v2] seed={s} 試行{info['attempts']}回 閉路{info['cycles']} "
              f"開通{info['open_edges']} 入口={info['gateway']}")

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(dict(generator="competition/maze_gen_v2.py",
                        rules=["IEEE R2SAC 4.3 (gateway=1, start 3 walls)",
                               "IEEE R2SAC 4.5 (wall-follower cannot reach goal, multiple paths)",
                               "NTF 注意9 (no walls/posts inside goal)",
                               "NTF 2-4 (every post has >=1 wall except goal center; outer walls complete)"],
                        extra_open_target=args.extra_open, mazes=infos), f, indent=2, ensure_ascii=False)
    cyc = [i["cycles"] for i in infos]
    att = [i["attempts"] for i in infos]
    print(f"\n生成 {len(infos)} 面: 閉路数 中央値 {np.median(cyc):.0f}（範囲 {min(cyc)}〜{max(cyc)}）"
          f"／試行回数 中央値 {np.median(att):.0f}（最大 {max(att)}）")


if __name__ == "__main__":
    main()
