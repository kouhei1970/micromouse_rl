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

■ 経路長の設計（2026-08-11 改修。docs/MAZE_DIFFICULTY_REPORT.md §5 案 3）
改修前は「内部壁をランダムに 30 枚除去」していたため、除去した壁の一部が
スタート–ゴール最短経路をまたぐ「弦」に当たり、最短距離が一気に短縮されていた。
実測では大会実迷路 42 面の真の最短距離 D_true が中央値 63 区画（迂回率 4.80）
なのに対し、改修前の生成迷路 20 面は中央値 20 区画（迂回率 1.43）で、
**両者の分布はまったく重ならなかった**（同レポート §1）。

本改修は経路長を大会迷路の水準へ引き上げる。手順は 2 点だけ変える。

  (1) 受理窓: 全域木＋ゴールリング処理の直後に D0（この時点の真の最短距離）を測り、
      D0_WINDOW = [45, 110] 区画に入らない試行を破棄する（既存のリジェクト
      サンプリング機構をそのまま使う）。ここで下限を課すのは、手順 4 の強制開放
      （ゴールリングの島化）が最短経路の終端に必ずショートカットを作り、
      平均 −30 区画の短縮を生むため。強制開放は壁づたい対策の根幹なので削らず、
      短くなった試行を捨てることで対処する。
  (2) 経路保護型の内壁除去: 除去候補を 1 枚開けるたびに BFS で最短距離を測り直し、
      **1 区画でも縮むなら取り消す**。つまり最短距離を縮めない壁だけを開ける。

(2) が成立する根拠は反証実験にある（同レポート §4.2）。「閉路を増やすと最短経路が
短くなる」は**偽**で、正しくは「**ランダムな位置に**閉路を作ると短くなる」。
経路を保護しながら閉路を作れば、除去枚数を 10→25 枚（β を 15→30）に増やしても
D_true の分布は 1 区画も動かない。すなわち**経路長（受理窓）と閉路数（除去枚数）は
直交する 2 つの難度パラメータ**として独立に制御できる。

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
#
# 【2026-08-11 改修】30 → 15。除去は「最短距離を縮めない壁だけ」に限定した
# （経路保護型除去）。docs/MAZE_DIFFICULTY_REPORT.md §5 案 3。
# 経路保護版で 3 つの独立 seed ブロック（各 20 面）を試作した実測:
#   除去目標  独立閉路 β 中央値  D_true 中央値  行止り  最短経路本数
#       10          15               72          —        —
#       15          20               72          22       3        ← 採用
#       20          25               72          —        —
#       25          30               72          —        —
# **D_true が除去枚数に依存しない**（β を 15→30 に倍増しても 1 区画も動かない）
# のが経路保護型除去の要点で、これにより経路長と閉路数を独立に決められる。
# 15 を採用する理由は β の実現値 19〜20 が大会実迷路 42 面の β 中央値 20 と一致し、
# 行き止まり数 22（大会 24）・最短経路本数 3（大会 5）も同時に整合するため。
#
# 改修前（ランダム除去）の設計根拠（seed 1000-1003 で実測した分岐次数分布）。
# 経路保護型では除去枚数と次数分布の対応が変わるため、この表は**履歴**として残す:
#   目標  閉路  平均次数  行止り/通路・角/T字/十字
#      0     4     2.02    29 / 194 / 31 /  1   ← 完全迷路。複数経路がなく IEEE 4.5 に反する
#     15    18     2.14    22 / 178 / 51 /  3
#     30    34     2.25    17 / 161 / 71 /  5   ← 旧採用値（経路が短くなる欠陥あり）
#     60    64     2.49     8 / 125 /111 / 10   ← 分岐が通路を上回り「広場」的で実競技と乖離
# 除去枚数の上限は「終点中央を除く全格子点に壁が最低 1 枚」の規定
# （＝ 2x2 の広場を禁じる）が自動的に与える。
EXTRA_OPEN_TARGET = 15

# 真の最短距離 D0 の受理窓 [下限, 上限]（区画数）。全域木＋ゴールリング処理の
# 直後に測り、窓の外なら試行を破棄する。
# 設計根拠: 大会実迷路 42 面の D_true は中央値 63・範囲 40〜249（同レポート §2）。
# 下限 45 は大会の最小値 40 をわずかに上回る水準、上限 110 は生成コストと
# エピソード長の上限から置いた実務的な頭打ち（大会の 249 のような極端な長距離面は
# 学習・評価のコストが跳ね上がるため意図的に除外している）。
D0_WINDOW = (45, 110)

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


def shortest_distance_to_goal(v, h):
    """スタート (0,0) からゴール 2x2 までの真の最短距離（区画数）。

    壁が完全に既知である前提の 4 近傍 BFS。ゴールへ到達できない場合は -1 を返す。
    区画数を返すので、物理距離に直すには区画幅 0.18 m を掛ける。
    """
    dist = {(0, 0): 0}
    dq = deque([(0, 0)])
    while dq:
        c = dq.popleft()
        if c in GOAL_CELLS:
            return dist[c]
        cx, cy = c
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            n = (cx + dx, cy + dy)
            if 0 <= n[0] < W and 0 <= n[1] < H and n not in dist and cells_open(v, h, c, n):
                dist[n] = dist[c] + 1
                dq.append(n)
    return -1


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


def generate_maze(seed, extra_open_target=EXTRA_OPEN_TARGET, max_attempts=MAX_ATTEMPTS,
                  d0_window=D0_WINDOW):
    """規定準拠の 16x16 迷路を生成する。

    返り値: (v_walls, h_walls, info)。info には試行回数・閉路数・最短距離等を含む。
    受け入れ条件を満たすまで同一 seed 内で内部乱数を進めて再試行する
    （リジェクトサンプリング）。したがって seed と本関数の実装だけから
    決定的に再現できる（研究計画書 §9-2）。

    d0_window: 手順 5 の直後に測った最短距離 D0 の受理窓 [下限, 上限]（区画数）。
               None を渡すと窓による棄却を行わない（改修前の挙動の再現用）。
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

        # 5.5 受理窓: この時点（全域木＋ゴールリング処理の直後）の最短距離 D0 を測る。
        #     窓の外なら以降の処理をせずに次の試行へ。手順 4 の強制開放が最短経路の
        #     終端にショートカットを作って距離を縮めるため、ここで短い骨格を捨てる。
        d0 = shortest_distance_to_goal(v, h)
        if d0 < 0:
            continue
        if d0_window is not None and not (d0_window[0] <= d0 <= d0_window[1]):
            continue

        # 6. 閉路を作る: 内部壁をランダム順に走査して除去（格子点の孤立を招くもの、
        #    および**最短距離を 1 区画でも縮めるものは取り消す**＝経路保護型除去）
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
                _set(v, h, e, 1)   # 取り消し（広場禁止規定）
                continue
            # 経路保護: 開けた結果 D が縮むならこの壁は最短経路をまたぐ「弦」なので戻す。
            # 縮まない壁だけを開けることで、閉路数と経路長を独立に制御できる。
            if shortest_distance_to_goal(v, h) < d0:
                _set(v, h, e, 1)   # 取り消し（最短距離が縮む）
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
        # D_final は手順 7（孤立格子点の修復）で壁を**足した**後の最短距離。
        # 手順 6 は最短距離を縮めない壁しか開けず、手順 7 は壁を足すだけなので
        # 設計上 D_final >= D0 が常に成り立つ（等号が普通）。破れたら実装バグ。
        d_final = shortest_distance_to_goal(v, h)
        if d_final < d0:
            raise AssertionError(
                f"seed={seed}: 経路保護の不変条件が破れています D_final={d_final} < D0={d0}")
        # 窓の判定は手順 7（壁を足す修復）の**前**に行っているので、修復が距離を
        # 伸ばした結果、最終的な迷路が窓の外に出ることがありうる。最終形でも
        # 窓に入っていることを受け入れ条件として確認する（黙って外れるのを防ぐ）。
        if d0_window is not None and not (d0_window[0] <= d_final <= d0_window[1]):
            continue
        info = dict(seed=seed, attempts=attempt, cycles=int(cycles),
                    open_edges=int(open_edges), gateway=list(gateway),
                    extra_opened=int(opened), d0=int(d0), d_shortest=int(d_final))
        return v, h, info

    raise RuntimeError(f"seed={seed}: {max_attempts} 回試行しても受け入れ条件を満たせませんでした")


def main():
    ap = argparse.ArgumentParser(description="規定準拠 評価迷路の生成（v2）")
    ap.add_argument("--seeds", default="1000-1019")
    ap.add_argument("--out-dir", default="competition/mazes/eval")
    ap.add_argument("--extra-open", type=int, default=EXTRA_OPEN_TARGET)
    ap.add_argument("--d0-window", default="{}-{}".format(*D0_WINDOW),
                    help="最短距離 D0 の受理窓（例 45-110）。'none' で窓なし")
    args = ap.parse_args()

    if args.d0_window.lower() == "none":
        window = None
    else:
        lo, _, hi = args.d0_window.partition("-")
        window = (int(lo), int(hi))

    a, _, b = args.seeds.partition("-")
    seeds = range(int(a), int(b) + 1) if b else [int(a)]

    from mouse.mjcf import build_maze_robot_xml
    from mouse.params import RobotParams
    params = RobotParams()
    os.makedirs(args.out_dir, exist_ok=True)

    infos = []
    for s in seeds:
        v, h, info = generate_maze(s, extra_open_target=args.extra_open, d0_window=window)
        npz = os.path.join(args.out_dir, f"maze_{s}.npz")
        np.savez(npz, v_walls=v, h_walls=h, seed=s, width=W, height=H)
        build_maze_robot_xml(v, h, npz[:-4] + ".xml", model_name=f"maze_{s}", params=params)
        infos.append(info)
        print(f"[maze_gen_v2] seed={s} 試行{info['attempts']}回 閉路{info['cycles']} "
              f"開通{info['open_edges']} 入口={info['gateway']} "
              f"最短{info['d_shortest']}区画(D0={info['d0']}) 除去{info['extra_opened']}枚")

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(dict(generator="competition/maze_gen_v2.py",
                        rules=["IEEE R2SAC 4.3 (gateway=1, start 3 walls)",
                               "IEEE R2SAC 4.5 (wall-follower cannot reach goal, multiple paths)",
                               "NTF 注意9 (no walls/posts inside goal)",
                               "NTF 2-4 (every post has >=1 wall except goal center; outer walls complete)"],
                        extra_open_target=args.extra_open,
                        d0_window=list(window) if window else None,
                        path_protected_open=True, mazes=infos), f, indent=2, ensure_ascii=False)
    cyc = [i["cycles"] for i in infos]
    att = [i["attempts"] for i in infos]
    dsh = [i["d_shortest"] for i in infos]
    opn = [i["extra_opened"] for i in infos]
    print(f"\n生成 {len(infos)} 面: 閉路数 中央値 {np.median(cyc):.0f}（範囲 {min(cyc)}〜{max(cyc)}）"
          f"／試行回数 中央値 {np.median(att):.0f}（最大 {max(att)}、合計 {sum(att)}）")
    print(f"  最短距離 中央値 {np.median(dsh):.0f} 区画（範囲 {min(dsh)}〜{max(dsh)}、"
          f"迂回率 中央値 {np.median(dsh)/14:.2f}）／実除去 中央値 {np.median(opn):.0f} 枚"
          f"（範囲 {min(opn)}〜{max(opn)}、目標 {args.extra_open}）")


if __name__ == "__main__":
    main()
