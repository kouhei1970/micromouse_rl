#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
監査 040: 壁伝い法の到達性の独立再計算（exp_021 カード §1-2 の 604/1000 の照合）

准教授セッション（8 代目）・2026-08-14
**判定形は `verification/AUDIT_040_PREREG_wall_follower.md` に実装前にコミット済み（90f2dc5）。**

盲検: 相手の実装 `research_notes/scripts/check_wall_follower_reachability.py` と
生出力 `outputs/wall_follower_reachability.json` は読んでいない。
壁伝い法の中核は `verification/wall_follower_core.py` に**迷路の壁の形式を見る前に**書いた。

限界: 迷路生成器 `mouse/maze6_gen.py` は共有しているので、生成器に誤りがあれば
両者は同じように誤る（作法 36）。また結果の数値を先に読んでいるので、
一致したときの情報量は数値を伏せた盲検より落ちる（不一致のときは落ちない）。
"""

import json
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from common.seed_bands import assert_seeds_allowed, describe_seeds  # noqa: E402
from mouse.maze6_gen import generate_maze  # noqa: E402
from verification.wall_follower_core import (DIR_NAMES, goal_walls_connected_to_outer,  # noqa: E402
                                             wall_follow)

SEEDS = list(range(8000, 9000))   # 学習用の 1000 迷路（カード §1-2 と同じ範囲）
MODE = "loop"                     # 学習環境の既定（mouse/maze6_env.py:267）

# ---- カード §1-2 が主張する値（AUDIT_040 事前登録 §2 の表） ----
CLAIM = dict(
    left=396, right=396, either=396, neither=604,
    step_limit=0, topo_disconnected=604, topo_diff=0,
    first20_reached=7,
    first20_reached_seeds={8002, 8003, 8004, 8007, 8011, 8013, 8017},
    first20_d0_4={8012, 8015, 8017, 8018, 8019},
    first20_d0_4_unreached={8012, 8015, 8018, 8019},
)

results = []


def record(item, ok, detail):
    results.append((item, ok, detail))
    mark = "PASS" if ok is True else ("FAIL" if ok is False else "INFO")
    print(f"  [{mark}] {item}: {detail}")


def make_blocked(m):
    """壁の配列から `blocked((x,y), d)` を作る。

    生成器の規約（mouse/maze6_gen.py:21-24）:
      v_walls[x, y] = 区画 (x-1, y) と (x, y) の間の縦壁   shape (W+1, H)
      h_walls[x, y] = 区画 (x, y-1) と (x, y) の間の横壁   shape (W, H+1)
      1 = 壁あり / 0 = 開通
    したがって区画 (x, y) について:
      東(0) = v_walls[x+1, y]   北(1) = h_walls[x, y+1]
      西(2) = v_walls[x,   y]   南(3) = h_walls[x, y]
    """
    v, h, W, H = m["v_walls"], m["h_walls"], m["width"], m["height"]

    def blocked(cell, d):
        x, y = cell
        if d == 0:
            return not (0 <= x + 1 < W + 1) or bool(v[x + 1, y])
        if d == 1:
            return not (0 <= y + 1 < H + 1) or bool(h[x, y + 1])
        if d == 2:
            return bool(v[x, y])
        return bool(h[x, y])

    return blocked


def bfs_distance(start, goal_cells, blocked, W, H):
    """開始区画からゴールまでの区画数（D_0）。壁を通れない幅優先探索。"""
    from collections import deque
    dist = {tuple(start): 0}
    q = deque([tuple(start)])
    while q:
        c = q.popleft()
        if c in goal_cells:
            return dist[c]
        for d, (dx, dy) in enumerate([(1, 0), (0, 1), (-1, 0), (0, -1)]):
            if blocked(c, d):
                continue
            n = (c[0] + dx, c[1] + dy)
            if 0 <= n[0] < W and 0 <= n[1] < H and n not in dist:
                dist[n] = dist[c] + 1
                q.append(n)
    return None


def main():
    print(describe_seeds(SEEDS, "maze6"))
    # 評価用に確保された seed を使っていないことを機械的に検査する（研究計画書 §9-7）
    assert_seeds_allowed(SEEDS, namespace="maze6", purpose="train")
    print(f"生成モード: {MODE}／{len(SEEDS)} 迷路\n")

    rows = []
    for s in SEEDS:
        m = generate_maze(s, mode=MODE)
        blocked = make_blocked(m)
        goal = set(map(tuple, m["goal_cells"]))
        start = tuple(m["start"])
        W, H = m["width"], m["height"]

        # スタート区画で開いている向き（生成器は 1 方向だけ開ける: _isolate_start）
        open_dirs = [d for d in range(4) if not blocked(start, d)]

        # W-9: 初期の向き 4 通りすべてで走らせる
        per_dir = {}
        for hand in ("left", "right"):
            for d0 in range(4):
                per_dir[(hand, d0)] = wall_follow(start, d0, goal, blocked,
                                                 max_steps=100_000, hand=hand)

        left = per_dir[("left", open_dirs[0] if open_dirs else 0)]
        right = per_dir[("right", open_dirs[0] if open_dirs else 0)]

        connected, n_goal_segs = goal_walls_connected_to_outer(W, H, goal, blocked)
        d0 = bfs_distance(start, goal, blocked, W, H)

        rows.append(dict(
            seed=s, start=start, n_open_at_start=len(open_dirs), d0=d0,
            left_reached=left["reached"], left_reason=left["reason"], left_steps=left["steps"],
            right_reached=right["reached"], right_reason=right["reason"], right_steps=right["steps"],
            topo_connected=connected, n_goal_wall_segments=n_goal_segs,
            # W-9: 向きに依存するか
            left_by_dir=[per_dir[("left", d)]["reached"] for d in range(4)],
            right_by_dir=[per_dir[("right", d)]["reached"] for d in range(4)],
            # W-10: 訪れた区画にゴールが含まれるか（到達判定とは別経路）
            left_visited_goal=bool(left["visited"] & goal),
            right_visited_goal=bool(right["visited"] & goal),
        ))

    # ================================================================
    print("=" * 72)
    print("W-1〜W-4: 到達件数")
    print("=" * 72)
    L = {r["seed"] for r in rows if r["left_reached"]}
    R = {r["seed"] for r in rows if r["right_reached"]}
    either, neither = L | R, {r["seed"] for r in rows} - (L | R)

    record("W-1 左手法で到達", len(L) == CLAIM["left"],
           f"再計算 {len(L)}/1000 = {len(L) / 10:.1f}% / 主張 {CLAIM['left']}")
    record("W-2 右手法で到達", len(R) == CLAIM["right"],
           f"再計算 {len(R)}/1000 = {len(R) / 10:.1f}% / 主張 {CLAIM['right']}")
    record("W-3 左右どちらかで到達", len(either) == CLAIM["either"],
           f"再計算 {len(either)}/1000 / 主張 {CLAIM['either']}")
    record("W-4 左右とも未到達", len(neither) == CLAIM["neither"],
           f"再計算 {len(neither)}/1000 = {len(neither) / 10:.1f}% / 主張 {CLAIM['neither']}")
    record("W-3b 左と右の到達集合が同一か（件数一致だけでは同じ集合とは限らない）",
           L == R, "完全一致" if L == R else f"左のみ {len(L - R)} 件 / 右のみ {len(R - L)} 件")

    # ================================================================
    print("\n" + "=" * 72)
    print("W-5: 未到達の確定の仕方（打ち切りが 0 件か）")
    print("=" * 72)
    reasons = Counter(r["left_reason"] for r in rows) + Counter(r["right_reason"] for r in rows)
    n_limit = reasons.get("step_limit", 0)
    record("W-5 歩数の上限による打ち切り", n_limit == CLAIM["step_limit"],
           f"再計算 {n_limit} 件（左右合計 2000 走行）/ 主張 {CLAIM['step_limit']} 件。内訳 {dict(reasons)}")

    # ================================================================
    print("\n" + "=" * 72)
    print("W-6: 位相による説明（ゴールを囲む壁が外周と繋がっていない）")
    print("=" * 72)
    disc = {r["seed"] for r in rows if not r["topo_connected"]}
    record("W-6a 繋がっていない迷路の数", len(disc) == CLAIM["topo_disconnected"],
           f"再計算 {len(disc)}/1000 / 主張 {CLAIM['topo_disconnected']}")
    diff = (disc ^ neither)
    record("W-6b 未到達の集合と完全一致するか", len(diff) == CLAIM["topo_diff"],
           f"対称差 {len(diff)} 件 / 主張 {CLAIM['topo_diff']} 件"
           + (f"　→ {sorted(diff)[:10]}" if diff else ""))

    # ================================================================
    print("\n" + "=" * 72)
    print("W-9（私の追加）: 壁伝いの結果は初期の向きに依存するか")
    print("=" * 72)
    n_open = Counter(r["n_open_at_start"] for r in rows)
    dep_l = sum(1 for r in rows if len(set(r["left_by_dir"])) > 1)
    dep_r = sum(1 for r in rows if len(set(r["right_by_dir"])) > 1)
    record("W-9a スタート区画で開いている向きの数", set(n_open) == {1},
           f"内訳 {dict(n_open)}（生成器 _isolate_start が 1 方向だけ開ける仕様）")
    record("W-9b 初期の向き 4 通りで到達結果が変わる迷路", dep_l == 0 and dep_r == 0,
           f"左手法 {dep_l} 件 / 右手法 {dep_r} 件")

    # ================================================================
    print("\n" + "=" * 72)
    print("W-10（私の追加）: 訪れた区画にゴールが含まれるか（到達判定と別経路）")
    print("=" * 72)
    mism = sum(1 for r in rows
               if r["left_visited_goal"] != r["left_reached"]
               or r["right_visited_goal"] != r["right_reached"])
    record("W-10 到達判定と訪問集合の整合", mism == 0, f"不整合 {mism}/1000")

    # ================================================================
    print("\n" + "=" * 72)
    print("W-7 / W-8: seed 8000〜8019 の内訳")
    print("=" * 72)
    first20 = [r for r in rows if r["seed"] < 8020]
    got = {r["seed"] for r in first20 if r["left_reached"] or r["right_reached"]}
    record("W-7a 到達した迷路の数", len(got) == CLAIM["first20_reached"],
           f"再計算 {len(got)}/20 / 主張 {CLAIM['first20_reached']}")
    record("W-7b 到達した seed の集合", got == CLAIM["first20_reached_seeds"],
           f"再計算 {sorted(got)} / 主張 {sorted(CLAIM['first20_reached_seeds'])}")

    d4 = {r["seed"] for r in first20 if r["d0"] == 4}
    d4_un = {r["seed"] for r in first20 if r["d0"] == 4
             and not (r["left_reached"] or r["right_reached"])}
    record("W-8a D_0=4 の迷路", d4 == CLAIM["first20_d0_4"],
           f"再計算 {sorted(d4)} / 主張 {sorted(CLAIM['first20_d0_4'])}")
    record("W-8b うち未到達", d4_un == CLAIM["first20_d0_4_unreached"],
           f"再計算 {sorted(d4_un)} / 主張 {sorted(CLAIM['first20_d0_4_unreached'])}")

    print("\n  seed 8000〜8019 の明細:")
    print("  seed   start   D_0  左   右   位相")
    for r in first20:
        print(f"  {r['seed']}  {str(r['start']):>7}  {r['d0']:>3}  "
              f"{'到達' if r['left_reached'] else ' ✗ '}  "
              f"{'到達' if r['right_reached'] else ' ✗ '}  "
              f"{'外周と連結' if r['topo_connected'] else '**切離**'}")

    # ================================================================
    print("\n" + "=" * 72)
    n_pass = sum(1 for _, ok, _ in results if ok is True)
    n_fail = sum(1 for _, ok, _ in results if ok is False)
    print(f"総括: PASS {n_pass} / FAIL {n_fail} / INFO "
          f"{sum(1 for _, ok, _ in results if ok is None)}")
    if n_fail:
        print("\n  🔴 不一致（**まず自分の実装を疑うこと** — 作法 35）:")
        for item, ok, detail in results:
            if ok is False:
                print(f"    {item}: {detail}")

    out = os.path.join(REPO, "outputs/verification/wall_follower_independent.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(seeds=[SEEDS[0], SEEDS[-1]], mode=MODE, rows=rows,
                       summary=dict(left=len(L), right=len(R), either=len(either),
                                    neither=len(neither), topo_disconnected=len(disc),
                                    topo_symmetric_difference=sorted(diff))),
                  f, ensure_ascii=False, indent=1)
    print(f"\n再計算の生記録: {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
