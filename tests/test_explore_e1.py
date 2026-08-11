"""
tests/test_explore_e1.py
================
探索戦略 E1（competition/explore_e1.py）の単体テスト。

最重要の検証: **確定判定の正しさ**
「relevant な未知壁が空 ⇒ 楽観最短距離 = 真の最短距離」という理論を、
実際の評価迷路 20 面に対して**探索の全過程をシミュレートして反証形式で確認**する。
確定を宣言した時点の楽観距離が真の最短距離と 1 区画でも違えば理論が破れる。

実行: .venv/bin/python tests/test_explore_e1.py
"""
import glob
import os
import re
import sys
from collections import deque

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from competition.explore_e1 import (  # noqa: E402
    UNKNOWN, OPEN, WALL,
    bfs_distances, is_shortest_confirmed, relevant_unknown_walls,
    shortest_path_cells, wall_state,
)

W = H = 16
GOALS = [(7, 7), (7, 8), (8, 7), (8, 8)]
START = (0, 0)
RESULTS = []


def record(name, expected, actual, ok, note=""):
    RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: expected={expected}, actual={actual} {note}")


def blank_known():
    """全て未知の地図（外周だけは既知の壁）。実機も外周は既知として扱える。"""
    v = np.full((W + 1, H), UNKNOWN, dtype=int)
    h = np.full((W, H + 1), UNKNOWN, dtype=int)
    v[0, :] = WALL
    v[W, :] = WALL
    h[:, 0] = WALL
    h[:, H] = WALL
    return v, h


def observe_cell(vk, hk, vt, ht, x, y):
    """区画 (x,y) の 4 方向の壁を真の迷路から観測して既知地図へ書き込む。"""
    if x + 1 <= W:
        vk[x + 1, y] = vt[x + 1, y]
    vk[x, y] = vt[x, y]
    if y + 1 <= H:
        hk[x, y + 1] = ht[x, y + 1]
    hk[x, y] = ht[x, y]


def true_shortest(vt, ht):
    d = bfs_distances(vt, ht, W, H, GOALS)
    return d.get(START)


def simulate_e1(vt, ht, max_steps=20000):
    """E1 の探索を区画単位でシミュレートし、確定時の楽観距離と訪問区画数を返す。

    ①往路: 楽観的にゴールへ ②追加探索: relevant が空になるまで
    （移動は「楽観地図での最短経路を 1 区画進む」を繰り返す簡略版）
    """
    vk, hk = blank_known()
    cur = START
    observe_cell(vk, hk, vt, ht, *cur)
    visited = {cur}

    # ① 往路: ゴールへ
    for _ in range(max_steps):
        if cur in set(GOALS):
            break
        path = shortest_path_cells(vk, hk, W, H, cur, GOALS)
        if path is None or len(path) < 2:
            break
        cur = path[1]
        observe_cell(vk, hk, vt, ht, *cur)
        visited.add(cur)

    # ② 追加探索ループ
    for _ in range(max_steps):
        rel = relevant_unknown_walls(vk, hk, W, H, START, GOALS)
        if not rel:
            break
        targets = {c for pair in rel for c in pair}
        path = shortest_path_cells(vk, hk, W, H, cur, targets)
        if path is None or len(path) < 2:
            break
        cur = path[1]
        observe_cell(vk, hk, vt, ht, *cur)
        visited.add(cur)

    d_opt = bfs_distances(vk, hk, W, H, GOALS).get(START)
    return d_opt, len(visited), is_shortest_confirmed(vk, hk, W, H, START, GOALS)


def eval_mazes():
    """評価帯の npz を **seed 昇順**で返す。

    ⚠️ **面の名前をハードコードしないこと（2026-08-12 是正）。**
    帯を v4 へ入れ替えたとき、`maze_1000.npz` / `maze_1003.npz` を直接開いていた
    テスト 2・3 が `FileNotFoundError` で落ちた。**採用 seed は帯ごとに変わる**ので、
    「何番目の面か」で選ぶ（どの帯でも成立し、かつ決定的）。
    """
    fs = glob.glob(os.path.join(REPO_ROOT, "competition/mazes/eval/maze_*.npz"))
    return sorted(fs, key=lambda p: int(re.search(r"\d+", os.path.basename(p)).group()))


def test1_confirmation_theory():
    print("\n=== テスト1: 確定判定の理論（評価迷路 20 面で反証形式に検証） ===")
    files = eval_mazes()
    ok_all = True
    rows = []
    for f in files:
        z = np.load(f)
        vt, ht = z["v_walls"], z["h_walls"]
        d_true = true_shortest(vt, ht)
        d_opt, n_visited, confirmed = simulate_e1(vt, ht)
        ok = confirmed and (d_opt == d_true)
        ok_all = ok_all and ok
        rows.append((int(z["seed"]), d_true, d_opt, n_visited, confirmed))
    for seed, dt, do, nv, cf in rows[:5]:
        print(f"    seed={seed}: 真の最短 {dt} / 確定時の楽観距離 {do} / 訪問区画 {nv}/256 / 確定 {cf}")
    n_ok = sum(1 for _, dt, do, nv, cf in rows if cf and dt == do)
    record("確定時の楽観距離 = 真の最短距離（20面）", "20/20", f"{n_ok}/{len(rows)}", ok_all)
    visited = [nv for _, _, _, nv, _ in rows]
    print(f"    訪問区画数: 中央値 {int(np.median(visited))}/256（全面探索が不要であることの実証）")
    return ok_all


def test2_pessimistic_return():
    print("\n=== テスト2: 帰路は悲観的（未知壁を通らない） ===")
    f = eval_mazes()[0]          # 帯の 1 番目（名前ではなく順番で選ぶ）
    z = np.load(f)
    print(f"    使用: {os.path.basename(f)}")
    vt, ht = z["v_walls"], z["h_walls"]
    vk, hk = blank_known()
    # 往路をシミュレートして地図を作る
    cur = START
    observe_cell(vk, hk, vt, ht, *cur)
    for _ in range(2000):
        if cur in set(GOALS):
            break
        p = shortest_path_cells(vk, hk, W, H, cur, GOALS)
        if p is None or len(p) < 2:
            break
        cur = p[1]
        observe_cell(vk, hk, vt, ht, *cur)
    # 悲観的な帰路が、既知で開いている辺のみを使うこと
    back = shortest_path_cells(vk, hk, W, H, cur, [START], pessimistic=True)
    ok_exists = back is not None
    uses_unknown = False
    if back:
        for a, b in zip(back, back[1:]):
            if wall_state(vk, hk, a[0], a[1], b[0], b[1]) != OPEN:
                uses_unknown = True
    record("悲観的帰路が存在する", True, ok_exists, ok_exists)
    record("帰路が未知壁を一切使わない", False, uses_unknown, not uses_unknown,
           f"経路長 {len(back) if back else '-'} 区画")
    # 楽観的帰路と比べて長いか同じ（悲観は制約が強いので短くはならない）
    opt_back = shortest_path_cells(vk, hk, W, H, cur, [START], pessimistic=False)
    ok_len = (opt_back is None) or (len(back) >= len(opt_back))
    record("悲観的帰路は楽観的帰路以上の長さ", True, ok_len, ok_len,
           f"悲観 {len(back)} vs 楽観 {len(opt_back) if opt_back else '-'}")
    return ok_exists and not uses_unknown and ok_len


def test3_confirmed_on_full_map():
    print("\n=== テスト3: 地図が完全に既知なら即座に確定 ===")
    f = eval_mazes()[3]          # 帯の 4 番目（名前ではなく順番で選ぶ）
    z = np.load(f)
    print(f"    使用: {os.path.basename(f)}")
    vt, ht = z["v_walls"], z["h_walls"]
    conf = is_shortest_confirmed(vt, ht, W, H, START, GOALS)
    rel = relevant_unknown_walls(vt, ht, W, H, START, GOALS)
    record("全既知地図で確定", True, conf, conf, f"relevant={len(rel)} 件")
    # 逆に全未知（外周のみ既知）では確定していない
    vk, hk = blank_known()
    conf0 = is_shortest_confirmed(vk, hk, W, H, START, GOALS)
    record("全未知地図では未確定", False, conf0, not conf0)
    return conf and not conf0


def main():
    for fn in (test1_confirmation_theory, test2_pessimistic_return, test3_confirmed_on_full_map):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] {fn.__name__}: {e}")
            RESULTS.append((fn.__name__, False))
    n_ok = sum(1 for _, ok in RESULTS if ok)
    print("\n" + "=" * 78)
    print(f"合計: {n_ok}/{len(RESULTS)} PASS")
    print("=" * 78)
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
