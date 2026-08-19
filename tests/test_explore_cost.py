"""
tests/test_explore_cost.py
==========================
`competition/explore_cost.py`（経路比 R の計算）の単体テスト。

R は評価迷路の受理条件に使う量なので、**独立実装との一致**を全数で確認する。
照合先は `verification/maze_exploration_cost.py`（准教授セッションが古典
アルゴリズムの定義から独立に書いたもの）。両者は同じアルゴリズムの別実装で、
**依存関係はない**（評価迷路は凍結物なので、生成が他セッションの管理下にある
ファイルに依存してはならない）。

実行: .venv/bin/python tests/test_explore_cost.py
"""
import glob
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "verification")):
    if p not in sys.path:
        sys.path.insert(0, p)

from competition.explore_cost import (  # noqa: E402
    detour_ratio, first_run_cells, first_run_path, true_shortest, true_shortest_path)

RESULTS = []


def record(name, expected, actual, ok, note=""):
    RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: expected={expected}, actual={actual} {note}")


def maze_files():
    pats = ["competition/reference_mazes/contest/contest_*.npz",
            "competition/mazes/eval/maze_*.npz",
            "competition/mazes/validation/maze_*.npz",
            "competition/mazes/eval_v2_short/maze_*.npz"]
    out = []
    for p in pats:
        out += sorted(glob.glob(os.path.join(REPO_ROOT, p)))
    return out


def load(f):
    z = np.load(f)
    v, h = z["v_walls"], z["h_walls"]
    if "start_x" in z.files:
        start = (int(z["start_x"]), int(z["start_y"]))
        goals = tuple((int(a), int(b)) for a, b in zip(z["goals_x"], z["goals_y"]))
    else:
        start, goals = (0, 0), ((7, 7), (7, 8), (8, 7), (8, 8))
    return v, h, start, goals


def test1_matches_independent_implementation():
    print("\n=== テスト1: 独立実装（verification/maze_exploration_cost.py）との全数一致 ===")
    try:
        from maze_exploration_cost import d_true as d_true_ap
        from maze_exploration_cost import explore_first_run, load as load_ap
    except Exception as e:  # noqa: BLE001
        record("独立実装の読み込み", "成功", f"失敗: {e}", False,
               "（准教授セッションのファイルが無い場合はこのテストを飛ばす）")
        return False
    from pathlib import Path
    files = maze_files()
    n_s = n_c = n_d = 0
    for f in files:
        m = load_ap(Path(f))
        v, h, start, goals = load(f)
        n_s += first_run_cells(v, h, start, goals, "straight") == explore_first_run(m, "straight")
        n_c += first_run_cells(v, h, start, goals, "compass") == explore_first_run(m, "compass")
        n_d += true_shortest(v, h, start, goals) == d_true_ap(m)
    n = len(files)
    ok = (n_s == n) and (n_c == n) and (n_d == n)
    record("初回探索の区画数（同点処理 straight）", f"{n}/{n}", f"{n_s}/{n}", n_s == n)
    record("初回探索の区画数（同点処理 compass）", f"{n}/{n}", f"{n_c}/{n}", n_c == n)
    record("真の最短距離 D_0", f"{n}/{n}", f"{n_d}/{n}", n_d == n)
    return ok


def test2_path_and_count_consistent():
    print("\n=== テスト2: 経路と区画数の整合（len(path)-1 == cells） ===")
    ok_all = True
    for f in maze_files():
        v, h, start, goals = load(f)
        path = first_run_path(v, h, start, goals)
        cells = first_run_cells(v, h, start, goals)
        if path is None or cells is None:
            ok_all = False
            break
        if len(path) - 1 != cells:
            ok_all = False
            break
        # 経路は隣接セルの連なりで、開通している辺だけを通る
        for a, b in zip(path, path[1:]):
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
                ok_all = False
                break
    record("全面で len(経路)-1 == 区画数 かつ経路が隣接連結", True, ok_all, ok_all)
    return ok_all


def test3_true_shortest_path_consistent():
    print("\n=== テスト3: 真の最短経路と最短距離の整合 ===")
    ok_all = True
    for f in maze_files():
        v, h, start, goals = load(f)
        d = true_shortest(v, h, start, goals)
        p = true_shortest_path(v, h, start, goals)
        if p is None or len(p) - 1 != d or p[0] != tuple(start) or p[-1] not in set(goals):
            ok_all = False
            break
    record("全面で len(最短経路)-1 == D_0・端点が正しい", True, ok_all, ok_all)
    return ok_all


def test4_detour_ratio_ge_one():
    print("\n=== テスト4: 経路比 R >= 1.0（初回探索は最短経路より短くなり得ない） ===")
    bad = []
    for f in maze_files():
        v, h, start, goals = load(f)
        r = detour_ratio(v, h, start, goals)
        if not (r >= 1.0 - 1e-12):
            bad.append((os.path.basename(f), r))
    record("全面で R >= 1.0", 0, len(bad), len(bad) == 0, str(bad[:3]))
    return not bad


def main():
    print(f"対象: {len(maze_files())} 面")
    for fn in (test1_matches_independent_implementation, test2_path_and_count_consistent,
               test3_true_shortest_path_consistent, test4_detour_ratio_ge_one):
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
