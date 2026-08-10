"""
tests/test_maze_gen_v2.py
================
規定準拠 評価迷路生成器 v2（competition/maze_gen_v2.py）の単体テスト。

研究計画書 §9-2（seed のみから決定的に再現できること）と §2（評価規約）を検証する:
  1. 決定性: 同一 seed で 2 回生成し壁配列の全要素一致（**リジェクトサンプリングの
     試行回数も含めて**一意に定まること）
  2. 凍結物との一致: 保存済み npz（test/validation）が再生成と完全一致
  3. 受け入れ条件 7 項: 保存済みの全 40 面が規定に適合
  4. seed 帯の分離: test(1000-1019) と validation(4000-4019) が異なる迷路であること

実行: .venv/bin/python tests/test_maze_gen_v2.py
"""
import glob
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from competition.audit_maze_rules import (  # noqa: E402
    goal_gateway_count,
    goal_interior_walls,
    isolated_posts,
    outer_walls_complete,
    start_cell_walls,
    wall_follow_reaches_goal,
    center_post_attached_walls,
    bfs_goal_reachable,
    independent_cycles,
)
from competition.maze_gen_v2 import generate_maze, all_cells_reachable  # noqa: E402

RESULTS = []


def record(name, expected, actual, ok, note=""):
    RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: expected={expected}, actual={actual} {note}")


def test1_determinism():
    print("\n=== テスト1: 決定性（同一 seed → 同一迷路。試行回数も一意） ===")
    ok_all = True
    for seed in (1000, 1007, 4000, 4019):
        v1, h1, i1 = generate_maze(seed)
        v2, h2, i2 = generate_maze(seed)
        same = bool(np.array_equal(v1, v2) and np.array_equal(h1, h2))
        same_attempts = i1["attempts"] == i2["attempts"]
        ok_all = ok_all and same and same_attempts
        record(f"seed={seed} 壁配列が完全一致", True, same,
               same, f"試行回数 {i1['attempts']}=={i2['attempts']}: {same_attempts}")
    return ok_all


def test2_frozen_npz_matches():
    print("\n=== テスト2: 凍結 npz と再生成の完全一致 ===")
    ok_all = True
    for d, band in (("competition/mazes/eval", "test"),
                     ("competition/mazes/validation", "validation")):
        files = sorted(glob.glob(os.path.join(REPO_ROOT, d, "maze_*.npz")))
        n_ok = 0
        for f in files:
            z = np.load(f)
            v, h, _ = generate_maze(int(z["seed"]))
            if np.array_equal(v, z["v_walls"]) and np.array_equal(h, z["h_walls"]):
                n_ok += 1
        ok = (n_ok == len(files)) and len(files) == 20
        ok_all = ok_all and ok
        record(f"{band}({d}) の凍結 npz が再生成と一致", "20/20", f"{n_ok}/{len(files)}", ok)
    return ok_all


def test3_acceptance_conditions():
    print("\n=== テスト3: 受け入れ条件 7 項（保存済み全 40 面） ===")
    checks = {k: 0 for k in ("gateway1", "wallfollow", "connected", "start3",
                              "goal_inner0", "center_post0", "isolated0", "outer", "cycles>0")}
    total = 0
    for d in ("competition/mazes/eval", "competition/mazes/validation"):
        for f in sorted(glob.glob(os.path.join(REPO_ROOT, d, "maze_*.npz"))):
            z = np.load(f); v, h = z["v_walls"], z["h_walls"]
            total += 1
            gw, _ = goal_gateway_count(v, h)
            cyc, _ = independent_cycles(v, h)
            checks["gateway1"] += (gw == 1)
            checks["wallfollow"] += (not wall_follow_reaches_goal(v, h, "left")
                                      and not wall_follow_reaches_goal(v, h, "right"))
            checks["connected"] += bool(all_cells_reachable(v, h) and bfs_goal_reachable(v, h))
            checks["start3"] += (start_cell_walls(v, h) == 3)
            checks["goal_inner0"] += (goal_interior_walls(v, h) == 0)
            checks["center_post0"] += (center_post_attached_walls(v, h) == 0)
            checks["isolated0"] += (isolated_posts(v, h) == 0)
            checks["outer"] += bool(outer_walls_complete(v, h))
            checks["cycles>0"] += (cyc > 0)
    labels = {
        "gateway1": "ゴール入口ちょうど1箇所 (IEEE 4.3)",
        "wallfollow": "左手法・右手法ともゴール到達不可 (IEEE 4.5)",
        "connected": "全256セル到達可能かつゴール到達可能",
        "start3": "スタート区画3方向壁 (IEEE 4.3)",
        "goal_inner0": "ゴール4区画内部に壁なし (NTF 注意9)",
        "center_post0": "ゴール中央の格子点に壁なし (NTF 2-4)",
        "isolated0": "孤立格子点ゼロ (NTF 2-4)",
        "outer": "外周壁完備 (NTF 2-4)",
        "cycles>0": "複数経路あり（閉路>0、IEEE 4.5）",
    }
    ok_all = True
    for k, lab in labels.items():
        ok = checks[k] == total
        ok_all = ok_all and ok
        record(lab, f"{total}/{total}", f"{checks[k]}/{total}", ok)
    return ok_all


def test4_band_separation():
    print("\n=== テスト4: seed 帯の分離（研究計画書 §9-7 データ三分割） ===")
    def load(d):
        return {int(np.load(f)["seed"]): np.load(f)["v_walls"].tobytes()
                for f in sorted(glob.glob(os.path.join(REPO_ROOT, d, "maze_*.npz")))}
    test = load("competition/mazes/eval")
    val = load("competition/mazes/validation")
    ok_seeds = (set(test) == set(range(1000, 1020))) and (set(val) == set(range(4000, 4020)))
    record("seed 帯 test=1000-1019 / validation=4000-4019", True, ok_seeds, ok_seeds)
    overlap = set(test.values()) & set(val.values())
    ok_distinct = len(overlap) == 0
    record("test と validation の迷路が重複しない", 0, len(overlap), ok_distinct)
    return ok_seeds and ok_distinct


def main():
    fns = [test1_determinism, test2_frozen_npz_matches, test3_acceptance_conditions,
           test4_band_separation]
    for fn in fns:
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
