"""
tests/test_maze_gen_v3.py
=========================
選抜込み評価迷路生成器 v3（`competition/maze_gen_v3.py`）の単体テスト。

**選抜を入れると帯は「無作為抽出」ではなく選ばれた部分集合になる**ので、
研究計画書 §9-2（生成は seed のみから決定的）を満たすことを、
**選抜まで含めて**確認する必要がある（教授の追加条件 C）。

  1. 候補の決定性: 同一 seed で 2 回作り、壁配列が全要素一致
  2. **選抜の決定性**: 同じ seed 範囲・同じ規則から**常に同じ 20 面**が出る
  3. 凍結物との一致: 保存済み npz が再生成と完全一致
  4. 受け入れ条件（規定 6 項目 + D の窓）が保存済み全 40 面で成立
  5. seed 帯の分離: eval_v3 は 1000-1999、validation_v3 は 4000-4999 から採られ、
     迷路が重複しない
  6. manifest に候補全数の採否が残っている（なぜこの面が選ばれたかを追える）

実行: .venv/bin/python tests/test_maze_gen_v3.py
"""
import glob
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from competition.audit_maze_rules import (  # noqa: E402
    bfs_goal_reachable, center_post_attached_walls, goal_gateway_count,
    goal_interior_walls, independent_cycles, isolated_posts, outer_walls_complete,
    start_cell_walls, wall_follow_reaches_goal)
from competition.explore_cost import detour_ratio, true_shortest  # noqa: E402
from competition.maze_gen_v2 import all_cells_reachable  # noqa: E402
from competition.maze_gen_v3 import (  # noqa: E402
    D_WINDOW, R_BAND_EDGES, band_of, generate_candidate, select_stratified)

BANDS = [("competition/mazes/eval_v3", "eval_v3", (1000, 1999)),
         ("competition/mazes/validation_v3", "validation_v3", (4000, 4999))]
RESULTS = []


def record(name, expected, actual, ok, note=""):
    RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: expected={expected}, actual={actual} {note}")


def npz_files(d):
    return sorted(glob.glob(os.path.join(REPO_ROOT, d, "maze_*.npz")))


def test1_candidate_determinism():
    print("\n=== テスト1: 候補の決定性（同一 seed → 同一迷路） ===")
    ok_all = True
    tried = 0
    for seed in range(1000, 1200):
        a = generate_candidate(seed)
        if a is None:
            continue
        b = generate_candidate(seed)
        same = bool(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
        ok_all = ok_all and same and a[2] == b[2]
        tried += 1
        if tried >= 8:
            break
    record("候補 8 件で壁配列と info が完全一致", True, ok_all, ok_all, f"（{tried} 件で確認）")
    return ok_all


def test2_selection_determinism():
    print("\n=== テスト2: **選抜の決定性**（同じ seed 範囲・規則 → 同じ 20 面） ===")
    ok_all = True
    for d, band, (lo, hi) in BANDS:
        mf = os.path.join(REPO_ROOT, d, "manifest.json")
        if not os.path.exists(mf):
            record(f"{band} の manifest", "存在", "なし", False)
            ok_all = False
            continue
        man = json.load(open(mf, encoding="utf-8"))
        chosen, _log = select_stratified(man["seed_from"], man["seed_to"],
                                          man["n_per_band"], tuple(man["r_band_edges"]),
                                          man["extra_open_target"], tuple(man["d_window"]),
                                          progress_every=0)
        got = sorted(x[2]["seed"] for b in chosen for x in b)
        want = sorted(m["seed"] for m in man["mazes"])
        ok = got == want
        ok_all = ok_all and ok
        record(f"{band} の選抜が manifest と一致", f"{len(want)} 面", f"{len(got)} 面", ok,
               "" if ok else f"差: {set(got) ^ set(want)}")
    return ok_all


def test3_frozen_npz_matches():
    print("\n=== テスト3: 凍結 npz と再生成の完全一致 ===")
    ok_all = True
    for d, band, _ in BANDS:
        files = npz_files(d)
        n_ok = 0
        for f in files:
            z = np.load(f)
            out = generate_candidate(int(z["seed"]))
            if out is None:
                continue
            if np.array_equal(out[0], z["v_walls"]) and np.array_equal(out[1], z["h_walls"]):
                n_ok += 1
        ok = (n_ok == len(files)) and len(files) == 20
        ok_all = ok_all and ok
        record(f"{band} の凍結 npz が再生成と一致", "20/20", f"{n_ok}/{len(files)}", ok)
    return ok_all


def test4_acceptance_conditions():
    print("\n=== テスト4: 受け入れ条件（規定 6 項目 + D の窓） ===")
    checks = {k: 0 for k in ("gateway1", "wallfollow", "connected", "start3", "goal_inner0",
                             "center_post0", "isolated0", "outer", "cycles>0", "d_window")}
    total = 0
    for d, _band, _ in BANDS:
        for f in npz_files(d):
            z = np.load(f)
            v, h = z["v_walls"], z["h_walls"]
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
            checks["d_window"] += bool(D_WINDOW[0] <= true_shortest(v, h) <= D_WINDOW[1])
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
        "d_window": f"**最終** D_true が窓 {list(D_WINDOW)} 内（v3 で直した点）",
    }
    ok_all = True
    for k, lab in labels.items():
        ok = checks[k] == total
        ok_all = ok_all and ok
        record(lab, f"{total}/{total}", f"{checks[k]}/{total}", ok)
    return ok_all


def test5_band_separation_and_strata():
    print("\n=== テスト5: seed 帯の分離と、層が埋まっていること ===")
    walls = {}
    ok_all = True
    for d, band, (lo, hi) in BANDS:
        seeds, per_band = [], {}
        for f in npz_files(d):
            z = np.load(f)
            s = int(z["seed"])
            seeds.append(s)
            walls.setdefault(band, set()).add(z["v_walls"].tobytes() + z["h_walls"].tobytes())
            b = band_of(detour_ratio(z["v_walls"], z["h_walls"]), R_BAND_EDGES)
            per_band[b] = per_band.get(b, 0) + 1
        in_range = all(lo <= s <= hi for s in seeds)
        record(f"{band} の seed が {lo}-{hi} の範囲内", True, in_range, in_range,
               f"実際の範囲 {min(seeds)}-{max(seeds)}")
        n_bands = len(R_BAND_EDGES) - 1
        filled = all(per_band.get(i, 0) == 5 for i in range(n_bands))
        record(f"{band} の各層がちょうど 5 面", {i: 5 for i in range(n_bands)},
               {i: per_band.get(i, 0) for i in range(n_bands)}, filled)
        ok_all = ok_all and in_range and filled
    overlap = walls["eval_v3"] & walls["validation_v3"]
    record("eval_v3 と validation_v3 の迷路が重複しない", 0, len(overlap), len(overlap) == 0)
    return ok_all and not overlap


def test6_manifest_has_candidate_log():
    print("\n=== テスト6: manifest に候補全数の採否が残っている ===")
    ok_all = True
    for d, band, _ in BANDS:
        man = json.load(open(os.path.join(REPO_ROOT, d, "manifest.json"), encoding="utf-8"))
        log = man.get("candidate_log", [])
        n_acc = sum(1 for r in log if r.get("accepted"))
        ok = len(log) > 20 and n_acc == 20 and all("reason" in r for r in log)
        ok_all = ok_all and ok
        record(f"{band} の候補記録", "候補>20・採用20・理由つき",
               f"候補{len(log)}・採用{n_acc}", ok)
    return ok_all


def main():
    for fn in (test1_candidate_determinism, test2_selection_determinism,
               test3_frozen_npz_matches, test4_acceptance_conditions,
               test5_band_separation_and_strata, test6_manifest_has_candidate_log):
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
