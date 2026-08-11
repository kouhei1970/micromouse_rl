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
  5. **候補プールの分離**: eval と validation の seed 範囲が重ならず、迷路も重複しない
  6. manifest に候補全数の採否が残っている（なぜこの面が選ばれたかを追える）
  7. **K の階級ごとの面数**が manifest の目標どおり（v4 で足した軸）

**⚠️ 数値をこのファイルに書かないこと（2026-08-12 是正）**

当初は seed 範囲を `(1000, 1999)` / `(4000, 4999)` とハードコードしていたが、
**帯を v4 へ入れ替えた 2026-08-12 に 4 件 FAIL した。**現行の候補プールは
eval [1000, 20999] / validation [21000, 40999] で、**採用 seed は連番ではない**
（eval 1018〜14037、validation 21003〜25842）。

**したがって受理条件・seed 範囲・K の目標はすべて `manifest.json` から読む。**
テスト側に値を持つと、帯を入れ替えるたびにテストが正本と食い違う。

**⚠️ テスト2 は遅い（合計 7 分程度）。**候補プール全体（13038 + 4843 seed）を
走査し直すため。**これは §9-2 が要求する「棄却再試行を含めて決定的」の保証**なので
短縮しない。急ぐときは `SKIP_SLOW=1` で飛ばせるが、**飛ばしたことは合格数に現れる**。

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
from competition.explore_cost import GOAL_CELLS, detour_ratio, true_shortest  # noqa: E402
from competition.maze_gen_v2 import all_cells_reachable  # noqa: E402
from competition.maze_gen_v3 import (  # noqa: E402
    D_WINDOW, R_BAND_EDGES, band_of, generate_candidate, select_stratified)

# 現行帯は v4（2026-08-12 凍結。教授裁定 R3）。
# **seed 範囲はここに書かない** — manifest から読む（上の注意書きを参照）。
BANDS = [("competition/mazes/eval", "eval(v4)"),
         ("competition/mazes/validation", "validation(v4)")]
SKIP_SLOW = os.environ.get("SKIP_SLOW", "") not in ("", "0")
RESULTS = []


def manifest(d):
    """帯の受理条件を manifest から読む（テスト側に数値を持たない）。"""
    return json.load(open(os.path.join(REPO_ROOT, d, "manifest.json"), encoding="utf-8"))


def k_targets_of(man):
    """manifest の k_targets は JSON なので鍵が文字列。int へ直す。"""
    kt = man.get("k_targets")
    return {int(k): int(v) for k, v in kt.items()} if kt else None


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
    if SKIP_SLOW:
        record("選抜の決定性", "実行", "SKIP_SLOW=1 で飛ばした", False,
               "← §9-2 の保証が未確認のまま")
        return False
    ok_all = True
    for d, band in BANDS:
        mf = os.path.join(REPO_ROOT, d, "manifest.json")
        if not os.path.exists(mf):
            record(f"{band} の manifest", "存在", "なし", False)
            ok_all = False
            continue
        man = manifest(d)
        # **k_targets を渡すのを忘れないこと。**渡さないと K の階級を無視した
        # 別の 20 面が返り、manifest と食い違う（2026-08-12 に実際に FAIL した）。
        print(f"  {band}: seed {man['seed_from']}〜{man['seed_to']} を走査中"
              f"（数分かかる）…", flush=True)
        chosen, _log = select_stratified(man["seed_from"], man["seed_to"],
                                          man["n_per_band"], tuple(man["r_band_edges"]),
                                          man["extra_open_target"], tuple(man["d_window"]),
                                          progress_every=0, k_targets=k_targets_of(man))
        got = sorted(x[2]["seed"] for b in chosen for x in b)
        want = sorted(m["seed"] for m in man["mazes"])
        ok = got == want
        ok_all = ok_all and ok
        record(f"{band} の選抜が manifest と一致", f"{len(want)} 面", f"{len(got)} 面", ok,
               "" if ok else f"差: {sorted(set(got) ^ set(want))}")
    return ok_all


def test3_frozen_npz_matches():
    print("\n=== テスト3: 凍結 npz と再生成の完全一致 ===")
    ok_all = True
    for d, band in BANDS:
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
    for d, _band in BANDS:
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
    print("\n=== テスト5: 候補プールの分離と、層が埋まっていること ===")
    walls, pools = {}, {}
    ok_all = True
    for d, band in BANDS:
        man = manifest(d)
        lo, hi = man["seed_from"], man["seed_to"]
        pools[band] = (lo, hi)
        seeds, per_band = [], {}
        for f in npz_files(d):
            z = np.load(f)
            s = int(z["seed"])
            seeds.append(s)
            walls.setdefault(band, set()).add(z["v_walls"].tobytes() + z["h_walls"].tobytes())
            b = band_of(detour_ratio(z["v_walls"], z["h_walls"]), R_BAND_EDGES)
            per_band[b] = per_band.get(b, 0) + 1
        # 採用 seed は連番ではない。**候補プールの中に収まっていること**だけを見る。
        in_range = all(lo <= s <= hi for s in seeds)
        record(f"{band} の採用 seed が候補プール {lo}-{hi} の内側", True, in_range, in_range,
               f"実際の範囲 {min(seeds)}-{max(seeds)}（連番ではない）")
        n_bands = len(R_BAND_EDGES) - 1
        filled = all(per_band.get(i, 0) == 5 for i in range(n_bands))
        record(f"{band} の各層がちょうど 5 面", {i: 5 for i in range(n_bands)},
               {i: per_band.get(i, 0) for i in range(n_bands)}, filled)
        ok_all = ok_all and in_range and filled
    # **候補プールどうしが重ならないこと**（同じ迷路が両帯に出るのを構造的に防ぐ）
    (alo, ahi), (blo, bhi) = pools[BANDS[0][1]], pools[BANDS[1][1]]
    disjoint = (ahi < blo) or (bhi < alo)
    record("eval と validation の候補プールが重ならない", True, disjoint, disjoint,
           f"[{alo}, {ahi}] と [{blo}, {bhi}]")
    overlap = walls[BANDS[0][1]] & walls[BANDS[1][1]]
    record("eval と validation の迷路が重複しない", 0, len(overlap), len(overlap) == 0)
    return ok_all and disjoint and not overlap


def test7_k_class_targets():
    """v4 で足した軸: 候補経路の本数 K の階級ごとの面数が manifest の目標どおりか。"""
    print("\n=== テスト7: K の階級ごとの面数（v4 で足した軸） ===")
    from competition.macro_routes import macro_routes  # noqa: PLC0415
    ok_all = True
    for d, band in BANDS:
        man = manifest(d)
        want = k_targets_of(man)
        if not want:
            record(f"{band} の k_targets", "manifest に存在", "なし", False)
            ok_all = False
            continue
        got = {1: 0, 2: 0, 3: 0}
        for f in npz_files(d):
            z = np.load(f)
            start = (int(z["start_x"]), int(z["start_y"])) if "start_x" in z.files else (0, 0)
            goals = tuple((int(a), int(b)) for a, b in zip(z["goals_x"], z["goals_y"])) \
                if "goals_x" in z.files else GOAL_CELLS
            k = len(macro_routes(z["v_walls"], z["h_walls"], start, goals,
                                 delta=man["macro_delta"], theta=man["macro_theta"]))
            got[min(k, 3)] += 1
        ok = got == want
        ok_all = ok_all and ok
        record(f"{band} の K 階級ごとの面数", want, got, ok)
    return ok_all


def test6_manifest_has_candidate_log():
    print("\n=== テスト6: manifest に候補全数の採否が残っている ===")
    ok_all = True
    for d, band in BANDS:
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
               test5_band_separation_and_strata, test6_manifest_has_candidate_log,
               test7_k_class_targets):
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
