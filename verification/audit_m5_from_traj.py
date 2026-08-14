#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
課題 J の第 2 層: 軌跡から (d) 最速タイムを自前で再計算し、確定基準表と照合する
Task J, layer 2: recompute (d) from trajectories and compare against the reference table

准教授セッション（8 代目）・2026-08-14
**判定形はデータが出る前に確定させてある**（本ファイルは再実行の完了前にコミットする）。

## なぜこれが要るか / why

`verification/audit_m5_baseline_table.py`（第 1 層）が確かめたのは
「教授の抽出が生データの集計値と合っているか」であって、
**集計値そのものが正しいか**ではない。旧基準表には
`verification/REPORT_003` r2 による軌跡からの再計算があるが、
新表の測定 M-1 のランナー（`experiments/exp_016_diagonal/run_016cal.py`）は
**軌跡を保存しない**ため、同じ検査が現存物には当てられない。

そこで `verification/rerun_m1_with_traj.py` で軌跡を記録しながら M-1 を再実行し、
本スクリプトが**評価器の集計を一切使わずに**軌跡だけから (d) を出して照合する。

## 独立性の限界（報告に必ず併記する）/ limits of independence

物理シミュレーション本体（MuJoCo・`mouse/sim.py`・方策 `SlalomPolicy`）は共有している。
**独立なのは KPI（最速タイム）の計算部分だけ**であり、これは `REPORT_003` r2 と同じ水準・
同じ限界である。シミュレータや方策に誤りがあれば、両者は同じように誤る。

## 事前に決めた判定 / pre-registered verdicts

  J-1  再実行の集約値が M-1 の記録と 20/20 で一致するか（再現性。許容 0 = 完全一致）
  J-2  軌跡から自前で計算した (d) が M-1 の fast_time と 20/20 で一致するか（許容 1e-9 秒）
  J-3  軌跡から自前で計算した (d) が研究計画書 §5 の表 20 値と一致するか（許容 5e-3 秒 = 丸め幅の半分）
  J-4  中央値が 14.690 s になるか（許容 5e-3 秒）
  J-5  経路が旧測定と同一か（安全率は速度計画のみを変えるはず。区画列の完全一致で見る）
  J-6  時間の分解: 安全率で伸縮する時間の割合。上界 −3.39%（sqrt 則）を超えないか

**J-1〜J-4 のいずれかが外れたら、確定基準表は「独立に確認済み」と書けない。**
**J-5 が外れたら、−2.66% を安全率だけに帰属できない**（経路も変わっている）。
**J-6 は機構の検査であって合否ではない**（外れても表の正しさは揺るがない。解釈が変わる）。
"""

import glob
import json
import math
import os
import statistics
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RERUN_DIR = os.path.join(REPO, "outputs/verification/m1_rerun_traj")
M1_JSON = os.path.join(REPO, "outputs/exp_016_diagonal/016cal_switch/m1_plain_l0c_sf0.75.json")
OLD_TRAJ_DIR = os.path.join(REPO, "outputs/exp_013_band_v4_reeval/l0c/traj")
OLD_JSON = os.path.join(REPO, "outputs/exp_013_band_v4_reeval/l0c/runs_detail.json")

CELL = 0.18  # 区画の一辺 [m]（docs/RESEARCH_PLAN.md §2）

# docs/RESEARCH_PLAN.md:164-170 の確定基準表（現行）
TABLE_NEW = {
    "maze_1018": 18.58, "maze_1019": 18.78, "maze_1021": 11.54, "maze_1023": 18.18,
    "maze_1024": 10.97, "maze_1041": 15.14, "maze_1046": 9.65, "maze_1073": 19.82,
    "maze_1103": 16.06, "maze_1126": 11.33, "maze_1138": 12.68, "maze_1161": 11.14,
    "maze_1240": 15.17, "maze_14037": 17.34, "maze_1804": 14.20, "maze_2304": 14.24,
    "maze_3707": 13.45, "maze_5188": 20.72, "maze_7837": 21.68, "maze_8209": 11.20,
}
TABLE_MEDIAN = 14.690

SF_OLD, SF_NEW = 0.70, 0.75
# 安全率は v_cap（線形）・a_max（線形）・a_lat（線形）の 3 つを同時にスケールする
# （competition/baseline_slalom.py:660-662）。
#   速度上限に張りつく区間: 時間 ∝ 1/v_cap        → 変化 = SF_OLD/SF_NEW − 1 = −6.67%
#   加速度・横加速度で決まる区間: 速度 ∝ sqrt(sf) → 変化 = sqrt(SF_OLD/SF_NEW) − 1 = −3.39%
# したがって**全区間が安全率で伸縮するなら改善は最低でも −3.39%**になる。
BOUND_SQRT = (math.sqrt(SF_OLD / SF_NEW) - 1.0) * 100.0
BOUND_LIN = (SF_OLD / SF_NEW - 1.0) * 100.0

results = []


def record(tag, item, ok, detail):
    results.append((tag, item, ok, detail))
    mark = "PASS" if ok is True else ("FAIL" if ok is False else "INFO")
    print(f"  [{mark}] {item}: {detail}")


def fast_time_from_traj(npz):
    """軌跡 npz から (d) 最速タイムを自前で計算する。

    評価器の集計値は一切見ない。走行の区切りと結末ラベルだけを使う。
    §2 の条文「(d) 最速タイム = 完走走行の最速値」と、
    実装 `competition/evaluator.py`:378-387 の「初回ゴール到達より後に開始した走行の最速値」を
    **両方**計算して返す（両者は同じとは限らない。条文と実装が食い違っているため）。
    """
    ts = np.asarray(npz["run_t_start"], dtype=float)
    te = np.asarray(npz["run_t_end"], dtype=float)
    oc = [str(s) for s in npz["run_outcome"]]
    rt = te - ts
    goal = [i for i in range(len(oc)) if oc[i] == "goal"]
    if not goal:
        return None, None, []
    by_text = float(min(rt[i] for i in goal))          # 条文どおり
    first = min(goal, key=lambda i: te[i])
    later = [i for i in goal if ts[i] > te[first]]
    by_impl = float(min(rt[i] for i in later)) if later else None  # 実装どおり
    return by_text, by_impl, [i + 1 for i in goal]


def cell_seq_from_traj(npz):
    """軌跡の (x, y) から通過区画の列を復元する（重複する連続の区画は畳む）。

    経路が変わっていないかを見るための量。安全率は速度計画だけを変えるはずなので、
    区画列は完全に一致していなければならない。
    """
    x = np.asarray(npz["x"], dtype=float)
    y = np.asarray(npz["y"], dtype=float)
    cx = np.floor(x / CELL).astype(int)
    cy = np.floor(y / CELL).astype(int)
    seq, prev = [], None
    for a, b in zip(cx, cy):
        cur = (int(a), int(b))
        if cur != prev:
            seq.append(cur)
            prev = cur
    return seq


def main():
    if not os.path.isdir(os.path.join(RERUN_DIR, "traj")):
        print(f"再実行の軌跡がまだ無い: {RERUN_DIR}/traj")
        print("先に verification/rerun_m1_with_traj.py を走らせること。")
        return 2

    with open(M1_JSON) as f:
        m1 = json.load(f)
    m1_ft = {r["maze"]: r["fast_time"] for r in m1["rows"]}

    paths = sorted(glob.glob(os.path.join(RERUN_DIR, "traj", "*.npz")))
    print(f"\n再実行の軌跡: {len(paths)} 本")

    # ---------------------------------------------------------------
    # J-1 再実行の集約値 対 M-1 の記録（再現性）
    # ---------------------------------------------------------------
    print("\n=== J-1 再実行の再現性（集約値どうし） ===")
    rr_json = os.path.join(RERUN_DIR, "rerun_detail.json")
    if os.path.exists(rr_json):
        with open(rr_json) as f:
            rr = json.load(f)
        # 🔧 是正（2026-08-14・初回実行後）: 再実行側の値は `harness_kpi` の中にある。
        # 当初の抽出は `harness_fast_time` / `fast_time` を直に探しており空振りして
        # 「0/20 不一致」を出した。**不一致の原因は私の抽出コードであって測定ではない。**
        rows_rr = rr if isinstance(rr, list) else rr.get("rows", rr.get("mazes", []))
        rr_ft, rr_et = {}, {}
        for row in rows_rr:
            kpi = row.get("harness_kpi") or {}
            v = row.get("harness_fast_time", kpi.get("fast_time"))
            if v is not None:
                rr_ft[row["maze"]] = v
            e = kpi.get("explore_time")
            if e is not None:
                rr_et[row["maze"]] = e

        n = sum(1 for m in m1_ft if m in rr_ft and rr_ft[m] == m1_ft[m])
        record("J-1", "最速タイム: 再実行 対 M-1（bit 単位）", n == 20,
               f"{n}/20 が bit 単位で一致")
        near = sum(1 for m in m1_ft if m in rr_ft and abs(rr_ft[m] - m1_ft[m]) <= 1e-9)
        if near != n:
            record("J-1", "（参考）1e-9 以内", None, f"{near}/20")

        # 探索走行タイムも M-1 に記録がある。独立な第 2 の値として照合する。
        m1_et = {r["maze"]: r.get("explore_time") for r in m1["rows"]}
        ne = sum(1 for m in m1_et if m1_et[m] is not None
                 and m in rr_et and rr_et[m] == m1_et[m])
        record("J-1", "探索走行タイム: 再実行 対 M-1（bit 単位）", ne == 20,
               f"{ne}/20 が bit 単位で一致")
    else:
        record("J-1", "再実行の集約 json", False, f"{rr_json} が無い")

    # ---------------------------------------------------------------
    # J-1b 標本数の一致（**再実行の開始後に追加した検査**・2026-08-14）
    # ---------------------------------------------------------------
    # 🔧 当初この検査は「軌跡の標本数 = M-1 の n_ticks」を課していたが、**前提が誤り**だった。
    # M-1 の `n_ticks` は `int(ey.size)`（run_016cal.py:143）で、横位置誤差の標本数である。
    # その `ey` は `ProbedCalPolicy._do_drive_control` が **`self._path is not None` のときだけ**
    # 追記する（run_016cal.py:67-70）ので、**経路追従中の周期しか数えていない**。
    # 一方こちらの軌跡は `act()` の全呼び出しを記録している（その場旋回・帰路も含む）。
    # **数えている対象が違うので一致しなくて当然**であり、当初の 0/20 は測定の差ではなく
    # 私の前提の誤りだった。比較可能な形（軌跡の標本数 ≥ n_ticks）に置き換える。
    print("\n=== J-1b 標本数の整合（前提を是正した版） ===")
    m1_ticks = {r["maze"]: r.get("n_ticks") for r in m1["rows"]}
    n1b, chk1b, ratios = 0, 0, []
    for p in paths:
        maze = os.path.basename(p)[:-4]
        if m1_ticks.get(maze) is None:
            continue
        chk1b += 1
        n = int(np.load(p, allow_pickle=False)["t"].shape[0])
        n1b += (n >= m1_ticks[maze])
        ratios.append(m1_ticks[maze] / n)
    record("J-1b", "軌跡の標本数 ≥ M-1 の n_ticks（経路追従中の部分集合）",
           n1b == chk1b and chk1b == 20,
           f"{n1b}/{chk1b} で成立。経路追従が占める割合は "
           f"{min(ratios) * 100:.1f}〜{max(ratios) * 100:.1f} %"
           f"（中央値 {statistics.median(ratios) * 100:.1f} %）")

    # ---------------------------------------------------------------
    # J-2 / J-3 / J-4 軌跡からの自前計算
    # ---------------------------------------------------------------
    print("\n=== J-2/J-3 軌跡から自前で計算した (d) との照合 ===")
    print("  迷路          自前(条文)  自前(実装)   M-1 記録    §5 の表   判定")
    mine, n2, n3, mismatch_def = {}, 0, 0, 0
    for p in paths:
        maze = os.path.basename(p)[:-4]
        z = np.load(p, allow_pickle=False)
        by_text, by_impl, _ = fast_time_from_traj(z)
        if by_impl is None:
            record("J-2", maze, False, "初回ゴール後に開始した完走走行が無い")
            continue
        mine[maze] = by_impl
        mismatch_def += abs(by_text - by_impl) > 1e-9
        ok2 = maze in m1_ft and abs(by_impl - m1_ft[maze]) <= 1e-9
        ok3 = maze in TABLE_NEW and abs(by_impl - TABLE_NEW[maze]) <= 5e-3
        n2 += ok2
        n3 += ok3
        print(f"  {maze:<13}{by_text:>11.4f}{by_impl:>12.4f}"
              f"{m1_ft.get(maze, float('nan')):>11.4f}{TABLE_NEW.get(maze, float('nan')):>11.2f}"
              f"   {'一致' if (ok2 and ok3) else '🔴 不一致'}")

    record("J-2", "自前計算 対 M-1 の fast_time", n2 == 20, f"{n2}/20 一致（許容 1e-9 秒）")
    record("J-3", "自前計算 対 §5 の表 20 値", n3 == 20, f"{n3}/20 一致（許容 5e-3 秒）")
    record("J-2", "条文の定義 対 実装の定義", None,
           f"値が食い違う迷路 {mismatch_def}/20"
           "（条文=完走走行の最速値／実装=初回ゴール後に開始した走行の最速値）")

    if len(mine) == 20:
        med = statistics.median(mine.values())
        record("J-4", "中央値", abs(med - TABLE_MEDIAN) <= 5e-3,
               f"自前計算 {med:.4f} s / 文書記載 {TABLE_MEDIAN} s")

    # ---------------------------------------------------------------
    # J-5 経路の同一性（旧測定の軌跡と比べる）
    # ---------------------------------------------------------------
    print("\n=== J-5 経路の同一性（安全率は速度計画だけを変えるはず） ===")
    n5, checked = 0, 0
    for p in paths:
        maze = os.path.basename(p)[:-4]
        old_p = os.path.join(OLD_TRAJ_DIR, maze + ".npz")
        if not os.path.exists(old_p):
            continue
        checked += 1
        s_new = cell_seq_from_traj(np.load(p, allow_pickle=False))
        s_old = cell_seq_from_traj(np.load(old_p, allow_pickle=False))
        n5 += (s_new == s_old)
    record("J-5", "通過区画の列が旧測定と一致", n5 == checked and checked == 20,
           f"{n5}/{checked} の迷路で完全一致")

    # ---------------------------------------------------------------
    # J-6 時間の分解（機構の検査。合否ではない）
    # ---------------------------------------------------------------
    print("\n=== J-6 機構: 改善率と安全率の理論上界 ===")
    print(f"  理論上界: 全区間が加速度・横加速度で決まるなら {BOUND_SQRT:+.2f} %")
    print(f"            全区間が速度上限に張りつくなら       {BOUND_LIN:+.2f} %")
    if os.path.exists(OLD_JSON) and mine:
        with open(OLD_JSON) as f:
            old = json.load(f)
        bym = {}
        for r in old.get("runs", []):
            bym.setdefault(r["maze"], []).append(r)
        pct = {}
        for maze, v in mine.items():
            if maze in bym:
                g = [r["run_time"] for r in bym[maze] if r["outcome"] == "goal"]
                if g:
                    pct[maze] = (v - min(g)) / min(g) * 100.0
        if pct:
            worst = min(pct.values())
            record("J-6", "改善率が sqrt 則の上界を超えないか", worst >= BOUND_SQRT,
                   f"最大改善 {worst:+.2f} % 対 上界 {BOUND_SQRT:+.2f} %")
            med_pct = statistics.median(pct.values())
            frac = med_pct / BOUND_SQRT
            record("J-6", "安全率で伸縮する時間の割合（sqrt 則を仮定）", None,
                   f"中央値 {med_pct:+.2f} % / 上界 {BOUND_SQRT:+.2f} % = {frac * 100:.1f} %"
                   f" → 残り {100 - frac * 100:.1f} % は安全率で変わらない時間")

    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    n_pass = sum(1 for _, _, ok, _ in results if ok is True)
    n_fail = sum(1 for _, _, ok, _ in results if ok is False)
    print(f"総括: PASS {n_pass} / FAIL {n_fail} / INFO "
          f"{sum(1 for _, _, ok, _ in results if ok is None)}")
    if n_fail:
        print("\n  🔴 不合格:")
        for tag, item, ok, detail in results:
            if ok is False:
                print(f"    [{tag}] {item}: {detail}")
    print("""
⚠️ 独立性の限界: 物理シミュレーション・方策は共有している。
   独立なのは最速タイムの計算部分だけで、REPORT_003 r2 と同じ水準・同じ限界である。
""")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
