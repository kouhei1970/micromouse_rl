#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_007 の評価結果を集計する（3 方式 × 3 腕、および是正前との新旧比較）。

`competition/results/exp007/<腕>/<方策>_<時刻>/summary.json` を読み、
研究計画書 §2 の 5 指標 (a)〜(e) と、経路長あたりの所要時間を表にする。

**(e) の読み方（評価器の注記より）**: 未定義面（初回の最短走行が成立しなかった面）を
除いた値なので、`n_undefined` と (c) 有効最短走行率を必ず併記して読むこと。
(e)=1.00 は「探索が優秀」な場合と「走り直しても速くならない迷路だった」場合の
両方で生じる。

使い方:
    .venv/bin/python experiments/exp_007_maze_reeval/aggregate.py
"""
import glob
import json
import os
import sys
from collections import OrderedDict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "competition", "reference_mazes"))

RES = os.path.join(REPO, "competition", "results", "exp007")
ARMS = OrderedDict([
    ("eval", "腕1 生成eval (1000-1019)"),
    ("validation", "腕2 生成validation (4000-4019)"),
    ("contest_reference", "腕3 大会実迷路 (中央ゴール∩窓内)"),
    ("eval_v2_short", "（対照）是正前 eval"),
])
POLICIES = OrderedDict([
    ("adachi_classical", "L0-a 超信地旋回走行（区画ごと停止）"),
    ("l0b_straightrun", "L0-b 超信地旋回走行（直進連続）"),
    ("l0c_slalom", "L0-c スラローム走行"),
])
MAZE_DIRS = {
    "eval": "competition/mazes/eval",
    "validation": "competition/mazes/validation",
    "contest_reference": "competition/mazes/contest_reference",
    "eval_v2_short": "competition/mazes/eval_v2_short",
}


def latest_summary(arm, policy):
    """同じ腕・方策で複数回走っている場合は最新のものを採る。"""
    cands = sorted(glob.glob(os.path.join(RES, arm, f"{policy}_*", "summary.json")))
    return cands[-1] if cands else None


def maze_d_true():
    """各迷路の真の最短距離 D_true を測る（(腕, maze_id) → 区画数）。

    **腕でキーを分けるのが必須**: eval と eval_v2_short はどちらも maze_1000〜1019
    という同じファイル名を使う（是正前後で中身が違う）。maze_id だけでキーにすると
    後から読んだ方が前を上書きし、s/区画 が桁で狂う。
    """
    from compare_generated_vs_contest import load_generated, load_contest, d_true
    out = {}
    for arm, d in MAZE_DIRS.items():
        for f in sorted(glob.glob(os.path.join(REPO, d, "*.npz"))):
            stem = os.path.basename(f)[:-4]
            z = np.load(f)
            loader = load_contest if "goals_x" in z else load_generated
            v, h, s, g, _ = loader(f)
            out[(arm, stem)] = d_true(v, h, s, g)
    return out


def fmt(x, n=2, unit=""):
    return "—" if x is None else f"{x:.{n}f}{unit}"


def main():
    D = maze_d_true()
    rows, detail = [], {}
    for arm in ARMS:
        for pol in POLICIES:
            p = latest_summary(arm, pol)
            if not p:
                continue
            j = json.load(open(p, encoding="utf-8"))
            k = j["kpi"]
            bt = j.get("best_times", {})
            # 経路長あたりの所要時間（面ごとに best_time / D_true を出して中央値）
            per_cell = [bt[m] / D[(arm, m)] for m in bt
                        if bt[m] is not None and D.get((arm, m))]
            # 走行回数（持ち時間内に何回走れたか）を面ごとの JSON から拾う
            nruns = []
            for mf in sorted(glob.glob(os.path.join(os.path.dirname(p), "maze_*.json"))
                             + glob.glob(os.path.join(os.path.dirname(p), "contest_*.json"))):
                nruns.append(len(json.load(open(mf, encoding="utf-8"))["runs"]))
            rows.append(dict(
                arm=arm, policy=pol, n=j["n_mazes"],
                a=k["a_goal_reached"]["rate"], b=k["b_fast_run_done"]["rate"],
                c=k["c_fast_run_effective"]["rate"],
                d_med=k["d_best_time"]["median"], d_min=k["d_best_time"]["min"],
                d_max=k["d_best_time"]["max"],
                e_med=k["e_first_fast_efficiency"]["median"],
                e_undef=k["e_first_fast_efficiency"]["n_undefined"],
                explore=k["explore_time"]["median"],
                per_cell=float(np.median(per_cell)) if per_cell else None,
                nruns_med=float(np.median(nruns)) if nruns else None,
                nruns_max=max(nruns) if nruns else None,
                d_true_med=(float(np.median([D[(arm, m)] for m in bt if D.get((arm, m))]))
                            if bt else None),
                dir=os.path.dirname(p),
            ))
            detail[f"{arm}/{pol}"] = {m: dict(best_time=bt[m], d_true=D.get((arm, m)))
                                      for m in bt}

    if not rows:
        print(f"結果が見つかりません: {RES}")
        return 1

    print("=" * 132)
    print("exp_007: 是正後の迷路での L0-a/b/c 再評価（研究計画書 §2 の 5 指標）")
    print("=" * 132)
    hdr = (f"{'腕':<34}{'方式':<32}{'n':>3} {'(a)':>6} {'(b)':>6} {'(c)':>6} "
           f"{'(d)中央値':>10} {'(d)範囲':>17} {'(e)':>6} {'未定義':>6} "
           f"{'探索中央':>9} {'s/区画':>8} {'走行回数':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{ARMS[r['arm']]:<34}{POLICIES[r['policy']]:<32}{r['n']:>3} "
              f"{fmt(r['a'] and r['a'] * 100, 0, '%'):>6} "
              f"{fmt(r['b'] and r['b'] * 100, 0, '%'):>6} "
              f"{fmt(r['c'] and r['c'] * 100, 0, '%'):>6} "
              f"{fmt(r['d_med'], 2, ' s'):>10} "
              f"{fmt(r['d_min'], 1) + '〜' + fmt(r['d_max'], 1):>17} "
              f"{fmt(r['e_med'], 2):>6} {r['e_undef']:>6} "
              f"{fmt(r['explore'], 1, ' s'):>9} "
              f"{fmt(r['per_cell'], 3):>8} "
              f"{fmt(r['nruns_med'], 0) + '/' + str(r['nruns_max']):>9}")

    print("\n【新旧比較】経路長が伸びたときのタイムの変化（同一方策・同一コード）")
    print(f"{'方式':<32}{'是正前 (d)':>12}{'是正後eval (d)':>16}{'倍率':>8}"
          f"{'是正前 D':>10}{'是正後 D':>10}{'D倍率':>8}{'s/区画 前→後':>18}")
    for pol in POLICIES:
        o = next((r for r in rows if r["arm"] == "eval_v2_short" and r["policy"] == pol), None)
        n_ = next((r for r in rows if r["arm"] == "eval" and r["policy"] == pol), None)
        if not (o and n_):
            print(f"{POLICIES[pol]:<32}{'（対照が未実施）' if not o else '（新が未実施）'}")
            continue
        print(f"{POLICIES[pol]:<32}{fmt(o['d_med'], 2, ' s'):>12}{fmt(n_['d_med'], 2, ' s'):>16}"
              f"{fmt(n_['d_med'] / o['d_med'], 2, '倍'):>8}"
              f"{fmt(o['d_true_med'], 0):>10}{fmt(n_['d_true_med'], 0):>10}"
              f"{fmt(n_['d_true_med'] / o['d_true_med'], 2, '倍'):>8}"
              f"{fmt(o['per_cell'], 3) + ' → ' + fmt(n_['per_cell'], 3):>18}")

    print("\n【腕1 vs 腕3】評価帯は大会実迷路と同じように振る舞うか")
    print("  （両腕は D_true 中央値がほぼ同じなので、差が出れば経路長以外の構造に帰属する）")
    print(f"{'方式':<32}{'腕1 (d)':>10}{'腕3 (d)':>10}{'比':>8}"
          f"{'腕1 探索':>10}{'腕3 探索':>10}{'比':>8}{'腕1 D':>8}{'腕3 D':>8}")
    for pol in POLICIES:
        a1 = next((r for r in rows if r["arm"] == "eval" and r["policy"] == pol), None)
        a3 = next((r for r in rows if r["arm"] == "contest_reference" and r["policy"] == pol), None)
        if not (a1 and a3):
            continue
        rd = (a3["d_med"] / a1["d_med"]) if (a1["d_med"] and a3["d_med"]) else None
        re_ = (a3["explore"] / a1["explore"]) if (a1["explore"] and a3["explore"]) else None
        print(f"{POLICIES[pol]:<32}{fmt(a1['d_med'], 2):>10}{fmt(a3['d_med'], 2):>10}"
              f"{fmt(rd, 2, '倍'):>8}{fmt(a1['explore'], 1):>10}{fmt(a3['explore'], 1):>10}"
              f"{fmt(re_, 2, '倍'):>8}{fmt(a1['d_true_med'], 0):>8}{fmt(a3['d_true_med'], 0):>8}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aggregate.json")
    json.dump(dict(rows=rows, per_maze=detail), open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n数値 JSON: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
