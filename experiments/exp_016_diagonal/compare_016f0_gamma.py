#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""016-F0 の (γ) — **設計帯での走行タイムを、面ごとに対応をとって比べる**（G5 の判定）。

カード `card_016f0.md` §5-1 (γ)。**退行確認の代理測定であって参照線の更新ではない**
（参照線 15.06 s と M5 の凍結表は動かさない。教授裁定 2026-08-14）。

§9-15（裁定 R17）に従い、**要約統計量どうしを比べない**。
**面ごとに対応をとった差の分布**で報告し、**二値指標は McNemar** で見る。

使い方:
    .venv/bin/python experiments/exp_016_diagonal/compare_016f0_gamma.py \
        --before outputs/exp_013_band_v4_reeval/016f0_base_l0c_e1t_tr_design/runs_detail.json \
        --after  outputs/exp_013_band_v4_reeval/016f0_f0_l0c_e1t_tr_design/runs_detail.json
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path):
    """面 → {best_time, n_goal, n_runs, e_prime(最良走行), 全走行} を返す。

    (e') は裁定 R14 に従い **分子 = 節点数 = 移動回数 + 1** で割る。
    """
    d = json.load(open(path, encoding="utf-8"))
    by_face = defaultdict(list)
    for r in d["runs"]:
        by_face[r["maze"]].append(r)
    out = {}
    for face, runs in by_face.items():
        goals = [q for q in runs if q["outcome"] == "goal" and q["run_time"]]
        best = min(goals, key=lambda q: q["run_time"]) if goals else None
        out[face] = dict(
            n_runs=len(runs), n_goal=len(goals),
            best_time=(best["run_time"] if best else None),
            # (e') = 計時窓内で経由した節点数 ÷ 真の最短距離（R14: n_cells は移動回数）
            e_prime=((best["n_cells"] + 1) / best["d_true"]) if best else None,
            d_true=runs[0]["d_true"],
            reached=bool(goals))
    return d, out


def mcnemar_exact(b, c):
    """対応のある二値指標の正確検定（両側）。b, c は不一致ペアの数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    db, before = load(args.before)
    da, after = load(args.after)
    faces = sorted(set(before) & set(after), key=lambda s: int(s.split("_")[1]))
    only = (set(before) ^ set(after))
    if only:
        print(f"⚠️ 片方にしか無い面: {sorted(only)}（対応が取れないので除外）")

    print(f"対照 = {db['policy']}")
    print(f"処理 = {da['policy']}")
    print(f"迷路 = {db['maze_dir']}／対応の取れた面 {len(faces)}\n")

    rows, dt, de = [], [], []
    for f in faces:
        b, a = before[f], after[f]
        d_time = (a["best_time"] - b["best_time"]) if (a["best_time"] and b["best_time"]) else None
        d_ep = (a["e_prime"] - b["e_prime"]) if (a["e_prime"] and b["e_prime"]) else None
        if d_time is not None:
            dt.append(d_time)
        if d_ep is not None:
            de.append(d_ep)
        rows.append(dict(maze=f, d_true=b["d_true"],
                         before_best=b["best_time"], after_best=a["best_time"],
                         d_best=d_time, before_e=b["e_prime"], after_e=a["e_prime"],
                         d_e=d_ep, before_reached=b["reached"], after_reached=a["reached"],
                         before_runs=b["n_runs"], after_runs=a["n_runs"]))

    print(f"{'面':<12}{'D':>4}{'前 best':>10}{'後 best':>10}{'差':>9}{'比':>8}"
          f"{'前 (e\')':>9}{'後 (e\')':>9}")
    for r in rows:
        f = lambda v, w, p: (f"{v:>{w}.{p}f}" if v is not None else " " * (w - 1) + "-")  # noqa: E731
        ratio = (r["after_best"] / r["before_best"]) if (r["after_best"] and r["before_best"]) else None
        print(f"{r['maze']:<12}{r['d_true']:>4}{f(r['before_best'],10,2)}{f(r['after_best'],10,2)}"
              f"{f(r['d_best'],9,3)}{f(ratio,8,4)}{f(r['before_e'],9,3)}{f(r['after_e'],9,3)}")

    print("\n【面ごとに対応をとった差の分布】（§9-15。要約統計量どうしを比べない）")
    if dt:
        dt = np.array(dt)
        print(f"  最良走行タイムの差 [s]: n={len(dt)}  中央値 {np.median(dt):+.3f}  "
              f"四分位 {np.percentile(dt,25):+.3f}〜{np.percentile(dt,75):+.3f}  "
              f"範囲 {dt.min():+.3f}〜{dt.max():+.3f}")
        print(f"    悪化（遅くなった）{int((dt>1e-9).sum())} 面 / "
              f"改善 {int((dt<-1e-9).sum())} 面 / 不変 {int((np.abs(dt)<=1e-9).sum())} 面")
        rel = np.array([r["after_best"] / r["before_best"] - 1.0 for r in rows
                        if r["after_best"] and r["before_best"]])
        print(f"  相対の差: 中央値 {np.median(rel)*100:+.2f} %  最悪 {rel.max()*100:+.2f} %")
    if de:
        de = np.array(de)
        print(f"  (e') の差: n={len(de)}  中央値 {np.median(de):+.4f}  "
              f"範囲 {de.min():+.4f}〜{de.max():+.4f}")

    b = sum(1 for r in rows if r["before_reached"] and not r["after_reached"])
    c = sum(1 for r in rows if not r["before_reached"] and r["after_reached"])
    print(f"\n  ゴール到達（二値・対応あり）: 前のみ成立 {b} 面 / 後のみ成立 {c} 面 "
          f"→ McNemar 正確検定 p = {mcnemar_exact(b, c):.4f}")
    print("  ⚠️ 幅で語らないこと。言えるのは機構の観測までである（note_016）")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(before=args.before, after=args.after, rows=rows,
                       n_faces=len(faces)),
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
