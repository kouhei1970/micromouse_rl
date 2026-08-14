#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""016-cal の集計と **C1〜C6 の判定**（カード `card_016cal.md` §4）。

**判定は条文の文言のみで行う。**外れの読みは §4-1／§7 に書く。

--------------------------------------------------------------------------
⚠️ C3 は 2 通り出す（カード §7-1）
--------------------------------------------------------------------------
    - **全段**（登録した条文どおり。**判定はこちら**）
    - **完走率 ≥ 90% の範囲に限ったもの**（参考）

🔴 **当初「低い完走率の段は生き残った迷路だけの中央値だから母集団が違う」と考えたが、
実測すると誤りだった**（カード §7-1 の訂正）。**全 60 迷路が最速タイムを持っている**
（衝突しても後の走行で最短走行が成立する）。**2 つの値が違う理由は母集団ではなく、
「ある速度を超えると、速くしようとするほど遅くなる」という実在の現象である。**

**判定は条文どおり「全段」で行い、参考値を併記する**（教授裁定 2026-08-14）。

使い方:
    .venv/bin/python experiments/exp_016_diagonal/aggregate_016cal.py
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

LOWER95_TARGET = 0.95      # REQ_001 §2-3 の判定規則
C2_BAND = (0.75, 0.90)     # C2 の予測帯
C3_BAND = (-1.3, -0.7)     # C3 の予測帯（−1.0 ± 0.3）
C5_TOP3_SHARE = 0.50       # C5: 最初に落ちる 3 迷路が全失敗の 50% 未満


def load(out_dir):
    rows = []
    for f in sorted(glob.glob(str(Path(out_dir) / "sf*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        rows.append(d)
    rows.sort(key=lambda d: d["summary"]["safety_factor"])
    return rows


def power_fit(sfs, times):
    m = [i for i, t in enumerate(times) if t and np.isfinite(t)]
    if len(m) < 2:
        return float("nan")
    x = np.log(np.array([sfs[i] for i in m]))
    y = np.log(np.array([times[i] for i in m]))
    return float(np.polyfit(x, y, 1)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "exp_016_diagonal" / "016cal"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = load(args.out_dir)
    if not data:
        print("結果が 1 つも無い")
        return 1
    S = [d["summary"] for d in data]
    sfs = [s["safety_factor"] for s in S]
    print(f"段数 {len(S)}／安全率 {sfs}")
    if len(S) < 8:
        print("⚠️ **8 段が揃っていない。判定は確定ではない**（中間の集計として読むこと）\n")

    # ---- 表 ----
    print(f"{'安全率':>7}{'完走':>9}{'完走率':>9}{'片側95%下限':>13}{'最速中央値':>12}"
          f"{'n(最速)':>9}{'e_y最大 中央値':>15}{'e_y最大 最悪':>13}{'e_y RMS 中央値':>15}")
    for s in S:
        ft = s["fast_time_median"]
        print(f"{s['safety_factor']:>7.2f}{s['n_completed']:>6}/{s['n']:<3}"
              f"{s['completion_rate']*100:>8.1f}%{s['completion_lower95']:>13.3f}"
              f"{(ft if ft else float('nan')):>12.2f}{s['n_fast']:>9}"
              f"{s['e_y_max_median_m']*1000:>15.2f}{s['e_y_max_max_m']*1000:>13.2f}"
              f"{s['e_y_rms_median_m']*1000:>15.2f}")

    # ---- C1: 完走率の単調非増加 ----
    rates = [s["completion_rate"] for s in S]
    n0 = S[0]["n"]
    ups = [(sfs[i], rates[i] - rates[i - 1]) for i in range(1, len(rates))
           if rates[i] > rates[i - 1] + 1e-12]
    big_ups = [u for u in ups if u[1] > 1.0 / n0 + 1e-12]   # 1 迷路ぶんを超える上昇
    c1 = not big_ups
    print(f"\n【C1】完走率の単調非増加: {'的中' if c1 else '外れ'}"
          f"（上昇した段 {len(ups)} 件・うち 1 迷路ぶん〔{1/n0:.3f}〕を超える上昇 {len(big_ups)} 件）")
    for sf, du in ups:
        print(f"    安全率 {sf:.2f} で +{du*100:.1f} 点"
              f"{'（分解能の内側）' if du <= 1.0/n0 + 1e-12 else ' ← **分解能を超える**'}")

    # ---- C2: 判定を満たす最大の安全率 ----
    ok = [s["safety_factor"] for s in S if s["completion_lower95"] >= LOWER95_TARGET]
    cal = max(ok) if ok else None
    if cal is None:
        c2, why = False, "**どの安全率も下限 0.95 を満たさない → F-2 の族**"
    elif C2_BAND[0] - 1e-9 <= cal <= C2_BAND[1] + 1e-9:
        c2, why = True, ""
    elif cal == 0.70:
        c2, why = False, "**= 0.70。現行が既に最適だった**（未校正だが偶然正しかった）"
    elif cal < 0.70:
        c2, why = False, "🔴 **< 0.70。現行が危険側だった** — これは重い"
    else:
        c2, why = False, "**> 0.90 → F-1: 物理限界の実測値そのものを疑う**"
    print(f"\n【C2】判定を満たす最大の安全率 = {cal}"
          f"（予測帯 {C2_BAND[0]}〜{C2_BAND[1]}）: {'的中' if c2 else '外れ'} {why}")

    # ---- C3: 最速タイムの冪（**2 通り出す**） ----
    ft_all = [s["fast_time_median"] for s in S]
    n_all = power_fit(sfs, ft_all)
    idx = [i for i, s in enumerate(S) if s["completion_rate"] >= 0.90]
    n_sub = power_fit([sfs[i] for i in idx], [ft_all[i] for i in idx])
    c3 = C3_BAND[0] <= n_all <= C3_BAND[1]
    print(f"\n【C3】最速タイムの冪（**判定は条文どおり全段**）: n = {n_all:+.3f}"
          f"（予測帯 {C3_BAND[0]}〜{C3_BAND[1]}）: {'的中' if c3 else '外れ'}")
    print(f"    参考（完走率 ≥90% の {len(idx)} 段に限る）: n = {n_sub:+.3f}")
    ns = [s["n_fast"] for s in S]
    print(f"    最速タイムを持つ迷路の数（各段）: {ns}"
          + ("  ← **全段で同じなら母集団は減っていない**" if len(set(ns)) == 1 else
             "  ← ⚠️ **段によって母集団が違う**"))

    # ---- C4: 横偏差の単調増加 ----
    ey = [s["e_y_max_median_m"] for s in S]
    downs = [(sfs[i], ey[i] - ey[i - 1]) for i in range(1, len(ey)) if ey[i] < ey[i - 1] - 1e-9]
    c4 = not downs
    print(f"\n【C4】横偏差 最大の中央値の単調増加: {'的中' if c4 else '外れ'}"
          f"（下降した段 {len(downs)} 件）"
          + ("" if c4 else " → **F-3: 追従制御が非線形。曲率不連続 = 016-G が疑わしい**"))
    for sf, d in downs:
        print(f"    安全率 {sf:.2f} で {d*1000:+.2f} mm")

    # ---- C5: 失敗の集中 ----
    from collections import Counter
    cnt = Counter(f for s in S for f in s["failed_faces"])
    tot = sum(cnt.values())
    top3 = sum(v for _, v in cnt.most_common(3))
    share = (top3 / tot) if tot else 0.0
    c5 = (tot == 0) or (share < C5_TOP3_SHARE)
    print(f"\n【C5】失敗の集中: 全失敗 {tot} 件・上位 3 迷路が {top3} 件（{share*100:.1f}%）"
          f": {'的中' if c5 else '外れ'}"
          + ("" if c5 else " → **F-4: 完走率 1 つで扱うのは不適切。迷路の性質で層別する**"))
    if cnt:
        print(f"    最も落ちた迷路: {cnt.most_common(5)}")

    # ---- C6: 0.60 で完走率 100% ----
    s60 = next((s for s in S if abs(s["safety_factor"] - 0.60) < 1e-9), None)
    c6 = bool(s60 and s60["completion_rate"] >= 1.0 - 1e-12)
    print(f"\n【C6】安全率 0.60 の完走率 = "
          f"{(s60['completion_rate']*100 if s60 else float('nan')):.1f}%"
          f": {'的中' if c6 else '外れ'}"
          + ("" if c6 else " → **F-2: 安全率以外に律速がある。校正では解けない**"))

    verd = dict(C1=c1, C2=c2, C3=c3, C4=c4, C5=c5, C6=c6)
    print(f"\n**的中 {sum(verd.values())} / 外れ {len(verd)-sum(verd.values())}**"
          f"  {verd}")
    print(f"**校正値（判定を満たす最大の安全率）= {cal}**"
          f"（現行の既定は 0.70）")

    if args.out:
        json.dump(dict(safety_factors=sfs, calibrated=cal, verdicts=verd,
                       power_all=n_all, power_high_completion=n_sub,
                       summaries=S),
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
