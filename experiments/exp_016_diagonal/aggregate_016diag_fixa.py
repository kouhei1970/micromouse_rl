#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**是正 (A) の判定 A3・A4**（`card_016diag_fixA.md` §2）。

**読むだけで走らせない。**

| # | 予測 | 量 | 閾値 |
|---|---|---|---|
| **A3** | 回収が起きた迷路で斜めが使われ続ける | 最速走行で**斜め経路に乗っていたティックの割合** | **回収が起きた迷路で、是正 (B) 後より高い** |
| **A4** | 完走率は落ちない | **(a) ゴール到達**が成立した迷路数 | **20 / 20 迷路** |

> 🔴 **対照は「是正 (B) 後」である**（`card_016diag_fixA.md` §2）。
> **「是正前の壊れた状態」ではない** — `card_016diag_fixB.md` §6-3 の教訓。

**⚠️ 走行タイムは判定に使わない**（カード §2-1）。

**⚠️ 集計の前に全数の存在検査をする**（`card_016diag_fixB.md` §5-2 の教訓。
**欠測が黙って「良い側の値」に化ける型**を防ぐ）。

使い方:
    .venv/bin/python experiments/exp_016_diagonal/aggregate_016diag_fixa.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "experiments" / "exp_016_diagonal"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from aggregate_016diag_fixb import (diag_fraction_and_vmax,  # noqa: E402
                                    fast_run_window)
from aggregate_016diag_switch import d_best_time, load_arm  # noqa: E402

ROOT = REPO_ROOT / "outputs" / "exp_016_diagonal" / "016diag_switch"


def maze_level(runs):
    """迷路ごとの量は**最後の走行の記録**に付いている（先頭を見ると None になる）。"""
    return next((r for r in reversed(runs) if r.get("n_incidents") is not None),
                runs[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--before", default="diag_fixb", help="**対照 = 是正 (B) 後**")
    ap.add_argument("--after", default="diag_fixab", help="是正 (A) 後")
    ap.add_argument("--control", default="control", help="参考（斜めなしの新既定）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out or (root / "fixa_aggregate.json"))
    (mb, B) = load_arm(root / args.before)
    (ma, A) = load_arm(root / args.after)
    (mc, C) = load_arm(root / args.control)
    common = sorted(set(A) & set(B) & set(C))
    print(f"対照（是正 (B) 後） = {mb['policy']}\n処理（是正 (A) 後） = {ma['policy']}\n"
          f"参考（斜めなし）    = {mc['policy']}\n迷路 {len(common)} 件\n")

    rows = []
    print(f"{'迷路':<12}{'回収 前':>8}{'回収 後':>8}{'事故 前':>8}{'事故 後':>8}"
          f"{'斜め割合 前':>12}{'斜め割合 後':>12}{'最大速度 後':>12}{'ゴール':>7}")
    for mz in common:
        db, da = maze_level(B[mz]), maze_level(A[mz])
        zb, wb = fast_run_window(root / args.before, mz, B[mz])
        za, wa = fast_run_window(root / args.after, mz, A[mz])
        fb, vb, _ = diag_fraction_and_vmax(zb, wb)
        fa, va, _ = diag_fraction_and_vmax(za, wa)
        goal = any(r["outcome"] == "goal" for r in A[mz])
        rows.append(dict(maze=mz, n_retr_before=db.get("n_retrieval"),
                         n_retr_after=da.get("n_retrieval"),
                         n_inc_before=db.get("n_incidents"),
                         n_inc_after=da.get("n_incidents"),
                         diag_frac_before=fb, diag_frac_after=fa,
                         v_max_after=va, goal=goal,
                         n_flag_saved=da.get("n_flag_saved_diag"),
                         d_after=d_best_time(A[mz]), d_before=d_best_time(B[mz]),
                         d_control=d_best_time(C[mz])))
        q = rows[-1]
        print(f"{mz:<12}{q['n_retr_before']!s:>8}{q['n_retr_after']!s:>8}"
              f"{q['n_inc_before']!s:>8}{q['n_inc_after']!s:>8}"
              f"{fb*100:>11.2f}%{fa*100:>11.2f}%{va:>12.3f}{'○' if goal else '**×**':>7}")

    # ---- 🔴 集計の前に全数の存在検査（欠測が「良い側」に化けるのを防ぐ）----
    missing = [k for k in ("n_retr_before", "n_retr_after")
               if sum(1 for q in rows if q[k] is None)]
    if missing:
        print(f"\n🔴 **判定不能** — 記録が欠けている項目がある: {missing}")
        return 1

    # ---------------- A3 ------------------------------------------------
    retr = [q for q in rows if q["n_retr_before"]]
    print(f"\n【A3】回収が起きた迷路で斜めが使われ続けるか"
          f"（対照 = 是正 (B) 後。回収が起きた迷路 {len(retr)} 件）")
    if not retr:
        a3 = None
        print("  ⚠️ **対照で回収が起きた迷路が無い → 判定不能**")
    else:
        ups = [q for q in retr if q["diag_frac_after"] > q["diag_frac_before"]]
        a3 = len(ups) == len(retr)
        for q in retr:
            print(f"  {q['maze']}: 斜め割合 {q['diag_frac_before']*100:.2f} % → "
                  f"{q['diag_frac_after']*100:.2f} %"
                  f"（回収 {q['n_retr_before']} → {q['n_retr_after']} 回"
                  f"／印が効いた回数 {q['n_flag_saved']}）")
        print(f"  → **{'的中' if a3 else '外れ'}**（上がった {len(ups)}/{len(retr)} 迷路）")

    # ---------------- A4 ------------------------------------------------
    n_goal = sum(1 for q in rows if q["goal"])
    a4 = (n_goal == len(rows))
    print(f"\n【A4】完走率（予測: 20/20 迷路）: **{n_goal}/{len(rows)}** → "
          f"**{'的中' if a4 else '外れ'}**")

    # ---------------- 全体の様子（判定に使わない）------------------------
    fa_all = [q["diag_frac_after"] for q in rows]
    fb_all = [q["diag_frac_before"] for q in rows]
    print(f"\n【参考・判定に使わない】")
    print(f"  斜め割合の中央値: 是正 (B) 後 {np.median(fb_all)*100:.2f} %"
          f" → **是正 (A) 後 {np.median(fa_all)*100:.2f} %**")
    print(f"  事故 合計: {sum(q['n_inc_before'] for q in rows)} 件"
          f" → **{sum(q['n_inc_after'] for q in rows)} 件**"
          f"／回収 合計: {sum(q['n_retr_before'] for q in rows)} 回"
          f" → **{sum(q['n_retr_after'] for q in rows)} 回**")
    rel = [(q["d_after"] - q["d_control"]) / q["d_control"] for q in rows
           if q["d_after"] and q["d_control"]]
    relb = [(q["d_before"] - q["d_control"]) / q["d_control"] for q in rows
            if q["d_before"] and q["d_control"]]
    if rel:
        print(f"  対照（斜めなし）との対応差の中央値: 是正 (B) 後 {np.median(relb)*100:+.1f} %"
              f" → **是正 (A) 後 {np.median(rel)*100:+.1f} %**"
              f"（**走行タイムの判定は P1〜P5 で行う**）")

    json.dump(dict(before=mb["policy"], after=ma["policy"], n_mazes=len(rows),
                   A3=a3, A4=a4, per_maze=rows,
                   diag_frac_median_before=float(np.median(fb_all)),
                   diag_frac_median_after=float(np.median(fa_all)),
                   rel_median_before=float(np.median(relb)) if relb else None,
                   rel_median_after=float(np.median(rel)) if rel else None),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"\n書き出し: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
