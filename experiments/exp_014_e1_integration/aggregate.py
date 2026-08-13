#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_014（E1 統合）を集計し、**対照 exp_013 との面ごと対応差**を出す。

**対照は exp_013 の実測**（同一の凍結帯 v4・20 面なので面ごとに対応が取れる）。
処理条件と対照条件の対応:

    l0b_e1 ← outputs/exp_013_band_v4_reeval/l0b   （L0-b, E1 なし）
    l0c_e1 ← outputs/exp_013_band_v4_reeval/l0c   （L0-c, E1 なし）

**指標の定義は exp_013 の `aggregate.py` をそのまま読み込んで使う**
（ここで書き直すと乖離する。裁定 R14 の (e')・R15 の (e) を含む）。

**報告は §9-15 準拠**（裁定 R17）:
  - 条件ごとの中央値どうしを比べない。**面ごとに対応をとった差の分布**を出す
  - **二値指標 (a)(b)(c) の対応比較は McNemar**（Fisher は独立標本用で不適切）

使い方:
    .venv/bin/python experiments/exp_014_e1_integration/aggregate.py
"""
import argparse
import importlib.util
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# exp_013 の集計器を「定義の出どころ」として読み込む（コピーしない）
_spec = importlib.util.spec_from_file_location(
    "exp013_agg", REPO_ROOT / "experiments" / "exp_013_band_v4_reeval" / "aggregate.py")
E13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E13)

OUT_NEW = REPO_ROOT / "outputs" / "exp_014_e1_integration"
OUT_OLD = REPO_ROOT / "outputs" / "exp_013_band_v4_reeval"
PAIRS = [("l0b_e1", "l0b", "L0-b"), ("l0c_e1", "l0c", "L0-c")]

# 本実験の条件名を読み込み側で登録する。**exp_013 の aggregate.py は変更しない**
# （完了した実験の記録なので、後の実験の都合で書き換えない）
E13.ARM_LABEL.update({"l0b_e1": "L0-b+E1", "l0c_e1": "L0-c+E1"})


def load(root, arm):
    """exp_013 の loader を、出力ルートを差し替えて使う。"""
    saved = E13.OUT_ROOT
    E13.OUT_ROOT = root
    try:
        return E13.load_arm(arm)
    finally:
        E13.OUT_ROOT = saved


def mcnemar_exact(b, c):
    """対応のある二値比較の McNemar 正確検定（両側）。
    b = 対照のみ成立した面数, c = 処理のみ成立した面数。
    不一致ペアが 0 なら判定不能（None）。"""
    n = b + c
    if n == 0:
        return None
    lo = sum(comb(n, k) for k in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * lo / 2 ** n)


def paired(rows_new, rows_old, key, higher_is_better=False):
    """面ごとに対応をとった差（処理 − 対照）の分布。両方定義された面のみ。"""
    ms = [m for m in sorted(set(rows_new) & set(rows_old))
          if rows_new[m][key] is not None and rows_old[m][key] is not None]
    d = np.array([rows_new[m][key] - rows_old[m][key] for m in ms], dtype=float)
    if not len(d):
        return {"n": 0}
    better = (d > 1e-9) if higher_is_better else (d < -1e-9)
    worse = (d < -1e-9) if higher_is_better else (d > 1e-9)
    return {"n": len(d), "median": float(np.median(d)),
            "q1": float(np.percentile(d, 25)), "q3": float(np.percentile(d, 75)),
            "min": float(d.min()), "max": float(d.max()),
            "n_better": int(better.sum()), "n_worse": int(worse.sum()),
            "n_same": int(len(d) - better.sum() - worse.sum()),
            "mazes_worse": [m for m, w in zip(ms, worse) if w],
            "mazes_better": [m for m, w in zip(ms, better) if w]}


def fmt(p, unit="", prec=2):
    if not p.get("n"):
        return "—（n=0）"
    return (f"{p['median']:+.{prec}f}{unit}（{p['q1']:+.{prec}f}〜{p['q3']:+.{prec}f}, "
            f"最小 {p['min']:+.{prec}f} / 最大 {p['max']:+.{prec}f}, n={p['n']}）")


def e1_fired(rows_new, rows_old):
    """E1 の発動面: 走行構成（本数・outcome・タイム）が対照と違う面。
    E1 は探索後に寄り道するので、発動すれば必ず走行構成が変わる。"""
    fired, detail = [], []
    for m in sorted(set(rows_new) & set(rows_old)):
        a, b = rows_new[m], rows_old[m]
        why = None
        if a["n_runs"] != b["n_runs"]:
            why = f"走行本数 {b['n_runs']}→{a['n_runs']}"
        elif abs((a["explore_time"] or 0) - (b["explore_time"] or 0)) > 1e-6:
            why = f"探索走行 {b['explore_time']:.2f}→{a['explore_time']:.2f} s"
        elif abs((a["best_time"] or 0) - (b["best_time"] or 0)) > 1e-6:
            why = f"最速 {b['best_time']:.2f}→{a['best_time']:.2f} s"
        elif a["first_fast_n_cells"] != b["first_fast_n_cells"]:
            why = f"初回最短走行の歩数 {b['first_fast_n_cells']}→{a['first_fast_n_cells']}"
        if why:
            fired.append(m)
            detail.append(f"{m}: {why}")
    return fired, detail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT_NEW / "aggregate"))
    args = ap.parse_args()

    L, missing = [], []
    w = L.append
    w("# exp_014 集計結果 — E1 を L0-b / L0-c へ統合（対照: exp_013・同一の凍結帯 v4）\n")
    w("> **予測は投入前に確定済み**（カード §4。commit `8e437c8`。実装コミット `d9a9b2a` より前）。\n")

    payload = {}
    for new_arm, old_arm, label in PAIRS:
        dn, en = load(OUT_NEW, new_arm)
        do, eo = load(OUT_OLD, old_arm)
        if en or eo:
            missing.append(en or eo)
            continue
        sn, so = E13.summarize(dn), E13.summarize(do)
        rn = {r["maze"]: r for r in sn["rows"]}
        ro = {r["maze"]: r for r in so["rows"]}
        fired, fdetail = e1_fired(rn, ro)

        w(f"\n## {label} + E1（対照: {label}）\n")
        w("### 主要指標（左が対照 = exp_013、右が処理 = exp_014）\n")
        w("| 指標 | 対照 | 処理 |")
        w("|---|---|---|")
        for k, nm, f_ in [("a", "(a) ゴール到達率", None), ("b", "(b) 最短走行成立率", None),
                          ("c", "(c) 有効最短走行率", None)]:
            w(f"| {nm} | {so[k]['rate']*100:.0f}%（{so[k]['n']}/{so['n_mazes']}） "
              f"| {sn[k]['rate']*100:.0f}%（{sn[k]['n']}/{sn['n_mazes']}） |")
        w(f"| (d) 最速タイム 中央値 | {so['d']['median']:.2f} s | {sn['d']['median']:.2f} s |")
        w(f"| **(e') 経路効率 中央値** | {so['e_prime']['median']:.3f} | **{sn['e_prime']['median']:.3f}** |")
        w(f"| 超過区画 0 の面 | {so['n_excess_zero']}/{so['n_excess_defined']} "
          f"| **{sn['n_excess_zero']}/{sn['n_excess_defined']}** |")
        w(f"| **(e) 初回最短走行効率 中央値** | "
          f"{so['e']['median']:.3f}（n={so['e']['n']}） | "
          f"**{sn['e']['median']:.3f}（n={sn['e']['n']}）** |")
        w(f"| 探索走行 中央値 | {so['explore']['median']:.2f} s | {sn['explore']['median']:.2f} s |")
        w(f"| 走行本数 中央値 | {so['n_runs']['median']:.1f} | {sn['n_runs']['median']:.1f} |")
        w("\n> ⚠️ **上の表の中央値どうしを引き算して効果を読まないこと**（§9-15・`note_016`）。"
          "効果は下の**面ごとに対応をとった差**で読む。\n")

        w("### 面ごとに対応をとった差（処理 − 対照）— **§9-15 準拠**\n")
        w("| 量 | 差の分布 | 改善 | 悪化 | 不変 |")
        w("|---|---|---|---|---|")
        specs = [("(d) 最速タイム [s]", "best_time", False, " s", 2),
                 ("(e') 経路効率", "e_prime", False, "", 3),
                 ("超過区画数", "excess_cells", False, " 区画", 1),
                 ("(e) 初回最短走行効率", "e", False, "", 3),
                 ("探索走行タイム [s]", "explore_time", False, " s", 2),
                 ("走行本数", "n_runs", True, " 本", 1)]
        pay = {}
        for nm, key, hib, unit, prec in specs:
            p = paired(rn, ro, key, higher_is_better=hib)
            pay[key] = p
            if not p.get("n"):
                w(f"| {nm} | — | — | — | — |")
                continue
            w(f"| {nm} | {fmt(p, unit, prec)} | **{p['n_better']}** | **{p['n_worse']}** | {p['n_same']} |")
        w("")
        for nm, key in [("(d) が悪化した面", "best_time"), ("(e') が悪化した面", "e_prime")]:
            mz = pay[key].get("mazes_worse") or []
            if mz:
                w(f"- **{nm}**: {', '.join(mz)}")
        w("")

        w("### 二値指標の対応比較 — **McNemar**（Fisher は独立標本用なので使わない）\n")
        w("| 指標 | 対照のみ成立 | 処理のみ成立 | McNemar 正確検定 $p$ |")
        w("|---|---|---|---|")
        for k, nm in [("a", "(a)"), ("b", "(b)"), ("c", "(c)")]:
            ms = sorted(set(rn) & set(ro))
            bb = sum(1 for m in ms if ro[m][k] and not rn[m][k])
            cc = sum(1 for m in ms if rn[m][k] and not ro[m][k])
            p = mcnemar_exact(bb, cc)
            w(f"| {nm} | {bb} 面 | {cc} 面 | "
              f"{'—（不一致ペア 0）' if p is None else f'{p:.4f}'} |")
        w("")

        w(f"### E1 の発動\n")
        w(f"- **発動面数: {len(fired)} / {len(set(rn) & set(ro))} 面**"
          + ("（**発動 0。効果の測定になっていない**）" if not fired else ""))
        for d in fdetail[:25]:
            w(f"  - {d}")
        w("")
        payload[new_arm] = {"summary_new": sn, "summary_old": so,
                            "paired": pay, "fired": fired}

    if missing:
        w("\n> ⚠️ **未完走**: " + " / ".join(missing) + "\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = "\n".join(L)
    Path(str(out) + ".md").write_text(md, encoding="utf-8")
    json.dump(payload, open(str(out) + ".json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    print(md)
    print(f"\n書き出し: {out}.md / {out}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
