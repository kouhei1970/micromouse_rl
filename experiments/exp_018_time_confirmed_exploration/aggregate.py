#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_018（確定判定を経路と同じ費用モデルで行う E1 → E1T）を集計し、
**対照 exp_015 の L0-c+E1+TR との面ごと対応差**を出す。

**対照は exp_015 の L0-c+E1+TR の実測**（同一の凍結帯・同一の 20 面なので面ごとに
対応が取れる）。処理条件と対照条件の対応:

    l0c_e1t_tr ← outputs/exp_015_time_optimal_route/l0c_e1_tr   （L0-c+E1+TR, 確定判定は歩数）

**指標の定義は exp_013 の `aggregate.py` をそのまま読み込んで使う**
（ここで書き直すと乖離する。裁定 R14 の (e')・R15 の (e) を含む）。
**面ごとに対応をとった差・McNemar・書式は exp_014 の `aggregate.py` を、
発動判定・持ち時間の使い方は exp_017 の `aggregate.py` をそのまま読み込んで使う**
（裁定 R23。ここで再実装しない）。

**追加探索の発動は「最初のゴール走行の直後の隙間」の変化で検出する**（exp_017 と
同じ考え方）: E1 と E1T の違いは「追加探索をどこまで続けるか（＝確定の基準）」
だけなので、確定の基準が変われば探索の終わり方が変わり、隙間
（= 次の走行の開始時刻 − 最初のゴール走行の終了時刻）が対照と処理で必ず変わる。

使い方:
    .venv/bin/python experiments/exp_018_time_confirmed_exploration/aggregate.py
    .venv/bin/python experiments/exp_018_time_confirmed_exploration/aggregate.py \\
        --new-root outputs/exp_018_design_check --old-root outputs/exp_015_design_check \\
        --out /path/to/scratch/agg_design
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module(name, relpath):
    """他実験の .py を importlib でモジュールとして読み込む（コピーしない。裁定 R23）。
    `sys.modules` に登録しない「切り離した」読み込みなので、E17 が内部で自前の
    E13/E14 コピーを読み込んでも、ここで読み込む E13/E14 とは干渉しない
    （E17 の `cutoff_activation` 等は走行 dict を直接受け取るだけで、
    E13/E14 のどのインスタンスかには依存しない）。"""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 指標の定義は exp_013、面ごと対応差・McNemar・書式は exp_014、
# 発動判定・持ち時間の使い方は exp_017 から読み込む（裁定 R23）
E13 = _load_module("exp013_agg", "experiments/exp_013_band_v4_reeval/aggregate.py")
E14 = _load_module("exp014_agg", "experiments/exp_014_e1_integration/aggregate.py")
E17 = _load_module("exp017_agg", "experiments/exp_017_budget_cutoff_l0b/aggregate.py")

# 本実験の条件名を読み込み側で登録する。**exp_013/exp_014/exp_017 の aggregate.py は変更しない**
E13.ARM_LABEL.update({"l0c_e1t_tr": "L0-c+E1T+TR", "l0c_e1_tr": "L0-c+E1+TR"})

# M5 gate 基準値（L0-c+E1・TR 無し）の出どころ。本実験の --old-root/--old-arm は
# 「対照 = exp_015 の L0-c+E1+TR」を指すので M5 gate 基準そのものではない。
# 基準は常に exp_014 の L0-c+E1 の実測（固定の帯・固定の条件）であり、
# --new-root/--old-root を設計帯へ差し替えても動かない（数値はハードコードせず、
# exp_014 自身の集計 JSON から読む）
M5_GATE_ROOT = REPO_ROOT / "outputs" / "exp_014_e1_integration"
M5_GATE_ARM = "l0c_e1"


def resolve(p):
    """相対パスはリポジトリルートからの相対とみなす。"""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load(root, arm):
    """exp_013 の loader を、出力ルートを差し替えて使う（exp_014/15/17 の `load()` を真似る）。"""
    saved = E13.OUT_ROOT
    E13.OUT_ROOT = root
    try:
        return E13.load_arm(arm)
    finally:
        E13.OUT_ROOT = saved


def read_m5_gate_reference():
    """exp_014 自身の集計 JSON から L0-c+E1 の (d) 中央値（M5 gate 基準値）を読む。
    見つからなければ None（本節は判定不能として報告する）。"""
    p = M5_GATE_ROOT / "aggregate.json"
    if not p.exists():
        return None, f"`{p}` が無い"
    try:
        payload = json.load(open(p, encoding="utf-8"))
        return payload[M5_GATE_ARM]["summary_new"]["d"]["median"], None
    except (KeyError, json.JSONDecodeError) as e:
        return None, f"`{p}` はあるが期待した形で読めなかった（{e}）"


# ==========================================================================
# 予測 U1〜U6 の機械的な照合
# ==========================================================================
def judge_predictions(rn, ro, common, pay, act, sn, so, mcnemar):
    j = {}

    # ---- U1: (e) が改善した面 >= 4 かつ 悪化した面 <= 1
    pe = pay["e"]
    if pe.get("n"):
        n_better, n_worse = pe["n_better"], pe["n_worse"]
        ok = (n_better >= 4) and (n_worse <= 1)
        j["U1"] = dict(
            pred="(e) が改善する面が 4 面以上、悪化する面は 1 面以下",
            measured=f"改善 {n_better} 面 / 悪化 {n_worse} 面（n={pe['n']}）",
            verdict="的中" if ok else "外れ")
    else:
        j["U1"] = dict(pred="(e) が改善する面が 4 面以上、悪化する面は 1 面以下",
                       measured="対応がとれた面が 0", verdict="判定不能")

    # ---- U2: 対照で超過区画 20 以上だった面が、処理ではすべて超過 5 以内
    qualifying = [m for m in common
                  if ro[m]["excess_cells"] is not None and ro[m]["excess_cells"] >= 20]
    if not qualifying:
        j["U2"] = dict(pred="対照で超過区画 20 以上だった面は、処理ではすべて超過 5 以内",
                       measured="対照で超過区画 20 以上の面が 0", verdict="判定不能")
    else:
        rows = [(m, ro[m]["excess_cells"], rn[m].get("excess_cells")) for m in qualifying]
        ok = all(new is not None and new <= 5 for _, _, new in rows)
        detail = ", ".join(f"{m}: {old}→{('未定義' if new is None else new)}"
                           for m, old, new in rows)
        j["U2"] = dict(
            pred="対照で超過区画 20 以上だった面は、処理ではすべて超過 5 以内",
            measured=f"対象 {len(qualifying)} 面（{detail}）",
            verdict="的中" if ok else "外れ")

    # ---- U3: (d) の対応差の中央値の絶対値 < 0.005 かつ 悪化面 0 かつ 改善の最大が 0.1 s 以内
    pd = pay["best_time"]
    if pd.get("n"):
        med = pd["median"]
        n_worse_d = pd["n_worse"]
        better_mazes = pd.get("mazes_better", [])
        improvements = [ro[m]["best_time"] - rn[m]["best_time"] for m in better_mazes]
        max_improve = max(improvements, default=0.0)
        ok = (abs(med) < 0.005) and (n_worse_d == 0) and (max_improve <= 0.1)
        j["U3"] = dict(
            pred="(d) の対応差の中央値は 0.00 s（|中央値|<0.005 s）、悪化面 0、改善は最大 0.1 s 以内",
            measured=(f"中央値 {med:+.4f} s / 悪化 {n_worse_d} 面 / "
                      f"改善の最大 {max_improve:.4f} s（n={pd['n']}）"),
            verdict="的中" if ok else "外れ")
    else:
        j["U3"] = dict(pred="(d) の対応差の中央値は 0.00 s（|中央値|<0.005 s）、悪化面 0、改善は最大 0.1 s 以内",
                       measured="対応がとれた面が 0", verdict="判定不能")

    # ---- U4: 隙間が延びた面（処理−対照 > 1e-6）>= 10 かつ 探索走行タイムが全面で不変
    n_extended = sum(1 for r in act["rows"] if r["diff"] > 1e-6)
    pex = pay["explore_time"]
    n_pay = pex.get("n", 0)
    n_same = pex.get("n_same", 0)
    n_common = len(common)
    if n_pay:
        ok = (n_extended >= 10) and (n_pay == n_common) and (n_same == n_common)
        j["U4"] = dict(
            pred="隙間が延びた面が 10 面以上、かつ探索走行タイムは全面で不変",
            measured=(f"隙間延長 {n_extended} 面 / 探索走行タイム不変 "
                      f"{n_same}/{n_pay} 面（対応 {n_common} 面）"),
            verdict="的中" if ok else "外れ")
    else:
        j["U4"] = dict(pred="隙間が延びた面が 10 面以上、かつ探索走行タイムは全面で不変",
                       measured="探索走行タイムの対応がとれた面が 0", verdict="判定不能")

    # ---- U5: (b) が落ちた面（対照で成立・処理で不成立）が 0〜1
    b_drop = mcnemar["b"]["b"]
    ok = 0 <= b_drop <= 1
    j["U5"] = dict(pred="(b) が落ちる面（対照で成立・処理で不成立）は 0〜1 面",
                   measured=f"落ちた {b_drop} 面", verdict="的中" if ok else "外れ")

    # ---- U6: (a)(c) が不一致 0 面、処理の (e') 中央値が対照以下、
    #          かつ「1.000 には戻らない」（処理で超過区画が 0 でない面が 1 面以上）
    mismatch_a = mcnemar["a"]["b"] + mcnemar["a"]["c"]
    mismatch_c = mcnemar["c"]["b"] + mcnemar["c"]["c"]
    ep_new, ep_old = sn["e_prime"]["median"], so["e_prime"]["median"]
    if ep_new is None or ep_old is None or sn["n_excess_defined"] == 0:
        j["U6"] = dict(
            pred="(a)(c) は不変、(e') は対照より改善するが 1.000 には戻らない",
            measured="(e') または超過区画が未定義", verdict="判定不能")
    else:
        cond1 = (mismatch_a == 0) and (mismatch_c == 0)
        cond2 = ep_new <= ep_old
        cond3 = sn["n_excess_zero"] < sn["n_excess_defined"]
        ok = cond1 and cond2 and cond3
        j["U6"] = dict(
            pred="(a)(c) は不変、(e') は対照より改善するが 1.000 には戻らない",
            measured=(f"(a) 不一致 {mismatch_a} 面 / (c) 不一致 {mismatch_c} 面 / "
                      f"(e') 中央値 {ep_old:.4f}→{ep_new:.4f} / "
                      f"超過区画 0 の面 {sn['n_excess_zero']}/{sn['n_excess_defined']}"),
            verdict="的中" if ok else "外れ")

    return j


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-root", default="outputs/exp_018_time_confirmed_exploration")
    ap.add_argument("--new-arm", default="l0c_e1t_tr")
    ap.add_argument("--old-root", default="outputs/exp_015_time_optimal_route")
    ap.add_argument("--old-arm", default="l0c_e1_tr")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    new_root = resolve(args.new_root)
    old_root = resolve(args.old_root)
    out = Path(args.out) if args.out else (new_root / "aggregate")

    L, missing = [], []
    w = L.append

    w("# exp_018 集計結果 — 確定判定を経路と同じ費用モデルで（E1 → E1T。対照: exp_015 の L0-c+E1+TR）\n")
    w("> **予測は実装前・exp_015 の凍結帯の結果を見る前に確定済み**（カード §4）。"
      "起票時点の記載: 「実装は 1 行もしていない。exp_015 の凍結帯の結果もまだ見ていない"
      "（設計帯の結果だけを見て書いている）」。\n")
    w(f"- 処理: `{args.new_arm}`（{new_root}）")
    w(f"- 対照: `{args.old_arm}`（{old_root}）\n")

    dn, en = load(new_root, args.new_arm)
    do, eo = load(old_root, args.old_arm)
    if en or eo:
        missing.append(en or eo)
        w("\n> ⚠️ **未完走**: " + " / ".join(missing) + "\n")
        out.parent.mkdir(parents=True, exist_ok=True)
        Path(str(out) + ".md").write_text("\n".join(L), encoding="utf-8")
        json.dump(dict(missing=missing), open(str(out) + ".json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("\n".join(L))
        print(f"\n書き出し: {out}.md / {out}.json")
        return 0

    sn, so = E13.summarize(dn), E13.summarize(do)
    rn = {r["maze"]: r for r in sn["rows"]}
    ro = {r["maze"]: r for r in so["rows"]}
    common = sorted(set(rn) & set(ro))

    # ---- 1. 主要指標 ----
    w("## 1. 主要指標（左が対照 = L0-c+E1+TR、右が処理 = L0-c+E1T+TR）\n")
    w("| 指標 | 対照 | 処理 |")
    w("|---|---|---|")
    for k, nm in [("a", "(a) ゴール到達率"), ("b", "(b) 最短走行成立率"),
                  ("c", "(c) 有効最短走行率")]:
        w(f"| {nm} | {so[k]['rate']*100:.0f}%（{so[k]['n']}/{so['n_mazes']}） "
          f"| {sn[k]['rate']*100:.0f}%（{sn[k]['n']}/{sn['n_mazes']}） |")
    w(f"| (d) 最速タイム 中央値 | {so['d']['median']:.2f} s | {sn['d']['median']:.2f} s |")
    w(f"| **(e') 経路効率 中央値** | {so['e_prime']['median']:.3f} | **{sn['e_prime']['median']:.3f}** |")
    w(f"| 超過区画 0 の面 | {so['n_excess_zero']}/{so['n_excess_defined']} "
      f"| {sn['n_excess_zero']}/{sn['n_excess_defined']} |")
    w(f"| **(e) 初回最短走行効率 中央値** | "
      f"{so['e']['median']:.3f}（n={so['e']['n']}） | "
      f"{sn['e']['median']:.3f}（n={sn['e']['n']}） |")
    w(f"| 探索走行 中央値 | {so['explore']['median']:.2f} s | {sn['explore']['median']:.2f} s |")
    w(f"| 走行本数 中央値 | {so['n_runs']['median']:.1f} | {sn['n_runs']['median']:.1f} |")
    w("\n> ⚠️ **上の表の中央値どうしを引き算して効果を読まないこと**（§9-15）。"
      "効果は下の**面ごとに対応をとった差**で読む。\n")

    # ---- 2. 面ごとに対応をとった差 ----
    w("## 2. 面ごとに対応をとった差（処理 − 対照）— **§9-15 準拠**\n")
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
        p = E14.paired(rn, ro, key, higher_is_better=hib)
        pay[key] = p
        if not p.get("n"):
            w(f"| {nm} | — | — | — | — |")
            continue
        w(f"| {nm} | {E14.fmt(p, unit, prec)} | **{p['n_better']}** | **{p['n_worse']}** | {p['n_same']} |")
    w("")
    for nm, key in [("(d) が悪化した面", "best_time"), ("(e') が悪化した面", "e_prime")]:
        mz = pay[key].get("mazes_worse") or []
        if mz:
            w(f"- **{nm}**: {', '.join(mz)}")
    w("")

    # ---- 3. McNemar ----
    w("## 3. 二値指標の対応比較 — **McNemar**（Fisher は独立標本用なので使わない）\n")
    w("| 指標 | 対照のみ成立 | 処理のみ成立 | McNemar 正確検定 $p$ |")
    w("|---|---|---|---|")
    mcnemar = {}
    for k, nm in [("a", "(a)"), ("b", "(b)"), ("c", "(c)")]:
        bb = sum(1 for m in common if ro[m][k] and not rn[m][k])
        cc = sum(1 for m in common if rn[m][k] and not ro[m][k])
        p = E14.mcnemar_exact(bb, cc)
        mcnemar[k] = dict(b=bb, c=cc, p=p)
        w(f"| {nm} | {bb} 面 | {cc} 面 | "
          f"{'—（不一致ペア 0）' if p is None else f'{p:.4f}'} |")
    w("")

    # ---- 4. 追加探索の発動 ----
    w("## 4. 追加探索の発動 — 最初のゴール走行直後の隙間で検出\n")
    w("E1 と E1T の違いは「確定の基準（歩数最短か時間最短か）」だけなので、"
      "基準が変われば**必ず**最初のゴール走行の直後の隙間"
      "（= 次の走行の開始時刻 − そのゴール走行の終了時刻）が変わる。"
      "差が 1e-6 s を超えたら発動とみなす（exp_017 と同じ判定）。\n")
    act = E17.cutoff_activation(dn["mazes"], do["mazes"], common)
    w(f"- **発動面数: {act['n_fired']} / {act['n_comparable']} 面**"
      + (f"（{act['n_skipped']} 面は最初のゴール走行の直後に次の走行が無く比較不可: "
         f"{', '.join(act['skipped'])}）" if act["n_skipped"] else ""))
    if not act["fired"]:
        w("- ⚠️ **発動 0。効果の測定になっていない**")
    else:
        w(f"- 発動面: {', '.join(act['fired'])}")
    w("")
    w("| 面 | 対照の隙間 [s] | 処理の隙間 [s] | 差 [s] | 発動 |")
    w("|---|---|---|---|---|")
    for r in act["rows"]:
        w(f"| {r['maze']} | {r['gap_old']:.2f} | {r['gap_new']:.2f} | {r['diff']:+.2f} | "
          f"{'○' if r['fired'] else ''} |")
    w("")

    # ---- 5. 持ち時間の使い方 ----
    w("## 5. 持ち時間の使い方 — 最終走行の終了時刻と残り\n")
    w(f"持ち時間は **{E17.BUDGET_S:.0f} s**（`docs/RESEARCH_PLAN.md` §2、exp_017 の定義を再利用）。"
      "走行タイムだけでなく、走行の隙間に落ちる費用も見る（`note_017`）。\n")
    bu = E17.budget_usage(dn["mazes"], do["mazes"], common)
    w("| 面 | 対照 終了 [s] | 対照 残り [s] | 処理 終了 [s] | 処理 残り [s] | 残りの差 [s] |")
    w("|---|---|---|---|---|---|")
    for r in bu["rows"]:
        w(f"| {r['maze']} | {r['end_old']:.2f} | {r['remain_old']:.2f} | "
          f"{r['end_new']:.2f} | {r['remain_new']:.2f} | {r['diff_remain']:+.2f} |")
    w("")
    st = bu["diff_remain_stats"]
    if st["median"] is not None:
        w(f"**残り時間の差（処理 − 対照）の分布**: 中央値 {st['median']:+.2f} s"
          f"（範囲 {st['min']:+.2f}〜{st['max']:+.2f} s、n={len(bu['rows'])}）\n")

    # ---- 6. 初回最短走行の比較（本実験の核心） ----
    w("## 6. 初回最短走行の比較（本実験の核心）\n")
    w("「初回ゴール走行より後に開始した最初のゴール走行」（= (e) が見ている走行。"
      "`first_fast_n_cells`/`first_fast_n_turns`/`excess_cells` は exp_013 の "
      "`per_maze_metrics` の定義をそのまま使う）を、面ごとに対照・処理で並べる。\n")
    ms7 = [m for m in common
           if rn[m]["first_fast_time"] is not None and ro[m]["first_fast_time"] is not None]
    pay7 = E14.paired(rn, ro, "first_fast_time", higher_is_better=False)
    w(f"- 対応がとれた面: **{len(ms7)} / {len(common)} 面**")
    if pay7.get("n"):
        w(f"- **改善（run_time 短縮）した面: {pay7['n_better']} 面** / "
          f"**悪化した面: {pay7['n_worse']} 面** / 不変 {pay7['n_same']} 面")
        if pay7.get("mazes_worse"):
            w(f"  - 悪化面: {', '.join(pay7['mazes_worse'])}")
    w("")
    w("| 面 | 対照 run_time [s] | 対照 (歩数,折れ数) | 対照 超過区画 | "
      "処理 run_time [s] | 処理 (歩数,折れ数) | 処理 超過区画 | 差 [s] | 判定 |")
    w("|---|---|---|---|---|---|---|---|---|")
    better_set = set(pay7.get("mazes_better") or [])
    worse_set = set(pay7.get("mazes_worse") or [])
    for m in ms7:
        a, b = ro[m], rn[m]
        diff = b["first_fast_time"] - a["first_fast_time"]
        mark = "改善" if m in better_set else ("悪化" if m in worse_set else "不変")
        w(f"| {m} | {a['first_fast_time']:.2f} | ({a['first_fast_n_cells']}, {a['first_fast_n_turns']}) "
          f"| {a['excess_cells']:+d} | {b['first_fast_time']:.2f} "
          f"| ({b['first_fast_n_cells']}, {b['first_fast_n_turns']}) | {b['excess_cells']:+d} "
          f"| {diff:+.2f} | {mark} |")
    w("")

    # ---- 7. 予測 U1〜U6 ----
    w("## 7. 予測 U1〜U6 の照合\n")
    judgments = judge_predictions(rn, ro, common, pay, act, sn, so, mcnemar)
    w("| # | 予測 | 実測（根拠） | 判定 |")
    w("|---|---|---|---|")
    for key in ["U1", "U2", "U3", "U4", "U5", "U6"]:
        j = judgments[key]
        mark = {"的中": "✓", "外れ": "✗", "判定不能": "—"}[j["verdict"]]
        w(f"| **{key}** | {j['pred']} | {j['measured']} | **{j['verdict']} {mark}** |")
    w("")

    # ---- 8. 参照線 ----
    w("## 8. 参照線\n")
    w(f"- 対照（L0-c+E1+TR）の (d) 中央値: **{so['d']['median']:.2f} s**")
    w(f"- 処理（L0-c+E1T+TR）の (d) 中央値: **{sn['d']['median']:.2f} s**\n")
    m5_ref, m5_err = read_m5_gate_reference()
    ref_check = dict(source=str(M5_GATE_ROOT / "aggregate.json"), arm=M5_GATE_ARM,
                     m5_ref=m5_ref, error=m5_err)
    if m5_ref is None:
        w(f"> `{M5_GATE_ROOT / 'aggregate.json'}` の M5 gate 基準値（L0-c+E1）を読めなかった"
          f"（{m5_err}）。参照線の突き合わせは未実施。\n")
    else:
        w(f"> **M5 gate の参照線は L0-c の {m5_ref:.2f} s（exp_014・L0-c+E1）であり、"
          "本実験の (d) が動けば参照線が動く。**\n")
        moved = abs(sn["d"]["median"] - m5_ref) > 1e-2
        ref_check["moved"] = moved
        if moved:
            w(f"> ⚠️ **参照線が動いている**: M5 gate 基準 {m5_ref:.2f} s に対し、"
              f"処理（L0-c+E1T+TR）の (d) 中央値は {sn['d']['median']:.2f} s"
              f"（差 {sn['d']['median']-m5_ref:+.2f} s）。**准教授の独立確認が必要**（裁定 R29）。\n")
        else:
            w(f"> 参照線は動いていない（M5 gate 基準 {m5_ref:.2f} s と "
              f"処理の (d) 中央値 {sn['d']['median']:.2f} s がほぼ一致）。\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    md = "\n".join(L)
    Path(str(out) + ".md").write_text(md, encoding="utf-8")
    payload = dict(
        new_arm=args.new_arm, old_arm=args.old_arm,
        new_root=str(new_root), old_root=str(old_root),
        summary_new=sn, summary_old=so, paired=pay, mcnemar=mcnemar,
        cutoff_activation=act, budget_usage=bu,
        first_fast_comparison=dict(n_comparable=len(ms7), n_common=len(common), paired=pay7),
        predictions=judgments, reference_line=ref_check,
    )
    json.dump(payload, open(str(out) + ".json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    print(md)
    print(f"\n書き出し: {out}.md / {out}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
