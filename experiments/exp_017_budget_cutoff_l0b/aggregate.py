#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_017（予算つき打ち切り E2 を L0-b へ）を集計し、**対照 exp_014 の L0-b+E1
との面ごと対応差**を出す。

**対照は exp_014 の L0-b+E1 の実測**（同一の凍結帯・同一の 20 面なので面ごとに
対応が取れる）。処理条件と対照条件の対応:

    l0b_e2 ← outputs/exp_014_e1_integration/l0b_e1   （L0-b+E1, 打ち切りなし）

**指標の定義は exp_013 の `aggregate.py` をそのまま読み込んで使う**
（ここで書き直すと乖離する。裁定 R14 の (e')・R15 の (e) を含む）。
**面ごとに対応をとった差・McNemar・書式は exp_014 の `aggregate.py` をそのまま
読み込んで使う**（裁定 R23。ここで再実装しない）。

**打ち切りの発動は「最初のゴール走行の直後の隙間」の変化で検出する**: E1 と
E2 の違いは追加探索を打ち切るかどうかだけなので、打ち切りが効けばその分だけ
次の走行の開始が早まり、隙間（= 次の走行の開始時刻 − 最初のゴール走行の
終了時刻）が対照と処理で必ず変わる（`_cut_at` 自体は走行ログに残らないので、
この間接指標で発動を判定する）。

使い方:
    .venv/bin/python experiments/exp_017_budget_cutoff_l0b/aggregate.py
    .venv/bin/python experiments/exp_017_budget_cutoff_l0b/aggregate.py \\
        --new-root outputs/exp_017_design_check --old-root outputs/exp_014_design_check \\
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
    """他実験の .py を importlib でモジュールとして読み込む（コピーしない。裁定 R23）。"""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 指標の定義は exp_013、面ごと対応差・McNemar・書式は exp_014 から読み込む（裁定 R23）
E13 = _load_module("exp013_agg", "experiments/exp_013_band_v4_reeval/aggregate.py")
E14 = _load_module("exp014_agg", "experiments/exp_014_e1_integration/aggregate.py")

# 本実験の条件名を読み込み側で登録する。**exp_013/exp_014 の aggregate.py は変更しない**
E13.ARM_LABEL.update({"l0b_e2": "L0-b+E2", "l0b_e1": "L0-b+E1"})

# 持ち時間 [s]。docs/RESEARCH_PLAN.md §2（競技プロトコルの持ち時間規定）と
# 本実験カード §4 に記載された固定値。**数値ハードコード禁止の唯一の例外**としてここに置く
BUDGET_S = 420.0

# カード §5-1 で「面を指定して」事前登録された検証対象の面
MAZE_7837 = "maze_7837"


def resolve(p):
    """相対パスはリポジトリルートからの相対とみなす。"""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load(root, arm):
    """exp_013 の loader を、出力ルートを差し替えて使う（exp_014/exp_015 の `load()` を真似る）。"""
    saved = E13.OUT_ROOT
    E13.OUT_ROOT = root
    try:
        return E13.load_arm(arm)
    finally:
        E13.OUT_ROOT = saved


# ==========================================================================
# 打ち切りの発動: 最初のゴール走行の直後の隙間を対照・処理で比べる
# ==========================================================================
def first_goal_gap(mazes, maze):
    """最初にゴールした走行の直後の隙間
    （= 次の走行の開始時刻 − そのゴール走行の終了時刻）を返す。
    ゴール走行が無い、またはそれが最終走行（次が無い）なら None。"""
    runs = mazes[maze]["runs"]  # index 昇順（load_arm でソート済み）
    goal_runs = [r for r in runs if r["outcome"] == "goal" and r["run_time"] is not None]
    if not goal_runs:
        return None
    first_goal = min(goal_runs, key=lambda r: r["t_end"])
    pos = runs.index(first_goal)
    if pos + 1 >= len(runs):
        return None
    return runs[pos + 1]["t_start"] - first_goal["t_end"]


def cutoff_activation(mazes_new, mazes_old, common):
    """面ごとに、最初のゴール走行直後の隙間を対照・処理で比べる。
    差が 1e-6 s を超えたら「発動」とみなす。"""
    rows, fired, skipped = [], [], []
    for m in common:
        g_old = first_goal_gap(mazes_old, m)
        g_new = first_goal_gap(mazes_new, m)
        if g_old is None or g_new is None:
            skipped.append(m)
            continue
        diff = g_new - g_old
        is_fired = abs(diff) > 1e-6
        if is_fired:
            fired.append(m)
        rows.append(dict(maze=m, gap_old=g_old, gap_new=g_new, diff=diff, fired=is_fired))
    return dict(n_comparable=len(rows), n_skipped=len(skipped), skipped=skipped,
                fired=fired, n_fired=len(fired), rows=rows)


# ==========================================================================
# 持ち時間の使い方: 面ごとに最終走行の終了時刻・残りを対照・処理で比べる
# ==========================================================================
def final_run_end(mazes, maze):
    """その面の全走行のうち最後に終わった時刻（= run_t_end の最大）。"""
    runs = mazes[maze]["runs"]
    return max(r["t_end"] for r in runs)


def budget_usage(mazes_new, mazes_old, common):
    rows = []
    for m in common:
        end_old = final_run_end(mazes_old, m)
        end_new = final_run_end(mazes_new, m)
        rem_old = BUDGET_S - end_old
        rem_new = BUDGET_S - end_new
        rows.append(dict(maze=m, end_old=end_old, end_new=end_new,
                         remain_old=rem_old, remain_new=rem_new,
                         diff_remain=rem_new - rem_old))
    diffs = np.array([r["diff_remain"] for r in rows], dtype=float)
    if len(diffs):
        stats = dict(median=float(np.median(diffs)), min=float(diffs.min()), max=float(diffs.max()))
    else:
        stats = dict(median=None, min=None, max=None)
    return dict(rows=rows, diff_remain_stats=stats)


# ==========================================================================
# maze_7837 の面指定検証（カード §5-1 の事前登録）
# ==========================================================================
def maze_7837_detail(mazes_new, mazes_old):
    if MAZE_7837 not in mazes_new or MAZE_7837 not in mazes_old:
        return None
    runs_new = mazes_new[MAZE_7837]["runs"]
    runs_old = mazes_old[MAZE_7837]["runs"]
    by_new = {r["index"]: r for r in runs_new}
    by_old = {r["index"]: r for r in runs_old}
    idxs = sorted(set(by_new) | set(by_old))
    rows = [dict(index=i, old=by_old.get(i), new=by_new.get(i)) for i in idxs]
    return dict(rows=rows, run2_new=by_new.get(2), run2_old=by_old.get(2))


# ==========================================================================
# 予測 V1〜V6 の機械的な照合
# ==========================================================================
def judge_predictions(rn, ro, common, pay, act, m7837, sn, mcnemar):
    j = {}

    # ---- V1: maze_7837 の走行2 が goal かつ 走行2 の開始 < 310 s（処理側）
    r2n = m7837["run2_new"] if m7837 else None
    if r2n is None:
        j["V1"] = dict(
            pred="maze_7837 の走行2 が goal に回復し、開始が 310 s 以前になる",
            measured="maze_7837 が対応面に無い、または処理側に走行2 が存在しない",
            verdict="判定不能")
    else:
        ok = (r2n["outcome"] == "goal") and (r2n["t_start"] < 310.0)
        j["V1"] = dict(
            pred="maze_7837 の走行2 が goal に回復し、開始が 310 s 以前になる（登録値）",
            measured=f"走行2 outcome={r2n['outcome']} / 開始 {r2n['t_start']:.2f} s",
            verdict="的中" if ok else "外れ")

    # ---- V2: (b) が全面成立（処理）、かつ「対照で成立し処理で不成立」の面が 0
    b_full = sn["b"]["n"] == sn["n_mazes"]
    b_regress = mcnemar["b"]["b"]  # 対照のみ成立 = 処理で新たに落ちた面
    ok = b_full and (b_regress == 0)
    j["V2"] = dict(
        pred="(b) が全面成立、かつ新たに (b) を落とす面は 0",
        measured=f"処理の (b) {sn['b']['n']}/{sn['n_mazes']} 面 / 退行 {b_regress} 面",
        verdict="的中" if ok else "外れ")

    # ---- V3: 発動面数 >= 10
    ok = act["n_fired"] >= 10
    j["V3"] = dict(
        pred="打ち切りの発動面数は 10 面以上",
        measured=f"発動 {act['n_fired']} / {act['n_comparable']} 面（比較不可 {act['n_skipped']} 面）",
        verdict="的中" if ok else "外れ")

    # ---- V4: (e') 悪化面が 2〜8、かつ処理の (e') 中央値 <= 1.010
    ep = pay["e_prime"]
    if not ep.get("n"):
        j["V4"] = dict(pred="(e') 悪化面が 2〜8 面、処理の (e') 中央値は 1.010 以内",
                       measured="対応がとれた面が 0", verdict="判定不能")
    else:
        n_worse = ep["n_worse"]
        med_new = sn["e_prime"]["median"]
        ok = (2 <= n_worse <= 8) and (med_new is not None and med_new <= 1.010)
        med_txt = f"{med_new:.4f}" if med_new is not None else "—"
        j["V4"] = dict(
            pred="(e') 悪化面が 2〜8 面、処理の (e') 中央値は 1.010 以内",
            measured=f"悪化 {n_worse} 面 / 処理 (e') 中央値 {med_txt}",
            verdict="的中" if ok else "外れ")

    # ---- V5: maze_7837 の (d) < 100 s、他の面の (d) 対応差が全て |差|<=1.0 s、
    #          悪化面（maze_7837 を除く）が 0〜2
    d_7837 = rn.get(MAZE_7837, {}).get("best_time")
    if d_7837 is None:
        j["V5"] = dict(
            pred="maze_7837 の (d) が 100 s 未満に改善し、他の面は対応差 ±1.0 s 以内・悪化 0〜2 面",
            measured="maze_7837 の (d) が未定義、または対応面に無い",
            verdict="判定不能")
    else:
        others = [m for m in common if m != MAZE_7837
                  and rn[m].get("best_time") is not None and ro[m].get("best_time") is not None]
        other_diffs = {m: rn[m]["best_time"] - ro[m]["best_time"] for m in others}
        within = all(abs(v) <= 1.0 for v in other_diffs.values()) if other_diffs else True
        n_worse_others = sum(1 for v in other_diffs.values() if v > 1e-9)
        max_abs = max((abs(v) for v in other_diffs.values()), default=0.0)
        ok = (d_7837 < 100.0) and within and (0 <= n_worse_others <= 2)
        j["V5"] = dict(
            pred="maze_7837 の (d) が 100 s 未満に改善し、他の面は対応差 ±1.0 s 以内・悪化 0〜2 面",
            measured=(f"maze_7837 (d)={d_7837:.2f} s / 他 {len(others)} 面の対応差 "
                      f"最大|差| {max_abs:.3f} s / 悪化 {n_worse_others} 面"),
            verdict="的中" if ok else "外れ")

    # ---- V6: 走行本数が増えた面 >= 4
    nr = pay["n_runs"]
    if not nr.get("n"):
        j["V6"] = dict(pred="走行本数が増えた面が 4 面以上", measured="対応がとれた面が 0",
                       verdict="判定不能")
    else:
        ok = nr["n_better"] >= 4
        j["V6"] = dict(
            pred="走行本数が増えた面が 4 面以上",
            measured=f"増加 {nr['n_better']} 面（{', '.join(nr['mazes_better']) or 'なし'}）",
            verdict="的中" if ok else "外れ")

    return j


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-root", default="outputs/exp_017_budget_cutoff_l0b")
    ap.add_argument("--new-arm", default="l0b_e2")
    ap.add_argument("--old-root", default="outputs/exp_014_e1_integration")
    ap.add_argument("--old-arm", default="l0b_e1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    new_root = resolve(args.new_root)
    old_root = resolve(args.old_root)
    out = Path(args.out) if args.out else (new_root / "aggregate")

    L, missing = [], []
    w = L.append

    w("# exp_017 集計結果 — 予算つき打ち切り（E2）を L0-b へ（対照: exp_014 の L0-b+E1）\n")
    w("> **予測は実装前に確定済み**（カード §5。起票時点の記載: 「実装は 1 行もしていない」）。\n")
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
    w("## 1. 主要指標（左が対照 = L0-b+E1、右が処理 = L0-b+E2）\n")
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

    # ---- 4. 打ち切りの発動 ----
    w("## 4. 打ち切りの発動 — 最初のゴール走行直後の隙間で検出\n")
    w("E1 と E2 の違いは「追加探索をいつ打ち切るか」だけなので、打ち切りが発動すれば"
      "**必ず**最初のゴール走行の直後の隙間"
      "（= 次の走行の開始時刻 − そのゴール走行の終了時刻）が変わる。"
      "差が 1e-6 s を超えたら発動とみなす。\n")
    act = cutoff_activation(dn["mazes"], do["mazes"], common)
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
    w(f"持ち時間は **{BUDGET_S:.0f} s**（`docs/RESEARCH_PLAN.md` §2）。"
      "走行タイムだけでなく、走行の隙間に落ちる費用も見る（`note_017`）。\n")
    bu = budget_usage(dn["mazes"], do["mazes"], common)
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

    # ---- 6. maze_7837 の面指定検証 ----
    w("## 6. maze_7837 の面指定検証（カード §5-1 の事前登録）\n")
    m7837 = maze_7837_detail(dn["mazes"], do["mazes"])
    if m7837 is None:
        w(f"- ⚠️ `{MAZE_7837}` が対応面に無い（このバンドには存在しない可能性がある）。"
          "本節は判定不能。\n")
    else:
        w("| 走行 | 対照 開始 [s] | 対照 終了 [s] | 対照 outcome | 対照 run_time [s] | "
          "処理 開始 [s] | 処理 終了 [s] | 処理 outcome | 処理 run_time [s] |")
        w("|---|---|---|---|---|---|---|---|---|")
        for row in m7837["rows"]:
            o, n = row["old"], row["new"]
            def fv(r, k, p=2):
                return "—" if r is None or r.get(k) is None else f"{r[k]:.{p}f}"
            def fo_(r, k):
                return "—" if r is None else str(r[k])
            w(f"| {row['index']} | {fv(o, 't_start')} | {fv(o, 't_end')} | {fo_(o, 'outcome')} | "
              f"{fv(o, 'run_time')} | {fv(n, 't_start')} | {fv(n, 't_end')} | {fo_(n, 'outcome')} | "
              f"{fv(n, 'run_time')} |")
        w("")
        r2n = m7837["run2_new"]
        d_new = rn.get(MAZE_7837, {}).get("best_time")
        w("**事前登録値との突き合わせ（処理側）**:\n")
        if r2n is not None:
            w(f"- 走行2 の開始: {r2n['t_start']:.2f} s（登録: 310 s 以前）→ "
              f"{'○' if r2n['t_start'] < 310.0 else '×'}")
            w(f"- 走行2 の outcome: {r2n['outcome']}（登録: goal）→ "
              f"{'○' if r2n['outcome'] == 'goal' else '×'}")
        else:
            w("- 走行2 が存在しない（登録値との突き合わせ不可）")
        if d_new is not None:
            w(f"- (d): {d_new:.2f} s（登録: 100 s 未満）→ {'○' if d_new < 100.0 else '×'}")
        else:
            w("- (d) が未定義（登録値との突き合わせ不可）")
        w("")

    # ---- 7. 予測 V1〜V6 ----
    w("## 7. 予測 V1〜V6 の照合\n")
    judgments = judge_predictions(rn, ro, common, pay, act, m7837, sn, mcnemar)
    w("| # | 予測 | 実測（根拠） | 判定 |")
    w("|---|---|---|---|")
    for key in ["V1", "V2", "V3", "V4", "V5", "V6"]:
        j = judgments[key]
        mark = {"的中": "✓", "外れ": "✗", "判定不能": "—"}[j["verdict"]]
        w(f"| **{key}** | {j['pred']} | {j['measured']} | **{j['verdict']} {mark}** |")
    w("")

    # ---- 8. 参照線 ----
    w("## 8. 参照線\n")
    w(f"- 対照（L0-b+E1）の (d) 中央値: **{so['d']['median']:.2f} s**")
    w(f"- 処理（L0-b+E2）の (d) 中央値: **{sn['d']['median']:.2f} s**\n")
    w("> **M5 gate の参照線は L0-c のものであり、本実験は L0-b なので参照線ではない**"
      "（カード §4）。上の (d) 中央値は本実験内の対照との突き合わせにのみ使う。\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    md = "\n".join(L)
    Path(str(out) + ".md").write_text(md, encoding="utf-8")
    payload = dict(
        new_arm=args.new_arm, old_arm=args.old_arm,
        new_root=str(new_root), old_root=str(old_root),
        summary_new=sn, summary_old=so, paired=pay, mcnemar=mcnemar,
        cutoff_activation=act, budget_usage=bu, maze_7837=m7837,
        predictions=judgments,
    )
    json.dump(payload, open(str(out) + ".json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    print(md)
    print(f"\n書き出し: {out}.md / {out}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
