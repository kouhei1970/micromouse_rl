#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_015（時間モデルによる経路選択 TR）を集計し、**対照 exp_014 の L0-c+E1 との
面ごと対応差**を出す。

**対照は exp_014 の L0-c+E1 の実測**（同一の凍結帯・同一の 20 面なので面ごとに
対応が取れる）。処理条件と対照条件の対応:

    l0c_e1_tr ← outputs/exp_014_e1_integration/l0c_e1   （L0-c+E1, TR なし）

**指標の定義は exp_013 の `aggregate.py` をそのまま読み込んで使う**
（ここで書き直すと乖離する。裁定 R14 の (e')・R15 の (e) を含む）。
**面ごとに対応をとった差・McNemar・書式は exp_014 の `aggregate.py` をそのまま
読み込んで使う**（裁定 R23。ここで再実装しない）。

**発動（TR が対照と別の経路を選んだか）は 2 通り出す**:
  1. **紙の上の予想**: `route_model.analyse_dir()` が地図から再現した経路どうし
     を時間モデルで採点した差（実走を回さない机上の予測）
  2. **実測での発動**: 各面の**最速のゴール走行**が実際に通過した区画列を
     `traj/<maze>.npz` の走行窓から復元し、対照と処理で比較する

使い方:
    .venv/bin/python experiments/exp_015_time_optimal_route/aggregate.py
    .venv/bin/python experiments/exp_015_time_optimal_route/aggregate.py \\
        --new-root outputs/exp_015_design_check --old-root outputs/exp_014_design_check \\
        --maze-dir competition/mazes/design_v4 --out /path/to/scratch/agg_design
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
    """他実験の .py を importlib でモジュールとして読み込む（コピーしない）。"""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 指標の定義は exp_013、面ごと対応差・McNemar・書式は exp_014 から読み込む（裁定 R23）
E13 = _load_module("exp013_agg", "experiments/exp_013_band_v4_reeval/aggregate.py")
E14 = _load_module("exp014_agg", "experiments/exp_014_e1_integration/aggregate.py")
RM = _load_module("exp015_route_model", "experiments/exp_015_time_optimal_route/route_model.py")

# 本実験の条件名を読み込み側で登録する。**exp_013/exp_014 の aggregate.py は変更しない**
E13.ARM_LABEL.update({"l0c_e1_tr": "L0-c+E1+TR", "l0c_e1": "L0-c+E1"})


def resolve(p):
    """相対パスはリポジトリルートからの相対とみなす。"""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load(root, arm):
    """exp_013 の loader を、出力ルートを差し替えて使う（exp_014 の `load()` を真似る）。"""
    saved = E13.OUT_ROOT
    E13.OUT_ROOT = root
    try:
        return E13.load_arm(arm)
    finally:
        E13.OUT_ROOT = saved


def read_cell_size(root, arm):
    p = Path(root) / arm / "runs_detail.json"
    return json.load(open(p, encoding="utf-8"))["cell_size"]


# ==========================================================================
# 実測での発動: 最速のゴール走行が通った区画列を traj から復元して比べる
# ==========================================================================
def fastest_goal_run(mazes, maze):
    """(d) の定義と同じ走行（探索走行を含む全走行中で最速のゴール到達）を返す。"""
    runs = mazes[maze]["runs"]
    goal_runs = [r for r in runs if r["outcome"] == "goal" and r["run_time"] is not None]
    if not goal_runs:
        return None
    return min(goal_runs, key=lambda r: r["run_time"])


def route_cells(traj_dir, maze, run, cell_size):
    """走行窓 [t_start, t_end] 内の試料から、連続重複を除いた区画列を復元する。"""
    z = np.load(traj_dir / f"{maze}.npz")
    bounds = {int(i): (float(ts), float(te)) for i, ts, te
              in zip(z["run_index"], z["run_t_start"], z["run_t_end"])}
    t_start, t_end = bounds[int(run["index"])]
    t, x, y = z["t"], z["x"], z["y"]
    mask = (t >= t_start) & (t <= t_end)
    cells = []
    for xv, yv in zip(x[mask], y[mask]):
        c = (int(xv // cell_size), int(yv // cell_size))
        if not cells or cells[-1] != c:
            cells.append(c)
    return cells


def measured_activation(rn, ro, mazes_new, mazes_old, traj_new, traj_old, cs_new, cs_old):
    """面ごとに、最速のゴール走行が通った区画列を対照・処理で比べる。"""
    common = sorted(set(rn) & set(ro))
    comparable = [m for m in common if rn[m]["a"] and ro[m]["a"]]
    fired, detail, skipped = [], [], []
    for m in comparable:
        run_new = fastest_goal_run(mazes_new, m)
        run_old = fastest_goal_run(mazes_old, m)
        if run_new is None or run_old is None:
            skipped.append(m)
            continue
        cells_new = route_cells(traj_new, m, run_new, cs_new)
        cells_old = route_cells(traj_old, m, run_old, cs_old)
        differs = cells_new != cells_old
        if differs:
            fired.append(m)
        detail.append(dict(maze=m, differs=differs,
                           old_n_cells=run_old["n_cells"], old_n_turns=run_old["n_turns"],
                           new_n_cells=run_new["n_cells"], new_n_turns=run_new["n_turns"]))
    return dict(n_comparable=len(comparable), n_skipped=len(skipped), skipped=skipped,
                fired=fired, detail=detail)


# ==========================================================================
# 予測 T1〜T6 の機械的な照合
# ==========================================================================
def judge_predictions(rn, ro, pay, sn, so):
    common = sorted(set(rn) & set(ro))
    j = {}

    # ---- T1: (d) が改善した面数が 2〜7、かつ改善も悪化もしなかった面は完全に不変
    pd = pay["best_time"]
    diffs_d = {m: rn[m]["best_time"] - ro[m]["best_time"] for m in common
               if rn[m]["best_time"] is not None and ro[m]["best_time"] is not None}
    same_mazes = [m for m in diffs_d
                  if m not in pd.get("mazes_worse", []) and m not in pd.get("mazes_better", [])]
    same_max_abs = max((abs(diffs_d[m]) for m in same_mazes), default=0.0)
    n_better = pd.get("n_better", 0)
    t1_ok = (2 <= n_better <= 7) and (same_max_abs < 1e-9)
    j["T1"] = dict(
        pred="(d) が改善する面は 2〜7 面、残りは完全に不変",
        measured=f"改善 {n_better} 面 / 不変面の最大|差| {same_max_abs:.2e} s（n={pd.get('n')}）",
        verdict="的中" if pd.get("n") else "判定不能",
    )
    if not pd.get("n"):
        j["T1"]["verdict"] = "判定不能"
    else:
        j["T1"]["verdict"] = "的中" if t1_ok else "外れ"

    # ---- T2: (d) の対応差の中央値の絶対値が 0.05 s 以内
    if pd.get("n"):
        med = pd["median"]
        t2_ok = abs(med) <= 0.05
        j["T2"] = dict(pred="(d) の対応差の中央値は 0.00 s（±0.05 s 以内）",
                       measured=f"中央値 {med:+.4f} s（n={pd['n']}）",
                       verdict="的中" if t2_ok else "外れ")
    else:
        j["T2"] = dict(pred="(d) の対応差の中央値は 0.00 s（±0.05 s 以内）",
                       measured="対応がとれた面が 0", verdict="判定不能")

    # ---- T3: 改善面の短縮率が 2〜5%、かつ絶対値が 0.2〜0.9 s
    better_mazes = pd.get("mazes_better", [])
    if better_mazes:
        pct = []
        sec = []
        for m in better_mazes:
            old_v, new_v = ro[m]["best_time"], rn[m]["best_time"]
            s = old_v - new_v
            sec.append(s)
            pct.append(s / old_v * 100.0 if old_v else float("nan"))
        pct_ok = all(2.0 <= p <= 5.0 for p in pct)
        sec_ok = all(0.2 <= s <= 0.9 for s in sec)
        t3_ok = pct_ok and sec_ok
        j["T3"] = dict(
            pred="改善面の短縮率は 2〜5%（絶対値 0.2〜0.9 s）",
            measured=(f"短縮率 {min(pct):.2f}〜{max(pct):.2f}%・"
                      f"短縮 {min(sec):.3f}〜{max(sec):.3f} s（改善 {len(better_mazes)} 面）"),
            verdict="的中" if t3_ok else "外れ")
    else:
        j["T3"] = dict(pred="改善面の短縮率は 2〜5%（絶対値 0.2〜0.9 s）",
                       measured="改善した面が 0", verdict="判定不能")

    # ---- T4: (d) が悪化した面が 0
    n_worse_d = pd.get("n_worse", 0) if pd.get("n") else 0
    j["T4"] = dict(pred="(d) が悪化する面は出ない",
                   measured=f"悪化 {n_worse_d} 面" + (f"（{', '.join(pd['mazes_worse'])}）"
                                                     if n_worse_d else ""),
                   verdict=("的中" if n_worse_d == 0 else "外れ") if pd.get("n") else "判定不能")

    # ---- T5: (e') が悪化する面が 1 面以上
    pep = pay["e_prime"]
    n_worse_ep = pep.get("n_worse", 0) if pep.get("n") else 0
    j["T5"] = dict(pred="(e') が悪化する面が 1 面以上出る",
                   measured=f"悪化 {n_worse_ep} 面" + (f"（{', '.join(pep['mazes_worse'])}）"
                                                       if n_worse_ep else ""),
                   verdict=("的中" if n_worse_ep >= 1 else "外れ") if pep.get("n") else "判定不能")

    # ---- T6: (a)(b)(c) が対照と完全一致（面ごとに）
    mismatch = [m for m in common
                if (rn[m]["a"], rn[m]["b"], rn[m]["c"])
                != (ro[m]["a"], ro[m]["b"], ro[m]["c"])]
    j["T6"] = dict(pred="(a)(b)(c) は不変（面ごとに完全一致）",
                   measured=(f"不一致 {len(mismatch)} 面"
                             + (f"（{', '.join(mismatch)}）" if mismatch else "")
                             + f" / 対応 {len(common)} 面"),
                   verdict="的中" if not mismatch else "外れ")
    return j


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-root", default="outputs/exp_015_time_optimal_route")
    ap.add_argument("--new-arm", default="l0c_e1_tr")
    ap.add_argument("--old-root", default="outputs/exp_014_e1_integration")
    ap.add_argument("--old-arm", default="l0c_e1")
    ap.add_argument("--maze-dir", default="competition/mazes/eval")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    new_root = resolve(args.new_root)
    old_root = resolve(args.old_root)
    out = Path(args.out) if args.out else (new_root / "aggregate")

    L, missing = [], []
    w = L.append

    w("# exp_015 集計結果 — 時間モデルによる経路選択 TR（対照: exp_014 の L0-c+E1）\n")
    w(f"> **予測は実装前に確定済み**（カード §4）。カード起票時点の記載: "
      "「時間モデルの再回帰は済ませたが、方策側は 1 行も実装していない」。\n")
    w(f"- 処理: `{args.new_arm}`（{new_root}）")
    w(f"- 対照: `{args.old_arm}`（{old_root}）")
    w(f"- 紙の上の予想に使う迷路帯: `{args.maze_dir}`\n")

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

    # ---- 2. 主要指標 ----
    w("## 1. 主要指標（左が対照 = L0-c+E1、右が処理 = L0-c+E1+TR）\n")
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

    # ---- 3. 面ごとに対応をとった差 ----
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

    # ---- 4. McNemar ----
    w("## 3. 二値指標の対応比較 — **McNemar**（Fisher は独立標本用なので使わない）\n")
    w("| 指標 | 対照のみ成立 | 処理のみ成立 | McNemar 正確検定 $p$ |")
    w("|---|---|---|---|")
    mcnemar = {}
    for k, nm in [("a", "(a)"), ("b", "(b)"), ("c", "(c)")]:
        ms = sorted(set(rn) & set(ro))
        bb = sum(1 for m in ms if ro[m][k] and not rn[m][k])
        cc = sum(1 for m in ms if rn[m][k] and not ro[m][k])
        p = E14.mcnemar_exact(bb, cc)
        mcnemar[k] = dict(b=bb, c=cc, p=p)
        w(f"| {nm} | {bb} 面 | {cc} 面 | "
          f"{'—（不一致ペア 0）' if p is None else f'{p:.4f}'} |")
    w("")

    # ---- 5. 発動 ----
    w("## 4. 発動 — 「紙の上の予想」と「実測」の 2 通り\n")

    w("### 4-1. 紙の上の予想（`route_model.analyse_dir`）\n")
    route_res = RM.analyse_dir(args.maze_dir)
    a_coef, b_coef = route_res["a"], route_res["b"]
    w(f"時間モデル（実測の回帰。ハードコードしない）: "
      f"t = {a_coef:.4f}×歩数 + {b_coef:.4f}×折れ数 → 旋回 1 回 = 移動 {b_coef/a_coef:.3f} 区画ぶん\n")
    fired_paper = [r for r in route_res["rows"] if r["gain_s"] > 1e-9]
    w(f"- **発動面数（時間モデル上で対照の経路より速い面）: "
      f"{len(fired_paper)} / {len(route_res['rows'])} 面**")
    if fired_paper:
        gp = np.array([r["gain_pct"] for r in fired_paper])
        w(f"- 発動面: {', '.join(r['maze'] for r in fired_paper)}")
        w(f"- 発動面の短縮率: 中央値 {np.median(gp):.2f}% ／ 最大 {gp.max():.2f}%")
    else:
        w("- ⚠️ **発動 0。効果の測定になっていない**")
    w("")
    w("| 面 | 対照 (歩,折) | 対照 t | TR (歩,折) | TR t | 短縮 [s] | 短縮 % | 発動 |")
    w("|---|---|---|---|---|---|---|---|")
    for r in route_res["rows"]:
        fire = r["gain_s"] > 1e-9
        w(f"| {r['maze']} | ({r['ctrl_moves']}, {r['ctrl_turns']}) | {r['ctrl_time']:.3f} "
          f"| ({r['tr_moves']}, {r['tr_turns']}) | {r['tr_time']:.3f} "
          f"| {r['gain_s']:+.3f} | {r['gain_pct']:+.2f}% | {'○' if fire else ''} |")
    w("")

    w("### 4-2. 実測での発動（最速のゴール走行が通った区画列を traj から復元）\n")
    cs_new = read_cell_size(new_root, args.new_arm)
    cs_old = read_cell_size(old_root, args.old_arm)
    if abs(cs_new - cs_old) > 1e-12:
        w(f"> ⚠️ cell_size が対照と処理で違う（対照 {cs_old} / 処理 {cs_new}）。"
          "区画復元の前提が崩れている可能性がある。\n")
    act = measured_activation(rn, ro, dn["mazes"], do["mazes"],
                              new_root / args.new_arm / "traj", old_root / args.old_arm / "traj",
                              cs_new, cs_old)
    w(f"- **発動面数（最速のゴール走行の経路が対照と処理で異なる面）: "
      f"{len(act['fired'])} / {act['n_comparable']} 面**"
      + (f"（{act['n_skipped']} 面は最速のゴール走行が定義できず比較不可）"
         if act["n_skipped"] else ""))
    if not act["fired"]:
        w("- ⚠️ **発動 0。効果の測定になっていない**")
    w("")
    if act["detail"]:
        w("| 面 | 経路 | 対照 (歩数,折れ数) | 処理 (歩数,折れ数) |")
        w("|---|---|---|---|")
        for d in act["detail"]:
            w(f"| {d['maze']} | {'**異なる**' if d['differs'] else '同じ'} "
              f"| ({d['old_n_cells']}, {d['old_n_turns']}) | ({d['new_n_cells']}, {d['new_n_turns']}) |")
        w("")

    # ---- 6. 予測 T1〜T6 ----
    w("## 5. 予測 T1〜T6 の照合\n")
    judgments = judge_predictions(rn, ro, pay, sn, so)
    w("| # | 予測 | 実測（根拠） | 判定 |")
    w("|---|---|---|---|")
    for key in ["T1", "T2", "T3", "T4", "T5", "T6"]:
        j = judgments[key]
        mark = {"的中": "✓", "外れ": "✗", "判定不能": "—"}[j["verdict"]]
        w(f"| **{key}** | {j['pred']} | {j['measured']} | **{j['verdict']} {mark}** |")
    w("")

    # ---- 7. 参照線 ----
    w("## 6. 参照線（裁定 R23）\n")
    w(f"- 対照（L0-c+E1）の (d) 中央値: **{so['d']['median']:.2f} s**")
    w(f"- 処理（L0-c+E1+TR）の (d) 中央値: **{sn['d']['median']:.2f} s**")
    w("\n**対照の値は exp_014 の実測をそのまま引いたものであり、本実験では作り直していない。**\n")
    ref_check = None
    old_agg_json = old_root / "aggregate.json"
    if old_agg_json.exists():
        try:
            payload = json.load(open(old_agg_json, encoding="utf-8"))
            ref_median = payload[args.old_arm]["summary_new"]["d"]["median"]
            moved = abs(ref_median - so["d"]["median"]) > 1e-9
            ref_check = dict(source=str(old_agg_json), ref_median=ref_median,
                             recomputed_median=so["d"]["median"], moved=moved)
            if moved:
                w(f"> ⚠️ **参照線が動いている**: exp_014 自身の報告（`{old_agg_json}`）では "
                  f"(d) 中央値 {ref_median:.2f} s だったが、本集計での再計算は "
                  f"{so['d']['median']:.2f} s。**准教授の独立確認が必要**（裁定 R29）。\n")
            else:
                w(f"> 参照線は動いていない（exp_014 自身の報告 `{old_agg_json}` と "
                  f"{ref_median:.2f} s で一致）。\n")
        except (KeyError, json.JSONDecodeError):
            w(f"> `{old_agg_json}` はあるが期待した形で読めなかった（参照線の突き合わせは未実施）。\n")
    else:
        w(f"> `{old_agg_json}` が無いため、exp_014 自身の報告との突き合わせは未実施。\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    md = "\n".join(L)
    Path(str(out) + ".md").write_text(md, encoding="utf-8")
    payload = dict(
        new_arm=args.new_arm, old_arm=args.old_arm,
        new_root=str(new_root), old_root=str(old_root), maze_dir=args.maze_dir,
        summary_new=sn, summary_old=so, paired=pay, mcnemar=mcnemar,
        activation_paper=dict(a=a_coef, b=b_coef, rows=route_res["rows"],
                              fired=[r["maze"] for r in fired_paper]),
        activation_measured=act, predictions=judgments, reference_line=ref_check,
    )
    json.dump(payload, open(str(out) + ".json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    print(md)
    print(f"\n書き出し: {out}.md / {out}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
