#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_013 の 4 腕を集計し、カード §5 の必須項目を出す。

**入力**（`run_arm.py` が残したもの。凍結ハーネスには触れない）:
  - `outputs/exp_013_band_v4_reeval/<腕>/runs_detail.json` … 走行ごとの
    (歩数 n_cells, 旋回数 n_turns=裁定R4, 実走距離, 平均速度, visited_cells)
  - `outputs/exp_013_band_v4_reeval/<腕>/traj/<maze>.npz`  … 走行境界
    (run_index / run_t_start / run_t_end / run_outcome)

`runs_detail.json` には **t_start / t_end が入っていない**が、指標 (b) の定義は
「初回ゴール到達**より後に開始**した走行」で、開始時刻が要る。そこで traj の
走行境界を突き合わせて走行 dict を復元し、**凍結ハーネスの `maze_kpi` を
そのまま呼ぶ**（定義をここで書き直すと乖離するため）。

**指標の出所**（`docs/RESEARCH_PLAN.md` §2 が正本）:
  (a) ゴール到達率 / (b) 最短走行成立率 / (c) 有効最短走行率（短縮率 10% 以上）
  (c1) 経路短縮率 = 1 − (最短走行の実走経路長 / 探索走行の実走経路長)
  (c2) 速度向上率 = (最短走行の平均速度 / 探索走行の平均速度) − 1
  (d) 最速タイム / (e) 初回最短走行効率 / (e') 初回最短走行の経路効率

使い方:
    .venv/bin/python experiments/exp_013_band_v4_reeval/aggregate.py
    .venv/bin/python experiments/exp_013_band_v4_reeval/aggregate.py --arms l0c
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition.evaluator import maze_kpi  # noqa: E402  凍結ハーネスの定義を再利用

OUT_ROOT = REPO_ROOT / "outputs" / "exp_013_band_v4_reeval"
ARMS = ["l0a_e1", "l0a_e2", "l0b", "l0c"]
ARM_LABEL = {"l0a_e1": "L0-a(E1)", "l0a_e2": "L0-a(E2)", "l0b": "L0-b", "l0c": "L0-c"}


# ==========================================================================
# 分布の要約（中央値・四分位・最小・最大・n）。空なら全て None
# ==========================================================================
def dist(xs):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return {"n": 0, "median": None, "q1": None, "q3": None, "min": None, "max": None}
    a = np.asarray(xs, dtype=float)
    return {"n": len(xs),
            "median": float(np.median(a)), "q1": float(np.percentile(a, 25)),
            "q3": float(np.percentile(a, 75)), "min": float(a.min()), "max": float(a.max())}


def fmt(d, unit="", scale=1.0, prec=2):
    """分布を「中央値（四分位, n=…）」の 1 行に。"""
    if d["n"] == 0:
        return "—（n=0）"
    f = lambda v: f"{v * scale:.{prec}f}"
    return (f"{f(d['median'])}{unit}（{f(d['q1'])}〜{f(d['q3'])}, "
            f"最小 {f(d['min'])} / 最大 {f(d['max'])}, n={d['n']}）")


# ==========================================================================
# 1 腕の読み込み: runs_detail.json × traj の走行境界 → 面ごとの走行列
# ==========================================================================
def load_arm(arm):
    arm_dir = OUT_ROOT / arm
    p = arm_dir / "runs_detail.json"
    if not p.exists():
        return None, f"{arm}: runs_detail.json が無い（未完走）"
    detail = json.load(open(p, encoding="utf-8"))

    # 面ごとに走行を集める（runs_detail.json の順序は評価器の走行順）
    by_maze = {}
    for r in detail["runs"]:
        by_maze.setdefault(r["maze"], []).append(r)

    mazes = {}
    for maze, rows in by_maze.items():
        z = np.load(arm_dir / "traj" / f"{maze}.npz")
        # 走行境界を index で引けるようにする
        bounds = {int(i): (float(ts), float(te), str(oc)) for i, ts, te, oc
                  in zip(z["run_index"], z["run_t_start"], z["run_t_end"], z["run_outcome"])}
        runs = []
        for r in rows:
            ts, te, oc = bounds[int(r["run"])]
            if oc != r["outcome"]:
                raise SystemExit(f"{arm}/{maze} run {r['run']}: outcome 不一致 "
                                 f"（traj={oc} / detail={r['outcome']}）")
            runs.append(dict(index=int(r["run"]), outcome=r["outcome"],
                             run_time=r["run_time"], t_start=ts, t_end=te,
                             n_cells=r["n_cells"], n_turns=r["n_turns"],
                             path_length_m=r["path_length_m"], mean_speed=r["mean_speed"],
                             visited_cells=r["visited_cells"]))
        runs.sort(key=lambda q: q["index"])
        mazes[maze] = dict(d_true=int(rows[0]["d_true"]), runs=runs)
    return dict(arm=arm, policy=detail["policy"], maze_dir=detail["maze_dir"],
                mazes=dict(sorted(mazes.items()))), None


# ==========================================================================
# 面ごとの指標
# ==========================================================================
def per_maze_metrics(maze, rec):
    runs, d_true = rec["runs"], rec["d_true"]
    k = maze_kpi(runs)

    goal_runs = [r for r in runs if r["outcome"] == "goal" and r["run_time"] is not None]
    best_time = min((r["run_time"] for r in goal_runs), default=None)

    m = dict(maze=maze, d_true=d_true, n_runs=len(runs), n_goal_runs=len(goal_runs),
             a=k["goal_reached"], b=k["fast_run_done"], c=k["fast_run_effective"],
             explore_time=k["explore_time"], fast_time=k["fast_time"],
             best_time=best_time, improvement=k["improvement_ratio"],
             c1=None, c2=None, identity_err=None,
             e=None, e_harness=k["first_fast_efficiency"], e_degenerate=False,
             e_prime=None, n_later_goals=0, excess_cells=None,
             first_fast_time=k["first_fast_time"], first_fast_n_cells=None,
             first_fast_n_turns=None)

    if not k["fast_run_done"]:
        return m

    first_goal = min(goal_runs, key=lambda r: r["t_end"])
    later = [r for r in goal_runs if r["t_start"] > first_goal["t_end"]]
    m["n_later_goals"] = len(later)
    fastest = min(later, key=lambda r: r["run_time"])
    first_fast = min(later, key=lambda r: r["t_start"])

    # ---- (c1)(c2): 分解は「探索走行」対「最短走行（= later のうち最速）」で取る
    #      (c) の分母・分子と同じ 2 本を使わないと恒等式が成立しない
    if first_goal["path_length_m"] and fastest["path_length_m"] is not None:
        m["c1"] = 1.0 - (fastest["path_length_m"] / first_goal["path_length_m"])
    if first_goal["mean_speed"] and fastest["mean_speed"] is not None:
        m["c2"] = (fastest["mean_speed"] / first_goal["mean_speed"]) - 1.0
    if m["c1"] is not None and m["c2"] is not None and m["improvement"] is not None:
        # 恒等式 1 − t_最短/t_探索 = 1 − (1−c1)/(1+c2)（面単位でのみ成立）
        m["identity_err"] = abs(m["improvement"] - (1.0 - (1.0 - m["c1"]) / (1.0 + m["c2"])))

    # ---- (e) 初回最短走行効率 = 初回の最短走行タイム ÷ **その迷路での最良タイム**
    #      （§2 の正本。凍結ハーネスの first_fast_efficiency は分母が
    #        「探索後の走行のうち最速」で、探索走行が最速の面で値が食い違う。
    #        両方を出して食い違いを可視化する）
    if best_time and best_time > 0:
        m["e"] = first_fast["run_time"] / best_time
    # 退化面: 探索後の走行が 1 回だけ **かつ** 最良タイムがその走行（§2）
    m["e_degenerate"] = bool(len(later) == 1 and best_time is not None
                             and abs(best_time - first_fast["run_time"]) < 1e-9)

    # ---- (e') 初回最短走行が通過した**区画数** ÷ D_true（距離で正規化しない）
    m["first_fast_n_cells"] = first_fast["n_cells"]
    m["first_fast_n_turns"] = first_fast["n_turns"]
    if d_true > 0:
        m["e_prime"] = first_fast["n_cells"] / d_true
    # 計時窓の端で遷移が 1 つ落ちる（計時は**機体前端**通過基準、区画の判定は
    # **機体中心**基準）ため、**真の最短を引けた走行でも n_cells = D_true − 1**
    # になる。したがって (e') の下限は 1.00 ではなく (D−1)/D である。
    # 診断量として超過区画数を出す（**0 = 真の最短**、1 以上が遠回り）。
    m["excess_cells"] = first_fast["n_cells"] - (d_true - 1)
    return m


def summarize(arm_data):
    rows = [per_maze_metrics(mz, rec) for mz, rec in arm_data["mazes"].items()]
    n = len(rows)
    n_a = sum(1 for r in rows if r["a"])
    n_b = sum(1 for r in rows if r["b"])
    n_c = sum(1 for r in rows if r["c"])
    # (e) は退化面を **1.00 と書かず未定義**として除く（§2）
    e_ok = [r["e"] for r in rows if r["e"] is not None and not r["e_degenerate"]]
    n_deg = sum(1 for r in rows if r["e_degenerate"])
    n_e_undef = sum(1 for r in rows if not r["b"])
    return dict(
        arm=arm_data["arm"], label=ARM_LABEL[arm_data["arm"]], policy=arm_data["policy"],
        n_mazes=n, rows=rows,
        a=dict(n=n_a, rate=n_a / n if n else None),
        b=dict(n=n_b, rate=n_b / n if n else None),
        c=dict(n=n_c, rate=n_c / n if n else None),
        improvement=dist(r["improvement"] for r in rows),
        c1=dist(r["c1"] for r in rows), c2=dist(r["c2"] for r in rows),
        identity_err_max=max([r["identity_err"] for r in rows
                              if r["identity_err"] is not None], default=None),
        d=dist(r["best_time"] for r in rows),
        explore=dist(r["explore_time"] for r in rows),
        fast=dist(r["fast_time"] for r in rows),
        e=dist(e_ok), e_n_degenerate=n_deg, e_n_undefined=n_e_undef,
        e_harness=dist(r["e_harness"] for r in rows
                       if r["e_harness"] is not None and not r["e_degenerate"]),
        e_prime=dist(r["e_prime"] for r in rows),
        excess=dist(r["excess_cells"] for r in rows),
        n_excess_zero=sum(1 for r in rows if r["excess_cells"] == 0),
        n_excess_defined=sum(1 for r in rows if r["excess_cells"] is not None),
        e_prime_floor=dist((r["d_true"] - 1) / r["d_true"] for r in rows
                           if r["e_prime"] is not None),
        n_later_goals=dist(r["n_later_goals"] for r in rows if r["b"]),
        n_runs=dist(r["n_runs"] for r in rows),
        first_fast_time=dist(r["first_fast_time"] for r in rows),
    )


# ==========================================================================
# E2 の発動面数: 腕1 と腕2 で**走行構成が違う面**を数える
#   （E2 は予算で打ち切るので、余裕のある面では E1 と完全一致する。
#     一致した面は「E2 が発動していない」＝効果の測定になっていない）
# ==========================================================================
def e2_activation(a1, a2, tol=1e-6):
    if a1 is None or a2 is None:
        return None
    fired, same, detail = [], [], []
    for mz in sorted(set(a1["mazes"]) & set(a2["mazes"])):
        r1, r2 = a1["mazes"][mz]["runs"], a2["mazes"][mz]["runs"]
        why = None
        if len(r1) != len(r2):
            why = f"走行本数 {len(r1)}→{len(r2)}"
        else:
            for x, y in zip(r1, r2):
                if x["outcome"] != y["outcome"]:
                    why = f"run{x['index']} outcome {x['outcome']}→{y['outcome']}"
                    break
                t1, t2 = x["run_time"], y["run_time"]
                if (t1 is None) != (t2 is None):
                    why = f"run{x['index']} run_time の有無が違う"
                    break
                if t1 is not None and abs(t1 - t2) > tol:
                    why = f"run{x['index']} run_time {t1:.3f}→{t2:.3f} s"
                    break
                if x["visited_cells"] != y["visited_cells"]:
                    why = (f"run{x['index']} 訪問区画 "
                           f"{x['visited_cells']}→{y['visited_cells']}")
                    break
        (fired if why else same).append(mz)
        if why:
            detail.append(f"{mz}: {why}")
    return dict(n_fired=len(fired), n_same=len(same), fired=fired, detail=detail,
                n_compared=len(fired) + len(same))


# ==========================================================================
# 報告本文
# ==========================================================================
def render(summaries, act, missing):
    L = []
    w = L.append
    w("# exp_013 集計結果（凍結帯 v4・20 面・持ち時間 420 s / 最大 5 走行）\n")
    if missing:
        w("> ⚠️ **未完走の腕がある**: " + " / ".join(missing) + "\n")

    w("## 1. 主要 5 指標\n")
    w("| 腕 | (a) ゴール到達率 | (b) 最短走行成立率 | (c) 有効最短走行率 | (d) 最速タイム 中央値 | **(e') 経路効率**（主） | (e) 初回最短走行効率 |")
    w("|---|---|---|---|---|---|---|")
    for s in summaries:
        ep, e = s["e_prime"], s["e"]
        e_txt = (f"{e['median']:.3f}" if e["n"] else "—")
        e_txt += f"<br>未定義 {s['e_n_undefined']} / 退化 {s['e_n_degenerate']} 面"
        w(f"| **{s['label']}** | {s['a']['rate']*100:.0f}%（{s['a']['n']}/{s['n_mazes']}） "
          f"| {s['b']['rate']*100:.0f}%（{s['b']['n']}/{s['n_mazes']}） "
          f"| {s['c']['rate']*100:.0f}%（{s['c']['n']}/{s['n_mazes']}） "
          f"| {s['d']['median']:.2f} s | **{ep['median']:.3f}**（n={ep['n']}） | {e_txt} |"
          if s["d"]["median"] is not None and ep["n"] else
          f"| **{s['label']}** | — | — | — | — | — | — |")
    w("\n**(e') が主報告**（区画数 ÷ D_true。距離で正規化していない）。"
      "**(e) の退化面（探索後の走行 1 回かつ最良がその走行）は 1.00 と書かず未定義**とし、"
      "中央値から除外して件数を併記した（§2）。\n")

    w("> ### ⚠️ (e') を 1.00 基準で読まないこと（本集計で判明・重要）\n>")
    w("> **(e') の下限は 1.00 ではない。**計時は**機体前端**のセンサ通過基準、"
      "区画の判定は**機体中心**基準なので、**計時窓の端で区画の遷移が 1 つ落ちる**。"
      "したがって**真の最短経路を引けた走行でも `n_cells = D_true − 1`** となり、"
      "**(e') = (D−1)/D ≒ 0.98〜0.99** になる（引き継ぎメモ §2 (3) と同じ機構。"
      "そこでは 18 面すべてで歩数 = $D_0-1$ が確認されている）。")
    w("> **0.98 を「最短より良い」と読んではならない。**そこで診断量として")
    w("> **超過区画数 = n_cells − (D_true − 1)** を併記する（**0 = 真の最短**、"
      "1 以上が遠回り）。判定にはこちらを使うのが安全である。\n")
    w("| 腕 | 超過区画数 中央値（四分位, 最小/最大） | **真の最短を引けた面** | (e') の下限 (D−1)/D 中央値 |")
    w("|---|---|---|---|")
    for s in summaries:
        ex = s["excess"]
        if ex["n"] == 0:
            w(f"| {s['label']} | — | — | — |")
            continue
        w(f"| **{s['label']}** | {ex['median']:.1f}（{ex['q1']:.1f}〜{ex['q3']:.1f}, "
          f"{ex['min']:.0f}/{ex['max']:.0f}） | **{s['n_excess_zero']}/{s['n_excess_defined']} 面** "
          f"| {s['e_prime_floor']['median']:.3f} |")
    w("")

    w("## 2. 分布（二値化した割合だけにしない）\n")
    for s in summaries:
        w(f"### {s['label']}（{s['policy']}）\n")
        w("| 量 | 中央値（四分位, 最小/最大, n） |")
        w("|---|---|")
        w(f"| **(c) 短縮率**（面ごとに対応をとった比） | {fmt(s['improvement'], '%', 100, 1)} |")
        w(f"| (c1) 経路短縮率 | {fmt(s['c1'], '%', 100, 2)} |")
        w(f"| (c2) 速度向上率 | {fmt(s['c2'], '%', 100, 2)} |")
        w(f"| (d) 最速タイム | {fmt(s['d'], ' s')} |")
        w(f"| 探索走行タイム | {fmt(s['explore'], ' s')} |")
        w(f"| 最短走行タイム | {fmt(s['fast'], ' s')} |")
        w(f"| 初回最短走行タイム | {fmt(s['first_fast_time'], ' s')} |")
        w(f"| **(e')** 経路効率 | {fmt(s['e_prime'], '', 1, 3)} |")
        w(f"| 超過区画数（0 = 真の最短） | {fmt(s['excess'], ' 区画', 1, 1)} |")
        w(f"| **(e)** 初回最短走行効率（退化面を除く） | {fmt(s['e'], '', 1, 3)} |")
        w(f"| 探索後に成立した走行回数 | {fmt(s['n_later_goals'], ' 本', 1, 1)} |")
        w(f"| 1 面あたりの走行本数 | {fmt(s['n_runs'], ' 本', 1, 1)} |")
        ie = s["identity_err_max"]
        w(f"\n恒等式 1 − t_最短/t_探索 = 1 − (1−c1)/(1+c2) の**面単位**の最大絶対誤差: "
          f"{ie:.2e}\n" if ie is not None else "")
        w("> ⚠️ **恒等式が成立するのは面単位であって、中央値どうしでは成立しない。**"
          "上の (c1)(c2) の中央値から (c) の中央値を復元しようとすると 1〜3 ポイントずれる（§2）。\n")

    w("## 3. E2（予算つき追加探索）の発動\n")
    if act is None:
        w("腕 1 または腕 2 が未完走のため未算定。\n")
    else:
        w(f"- **発動面数: {act['n_fired']} / {act['n_compared']} 面**"
          f"（残り {act['n_same']} 面は E1 と走行構成が完全一致＝打ち切りが効いていない）")
        if act["n_fired"] == 0:
            w("- ⚠️ **発動 0。E2 の効果の測定になっていない**（カード §4-3）。"
              "打ち切り条件を触る前に、面ごとの残り時間を出すこと")
        for d in act["detail"]:
            w(f"  - {d}")
        w("")

    w("## 4. 面ごとの内訳\n")
    for s in summaries:
        w(f"### {s['label']}\n")
        w("| 面 | D_true | 走行 | ゴール | 探索 s | 最短 s | 最速 s | (c) | (c1) | (c2) | (e') | 超過区画 | (e) |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in s["rows"]:
            g = lambda v, p=2, sc=1.0: ("—" if v is None else f"{v*sc:.{p}f}")
            e_txt = "**退化**" if r["e_degenerate"] else g(r["e"], 3)
            ex = "—" if r["excess_cells"] is None else f"{r['excess_cells']:+d}"
            w(f"| {r['maze']} | {r['d_true']} | {r['n_runs']} | {r['n_goal_runs']} "
              f"| {g(r['explore_time'])} | {g(r['fast_time'])} | {g(r['best_time'])} "
              f"| {g(r['improvement'], 1, 100)}% | {g(r['c1'], 2, 100)}% | {g(r['c2'], 2, 100)}% "
              f"| {g(r['e_prime'], 3)} | {ex} | {e_txt} |")
        w("")

    w("## 5. 限界（必ず併記する）\n")
    w("- **L0-c の速度プロファイル安全率 0.7 は校正されていない。**"
      "`docs/L0PLUS_DESIGN.md` §8 の探索記録の表は**プレースホルダのまま空欄**であり"
      "（`| 0.7 | — | — | — | — |`）、0.6 → 0.7 → 0.8 と上げて完走率 100% を確認する"
      "手順（同 §2.2 の教授裁定 ①）が**実行された記録がリポジトリに無い**。"
      "したがって **L0-c の (d) は「校正されていない安全率で得た値」**である。"
      "校正のやり直しは本実験の後に別実験として教授が裁定する（准教授の指摘）")
    w("- **v3 帯と v4 帯は 20 面中 10 面を共有している**"
      "（seed 1018, 1019, 1021, 1023, 1024, 1041, 1046, 1073, 1103, 1138。npz の SHA-256 一致）。"
      "**exp_007 との比較を「独立な 2 帯の結果」として扱ってはならない。**"
      "さらに exp_007 の eval 腕は **v2 帯**で走っており、v4 と重なるのは **2 面のみ**である")
    w("- **`n_turns` は裁定 R4 の正本定義**（区画列の進行方向変化。180° = 2）で記録した。"
      "±45° リセットあり定義は使っていない")
    w("- **(e') は 1.00 が下限ではない**（上記 §1 の注記）。計時窓の端で区画の遷移が"
      "1 つ落ちるので、**真の最短でも (D−1)/D ≒ 0.98〜0.99** になる。"
      "この端の切り取り量は `front_offset`（`competition/evaluator.py`）と速度から"
      "見積もれるはずだが**未実施**であり、(e') の定義そのものを是正すべきかは"
      "教授の裁定事項である（本集計では定義を変えず、超過区画数を併記して回避した）")
    w("- **(e) の分母**は §2 の正本（その迷路での**最良タイム**）で計算した。"
      "凍結ハーネス `maze_kpi` の `first_fast_efficiency` は分母が"
      "「探索後の走行のうち最速」なので、**探索走行が最速の面で両者は食い違う**。"
      "参考値として両方を JSON に出してある")
    w("- **L0-b・L0-c には E1 が統合されていない**ので、"
      "**E2 の効果を測れるのは腕 1 対腕 2 の対比だけ**である（カード §2）\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--out", default=str(OUT_ROOT / "aggregate"))
    args = ap.parse_args()

    loaded, missing, summaries = {}, [], []
    for arm in args.arms:
        data, err = load_arm(arm)
        if err:
            missing.append(err)
            continue
        loaded[arm] = data
        summaries.append(summarize(data))

    act = e2_activation(loaded.get("l0a_e1"), loaded.get("l0a_e2"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = render(summaries, act, missing)
    Path(str(out) + ".md").write_text(md, encoding="utf-8")
    json.dump(dict(summaries=summaries, e2_activation=act, missing=missing),
              open(str(out) + ".json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    print(md)
    print(f"\n書き出し: {out}.md / {out}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
