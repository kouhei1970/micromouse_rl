#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""016-cal 本体 — **安全率 1 点ぶんの校正走行**（カード `card_016cal.md`）。

**1 プロセス = 1 安全率**にしてある（**8 点を並行に回せる**ようにするため）。

--------------------------------------------------------------------------
測るもの（REQ_001 §2-4 の表を埋める）
--------------------------------------------------------------------------
- **完走**（**カード §5-2 の操作的定義**）:
  **(a) ゴール到達が真 かつ その迷路の全走行で outcome ∈ {goal, timeout}**
  （＝ **collision / tipover / stuck が 0 回**）
- **(d) 最速タイム = 完走走行の最速値**（研究計画書 §2。**探索走行も完走すれば算入する**）
- **(b) の最短走行タイム**（初回ゴールより後に開始した走行のうち最速）も**分けて記録する**
  （条文「探索走行タイムと最短走行タイムも分けて記録する」）。
  **指標の定義は凍結ハーネスの `maze_kpi` / `evaluate_maze` をそのまま呼ぶ**

  > ### 🔴 **2026-08-14 是正**（准教授の監査 AUDIT_039 §3-2・教授裁定）
  > **本スクリプトは (b) の最短走行タイム `fast_time` を
  > 「最速タイム」＝ (d) の位置で報告していた。**
  > **(d) は「完走走行の最速値」なので、探索走行が最速だった迷路では食い違う。**
  >
  > **影響**: **古典方策では両定義が一致する**（**確保済みの評価用 20 迷路で 20/20**。
  > 学生A が maze_1018 = 18.57999999956064 で実測・准教授も確認）。
  > **したがって M5 の確定基準表の 20 値は動かない。**
  > **食い違いうるのは、探索走行が最速になる方策**（RL 方策で起こりうる）である。
  >
  > **⚠️ 凍結評価ハーネス `competition/evaluator.py` は正しかったので触っていない**
  > （`:378-387` は (b) の定義・`:745` の `best_time` が (d)）。
- **横偏差 最大 / RMS**（**方策を包んで記録するだけ**。電圧も軌跡も変えない）
- 失敗様式（迷路 ID つき）

--------------------------------------------------------------------------
⚠️ 制御には一切触れない
--------------------------------------------------------------------------
**安全率はコンストラクタ引数**なので、**方策ファイルを 1 行も変更せずに振れる**。
`_do_drive_control` を包んでいるのは**記録のためだけ**で、**親を呼んでその返り値を
そのまま返す**（横偏差は親と同じ式で再計算する）。

使い方:
    .venv/bin/python experiments/exp_016_diagonal/run_016cal.py --safety 0.70
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.seed_bands import (assert_seeds_allowed,  # noqa: E402
                                describe_seeds)
import importlib  # noqa: E402

from competition.evaluator import CompetitionEvaluator, maze_kpi  # noqa: E402

# 走行が「壊れていない」とみなす outcome（カード §5-2）
OK_OUTCOMES = {"goal", "timeout"}


def make_probed(policy_spec: str):
    """**任意の方策を包んで横偏差だけ記録する**クラスを作る（**挙動は変えない**）。

    `_do_drive_control` は**親を呼んで返り値をそのまま返す**ので、
    **電圧も軌跡も 1 ビットも変わらない**。横偏差は**親と同じ式**で再計算している
    （親は `e_y = (x−p_x)(−sin ψ) + (y−p_y) cos ψ` を使う）。
    """
    mod, _, cls = policy_spec.partition(":")
    base = getattr(importlib.import_module(mod), cls)

    class ProbedCalPolicy(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._ey = []

        def _do_drive_control(self, obs, x, y, yaw):
            out = super()._do_drive_control(obs, x, y, yaw)
            path, i = self._path, self._cursor
            if path is not None:
                psi = float(path.heading[i])
                self._ey.append((x - float(path.x[i])) * (-math.sin(psi))
                                + (y - float(path.y[i])) * math.cos(psi))
            return out

    return ProbedCalPolicy


def wilson_lower(k: int, n: int, z: float = 1.645) -> float:
    """完走率の**片側 95% 下限**（Wilson）。REQ_001 §2-3 の判定に使う。

    **全数成功（k = n）でも 1.0 にはならない**のが要点である
    （**n=20 で 0.8808・n=60 で 0.9568**）。

    ⚠️ **2026-08-14 是正**（准教授の監査 AUDIT_039 の指摘）。
    ここに書いてあった **0.861 / 0.951 は正確法（Clopper-Pearson）の値**であり、
    **本関数が返す Wilson の値ではなかった**。**選定の結果には影響しない**
    （**両方式とも安全率 0.75 が選ばれる** — 准教授が確認済み）。

    **どちらの方式でも、n=60 の判定「下限 ≥ 0.95」は全数完走（60/60）でしか通らない**
    （Wilson で 60/60 → 0.9568 ／ **59/60 → 0.9287**）。
    **段階的な基準に見えて、運用上は全か無かの検査である**（`card_016cal.md` §8-1）。
    """
    if n == 0:
        return float("nan")
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d)


def best_time_of(runs) -> float:
    """**(d) 最速タイム = 完走走行の最速値** [s]（研究計画書 §2）。1 本も無ければ None。

    **`competition/evaluator.py:745` と同じ定義の写しである。**
    **写しがずれると気づけない**ので、`main()` は評価器が返す `best_time` と
    **1 走行ごとに突き合わせる**（食い違ったら例外で止まる）。
    ここに写しを置くのは、**シミュレータを回さずに単体検査できるようにするため**
    （`tests/test_run_016cal_d_metric.py`）。

    ⚠️ **(b) の `fast_time`（初回ゴールより後に開始した走行のうち最速）とは別物。**
    **探索走行が最速だった迷路では食い違う。**
    """
    ts = [float(r["run_time"]) for r in runs
          if r.get("outcome") == "goal" and r.get("run_time") is not None]
    return min(ts) if ts else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safety", type=float, required=True)
    ap.add_argument("--maze-dir", default="competition/mazes/cal_v4")
    ap.add_argument("--policy",
                    default="competition.baseline_slalom_e1t_tr_f0b_cal:SlalomE1TTRF0bCalPolicy",
                    help="方策（module:Class）。既定は校正済みの新既定")
    ap.add_argument("--gate-reason", default=None,
                    help="**確保済みの評価用迷路を使う場合のみ**。裁定 R40 の合言葉。"
                         "指定すると purpose='gate' で安全弁を通す（理由は記録される）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mazes = sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))
    seeds = [int(p.stem.split("_")[1]) for p in mazes]
    print(describe_seeds(seeds, "competition"))
    if args.gate_reason:
        # **確保済みの評価用迷路の使用は合言葉つきでのみ許す**（裁定 R40 条件 4）
        print(f"⚠️ **確保済みの評価用迷路を使う**（purpose='gate'）／理由: {args.gate_reason}")
        assert_seeds_allowed(seeds, namespace="competition", purpose="gate",
                             reason=args.gate_reason)
    else:
        assert_seeds_allowed(seeds, namespace="competition", purpose="validate")

    out = Path(args.out or (REPO_ROOT / "outputs" / "exp_016_diagonal" / "016cal"
                            / f"sf{args.safety:g}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"安全率 {args.safety:g}／{len(mazes)} 迷路／{args.maze_dir}"
          f"／方策 {args.policy}\n", flush=True)
    Probed = make_probed(args.policy)

    rows = []
    for m in mazes:
        pol = Probed(safety_factor=args.safety)
        ev = CompetitionEvaluator(maze_dir=args.maze_dir,
                                  out_dir=str(out.parent / f"sf{args.safety:g}_traj"))
        r = ev.evaluate_maze(m, pol)
        kpi = maze_kpi(r["runs"])
        outs = [q["outcome"] for q in r["runs"]]
        broken = [o for o in outs if o not in OK_OUTCOMES]
        ey = np.abs(np.asarray(pol._ey, dtype=float)) if pol._ey else np.array([0.0])
        # **(d) 最速タイム**（条文 = 完走走行の最速値）。**写しが凍結ハーネスから
        # ずれていないことを 1 迷路ごとに突き合わせる**
        best = best_time_of(r["runs"])
        if r.get("best_time") is not None or best is not None:
            if not (best is not None and r.get("best_time") is not None
                    and best == float(r["best_time"])):
                raise SystemExit(
                    f"{r['maze_id']}: (d) の写しが凍結ハーネスと食い違う "
                    f"（写し={best} / 評価器={r.get('best_time')}）")
        rows.append(dict(
            maze=r["maze_id"], n_runs=len(outs), outcomes=outs,
            # **完走の操作的定義**（カード §5-2）
            completed=bool(kpi["goal_reached"] and not broken),
            goal_reached=bool(kpi["goal_reached"]),
            n_broken=len(broken), broken_kinds=sorted(set(broken)),
            # 🔴 **(d) 最速タイム**（2026-08-14 是正。ここが判定に使う量）
            best_time=best,
            # **(b) の最短走行タイム**（条文「分けて記録する」。**(d) ではない**）
            fast_time=kpi["fast_time"], explore_time=kpi["explore_time"],
            fast_run_done=bool(kpi["fast_run_done"]),
            e_y_max_m=float(ey.max()), e_y_rms_m=float(np.sqrt(np.mean(ey ** 2))),
            n_ticks=int(ey.size)))
        c = rows[-1]
        print(f"  {c['maze']}: {'完走' if c['completed'] else '**未完走**'}"
              f" 走行 {c['n_runs']} 本 {outs}"
              f" (d)最速 {c['best_time'] if c['best_time'] else '-'}"
              f" (b)最短走行 {c['fast_time'] if c['fast_time'] else '-'}"
              f" e_y最大 {c['e_y_max_m']*1000:.2f} mm", flush=True)

    n = len(rows)
    k = sum(1 for c in rows if c["completed"])
    best = [c["best_time"] for c in rows if c["best_time"]]
    fast = [c["fast_time"] for c in rows if c["fast_time"]]
    eymax = [c["e_y_max_m"] for c in rows]
    eyrms = [c["e_y_rms_m"] for c in rows]
    summ = dict(safety_factor=args.safety, n=n, n_completed=k,
                completion_rate=k / n if n else float("nan"),
                completion_lower95=wilson_lower(k, n),
                # 🔴 **(d) 最速タイム**（2026-08-14 是正。判定はこちらを使う）
                best_time_median=(float(np.median(best)) if best else None),
                n_best=len(best),
                # **(b) の最短走行タイム**（分けて記録する。**(d) ではない**）
                fast_time_median=(float(np.median(fast)) if fast else None),
                n_fast=len(fast),
                e_y_max_median_m=float(np.median(eymax)),
                e_y_max_max_m=float(np.max(eymax)),
                e_y_rms_median_m=float(np.median(eyrms)),
                failed_faces=[c["maze"] for c in rows if not c["completed"]],
                failure_kinds=sorted({x for c in rows for x in c["broken_kinds"]}))
    print(f"\n【安全率 {args.safety:g}】完走 {k}/{n} = {summ['completion_rate']*100:.1f}%"
          f"／**片側 95% 下限 {summ['completion_lower95']:.3f}**")
    print(f"  **(d) 最速タイム（完走走行の最速値）中央値 {summ['best_time_median']}**"
          f"（n={summ['n_best']}）")
    print(f"  (b) 最短走行タイム 中央値 {summ['fast_time_median']}（n={summ['n_fast']}）"
          f"{'  ← (d) と同値' if summ['best_time_median'] == summ['fast_time_median'] else '  ← **(d) と食い違う**'}")
    print(f"  横偏差 最大の中央値 {summ['e_y_max_median_m']*1000:.2f} mm"
          f"／最悪 {summ['e_y_max_max_m']*1000:.2f} mm"
          f"／RMS 中央値 {summ['e_y_rms_median_m']*1000:.2f} mm")
    if summ["failed_faces"]:
        print(f"  未完走の迷路: {summ['failed_faces']}／様式 {summ['failure_kinds']}")

    json.dump(dict(summary=summ, rows=rows, maze_dir=args.maze_dir,
                   policy=args.policy, gate_reason=args.gate_reason),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
