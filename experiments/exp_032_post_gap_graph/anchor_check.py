"""
experiments/exp_032_post_gap_graph/anchor_check.py
================
exp_032 の錨の独立再計算。`run.py` が出す集計（`outputs/exp_032_post_gap_graph/
latest/metrics.json`）を一切読まず、**一次記録（raw_records.json）だけ**から
群ごと・競技区分ごとの ρ の中央値を数え直し、run.py の集計と照合する
（`docs/RESEARCH_PLAN.md` §12-9 (c) が要求する「錨の独立再計算スクリプト」）。

一致すれば run.py の集計処理（中央値の取り方・群分けの実装）に誤りが無いことが
確かめられる。一致しなければ、run.py 側の集計ロジックか、本スクリプトの
どちらかに誤りがあるので、両方を読み比べて原因を特定すること。

使い方:
    source .venv/bin/activate
    python experiments/exp_032_post_gap_graph/anchor_check.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "exp_032_post_gap_graph", "latest")
RAW_PATH = os.path.join(OUT_DIR, "raw_records.json")
METRICS_PATH = os.path.join(OUT_DIR, "metrics.json")

TOLERANCE = 1e-9  # 中央値の再計算は同じ浮動小数点演算のはずなので厳しめでよい


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    vs = [v for v in values if v is not None]
    if not vs:
        return None
    return float(np.median(vs))


def _classify_contest_category(maze_name: str) -> str:
    """run.py::_classify_contest_category の複製（一次記録には category が
    既に入っているので本来は不要だが、「一次記録だけから数え直す」ことを
    徹底するため、maze_id からも独立に再現できることを確かめる）。"""
    if "frsh" in maze_name:
        return "フレッシュマン"
    if "_fin" in maze_name or maze_name.endswith("fin"):
        return "決勝"
    if "_pre" in maze_name or maze_name.endswith("pre"):
        return "予選"
    return "無区分"


def _check_equal(label: str, a: Optional[float], b: Optional[float], errors: List[str]) -> None:
    if a is None and b is None:
        return
    if a is None or b is None or abs(a - b) > 1e-6:
        errors.append(f"{label}: run.py集計={b} 独立再計算={a}")


def main() -> int:
    if not os.path.exists(RAW_PATH):
        print(f"一次記録が見つかりません: {RAW_PATH}（先に run.py を実行すること）")
        return 1

    raw = json.loads(open(RAW_PATH, encoding="utf-8").read())
    metrics = json.loads(open(METRICS_PATH, encoding="utf-8").read())
    summary = metrics["phase_specific"]

    errors: List[str] = []

    # --- design_v4 ---
    design_records = raw["design_v4"]
    design_rho = [r["rho"] for r in design_records if r["rho"] is not None]
    design_rho_median = _median(design_rho)
    _check_equal("design_v4.rho_median", design_rho_median, summary["design_v4"]["rho_median"], errors)
    n_anchor_fail_design = sum(1 for r in design_records if not r["anchor_ok"])
    if n_anchor_fail_design != summary["design_v4"]["n_anchor_fail"]:
        errors.append(
            f"design_v4.n_anchor_fail: run.py集計={summary['design_v4']['n_anchor_fail']} "
            f"独立再計算={n_anchor_fail_design}"
        )

    # --- contest_historical（全体・競技区分ごと） ---
    contest_records = raw["contest_historical"]
    contest_rho_all = [r["rho"] for r in contest_records if r["rho"] is not None]
    _check_equal(
        "contest_historical_all.rho_median",
        _median(contest_rho_all),
        summary["contest_historical_all"]["rho_median"],
        errors,
    )

    # category フィールドが maze_id からの再分類と一致することも確かめる
    # （一次記録の category を鵜呑みにせず、命名規約からも再現できるか）
    for r in contest_records:
        recomputed_cat = _classify_contest_category(r["maze_id"])
        if recomputed_cat != r["category"]:
            errors.append(
                f"category mismatch for {r['maze_id']}: raw record={r['category']} "
                f"maze_id から再分類={recomputed_cat}"
            )

    for cat in ["決勝", "予選", "フレッシュマン", "無区分"]:
        cat_records = [r for r in contest_records if r["category"] == cat]
        cat_rho = [r["rho"] for r in cat_records if r["rho"] is not None]
        _check_equal(
            f"contest_historical.{cat}.rho_median",
            _median(cat_rho),
            summary["contest_historical_by_category"][cat]["rho_median"],
            errors,
        )
        n_fail = sum(1 for r in cat_records if not r["anchor_ok"])
        summary_n_fail = summary["contest_historical_by_category"][cat]["n_anchor_fail"]
        if n_fail != summary_n_fail:
            errors.append(
                f"contest_historical.{cat}.n_anchor_fail: run.py集計={summary_n_fail} "
                f"独立再計算={n_fail}"
            )

    n_anchor_fail_contest = sum(1 for r in contest_records if not r["anchor_ok"])
    if n_anchor_fail_contest != summary["contest_historical_all"]["n_anchor_fail"]:
        errors.append(
            f"contest_historical_all.n_anchor_fail: run.py集計="
            f"{summary['contest_historical_all']['n_anchor_fail']} 独立再計算={n_anchor_fail_contest}"
        )

    print(f"design_v4: n={len(design_records)} rho中央値(独立再計算)={design_rho_median} "
          f"等価性の錨の不一致={n_anchor_fail_design}")
    print(f"contest_historical: n={len(contest_records)} "
          f"rho中央値(独立再計算)={_median(contest_rho_all)} "
          f"等価性の錨の不一致={n_anchor_fail_contest}")
    for cat in ["決勝", "予選", "フレッシュマン", "無区分"]:
        cat_records = [r for r in contest_records if r["category"] == cat]
        cat_rho = [r["rho"] for r in cat_records if r["rho"] is not None]
        print(f"  {cat}: n={len(cat_records)} rho中央値(独立再計算)={_median(cat_rho)}")

    if errors:
        print("\n不一致あり:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nrun.py の集計と独立再計算は全て一致した。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
