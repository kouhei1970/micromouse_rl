"""
experiments/exp_036_racing_line/anchor_check.py
================
exp_036の錨の独立再計算。`run.py`が出す集計（`outputs/exp_036_racing_line/
latest/metrics.json`）・`run.py`のコード自体を一切読まず、**一次記録
（raw_records.json）だけ**からτの中央値・余裕の最小値・曲率の跳びの最大値を
数え直し、run.pyの集計と照合する（`docs/RESEARCH_PLAN.md` §12-9(c)が要求する
「錨の独立再計算スクリプト」・`PREREG.md` §5）。

一致すればrun.pyの集計処理（中央値の取り方・最小値/最大値の取り方）に誤りが
無いことが確かめられる。一致しなければ、run.py側の集計ロジックか、本
スクリプトのどちらかに誤りがあるので、両方を読み比べて原因を特定すること。

🔴 `run.py`・`classic/racing_line.py`のどちらの関数もimportしない
（PREREG §5「run.pyを読まない」の徹底。中央値・最小値・最大値という
単純な集計だけをNumPyで数え直す）。

使い方:
    source .venv/bin/activate
    python experiments/exp_036_racing_line/anchor_check.py
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

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "exp_036_racing_line", "latest")
RAW_PATH = os.path.join(OUT_DIR, "raw_records.json")
METRICS_PATH = os.path.join(OUT_DIR, "metrics.json")

TOLERANCE = 1e-9  # 同じ浮動小数点演算のはずなので厳しめでよい


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    vs = [v for v in values if v is not None]
    if not vs:
        return None
    return float(np.median(vs))


def _check_equal(label: str, independent: Optional[float], from_run: Optional[float], errors: List[str]) -> None:
    if independent is None and from_run is None:
        return
    if independent is None or from_run is None or abs(independent - from_run) > TOLERANCE:
        errors.append(f"{label}: run.py集計={from_run} 独立再計算={independent}")


def _check_equal_int(label: str, independent: int, from_run: int, errors: List[str]) -> None:
    if independent != from_run:
        errors.append(f"{label}: run.py集計={from_run} 独立再計算={independent}")


def main() -> int:
    if not os.path.exists(RAW_PATH):
        print(f"一次記録が見つかりません: {RAW_PATH}（先にrun.pyを実行すること）")
        return 1

    raw = json.loads(open(RAW_PATH, encoding="utf-8").read())
    metrics = json.loads(open(METRICS_PATH, encoding="utf-8").read())
    summary = metrics["phase_specific"]

    errors: List[str] = []
    records = raw["design_v4"]

    # --- 主判定量τの中央値（PREREG §2・§5） ---
    taus = [r["tau"] for r in records if r["tau"] is not None]
    tau_median = _median(taus)
    _check_equal("tau_median", tau_median, summary["tau_median"], errors)
    _check_equal("tau_min", float(np.min(taus)) if taus else None, summary["tau_min"], errors)
    _check_equal("tau_max", float(np.max(taus)) if taus else None, summary["tau_max"], errors)

    # --- 合格条件1: 余裕の最小値（迷路ごとの一次記録のκ(s)格子と対で
    #     記録されているmin_clearance_mを、run.pyの計算をなぞらず
    #     「全迷路の最小」を取るだけで数え直す） ---
    min_clears = [r["racing_line"]["min_clearance_m"] for r in records]
    min_clear_overall = float(np.min(min_clears)) if min_clears else None
    _check_equal("min_clearance_overall_m", min_clear_overall, summary["min_clearance_overall_m"], errors)

    n_below_margin = sum(1 for c in min_clears if c < 0.005 - 1e-9)
    _check_equal_int("n_clearance_below_margin", n_below_margin, summary["n_clearance_below_margin"], errors)

    # --- 合格条件2: 曲率の跳びの最大値 ---
    max_jumps = [r["racing_line"]["max_kappa_jump"] for r in records]
    max_jump_overall = float(np.max(max_jumps)) if max_jumps else None
    _check_equal("max_kappa_jump_overall", max_jump_overall, summary["max_kappa_jump_overall"], errors)

    # --- κ(s)格子そのものの整合性: 一次記録のs_grid/kappa_gridから
    #     曲率の跳びの最大値を「run.pyのmax_kappa_jump関数を使わず」
    #     numpy.diffだけで独立に数え直し、一次記録に書かれた値と一致するか
    #     （一次記録の値そのものが正しく計算されていたかの検算）。
    n_grid_mismatch = 0
    for r in records:
        kappa_grid = np.asarray(r["racing_line"]["kappa_grid"], dtype=np.float64)
        if len(kappa_grid) < 2:
            recomputed_jump = 0.0
        else:
            recomputed_jump = float(np.max(np.abs(np.diff(kappa_grid))))
        if abs(recomputed_jump - r["racing_line"]["max_kappa_jump"]) > 1e-6:
            n_grid_mismatch += 1
            errors.append(
                f"{r['maze_id']}: kappa_gridから再計算した跳び={recomputed_jump} "
                f"一次記録のmax_kappa_jump={r['racing_line']['max_kappa_jump']}"
            )

    # --- 否定対照（PREREG §4）: 判定条件が成立した件数を数え直す ---
    n1_fired = sum(1 for r in records if r["n1_unrounded_time_s"] > r["racing_line"]["ideal_time_s"])
    _check_equal_int("N1_n_fired", n1_fired, summary["negative_controls"]["N1_n_fired"], errors)

    n2_fired = sum(
        1 for r in records
        if r["n2_clearance_at_that_r_m"] is not None and r["n2_clearance_at_that_r_m"] < 0.0
    )
    _check_equal_int("N2_n_fired", n2_fired, summary["negative_controls"]["N2_n_fired"], errors)

    n3_fired = sum(
        1 for r in records
        if r["tau"] is not None and r["n3_tau_nodiag"] is not None
        and abs(r["n3_tau_nodiag"] - 1.0) < abs(r["tau"] - 1.0)
    )
    _check_equal_int("N3_n_fired", n3_fired, summary["negative_controls"]["N3_n_fired"], errors)

    n3_tau_nodiag_median = _median([r["n3_tau_nodiag"] for r in records])
    _check_equal(
        "N3_tau_nodiag_median", n3_tau_nodiag_median,
        summary["negative_controls"]["N3_tau_nodiag_median"], errors,
    )

    print(f"迷路数: {len(records)}")
    print(f"tau中央値(独立再計算)={tau_median} / run.py集計={summary['tau_median']}")
    print(f"余裕の最小値(全迷路, 独立再計算)={min_clear_overall}m / run.py集計={summary['min_clearance_overall_m']}m")
    print(f"曲率の跳びの最大値(独立再計算)={max_jump_overall} / run.py集計={summary['max_kappa_jump_overall']}")
    print(f"margin未満の迷路数(独立再計算)={n_below_margin} / run.py集計={summary['n_clearance_below_margin']}")
    print(f"κ(s)格子から跳びを再計算して一次記録と食い違った迷路数: {n_grid_mismatch}")
    print(f"N1成立数(独立再計算)={n1_fired} / N2成立数(独立再計算)={n2_fired} / N3成立数(独立再計算)={n3_fired}")

    if errors:
        print("\n不一致あり:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nrun.pyの集計と独立再計算は全て一致した。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
