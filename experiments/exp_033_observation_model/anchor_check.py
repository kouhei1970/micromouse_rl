"""
experiments/exp_033_observation_model/anchor_check.py
================
PREREG §5「錨の独立再計算」。`run.py` の一次記録（`raw_records.json`）**だけ**を読み、
q_95（主判定量）を数え直して `run.py` の集計（`metrics.json`）と照合する。

🔴 `run.py` の集計出力（metrics.json の値そのもの）は読まない。`run.py` の関数も
import しない。標準ライブラリ + numpy だけで、一次記録の `predicted`/`actual` から
独立に計算する。

使い方:
    python3 experiments/exp_033_observation_model/anchor_check.py \
        [outputs/exp_033_observation_model/latest/raw_records.json]
"""
from __future__ import annotations

import json
import sys
from typing import Dict, List


def _load_raw_records(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["records"]


def _percentile95_nearest_rank(values: List[float]) -> float:
    """95 パーセンタイルを、外部ライブラリの補間方式に頼らず**独立に**計算する
    （numpy と同じ「線形補間」方式を使うと `run.py` と実装が近くなりすぎるため、
    ここでは教科書的な最近接順位法 (nearest-rank method) で計算し、2 つの
    独立な実装が近い値に収束することを確かめる）。

    最近接順位法: n 個をソートし、ceil(0.95*n) 番目（1始まり）の値を使う。
    """
    if not values:
        raise ValueError("空の配列の分位点は計算できない")
    vs = sorted(values)
    n = len(vs)
    # ceil(0.95*n) の 1-始まり順位 -> 0-始まり添字
    import math
    rank = math.ceil(0.95 * n)
    rank = max(1, min(n, rank))
    return vs[rank - 1]


def _percentile95_linear_interp(values: List[float]) -> float:
    """比較用にもう一方（numpy 既定と同じ線形補間方式）も計算しておく。
    2 つの方式は定義が違うのでビット一致はしないが、近い値になるはずである。"""
    if not values:
        raise ValueError("空の配列の分位点は計算できない")
    vs = sorted(values)
    n = len(vs)
    if n == 1:
        return vs[0]
    pos = 0.95 * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return vs[lo] * (1.0 - frac) + vs[hi] * frac


def recompute_q95(records: List[Dict]) -> Dict:
    """主判定量 q_95 を一次記録だけから数え直す
    （PREREG §2: 姿勢の集合とセンサ4本のすべての組み合わせをまとめて1つの分布にする）。
    """
    abs_diffs: List[float] = []
    for r in records:
        pred = r["predicted"]
        actual = r["actual"]
        if len(pred) != len(actual):
            raise ValueError(f"predicted/actual の長さが一致しない: {r.get('maze_id')}")
        for p, a in zip(pred, actual):
            abs_diffs.append(abs(p - a))

    return {
        "n_pairs": len(abs_diffs),
        "q95_nearest_rank": _percentile95_nearest_rank(abs_diffs),
        "q95_linear_interp": _percentile95_linear_interp(abs_diffs),
        "max_abs_diff": max(abs_diffs) if abs_diffs else None,
        "min_abs_diff": min(abs_diffs) if abs_diffs else None,
    }


def recompute_exclusion_count(records: List[Dict], expected_total_per_maze: Dict[str, int]) -> Dict:
    """迷路ごとの一次記録の件数から、除外された姿勢の数を数え直す
    （`n_poses_total`（直積の総数）は一次記録に含まれないため、`run.py` の
    直積の定義（区画16×5×5×24=9600/迷路）をこのファイル自身でも明記して使う。
    `run.py` の定数は import しない — 独立に同じ値をここに書く）。
    """
    # PREREG §3 の直積（run.py の WITHIN_CELL_OFFSETS_M/YAW_DEG/_select_cells と
    # 同じ値を、import せずこのファイル単独で明記する）。
    n_cells = 16
    n_offsets = 5
    n_yaw = 24
    poses_per_maze = n_cells * n_offsets * n_offsets * n_yaw

    per_maze_valid: Dict[str, int] = {}
    for r in records:
        per_maze_valid[r["maze_id"]] = per_maze_valid.get(r["maze_id"], 0) + 1

    result = {}
    for maze_id, n_valid in per_maze_valid.items():
        result[maze_id] = {
            "n_valid": n_valid,
            "n_poses_total_expected": poses_per_maze,
            "n_excluded_inferred": poses_per_maze - n_valid,
        }
    return result


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/exp_033_observation_model/latest/raw_records.json"
    records = _load_raw_records(path)
    print(f"一次記録を読み込んだ: {path} ({len(records)} 件)")

    q95_result = recompute_q95(records)
    print(json.dumps(q95_result, ensure_ascii=False, indent=2))

    exclusion = recompute_exclusion_count(records, {})
    print(json.dumps(exclusion, ensure_ascii=False, indent=2))

    q95 = q95_result["q95_nearest_rank"]
    bucket = "不明"
    if q95 < 0.002:
        bucket = "[0, 0.002) 幾何が合っている"
    elif q95 < 0.010:
        bucket = "[0.002, 0.010) ずれがある（原因特定要）"
    else:
        bucket = "[0.010, inf) 幾何モデルが足りない"
    print(f"\n独立再計算した q_95（最近接順位法） = {q95:.3e} m -> {bucket}")
    print(f"（比較用: 線形補間法では {q95_result['q95_linear_interp']:.3e} m）")


if __name__ == "__main__":
    main()
