"""tests/test_exp024_anchors.py — exp_024 錨の独立再計算の検査

PREREG.md §9 の指示どおり、**人工の一次記録（合成データ）で境界の両側**を
実測する。`docs/RESEARCH_PLAN.md` §12-9 (c) の作法（`tests/test_classic_checks.py`
と同じ）にならい、片側だけの検査は認めない — 合格側・不合格側を必ず対で書く。

🔴 合成データは実際の一次記録と**同じ形**にする（キー名を実物に合わせる）:
`{"runs": [{"index": int, "outcome": str, "run_time": float|None, ...}, ...],
  "run_phases": [{"run_index": int, "phase": str}, ...]}`
（`classic/policy.py` の `ClassicExplorerPolicy.get_run_phases()`・
`competition/evaluator.py` の `evaluate_maze()` が返す `runs` と同じ形）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from experiments.exp_024_s3_fast_run.recompute_anchors import (
    MazeAnchor,
    build_anchor,
    compute_C,
    compute_E,
    compute_R,
    compute_t_explore,
    compute_t_fast,
    judge_R,
    judge_declaration,
)


def _record(runs, run_phases):
    """実物の一次記録 JSON と同じ形の dict（`runs`/`run_phases` キーのみ）。"""
    return {"runs": runs, "run_phases": run_phases}


def _run(index, outcome, run_time):
    return {"index": index, "outcome": outcome, "run_time": run_time}


def _phase(run_index, phase):
    return {"run_index": run_index, "phase": phase}


# ==========================================================================
# 1. 主判定量 R の境界（PREREG §6）: R<1.0 合格側・R=1.0 不合格側・S空(+∞) 不合格側
# ==========================================================================
def test_R_below_one_is_pass():
    """R = 0.999（合格側）。judge_R と、compute_R を通した実測の両方で示す。"""
    assert judge_R(0.999) == "合格"

    ext = _record(
        runs=[_run(1, "goal", 100.0), _run(2, "goal", 99.9)],
        run_phases=[_phase(1, "EXPLORE"), _phase(2, "FAST")],
    )
    a = build_anchor(seed=41001, d0=52, extended_record=ext)
    assert a.ratio == pytest.approx(0.999)
    R = compute_R([a])
    assert R == pytest.approx(0.999)
    assert judge_R(R) == "合格"


def test_R_exactly_one_is_fail():
    """R = 1.0 ちょうど（不合格側）。同値の扱いは PREREG §6「R = 1.0 ちょうどは
    不合格側に入れる」を機械的に検査する。"""
    assert judge_R(1.0) == "不合格"

    ext = _record(
        runs=[_run(1, "goal", 100.0), _run(2, "goal", 100.0)],
        run_phases=[_phase(1, "EXPLORE"), _phase(2, "FAST")],
    )
    a = build_anchor(seed=41009, d0=57, extended_record=ext)
    assert a.ratio == pytest.approx(1.0)
    R = compute_R([a])
    assert R == pytest.approx(1.0)
    assert judge_R(R) == "不合格"


def test_R_is_plus_infinity_when_S_is_empty():
    """S が空（どの迷路も T_fast が定義されない）→ R = +∞（不合格側）。"""
    ext = _record(
        runs=[_run(1, "goal", 100.0)],  # 最短走行(FAST)の走行が1本も無い
        run_phases=[_phase(1, "EXPLORE")],
    )
    a = build_anchor(seed=41012, d0=66, extended_record=ext)
    assert a.in_s is False
    R = compute_R([a])
    import math
    assert math.isinf(R) and R > 0
    assert judge_R(R) == "不合格"


# ==========================================================================
# 2. T_fast: FAST の走行があっても outcome != "goal" なら未定義
# ==========================================================================
def test_t_fast_is_undefined_when_the_fast_run_did_not_reach_goal():
    """走行が始まった瞬間の段階が FAST であっても、その走行の outcome が
    "goal" でなければ T_fast は未定義（PREREG §5「かつゴールした走行」）。"""
    runs = [
        _run(1, "goal", 50.0),        # 探索走行はゴールしている(T_explore用)
        _run(2, "collision", 30.0),   # 最短走行(FAST)を試みたが衝突
    ]
    run_phases = [_phase(1, "EXPLORE"), _phase(2, "FAST")]

    t_fast = compute_t_fast(runs, run_phases)
    assert t_fast is None

    ext = _record(runs, run_phases)
    a = build_anchor(seed=41005, d0=75, extended_record=ext)
    assert a.t_explore == pytest.approx(50.0)
    assert a.t_fast_extended is None
    assert a.in_s is False


# ==========================================================================
# 3. T_explore: 「最初にゴールした走行」から取ること（最初の走行の失敗を無視）
# ==========================================================================
def test_t_explore_comes_from_the_first_run_that_reached_goal():
    """1本目が collision、2本目が goal のとき、T_explore は 2本目の
    run_time になること（1本目の run_time と混同しないこと）。"""
    runs = [
        _run(1, "collision", 15.0),
        _run(2, "goal", 80.0),
    ]
    t_explore = compute_t_explore(runs)
    assert t_explore == pytest.approx(80.0)
    assert t_explore != pytest.approx(15.0)

    ext = _record(runs, run_phases=[_phase(1, "EXPLORE"), _phase(2, "EXPLORE")])
    a = build_anchor(seed=41008, d0=78, extended_record=ext)
    assert a.t_explore == pytest.approx(80.0)


def test_t_explore_is_undefined_when_no_run_reaches_goal():
    """空振り防止: 1本もゴールしていなければ T_explore は未定義。"""
    runs = [_run(1, "collision", 15.0), _run(2, "timeout", 400.0)]
    assert compute_t_explore(runs) is None


# ==========================================================================
# 4. 副 1: 成立率 C の境界（PREREG §8「C >= 0.5」）: |S|=3(6件中) が合格側・
#    |S|=2(6件中) が不合格側
# ==========================================================================
def _anchor_in_s(seed: int) -> MazeAnchor:
    return MazeAnchor(seed=seed, d0=0, t_explore=100.0, t_fast_extended=90.0, t_fast_percell=None)


def _anchor_not_in_s(seed: int) -> MazeAnchor:
    return MazeAnchor(seed=seed, d0=0, t_explore=100.0, t_fast_extended=None, t_fast_percell=None)


def test_C_equal_half_is_pass_side():
    """|S|=3 / 6迷路 → C=0.5（合格側、境界を含む）。"""
    anchors = [_anchor_in_s(i) for i in range(3)] + [_anchor_not_in_s(i) for i in range(3, 6)]
    C = compute_C(anchors)
    assert C == pytest.approx(0.5)
    assert C >= 0.5  # PREREG §8 の宣言条件はこの実測がそのまま境界を満たす


def test_C_below_half_is_fail_side():
    """|S|=2 / 6迷路 → C≈0.333（不合格側）。"""
    anchors = [_anchor_in_s(i) for i in range(2)] + [_anchor_not_in_s(i) for i in range(2, 6)]
    C = compute_C(anchors)
    assert C == pytest.approx(2.0 / 6.0)
    assert C < 0.5


# ==========================================================================
# 5. 副 2: E の中央値（偶数個のとき、中央 2 値の平均になること）
# ==========================================================================
def test_E_median_with_even_count():
    """S' が 4 迷路（偶数）のとき、中央値が中央 2 値の平均になること
    （`statistics.median` の仕様の実測）。比の列は [0.5, 0.6, 0.7, 0.8]
    → 中央値 = (0.6+0.7)/2 = 0.65。"""
    ratios = [0.5, 0.6, 0.7, 0.8]
    anchors = [
        MazeAnchor(seed=100 + i, d0=0, t_explore=None,
                   t_fast_extended=r, t_fast_percell=1.0)
        for i, r in enumerate(ratios)
    ]
    E = compute_E(anchors)
    assert E == pytest.approx(0.65)


def test_E_is_plus_infinity_when_S_prime_is_empty():
    """空振り防止: percell 側の T_fast が誰も定義されていなければ E = +∞。"""
    anchors = [
        MazeAnchor(seed=1, d0=0, t_explore=None, t_fast_extended=90.0, t_fast_percell=None),
        MazeAnchor(seed=2, d0=0, t_explore=None, t_fast_extended=None, t_fast_percell=None),
    ]
    import math
    E = compute_E(anchors)
    assert math.isinf(E) and E > 0


# ==========================================================================
# 6. 宣言条件（PREREG §8 の表）: 4 通りをすべて機械的に確認する
# ==========================================================================
@pytest.mark.parametrize(
    "R, C, expect_keyword",
    [
        (0.5, 0.6, "S3完了"),
        (0.5, 0.3, "S1"),
        (1.2, 0.6, "S3の実装に不足"),
        (float("inf"), 0.0, "R=+∞"),
    ],
)
def test_declaration_table_covers_all_four_outcomes(R, C, expect_keyword):
    got = judge_declaration(R, C)
    assert expect_keyword in got, got
