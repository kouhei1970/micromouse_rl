#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**(d) 最速タイムの定義を固定する検査**（AUDIT_039 §3-2 の是正・2026-08-14）。

条文（`docs/RESEARCH_PLAN.md` §2）:

> **(d) 最速タイム: 完走走行の最速値。**中央値と分布で報告する。
> **探索走行タイムと最短走行タイムも分けて記録する。**

**是正前の `run_016cal.py` は、(b) の最短走行タイム（初回ゴールより後に開始した
走行のうち最速）を「最速タイム」として報告していた。**
**探索走行が最速だった迷路では両者が食い違う。**

**本検査が固定するのはその 1 点である**:

    **探索走行（初回ゴール走行）が最速なら、その値が (d) になる。**

⚠️ **凍結評価ハーネス `competition/evaluator.py` は正しかったので触っていない。**
`maze_kpi` の `fast_time` は (b) の定義として条文どおりであり、
**本検査は「(b) と (d) は別物である」ことも同時に固定する**
（`fast_time` が変わっていないことを確かめる）。

    .venv/bin/python -m pytest tests/test_run_016cal_d_metric.py -q
"""
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition.evaluator import maze_kpi  # noqa: E402


def _load_run_016cal():
    """`run_016cal.py` を読み込む（ファイル名が識別子として使えないので直接読む）。"""
    p = REPO_ROOT / "experiments" / "exp_016_diagonal" / "run_016cal.py"
    spec = importlib.util.spec_from_file_location("run_016cal_for_test", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_016cal_for_test"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_run_016cal()


def _run(index, outcome, t_start, t_end, run_time):
    return dict(index=index, outcome=outcome, t_start=t_start, t_end=t_end,
                run_time=run_time)


# ==========================================================================
def test_exploration_run_counts_toward_d():
    """🔴 **探索走行が最速なら、それが (d) になる**（是正の本体）。

    走行 1（探索）が 10.0 s でゴール、その後の走行 2・3 は 12.0 / 11.0 s。
    - **(d) = 10.0**（完走走行の最速値 ＝ 探索走行）
    - **(b) の最短走行タイム = 11.0**（初回ゴールより後に開始した走行のうち最速）
    """
    runs = [_run(0, "goal", 0.0, 10.0, 10.0),
            _run(1, "goal", 11.0, 23.0, 12.0),
            _run(2, "goal", 24.0, 35.0, 11.0)]
    assert M.best_time_of(runs) == 10.0, "(d) が探索走行を算入していない"
    assert maze_kpi(runs)["fast_time"] == 11.0, "(b) の定義が変わってしまっている"
    assert M.best_time_of(runs) != maze_kpi(runs)["fast_time"], \
        "この標本では (d) と (b) は食い違うはず"


def test_matches_b_when_fast_run_is_fastest():
    """**最短走行が最速の場合は (d) と (b) が一致する**（古典方策の実際の姿）。"""
    runs = [_run(0, "goal", 0.0, 28.0, 28.0),
            _run(1, "goal", 29.0, 48.0, 19.0),
            _run(2, "goal", 49.0, 67.0, 18.58)]
    assert M.best_time_of(runs) == 18.58
    assert maze_kpi(runs)["fast_time"] == 18.58


def test_non_goal_runs_are_excluded():
    """**完走していない走行は算入しない**（条文「完走走行の」）。"""
    runs = [_run(0, "timeout", 0.0, 420.0, None),
            _run(1, "collision", 421.0, 425.0, None),
            _run(2, "goal", 426.0, 440.0, 14.0)]
    assert M.best_time_of(runs) == 14.0


def test_none_when_no_goal():
    """**1 本も完走しなければ None**（DNF）。"""
    runs = [_run(0, "timeout", 0.0, 420.0, None),
            _run(1, "stuck", 421.0, 430.0, None)]
    assert M.best_time_of(runs) is None
    assert maze_kpi(runs)["fast_time"] is None


def test_copy_matches_frozen_harness():
    """**写しが凍結評価ハーネスの定義からずれていない**ことを確かめる。

    `competition/evaluator.py:745` の `best_time` と同じ式であること
    （評価器の該当行と同じ計算を、ここで独立に書いて突き合わせる）。
    """
    cases = [
        [_run(0, "goal", 0.0, 10.0, 10.0), _run(1, "goal", 11.0, 23.0, 12.0)],
        [_run(0, "goal", 0.0, 28.0, 28.0), _run(1, "goal", 29.0, 47.0, 18.0)],
        [_run(0, "timeout", 0.0, 420.0, None), _run(1, "goal", 421.0, 435.0, 14.0)],
        [_run(0, "goal", 0.0, 9.5, 9.5)],
    ]
    for runs in cases:
        expect = min((r["run_time"] for r in runs if r["outcome"] == "goal"), default=None)
        assert M.best_time_of(runs) == expect
