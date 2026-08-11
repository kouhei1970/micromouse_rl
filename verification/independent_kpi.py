#!/usr/bin/env python3
"""独立集計器 — docs/RESEARCH_PLAN.md §2 の定義文のみから実装した (a)〜(e) の再計算。

作成: 2026-08-11 准教授セッション（独立検証担当）
目的: 評価器 (competition/) の実装を一切参照せずに、生の結果 JSON
      (competition/results/exp007/<帯>/<方式>_<日時>/maze_*.json) の
      `runs` 配列だけから §2 の成績指標を再計算し、評価器自身が出した
      `summary.json` / 各 JSON の `kpi` ブロックと突き合わせる。

独立性の担保:
  - 本スクリプトは各 JSON の `kpi` ブロック・`best_time` ・`success` を
    「入力」としては一切使わない（突き合わせ段でのみ読む）。
  - 使う生データは runs[].{index,t_start,t_end,run_time,outcome} のみ。

§2 の定義文（引用）と、それに対する本実装の解釈は verification/REPORT.md に記載。
曖昧な箇所は両方の解釈を計算して併記する（*_A / *_B）。
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
# 既定は本番パス。第1引数で凍結スナップショットを指定できる
# （contest_reference 帯は学生A が実行中でファイルが増え続けるため、
#   集計は必ず凍結スナップショットに対して行う）。
RESULTS = REPO / "competition" / "results" / "exp007"

EPS = 1e-9
# (c) の閾値: 「最短走行タイムが探索走行タイムより 10% 以上短い」
IMPROVE_THRESHOLD = 0.10


# ---------------------------------------------------------------- 統計ユーティリティ
def quantile(xs: list[float], q: float) -> float | None:
    """線形補間の分位点（numpy に依存しない実装。R type-7 / numpy 既定と同一）。"""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[int(pos)]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def describe(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "min": min(xs),
        "q1": quantile(xs, 0.25),
        "median": quantile(xs, 0.50),
        "q3": quantile(xs, 0.75),
        "max": max(xs),
        "mean": sum(xs) / len(xs),
    }


# ---------------------------------------------------------------- 1 迷路の集計
@dataclass
class MazeResult:
    maze_id: str
    path: str
    n_runs: int                      # 走行回数（§2「スタート区画から出発した回数」＝ runs の要素数）
    n_goal_runs: int                 # ゴールへ到達した走行の回数
    outcomes: list[str] = field(default_factory=list)

    # (a)
    goal_reached: bool = False
    # (b)
    fast_run_done: bool = False
    n_fast_runs: int = 0             # 初回ゴール到達より後に開始し、ゴールへ到達した走行の回数
    # タイム
    explore_time: float | None = None       # 初回ゴール到達を果たした走行の走行タイム [s]
    fast_time: float | None = None          # (b) 該当走行のうち最速 [s]（解釈 A）
    best_time_all: float | None = None      # 全ゴール走行のうち最速 [s]（探索走行を含む）
    first_fast_time: float | None = None    # 初回の最短走行のタイム [s]
    # (c)
    improvement_ratio: float | None = None  # (explore - fast) / explore
    fast_run_effective: bool = False
    # (e)
    eff_A: float | None = None       # 初回最短走行タイム ÷ 全ゴール走行の最速
    eff_B: float | None = None       # 初回最短走行タイム ÷ 最短走行群の最速
    # (e) 改訂版（2026-08-11 §2 改訂 b2280dd）:
    #   分母は「その迷路での最良タイム」＝ 全ゴール走行の最速（＝解釈 A）。
    #   退化ガード: 探索後に成立した走行が 1 回だけで、かつ最良タイムがその走行なら
    #   比は定義上必ず 1.00 になり測定にならないので **未定義**とする。
    #   ただし その 1 回が探索走行より遅ければ最良タイムは探索走行側になり (e)>1 と
    #   なって情報を持つので、走行回数だけで機械的に落としてはならない。
    eff_v2: float | None = None
    eff_v2_degenerate: bool = False   # 退化により未定義に落とした面か
    # 追加量
    explore_over_fast: float | None = None  # 探索走行タイム ÷ 最短走行タイム（連続量）
    # (c) の分解: 時間短縮が「経路が短くなった」ためか「速く走った」ためか
    #   t = L / v  なので  t_exp/t_fast = (L_exp/L_fast) · (v_fast/v_exp)
    path_ratio: float | None = None   # L_exp / L_fast  … 地図による経路短縮の寄与
    speed_ratio: float | None = None  # v_fast / v_exp  … 走行方式（速度）の寄与
    # 正本 §2（7bb7f3c）の定義そのもの
    c1: float | None = None   # (c1) 経路短縮率 = 1 − (L_最短 / L_探索)
    c2: float | None = None   # (c2) 速度向上率 = (v_最短 / v_探索) − 1
    time_used: float | None = None          # 最終走行の t_end [s]
    hit_run_cap: bool = False               # 走行回数が上限に達したか
    last_outcome: str | None = None

    # 突き合わせ用（評価器の自己申告値。集計には使わない）
    ev_kpi: dict[str, Any] = field(default_factory=dict)
    ev_best_time: float | None = None
    ev_success: bool | None = None


def analyze_maze(path: Path) -> MazeResult:
    d = json.loads(path.read_text())
    runs = d["runs"]
    proto = d.get("protocol", {})
    max_runs = proto.get("max_runs")

    n_runs = len(runs)
    goal_runs = [r for r in runs if r["outcome"] == "goal"]

    m = MazeResult(
        maze_id=d["maze_id"],
        path=str(path),
        n_runs=n_runs,
        n_goal_runs=len(goal_runs),
        outcomes=[r["outcome"] for r in runs],
        ev_kpi=d.get("kpi", {}),
        ev_best_time=d.get("best_time"),
        ev_success=d.get("success"),
    )
    if runs:
        m.time_used = max(r["t_end"] for r in runs)
        m.last_outcome = runs[-1]["outcome"]
    if max_runs is not None:
        m.hit_run_cap = n_runs >= max_runs

    # ---- (a) ゴール到達率: 持ち時間内にゴール区画へ到達した走行が 1 回以上ある
    m.goal_reached = len(goal_runs) >= 1
    if not m.goal_reached:
        return m

    first_goal = goal_runs[0]
    m.explore_time = first_goal["run_time"]

    # ---- (b) 最短走行成立率: 「初回ゴール到達より後に開始した走行」でゴールへ到達
    #      初回ゴール到達の時刻 = 初回ゴール走行の t_end（先端がゴールセンサを通過した時点）
    t_first_goal = first_goal["t_end"]
    fast_runs = [r for r in goal_runs if r["t_start"] > t_first_goal + EPS]
    m.n_fast_runs = len(fast_runs)
    m.fast_run_done = len(fast_runs) >= 1

    # ---- タイム
    m.best_time_all = min(r["run_time"] for r in goal_runs)
    if fast_runs:
        m.fast_time = min(r["run_time"] for r in fast_runs)
        m.first_fast_time = fast_runs[0]["run_time"]

        # ---- (c) 有効最短走行: 最短走行タイムが探索走行タイムより 10% 以上短い
        m.improvement_ratio = (m.explore_time - m.fast_time) / m.explore_time
        m.fast_run_effective = m.improvement_ratio >= IMPROVE_THRESHOLD - EPS

        # ---- (e) 初回最短走行効率 = 初回の最短走行タイム ÷ その迷路での最良タイム
        m.eff_A = m.first_fast_time / m.best_time_all       # 最良＝全ゴール走行の最速
        m.eff_B = m.first_fast_time / m.fast_time           # 最良＝最短走行群の最速

        # ---- (e) 改訂版（§2 b2280dd）
        # 「最良タイムがその走行である」＝ 初回最短走行が全ゴール走行の最速と一致する
        best_is_that_run = abs(m.first_fast_time - m.best_time_all) <= EPS
        if len(fast_runs) == 1 and best_is_that_run:
            m.eff_v2 = None
            m.eff_v2_degenerate = True
        else:
            m.eff_v2 = m.eff_A

        # ---- 追加量: 探索走行タイム ÷ 最短走行タイム
        m.explore_over_fast = m.explore_time / m.fast_time

        # ---- (c) の分解: 最速の最短走行と、初回ゴール走行（探索走行）を比べる
        best_fast_run = min(fast_runs, key=lambda r: r["run_time"])
        l_exp, l_fast = first_goal["path_length_m"], best_fast_run["path_length_m"]
        if l_exp and l_fast:
            v_exp = l_exp / m.explore_time
            v_fast = l_fast / m.fast_time
            m.path_ratio = l_exp / l_fast      # >1 なら地図で経路が短くなった
            m.speed_ratio = v_fast / v_exp     # >1 なら最短走行の方が速く走っている
            # --- 正本 §2（7bb7f3c）の (c1) (c2) ---
            m.c1 = 1.0 - (l_fast / l_exp)      # 経路短縮率
            m.c2 = (v_fast / v_exp) - 1.0      # 速度向上率

    return m


# ---------------------------------------------------------------- 1 条件（帯×方式）の集計
def aggregate(mazes: list[MazeResult]) -> dict[str, Any]:
    n = len(mazes)
    a_hits = [m for m in mazes if m.goal_reached]
    b_hits = [m for m in mazes if m.fast_run_done]
    c_hits = [m for m in mazes if m.fast_run_effective]

    out: dict[str, Any] = {
        "n_mazes": n,
        # (a)
        "a_goal_rate": len(a_hits) / n if n else None,
        "a_count": f"{len(a_hits)}/{n}",
        # (b)
        "b_fast_done_rate": len(b_hits) / n if n else None,
        "b_count": f"{len(b_hits)}/{n}",
        # (c) 分母の解釈が一意でないため両方出す
        "c_effective_rate_over_all": len(c_hits) / n if n else None,          # 解釈 A: 全迷路が分母
        "c_effective_rate_over_b": (len(c_hits) / len(b_hits)) if b_hits else None,  # 解釈 B: (b) 該当面が分母
        "c_count_over_all": f"{len(c_hits)}/{n}",
        "c_count_over_b": f"{len(c_hits)}/{len(b_hits)}",
        # (d)
        "d_best_time": describe([m.best_time_all for m in a_hits if m.best_time_all is not None]),
        "d_explore_time": describe([m.explore_time for m in a_hits if m.explore_time is not None]),
        "d_fast_time": describe([m.fast_time for m in b_hits if m.fast_time is not None]),
        # --- (e) 改訂版（§2 b2280dd）: 分母＝全ゴール走行の最速、退化面は未定義 ---
        "e_v2": describe([m.eff_v2 for m in mazes if m.eff_v2 is not None]),
        "e_v2_undefined_no_fast_run": n - len(b_hits),                       # (b) 不成立で未定義
        "e_v2_undefined_degenerate": sum(1 for m in mazes if m.eff_v2_degenerate),  # 退化で未定義
        "e_v2_defined_count": sum(1 for m in mazes if m.eff_v2 is not None),
        # 退化ガードの内訳: 最短走行 1 回でも「探索走行の方が速い」面は情報を持つので残す
        "e_v2_single_fast_but_informative": sum(
            1 for m in b_hits if m.n_fast_runs == 1 and not m.eff_v2_degenerate),
        # --- (c) 連続量の併記（§2 b2280dd で必須化）: 短縮率 = 1 − t_最短/t_探索 ---
        #     面ごとに比を取ってから集計している（中央値どうしの比とは一致しない）
        "c_shrink_rate": describe([m.improvement_ratio for m in b_hits
                                   if m.improvement_ratio is not None]),
        # --- 正本 §2（7bb7f3c）の (c1)(c2) ---
        # (c1) 経路短縮率 = 1 − L_最短/L_探索 ／ (c2) 速度向上率 = v_最短/v_探索 − 1
        # 分解の恒等式: 1 − t_最短/t_探索 = 1 − (1−c1)/(1+c2)
        "c1_path_shrink": describe([m.c1 for m in b_hits if m.c1 is not None]),
        "c2_speed_gain": describe([m.c2 for m in b_hits if m.c2 is not None]),
        # (e)
        "e_first_fast_efficiency_A": describe([m.eff_A for m in b_hits if m.eff_A is not None]),
        "e_first_fast_efficiency_B": describe([m.eff_B for m in b_hits if m.eff_B is not None]),
        "e_undefined_count": n - len(b_hits),
        "e_A_equals_one_count": sum(1 for m in b_hits if m.eff_A is not None and m.eff_A <= 1.0 + 1e-9),
        # 追加量 1: 持ち時間内の走行回数
        "runs_per_maze": describe([float(m.n_runs) for m in mazes]),
        "goal_runs_per_maze": describe([float(m.n_goal_runs) for m in mazes]),
        # 追加量 2: 探索走行の後に成立した走行回数
        "fast_runs_per_maze": describe([float(m.n_fast_runs) for m in mazes]),
        "fast_runs_per_maze_among_a": describe([float(m.n_fast_runs) for m in a_hits]),
        # 追加量 3: 探索走行タイム ÷ 最短走行タイム（連続量）
        "explore_over_fast": describe([m.explore_over_fast for m in b_hits if m.explore_over_fast is not None]),
        # 改善率（連続量）
        "improvement_ratio": describe([m.improvement_ratio for m in b_hits if m.improvement_ratio is not None]),
        # 補助情報
        # (c) の分解
        "path_ratio": describe([m.path_ratio for m in b_hits if m.path_ratio is not None]),
        "speed_ratio": describe([m.speed_ratio for m in b_hits if m.speed_ratio is not None]),
        # --- 任務2 用の構造診断 ---
        # (e) が構造的に 1.00 に固定される面数（最短走行が 1 回しかない面では
        #     解釈 B の分母＝初回最短走行タイム自身になり、比は恒等的に 1.00）
        "e_forced_one_count": sum(1 for m in b_hits if m.n_fast_runs == 1),
        "e_forced_one_frac_of_b": (sum(1 for m in b_hits if m.n_fast_runs == 1) / len(b_hits)) if b_hits else None,
        # (c) の 10% 閾値の近傍に改善率が集中しているか（閾値が現象を潰しているかの診断）
        "c_threshold_sensitivity": {
            f"{t:.2f}": sum(1 for m in b_hits if m.improvement_ratio is not None
                            and m.improvement_ratio >= t - EPS)
            for t in (0.00, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30)
        },
        "c_near_threshold_count": sum(
            1 for m in b_hits if m.improvement_ratio is not None
            and abs(m.improvement_ratio - IMPROVE_THRESHOLD) <= 0.05
        ),
        "hit_run_cap_count": sum(1 for m in mazes if m.hit_run_cap),
        "last_run_timeout_count": sum(1 for m in mazes if m.last_outcome == "timeout"),
        "time_used": describe([m.time_used for m in mazes if m.time_used is not None]),
    }
    # 走行回数のヒストグラム
    hist: dict[str, int] = {}
    for m in mazes:
        hist[str(m.n_runs)] = hist.get(str(m.n_runs), 0) + 1
    out["runs_per_maze_hist"] = dict(sorted(hist.items(), key=lambda kv: int(kv[0])))
    hist2: dict[str, int] = {}
    for m in mazes:
        hist2[str(m.n_fast_runs)] = hist2.get(str(m.n_fast_runs), 0) + 1
    out["fast_runs_per_maze_hist"] = dict(sorted(hist2.items(), key=lambda kv: int(kv[0])))
    return out


# ---------------------------------------------------------------- 評価器出力との突き合わせ
def crosscheck_per_maze(mazes: list[MazeResult], cond: str = "") -> list[dict[str, Any]]:
    """各 maze JSON の `kpi` ブロックと自分の値を照合する。"""
    diffs = []
    for m in mazes:
        k = m.ev_kpi
        if not k:
            continue
        checks = [
            ("goal_reached", m.goal_reached, k.get("goal_reached")),
            ("fast_run_done", m.fast_run_done, k.get("fast_run_done")),
            ("fast_run_effective", m.fast_run_effective, k.get("fast_run_effective")),
            ("explore_time", m.explore_time, k.get("explore_time")),
            ("fast_time", m.fast_time, k.get("fast_time")),
            ("improvement_ratio", m.improvement_ratio, k.get("improvement_ratio")),
            ("first_fast_time", m.first_fast_time, k.get("first_fast_time")),
            ("first_fast_efficiency_A", m.eff_A, k.get("first_fast_efficiency")),
            ("first_fast_efficiency_B", m.eff_B, k.get("first_fast_efficiency")),
            ("best_time", m.best_time_all, m.ev_best_time),
            ("success", m.goal_reached, m.ev_success),
        ]
        for name, mine, theirs in checks:
            if mine is None and theirs is None:
                continue
            if isinstance(mine, bool) or isinstance(theirs, bool):
                same = bool(mine) == bool(theirs) and (mine is not None) == (theirs is not None)
            elif mine is None or theirs is None:
                same = False
            else:
                same = abs(float(mine) - float(theirs)) <= 1e-6 * max(1.0, abs(float(theirs)))
            if not same:
                diffs.append({
                    "condition": cond, "maze": m.maze_id, "path": m.path, "field": name,
                    "mine": mine, "evaluator": theirs,
                })
    return diffs


def compare_summary(cond_key: str, mine: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """評価器が出した summary.json と、自分の集計値を突き合わせる。

    summary.json は「突き合わせ対象」であり、集計の入力ではない。
    """
    sp = run_dir / "summary.json"
    if not sp.exists():
        return [{"condition": cond_key, "field": "(summary.json 自体)", "mine": "-",
                 "evaluator": "ファイルなし（実行未完）", "status": "MISSING"}]
    s = json.loads(sp.read_text())
    k = s.get("kpi", {})
    e_mine = mine["e_first_fast_efficiency_B"]     # 評価器の (e) 定義は解釈 B（下記 REPORT.md 参照）
    d_mine = mine["d_best_time"]
    pairs: list[tuple[str, Any, Any]] = [
        ("n_mazes", mine["n_mazes"], s.get("n_mazes")),
        ("(a) rate", mine["a_goal_rate"], k.get("a_goal_reached", {}).get("rate")),
        ("(a) n", sum(1 for _ in range(0)) or int(round((mine["a_goal_rate"] or 0) * mine["n_mazes"])),
         k.get("a_goal_reached", {}).get("n")),
        ("success_rate", mine["a_goal_rate"], s.get("success_rate")),
        ("(b) rate", mine["b_fast_done_rate"], k.get("b_fast_run_done", {}).get("rate")),
        ("(b) n", int(round((mine["b_fast_done_rate"] or 0) * mine["n_mazes"])),
         k.get("b_fast_run_done", {}).get("n")),
        ("(c) rate", mine["c_effective_rate_over_all"], k.get("c_fast_run_effective", {}).get("rate")),
        ("(c) n", int(round((mine["c_effective_rate_over_all"] or 0) * mine["n_mazes"])),
         k.get("c_fast_run_effective", {}).get("n")),
        ("(d) median", d_mine.get("median"), k.get("d_best_time", {}).get("median")),
        ("(d) mean", d_mine.get("mean"), k.get("d_best_time", {}).get("mean")),
        ("(d) min", d_mine.get("min"), k.get("d_best_time", {}).get("min")),
        ("(d) max", d_mine.get("max"), k.get("d_best_time", {}).get("max")),
        ("median_best_time", d_mine.get("median"), s.get("median_best_time")),
        ("mean_best_time", d_mine.get("mean"), s.get("mean_best_time")),
        ("explore_time median", mine["d_explore_time"].get("median"),
         k.get("explore_time", {}).get("median")),
        ("fast_time median", mine["d_fast_time"].get("median"), k.get("fast_time", {}).get("median")),
        ("(e) median", e_mine.get("median"), k.get("e_first_fast_efficiency", {}).get("median")),
        ("(e) mean", e_mine.get("mean"), k.get("e_first_fast_efficiency", {}).get("mean")),
        ("(e) min", e_mine.get("min"), k.get("e_first_fast_efficiency", {}).get("min")),
        ("(e) max", e_mine.get("max"), k.get("e_first_fast_efficiency", {}).get("max")),
        ("(e) n_defined", e_mine.get("n"), k.get("e_first_fast_efficiency", {}).get("n_defined")),
        ("(e) n_undefined", mine["e_undefined_count"],
         k.get("e_first_fast_efficiency", {}).get("n_undefined")),
    ]
    rows = []
    for name, a, b in pairs:
        if a is None and b is None:
            status = "both-None"
        elif a is None or b is None:
            status = "MISMATCH"
        else:
            status = "OK" if abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b))) else "MISMATCH"
        rows.append({"condition": cond_key, "field": name, "mine": a, "evaluator": b, "status": status})
    return rows


def main() -> None:
    global RESULTS
    if len(sys.argv) > 1:
        RESULTS = Path(sys.argv[1])
    print(f"# 入力ルート: {RESULTS}")
    conditions: dict[str, dict[str, Any]] = {}
    per_maze_dump: dict[str, list[dict[str, Any]]] = {}
    all_diffs: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []

    for band_dir in sorted(RESULTS.iterdir()):
        if not band_dir.is_dir():
            continue
        for run_dir in sorted(band_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            files = sorted(run_dir.glob("maze_*.json"))
            if not files:
                continue
            key = f"{band_dir.name}/{run_dir.name}"
            mazes = [analyze_maze(p) for p in files]
            conditions[key] = aggregate(mazes)
            conditions[key]["maze_ids"] = [m.maze_id for m in mazes]
            per_maze_dump[key] = [
                {
                    "maze_id": m.maze_id, "n_runs": m.n_runs, "outcomes": m.outcomes,
                    "n_goal_runs": m.n_goal_runs, "n_fast_runs": m.n_fast_runs,
                    "goal_reached": m.goal_reached, "fast_run_done": m.fast_run_done,
                    "fast_run_effective": m.fast_run_effective,
                    "explore_time": m.explore_time, "fast_time": m.fast_time,
                    "best_time_all": m.best_time_all, "first_fast_time": m.first_fast_time,
                    "improvement_ratio": m.improvement_ratio,
                    "eff_A": m.eff_A, "eff_B": m.eff_B,
                    "explore_over_fast": m.explore_over_fast,
                    "c1": m.c1, "c2": m.c2,
                    "time_used": m.time_used, "hit_run_cap": m.hit_run_cap,
                }
                for m in mazes
            ]
            for p in files:
                st = p.stat()
                inventory.append({"path": str(p), "bytes": st.st_size, "mtime": st.st_mtime})
            all_diffs.extend(crosscheck_per_maze(mazes, key))
            summary_rows.extend(compare_summary(key, conditions[key], run_dir))

    out = {
        "generated_by": "verification/independent_kpi.py",
        "note": "docs/RESEARCH_PLAN.md §2 の定義文のみから独立実装。評価器コード未参照。",
        "conditions": conditions,
        "per_maze": per_maze_dump,
        "per_maze_kpi_mismatches": all_diffs,
        "summary_json_crosscheck": summary_rows,
        "file_inventory_count": len(inventory),
    }
    outdir = REPO / "verification" / "out"
    outdir.mkdir(exist_ok=True)
    (outdir / "independent_kpi.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (outdir / "file_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=1))

    # 端末向けの要約
    for key, c in conditions.items():
        print(f"\n=== {key}  (n={c['n_mazes']}) ===")
        print(f"  (a) ゴール到達率        : {c['a_count']} = {c['a_goal_rate']*100:.1f}%")
        print(f"  (b) 最短走行成立率      : {c['b_count']} = {c['b_fast_done_rate']*100:.1f}%")
        print(f"  (c) 有効最短走行率 /全面: {c['c_count_over_all']} = "
              f"{(c['c_effective_rate_over_all'] or 0)*100:.1f}%   /(b)面: {c['c_count_over_b']}")
        d = c["d_best_time"]
        if d["n"]:
            print(f"  (d) 最速タイム [s]      : median {d['median']:.2f}  "
                  f"(min {d['min']:.2f} / max {d['max']:.2f}, n={d['n']})")
        de, df = c["d_explore_time"], c["d_fast_time"]
        if de["n"]:
            print(f"      探索走行 [s]        : median {de['median']:.2f} (n={de['n']})")
        if df["n"]:
            print(f"      最短走行 [s]        : median {df['median']:.2f} (n={df['n']})")
        e = c["e_first_fast_efficiency_A"]
        if e["n"]:
            print(f"  (e) 初回最短走行効率 A  : median {e['median']:.4f}  max {e['max']:.4f}  "
                  f"(n={e['n']}, 未定義 {c['e_undefined_count']} 面, =1.00 は {c['e_A_equals_one_count']} 面)")
        r = c["runs_per_maze"]
        print(f"  走行回数/迷路           : median {r['median']:.1f} (min {r['min']:.0f} / max {r['max']:.0f}) "
              f"hist={c['runs_per_maze_hist']}")
        f = c["fast_runs_per_maze"]
        print(f"  探索後に成立した走行数  : median {f['median']:.1f} (min {f['min']:.0f} / max {f['max']:.0f}) "
              f"hist={c['fast_runs_per_maze_hist']}")
        eo = c["explore_over_fast"]
        if eo["n"]:
            print(f"  探索/最短 比            : median {eo['median']:.3f}  "
                  f"Q1 {eo['q1']:.3f}  Q3 {eo['q3']:.3f}  (min {eo['min']:.3f} / max {eo['max']:.3f}, n={eo['n']})")
        print(f"  走行上限到達 {c['hit_run_cap_count']} 面 / 最終走行が timeout {c['last_run_timeout_count']} 面 / "
              f"使用時間 median {c['time_used']['median']:.1f} s")

    print("\n=== summary.json との突き合わせ ===")
    bad = [r for r in summary_rows if r["status"] not in ("OK", "both-None")]
    ok_n = sum(1 for r in summary_rows if r["status"] == "OK")
    print(f"  照合項目 {len(summary_rows)} 件中 一致 {ok_n} 件 / 不一致・欠落 {len(bad)} 件")
    for r in bad:
        print(f"  [{r['status']}] {r['condition']:<48} {r['field']:>20}  "
              f"mine={r['mine']}  evaluator={r['evaluator']}")

    print("\n=== 改訂 §2（b2280dd）に基づく新定義 ===")
    for key, c in conditions.items():
        e, sr = c["e_v2"], c["c_shrink_rate"]
        print(f"  {key}  (n={c['n_mazes']})")
        if e["n"]:
            print(f"    (e) 改訂版: 中央値 {e['median']:.4f}  [Q1 {e['q1']:.4f}, Q3 {e['q3']:.4f}]  "
                  f"max {e['max']:.4f}  定義された面 {c['e_v2_defined_count']}/{c['n_mazes']}")
        else:
            print(f"    (e) 改訂版: 定義される面が 0（全面が未定義）")
        print(f"        未定義の内訳: 最短走行なし {c['e_v2_undefined_no_fast_run']} 面 / "
              f"退化（1 回のみ＆最良がその走行） {c['e_v2_undefined_degenerate']} 面 "
              f"／ 1 回のみだが情報を持つ面 {c['e_v2_single_fast_but_informative']} 面")
        if sr["n"]:
            print(f"    (c) 時間短縮率 1−t_最短/t_探索: 中央値 {sr['median']*100:6.2f}%  "
                  f"[Q1 {sr['q1']*100:.2f}%, Q3 {sr['q3']*100:.2f}%]  "
                  f"min {sr['min']*100:.2f}%  max {sr['max']*100:.2f}%  n={sr['n']}"
                  f"   ／二値化率 {c['c_effective_rate_over_all']*100:.0f}%")
        c1, c2 = c["c1_path_shrink"], c["c2_speed_gain"]
        if c1["n"]:
            print(f"    **(c1) 経路短縮率** : 中央値 {c1['median']*100:6.2f}%  "
                  f"[Q1 {c1['q1']*100:.2f}%, Q3 {c1['q3']*100:.2f}%]  "
                  f"min {c1['min']*100:.2f}%  max {c1['max']*100:.2f}%")
            print(f"    **(c2) 速度向上率** : 中央値 {c2['median']*100:6.2f}%  "
                  f"[Q1 {c2['q1']*100:.2f}%, Q3 {c2['q3']*100:.2f}%]  "
                  f"min {c2['min']*100:.2f}%  max {c2['max']*100:.2f}%")

    print("\n--- (c) の分解: 時間短縮は「経路が短くなった」からか「速く走った」からか ---")
    print("      t_exp/t_fast = (L_exp/L_fast) × (v_fast/v_exp)")
    for key, c in conditions.items():
        pr, sr, eo = c["path_ratio"], c["speed_ratio"], c["explore_over_fast"]
        if not pr["n"]:
            continue
        print(f"  {key}")
        print(f"    時間比 t_exp/t_fast : median {eo['median']:.3f}")
        print(f"    ├ 経路比 L_exp/L_fast: median {pr['median']:.3f}  (Q1 {pr['q1']:.3f} / Q3 {pr['q3']:.3f})  ← 地図の寄与")
        print(f"    └ 速度比 v_fast/v_exp: median {sr['median']:.3f}  (Q1 {sr['q1']:.3f} / Q3 {sr['q3']:.3f})  ← 走行方式の寄与")

    print("\n--- (c) 閾値の感度 / (e) の構造的固定 ---")
    for key, c in conditions.items():
        print(f"  {key}")
        print(f"    (c) 閾値別の該当面数 (分母 n={c['n_mazes']}): {c['c_threshold_sensitivity']}"
              f"  ／10%±5pp に入る面: {c['c_near_threshold_count']}")
        print(f"    (e) 最短走行が1回のみ＝比が恒等的に1.00になる面: "
              f"{c['e_forced_one_count']} / (b)該当 {c['b_count'].split('/')[0]} 面")

    print(f"\n--- 各 maze JSON の kpi ブロックとの不一致: {len(all_diffs)} 件 ---")
    by_field: dict[str, int] = {}
    for d in all_diffs:
        by_field[d["field"]] = by_field.get(d["field"], 0) + 1
    print(f"  項目別: {by_field}")
    for d in all_diffs[:80]:
        print(f"  {d['condition']:<48} {d['maze']:>22} {d['field']:>24}  "
              f"mine={d['mine']}  evaluator={d['evaluator']}")
    if len(all_diffs) > 80:
        print(f"  ... 他 {len(all_diffs)-80} 件")


if __name__ == "__main__":
    main()
