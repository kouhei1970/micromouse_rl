"""
experiments/exp_028_margin_sweep/judge.py
================
exp_028（`clearance_margin_m` の掃引: margin 5/10/15/20/25/30mm × u 0.30/0.50 ×
北向き/南向き開始）の一次記録から、水準ごとに

    η       = T_ideal / T_measured    （完走した迷路だけで定義される）
    η_map   = T_ideal / t_plan        （margin を上げたときの「η の上限」。
                                         `摩擦円の使用率 u` の η_map と同じ意味）

を計算し、①完走した迷路数 ②到達距離(path_length_m)の中央値 ③完走した迷路の η
を条件（margin, u, 開始方位）ごとに表にする。

- `T_ideal` は `experiments/exp_025_s4_slalom/ideal_table.json` の
  `rows[].t_ideal_slalom`（margin=0.005 固定・触っていない。分母は動かさない）。
- `t_plan` は `run_exp028.py::run_single_attempt` が測定時に `_begin_fast_run`
  の結果（`ex._fast_plan.t_plan`）をそのまま一次記録へ書き足したものを読む
  （🔴 呼び直さない。margin=25/30mmのような幾何探索が重い条件では1回の
  再計算に数十秒かかり、240条件を呼び直すと現実的な時間に収まらないため）。

使い方:
    .venv/bin/python experiments/exp_028_margin_sweep/judge.py
"""
from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_ROOT = REPO_ROOT / "outputs" / "exp_028"
MANIFEST_PATH = REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "manifest.json"
IDEAL_TABLE_PATH = REPO_ROOT / "experiments" / "exp_025_s4_slalom" / "ideal_table.json"

MARGIN_LEVELS_MM: List[float] = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
U_LEVELS: List[float] = [0.30, 0.50]
HEADINGS = ("north", "south")


def _margin_dirname(margin_mm: float) -> str:
    return f"margin_{margin_mm:04.1f}mm"


def _u_dirname(u: float) -> str:
    return f"u_{u:.2f}"


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_ideal_table(path: Path = IDEAL_TABLE_PATH) -> Dict[int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["seed"]): float(row["t_ideal_slalom"]) for row in data["rows"]}


def _target_mazes(manifest_path: Path = MANIFEST_PATH) -> List[Dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ordered = sorted(manifest["mazes"], key=lambda m: int(m["seed"]))
    return [{"seed": int(m["seed"]), "d0": int(m["d0"])} for m in ordered]


# ==========================================================================
# 迷路ごとの錨
# ==========================================================================
@dataclass
class Row:
    seed: int
    heading: str
    t_ideal: Optional[float] = None
    t_plan: Optional[float] = None
    t_measured: Optional[float] = None
    outcome: Optional[str] = None
    path_length_m: Optional[float] = None
    note: str = ""

    @property
    def eta(self) -> Optional[float]:
        if self.t_ideal is None or self.t_measured is None or self.t_measured <= 0:
            return None
        return self.t_ideal / self.t_measured

    @property
    def eta_map(self) -> Optional[float]:
        if self.t_ideal is None or self.t_plan is None or self.t_plan <= 0:
            return None
        return self.t_ideal / self.t_plan


def build_rows(margin_mm: float, u: float, heading: str,
                out_root: Path = OUT_ROOT) -> List[Row]:
    targets = _target_mazes()
    ideal_table = load_ideal_table()

    rows: List[Row] = []
    missing: List[str] = []
    for m in targets:
        seed = m["seed"]
        row = Row(seed=seed, heading=heading, t_ideal=ideal_table.get(seed))

        path = out_root / _margin_dirname(margin_mm) / _u_dirname(u) / f"maze_{seed}.json"
        record = _load_json(path)
        if record is None:
            missing.append(str(path))
            rows.append(row)
            continue

        leg = record.get(heading, {})
        row.outcome = leg.get("outcome")
        row.path_length_m = leg.get("path_length_m")
        row.t_measured = leg.get("run_time") if row.outcome == "goal" else None
        if row.t_measured is None:
            row.note += f"未完走(outcome={row.outcome});"

        row.t_plan = leg.get("t_plan")
        if row.t_plan is None:
            row.note += "t_planが記録されていない(plan_failed);"

        rows.append(row)

    if missing:
        print(f"[警告] margin={margin_mm}mm u={u:.2f} {heading}: 一次記録ファイルが見つからない:")
        for p in missing:
            print(f"  - {p}")

    return rows


@dataclass
class LevelSummary:
    margin_mm: float
    u: float
    heading: str
    rows: List[Row]

    @property
    def n_completed(self) -> int:
        return sum(1 for r in self.rows if r.outcome == "goal")

    @property
    def n_total(self) -> int:
        return len(self.rows)

    @property
    def reach_distances(self) -> List[float]:
        return [r.path_length_m for r in self.rows if r.path_length_m is not None]

    @property
    def etas(self) -> List[float]:
        return [r.eta for r in self.rows if r.eta is not None]

    @property
    def eta_maps(self) -> List[float]:
        return [r.eta_map for r in self.rows if r.eta_map is not None]


def _median(xs: List[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


def _fmt(x: Optional[float], nd: int = 3) -> str:
    return f"{x:.{nd}f}" if x is not None else "-"


def print_level_table(summaries: List[LevelSummary]) -> None:
    print("水準ごとの集計（margin[mm] × u × 開始方位 -> 完走数・到達距離中央値[m]・"
          "η中央値(完走のみ)・η_map中央値(=η の上限)）:")
    header = (f"  {'margin':>7} {'u':>5} {'heading':>7} {'n_goal':>7} "
              f"{'reach_med[m]':>13} {'eta_med':>8} {'eta_map_med':>12}")
    print(header)
    for s in summaries:
        print(f"  {s.margin_mm:>5.1f}mm {s.u:>5.2f} {s.heading:>7} "
              f"{s.n_completed:>2}/{s.n_total:<4} "
              f"{_fmt(_median(s.reach_distances)):>13} {_fmt(_median(s.etas)):>8} "
              f"{_fmt(_median(s.eta_maps)):>12}")


def print_completions(summaries: List[LevelSummary]) -> None:
    print("\n[本実験の主要な問い] 完走した迷路・条件（1本でも出たら本実験の最大の成果）:")
    any_completion = False
    for s in summaries:
        for r in s.rows:
            if r.outcome == "goal":
                any_completion = True
                print(f"  seed={r.seed} margin={s.margin_mm:.1f}mm u={s.u:.2f} "
                      f"heading={s.heading}: t_measured={_fmt(r.t_measured)}s "
                      f"t_ideal={_fmt(r.t_ideal)}s eta={_fmt(r.eta)}")
    if not any_completion:
        print("  無し（掃引した範囲すべてで衝突・その他の未完走）")


def print_maze_detail(summaries: List[LevelSummary]) -> None:
    print("\n迷路ごとの内訳:")
    for s in summaries:
        print(f"  --- margin={s.margin_mm:.1f}mm u={s.u:.2f} heading={s.heading} ---")
        for r in s.rows:
            print(f"    seed={r.seed:>5} outcome={str(r.outcome):>10} "
                  f"reach={_fmt(r.path_length_m):>8}m t_ideal={_fmt(r.t_ideal):>8} "
                  f"t_plan={_fmt(r.t_plan):>8} t_measured={_fmt(r.t_measured):>8} "
                  f"eta={_fmt(r.eta):>7} eta_map={_fmt(r.eta_map):>7}"
                  + (f"  注記: {r.note}" if r.note else ""))


def check_monotonicity(all_rows: Dict[tuple, List[Row]]) -> None:
    """期待: margin を上げると t_plan は単調に伸びる（=eta_mapは単調に下がる）
    ＝ friction_use と同じ構造。⚠️ これまでの掃引で隣接水準の反転が繰り返し
    出ているため、合わせ込まず違反があればそのまま件数を報告する。"""
    print("\n[単調性] 同じ迷路・u・開始方位で margin を上げるほど t_plan が伸びること:")
    for u in U_LEVELS:
        for heading in HEADINGS:
            violations = 0
            checked = 0
            margins_sorted = sorted(MARGIN_LEVELS_MM)
            for m in _target_mazes():
                seed = m["seed"]
                t_plans = []
                for margin_mm in margins_sorted:
                    rows = all_rows.get((margin_mm, u, heading), [])
                    row = next((r for r in rows if r.seed == seed), None)
                    t_plans.append(row.t_plan if row is not None else None)
                pairs = [(a, b) for a, b in zip(t_plans, t_plans[1:]) if a is not None and b is not None]
                for a, b in pairs:
                    checked += 1
                    if b < a - 1e-9:
                        violations += 1
            print(f"  u={u:.2f} heading={heading}: 検査した隣接ペア={checked}  違反={violations}")


def main() -> int:
    summaries: List[LevelSummary] = []
    all_rows: Dict[tuple, List[Row]] = {}
    for margin_mm in MARGIN_LEVELS_MM:
        for u in U_LEVELS:
            for heading in HEADINGS:
                rows = build_rows(margin_mm, u, heading)
                all_rows[(margin_mm, u, heading)] = rows
                summaries.append(LevelSummary(margin_mm=margin_mm, u=u, heading=heading, rows=rows))

    print_level_table(summaries)
    print_completions(summaries)
    print_maze_detail(summaries)
    check_monotonicity(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
