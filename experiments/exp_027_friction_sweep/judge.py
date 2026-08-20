"""
experiments/exp_027_friction_sweep/judge.py
================
exp_027（摩擦円の使用率 u の掃引と、壁センサ補正の有無）の一次記録から、
水準（`u`, `wall_correction`）ごとに η・η_track・η_map・衝突した迷路数を計算し、
「壊れずに走り切れた最大の u」を条件ごとに印字する
（判定条文は `research_notes/note_031_profile_planner_and_eta.md` を参照。
「壊れずに走り切れた」の定義は本実験の `PREREG.md` §6 に確定済み）。

    η       = T_ideal / T_measured
    η_track = t_plan   / T_measured
    η_map   = T_ideal / t_plan

- `T_ideal` は `experiments/exp_025_s4_slalom/ideal_table.json` の
  `rows[].t_ideal_slalom`（`u=1.00` の物理限界。作り直さない）。
- `T_measured` は `outputs/exp_027/u_<u:.2f>/wc_<off|on>/maze_<seed>.json` の
  一次記録（`runs[]`/`run_phases`）から、「走行が始まった瞬間の段階が FAST
  であり、かつゴールした走行」のタイムの最小値として独立に計算する
  （`experiments/exp_026_profile_run/judge.py::compute_t_fast` と同じ定義）。
- `t_plan` は一次記録に残らない（PREREG §5）ため、一次記録に書き足された
  学習地図に対して `classic.fast_planner.plan_fast_run(..., friction_use=u)`
  を、**その走行が実際に使った `u` と同じ値で**呼び直して独立に再計算する。

判定条文以外に、本スクリプトは PREREG §7・§8 の否定対照・単調性も検査する。

使い方:
    .venv/bin/python experiments/exp_027_friction_sweep/judge.py
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

OUT_ROOT = REPO_ROOT / "outputs" / "exp_027"
EXP026_PROFILE_ROOT = REPO_ROOT / "outputs" / "exp_026" / "profile"
MANIFEST_PATH = REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "manifest.json"
IDEAL_TABLE_PATH = REPO_ROOT / "experiments" / "exp_025_s4_slalom" / "ideal_table.json"

U_LEVELS: List[float] = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
WALL_CORRECTION_LEVELS: List[bool] = [False, True]


def _u_dirname(u: float) -> str:
    return f"u_{u:.2f}"


def _wc_dirname(wc: bool) -> str:
    return "wc_on" if wc else "wc_off"


# ==========================================================================
# T_fast の再構築（exp_026 judge.py::compute_t_fast と同じ定義）
# ==========================================================================
def compute_t_fast(runs: List[dict], run_phases: List[dict]) -> Optional[float]:
    """T_fast = 「走行が始まった瞬間の段階が FAST であり、かつゴールした
    走行」のタイム（run_time）の最小値。該当が無ければ None。"""
    phase_by_run_index = {p["run_index"]: p["phase"] for p in run_phases}
    fast_goal_times = [
        float(r["run_time"])
        for r in runs
        if r.get("outcome") == "goal"
        and r.get("run_time") is not None
        and phase_by_run_index.get(r["index"]) == "FAST"
    ]
    if not fast_goal_times:
        return None
    return min(fast_goal_times)


# ==========================================================================
# t_plan の独立再計算（PREREG §5。走行が実際に使った u と同じ値で呼び直す）
# ==========================================================================
def compute_t_plan_from_saved_map(record: dict, friction_use: float) -> Optional[float]:
    """一次記録に書き足された学習地図から、`classic.fast_planner.
    plan_fast_run()` を `friction_use` 付きで呼び直して `t_plan` を得る。

    地図が記録されていない場合は None を返す。"""
    if "maze_v_walls" not in record or "maze_h_walls" not in record:
        return None

    import numpy as np

    from classic.fast_planner import plan_fast_run
    from classic.maze_map import Direction, MazeMap
    from competition.evaluator import goal_cells

    v = np.asarray(record["maze_v_walls"], dtype=np.int8)
    h = np.asarray(record["maze_h_walls"], dtype=np.int8)
    width = int(v.shape[0] - 1)
    height = int(v.shape[1])

    maze = MazeMap(width, height)
    maze.v_walls[:, :] = v
    maze.h_walls[:, :] = h

    goals = goal_cells(width, height)
    plan = plan_fast_run(maze, start=(0, 0), goals=goals, start_heading=Direction.N,
                          friction_use=friction_use)
    return None if plan is None else float(plan.t_plan)


# ==========================================================================
# 迷路ごとの錨
# ==========================================================================
@dataclass
class MazeRow:
    seed: int
    d0: Optional[int]
    t_ideal: Optional[float] = None
    t_plan: Optional[float] = None
    t_measured: Optional[float] = None
    outcomes: List[str] = field(default_factory=list)
    plan_ids: List[str] = field(default_factory=list)
    note: str = ""

    @property
    def eta(self) -> Optional[float]:
        if self.t_ideal is None or self.t_measured is None or self.t_measured <= 0:
            return None
        return self.t_ideal / self.t_measured

    @property
    def eta_track(self) -> Optional[float]:
        if self.t_plan is None or self.t_measured is None or self.t_measured <= 0:
            return None
        return self.t_plan / self.t_measured

    @property
    def eta_map(self) -> Optional[float]:
        if self.t_ideal is None or self.t_plan is None or self.t_plan <= 0:
            return None
        return self.t_ideal / self.t_plan


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_ideal_table(path: Path = IDEAL_TABLE_PATH) -> Dict[int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["seed"]): float(row["t_ideal_slalom"]) for row in data["rows"]}


def _target_mazes_from_manifest(manifest_path: Path = MANIFEST_PATH) -> List[Dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ordered = sorted(manifest["mazes"], key=lambda m: int(m["seed"]))
    return [{"seed": int(m["seed"]), "d0": int(m["d0"])} for m in ordered]


def build_rows(u: float, wc: bool, out_root: Path = OUT_ROOT,
               manifest_path: Path = MANIFEST_PATH,
               ideal_table_path: Path = IDEAL_TABLE_PATH) -> List[MazeRow]:
    targets = _target_mazes_from_manifest(manifest_path)
    ideal_table = load_ideal_table(ideal_table_path)

    rows: List[MazeRow] = []
    missing: List[str] = []
    for m in targets:
        seed, d0 = m["seed"], m["d0"]
        row = MazeRow(seed=seed, d0=d0, t_ideal=ideal_table.get(seed))
        if row.t_ideal is None:
            row.note += "T_idealがideal_table.jsonに無い;"

        path = out_root / _u_dirname(u) / _wc_dirname(wc) / f"maze_{seed}.json"
        record = _load_json(path)
        if record is None:
            missing.append(str(path))
            rows.append(row)
            continue

        row.t_measured = compute_t_fast(record.get("runs", []), record.get("run_phases", []))
        row.outcomes = [r.get("outcome") for r in record.get("runs", [])]
        row.plan_ids = record.get("plan_ids", [])
        if row.t_measured is None:
            row.note += "FASTがゴールした走行が無い(衝突/失敗);"
        row.t_plan = compute_t_plan_from_saved_map(record, friction_use=u)
        if row.t_plan is None:
            row.note += "t_planを計算できない(学習地図が悲観到達不能、または未記録);"

        rows.append(row)

    if missing:
        print(f"[警告] u={u:.2f} wc={_wc_dirname(wc)}: 一次記録ファイルが見つからない"
              "（未測定、または測定途中）:")
        for p in missing:
            print(f"  - {p}")

    return rows


# ==========================================================================
# 水準ごとの集計
# ==========================================================================
@dataclass
class LevelSummary:
    u: float
    wc: bool
    rows: List[MazeRow]

    @property
    def n_collided(self) -> int:
        """PREREG §6: ηが計算できなかった迷路の数。"""
        return sum(1 for r in self.rows if r.eta is None)

    @property
    def n_total(self) -> int:
        return len(self.rows)

    @property
    def etas(self) -> List[float]:
        return [r.eta for r in self.rows if r.eta is not None]

    @property
    def eta_tracks(self) -> List[float]:
        return [r.eta_track for r in self.rows if r.eta_track is not None]

    @property
    def eta_maps(self) -> List[float]:
        return [r.eta_map for r in self.rows if r.eta_map is not None]

    @property
    def is_unbroken(self) -> bool:
        """PREREG §6: 「壊れずに走り切れた」= 10迷路すべてでηが計算できた。"""
        return self.n_collided == 0 and self.n_total > 0


def _median(xs: List[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


def _fmt(x: Optional[float], nd: int = 4) -> str:
    return f"{x:.{nd}f}" if x is not None else "—"


# ==========================================================================
# PREREG §7: 否定対照
# ==========================================================================
def check_regression_against_exp026(rows_u100_off: List[MazeRow]) -> None:
    """対照1: u=1.00・wall_correction=False が exp_026 の profile 条件と
    完全に同一であること（plan_ids・runs[].outcome・runs[].run_time を突き合わせる）。"""
    print("\n[否定対照1: 回帰] u=1.00・wall_correction=False は exp_026(profile)と同一であるべき:")
    all_ok = True
    for row in rows_u100_off:
        exp026_path = EXP026_PROFILE_ROOT / f"maze_{row.seed}.json"
        exp026_record = _load_json(exp026_path)
        exp027_path = OUT_ROOT / _u_dirname(1.00) / _wc_dirname(False) / f"maze_{row.seed}.json"
        exp027_record = _load_json(exp027_path)
        if exp026_record is None or exp027_record is None:
            print(f"  seed={row.seed}: —(未測定。比較不能)")
            continue

        exp026_outcomes = [(r.get("outcome"), r.get("run_time")) for r in exp026_record.get("runs", [])]
        exp027_outcomes = [(r.get("outcome"), r.get("run_time")) for r in exp027_record.get("runs", [])]
        exp026_plan_ids = exp026_record.get("plan_ids", [])
        exp027_plan_ids = exp027_record.get("plan_ids", [])

        same_outcomes = exp026_outcomes == exp027_outcomes
        same_plan_ids = exp026_plan_ids == exp027_plan_ids
        ok = same_outcomes and same_plan_ids
        all_ok = all_ok and ok
        print(f"  seed={row.seed}: {'OK' if ok else 'NG'}"
              f"(outcomes一致={same_outcomes}, plan_ids一致={same_plan_ids})")
    print(f"  -> 総合: {'OK(回帰なし)' if all_ok else 'NG(exp_026から挙動が変わった)'}")


def check_wall_correction_pair(u: float, rows_off: List[MazeRow], rows_on: List[MazeRow]) -> None:
    """対照2: 同じuで wall_correction=True/False を比べ、True側は False側と
    異なること（False側はexp_026と一致する。対照1で別途確認済み）。"""
    by_seed_off = {r.seed: r for r in rows_off}
    diffs = 0
    total = 0
    for row_on in rows_on:
        row_off = by_seed_off.get(row_on.seed)
        if row_off is None:
            continue
        total += 1
        if row_on.plan_ids != row_off.plan_ids or row_on.outcomes != row_off.outcomes:
            diffs += 1
    print(f"  u={u:.2f}: wall_correction=True が False と異なる迷路 = {diffs}/{total}")


# ==========================================================================
# PREREG §8: 単調性（uを下げるとt_planが伸びる）
# ==========================================================================
def check_monotonicity(levels: Dict[tuple, List[MazeRow]]) -> None:
    print("\n[単調性] 同じ学習地図で u を下げるほど t_plan が伸びること:")
    for wc in WALL_CORRECTION_LEVELS:
        violations = 0
        checked = 0
        u_sorted = sorted(U_LEVELS)  # 昇順
        for seed_idx in range(10):
            t_plans = []
            for u in u_sorted:
                rows = levels.get((u, wc))
                if rows is None or seed_idx >= len(rows):
                    t_plans.append(None)
                    continue
                t_plans.append(rows[seed_idx].t_plan)
            pairs = [(a, b) for a, b in zip(t_plans, t_plans[1:]) if a is not None and b is not None]
            for a, b in pairs:
                checked += 1
                if b < a - 1e-9:  # u昇順でt_planは非増加のはず(次のuでは増えない)
                    violations += 1
        print(f"  wall_correction={_wc_dirname(wc)}: 検査した隣接ペア={checked}  違反={violations}")


# ==========================================================================
# 印字
# ==========================================================================
def print_level_table(summaries: List[LevelSummary]) -> None:
    header = (f"  {'u':>5} {'wc':>6} {'n_collided':>10} {'eta_median':>11} {'eta_min':>8} "
              f"{'eta_track_med':>14} {'eta_map_med':>12}")
    print("水準ごとの集計（η_medianは算出できた迷路のみの中央値）:")
    print(header)
    for s in summaries:
        etas = s.etas
        eta_median = _median(etas)
        eta_min = min(etas) if etas else None
        eta_track_median = _median(s.eta_tracks)
        eta_map_median = _median(s.eta_maps)
        print(f"  {s.u:>5.2f} {_wc_dirname(s.wc):>6} {s.n_collided:>3}/{s.n_total:<6} "
              f"{_fmt(eta_median):>11} {_fmt(eta_min):>8} {_fmt(eta_track_median):>14} "
              f"{_fmt(eta_map_median):>12}")


def print_maze_detail(summaries: List[LevelSummary]) -> None:
    print("\n迷路ごとの内訳:")
    for s in summaries:
        print(f"  --- u={s.u:.2f} wall_correction={_wc_dirname(s.wc)} ---")
        for r in s.rows:
            print(f"    seed={r.seed:>5} D0={r.d0:>3} T_ideal={_fmt(r.t_ideal,3):>8} "
                  f"t_plan={_fmt(r.t_plan,3):>8} T_measured={_fmt(r.t_measured,3):>9} "
                  f"eta={_fmt(r.eta):>7} outcomes={r.outcomes}"
                  + (f"  注記: {r.note}" if r.note else ""))


def print_max_unbroken_u(summaries: List[LevelSummary]) -> None:
    print("\n[本実験の主要な問い] 壊れずに走り切れた（10迷路すべてでηが計算できた）最大のu:")
    for wc in WALL_CORRECTION_LEVELS:
        candidates = sorted(
            (s.u for s in summaries if s.wc == wc and s.is_unbroken), reverse=True
        )
        best = candidates[0] if candidates else None
        print(f"  wall_correction={_wc_dirname(wc)}: "
              f"{f'u={best:.2f}' if best is not None else '無し（掃引した水準すべてで壊れた）'}")


def main() -> int:
    summaries: List[LevelSummary] = []
    levels: Dict[tuple, List[MazeRow]] = {}
    for u in U_LEVELS:
        for wc in WALL_CORRECTION_LEVELS:
            rows = build_rows(u, wc)
            levels[(u, wc)] = rows
            summaries.append(LevelSummary(u=u, wc=wc, rows=rows))

    print_level_table(summaries)
    print_maze_detail(summaries)
    print_max_unbroken_u(summaries)

    rows_u100_off = levels.get((1.00, False), [])
    check_regression_against_exp026(rows_u100_off)

    print("\n[否定対照2: 壁センサ補正の対] 同じuで wall_correction=True/False の走行が異なること:")
    for u in U_LEVELS:
        check_wall_correction_pair(u, levels.get((u, False), []), levels.get((u, True), []))

    check_monotonicity(levels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
