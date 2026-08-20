"""
experiments/exp_026_profile_run/judge.py
================
exp_026（最短走行のプロファイル追従化）の一次記録から η を計算する
（判定条文は `research_notes/note_031_profile_planner_and_eta.md` を参照。
本スクリプトはそこに確定済みの式をそのまま実装するだけで、新設・変更しない）。

    η       = T_ideal / T_measured
    η_track = t_plan   / T_measured
    η_map   = T_ideal / t_plan

- `T_ideal` は `experiments/exp_025_s4_slalom/ideal_table.json` の
  `rows[].t_ideal_slalom`（版管理下・既に計算済み。作り直さない）。
- `T_measured` は `outputs/exp_026/profile/maze_<seed>.json` の一次記録
  （`runs[]`/`run_phases`）から、「走行が始まった瞬間の段階が FAST であり、
  かつゴールした走行」のタイムの最小値として独立に計算する
  （`experiments/exp_024_s3_fast_run/recompute_anchors.py::compute_t_fast`
  と同じ定義）。評価器の `best_time`/`maze_kpi()` は使わない。
- `t_plan` は一次記録に残らない（PREREG §5）ため、`profile` 条件の一次記録に
  書き足された学習地図（`maze_v_walls`/`maze_h_walls`。読み取り専用の後処理で
  保存したもの）に対して `classic.fast_planner.plan_fast_run()` を呼び直して
  独立に再計算する。

否定対照 N1/N2（`plan_ids` に "profile" を含む/含まない）もここで検査する。

使い方:
    .venv/bin/python experiments/exp_026_profile_run/judge.py
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

OUT_ROOT = REPO_ROOT / "outputs" / "exp_026"
MANIFEST_PATH = REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "manifest.json"
IDEAL_TABLE_PATH = REPO_ROOT / "experiments" / "exp_025_s4_slalom" / "ideal_table.json"

CONDITION_COMMAND = "command"
CONDITION_PROFILE = "profile"


# ==========================================================================
# T_fast の再構築（exp_024 recompute_anchors.py::compute_t_fast と同じ定義）
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
# t_plan の独立再計算（PREREG §5）
# ==========================================================================
def compute_t_plan_from_saved_map(record: dict) -> Optional[float]:
    """`profile` 条件の一次記録に書き足された学習地図から、
    `classic.fast_planner.plan_fast_run()` を呼び直して `t_plan` を得る。

    地図が記録されていない（`maze_v_walls`/`maze_h_walls` が無い。measured 側が
    そもそも profile 条件でない、または測定時点でまだ実装していなかった等）
    場合は None を返す。"""
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
    plan = plan_fast_run(maze, start=(0, 0), goals=goals, start_heading=Direction.N)
    return None if plan is None else float(plan.t_plan)


# ==========================================================================
# 否定対照 N1/N2（plan_ids に "profile" を含む/含まない）
# ==========================================================================
def check_plan_id_vocabulary(record: dict, condition: str) -> str:
    """N1（`profile` 条件で "profile" を含む plan_id が現れる）・
    N2（`command` 条件で一切現れない）を1迷路ぶん検査し、結果を文字列で返す。"""
    plan_ids = record.get("plan_ids", [])
    has_profile = any("profile" in p for p in plan_ids)
    if condition == CONDITION_PROFILE:
        return "OK(作動)" if has_profile else "NG(profileのplan_idが1件も無い)"
    return "OK(空振り)" if not has_profile else "NG(commandなのにprofileのplan_idが混入)"


# ==========================================================================
# 迷路ごとの錨
# ==========================================================================
@dataclass
class MazeRow:
    seed: int
    d0: Optional[int]
    t_ideal: Optional[float] = None
    t_plan: Optional[float] = None
    t_measured: Optional[float] = None          # profile 条件の T_fast
    t_measured_command: Optional[float] = None   # command 条件の T_fast（参考）
    outcomes_profile: List[str] = field(default_factory=list)
    outcomes_command: List[str] = field(default_factory=list)
    n1_n2: str = "—"
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


def build_rows(out_root: Path = OUT_ROOT, manifest_path: Path = MANIFEST_PATH,
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

        cmd_path = out_root / CONDITION_COMMAND / f"maze_{seed}.json"
        prof_path = out_root / CONDITION_PROFILE / f"maze_{seed}.json"
        cmd_record = _load_json(cmd_path)
        prof_record = _load_json(prof_path)
        if cmd_record is None:
            missing.append(str(cmd_path))
        if prof_record is None:
            missing.append(str(prof_path))

        if cmd_record is not None:
            row.t_measured_command = compute_t_fast(cmd_record.get("runs", []), cmd_record.get("run_phases", []))
            row.outcomes_command = [r.get("outcome") for r in cmd_record.get("runs", [])]

        if prof_record is not None:
            row.t_measured = compute_t_fast(prof_record.get("runs", []), prof_record.get("run_phases", []))
            row.outcomes_profile = [r.get("outcome") for r in prof_record.get("runs", [])]
            if row.t_measured is None:
                row.note += "profile条件でFASTがゴールした走行が無い;"
            row.t_plan = compute_t_plan_from_saved_map(prof_record)
            if row.t_plan is None:
                row.note += "t_planを計算できない(学習地図が悲観到達不能、または未記録);"

            n1 = check_plan_id_vocabulary(prof_record, CONDITION_PROFILE)
            n2 = check_plan_id_vocabulary(cmd_record, CONDITION_COMMAND) if cmd_record is not None else "—(command欠測)"
            row.n1_n2 = f"N1:{n1} N2:{n2}"

        rows.append(row)

    if missing:
        print("[警告] 一次記録ファイルが見つからない（未測定、または測定途中）:")
        for p in missing:
            print(f"  - {p}")
        print()

    return rows


# ==========================================================================
# 印字
# ==========================================================================
def _fmt(t: Optional[float], nd: int = 3) -> str:
    return f"{t:.{nd}f}" if t is not None else "—"


def print_report(rows: List[MazeRow]) -> None:
    header = (f"  {'seed':>6} {'D0':>4} {'T_ideal':>9} {'t_plan':>9} {'T_measured':>10} "
              f"{'eta':>7} {'eta_track':>9} {'eta_map':>8}  outcome(profile)")
    print("迷路ごとの η（T_ideal は experiments/exp_025_s4_slalom/ideal_table.json の t_ideal_slalom）:")
    print(header)
    for r in rows:
        print(f"  {r.seed:>6} {r.d0:>4} {_fmt(r.t_ideal):>9} {_fmt(r.t_plan):>9} {_fmt(r.t_measured):>10} "
              f"{_fmt(r.eta,4):>7} {_fmt(r.eta_track,4):>9} {_fmt(r.eta_map,4):>8}  {r.outcomes_profile}")
        if r.note:
            print(f"         注記: {r.note}")
        print(f"         {r.n1_n2}")

    etas = [r.eta for r in rows if r.eta is not None]
    print()
    if etas:
        print(f"η の中央値 = {statistics.median(etas):.4f}  最小値 = {min(etas):.4f}  (n={len(etas)}/{len(rows)})")
    else:
        print("η を計算できた迷路が1つも無い。")

    print()
    print("command条件のT_fast（参考。commandとprofileの探索・帰還が同一かどうかの目安）:")
    for r in rows:
        print(f"  seed={r.seed}: T_fast(command)={_fmt(r.t_measured_command)}  "
              f"T_fast(profile)={_fmt(r.t_measured)}  outcomes(command)={r.outcomes_command}")


def main() -> int:
    rows = build_rows()
    print_report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
