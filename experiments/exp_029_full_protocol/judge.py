"""
experiments/exp_029_full_protocol/judge.py
================
exp_029（競技の全手順を通してηを測る）の一次記録から、判定条文
（`research_notes/note_031_profile_planner_and_eta.md` の「判定条文」節。
本ファイルは写さず参照するだけ）に従って迷路ごとに

    T_ideal    = experiments/exp_025_s4_slalom/ideal_table.json の t_ideal_slalom
                 （余裕5mm・u=1.00固定。分母は動かさない）
    T_plan     = マウス自身が学習した地図から plan_fast_run が計算した理論時間
                 （run_exp029.py が on_run_start の瞬間に記録した
                 fast_plan_by_run[run_index]。T_measured を出した run が
                 FAST計画そのものならその値、そうでなければ最初に計画された
                 値にフォールバックする — 詳細は _pick_t_plan の docstring）
    T_measured = その迷路で成立した最良の走行の時間（evaluate_maze の
                 best_time。探索走行がゴールに達した場合はその時間も含む
                 — 実競技と同じ規約）
    η          = T_ideal / T_measured（🔴 1本も成立しなければ η=0。判定条文§5-0）
    η_map      = T_ideal / T_plan
    η_track    = T_plan   / T_measured

を計算し、持ち時間420秒/1500秒それぞれで表にする。exp_028（真の地図を直接
最短走行させた測定。余裕25mm・u=0.30・北向き開始）の実測 η とも突き合わせ、
差が η_map にどれだけ現れているかを見る。

使い方:
    .venv/bin/python experiments/exp_029_full_protocol/judge.py
"""
from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_ROOT = REPO_ROOT / "outputs" / "exp_029"
MANIFEST_PATH = REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "manifest.json"
IDEAL_TABLE_PATH = REPO_ROOT / "experiments" / "exp_025_s4_slalom" / "ideal_table.json"
EXP028_ROOT = REPO_ROOT / "outputs" / "exp_028"
EXP028_MARGIN_MM = 25.0
EXP028_U = 0.30

TIME_BUDGETS_S: List[float] = [1500.0, 420.0]


def _tb_dirname(time_budget: float) -> str:
    return f"tb_{int(round(time_budget)):04d}s"


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_ideal_table(path: Path = IDEAL_TABLE_PATH) -> Dict[int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["seed"]): float(row["t_ideal_slalom"]) for row in data["rows"]}


def target_mazes(manifest_path: Path = MANIFEST_PATH) -> List[Dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ordered = sorted(manifest["mazes"], key=lambda m: int(m["seed"]))
    return [{"seed": int(m["seed"]), "d0": int(m["d0"])} for m in ordered]


def load_exp028_north(margin_mm: float = EXP028_MARGIN_MM, u: float = EXP028_U) -> Dict[int, dict]:
    """exp_028（真の地図を直接FAST走行させた測定。北向き開始）の一次記録を
    seed をキーに読み込む。`research_notes/note_031` §exp_028 の実測（η中央値
    0.533）と同じ一次記録。"""
    d = EXP028_ROOT / f"margin_{margin_mm:04.1f}mm" / f"u_{u:.2f}"
    out: Dict[int, dict] = {}
    for m in target_mazes():
        rec = _load_json(d / f"maze_{m['seed']}.json")
        if rec is not None:
            out[m["seed"]] = rec.get("north", {})
    return out


# ==========================================================================
# T_plan の選び方
# ==========================================================================
def _pick_t_plan(fast_plan_by_run: Dict[str, float], best_run_index: Optional[int]):
    """T_measured を出した run が FAST計画そのものならその t_plan を使う。
    そうでなければ（例: 探索走行自身がゴールした時間が最良だった場合。
    420秒条件で実際に発生した）、記録されている中で最初の run_index の
    t_plan にフォールバックする（地図はFAST突入後は更新されないので、
    複数回計画されていれば通常どれも同じ値のはず。run_exp029.py
    docstring参照）。戻り値: (t_plan, is_from_best_run: bool)。
    どちらも取れなければ (None, False)。"""
    if not fast_plan_by_run:
        return None, False
    by_run = {int(k): float(v) for k, v in fast_plan_by_run.items()}
    if best_run_index is not None and best_run_index in by_run:
        return by_run[best_run_index], True
    first_key = min(by_run.keys())
    return by_run[first_key], False


def _best_run_index(runs: List[dict], best_time: Optional[float]) -> Optional[int]:
    if best_time is None:
        return None
    for r in runs:
        if r.get("outcome") == "goal" and r.get("run_time") is not None:
            if abs(r["run_time"] - best_time) < 1e-9:
                return int(r["index"])
    return None


def _explore_return_time_used(runs: List[dict], fast_plan_by_run: Dict[str, float],
                               time_budget: float) -> Optional[float]:
    """最初のFAST計画（速度プロファイル追従の最短走行の1本目）が始まる
    までに使った時間 [s]（探索＋帰還が消費した時間。note_030 §8-3 と同じ
    定義）。1本もFAST計画に到達しなかった場合は None（探索・帰還だけで
    持ち時間を使い切った、または全く別の理由で頭打ちになったことを示す）。"""
    if not fast_plan_by_run:
        return None
    first_run_index = min(int(k) for k in fast_plan_by_run.keys())
    for r in runs:
        if int(r["index"]) == first_run_index:
            return float(r["t_start"])
    return None


# ==========================================================================
# 迷路ごとの錨
# ==========================================================================
@dataclass
class Row:
    seed: int
    time_budget: float
    t_ideal: Optional[float] = None
    t_plan: Optional[float] = None
    t_plan_is_best_run: bool = False
    t_measured: Optional[float] = None
    best_run_index: Optional[int] = None
    best_run_outcome: Optional[str] = None  # "goal"（成立） or None（不成立）
    n_runs: int = 0
    n_goal: int = 0
    explore_return_time_used: Optional[float] = None
    note: str = ""

    @property
    def eta(self) -> float:
        # 判定条文§5-0: 走行が1本も成立しなかった迷路は η=0
        if self.t_ideal is None or self.t_measured is None or self.t_measured <= 0:
            return 0.0
        return self.t_ideal / self.t_measured

    @property
    def eta_map(self) -> Optional[float]:
        if self.t_ideal is None or self.t_plan is None or self.t_plan <= 0:
            return None
        return self.t_ideal / self.t_plan

    @property
    def eta_track(self) -> Optional[float]:
        if self.t_plan is None or self.t_measured is None or self.t_measured <= 0:
            return None
        return self.t_plan / self.t_measured

    @property
    def explore_return_frac(self) -> Optional[float]:
        if self.explore_return_time_used is None:
            return None
        return self.explore_return_time_used / self.time_budget


def build_rows(time_budget: float, ideal_table: Dict[int, float]) -> List[Row]:
    rows: List[Row] = []
    missing: List[str] = []
    for m in target_mazes():
        seed = m["seed"]
        row = Row(seed=seed, time_budget=time_budget, t_ideal=ideal_table.get(seed))

        path = OUT_ROOT / _tb_dirname(time_budget) / f"maze_{seed}.json"
        record = _load_json(path)
        if record is None:
            missing.append(str(path))
            rows.append(row)
            continue

        result = record["result"]
        runs = result["runs"]
        row.n_runs = len(runs)
        row.n_goal = sum(1 for r in runs if r["outcome"] == "goal")
        row.t_measured = result["best_time"]  # None なら不成立(η=0)
        row.best_run_index = _best_run_index(runs, row.t_measured)
        row.best_run_outcome = "goal" if row.t_measured is not None else None
        if row.t_measured is None:
            row.note += "5走行とも不成立(または探索が完了しなかった);"

        t_plan, is_best = _pick_t_plan(record.get("fast_plan_by_run", {}), row.best_run_index)
        row.t_plan = t_plan
        row.t_plan_is_best_run = is_best
        if t_plan is None:
            row.note += "FAST計画(profile)に一度も到達しなかった;"
        elif not is_best and row.t_measured is not None:
            row.note += "最良走行はFAST計画由来ではない(探索走行自体がゴール);T_planは最初のFAST計画にフォールバック;"

        row.explore_return_time_used = _explore_return_time_used(
            runs, record.get("fast_plan_by_run", {}), time_budget)

        rows.append(row)

    if missing:
        print(f"[警告] tb={time_budget:.0f}s: 一次記録ファイルが見つからない:")
        for p in missing:
            print(f"  - {p}")

    return rows


def _fmt(x, nd: int = 3) -> str:
    return f"{x:.{nd}f}" if x is not None else "-"


def print_table(time_budget: float, rows: List[Row]) -> None:
    print(f"\n=== 持ち時間 {time_budget:.0f}s ===")
    header = (f"  {'seed':>6} {'T_ideal':>8} {'T_plan':>8} {'T_measured':>10} "
              f"{'eta':>6} {'eta_map':>7} {'eta_track':>9} "
              f"{'n_run':>5} {'n_goal':>6} {'探索消費[s]':>10} {'探索消費%':>8}  結末")
    print(header)
    for r in rows:
        outcome_str = "goal" if r.t_measured is not None else "DNF"
        frac = r.explore_return_frac
        frac_str = f"{frac*100:5.1f}%" if frac is not None else "     -"
        print(f"  {r.seed:>6} {_fmt(r.t_ideal):>8} {_fmt(r.t_plan):>8} {_fmt(r.t_measured):>10} "
              f"{_fmt(r.eta):>6} {_fmt(r.eta_map):>7} {_fmt(r.eta_track):>9} "
              f"{r.n_runs:>5} {r.n_goal:>6} {_fmt(r.explore_return_time_used, 1):>10} {frac_str:>8}  "
              f"{outcome_str}" + (f"  注記: {r.note}" if r.note else ""))

    etas = [r.eta for r in rows]
    eta_nonzero = [e for e in etas if e > 0]
    n_zero = sum(1 for e in etas if e == 0.0)
    print(f"\n  η 中央値(10迷路, DNFはη=0で算入) = {_fmt(statistics.median(etas))}")
    print(f"  η 最小値(10迷路, DNFはη=0で算入) = {_fmt(min(etas))}")
    print(f"  η=0 の迷路数 = {n_zero} / {len(rows)}")
    if eta_nonzero:
        print(f"  (参考)成立した走行だけのη 中央値 = {_fmt(statistics.median(eta_nonzero))}"
              f"  最小値 = {_fmt(min(eta_nonzero))}  (n={len(eta_nonzero)})")

    eta_maps = [r.eta_map for r in rows if r.eta_map is not None]
    if eta_maps:
        print(f"  η_map 中央値(取得できた迷路のみ, n={len(eta_maps)}) = {_fmt(statistics.median(eta_maps))}")
    eta_tracks = [r.eta_track for r in rows if r.eta_track is not None]
    if eta_tracks:
        print(f"  η_track 中央値(取得できた迷路のみ, n={len(eta_tracks)}) = {_fmt(statistics.median(eta_tracks))}")

    fracs = [r.explore_return_frac for r in rows if r.explore_return_frac is not None]
    n_no_fast = sum(1 for r in rows if r.explore_return_time_used is None)
    if fracs:
        print(f"  探索+帰還が消費した持ち時間の割合 中央値(FAST到達迷路のみ, n={len(fracs)}) "
              f"= {_fmt(statistics.median(fracs)*100, 1)}%  範囲 = "
              f"[{_fmt(min(fracs)*100,1)}%, {_fmt(max(fracs)*100,1)}%]")
    print(f"  FAST計画(profile)に一度も到達しなかった迷路数 = {n_no_fast} / {len(rows)}")


def print_exp028_comparison(rows_by_seed_1500: Dict[int, Row]) -> None:
    exp028 = load_exp028_north()
    ideal_table = load_ideal_table()
    print("\n=== exp_028（真の地図を直接FAST走行）vs exp_029（学習した地図。持ち時間1500s）比較 ===")
    print(f"  {'seed':>6} {'eta_exp028':>10} {'eta_exp029':>10} {'差(028-029)':>11} "
          f"{'eta_map(exp029)':>15}  備考")
    diffs = []
    for seed, leg in sorted(exp028.items()):
        t_ideal = ideal_table.get(seed)
        t_measured_028 = leg.get("run_time") if leg.get("outcome") == "goal" else None
        eta_028 = (t_ideal / t_measured_028) if (t_ideal and t_measured_028) else None
        row029 = rows_by_seed_1500.get(seed)
        eta_029 = row029.eta if row029 is not None else None
        eta_map_029 = row029.eta_map if row029 is not None else None
        diff = (eta_028 - eta_029) if (eta_028 is not None and eta_029 is not None) else None
        if diff is not None:
            diffs.append(diff)
        note = ""
        if row029 is not None and row029.note:
            note = row029.note
        print(f"  {seed:>6} {_fmt(eta_028):>10} {_fmt(eta_029):>10} {_fmt(diff):>11} "
              f"{_fmt(eta_map_029):>15}  {note}")
    if diffs:
        print(f"\n  差(eta_028 - eta_029)の中央値 = {_fmt(statistics.median(diffs))}"
              f"  (n={len(diffs)}／exp_028で完走した5迷路のみ)")


def main() -> int:
    ideal_table = load_ideal_table()
    rows_by_tb: Dict[float, List[Row]] = {}
    for tb in TIME_BUDGETS_S:
        rows = build_rows(tb, ideal_table)
        rows_by_tb[tb] = rows
        print_table(tb, rows)

    rows_1500_by_seed = {r.seed: r for r in rows_by_tb.get(1500.0, [])}
    print_exp028_comparison(rows_1500_by_seed)

    # 420s vs 1500s: 420sで最短走行(FAST計画)に到達できなかった迷路の数
    rows_420 = rows_by_tb.get(420.0, [])
    n_no_fast_420 = sum(1 for r in rows_420 if r.t_plan is None)
    n_dnf_420 = sum(1 for r in rows_420 if r.t_measured is None)
    print(f"\n=== 持ち時間420s固有の内訳 ===")
    print(f"  FAST計画に一度も到達しなかった迷路 = {n_no_fast_420} / {len(rows_420)}")
    print(f"  走行が1本も成立しなかった(DNF, eta=0)迷路 = {n_dnf_420} / {len(rows_420)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
