"""
experiments/exp_025_s4_slalom/recompute_anchors.py
================
exp_025（段階 S4「スラローム」の完了判定）の錨の独立再計算（PREREG §9・追記 1〜3）。

`outputs/exp_025_s4/<condition>/maze_<seed>.json`（条件 = "slalom"/"control"）の
**一次記録だけ**を入力として、PREREG §5〜§7 の定義どおりに `t_fast`・`R`・`S`・`G`・
A1（弧 1 回の所要時間）を自分で組み立てて計算し直す。

🔴 評価器が返した `best_time` や `maze_kpi()` の出力は**使わない**
（`runs[]` と `run_phases[]` から独自に計算する。集約結果の引き写しの禁止）。

🔴 `t_fast(条件, 迷路)` の取り方（PREREG §5・本タスクの指示を一字一句そのまま
実装する）:
    `run_phases` の `phase == "FAST"` かつ結果が goal である走行のうち
    **最初のもの**の `run_time` を取る。
    exp_024 の `recompute_anchors.py` は複数の成立走行から run_time が
    **最小**のものを選んでいたが、本実験は「最初に成立した最短走行」を
    対の相手にする（結果が複数あっても 2 本目以降は見ない）。
    あわせてその走行の**終了時刻**（迷路開始からの絶対時刻 `t_end`）も返す
    — PREREG 追記 3 の副 2（G。終了時刻 ≤ 420 s か）に使う。

🔴 本スクリプトは判定条文そのものを新設・変更しない —
PREREG.md に既に確定・コミット済みの条文をそのまま定数化して転記しているだけ
である（食い違いを見つけたら PREREG.md を正として本スクリプトを直す）。

使い方:
    .venv/bin/python experiments/exp_025_s4_slalom/recompute_anchors.py
    .venv/bin/python experiments/exp_025_s4_slalom/recompute_anchors.py \\
        --out-root outputs/exp_024_s3   # 別実験の出力を与えても壊れないかの確認用
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MANIFEST_PATH = REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "manifest.json"
OUT_ROOT_DEFAULT = REPO_ROOT / "outputs" / "exp_025_s4"

CONDITION_SLALOM = "slalom"
CONDITION_CONTROL = "control"

# 一次記録に必須の列（無ければ「壊れる」のではなく、印字して SystemExit する）。
REQUIRED_RECORD_FIELDS = ("runs", "run_phases", "plan_ids", "protocol")

# ==========================================================================
# 判定表（PREREG §6・§7・§8・追記 2 の定数化。ここで新設・変更はしない）
# ==========================================================================
R_PASS_THRESHOLD = 0.80        # PREREG §6: R <= 0.80 が合格域
R_PER_MAZE_THRESHOLD = 1.0     # PREREG §8-2: 全迷路で R(迷路) < 1.0
S_PASS_THRESHOLD = 1.0         # PREREG §8-3: S = 1.0（副 1・健全性）
EXCLUDE_LIMIT = 3              # PREREG 追記 2: 除外が 3 迷路を超えたら中央値は使わない
COMPETITION_TIME_BUDGET_S = 420.0  # PREREG 追記 3: 副 2 G の判定時刻

# 錨 A1（PREREG §9）: 弧 1 回の設計所要時間。
# 値の出所は geometry_anchor.py と同じ（`mouse/params.py` cell_size、
# `classic/motion.py` DEFAULT_V_CRUISE）。ここでは「この式から計算する」と
# いう指示どおり、定数から式で導出する（1.178 s を直書きしない）。
CELL = 0.180        # mouse/params.py cell_size
R_ARC = CELL / 2.0  # 弧の半径（geometry_anchor.py と同じ置き方）
V_CRUISE = 0.12     # classic/motion.py DEFAULT_V_CRUISE
ARC_DESIGN_S = R_ARC * math.pi / 2.0 / V_CRUISE
ARC_TOLERANCE = 0.10  # ±10%（PREREG §9 A1）
ARC_PLAN_IDS = ("fast:slalom_left", "fast:slalom_right")  # PREREG §5「弧 1 回」の定義


def judge_R(R: Optional[float]) -> str:
    """PREREG §6 の判定表。`R <= 0.80` のみ合格域。R が未定義（None）なら判定不能。"""
    if R is None:
        return "判定不能（R の中央値が計算できない）"
    return "合格域" if R <= R_PASS_THRESHOLD else "不合格域"


def judge_S(S: Optional[float]) -> str:
    """PREREG §8-3 の宣言条件。S == 1.0 のみ合格。"""
    if S is None:
        return "判定不能（S の分母が 0）"
    return "合格" if S >= S_PASS_THRESHOLD - 1e-9 else "不合格"


# ==========================================================================
# PREREG §5・追記 3 の定義どおりの再構築（1 迷路・1 条件の runs[]/run_phases[] から）
# ==========================================================================
def compute_t_fast_first(runs: List[dict], run_phases: List[dict]) -> Tuple[Optional[float], Optional[float]]:
    """『最初に成立した最短走行』の (run_time, t_end) を返す。

    定義（PREREG §5・本タスクの指示のとおり。exp_024 のように run_time の
    最小値を選ぶのではない）:
        run_phases（走行開始時点の phase）が "FAST" であり、かつ結果が
        goal である走行のうち、run index が最小のもの（＝時系列で最初に
        成立した最短走行）を無条件に取る。
    該当が無ければ (None, None)。
    """
    phase_by_run_index = {p["run_index"]: p["phase"] for p in run_phases}
    candidates = [
        r for r in runs
        if r.get("outcome") == "goal"
        and r.get("run_time") is not None
        and phase_by_run_index.get(r["index"]) == "FAST"
    ]
    if not candidates:
        return None, None
    first = min(candidates, key=lambda r: r["index"])
    return float(first["run_time"]), float(first["t_end"])


def first_fast_tick_index(plan_ids: List[str]) -> int:
    """plan_ids の中で最初に `"fast:"` 接頭辞が現れるティック番号（0 始まり）を
    返す。1 度も現れなければ `len(plan_ids)`（＝全列を比較対象にする）。"""
    for i, pid in enumerate(plan_ids):
        if pid.startswith("fast:"):
            return i
    return len(plan_ids)


def premise_plan_ids_match(plan_ids_a: List[str], plan_ids_b: List[str]) -> bool:
    """PREREG 追記 1 の前提検査: 『同じ迷路の 2 条件で、最初の FAST が始まる
    までの plan_ids が完全一致する』こと（探索と帰還はスラロームの影響を
    受けないので一致するはず、という前提の検査）。"""
    ia = first_fast_tick_index(plan_ids_a)
    ib = first_fast_tick_index(plan_ids_b)
    if ia != ib:
        return False
    return plan_ids_a[:ia] == plan_ids_b[:ib]


def find_arc_durations(plan_ids: List[str], control_dt: float) -> List[float]:
    """錨 A1: plan_id が `fast:slalom_left`/`fast:slalom_right` である連続
    ティック区間（PREREG §5「弧 1 回」の定義）ごとに、ティック数 * control_dt
    を弧 1 回の実測所要時間として返す。"""
    durations = []
    for key, group in itertools.groupby(plan_ids):
        if key in ARC_PLAN_IDS:
            n_ticks = sum(1 for _ in group)
            durations.append(n_ticks * control_dt)
    return durations


# ==========================================================================
# 迷路ごとの錨
# ==========================================================================
@dataclass(frozen=True)
class MazeAnchor:
    seed: int
    d0: Optional[int]
    t_fast_slalom: Optional[float]
    t_end_slalom: Optional[float]
    t_fast_control: Optional[float]
    t_end_control: Optional[float]
    premise_ok: Optional[bool]          # None = 比較不能（どちらかのデータが無い）
    arc_durations: List[float] = field(default_factory=list)  # A1（slalom 条件のみ）
    outcomes_slalom: List[str] = field(default_factory=list)
    outcomes_control: List[str] = field(default_factory=list)

    @property
    def control_defined(self) -> bool:
        """PREREG 追記 2: 対照条件で最短走行が成立したか（R の中央値・S の
        分母の基準）。"""
        return self.t_fast_control is not None

    @property
    def in_r_domain(self) -> bool:
        """R(迷路) が定義できるか = 両条件の t_fast がともに定義される
        （追記 1 の前提が崩れている迷路は判定から外す）。"""
        return (self.t_fast_slalom is not None and self.t_fast_control is not None
                and self.premise_ok is not False)

    @property
    def ratio(self) -> Optional[float]:
        return (self.t_fast_slalom / self.t_fast_control) if self.in_r_domain else None


def build_anchor(seed: int, d0: Optional[int], slalom_record: Optional[dict],
                  control_record: Optional[dict]) -> MazeAnchor:
    """1 迷路ぶんの一次記録（実物の JSON をそのまま `json.load` したもの）から
    MazeAnchor を組み立てる。どちらかが無ければ None のまま扱う。"""
    t_fast_slalom = t_end_slalom = None
    outcomes_slalom: List[str] = []
    arc_durations: List[float] = []
    plan_ids_slalom: Optional[List[str]] = None
    if slalom_record is not None:
        _check_required_fields(slalom_record, f"slalom/maze_{seed}.json")
        s_runs = slalom_record.get("runs", [])
        s_phases = slalom_record.get("run_phases", [])
        t_fast_slalom, t_end_slalom = compute_t_fast_first(s_runs, s_phases)
        outcomes_slalom = [r.get("outcome") for r in s_runs]
        plan_ids_slalom = slalom_record.get("plan_ids", [])
        control_hz = slalom_record.get("protocol", {}).get("control_hz")
        control_dt = (1.0 / control_hz) if control_hz else None
        if control_dt is not None:
            arc_durations = find_arc_durations(plan_ids_slalom, control_dt)

    t_fast_control = t_end_control = None
    outcomes_control: List[str] = []
    plan_ids_control: Optional[List[str]] = None
    if control_record is not None:
        _check_required_fields(control_record, f"control/maze_{seed}.json")
        c_runs = control_record.get("runs", [])
        c_phases = control_record.get("run_phases", [])
        t_fast_control, t_end_control = compute_t_fast_first(c_runs, c_phases)
        outcomes_control = [r.get("outcome") for r in c_runs]
        plan_ids_control = control_record.get("plan_ids", [])

    premise_ok: Optional[bool] = None
    if plan_ids_slalom is not None and plan_ids_control is not None:
        premise_ok = premise_plan_ids_match(plan_ids_slalom, plan_ids_control)

    return MazeAnchor(
        seed=seed, d0=d0,
        t_fast_slalom=t_fast_slalom, t_end_slalom=t_end_slalom,
        t_fast_control=t_fast_control, t_end_control=t_end_control,
        premise_ok=premise_ok, arc_durations=arc_durations,
        outcomes_slalom=outcomes_slalom, outcomes_control=outcomes_control,
    )


def _check_required_fields(record: dict, label: str) -> None:
    """一次記録に必要な列が揃っているかを検査する。足りなければ、原因不明の
    例外で落ちる前に理由を印字して `SystemExit` する（壊れないこと、の意味）。"""
    missing = [f for f in REQUIRED_RECORD_FIELDS if f not in record]
    if missing:
        raise SystemExit(
            f"🔴 一次記録に必要な列が無い: {label} に {missing} が無い。"
            "本スクリプトは exp_025 の一次記録形式（run_exp025.py が出力する "
            "runs/run_phases/plan_ids/protocol を持つ JSON）を前提とする。"
            "実行を中止する。"
        )


# ==========================================================================
# PREREG §6・§7・追記 2・追記 3 の主・副判定量
# ==========================================================================
def compute_R(anchors: List[MazeAnchor]) -> Tuple[Optional[float], List[MazeAnchor]]:
    """R = 10 迷路の R(迷路) の中央値（PREREG §6）。
    追記 2: 除外（対照未成立、または premise_ok=False）が EXCLUDE_LIMIT を
    超えたら None を返す（中央値は判定に使わない）。
    戻り値: (R または None, R の計算に使った迷路の一覧)"""
    used = [a for a in anchors if a.in_r_domain]
    excluded = [a for a in anchors if not a.in_r_domain]
    if len(excluded) > EXCLUDE_LIMIT:
        return None, used
    if not used:
        return None, used
    ratios = [a.ratio for a in used]
    return statistics.median(ratios), used


def compute_S(anchors: List[MazeAnchor]) -> Tuple[Optional[float], int, int]:
    """S = (条件 A が正しく完走した迷路数) / (対照条件で最短走行が成立した
    迷路数)（PREREG §7 副 1・追記 2 の分母の定義）。
    「条件 A が正しく完走した」= t_fast_slalom が定義される
    （= 衝突・スタック・転倒なく FAST でゴールへ到達した）。
    分母が 0 なら (None, 0, 0)。"""
    denom_anchors = [a for a in anchors if a.control_defined]
    denom = len(denom_anchors)
    if denom == 0:
        return None, 0, 0
    numer = sum(1 for a in denom_anchors if a.t_fast_slalom is not None)
    return numer / denom, numer, denom


def compute_G(anchors: List[MazeAnchor], condition: str, total_target: int) -> Tuple[int, List[int]]:
    """副 2 G（PREREG §7・追記 3）: 『最短走行の終了時刻 ≤ 420.0 s』であった
    迷路数と、その seed の一覧を返す。分母（対象迷路の総数）は呼び出し側が
    total_target で渡す（この関数は分子と該当 seed だけを返す）。
    condition は "slalom" または "control"。"""
    qualifying = []
    for a in anchors:
        t_end = a.t_end_slalom if condition == CONDITION_SLALOM else a.t_end_control
        if t_end is not None and t_end <= COMPETITION_TIME_BUDGET_S:
            qualifying.append(a.seed)
    return len(qualifying), qualifying


def compute_A1(anchors: List[MazeAnchor]) -> Dict:
    """錨 A1（PREREG §9）: 全迷路の弧所要時間を集約し、設計値との比較を返す。"""
    all_durations: List[float] = []
    for a in anchors:
        all_durations.extend(a.arc_durations)
    if not all_durations:
        return {"n": 0, "mean": None, "min": None, "max": None,
                "design_s": ARC_DESIGN_S, "rel_err": None, "within_tolerance": None}
    mean_s = statistics.mean(all_durations)
    rel_err = abs(mean_s - ARC_DESIGN_S) / ARC_DESIGN_S
    return {
        "n": len(all_durations), "mean": mean_s,
        "min": min(all_durations), "max": max(all_durations),
        "design_s": ARC_DESIGN_S, "rel_err": rel_err,
        "within_tolerance": rel_err <= ARC_TOLERANCE,
    }


# ==========================================================================
# 対象迷路（manifest.json から seed 昇順で全件。run_exp025.py の
# select_target_mazes() と同じ選び方。表示用の D0 の出所であり、判定量
# そのものの計算には使わない）
# ==========================================================================
def _target_mazes_from_manifest(manifest_path: Path = MANIFEST_PATH) -> List[Dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ordered = sorted(manifest["mazes"], key=lambda m: int(m["seed"]))
    return [{"seed": int(m["seed"]), "d0": int(m["d0"])} for m in ordered]


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_anchors_from_outputs(out_root: Path = OUT_ROOT_DEFAULT,
                               manifest_path: Path = MANIFEST_PATH) -> List[MazeAnchor]:
    """`outputs/exp_025_s4/`（または --out-root で指定した別ディレクトリ）の
    実測ファイルから MazeAnchor の列を組み立てる。ファイルが無い迷路・条件は
    None のまま扱う（欠測は R/S の分母に入らないだけで処理は止めない —
    部分的な実測でも現況を印字できるようにするため）。"""
    targets = _target_mazes_from_manifest(manifest_path)
    anchors = []
    missing: List[str] = []
    for m in targets:
        seed, d0 = m["seed"], m["d0"]
        slalom_path = out_root / CONDITION_SLALOM / f"maze_{seed}.json"
        control_path = out_root / CONDITION_CONTROL / f"maze_{seed}.json"
        slalom_record = _load_json(slalom_path)
        control_record = _load_json(control_path)
        if slalom_record is None:
            missing.append(str(slalom_path))
        if control_record is None:
            missing.append(str(control_path))
        anchors.append(build_anchor(seed, d0, slalom_record, control_record))
    if missing:
        print("[警告] 一次記録ファイルが見つからない（未測定、または測定途中）:")
        for p in missing:
            print(f"  - {p}")
    return anchors


# ==========================================================================
# 印字
# ==========================================================================
def _fmt_t(t: Optional[float]) -> str:
    return f"{t:.3f}s" if t is not None else "未定義"


def _fmt_ratio(r: Optional[float]) -> str:
    return f"{r:.4f}" if r is not None else "—"


def print_report(anchors: List[MazeAnchor]) -> None:
    print("迷路ごとの内訳（t_fast は『最初に成立した最短走行』の run_time）:")
    header = (f"  {'seed':>6} {'D0':>4} {'t_fast(slalom)':>15} {'t_fast(control)':>16} "
              f"{'R(迷路)':>8} {'前提一致':>8}")
    print(header)
    for a in anchors:
        premise_str = "—" if a.premise_ok is None else ("一致" if a.premise_ok else "不一致")
        print(f"  {a.seed:>6} {a.d0 if a.d0 is not None else '?':>4} "
              f"{_fmt_t(a.t_fast_slalom):>15} {_fmt_t(a.t_fast_control):>16} "
              f"{_fmt_ratio(a.ratio):>8} {premise_str:>8}")

    R, used = compute_R(anchors)
    excluded = [a for a in anchors if not a.in_r_domain]
    excluded_control_fail = [a for a in anchors if not a.control_defined]
    excluded_slalom_fail_only = [a for a in anchors
                                  if a.control_defined and a.t_fast_slalom is None]
    excluded_premise_mismatch = [a for a in anchors if a.premise_ok is False]

    print()
    print(f"R の計算に使った迷路 = {{{', '.join(str(a.seed) for a in used)}}}  (n={len(used)})")
    print(f"除外した迷路 = {{{', '.join(str(a.seed) for a in excluded)}}}  "
          f"(n={len(excluded)}、うち対照未成立={len(excluded_control_fail)}、"
          f"slalom のみ未成立={len(excluded_slalom_fail_only)}、"
          f"前提不一致={len(excluded_premise_mismatch)})")
    if len(excluded) > EXCLUDE_LIMIT:
        print(f"🔴 除外が {EXCLUDE_LIMIT} 迷路を超えた（{len(excluded)} 迷路）。"
              "PREREG 追記 2 により R の中央値は判定に使わない。")
    print(f"R = {R if R is not None else '未定義'}  -> {judge_R(R)}")

    if R is not None:
        n_over_1 = sum(1 for a in used if a.ratio is not None and a.ratio >= R_PER_MAZE_THRESHOLD)
        print(f"R(迷路) >= {R_PER_MAZE_THRESHOLD} だった迷路数 = {n_over_1} "
              f"（PREREG §8-2「全迷路で R(迷路) < 1.0」の判定に使う。0 でなければ不合格）")

    S, s_numer, s_denom = compute_S(anchors)
    print()
    print(f"S = {s_numer}/{s_denom} = {S if S is not None else '未定義'}  -> {judge_S(S)}")

    total_target = len(anchors)
    g_slalom, g_slalom_seeds = compute_G(anchors, CONDITION_SLALOM, total_target)
    g_control, g_control_seeds = compute_G(anchors, CONDITION_CONTROL, total_target)
    print()
    print(f"G（副 2・判定には使わない。持ち時間 {COMPETITION_TIME_BUDGET_S:.0f}s 以内に"
          f"最短走行が終了した迷路数）:")
    print(f"  slalom : {g_slalom}/{total_target}  seeds={g_slalom_seeds}")
    print(f"  control: {g_control}/{total_target}  seeds={g_control_seeds}")

    a1 = compute_A1(anchors)
    print()
    print(f"A1（弧 1 回の所要時間。設計値 = {a1['design_s']:.3f}s "
          f"= {R_ARC*1000:.1f}mm・π/2 / {V_CRUISE}m/s）:")
    if a1["n"] == 0:
        print("  該当区間 0 件（plan_ids に fast:slalom_left/right が現れていない。"
              "slalom 未実装、または slalom=False のみで測定した可能性がある）。")
    else:
        verdict = "±10%以内" if a1["within_tolerance"] else "🔴 ±10%を超えている"
        print(f"  n={a1['n']}  mean={a1['mean']:.3f}s  min={a1['min']:.3f}s  "
              f"max={a1['max']:.3f}s  相対誤差={a1['rel_err']*100:.2f}%  -> {verdict}")

    print()
    print("🔴 本スクリプトは §8 の宣言条件（4 項目すべて）を自動判定しない"
          "（N1〜N6 否定対照は classic/checks.py 側の検査であり、本スクリプトの"
          "入力である一次記録には現れない）。上記 R・S・A1 の数字を "
          "judgment.md へ転記し、教授セッションが §8 と突き合わせて判定すること。")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT,
                         help="一次記録のルートディレクトリ（既定: outputs/exp_025_s4）")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH,
                         help="manifest.json のパス（既定: design_turn_v1）")
    args = parser.parse_args(argv)

    anchors = load_anchors_from_outputs(args.out_root, args.manifest)
    print_report(anchors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
