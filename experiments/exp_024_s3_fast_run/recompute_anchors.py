"""
experiments/exp_024_s3_fast_run/recompute_anchors.py
================
exp_024（段階 S3「最短走行」の完了判定）の錨の独立再計算（PREREG §9）。

`outputs/exp_024_s3/<condition>/maze_<seed>.json` の**一次記録だけ**を入力として、
PREREG §5 の定義どおりに `T_explore`・`T_fast`・`R`・`C`・`E` を自分で組み立てて
計算し直す。

🔴 評価器が返した `best_time` や `maze_kpi()` の出力は**使わない**
（`runs[]` と `run_phases` から独自に計算する。集約結果の引き写しの禁止）。

事前登録した判定表（PREREG §6 の境界・§8 の宣言条件）をスクリプト内に定数
として持ち、再計算した値と突き合わせて合否を印字する。

🔴 本スクリプトは判定条文そのものを新設・変更しない —
PREREG.md に既に確定・コミット済みの条文をそのまま定数化して転記しているだけ
である（食い違いを見つけたら PREREG.md を正として本スクリプトを直す）。

使い方:
    .venv/bin/python experiments/exp_024_s3_fast_run/recompute_anchors.py
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_ROOT = REPO_ROOT / "outputs" / "exp_024_s3"
MANIFEST_PATH = REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "manifest.json"

CONDITION_EXTENDED = "extended"
CONDITION_PERCELL = "percell"


# ==========================================================================
# 判定表（PREREG §6・§8 の定数化。ここで新設・変更はしない）
# ==========================================================================
def judge_R(R: float) -> str:
    """PREREG §6 の判定表。`R < 1.0` のみ合格（`R == 1.0` ちょうども不合格側）。"""
    return "合格" if R < 1.0 else "不合格"


def judge_declaration(R: float, C: float) -> str:
    """PREREG §8 の宣言条件（`R < 1.0` かつ `C >= 0.5`）と、
    欠けた場合の読み方の表をそのまま定数化したもの。"""
    if math.isinf(R):
        return (
            "最短走行が1本も成立していない（R=+∞、Sが空）。"
            "まず成立しない原因（探索未完了か、最短走行の実行時の失敗か）を"
            "run_phasesとoutcomeで切り分ける。"
        )
    if R >= 1.0:
        return "S3の実装に不足がある。最短走行が探索より遅い原因を一次記録から切り分ける。"
    if C < 0.5:
        return "不足はS3ではなくS1（探索の到達率）。次の仕事を到達率へ向ける。"
    return "S3完了。次はS4（速度向上）。"


# ==========================================================================
# PREREG §5 の定義どおりの再構築（1 迷路・1 条件の runs[]/run_phases[] から）
# ==========================================================================
def compute_t_explore(runs: List[dict]) -> Optional[float]:
    """T_explore(i) = 最初にゴールした走行のタイム。該当が無ければ None。

    「最初」は走行番号（runs[].index、評価器が発行する 1 始まりの通し番号。
    出発順と一致する）の昇順で決める。"""
    goal_runs = [r for r in runs if r.get("outcome") == "goal" and r.get("run_time") is not None]
    if not goal_runs:
        return None
    first = min(goal_runs, key=lambda r: r["index"])
    return float(first["run_time"])


def compute_t_fast(runs: List[dict], run_phases: List[dict]) -> Optional[float]:
    """T_fast(i) = 「走行が始まった瞬間の段階が FAST であり、かつゴールした
    走行」のタイムの最小値。該当が無ければ None。

    🔴「最初のゴール走行より後」という順序だけでは決めない（PREREG §5）。
    run_phases（run_index -> 開始時の段階）と runs（index, outcome, run_time）
    を run_index で突き合わせて判定する。"""
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
# 迷路ごとの錨（判定量の入力になる最小限の値だけを保持する）
# ==========================================================================
@dataclass(frozen=True)
class MazeAnchor:
    seed: int
    d0: Optional[int]
    t_explore: Optional[float]           # 条件 extended から
    t_fast_extended: Optional[float]     # 条件 extended から
    t_fast_percell: Optional[float]      # 条件 percell から
    outcomes_extended: List[str] = field(default_factory=list)
    outcomes_percell: List[str] = field(default_factory=list)

    @property
    def in_s(self) -> bool:
        """PREREG §6 の S（条件 extended で T_fast(i) が定義される迷路の集合）
        に属するか。T_fast は T_explore あってこその比なので、両方が
        定義されていることを要求する。"""
        return self.t_explore is not None and self.t_fast_extended is not None

    @property
    def ratio(self) -> Optional[float]:
        return (self.t_fast_extended / self.t_explore) if self.in_s else None

    @property
    def in_s_prime(self) -> bool:
        """PREREG §7 副 2 の S'（extended と percell の両方で T_fast(i) が
        定義される迷路の集合）に属するか。"""
        return self.t_fast_extended is not None and self.t_fast_percell is not None

    @property
    def e_ratio(self) -> Optional[float]:
        return (self.t_fast_extended / self.t_fast_percell) if self.in_s_prime else None


def build_anchor(seed: int, d0: Optional[int], extended_record: dict,
                  percell_record: Optional[dict] = None) -> MazeAnchor:
    """1 迷路ぶんの一次記録（の生の dict。実物の JSON をそのまま `json.load`
    したもの）から MazeAnchor を組み立てる。

    Args:
        extended_record: `outputs/exp_024_s3/extended/maze_<seed>.json` の中身
            （`runs` と `run_phases` キーを持つ dict）。
        percell_record: 同 percell 側。無ければ None（副 2 は計算できない）。
    """
    ext_runs = extended_record.get("runs", [])
    ext_phases = extended_record.get("run_phases", [])
    t_explore = compute_t_explore(ext_runs)
    t_fast_extended = compute_t_fast(ext_runs, ext_phases)
    outcomes_extended = [r.get("outcome") for r in ext_runs]

    t_fast_percell = None
    outcomes_percell: List[str] = []
    if percell_record is not None:
        pc_runs = percell_record.get("runs", [])
        pc_phases = percell_record.get("run_phases", [])
        t_fast_percell = compute_t_fast(pc_runs, pc_phases)
        outcomes_percell = [r.get("outcome") for r in pc_runs]

    return MazeAnchor(
        seed=seed, d0=d0,
        t_explore=t_explore, t_fast_extended=t_fast_extended, t_fast_percell=t_fast_percell,
        outcomes_extended=outcomes_extended, outcomes_percell=outcomes_percell,
    )


# ==========================================================================
# PREREG §6・§7 の主・副判定量
# ==========================================================================
def compute_R(anchors: List[MazeAnchor]) -> float:
    """R = max(T_fast(i)/T_explore(i))（i∈S）。S が空なら +∞。"""
    ratios = [a.ratio for a in anchors if a.in_s]
    return max(ratios) if ratios else math.inf


def compute_C(anchors: List[MazeAnchor]) -> float:
    """C = |S| / 6（対象迷路の総数。実運用では常に 6 だが、検査のため
    分母は渡された anchors の総数そのものを使う）。"""
    if not anchors:
        return 0.0
    n_in_s = sum(1 for a in anchors if a.in_s)
    return n_in_s / len(anchors)


def compute_E(anchors: List[MazeAnchor]) -> float:
    """E = median(T_fast_extended(i)/T_fast_percell(i))（i∈S'）。
    S' が空なら +∞。"""
    ratios = [a.e_ratio for a in anchors if a.in_s_prime]
    return statistics.median(ratios) if ratios else math.inf


# ==========================================================================
# 対象迷路（manifest.json から D0 昇順で 6 迷路。表示用の D0 の出所であり、
# 判定量そのものの計算には使わない）
# ==========================================================================
def _target_mazes_from_manifest(manifest_path: Path = MANIFEST_PATH, n: int = 6) -> List[Dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ordered = sorted(manifest["mazes"], key=lambda m: (int(m["d0"]), int(m["seed"])))
    return [{"seed": int(m["seed"]), "d0": int(m["d0"])} for m in ordered[:n]]


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_anchors_from_outputs(out_root: Path = OUT_ROOT,
                               manifest_path: Path = MANIFEST_PATH) -> List[MazeAnchor]:
    """`outputs/exp_024_s3/` の実測ファイルから MazeAnchor の列を組み立てる。
    ファイルが無い迷路・条件は None のまま扱う（欠測は S/S' に入らないだけで
    処理は止めない — 部分的な実測でも現況を印字できるようにするため）。"""
    targets = _target_mazes_from_manifest(manifest_path)
    anchors = []
    missing: List[str] = []
    for m in targets:
        seed, d0 = m["seed"], m["d0"]
        ext_path = out_root / CONDITION_EXTENDED / f"maze_{seed}.json"
        pc_path = out_root / CONDITION_PERCELL / f"maze_{seed}.json"
        ext_record = _load_json(ext_path)
        pc_record = _load_json(pc_path)
        if ext_record is None:
            missing.append(str(ext_path))
        if pc_record is None:
            missing.append(str(pc_path))
        if ext_record is None:
            # extended が無ければ T_explore/T_fast_extended どちらも組み立てられない。
            anchors.append(MazeAnchor(seed=seed, d0=d0, t_explore=None,
                                       t_fast_extended=None, t_fast_percell=None))
            continue
        anchors.append(build_anchor(seed, d0, ext_record, pc_record))
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
    print("迷路ごとの内訳（条件 extended の T_explore・T_fast、比 = T_fast/T_explore）:")
    header = f"  {'seed':>6} {'D0':>4} {'T_explore':>10} {'T_fast':>10} {'比':>8}  outcomes(extended)"
    print(header)
    for a in anchors:
        print(f"  {a.seed:>6} {a.d0 if a.d0 is not None else '?':>4} "
              f"{_fmt_t(a.t_explore):>10} {_fmt_t(a.t_fast_extended):>10} "
              f"{_fmt_ratio(a.ratio):>8}  {a.outcomes_extended}")

    R = compute_R(anchors)
    C = compute_C(anchors)
    E = compute_E(anchors)
    n_in_s = sum(1 for a in anchors if a.in_s)
    n_in_s_prime = sum(1 for a in anchors if a.in_s_prime)

    print()
    print(f"S  = {{{', '.join(str(a.seed) for a in anchors if a.in_s)}}}  (|S|={n_in_s})")
    print(f"R  = {R if math.isfinite(R) else '+inf'}  -> {judge_R(R)}")
    print(f"C  = {n_in_s}/{len(anchors)} = {C:.4f}  (C >= 0.5 が条件)")
    print(f"S' = {{{', '.join(str(a.seed) for a in anchors if a.in_s_prime)}}}  (|S'|={n_in_s_prime})")
    print(f"E  = {E if math.isfinite(E) else '+inf'}")
    print()
    print(f"[段階 S3 完了の宣言] {judge_declaration(R, C)}")


def main() -> int:
    anchors = load_anchors_from_outputs()
    print_report(anchors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
