"""
experiments/exp_030_map_quality/trace_fatal.py
================
致命的な誤りが「その壁を記録した瞬間」を一次記録から特定する診断スクリプト
（任務指示「さらに: 致命的な誤りがどこで生まれたかを特定する」）。

`judge.py` が検出した致命的な誤りの箇所（真は壁なのに「開通」と判定した壁）を
対象に、その迷路だけを再実行する。`ClassicExplorer`/`ClassicExplorerPolicy` に
乱数は無く MuJoCo の物理も決定的なので、`run_exp030.py` が測定したのと
**ビット同一の**探索が再現されるはず（この前提は末尾で検算し、崩れていれば
そのまま報告する）。

再実行時だけ、以下をフックする（呼び出す内容・戻り値は一切変えない。継承・
オーバーライドで「記録を追加するだけ」）:
  1. `classic/explorer.py` の `_update_map_from_sensing`: 書き込み直前の
     `self.cell`（信じている区画）・`self.heading`（信じている向き）・
     センサ判定を記録する。
  2. `mouse/sim.py` の `MouseSim.__init__`: 生成されたインスタンスを捕まえておき、
     書き込みの瞬間に `sim.privileged_pose()`（診断専用。方策には一切渡さない）
     を読んで実際の位置を記録する。

🔴 これは診断専用の再実行であり、方策自身は真値を一切参照しない
（`requires_privileged=False` は不変）。

出力: 各致命的な誤りについて、書き込み時に信じていた区画・実際の区画、
位置ずれ [mm]（前後・横に分解）、方位のずれ [deg]、そのときのセンサ判定。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MAZE_DIR = REPO_ROOT / "competition" / "mazes" / "design_turn_v1"
OUT_ROOT = REPO_ROOT / "outputs" / "exp_030_map_quality"

CELL_SIZE = 0.18  # mouse/params.py RobotParams.cell_size の既定値と同じ

# Direction(N=0,E=1,S=2,W=3) → 期待される機体ヨー角 [deg]
# （mouse.sim.MouseSim.full_reset(heading_deg=90.0) が N=90° を定義。
#  classic/maze_map.py の _DIR_DELTA と整合: N=(0,1)=90°,E=(1,0)=0°,S=(0,-1)=-90°,W=(-1,0)=180°）
_HEADING_YAW_DEG = {0: 90.0, 1: 0.0, 2: -90.0, 3: 180.0}


def _wrap_deg(a: float) -> float:
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


def _install_sim_capture() -> Dict[str, object]:
    """MouseSim.__init__ をパッチし、生成されたインスタンスを捕まえる
    （このプロセス内でのみ有効。実行終了で消える。恒久的な変更ではない）。"""
    import mouse.sim as sim_mod

    holder: Dict[str, object] = {}
    orig_init = sim_mod.MouseSim.__init__

    def patched_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        holder["sim"] = self

    sim_mod.MouseSim.__init__ = patched_init
    return holder


def _make_tracing_policy(fast_mode: str):
    from classic.explorer import ClassicExplorer
    from classic.policy import ClassicExplorerPolicy

    write_log: List[Dict] = []
    sim_holder = _install_sim_capture()

    class TracingExplorer(ClassicExplorer):
        def __init__(self, *a, **kw) -> None:
            super().__init__(*a, **kw)
            self._tick_n = 0

        def tick(self, obs):
            self._tick_n += 1
            return super().tick(obs)

        def _update_map_from_sensing(self, sensing) -> None:
            cell_believed = tuple(self.cell)
            heading_believed = int(self.heading)
            sim = sim_holder.get("sim")
            pose = sim.privileged_pose() if sim is not None else None
            v_before = self.maze.v_walls.copy()
            h_before = self.maze.h_walls.copy()
            super()._update_map_from_sensing(sensing)

            def _log(arr_name, before, after):
                for idx in zip(*np.where(before != after)):
                    write_log.append({
                        "tick": self._tick_n, "arr": arr_name,
                        "idx": (int(idx[0]), int(idx[1])),
                        "new_state": int(after[idx]),
                        "cell_believed": cell_believed, "heading_believed": heading_believed,
                        "sensing": {"front": sensing.front.name, "left": sensing.left.name,
                                    "right": sensing.right.name},
                        "pose_true": None if pose is None else
                                     [float(pose[0]), float(pose[1]), float(pose[2])],
                    })

            _log("v", v_before, self.maze.v_walls)
            _log("h", h_before, self.maze.h_walls)

    class TracingPolicy(ClassicExplorerPolicy):
        def on_maze_start(self, maze_info: dict) -> None:
            width = int(maze_info["width"])
            height = int(maze_info["height"])
            self._explorer = TracingExplorer(
                width, height, params=self.params,
                extend_straights=self.extend_straights,
                fast_mode=self.fast_mode,
                friction_use=self.friction_use,
                clearance_margin_m=self.clearance_margin_m,
                wall_correction=self.wall_correction,
            )
            self._plan_ids = []
            self._run_phases = []

    return TracingPolicy(fast_mode=fast_mode), write_log


def rerun_with_trace(seed: int, fast_mode: str = "command",
                      time_budget: float = 1500.0, max_runs: int = 5):
    from competition.evaluator import CompetitionEvaluator

    policy, write_log = _make_tracing_policy(fast_mode)
    ev = CompetitionEvaluator(maze_dir=str(MAZE_DIR), time_budget=time_budget, max_runs=max_runs)
    result = ev.evaluate_maze(MAZE_DIR / f"maze_{seed}.npz", policy)
    learned = {
        "v_walls_known": policy.v_walls_known.tolist(),
        "h_walls_known": policy.h_walls_known.tolist(),
    }
    return result, write_log, learned


def analyze_fatal(fatal_locs: List[Tuple[str, Tuple[int, int]]], write_log: List[Dict]) -> List[Dict]:
    reports = []
    for arr, idx in fatal_locs:
        idx_t = tuple(idx)
        matches = [e for e in write_log if e["arr"] == arr and tuple(e["idx"]) == idx_t
                   and e["new_state"] == 2]  # 2 = WallState.OPEN
        if not matches:
            reports.append({"wall": (arr, idx_t), "found": False})
            continue
        e = matches[-1]  # 最後の書き込み＝最終状態(OPEN)を作った書き込み
        cx, cy = e["cell_believed"]
        heading = e["heading_believed"]
        expected_center = ((cx + 0.5) * CELL_SIZE, (cy + 0.5) * CELL_SIZE)
        expected_yaw_deg = _HEADING_YAW_DEG[heading]
        expected_yaw = math.radians(expected_yaw_deg)
        pose = e["pose_true"]
        if pose is None:
            reports.append({"wall": (arr, idx_t), "found": True, "pose_missing": True})
            continue
        ax, ay, ayaw = pose
        dx = ax - expected_center[0]
        dy = ay - expected_center[1]
        fwd = dx * math.cos(expected_yaw) + dy * math.sin(expected_yaw)
        lat = -dx * math.sin(expected_yaw) + dy * math.cos(expected_yaw)
        actual_cell = (int(ax // CELL_SIZE), int(ay // CELL_SIZE))
        yaw_err_deg = _wrap_deg(math.degrees(ayaw) - expected_yaw_deg)
        reports.append({
            "wall": (arr, idx_t), "found": True, "tick": e["tick"],
            "cell_believed": (cx, cy), "actual_cell": actual_cell,
            "cell_match": (actual_cell == (cx, cy)),
            "heading_believed": heading,
            "forward_offset_mm": fwd * 1000.0, "lateral_offset_mm": lat * 1000.0,
            "yaw_error_deg": yaw_err_deg,
            "sensing": e["sensing"],
        })
    return reports


def trace_one(row: Dict) -> Dict:
    """1 迷路ぶんの再実行＋解析（multiprocessing のワーカー関数。picklable にする
    ため top-level 関数にしてある）。"""
    seed = row["seed"]
    result, write_log, learned = rerun_with_trace(seed)

    orig = json.loads((OUT_ROOT / f"maze_{seed}.json").read_text(encoding="utf-8"))
    same_v = learned["v_walls_known"] == orig["v_walls_known"]
    same_h = learned["h_walls_known"] == orig["h_walls_known"]

    fatal_locs = [tuple(loc) for loc in row["fatal_locs"]]
    reports = analyze_fatal(fatal_locs, write_log)
    return {"seed": seed, "deterministic": bool(same_v and same_h), "reports": reports}


def main(argv=None) -> int:
    import multiprocessing as mp
    import os

    summary_path = OUT_ROOT / "summary.json"
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    targets = [r for r in rows if r["fatal"] > 0]
    if not targets:
        print("致命的な誤りを持つ迷路は無い（summary.json）。トレース対象なし。")
        return 0

    print(f"致命的な誤りを持つ {len(targets)} 迷路を再実行してトレースする "
          f"（seed={[r['seed'] for r in targets]}）。")

    n_workers = max(1, min(len(targets), (os.cpu_count() or 2) - 2))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        outcomes = pool.map(trace_one, targets)

    all_reports = {}
    for outcome in sorted(outcomes, key=lambda o: o["seed"]):
        seed = outcome["seed"]
        print(f"\n=== seed={seed} ===")
        print(f"決定性検算: {'一致' if outcome['deterministic'] else '🔴 不一致（以下は再実行時の地図に基づく）'}")
        all_reports[seed] = outcome["reports"]
        for rep in outcome["reports"]:
            arr, idx = rep["wall"]
            if not rep.get("found"):
                print(f"  壁 {arr}{idx}: 書き込み記録が見つからない（想定外）")
                continue
            if rep.get("pose_missing"):
                print(f"  壁 {arr}{idx}: 真値位置が取得できなかった（想定外）")
                continue
            print(f"  壁 {arr}{idx} (tick={rep['tick']}): "
                  f"信じていた区画={rep['cell_believed']} 実際の区画={rep['actual_cell']} "
                  f"(一致={rep['cell_match']}) heading={rep['heading_believed']} "
                  f"前後ずれ={rep['forward_offset_mm']:+.1f}mm 横ずれ={rep['lateral_offset_mm']:+.1f}mm "
                  f"方位ずれ={rep['yaw_error_deg']:+.2f}deg センサ={rep['sensing']}")

    trace_path = OUT_ROOT / "trace_fatal.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in all_reports.items()}, f, indent=2, ensure_ascii=False)
    print(f"\nトレース結果を保存: {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
