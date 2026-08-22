"""
research_notes/scripts/video_sensor_stage1.py
================
発表用動画（センサの読みの可視化）段階1: 走行データを作る。

やること（タスク指示のとおり）:
  1. 調整用の帯（competition namespace の 'free'、41000 以降）から未使用の seed を選び、
     `competition/maze_gen_turnmix.py` の仕組みで迷路を1面新しく生成する
     （🔴 評価用に予約された seed 1000〜40999 は使用禁止。common/seed_bands.py で確認する）。
  2. `classic/policy.py::ClassicExplorerPolicy` を `competition/evaluator.py` の
     `CompetitionEvaluator` に通して走らせる（方策には一切手を入れない。
     `experiments/exp_029_full_protocol/run_exp029.py` と同じ「初めて完走が出た組み合わせ」
     をそのまま使う）。
  3. 制御周期（100Hz）ごとに、模擬時刻・真の姿勢(x,y,θ)・機体速度・走行の段階
     （探索/帰還/最短走行）・走行回数を記録する。
  4. 記録した真の姿勢から、センサ4本の読みを `mouse/ir_sensor.py::response_table()`
     （表は `mouse/ir_table.py::load_cumulative_table()`）で計算する。
     🔴 ロボットの真の状態は既知という前提なので、模擬から読んだ真の姿勢をそのまま使う
     （方策が推定した位置ではない）。
  5. 記録を outputs/video_sensor/run_data.npz + run_meta.json に保存する。

🔴 `mouse/`・`classic/`・`competition/` のコードは一切変更しない（読み取り・import のみ）。
   `CompetitionEvaluator.evaluate_maze()` の内部ループには手が届かないので、
   `competition.evaluator` モジュールの `MouseSim` 参照だけを実行時に（このスクリプトの
   プロセス内でだけ）サブクラスへ差し替える「モンキーパッチ」で1ステップごとの記録を行う。
   `MouseSim.step_control()` 自体はオーバーライドしても中身は素通しで
   `super().step_control()` を呼ぶだけであり、物理・電圧計算には一切触れない。

センサ計算の重さ対策: 1 迷路の全制御ステップ（持ち時間 420 秒なら最大 42000+ 点）に
4本ぶんの `response_table()`（実測 1〜2ms/回）をすべて掛けると数分〜十数分かかりうる
（`verification/AUDIT_060_RESULT.md` 実測: 実際の走行姿勢で 1.134ms/回）。10分以内に
収める制約があるため、段階1では等間隔の間引き標本（既定上限 20000 点）で計算し、
その範囲（最小/中央値/最大）を報告する。動画に使う抜粋（2〜3分ぶん）の全点計算は
段階2で行う（そちらは対象点数が少ないので軽い）。

前景で最後まで走らせること（10分以内に収まるよう分割してよいが、バックグラウンドへは
投げない）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classic.geometry import wall_obstacles  # noqa: E402
from classic.policy import ClassicExplorerPolicy  # noqa: E402
from common.seed_bands import assert_seeds_allowed, describe_seeds  # noqa: E402
from competition.maze_gen_turnmix import (  # noqa: E402
    generate_turnmix_set,
    load_excluded_seeds,
)
from mouse.ir_sensor import (  # noqa: E402
    IrSensorSpec,
    SurfaceSpec,
    build_maze_cell_index,
    response_table,
)
from mouse.ir_table import load_cumulative_table  # noqa: E402
from mouse.mjcf import build_maze_robot_xml  # noqa: E402
from mouse.params import RobotParams  # noqa: E402
from mouse.sim import MouseSim  # noqa: E402

# 満量（正規化の基準。`verification/AUDIT_050_PREREG_ir_raycast.md` §2-2 以来、
# リポジトリ内の全監査スクリプトが使っている固定値。ここでもそれを踏襲する
# — 姿勢の標本に依存しない固定値なので、動画のセンサ計器の「満量比」もこれで割る）。
I_FULL = 0.8298934

DEFAULT_MAZE_OUT_DIR = REPO_ROOT / "outputs" / "video_sensor" / "maze"
DEFAULT_DATA_OUT = REPO_ROOT / "outputs" / "video_sensor" / "run_data.npz"
DEFAULT_META_OUT = REPO_ROOT / "outputs" / "video_sensor" / "run_meta.json"
DEFAULT_TABLE_PATH = REPO_ROOT / "mouse" / "data" / "ir_cumulative_table.npz"
DEFAULT_EXCLUDE_MANIFEST = REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "manifest.json"

# exp_029（`experiments/exp_029_full_protocol/run_exp029.py`）で「初めて完走が出た組み合わせ」
# をそのまま使う（本タスクは方策・評価器に手を入れないので、既知でうまく動くパラメータを選ぶ）。
FRICTION_USE = 0.30
CLEARANCE_MARGIN_M = 0.025
WALL_CORRECTION = True
FAST_MODE = "profile"


# ==========================================================================
# 記録用フック（プロセス内のみのモンキーパッチ。ファイルは一切書き換えない）
# ==========================================================================
_TRACE: list = []
_POLICY_REF = {"policy": None}


class RecordingMouseSim(MouseSim):
    """`step_control()` の直後（＝評価器の主ループが真の姿勢を読みに行くのと同じ時点）に
    真の姿勢・速度・方策の段階・走行番号を `_TRACE` へ積むだけのラッパー。
    `step_control()` 自体は `super().step_control()` を呼ぶだけで中身に一切触れない。"""

    def step_control(self, v_left: float, v_right: float) -> dict:
        result = super().step_control(v_left, v_right)
        x, y, yaw = self.privileged_pose()
        v_fwd, omega_z = self.privileged_velocity()
        pol = _POLICY_REF["policy"]
        phase = pol.current_phase if pol is not None else "INIT"
        run_index = pol.current_run_index if pol is not None else 0
        run_active = pol.current_run_active if pol is not None else False
        _TRACE.append((
            self.sim_time, x, y, yaw, v_fwd, omega_z,
            1.0 if result["collision"] else 0.0,
            1.0 if result["tipped"] else 0.0,
            phase, run_index, 1.0 if run_active else 0.0,
        ))
        return result


class RecordingPolicy(ClassicExplorerPolicy):
    """`ClassicExplorerPolicy` の薄いサブクラス。`act()`/`on_run_start`/`on_run_end` の
    呼び出しはそのまま素通しし、`RecordingMouseSim` が読みに行く現在の段階・走行番号・
    走行中フラグを更新するだけ（電圧計算・状態機械には一切触れない。
    `experiments/exp_029_full_protocol/run_exp029.py::RecordingPolicy` と同じ作法）。"""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.current_phase = "INIT"
        self.current_run_index = 0
        self.current_run_active = False

    def on_maze_start(self, maze_info: dict) -> None:
        super().on_maze_start(maze_info)
        _POLICY_REF["policy"] = self
        self.current_phase = self._explorer.phase.name
        self.current_run_index = 0
        self.current_run_active = False

    def on_run_start(self, run_index: int) -> None:
        super().on_run_start(run_index)
        self.current_run_index = int(run_index)
        self.current_run_active = True

    def on_run_end(self, outcome: str) -> None:
        super().on_run_end(outcome)
        self.current_run_active = False

    def act(self, obs: np.ndarray):
        vl, vr = super().act(obs)
        if self._explorer is not None:
            self.current_phase = self._explorer.phase.name
        return vl, vr


# ==========================================================================
# 迷路生成
# ==========================================================================
def make_maze(seed_start: int, exclude_manifest: Path, maze_out_dir: Path, max_scan: int):
    excluded = load_excluded_seeds(exclude_manifest)
    print(f"[maze] seed_start={seed_start} 除外manifest={exclude_manifest}"
          f"（除外seed数={len(excluded)}）", flush=True)
    accepted, n_scanned = generate_turnmix_set(
        seed_start=seed_start, count=1, max_scan=max_scan, excluded_seeds=excluded,
    )
    seed, v, h, gen_info, metrics, commands = accepted[0]
    print(describe_seeds([seed], namespace="competition"), flush=True)
    # 念のための二重の安全弁（generate_turnmix_set 内部でも確認済みだが、明示的に確認する）。
    assert_seeds_allowed([seed], namespace="competition", purpose="validate")

    maze_out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = maze_out_dir / f"maze_{seed}.npz"
    np.savez(npz_path, v_walls=v, h_walls=h, seed=seed, width=v.shape[0] - 1, height=v.shape[1])
    build_maze_robot_xml(v, h, str(npz_path.with_suffix(".xml")),
                          model_name=f"maze_{seed}", params=RobotParams())
    print(f"[maze] 採用 seed={seed}（走査{n_scanned}個） "
          f"最短{gen_info['d_shortest']}区画(D0={gen_info['d0']}) "
          f"右90={metrics['right90']} 左90={metrics['left90']} "
          f"連続ターン={metrics['consecutive_turns']} 長い直線={metrics['long_straights']}",
          flush=True)
    return seed, v, h, gen_info, metrics, npz_path


# ==========================================================================
# 走行 + 記録
# ==========================================================================
def run_and_record(npz_path: Path, maze_out_dir: Path, time_budget: float, max_runs: int):
    import competition.evaluator as ev_mod

    # このプロセス内でだけ MouseSim を差し替える（ファイルは書き換えない）。
    ev_mod.MouseSim = RecordingMouseSim

    _TRACE.clear()
    _POLICY_REF["policy"] = None

    ev = ev_mod.CompetitionEvaluator(
        maze_dir=str(maze_out_dir), time_budget=time_budget, max_runs=max_runs,
    )
    policy = RecordingPolicy(
        fast_mode=FAST_MODE, friction_use=FRICTION_USE,
        clearance_margin_m=CLEARANCE_MARGIN_M, wall_correction=WALL_CORRECTION,
    )
    print(f"[run] time_budget={time_budget}s max_runs={max_runs} "
          f"friction_use={FRICTION_USE} clearance_margin_m={CLEARANCE_MARGIN_M} "
          f"wall_correction={WALL_CORRECTION} fast_mode={FAST_MODE}", flush=True)

    t0 = time.time()
    result = ev.evaluate_maze(npz_path, policy)
    t_physics = time.time() - t0

    trace = _TRACE.copy()
    _TRACE.clear()
    print(f"[run] 完了: 記録点数={len(trace)} 実行時間={t_physics:.1f}s "
          f"n_runs={len(result['runs'])} best_time={result['best_time']}", flush=True)
    return result, trace, t_physics


def trace_to_arrays(trace: list) -> dict:
    n = len(trace)
    sim_time = np.empty(n, dtype=np.float64)
    x = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    yaw = np.empty(n, dtype=np.float64)
    v_fwd = np.empty(n, dtype=np.float64)
    omega_z = np.empty(n, dtype=np.float64)
    collision = np.empty(n, dtype=np.uint8)
    tipped = np.empty(n, dtype=np.uint8)
    phase = np.empty(n, dtype="<U8")
    run_index = np.empty(n, dtype=np.int32)
    run_active = np.empty(n, dtype=np.uint8)
    for i, row in enumerate(trace):
        (sim_time[i], x[i], y[i], yaw[i], v_fwd[i], omega_z[i],
         collision[i], tipped[i], phase[i], run_index[i], run_active[i]) = row
    return dict(
        sim_time=sim_time, x=x, y=y, yaw=yaw, v_fwd=v_fwd, omega_z=omega_z,
        collision=collision, tipped=tipped, phase=phase,
        run_index=run_index, run_active=run_active,
    )


# ==========================================================================
# センサ計算（間引き標本。段階1の報告用）
# ==========================================================================
def build_ir_specs(p: RobotParams) -> list:
    specs = []
    for s in p.sensors:
        pos = tuple(float(v) for v in s["pos"].split())
        axis = tuple(float(v) for v in s["zaxis"].split())
        specs.append(IrSensorSpec(name=s["name"], pos=pos, axis=axis))
    return specs


def compute_sensor_subsample(arrays: dict, v_walls, h_walls, cell_size: float,
                              table_path: Path, cap: int):
    n = len(arrays["sim_time"])
    if n <= cap:
        idx = np.arange(n, dtype=np.int64)
    else:
        idx = np.unique(np.linspace(0, n - 1, cap, dtype=np.int64))

    p = RobotParams()
    specs = build_ir_specs(p)
    surf = SurfaceSpec()
    rects = wall_obstacles(v_walls, h_walls, cell_size=cell_size)
    cell_index = build_maze_cell_index(rects, cell_size)
    table = load_cumulative_table(table_path)

    m = len(idx)
    raw = np.empty((m, len(specs)), dtype=np.float64)
    t0 = time.time()
    for row, i in enumerate(idx):
        pose = (float(arrays["x"][i]), float(arrays["y"][i]), float(arrays["yaw"][i]))
        for si, spec in enumerate(specs):
            raw[row, si] = response_table(spec, pose, rects, surf, table, cell_index, cell_size)
        if (row + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  [sensor] {row+1}/{m} 点 ({elapsed:.1f}s, {elapsed/(row+1)*1000:.3f}ms/点)",
                  flush=True)
    t_sensor = time.time() - t0
    ratio = raw / I_FULL
    return dict(
        subsample_idx=idx, sensor_names=[s.name for s in specs],
        sensor_raw=raw, sensor_ratio=ratio, t_sensor=t_sensor,
    )


# ==========================================================================
# メイン
# ==========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed-start", type=int, default=41200)
    ap.add_argument("--max-scan", type=int, default=500)
    ap.add_argument("--exclude-manifest", type=Path, default=DEFAULT_EXCLUDE_MANIFEST)
    ap.add_argument("--maze-out-dir", type=Path, default=DEFAULT_MAZE_OUT_DIR)
    ap.add_argument("--time-budget", type=float, default=420.0)
    ap.add_argument("--max-runs", type=int, default=5)
    ap.add_argument("--sensor-subsample-cap", type=int, default=20000)
    ap.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH)
    ap.add_argument("--data-out", type=Path, default=DEFAULT_DATA_OUT)
    ap.add_argument("--meta-out", type=Path, default=DEFAULT_META_OUT)
    args = ap.parse_args(argv)

    t_start_all = time.time()

    seed, v_walls, h_walls, gen_info, turn_metrics, npz_path = make_maze(
        args.seed_start, args.exclude_manifest, args.maze_out_dir, args.max_scan,
    )
    t_maze = time.time() - t_start_all

    result, trace, t_physics = run_and_record(
        npz_path, args.maze_out_dir, args.time_budget, args.max_runs,
    )
    arrays = trace_to_arrays(trace)

    cell_size = RobotParams().cell_size
    sensor = compute_sensor_subsample(
        arrays, v_walls, h_walls, cell_size, args.table_path, args.sensor_subsample_cap,
    )

    args.data_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.data_out,
        seed=seed, cell_size=cell_size,
        v_walls=v_walls, h_walls=h_walls,
        width=v_walls.shape[0] - 1, height=v_walls.shape[1],
        i_full=I_FULL,
        **arrays,
        subsample_idx=sensor["subsample_idx"],
        sensor_names=np.array(sensor["sensor_names"]),
        sensor_raw=sensor["sensor_raw"],
        sensor_ratio=sensor["sensor_ratio"],
    )

    t_total = time.time() - t_start_all
    meta = dict(
        seed=int(seed), gen_info=gen_info, turn_metrics=turn_metrics,
        time_budget=args.time_budget, max_runs=args.max_runs,
        friction_use=FRICTION_USE, clearance_margin_m=CLEARANCE_MARGIN_M,
        wall_correction=WALL_CORRECTION, fast_mode=FAST_MODE,
        n_steps_recorded=int(len(arrays["sim_time"])),
        sim_time_final_s=float(arrays["sim_time"][-1]) if len(arrays["sim_time"]) else 0.0,
        n_sensor_subsample=int(len(sensor["subsample_idx"])),
        i_full=I_FULL,
        sensor_ratio_min=float(np.min(sensor["sensor_ratio"])),
        sensor_ratio_median=float(np.median(sensor["sensor_ratio"])),
        sensor_ratio_max=float(np.max(sensor["sensor_ratio"])),
        timings_s=dict(maze_gen=t_maze, physics=t_physics,
                        sensor=sensor["t_sensor"], total=t_total),
        evaluator_result=result,
    )
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ---- 報告に必要な要約を印字 ----
    runs = result["runs"]
    phase_counts = {}
    for ph in arrays["phase"]:
        phase_counts[ph] = phase_counts.get(ph, 0) + 1
    print("\n===== 段階1 報告用サマリ =====")
    print(f"seed={seed}（namespace=competition, band=free。予約帯 1000-40999 の外）")
    print(f"迷路: 最短{gen_info['d_shortest']}区画(D0={gen_info['d0']})")
    print(f"time_budget={args.time_budget}s max_runs={args.max_runs}")
    print(f"走行結果: n_runs={len(runs)} best_time={result['best_time']}")
    for r in runs:
        print(f"  run{r['index']}: outcome={r['outcome']} "
              f"t_start={r['t_start']:.2f} t_end={r['t_end']:.2f} "
              f"run_time={r['run_time']}")
    print(f"記録点数={len(arrays['sim_time'])} 模擬時間の長さ={arrays['sim_time'][-1]:.2f}s")
    print(f"段階別の記録点数内訳: {phase_counts}")
    print(f"センサ計算: 間引き標本点数={len(sensor['subsample_idx'])} "
          f"(上限{args.sensor_subsample_cap})")
    print(f"満量比 min={meta['sensor_ratio_min']:.4f} "
          f"median={meta['sensor_ratio_median']:.4f} max={meta['sensor_ratio_max']:.4f}")
    print(f"所要時間: 迷路生成={t_maze:.1f}s 走行(物理)={t_physics:.1f}s "
          f"センサ計算={sensor['t_sensor']:.1f}s 合計={t_total:.1f}s")
    print(f"保存先: {args.data_out}")
    print(f"保存先(メタ): {args.meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
