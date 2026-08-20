"""
experiments/exp_027_friction_sweep/diagnose_retrieval.py
================
exp_027 の一次記録で観測された「最初の FAST（最短走行）だけ他と違う／2本目以降は
互いに完全に同一」という型を切り分けるための診断スクリプト（教授セッションの
委譲指示: `handover/professor.md` 相当の作業依頼を参照）。

対象: maze_41004・u=0.50・wall_correction=True（`outputs/exp_027/u_0.50/wc_on/
maze_41004.json` の run3=13.216m・run4/5=1.538m が再現対象）。

やること:
    `competition.evaluator.CompetitionEvaluator.evaluate_maze` を**そのまま**
    呼び、`classic.explorer.ClassicExplorer._begin_profile_run`
    （FAST/RETURN2 の profile 計画を作る箇所）と `mouse.sim.MouseSim.step_control`
    をモンキーパッチで包んで、1本目・2本目の FAST が始まる瞬間に次を記録する:
      - `FastPlan`（t_plan・cells 長・n_turns・n_forced_spins・先頭ステップ種別）
      - 呼び出しに使われた `start_heading`
      - `ProfileTracker` の内部状態（呼び出し **直前**の値。reset() で上書きされる
        前を見る — 持ち越しの有無を直接確認するため）
      - `Localizer.events` の件数（呼び出し直前）
      - 機体の真値姿勢・速度（`sim.privileged_pose()`/`privileged_velocity()`。
        🔴 診断スクリプトの中でだけ読む。`classic/` 側の実装は一切真値を読まない）

    評価そのもの（`evaluate_maze` の中身）は一切変更しない。3本目の FAST 計画まで
    記録できたら `_StopEarly` 例外で打ち切る（4本目以降は必要ないため）。

🔴 本スクリプトは診断専用。判定・修正は行わない（`PREREG.md` 参照）。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MAZE_DIR = REPO_ROOT / "competition" / "mazes" / "design_turn_v1"
MAZE_NPZ = MAZE_DIR / "maze_41004.npz"
U = 0.50
WALL_CORRECTION = True
N_CALLS_TO_CAPTURE = 3  # run3(1本目)・run4(2本目)・run5(3本目=2本目と比較する対照)


class _StopEarly(Exception):
    """診断に必要な回数だけ `_begin_profile_run` を捕まえたら評価ループを打ち切る合図。"""


def main() -> int:
    from classic.explorer import ClassicExplorer
    from classic.policy import ClassicExplorerPolicy
    from competition.evaluator import CompetitionEvaluator
    from mouse.sim import MouseSim

    captured_sim = {}
    calls = []

    orig_step_control = MouseSim.step_control

    def patched_step_control(self, v_left, v_right):
        # sim インスタンスを捕まえるためだけのフック（挙動は一切変えない）。
        captured_sim["sim"] = self
        return orig_step_control(self, v_left, v_right)

    orig_begin_profile_run = ClassicExplorer._begin_profile_run

    def patched_begin_profile_run(self, obs, start, goals, start_heading):
        sim = captured_sim.get("sim")
        tracker = self._tracker
        pre_tracker = None
        if tracker is not None:
            pre_tracker = dict(
                yaw_est_deg=__import__("math").degrees(tracker.yaw_estimate),
                s=tracker.s,
                t=tracker._t,
                e_lat=tracker._e_lat,
                integ_l=tracker._integ_l,
                integ_r=tracker._integ_r,
                mode=tracker._mode,
            )
        localizer_events_before = len(self.localizer.events)
        true_pose = sim.privileged_pose() if sim is not None else None
        true_vel = sim.privileged_velocity() if sim is not None else None

        orig_begin_profile_run(self, obs, start, goals, start_heading)

        plan = self._fast_plan
        info = dict(
            call_idx=len(calls) + 1,
            phase=self.phase.name,
            start=start,
            goals=list(goals),
            start_heading=str(start_heading),
            heading_int=int(start_heading),
            fallback=self._fast_command_fallback,
            pre_tracker=pre_tracker,
            localizer_events_before=localizer_events_before,
            true_pose=true_pose,
            true_vel=true_vel,
        )
        if plan is not None:
            first_step = plan.steps[0] if plan.steps else None
            info.update(
                t_plan=plan.t_plan,
                n_cells=len(plan.cells),
                n_turns=plan.n_turns,
                n_forced_spins=plan.n_forced_spins,
                first_step_kind=type(first_step).__name__ if first_step is not None else None,
                cells_head=plan.cells[:8],
                cells_tail=plan.cells[-8:],
            )
        calls.append(info)

        if len(calls) >= N_CALLS_TO_CAPTURE:
            raise _StopEarly()

    MouseSim.step_control = patched_step_control
    ClassicExplorer._begin_profile_run = patched_begin_profile_run

    try:
        evaluator = CompetitionEvaluator(
            maze_dir=str(MAZE_DIR), time_budget=1500.0, max_runs=5,
        )
        policy = ClassicExplorerPolicy(fast_mode="profile", friction_use=U, wall_correction=WALL_CORRECTION)
        try:
            evaluator.evaluate_maze(MAZE_NPZ, policy)
        except _StopEarly:
            pass
    finally:
        MouseSim.step_control = orig_step_control
        ClassicExplorer._begin_profile_run = orig_begin_profile_run

    # ------------------------------------------------------------------
    # 報告
    # ------------------------------------------------------------------
    print(f"maze={MAZE_NPZ.name} u={U} wall_correction={WALL_CORRECTION}")
    print(f"捕捉した _begin_profile_run 呼び出し回数: {len(calls)}\n")
    for c in calls:
        print(f"=== 呼び出し #{c['call_idx']} (この呼び出しで phase={c['phase']} の計画を作る) ===")
        print(f"  start={c['start']} goals={c['goals']} start_heading={c['start_heading']}(int={c['heading_int']})")
        print(f"  fallback(コマンド方式へ落ちた)={c['fallback']}")
        if "t_plan" in c:
            print(f"  FastPlan: t_plan={c['t_plan']:.4f}s n_cells={c['n_cells']} n_turns={c['n_turns']} "
                  f"n_forced_spins={c['n_forced_spins']} first_step={c['first_step_kind']}")
            print(f"  cells先頭8={c['cells_head']}")
            print(f"  cells末尾8={c['cells_tail']}")
        else:
            print("  FastPlan: None（trivialまたは計画失敗）")
        pt = c["pre_tracker"]
        if pt is not None:
            print(f"  ProfileTracker(直前の値。reset()で上書きされる前): "
                  f"mode={pt['mode']} yaw_est={pt['yaw_est_deg']:.3f}deg s={pt['s']:.6f}m t={pt['t']:.4f}s "
                  f"e_lat={pt['e_lat']:.6f}m integ_l={pt['integ_l']:.4f} integ_r={pt['integ_r']:.4f}")
        else:
            print("  ProfileTracker: 未構築(wall_correction/fast_mode設定を確認)")
        print(f"  Localizer.events件数(直前まで累積): {c['localizer_events_before']}")
        if c["true_pose"] is not None:
            x, y, yaw = c["true_pose"]
            v_fwd, om = c["true_vel"]
            import math
            print(f"  真値姿勢: x={x:.5f}m y={y:.5f}m yaw={math.degrees(yaw):.3f}deg")
            print(f"  真値速度: v_fwd={v_fwd:.5f}m/s omega={om:.5f}rad/s")
        print()

    # ------------------------------------------------------------------
    # 呼び出し#1(1本目=run3) と #2(2本目=run4) の差分だけを抜き出して比較
    # ------------------------------------------------------------------
    if len(calls) >= 2:
        a, b = calls[0], calls[1]
        print("=== 呼び出し#1 と #2 の差分 ===")
        for key in ("start_heading", "t_plan", "n_cells", "n_turns", "n_forced_spins", "first_step_kind"):
            va, vb = a.get(key), b.get(key)
            mark = "≠" if va != vb else "="
            print(f"  {key}: #1={va} {mark} #2={vb}")
        if a["true_pose"] is not None and b["true_pose"] is not None:
            xa, ya, yawa = a["true_pose"]
            xb, yb, yawb = b["true_pose"]
            va_fwd, oma = a["true_vel"]
            vb_fwd, omb = b["true_vel"]
            import math
            print(f"  真値位置差: dx={xb - xa:+.5f}m dy={yb - ya:+.5f}m dyaw={math.degrees(yawb - yawa):+.3f}deg")
            print(f"  真値速度差: dv_fwd={vb_fwd - va_fwd:+.5f}m/s domega={omb - oma:+.5f}rad/s")
        if a["pre_tracker"] is not None and b["pre_tracker"] is not None:
            same = a["pre_tracker"] == b["pre_tracker"]
            print(f"  ProfileTracker直前状態が完全一致: {same}"
                  f"（一致するのが自然 — どちらも直前の走行の名残であり reset() 前）")
        print(f"  Localizer.events件数: #1={a['localizer_events_before']} #2={b['localizer_events_before']}")

    if len(calls) >= 3:
        b, c = calls[1], calls[2]
        print("\n=== 呼び出し#2 と #3 の差分（2本目以降が同一という主張の直接確認） ===")
        for key in ("start_heading", "t_plan", "n_cells", "n_turns", "n_forced_spins", "first_step_kind"):
            vb, vc = b.get(key), c.get(key)
            mark = "≠" if vb != vc else "="
            print(f"  {key}: #2={vb} {mark} #3={vc}")
        if b["true_pose"] is not None and c["true_pose"] is not None:
            print(f"  真値姿勢: #2={b['true_pose']} #3={c['true_pose']}")
            print(f"  真値速度: #2={b['true_vel']} #3={c['true_vel']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
