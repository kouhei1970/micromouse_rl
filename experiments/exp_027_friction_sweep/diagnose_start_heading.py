"""
experiments/exp_027_friction_sweep/diagnose_start_heading.py
================
`diagnose_retrieval.py`（46d9183）で「開始方位の実在する違い」までは確定した。
本スクリプトはその続きの診断: **なぜ「先頭にその場旋回が無い計画」（北向き開始・
run4/5）のほうが「先頭にその場旋回がある計画」（南向き開始・run3）より脆いのか**を、
推測（計画上いるべき位置）と真値（真の位置）の弧長ごとのずれを実測して切り分ける。

対象: maze_41004・u=0.50・wall_correction=True（run3=南向き開始・run4=北向き開始）。

やること（教授セッションからの委譲指示どおり）:
    `_begin_profile_run` を包んで call#1(=run3・南)・call#2(=run4・北) の
    `FastPlan` を捕まえ、その FastPlan と同じ幾何（`classic.ideal._geometry_blocks`
    と `classic.ideal.ideal_time_for_path` を計画時と同じ入力で呼び直して再構成する
    — 半径探索・比例配分は決定的なので同じ答えになる。真値は一切使わない計画側の
    再計算）から「計画上、弧長 s のときにいるべき絶対姿勢」を求める。

    `_tick_profile` と `MouseSim.step_control` も包み、ティックごとに
    (a) このティックの制御に使った推測航法の弧長 s・方位推定・目標(v_ref/kappa_ref
        /psi_ref/a_ref)、(b) このティックの obs のもとになった真値姿勢・速度
        （🔴 診断のためだけに読む。`classic/` 側の実装は真値を一切読まない）
    を対にして記録する。両者は同じ瞬間（このティックの obs が作られた瞬間）を
    指すので、「計画上いるべき位置」と「真値位置」を弧長に対してそのまま比較できる。

🔴 本スクリプトは診断専用。判定・修正は行わない。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MAZE_DIR = REPO_ROOT / "competition" / "mazes" / "design_turn_v1"
MAZE_NPZ = MAZE_DIR / "maze_41004.npz"
U = 0.50
WALL_CORRECTION = True
N_CALLS_TO_CAPTURE = 3
RECORD_CALL_IDXS = (1, 2)  # 1=run3(南向き開始)・2=run4(北向き開始)
N_HEAD_TICKS_DETAIL = 25   # 発進直後の詳細表示ティック数
SAMPLE_STRIDE = 20         # 全体の推移を印字する間引き間隔（ティック）


class _StopEarly(Exception):
    """診断に必要な回数だけ `_begin_profile_run` を捕まえたら評価ループを打ち切る合図。"""


def _pose_at_block_s(segments, start_pose, s_query: float):
    """`segments`（PathSegment列。直線/円弧）を `start_pose` から辿り、
    ブロック内弧長 `s_query` の時点の絶対姿勢を返す（`classic.geometry.poses_along`
    と同じ厳密解 `_pose_at` を使うが、任意の s を1点だけ問い合わせる版）。
    範囲外は端でクランプする（`classic.tracker._interp_linear` と同じ規約）。"""
    from classic.geometry import _pose_at

    pose = start_pose
    remaining = max(s_query, 0.0)
    for i, seg in enumerate(segments):
        is_last = (i == len(segments) - 1)
        if remaining <= seg.length or is_last:
            local_s = min(max(remaining, 0.0), seg.length)
            return _pose_at(pose, seg.curvature, local_s)
        pose = _pose_at(pose, seg.curvature, seg.length)
        remaining -= seg.length
    return pose


def main() -> int:
    from classic.explorer import ClassicExplorer
    from classic.policy import ClassicExplorerPolicy
    from classic.fast_planner import PathBlock, SpinSegment, _bool_walls_from_maze
    from classic.ideal import _geometry_blocks, _turns_and_runs, ideal_time_for_path
    from competition.evaluator import CompetitionEvaluator
    from mouse.sim import MouseSim

    captured_sim: Dict[str, object] = {}
    calls: List[dict] = []
    run_logs: Dict[int, dict] = {}
    state = {"active_call_idx": None, "pending": None}

    # ------------------------------------------------------------------
    # フック
    # ------------------------------------------------------------------
    orig_step_control = MouseSim.step_control

    def patched_step_control(self, v_left, v_right):
        captured_sim["sim"] = self
        pending = state["pending"]
        if pending is not None:
            x, y, yaw = self.privileged_pose()
            v_fwd, om = self.privileged_velocity()
            pending["true_x"] = x
            pending["true_y"] = y
            pending["true_yaw"] = yaw
            pending["true_v"] = v_fwd
            pending["true_omega"] = om
            pending["sim_time"] = float(self.data.time)
            call_idx = state["active_call_idx"]
            if call_idx is not None:
                run_logs[call_idx]["ticks"].append(pending)
            state["pending"] = None
        result = orig_step_control(self, v_left, v_right)
        call_idx = state["active_call_idx"]
        if call_idx is not None and result["collision"]:
            run_logs[call_idx]["collided_at_tick"] = len(run_logs[call_idx]["ticks"]) - 1
            state["active_call_idx"] = None
        return result

    orig_apply_wall_correction = ClassicExplorer._apply_wall_correction

    def patched_apply_wall_correction(self, obs):
        call_idx = state["active_call_idx"]
        if call_idx is None:
            return orig_apply_wall_correction(self, obs)
        tracker = self._tracker
        s_before = tracker.s
        ref_before = self._wc_fwd_ref_s
        e_lat_before = tracker._e_lat
        orig_apply_wall_correction(self, obs)
        s_after = tracker.s
        ref_after = self._wc_fwd_ref_s
        e_lat_after = tracker._e_lat
        if abs(s_after - s_before) > 1e-9 or ref_before != ref_after:
            run_logs[call_idx].setdefault("wc_events", []).append(dict(
                tick=len(run_logs[call_idx]["ticks"]), s_before=s_before, s_after=s_after,
                ds=s_after - s_before, ref_before=ref_before, ref_after=ref_after,
                e_lat_before=e_lat_before, e_lat_after=e_lat_after,
            ))
        return None

    orig_tick_profile = ClassicExplorer._tick_profile

    def patched_tick_profile(self, obs):
        call_idx = state["active_call_idx"]
        if call_idx is None:
            return orig_tick_profile(self, obs)
        tracker = self._tracker
        step_idx_before = self._fast_plan_step_idx
        mode_before = tracker._mode
        s_before = tracker.s
        yaw_before = tracker.yaw_estimate
        t_before = tracker._t
        e_lat_before = tracker._e_lat
        wc_ref_before = self._wc_fwd_ref_s

        if mode_before == "path":
            v_ff = tracker._v_at(s_before)
            kappa_ref = tracker._kappa_at(s_before)
            psi_ref = tracker._psi_at(s_before)
            a_ref = tracker._a_ref_at(s_before)
        else:
            psi_ref, omega_ref, alpha_ref = tracker._spin_kinematics(t_before)
            v_ff = 0.0
            kappa_ref = None
            a_ref = 0.0

        vl, vr, plan_id = orig_tick_profile(self, obs)

        rec = dict(
            tick=len(run_logs[call_idx]["ticks"]),
            step_idx=step_idx_before,
            mode=mode_before,
            s=s_before,
            yaw_est=yaw_before,
            t_spin=t_before,
            e_lat=e_lat_before,
            wc_ref_s=wc_ref_before,
            v_ff=v_ff,
            kappa_ref=kappa_ref,
            a_ref=a_ref,
            psi_ref=psi_ref,
            voltage_saturated=tracker.voltage_saturated,
            vl=vl, vr=vr,
        )
        state["pending"] = rec
        return vl, vr, plan_id

    orig_begin_profile_run = ClassicExplorer._begin_profile_run

    def patched_begin_profile_run(self, obs, start, goals, start_heading):
        state["active_call_idx"] = None
        orig_begin_profile_run(self, obs, start, goals, start_heading)
        plan = self._fast_plan
        call_idx = len(calls) + 1
        info = dict(
            call_idx=call_idx, phase=self.phase.name, start_heading=str(start_heading),
            fallback=self._fast_command_fallback,
        )
        if plan is not None:
            info.update(
                t_plan=plan.t_plan, n_cells=len(plan.cells), n_turns=plan.n_turns,
                n_forced_spins=plan.n_forced_spins,
                first_step_kind=type(plan.steps[0]).__name__ if plan.steps else None,
            )
        calls.append(info)

        if plan is not None and plan.steps and call_idx in RECORD_CALL_IDXS:
            v_walls, h_walls = _bool_walls_from_maze(self.maze)
            result = ideal_time_for_path(
                list(plan.cells), v_walls, h_walls, start_heading,
                mode="slalom", margin=0.005, allocation="best",
            )
            _turns, runs = _turns_and_runs(list(plan.cells), start_heading)
            geo_blocks, block_starts = _geometry_blocks(list(plan.cells), start_heading, result.turns, runs)
            spin_turns = [tp for tp in result.turns if tp.radius <= 0.0]

            # plan.steps と1対1対応する幾何参照列を作る（fast_planner.py の
            # steps 構築ループと同じ順序で辿るだけ。半径探索はしない）。
            step_geo_ref = []
            spin_i = 0
            for i, block in enumerate(geo_blocks):
                if block:
                    step_geo_ref.append(("path", block, block_starts[i]))
                if i < len(spin_turns):
                    step_geo_ref.append(("spin", None, block_starts[i + 1]))
                    spin_i += 1
            assert len(step_geo_ref) == len(plan.steps), (
                f"幾何参照列がFastPlan.stepsと対応しません: {len(step_geo_ref)} vs {len(plan.steps)}"
            )

            run_logs[call_idx] = dict(
                info=info, step_geo_ref=step_geo_ref, steps=plan.steps,
                cells=plan.cells, start_heading=str(start_heading), ticks=[],
            )
            state["active_call_idx"] = call_idx

        if len(calls) >= N_CALLS_TO_CAPTURE:
            raise _StopEarly()

    MouseSim.step_control = patched_step_control
    ClassicExplorer._tick_profile = patched_tick_profile
    ClassicExplorer._apply_wall_correction = patched_apply_wall_correction
    ClassicExplorer._begin_profile_run = patched_begin_profile_run

    try:
        evaluator = CompetitionEvaluator(maze_dir=str(MAZE_DIR), time_budget=1500.0, max_runs=5)
        policy = ClassicExplorerPolicy(fast_mode="profile", friction_use=U, wall_correction=WALL_CORRECTION)
        try:
            evaluator.evaluate_maze(MAZE_NPZ, policy)
        except _StopEarly:
            pass
    finally:
        MouseSim.step_control = orig_step_control
        ClassicExplorer._tick_profile = orig_tick_profile
        ClassicExplorer._apply_wall_correction = orig_apply_wall_correction
        ClassicExplorer._begin_profile_run = orig_begin_profile_run

    # ------------------------------------------------------------------
    # 解析・報告
    # ------------------------------------------------------------------
    print(f"maze={MAZE_NPZ.name} u={U} wall_correction={WALL_CORRECTION}")
    for c in calls:
        print(f"  call#{c['call_idx']}: phase={c['phase']} start_heading={c['start_heading']} "
              f"first_step={c.get('first_step_kind')} n_forced_spins={c.get('n_forced_spins')} "
              f"t_plan={c.get('t_plan')}")
    print()

    for call_idx in RECORD_CALL_IDXS:
        log = run_logs.get(call_idx)
        if log is None:
            print(f"=== call#{call_idx}: 記録なし ===\n")
            continue
        ticks = log["ticks"]
        collided_at = log.get("collided_at_tick")
        print(f"=== call#{call_idx} (start_heading={log['start_heading']}・"
              f"n_tick={len(ticks)}・collided_at_tick={collided_at}) ===")

        # このブロックの幾何(直線/円弧の並び)を、最初の弧が始まる弧長を特定するために印字する
        for step_i, (kind, block, ref_pose) in enumerate(log["step_geo_ref"]):
            if kind != "path":
                print(f"  step#{step_i}: SpinSegment")
                continue
            cum = 0.0
            seg_desc = []
            for seg in block:
                seg_desc.append(f"{seg.kind}[{cum:.4f}-{cum+seg.length:.4f}]kappa={seg.curvature:.3f}")
                cum += seg.length
            print(f"  step#{step_i}: PathBlock len={cum:.4f}m segments=" + " ".join(seg_desc[:12])
                  + (" ..." if len(seg_desc) > 12 else ""))

        wc_events = log.get("wc_events", [])
        print(f"  壁センサ補正が tracker._s を動かした回数: {len(wc_events)}")
        for ev in wc_events:
            print(f"    tick={ev['tick']:4d} s: {ev['s_before']:.4f}->{ev['s_after']:.4f} "
                  f"(ds={ev['ds']*1000:+.2f}mm) ref: {ev['ref_before']}->{ev['ref_after']}")
        print()

        # 各ティックに計画上の絶対姿勢を付与し、真値との差(縦=経路接線方向・横=法線方向)を計算
        errs = []
        for rec in ticks:
            kind, block, ref_pose = log["step_geo_ref"][rec["step_idx"]]
            if kind == "path":
                plan_pose = _pose_at_block_s(block, ref_pose, rec["s"])
            else:
                plan_pose = type(ref_pose)(ref_pose.x, ref_pose.y, rec["psi_ref"])
            dx = rec["true_x"] - plan_pose.x
            dy = rec["true_y"] - plan_pose.y
            th = plan_pose.theta
            longitudinal = math.cos(th) * dx + math.sin(th) * dy
            lateral = -math.sin(th) * dx + math.cos(th) * dy
            yaw_err_true_vs_plan = math.atan2(math.sin(rec["true_yaw"] - plan_pose.theta),
                                               math.cos(rec["true_yaw"] - plan_pose.theta))
            yaw_err_est_vs_true = math.atan2(math.sin(rec["yaw_est"] - rec["true_yaw"]),
                                              math.cos(rec["yaw_est"] - rec["true_yaw"]))
            errs.append(dict(
                rec=rec, plan_x=plan_pose.x, plan_y=plan_pose.y, plan_theta=plan_pose.theta,
                longitudinal=longitudinal, lateral=lateral,
                yaw_err_true_vs_plan=math.degrees(yaw_err_true_vs_plan),
                yaw_err_est_vs_true=math.degrees(yaw_err_est_vs_true),
            ))

        # (1) 発進直後 N_HEAD_TICKS_DETAIL ティックの詳細
        print(f"  --- 発進直後 {N_HEAD_TICKS_DETAIL} ティック ---")
        print("  tick mode   s[m]    v_ff   a_ref   vsat  true_v  lat_err[mm] lon_err[mm] "
              "yawErr(真-計)[deg] yawErr(推-真)[deg]  vl,vr")
        for e in errs[:N_HEAD_TICKS_DETAIL]:
            r = e["rec"]
            print(f"  {r['tick']:4d} {r['mode']:6s} {r['s']:.4f}  {r['v_ff'] if r['v_ff'] is not None else float('nan'):.4f} "
                  f"{r['a_ref']:.3f}  {str(r['voltage_saturated'])[0]}    {r['true_v']:.4f}  "
                  f"{e['lateral']*1000:8.2f}   {e['longitudinal']*1000:8.2f}     "
                  f"{e['yaw_err_true_vs_plan']:7.3f}          {e['yaw_err_est_vs_true']:7.3f}   "
                  f"{r['vl']:.3f},{r['vr']:.3f}")
        print()

        # (2) 全体の推移(間引き) — どこでずれ始めるか
        print(f"  --- 全体の推移（{SAMPLE_STRIDE}ティックごと。lat/lon は plan 基準の誤差[mm]） ---")
        print("  tick   s[m]   lat_err[mm]  lon_err[mm]  |pos_err|[mm]  yawErr(真-計)[deg]")
        for i in range(0, len(errs), SAMPLE_STRIDE):
            e = errs[i]
            r = e["rec"]
            pos_err = math.hypot(e["lateral"], e["longitudinal"]) * 1000
            print(f"  {r['tick']:5d} {r['s']:7.4f}  {e['lateral']*1000:9.2f}   {e['longitudinal']*1000:9.2f}   "
                  f"{pos_err:9.2f}     {e['yaw_err_true_vs_plan']:7.3f}")
        # 最後のティックも必ず出す
        if errs:
            e = errs[-1]
            r = e["rec"]
            pos_err = math.hypot(e["lateral"], e["longitudinal"]) * 1000
            print(f"  {r['tick']:5d} {r['s']:7.4f}  {e['lateral']*1000:9.2f}   {e['longitudinal']*1000:9.2f}   "
                  f"{pos_err:9.2f}     {e['yaw_err_true_vs_plan']:7.3f}  <- 最終")
        print()

        # (3) 発散の起点を機械的に特定: |lateral| が 20mm を初めて超えたティック
        THRESH_MM = 20.0
        first_over = next((e for e in errs if abs(e["lateral"]) * 1000 > THRESH_MM), None)
        if first_over is not None:
            r = first_over["rec"]
            print(f"  横ずれが最初に{THRESH_MM}mmを超えたのは tick={r['tick']} s={r['s']:.4f} "
                  f"(lat={first_over['lateral']*1000:.2f}mm lon={first_over['longitudinal']*1000:.2f}mm)")
        else:
            print(f"  横ずれは記録区間内で{THRESH_MM}mmを一度も超えなかった")
        print()

        # (4) 「誤差(横 or 縦)が10mmを初めて超えた」地点の前後を詳細表示(初動の飛びを特定する)
        ZOOM_THRESH_MM = 10.0
        zoom_idx = next((i for i, e in enumerate(errs)
                          if max(abs(e["lateral"]), abs(e["longitudinal"])) * 1000 > ZOOM_THRESH_MM), None)
        if zoom_idx is not None:
            lo = max(0, zoom_idx - 15)
            hi = min(len(errs), zoom_idx + 30)
            print(f"  --- 誤差が{ZOOM_THRESH_MM}mmを最初に超えた idx={zoom_idx} の前後を詳細表示 ---")
            print("  tick mode   s[m]    kappa_ref  e_lat[mm] wc_ref_s   lat_err[mm] lon_err[mm]  vl,vr")
            for i in range(lo, hi):
                e = errs[i]
                r = e["rec"]
                kap = r["kappa_ref"]
                wc = r["wc_ref_s"]
                print(f"  {r['tick']:4d} {r['mode']:6s} {r['s']:.4f}  "
                      f"{('%.3f' % kap) if kap is not None else '  -  ':>9s}  "
                      f"{r['e_lat']*1000:8.3f}  {('%.4f' % wc) if wc is not None else '  -   ':>8s}  "
                      f"{e['lateral']*1000:8.2f}   {e['longitudinal']*1000:8.2f}   {r['vl']:.3f},{r['vr']:.3f}"
                      + ("  <== 閾値超え" if i == zoom_idx else ""))
            print()

        # (5) 衝突直前 N ティックの詳細（衝突原因そのものを見る）
        N_TAIL = 45
        tail_lo = max(0, len(errs) - N_TAIL)
        print(f"  --- 衝突直前 {len(errs) - tail_lo} ティックの詳細 ---")
        print("  tick mode   s[m]    kappa_ref  e_lat[mm] wc_ref_s   lat_err[mm] lon_err[mm]  vl,vr")
        for i in range(tail_lo, len(errs)):
            e = errs[i]
            r = e["rec"]
            kap = r["kappa_ref"]
            wc = r["wc_ref_s"]
            print(f"  {r['tick']:4d} {r['mode']:6s} {r['s']:.4f}  "
                  f"{('%.3f' % kap) if kap is not None else '  -  ':>9s}  "
                  f"{r['e_lat']*1000:8.3f}  {('%.4f' % wc) if wc is not None else '  -   ':>8s}  "
                  f"{e['lateral']*1000:8.2f}   {e['longitudinal']*1000:8.2f}   {r['vl']:.3f},{r['vr']:.3f}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
