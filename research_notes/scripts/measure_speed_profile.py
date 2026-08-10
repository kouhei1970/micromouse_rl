"""
research_notes/scripts/measure_speed_profile.py
================
古典ベースライン3方式（L0-a/L0-b/L0-c）の速度プロファイルと内部状態の
内訳を計測する診断スクリプト（新規・既存ファイル無変更）。

--------------------------------------------------------------------------
教授セッションからの問い（これに数値で答えるためのスクリプト）
--------------------------------------------------------------------------
問1: 「速度が0.1 m/s未満になっていた時間の割合」は3方式（L0-a/L0-b/L0-c）
     でどう違うか。
問2: L0-c（SlalomPolicy）の探索走行（第1走行）は本当にスラロームで
     走っているのか、それとも超信地旋回（その場旋回）に落ちているのか。
     → L0-cの内部状態 self._state の "TURN_INPLACE" が占める時間割合と、
       他状態→TURN_INPLACEへの遷移回数を第1走行について集計して判定する。

--------------------------------------------------------------------------
計測方法
--------------------------------------------------------------------------
research_notes/scripts/_video_l0_common.py の RecordingPolicyWrapper /
record_run_video と同じやり方（= competition/evaluator.py の
CompetitionEvaluator.evaluate_maze() をそのまま駆動し、on_run_start/
on_run_end フックで走行境界を取得する）を踏襲するが、描画は一切行わない
（レンダラを作らない・画像を書き出さない）。各 act() 呼び出しのたびに
sim.sim_time・sim.privileged_velocity()・走行番号・走行中/帰還中の別・
方策の内部状態変数を記録し、後段で集計する。

方策ごとの内部状態変数（ソースを読んで特定）:
  - L0-a (AdachiPolicy, competition/baseline_classical.py):
      self._state ∈ {"IDLE","PLAN","TURN","FORWARD"}。
      "TURN" が超信地旋回（区画ごとに停止して曲がる、L0-aの走行方式では
      これが唯一の旋回形態）に対応する。
  - L0-b (StraightRunPolicy, competition/baseline_straightrun.py):
      self._state ∈ {"IDLE","PLAN","TURN","FORWARD"}（L0-aと同じ状態名）。
      "TURN" が超信地旋回に対応する（直進区間は複数区画を停止せず走り抜ける
      が、曲がる箇所では依然として一旦停止しての超信地旋回）。
  - L0-c (SlalomPolicy, competition/baseline_slalom.py):
      self._state ∈ {"INIT","DRIVE","TURN_INPLACE","IDLE"}。
      self._path_end_reason ∈ {None,"goal","start","pre_turn","continue"}
      （DRIVE状態で、今張っている軌道の終端理由。速度計画がこの理由に応じて
      終端速度を0に落とすかどうかを決める。§2.3 参照）。
      "TURN_INPLACE" が例外処理としてのその場旋回（(a)先頭セルでの急な
      方位転換 (b)180°折返し、baseline_slalom.py 冒頭docstring参照）。
"""
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# リポジトリルートをsys.pathへ追加する（research_notes/scripts配下の他スクリプト
# （video_l0_run.py等）と同じ慣例。python foo/bar.py で実行するとsys.path[0]が
# スクリプト自身のディレクトリになりcwdは自動では乗らないため、どこから
# 実行されても `competition` パッケージ（リポジトリルート直下）をimportできる
# ようにする）。
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_REPO)
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from competition.baseline_classical import AdachiPolicy  # noqa: E402
from competition.baseline_slalom import SlalomPolicy
from competition.baseline_straightrun import StraightRunPolicy
from competition.evaluator import CompetitionEvaluator
from competition.policy_interface import MousePolicy

EVAL_MAZE_DIR = Path("competition/mazes/eval")
MAZE_ID = "maze_1015"
TIME_BUDGET_S = 420.0
MAX_RUNS = 5

V_THRESH = 0.1       # |v| < 0.1 m/s を「低速」と判定するしきい値 [m/s]
OMEGA_THRESH = 0.5    # |omega| >= 0.5 rad/s を「旋回中」と判定するしきい値 [rad/s]

OUT_JSON = Path("outputs/analysis/speed_profile_maze1015.json")

# 各方式の「超信地旋回に相当する内部状態名」（比較表・計数で共通に使う）
_INPLACE_TURN_STATE = {
    "l0a_adachi": "TURN",
    "l0b_straightrun": "TURN",
    "l0c_slalom": "TURN_INPLACE",
}


# ==========================================================================
# 方策ラッパー: 描画なし版 RecordingPolicyWrapper。
# CompetitionEvaluator.evaluate_maze() をそのまま駆動し、act() が呼ばれる
# たびに (t, v, omega, run, phase, policy_state, path_end_reason) を記録する。
# ==========================================================================
class MeasuringPolicyWrapper(MousePolicy):
    requires_privileged = True

    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", "unnamed")
        self.sim = None
        self.run_count = 0
        self.state = "FREE"   # "RUN_ACTIVE"（走行中） / "FREE"（帰還中・待機中）
        self.records = []     # list[dict]

    def bind_sim(self, sim):
        self.sim = sim
        self.inner.bind_sim(sim)

    def bind_maze(self, v_walls, h_walls):
        self.inner.bind_maze(v_walls, h_walls)

    def on_maze_start(self, maze_info):
        self.run_count = 0
        self.state = "FREE"
        self.inner.on_maze_start(maze_info)

    def on_run_start(self, run_index):
        self.run_count = run_index
        self.state = "RUN_ACTIVE"
        self.inner.on_run_start(run_index)

    def on_run_end(self, outcome):
        self.state = "FREE"
        self.inner.on_run_end(outcome)

    def on_retrieval(self):
        self.inner.on_retrieval()

    def act(self, obs):
        vl, vr = self.inner.act(obs)
        t = self.sim.sim_time
        v_fwd, omega_z = self.sim.privileged_velocity()
        self.records.append({
            "t": float(t),
            "v": float(v_fwd),
            "omega": float(omega_z),
            "run": int(self.run_count),
            "phase": self.state,
            "policy_state": getattr(self.inner, "_state", None),
            "path_end_reason": getattr(self.inner, "_path_end_reason", None),
        })
        return vl, vr


# ==========================================================================
# 集計ヘルパー
# ==========================================================================
def frac_stats(v_arr: np.ndarray, omega_arr: np.ndarray, dt: float) -> dict:
    """|v|<V_THRESH の時間割合、さらにomegaで「完全停止」と「その場旋回」に
    内訳分解した時間割合を返す。走行中のみのステップ列を渡すこと。"""
    n = len(v_arr)
    if n == 0:
        return dict(n_steps=0, total_time_s=0.0, frac_slow_pct=None,
                    frac_full_stop_pct=None, frac_inplace_turn_pct=None)
    v_abs = np.abs(v_arr)
    om_abs = np.abs(omega_arr)
    slow = v_abs < V_THRESH
    turn = slow & (om_abs >= OMEGA_THRESH)
    stop = slow & (om_abs < OMEGA_THRESH)
    return dict(
        n_steps=int(n),
        total_time_s=float(n * dt),
        frac_slow_pct=float(100.0 * slow.sum() / n),
        frac_full_stop_pct=float(100.0 * stop.sum() / n),
        frac_inplace_turn_pct=float(100.0 * turn.sum() / n),
    )


def state_fraction_table(states) -> dict:
    """状態文字列の列（Noneを含みうる）から、各状態が占める時間割合[%]を返す。"""
    n = len(states)
    if n == 0:
        return {}
    cnt = Counter(states)
    return {str(k): 100.0 * v / n for k, v in cnt.items()}


def count_transitions_into(states, target) -> int:
    """states（時系列順）の中で、target以外からtargetへ遷移した回数を数える。"""
    count = 0
    prev = None
    for s in states:
        if s == target and prev != target:
            count += 1
        prev = s
    return count


def lowspeed_breakdown_l0c(v_arr, policy_state_arr, path_end_reason_arr) -> dict:
    """L0-c専用: |v|<V_THRESH のステップを (policy_state, path_end_reason) で
    内訳分解する。TURN_INPLACE中はpath_end_reasonが直前DRIVE脚の値のまま
    残留する（_enter_turn_inplaceではクリアされない）ため、TURN_INPLACE中は
    path_end_reasonを無視してひとまとめにする。"""
    slow_mask = np.abs(v_arr) < V_THRESH
    n_slow = int(slow_mask.sum())
    if n_slow == 0:
        return {"n_slow_steps": 0, "breakdown_pct": {}}
    ps = policy_state_arr[slow_mask]
    per = path_end_reason_arr[slow_mask]
    cnt = Counter()
    for s, p in zip(ps, per):
        if s == "TURN_INPLACE":
            key = "TURN_INPLACE"
        elif s == "DRIVE":
            key = f"DRIVE(path_end_reason={p})"
        else:
            key = f"state={s}"
        cnt[key] += 1
    return {"n_slow_steps": n_slow,
            "breakdown_pct": {k: 100.0 * v / n_slow for k, v in cnt.items()}}


def fmt_pct(x):
    return "  N/A " if x is None else f"{x:6.2f}%"


def fmt_time(x):
    return "  N/A " if x is None else f"{x:7.2f}s"


# ==========================================================================
# 1方式ぶんの評価実行 + 集計
# ==========================================================================
def process_method(short_name: str, label: str, policy_cls) -> dict:
    print(f"\n{'=' * 78}")
    print(f"{label} ({short_name}) を評価中 …")
    print("=" * 78)

    npz_path = EVAL_MAZE_DIR / f"{MAZE_ID}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} が見つかりません")

    policy = policy_cls()
    wrapper = MeasuringPolicyWrapper(policy)
    evaluator = CompetitionEvaluator(maze_dir=str(EVAL_MAZE_DIR),
                                      time_budget=TIME_BUDGET_S, max_runs=MAX_RUNS)

    t_wall0 = time.time()
    result = evaluator.evaluate_maze(npz_path, wrapper)
    wall_clock_s = time.time() - t_wall0

    records = wrapper.records
    n_total = len(records)
    print(f"  評価完了: 実時間 {wall_clock_s:6.1f}s, 総制御ステップ数 {n_total}")

    dt = float(getattr(policy, "control_dt", 0.01))

    t_arr = np.array([r["t"] for r in records], dtype=float)
    v_arr = np.array([r["v"] for r in records], dtype=float)
    om_arr = np.array([r["omega"] for r in records], dtype=float)
    run_arr = np.array([r["run"] for r in records], dtype=int)
    phase_arr = np.array([r["phase"] for r in records], dtype=object)
    pstate_arr = np.array([r["policy_state"] for r in records], dtype=object)
    per_arr = np.array([r["path_end_reason"] for r in records], dtype=object)

    active_mask_all = (phase_arr == "RUN_ACTIVE")

    runs_info = result["runs"]
    for r in runs_info:
        print(f"    第{r['index']}走行: outcome={r['outcome']:10s} "
              f"run_time={r['run_time'] if r['run_time'] is not None else float('nan'):7.2f}s")

    # --- 全走行の合計（走行中のみ） ---
    overall_all = frac_stats(v_arr[active_mask_all], om_arr[active_mask_all], dt)

    # --- 最速走行のみ ---
    goal_runs = [r for r in runs_info if r["outcome"] == "goal" and r.get("run_time") is not None]
    best_run_info = min(goal_runs, key=lambda r: r["run_time"]) if goal_runs else None
    if best_run_info is not None:
        best_idx = best_run_info["index"]
        best_mask = active_mask_all & (run_arr == best_idx)
        overall_best = frac_stats(v_arr[best_mask], om_arr[best_mask], dt)
        overall_best["run_index"] = best_idx
        overall_best["run_time_s"] = best_run_info["run_time"]
    else:
        overall_best = dict(n_steps=0, total_time_s=0.0, frac_slow_pct=None,
                             frac_full_stop_pct=None, frac_inplace_turn_pct=None,
                             run_index=None, run_time_s=None)

    # --- 走行ごと ---
    per_run = []
    inplace_state_name = _INPLACE_TURN_STATE[short_name]
    for r in runs_info:
        idx = r["index"]
        mask = active_mask_all & (run_arr == idx)
        stats = frac_stats(v_arr[mask], om_arr[mask], dt)
        states_this_run = list(pstate_arr[mask])
        state_frac = state_fraction_table(states_this_run)
        n_inplace_entries = count_transitions_into(states_this_run, inplace_state_name)
        per_run.append({
            "run_index": idx,
            "outcome": r["outcome"],
            "run_time_s": r["run_time"],
            **stats,
            "policy_state_fraction_pct": state_frac,
            "n_inplace_turn_entries": n_inplace_entries,
        })

    # --- 表出力: 全走行合計 / 最速走行 ---
    print(f"\n  [全走行合計・走行中のみ]  ステップ数={overall_all['n_steps']:6d}  "
          f"時間={fmt_time(overall_all['total_time_s'])}")
    print(f"    |v|<0.1               : {fmt_pct(overall_all['frac_slow_pct'])}")
    print(f"      内訳 完全停止(|ω|<0.5): {fmt_pct(overall_all['frac_full_stop_pct'])}")
    print(f"      内訳 その場旋回(|ω|>=0.5): {fmt_pct(overall_all['frac_inplace_turn_pct'])}")

    if best_run_info is not None:
        print(f"\n  [最速走行のみ・第{overall_best['run_index']}走行 "
              f"({overall_best['run_time_s']:.2f}s)]  ステップ数={overall_best['n_steps']:6d}")
        print(f"    |v|<0.1               : {fmt_pct(overall_best['frac_slow_pct'])}")
        print(f"      内訳 完全停止(|ω|<0.5): {fmt_pct(overall_best['frac_full_stop_pct'])}")
        print(f"      内訳 その場旋回(|ω|>=0.5): {fmt_pct(overall_best['frac_inplace_turn_pct'])}")
    else:
        print("\n  [最速走行のみ] ゴール到達走行が無いため算出不能")

    # --- 表出力: 走行ごと ---
    print("\n  [走行ごと]")
    print("   run  outcome     run_time  |v|<0.1  完全停止  その場旋回  内部状態内訳")
    for pr in per_run:
        state_str = ", ".join(f"{k}={v:.1f}%" for k, v in sorted(pr["policy_state_fraction_pct"].items()))
        rt = pr["run_time_s"]
        rt_str = f"{rt:7.2f}s" if rt is not None else "    N/A"
        print(f"    {pr['run_index']:2d}  {pr['outcome']:10s} {rt_str}  "
              f"{fmt_pct(pr['frac_slow_pct'])} {fmt_pct(pr['frac_full_stop_pct'])} "
              f"{fmt_pct(pr['frac_inplace_turn_pct'])}  [{state_str}]"
              f"  {inplace_state_name}遷移回数={pr['n_inplace_turn_entries']}")

    # --- L0-c専用: 低速ステップの内訳（(i)その場旋回 vs (ii)経路終端の減速停止） ---
    lowspeed_detail_per_run = None
    if short_name == "l0c_slalom":
        print("\n  [L0-c専用] 低速(|v|<0.1)ステップの内訳（走行ごと）:")
        lowspeed_detail_per_run = []
        for r in runs_info:
            idx = r["index"]
            mask = active_mask_all & (run_arr == idx)
            detail = lowspeed_breakdown_l0c(v_arr[mask], pstate_arr[mask], per_arr[mask])
            detail["run_index"] = idx
            lowspeed_detail_per_run.append(detail)
            bstr = ", ".join(f"{k}={v:.1f}%" for k, v in sorted(detail["breakdown_pct"].items()))
            print(f"    第{idx}走行: 低速ステップ数={detail['n_slow_steps']:6d}  [{bstr}]")

    # --- 生時系列（10Hzへ間引き） ---
    target_hz = 10.0
    factor = max(1, int(round((1.0 / dt) / target_hz)))
    timeseries_10hz = [
        {"t": records[i]["t"], "v": records[i]["v"], "omega": records[i]["omega"],
         "run": records[i]["run"], "phase": records[i]["phase"],
         "policy_state": records[i]["policy_state"], "path_end_reason": records[i]["path_end_reason"]}
        for i in range(0, n_total, factor)
    ]

    return {
        "label": label,
        "short_name": short_name,
        "control_dt_s": dt,
        "wall_clock_s": wall_clock_s,
        "n_total_steps": n_total,
        "official_runs": runs_info,
        "best_time_s": result["best_time"],
        "overall_all_runs_active_only": overall_all,
        "overall_best_run_active_only": overall_best,
        "per_run": per_run,
        "inplace_turn_state_name": inplace_state_name,
        "lowspeed_breakdown_per_run_l0c": lowspeed_detail_per_run,
        "timeseries_10hz": timeseries_10hz,
    }


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"JSON化できない型です: {type(o)}")


def main():
    methods = [
        ("l0a_adachi", "L0-a AdachiPolicy（超信地旋回・区画ごと停止）", AdachiPolicy),
        ("l0b_straightrun", "L0-b StraightRunPolicy（超信地旋回・直進連続）", StraightRunPolicy),
        ("l0c_slalom", "L0-c SlalomPolicy（スラローム走行）", SlalomPolicy),
    ]

    results = {}
    for short_name, label, cls in methods:
        results[short_name] = process_method(short_name, label, cls)

    # --- 問1・問2への回答用まとめ表 ---
    print(f"\n{'=' * 78}")
    print("まとめ: 問1（|v|<0.1 の時間割合、3方式比較）")
    print("=" * 78)
    print("  方式               全走行合計|v|<0.1   最速走行のみ|v|<0.1")
    for short_name, label, _ in methods:
        res = results[short_name]
        a = res["overall_all_runs_active_only"]["frac_slow_pct"]
        b = res["overall_best_run_active_only"]["frac_slow_pct"]
        print(f"  {short_name:18s} {fmt_pct(a):>10s}          {fmt_pct(b):>10s}")

    print(f"\n{'=' * 78}")
    print("まとめ: 問2（L0-c 第1走行=探索走行 の TURN_INPLACE 内訳）")
    print("=" * 78)
    l0c_per_run = results["l0c_slalom"]["per_run"]
    run1 = next((pr for pr in l0c_per_run if pr["run_index"] == 1), None)
    if run1 is not None:
        turn_inplace_frac_time = run1["policy_state_fraction_pct"].get("TURN_INPLACE", 0.0)
        n_entries = run1["n_inplace_turn_entries"]
        print(f"  第1走行（探索走行）の内部状態時間割合: "
              f"{ {k: round(v,2) for k,v in run1['policy_state_fraction_pct'].items()} }")
        print(f"  TURN_INPLACE が占める時間割合          : {turn_inplace_frac_time:6.2f}%")
        print(f"  他状態→TURN_INPLACE への遷移回数        : {n_entries} 回")
    else:
        print("  第1走行のデータが記録されていません（走行が発生しなかった可能性）")

    # --- JSON保存 ---
    os.makedirs(OUT_JSON.parent, exist_ok=True)
    out_data = {
        "generated_at": datetime.now().isoformat(),
        "maze_id": MAZE_ID,
        "time_budget_s": TIME_BUDGET_S,
        "max_runs": MAX_RUNS,
        "v_thresh_mps": V_THRESH,
        "omega_thresh_radps": OMEGA_THRESH,
        "methods": results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"\nJSON保存先: {OUT_JSON.resolve()}")


if __name__ == "__main__":
    main()
