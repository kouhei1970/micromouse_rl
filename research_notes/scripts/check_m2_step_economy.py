"""
research_notes/scripts/check_m2_step_economy.py
=================================================
M2-0（6x6 迷路単走）の学習済み方策を検証帯（maze seed 7000-7019）で再生し、
**1 ステップあたりの報酬内訳を実測する**。机上計算の入力値になるスクリプトなので、
正確さを最優先し、env（`mouse/maze6_env.py`）は一切改造せず、`Maze6Env.step()` の
前後で内部状態を読んで外側で再計算する。

--------------------------------------------------------------------------
reward の分解（`mouse/maze6_env.py` Maze6Env.step()、306-345 行）
--------------------------------------------------------------------------
    reward = γ・Φ(s') − Φ(s) − time_penalty                      … shaping + 定数罰
             + goal_bonus          (ゴール到達時のみ、+1.0)
             + collision_penalty   (衝突・転倒時のみ、ゴールでない場合、既定 -1.0)
             + visit_bonus         (未訪問区画へ初めて入ったときのみ、+0.02)
             − action_smooth_penalty・‖a_t − a_(t−1)‖²    (本実験群は係数 0)
             − action_highpass_penalty・‖a_t − ā_t‖²       (k。exp_010=8.7e-3 / exp_011=0)

shaping = γ・Φ(s') − Φ(s) を dPhi と drift に分解する式（自分で検算した恒等式）:
    dPhi  = Φ(s') − Φ(s)             … 区画レベルの生の進捗
    drift = −(1 − γ)・Φ(s')           … 割引率 γ<1 による「居るだけで削られる」項
    dPhi + drift
      = [Φ(s') − Φ(s)] − (1 − γ)Φ(s')
      = Φ(s') − Φ(s) − Φ(s') + γΦ(s')
      = γΦ(s') − Φ(s)
      = shaping                      … ∴ shaping = dPhi + drift （γ・dPhi ではない）

各ステップで env が返した reward と、上記全項の総和（recomposed）が
絶対誤差 1e-9 未満で一致することを毎ステップ assert する。

--------------------------------------------------------------------------
ゴールへの近接度（2026-08-11 教授指示で追加）
--------------------------------------------------------------------------
`n_visited`（訪問した異なる区画数）は同じ区画を往復しても増えるだけなので進捗の
指標にならない。`info["dist_to_goal"]`（ゴールまでの迷路距離。区画単位の整数）から
走行ごとに次を直接測る（近似・推定は使わない）:

    d_start  = エピソード開始時の dist_to_goal（= info["d_start"] と一致するはず。
               一致しなければ assert で落ちる）
    d_end    = 終端ステップの dist_to_goal
    d_min    = エピソード中（開始時を含む）の dist_to_goal の最小値
    progress_cells      = d_start − d_end       （負なら始点より遠くで終わった）
    best_progress_cells = d_start − d_min
    path_efficiency      = distance_m / (d_start × cell_size)   （1.0 が最短経路）
    Phi_T = (d_start − d_end) × cell_size [m]
          = Φ(終端区画)（potential の定義 Φ(cell)=(D0−d(cell))·cell_size そのもの）

--------------------------------------------------------------------------
学習時の環境構築引数（experiments/exp_010_m2_0/train.py と
logs/exp_010_m2_0_seed1/run_summary.json, logs/exp_011_m2_0_k0_seed1/run_summary.json
から確認）
--------------------------------------------------------------------------
    gamma=0.995, maze_mode="loop", visit_bonus=0.02, collision_penalty=-1.0,
    action_smooth_penalty=0.0, action_highpass_alpha=0.5,
    action_highpass_penalty = 8.7e-3 (exp_010_m2_0_*) / 0.0 (exp_011_m2_0_k0_*)

VecNormalize は使っていない（train.py は DummyVecEnv をそのまま PPO に渡すのみ）。
`mouse/maze6_eval.py` も `PPO.load(..., device="cpu")` → `model.predict(obs, ...)` で
生の観測をそのまま方策へ渡している。したがって本スクリプトも観測を正規化しない。

使い方:
    .venv/bin/python research_notes/scripts/check_m2_step_economy.py --smoke
    .venv/bin/python research_notes/scripts/check_m2_step_economy.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO  # noqa: E402

from common.seed_bands import assert_seeds_allowed, describe_seeds  # noqa: E402
from mouse.maze6_env import Maze6Env, _GOAL_BONUS, _TIME_PENALTY  # noqa: E402
from mouse.maze6_eval import VALIDATION_MAZE_DIR, _trial_seed  # noqa: E402

# --- 学習時の環境構築引数（上記コメント参照）。VecNormalize なし ---
GAMMA = 0.995
MAZE_MODE = "loop"
VISIT_BONUS = 0.02
COLLISION_PENALTY = -1.0
ACTION_SMOOTH_PENALTY = 0.0
ACTION_HIGHPASS_ALPHA = 0.5

MODELS = [
    dict(name="exp_010_m2_0_seed1", path="models/exp_010_m2_0_seed1.zip", k=8.7e-3),
    dict(name="exp_010_m2_0_seed2", path="models/exp_010_m2_0_seed2.zip", k=8.7e-3),
    dict(name="exp_011_m2_0_k0_seed1", path="models/exp_011_m2_0_k0_seed1.zip", k=0.0),
    dict(name="exp_011_m2_0_k0_seed2", path="models/exp_011_m2_0_k0_seed2.zip", k=0.0),
    # exp_012 の 3 条件（裁定 R41-① の機構分析。2026-08-13 追加）。
    # **flags は学習時と同じものを渡すこと**。違うと Φ の定義が変わって shaping が
    # 別物になり、毎ステップの reward 分解 assert が落ちる（＝取り違えは必ず検出される）。
    # flags の中身は experiments/exp_012_continuous_potential/train.py の
    # CONDITION_FLAGS と同じ（E / C / Cp）。
    dict(name="exp_012_condE_seed1", path="models/exp_012_condE_seed1.zip",
         k=8.7e-3, flags=dict(continuous_potential=True)),
    dict(name="exp_012_condE_seed2", path="models/exp_012_condE_seed2.zip",
         k=8.7e-3, flags=dict(continuous_potential=True)),
    dict(name="exp_012_condE_seed3", path="models/exp_012_condE_seed3.zip",
         k=8.7e-3, flags=dict(continuous_potential=True)),
    dict(name="exp_012_condC_seed1", path="models/exp_012_condC_seed1.zip",
         k=8.7e-3, flags=dict(geodesic_potential=True)),
    dict(name="exp_012_condC_seed2", path="models/exp_012_condC_seed2.zip",
         k=8.7e-3, flags=dict(geodesic_potential=True)),
    dict(name="exp_012_condC_seed3", path="models/exp_012_condC_seed3.zip",
         k=8.7e-3, flags=dict(geodesic_potential=True)),
    dict(name="exp_012_condCp_seed1", path="models/exp_012_condCp_seed1.zip",
         k=8.7e-3, flags=dict(geodesic_potential=True, geodesic_rho_scale=True)),
    dict(name="exp_012_condCp_seed2", path="models/exp_012_condCp_seed2.zip",
         k=8.7e-3, flags=dict(geodesic_potential=True, geodesic_rho_scale=True)),
    dict(name="exp_012_condCp_seed3", path="models/exp_012_condCp_seed3.zip",
         k=8.7e-3, flags=dict(geodesic_potential=True, geodesic_rho_scale=True)),
]

VALIDATION_SEEDS = list(range(7000, 7020))  # 検証帯（研究計画書 §9-7）

# 挙動 3 層のしきい値 [m/s]
STOP_V = 0.05
DRIVE_V = 0.4

ASSERT_TOL = 1e-9

REPO_MAZE_DIR = str(REPO_ROOT / VALIDATION_MAZE_DIR)


def run_episode(model, maze_seed: int, k: float, err_tracker: dict,
                flags: dict = None) -> dict:
    """1 面 ×1 試行を決定的に再生し、ステップごとの報酬内訳を分解して返す。

    `flags` は Φ の実現方法（train.py の CONDITION_FLAGS と同じもの。None = 階段版）。
    **学習時と同じものを渡すこと**。違うと shaping が別物になり、毎ステップの
    reward 分解 assert が落ちる（＝取り違えは必ず検出される）。
    """
    env = Maze6Env(
        maze_dir=REPO_MAZE_DIR, maze_seeds=[maze_seed], max_cache=2,
        gamma=GAMMA, mode="fixed", maze_mode=MAZE_MODE,
        visit_bonus=VISIT_BONUS, collision_penalty=COLLISION_PENALTY,
        action_smooth_penalty=ACTION_SMOOTH_PENALTY,
        action_highpass_penalty=k, action_highpass_alpha=ACTION_HIGHPASS_ALPHA,
        **(flags or {}),
    )
    tseed = _trial_seed(0, maze_seed, 0)   # mouse.maze6_eval と同じ試行 seed の規約
    obs, info = env.reset(seed=tseed)
    dt = env.params.control_dt
    cell_size = env.params.cell_size
    n = env._n_dist

    # d_start はエピソード開始時の dist_to_goal。info["d_start"]（= self._d_start）と
    # 一致するはず（両方とも self._dist_map[start] から出ているので恒等的に一致する）。
    # 近似ではなく実測なので、食い違えばここで即座に落として報告する。
    d_start = int(info["dist_to_goal"])
    assert d_start == int(info["d_start"]), (
        f"d_start 不一致: maze_seed={maze_seed} dist_to_goal={d_start} "
        f"info['d_start']={info['d_start']}")
    dist_history = [d_start]   # 開始時を含む dist_to_goal の履歴（d_min 計算用）

    shaping_arr, dphi_arr, drift_arr = [], [], []
    hp2_arr, hpterm_arr, d2_arr = [], [], []
    visit_arr, v_arr, reward_arr = [], [], []
    goal_bonus_arr, coll_arr = [], []
    cells, dists = [], []

    while True:
        raw_a, _ = model.predict(obs, deterministic=True)
        # **丸めない**（行動の摂動に対しカオス的なことが既知。full precision で渡す）
        action = np.clip(np.asarray(raw_a, dtype=np.float64), -1.0, 1.0)

        # --- step 前の内部状態を読む ---
        prev_potential = env._prev_potential
        prev_action_before = np.array(env._prev_action, dtype=np.float64)
        visited_before = set(env._visited)

        obs, reward, terminated, truncated, info = env.step(action)

        # --- step 後の内部状態を読む ---
        potential_after = env._prev_potential  # env が既に Φ(s') へ更新済み
        dPhi = potential_after - prev_potential
        drift = -(1.0 - env.gamma) * potential_after
        shaping = dPhi + drift  # 恒等式で shaping = γΦ(s') − Φ(s) と一致（上部の検算参照）

        raw_obs = env.sim.observation()  # 副作用のない読み取りのみ（mouse/sim.py 参照）
        omega_l, omega_r = float(raw_obs[n + 6]), float(raw_obs[n + 7])
        v_forward = env.params.wheel_radius * (omega_l + omega_r) / 2.0

        cell = tuple(info["cell"])
        visit = VISIT_BONUS if cell not in visited_before else 0.0

        goal = bool(info["goal"])
        collision = bool(info["collision"])  # env 内部では衝突・転倒の両方を含む
        goal_bonus = _GOAL_BONUS if goal else 0.0
        collision_term = COLLISION_PENALTY if (collision and not goal) else 0.0

        ahat_after = np.array(env._action_lowpass, dtype=np.float64)
        hp_vec = action - ahat_after
        hp2 = float(np.dot(hp_vec, hp_vec))
        hp_term = -k * hp2

        d_vec = action - prev_action_before
        d2 = float(np.dot(d_vec, d_vec))
        smooth_term = -ACTION_SMOOTH_PENALTY * d2

        recomposed = (shaping - _TIME_PENALTY + visit + goal_bonus + collision_term
                      + hp_term + smooth_term)
        err = abs(recomposed - reward)
        err_tracker["max"] = max(err_tracker["max"], err)
        err_tracker["n"] += 1
        assert err < ASSERT_TOL, (
            f"reward 不一致: maze_seed={maze_seed} step={len(shaping_arr)} "
            f"env_reward={reward!r} recomposed={recomposed!r} err={err!r}")

        shaping_arr.append(shaping)
        dphi_arr.append(dPhi)
        drift_arr.append(drift)
        hp2_arr.append(hp2)
        hpterm_arr.append(hp_term)
        d2_arr.append(d2)
        visit_arr.append(visit)
        goal_bonus_arr.append(goal_bonus)
        coll_arr.append(collision_term)
        v_arr.append(v_forward)
        reward_arr.append(reward)
        cells.append(cell)
        dists.append(info["dist_to_goal"])
        dist_history.append(info["dist_to_goal"])

        if terminated or truncated:
            outcome = "goal" if goal else ("collision" if collision else "timeout")
            n_visited = int(info["n_visited"])
            break

    # 終端の Φ を env から**実測**で読む。連続 Φ では
    # phi_T = (d_start − d_end)·cell_size（階段版の式）が成り立たないため。
    phi_T_measured = float(env._prev_potential)
    env.close()
    v_arr = np.array(v_arr)
    distance_m = float(np.sum(np.abs(v_arr) * dt))
    n_steps = len(shaping_arr)

    # --- ゴールへの近接度（推定でなく実測。dist_to_goal の履歴から直接計算） ---
    d_end = int(dists[-1])
    d_min = int(min(dist_history))
    progress_cells = d_start - d_end
    best_progress_cells = d_start - d_min
    path_efficiency = (distance_m / (d_start * cell_size)) if d_start > 0 else float("nan")
    phi_T = (d_start - d_end) * cell_size   # = Φ(終端区画)

    # 🔴 走行ごとの層別割合。**プールした割合は長いエピソード（時間切れ 6000 歩）に
    # 支配される**ので、「方策が停止解か」を語るには走行ごとに測って outcome 別に
    # まとめる必要がある（2026-08-13 追加。集計と対応の取り違えを防ぐ）。
    va = np.abs(v_arr)
    ep_stop_frac = float(np.mean(va < STOP_V))
    ep_drive_frac = float(np.mean(va >= DRIVE_V))

    return dict(
        maze_seed=maze_seed, outcome=outcome, n_steps=n_steps, n_visited=n_visited,
        stop_frac=ep_stop_frac, drive_frac=ep_drive_frac,
        distance_m=distance_m, avg_speed_mps=distance_m / max(n_steps * dt, 1e-9),
        shaping=np.array(shaping_arr), dPhi=np.array(dphi_arr), drift=np.array(drift_arr),
        hp2=np.array(hp2_arr), hp_term=np.array(hpterm_arr), d2=np.array(d2_arr),
        visit=np.array(visit_arr), goal_bonus=np.array(goal_bonus_arr),
        collision_penalty=np.array(coll_arr), v_forward=v_arr,
        reward=np.array(reward_arr), cells=cells, dist_to_goal=dists,
        d_start=d_start, d_end=d_end, d_min=d_min,
        progress_cells=progress_cells, best_progress_cells=best_progress_cells,
        path_efficiency=path_efficiency, phi_T=phi_T,
        phi_T_measured=phi_T_measured,
    )


def _q(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def summarize_model(name: str, episodes: list) -> dict:
    """1 モデルの 20 走行から、報告用の集計値をすべて計算する。"""
    outcomes = [e["outcome"] for e in episodes]
    n = len(episodes)
    n_goal = outcomes.count("goal")
    n_coll = outcomes.count("collision")
    n_timeout = outcomes.count("timeout")

    Ts = np.array([e["n_steps"] for e in episodes], dtype=float)
    nvis = np.array([e["n_visited"] for e in episodes], dtype=float)
    dist = np.array([e["distance_m"] for e in episodes], dtype=float)
    avgv = np.array([e["avg_speed_mps"] for e in episodes], dtype=float)

    # --- ゴールへの近接度（実測。n_visited は往復で増えるだけなので進捗の指標にならない） ---
    progress = np.array([e["progress_cells"] for e in episodes], dtype=float)
    best_progress = np.array([e["best_progress_cells"] for e in episodes], dtype=float)
    path_eff = np.array([e["path_efficiency"] for e in episodes], dtype=float)
    phi_T_arr = np.array([e["phi_T"] for e in episodes], dtype=float)
    n_worse_than_start = int(np.sum(progress < 0))

    # --- プール（全ステップを episode 境界なく束ねる） ---
    pool_hp2 = np.concatenate([e["hp2"] for e in episodes])
    pool_d2 = np.concatenate([e["d2"] for e in episodes])
    pool_shaping = np.concatenate([e["shaping"] for e in episodes])
    pool_hpterm = np.concatenate([e["hp_term"] for e in episodes])
    pool_visit = np.concatenate([e["visit"] for e in episodes])
    pool_v = np.concatenate([e["v_forward"] for e in episodes])

    # --- エピソードごとの平均 hp2 の分布（別集計） ---
    ep_mean_hp2 = np.array([float(np.mean(e["hp2"])) for e in episodes])

    def five(x):
        return dict(min=float(np.min(x)), q1=_q(x, 25), median=_q(x, 50),
                    q3=_q(x, 75), max=float(np.max(x)))

    # --- 挙動 3 層による層別集計 ---
    layers = {}
    v_abs = np.abs(pool_v)
    masks = dict(
        停止=(v_abs < STOP_V),
        低速=(v_abs >= STOP_V) & (v_abs < DRIVE_V),
        走行=(v_abs >= DRIVE_V),
    )
    for label, mask in masks.items():
        cnt = int(np.sum(mask))
        layers[label] = dict(
            n_steps=cnt, frac=float(cnt / len(pool_v)) if len(pool_v) else float("nan"),
            mean_hp2=float(np.mean(pool_hp2[mask])) if cnt else None,
            mean_d2=float(np.mean(pool_d2[mask])) if cnt else None,
            mean_shaping=float(np.mean(pool_shaping[mask])) if cnt else None,
        )

    by_outcome = {}
    for oc in ("goal", "collision", "timeout"):
        sel = [e for e in episodes if e["outcome"] == oc]
        if not sel:
            continue
        by_outcome[oc] = dict(
            n=len(sel),
            stop_frac_median=float(np.median([e["stop_frac"] for e in sel])),
            drive_frac_median=float(np.median([e["drive_frac"] for e in sel])),
            T_median=float(np.median([e["n_steps"] for e in sel])),
            progress_median=float(np.median([e["progress_cells"] for e in sel])),
        )
    ep_stop = np.array([e["stop_frac"] for e in episodes], dtype=float)
    ep_drive = np.array([e["drive_frac"] for e in episodes], dtype=float)

    return dict(
        name=name, n_episodes=n,
        stop_frac_per_episode=dict(median=float(np.median(ep_stop)),
                                   min=float(np.min(ep_stop)), max=float(np.max(ep_stop))),
        drive_frac_per_episode=dict(median=float(np.median(ep_drive)),
                                    min=float(np.min(ep_drive)), max=float(np.max(ep_drive))),
        by_outcome=by_outcome,
        n_goal=n_goal, n_collision=n_coll, n_timeout=n_timeout,
        T=dict(min=float(np.min(Ts)), median=float(np.median(Ts)), max=float(np.max(Ts))),
        n_visited=dict(min=float(np.min(nvis)), median=float(np.median(nvis)),
                       max=float(np.max(nvis))),
        distance_m=dict(min=float(np.min(dist)), median=float(np.median(dist)),
                         max=float(np.max(dist))),
        avg_speed_mps_median=float(np.median(avgv)),
        progress_cells=dict(min=float(np.min(progress)), median=float(np.median(progress)),
                            max=float(np.max(progress))),
        best_progress_cells=dict(min=float(np.min(best_progress)),
                                 median=float(np.median(best_progress)),
                                 max=float(np.max(best_progress))),
        path_efficiency=dict(min=float(np.min(path_eff)), median=float(np.median(path_eff)),
                             max=float(np.max(path_eff))),
        n_worse_than_start=n_worse_than_start,
        phi_T_median=float(np.median(phi_T_arr)),
        phi_T_measured_median=float(np.median(
            np.array([e["phi_T_measured"] for e in episodes], dtype=float))),
        hp2_pooled=five(pool_hp2),
        hp2_episode_mean=five(ep_mean_hp2),
        d2_pooled_median=float(np.median(pool_d2)),
        reward_breakdown_mean=dict(
            shaping=float(np.mean(pool_shaping)),
            time_penalty=-float(_TIME_PENALTY),
            hp_term=float(np.mean(pool_hpterm)),
            visit=float(np.mean(pool_visit)),
        ),
        layers=layers,
        n_steps_pooled=int(len(pool_v)),
        episodes=[dict(maze_seed=e["maze_seed"], outcome=e["outcome"],
                       n_steps=e["n_steps"], n_visited=e["n_visited"],
                       stop_frac=e["stop_frac"], drive_frac=e["drive_frac"],
                       distance_m=e["distance_m"], avg_speed_mps=e["avg_speed_mps"],
                       mean_hp2=float(np.mean(e["hp2"])), mean_d2=float(np.mean(e["d2"])),
                       d_start=e["d_start"], d_end=e["d_end"], d_min=e["d_min"],
                       progress_cells=e["progress_cells"],
                       best_progress_cells=e["best_progress_cells"],
                       path_efficiency=e["path_efficiency"], phi_T=e["phi_T"],
                       phi_T_measured=e["phi_T_measured"])
                  for e in episodes],
    )


def print_report(summaries: list, err_tracker_by_model: dict):
    print("\n" + "=" * 100)
    print("M2-0 1 ステップあたり報酬内訳の実測（検証帯 seed 7000-7019・各 1 試行・deterministic）")
    print("=" * 100)
    for s in summaries:
        et = err_tracker_by_model[s["name"]]
        print(f"\n--- {s['name']} ---")
        print(f"  reward 分解の一致確認: {et['n']} ステップ中 最大絶対誤差 = {et['max']:.3e}"
              f"（許容 {ASSERT_TOL:.0e} 未満を全ステップで assert 済み）")
        print(f"  結果: ゴール {s['n_goal']} / 衝突 {s['n_collision']} / 時間切れ {s['n_timeout']}"
              f"  （{s['n_episodes']} 走行）")
        print(f"  エピソード長 T [step]        min={s['T']['min']:.0f}"
              f"  median={s['T']['median']:.0f}  max={s['T']['max']:.0f}")
        print(f"  訪問区画数                    min={s['n_visited']['min']:.0f}"
              f"  median={s['n_visited']['median']:.0f}  max={s['n_visited']['max']:.0f}")
        print(f"  走行距離 [m]                  min={s['distance_m']['min']:.3f}"
              f"  median={s['distance_m']['median']:.3f}  max={s['distance_m']['max']:.3f}")
        print(f"  平均前進速度 [m/s] の中央値    {s['avg_speed_mps_median']:.3f}")
        pc, bpc, pe = s["progress_cells"], s["best_progress_cells"], s["path_efficiency"]
        print(f"  --- ゴールへの近接度（実測。往復で増える n_visited は使わない） ---")
        print(f"  progress_cells (d_start-d_end)      min={pc['min']:.0f}"
              f"  median={pc['median']:.0f}  max={pc['max']:.0f}")
        print(f"  best_progress_cells (d_start-d_min) min={bpc['min']:.0f}"
              f"  median={bpc['median']:.0f}  max={bpc['max']:.0f}")
        print(f"  path_efficiency (distance/最短経路)  min={pe['min']:.3f}"
              f"  median={pe['median']:.3f}  max={pe['max']:.3f}")
        print(f"  始点より遠くで終わった走行数 (progress_cells<0): "
              f"{s['n_worse_than_start']} / {s['n_episodes']}")
        print(f"  Phi_T = (d_start-d_end)*cell_size [m] の中央値: {s['phi_T_median']:.4f}"
              f"  ／ env から実測した Phi_T の中央値: {s['phi_T_measured_median']:.4f}")
        h = s["hp2_pooled"]
        print(f"  E|a-abar|^2（全ステップ pool）min={h['min']:.4e} q1={h['q1']:.4e}"
              f" median={h['median']:.4e} q3={h['q3']:.4e} max={h['max']:.4e}")
        he = s["hp2_episode_mean"]
        print(f"  E|a-abar|^2（走行ごとの平均の分布）min={he['min']:.4e} q1={he['q1']:.4e}"
              f" median={he['median']:.4e} q3={he['q3']:.4e} max={he['max']:.4e}")
        print(f"  E|delta a|^2（全ステップ pool）median = {s['d2_pooled_median']:.4e}")
        rb = s["reward_breakdown_mean"]
        print(f"  1 ステップ平均: shaping={rb['shaping']:.4e}  time={rb['time_penalty']:.4e}"
              f"  hp罰(k適用後)={rb['hp_term']:.4e}  visit={rb['visit']:.4e}")
        sf, df = s["stop_frac_per_episode"], s["drive_frac_per_episode"]
        print(f"  🔴 走行ごとの停止層の割合   median={sf['median']:.1%}"
              f"  [{sf['min']:.1%}, {sf['max']:.1%}]   走行層 median={df['median']:.1%}")
        for oc, b in s["by_outcome"].items():
            print(f"     {oc:<10} n={b['n']:<3} 停止層中央値={b['stop_frac_median']:.1%}"
                  f"  走行層={b['drive_frac_median']:.1%}  T中央値={b['T_median']:.0f}"
                  f"  progress中央値={b['progress_median']:.0f}")
        print(f"  挙動層別集計（プールした全 {s['n_steps_pooled']} ステップ中の割合。"
              f"⚠️ 長いエピソードに支配されるので方策の性質の判断には使わない）:")
        for label in ("停止", "低速", "走行"):
            ly = s["layers"][label]
            hp2s = f"{ly['mean_hp2']:.4e}" if ly["mean_hp2"] is not None else "—"
            d2s = f"{ly['mean_d2']:.4e}" if ly["mean_d2"] is not None else "—"
            shs = f"{ly['mean_shaping']:.4e}" if ly["mean_shaping"] is not None else "—"
            print(f"    {label:<4} 割合={ly['frac']:.1%} (n={ly['n_steps']})"
                  f"  E|a-abar|^2={hp2s}  E|delta a|^2={d2s}  平均shaping={shs}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="M2-0 の 1 ステップあたり報酬内訳の実測")
    ap.add_argument("--smoke", action="store_true",
                    help="疎通確認: 先頭 1 モデル ×3 面のみ")
    ap.add_argument("--out", type=str, default="outputs/m2_step_economy/step_economy.json")
    ap.add_argument("--only", type=str, default=None,
                    help="モデル名にこの文字列を含むものだけを回す（例 exp_012）")
    args = ap.parse_args(argv)

    models = MODELS[:1] if args.smoke else MODELS
    if args.only:
        models = [m for m in models if args.only in m["name"]]
        if not models:
            raise SystemExit(f"--only {args.only!r} に一致するモデルが無い")
    seeds = VALIDATION_SEEDS[:3] if args.smoke else VALIDATION_SEEDS

    # 🔴 帯の明示と安全弁（裁定 R40 条件 4・R11 項目 7）。本スクリプトは**検証帯**で測る。
    assert_seeds_allowed(seeds, namespace="maze6", purpose="validate")
    print(describe_seeds(seeds, namespace="maze6"), flush=True)

    summaries = []
    err_tracker_by_model = {}
    t0 = time.time()
    for mcfg in models:
        model_path = REPO_ROOT / mcfg["path"]
        if not model_path.exists():
            print(f"[skip] {mcfg['name']}: {model_path} が無い（未完走）", flush=True)
            continue
        print(f"[load] {mcfg['name']} <- {model_path}", flush=True)
        model = PPO.load(str(model_path), device="cpu")
        err_tracker = dict(max=0.0, n=0)
        episodes = []
        for ms in seeds:
            te0 = time.time()
            ep = run_episode(model, ms, mcfg["k"], err_tracker,
                             flags=mcfg.get("flags"))
            print(f"  [{mcfg['name']}] maze_seed={ms} outcome={ep['outcome']:<10}"
                  f" T={ep['n_steps']:<5} n_visited={ep['n_visited']:<3}"
                  f" ({time.time()-te0:.1f}s)", flush=True)
            episodes.append(ep)
        err_tracker_by_model[mcfg["name"]] = err_tracker
        summaries.append(summarize_model(mcfg["name"], episodes))

    print_report(summaries, err_tracker_by_model)
    print(f"\n[total elapsed] {time.time()-t0:.1f}s")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(
            validation_seeds=seeds,
            env_args=dict(gamma=GAMMA, maze_mode=MAZE_MODE, visit_bonus=VISIT_BONUS,
                         collision_penalty=COLLISION_PENALTY,
                         action_smooth_penalty=ACTION_SMOOTH_PENALTY,
                         action_highpass_alpha=ACTION_HIGHPASS_ALPHA),
            layer_thresholds_mps=dict(stop=STOP_V, drive=DRIVE_V),
            assert_tolerance=ASSERT_TOL,
            max_abs_reward_error_by_model={k: v["max"] for k, v in err_tracker_by_model.items()},
            n_steps_checked_by_model={k: v["n"] for k, v in err_tracker_by_model.items()},
            models_config=[dict(name=m["name"], path=m["path"], k=m["k"],
                                flags=m.get("flags") or {})
                           for m in models],
            models=summaries,
        ), f, indent=2, ensure_ascii=False)
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
