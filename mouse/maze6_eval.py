"""
mouse/maze6_eval.py
===================
M2-0（6x6 迷路の単走）の評価器。`mouse/corridor_eval.py` と同じ様式で、
**方策を決定論的に走らせ、迷路ごとに複数試行して指標を集計する**。

## 帯の分割（研究計画書 §9-7、`experiments/m2_design.md` の裁定 D）

| 用途 | maze seed | 既定の試行数 |
|---|---|---|
| 学習 | 8000 以降（予約帯は `Maze6Env._next_maze_seed()` が決定的に読み飛ばす） | — |
| 日常判断・チェックポイント選択 | **検証帯 7000-7019** | 1 |
| gate 判定 | **評価帯 6000-6019** | 5 |

## 集計の作法（M1 で確立。`experiments/exp_006_action_smoothness/card.md` §7）

- **速度・符号反転・最小壁余裕は「壁接触なしでゴールした走行のみ」で集計する。**
  失敗走行は壁に当たるまでの短い区間の値であり、成功走行と同じ土俵の量ではない
- 完走できなかった方策の指標を、完走できた方策の指標と混ぜない
- 結論は「x / N seed」の形で書き、条件間の比較は同じ seed 数で揃える

使い方:
    .venv/bin/python -m mouse.maze6_eval --model models/exp_010_m2_0.zip
    .venv/bin/python -m mouse.maze6_eval --model ... --split validation
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from mouse.clearance import HF_WORST, ClearanceMeter, hf_energy_ratio
from mouse.maze6_env import Maze6Env
from mouse.params import RobotParams

EVAL_MAZE_DIR = "assets/maze6/loop/eval"              # seed 6000-6019
VALIDATION_MAZE_DIR = "assets/maze6/loop/validation"  # seed 7000-7019
DEFAULT_N_TRIALS = 5


def _git_rev() -> str:
    """現在の HEAD（dirty なら末尾に -dirty）。corridor_eval と同じ実装を使う。"""
    from mouse.corridor_eval import _git_rev as _r
    return _r()


def _trial_seed(base: int, maze_seed: int, trial_idx: int) -> int:
    """(base, maze_seed, trial) から決定的に試行 seed を導く（corridor_eval と同じ規約）。"""
    return (base * 1_000_003 + maze_seed * 10_007 + trial_idx) % (2 ** 31 - 1)


def _run_one_trial(env, policy_fn, tseed: int, keep_trace: bool = False) -> dict:
    obs, info = env.reset(seed=tseed)
    p = RobotParams()
    prev_sign = np.sign(np.zeros(2))
    flips = np.zeros(2, dtype=int)
    acts, cur, n_steps = [], [], 0
    omegas, poses, times = [], [], []
    outcome = "timeout"
    # 最小壁余裕（安全余裕そのもの。横偏差では測れない。mouse/clearance.py 参照）
    meter = ClearanceMeter(env.sim)
    s_ = env.sim
    wheel_adr = (s_._left_wheel_qvel_adr, s_._right_wheel_qvel_adr)
    # 旋回回数（時間の主要因。迷路では分岐ごとに出る）
    _, _, yaw_prev = s_.privileged_pose()
    yaw_acc, n_turns = 0.0, 0
    while True:
        a = np.clip(np.asarray(policy_fn(obs), dtype=np.float64), -1.0, 1.0)
        acts.append(a)
        sg = np.sign(a)
        flips += (sg * prev_sign < 0).astype(int)
        prev_sign = sg
        v = a * p.voltage_limit
        w0 = np.array([s_.data.qvel[wheel_adr[0]], s_.data.qvel[wheel_adr[1]]])
        obs, _r, term, trunc, info = env.step(a)
        w1 = np.array([s_.data.qvel[wheel_adr[0]], s_.data.qvel[wheel_adr[1]]])
        # モータ電流 I = (V − K_e·N·ω_w)/R。ω は制御周期の前後平均で代表
        cur.append((v - p.motor_Ke * p.gear_ratio * 0.5 * (w0 + w1)) / p.motor_R)
        meter.update()
        n_steps += 1

        x, y, yaw = s_.privileged_pose()
        dyaw = math.atan2(math.sin(yaw - yaw_prev), math.cos(yaw - yaw_prev))
        yaw_prev = yaw
        if yaw_acc * dyaw < 0.0:
            yaw_acc = 0.0
        yaw_acc += dyaw
        if abs(yaw_acc) >= math.pi / 4:
            n_turns += 1
            yaw_acc = 0.0
        if keep_trace:
            omegas.append(0.5 * (w0 + w1))
            poses.append((x, y, yaw))
            times.append(float(info.get("sim_time", 0.0)))
        if term or trunc:
            outcome = ("goal" if info.get("goal") else
                       "collision" if info.get("collision") else "timeout")
            break

    sim_time = float(info["sim_time"])
    acts = np.array(acts)
    cur = np.array(cur)
    d = np.diff(acts, axis=0, prepend=np.zeros((1, 2)))
    no_contact_goal = bool(outcome == "goal")
    return dict(
        outcome=outcome,
        no_contact_goal=no_contact_goal,
        n_steps=n_steps,
        sim_time=sim_time,
        # 到達までの所要時間（ゴールした走行のみ意味を持つ）
        goal_time_s=sim_time if no_contact_goal else None,
        # 1 区画あたり所要時間: 最短歩数で割る（迷路ごとの難度で正規化する）
        sec_per_cell=(sim_time / info["d_start"]) if (no_contact_goal
                                                      and info.get("d_start")) else None,
        n_visited=int(info.get("n_visited", 0)),
        odom_error_m=float(info.get("odom_error_m", float("nan"))),
        min_wall_clearance_m=(float(meter.worst)
                              if math.isfinite(meter.worst) else None),
        action_diff_rms=float(np.sqrt((d ** 2).sum(axis=1).mean())),
        sign_flip_rate_left=float(flips[0] / sim_time) if sim_time > 0 else 0.0,
        sign_flip_rate_right=float(flips[1] / sim_time) if sim_time > 0 else 0.0,
        # --- 駆動系の指標（M1 では後から足して測り直しになったので最初から入れる）---
        # モータ電流の RMS（左右平均）。M1 で一次指標に採用（card.md §3-9）
        i_rms=float(np.sqrt((cur ** 2).mean())),
        i_dc=float(np.abs(cur.mean(axis=0)).mean()),
        i_ac=float(np.sqrt(np.maximum(
            (cur ** 2).mean(axis=0) - cur.mean(axis=0) ** 2, 0.0)).mean()),
        n_turns=int(n_turns),
        trace=(dict(t=[round(v, 4) for v in times],
                    action=[[round(float(a[0]), 5), round(float(a[1]), 5)] for a in acts],
                    wheel_omega=[[round(float(w[0]), 4), round(float(w[1]), 4)]
                                 for w in omegas],
                    pose=[[round(v, 5) for v in q] for q in poses])
               if keep_trace else None),
        # 車輪が追従できない帯域の指令成分。理論最悪値 HF_WORST=0.762 で正規化して読む
        hf_ratio=float(hf_energy_ratio(acts)),
    )


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def evaluate_maze6(policy_fn, maze_dir=EVAL_MAZE_DIR, n_trials=DEFAULT_N_TRIALS,
                   seed=0, gamma=0.995, maze_mode="loop",
                   keep_traces: bool = True, model_sha256: str = "unknown") -> dict:
    """M2-0 の評価を実行する。

    Args:
        policy_fn: obs -> action。SB3 なら
            `lambda o: model.predict(o, deterministic=True)[0]`。
        maze_dir: 迷路 npz+xml のディレクトリ（既定は評価帯 6000-6019）。
        n_trials: 迷路あたりの試行数（gate は 20 面 ×5 = 100 試行）。

    Returns:
        dict: goal_rate（gate 本体）ほかの指標を含む summary。
        **速度・反転・電流はゴールした走行のみで集計する。**
    """
    maze_dir = Path(maze_dir)
    npz_paths = sorted(maze_dir.glob("maze6_*.npz"))
    if not npz_paths:
        raise ValueError(f"迷路が見つかりません: {maze_dir}。"
                         f"`python -m mouse.maze6_gen --split ... --seeds ...` で生成してください")
    maze_seeds = [int(np.load(p)["seed"]) for p in npz_paths]

    per_maze, all_trials = [], []
    for ms in maze_seeds:
        env = Maze6Env(maze_dir=maze_dir, maze_seeds=[ms], max_cache=2,
                       gamma=gamma, mode="fixed", maze_mode=maze_mode)
        trials = [_run_one_trial(env, policy_fn, _trial_seed(seed, ms, t),
                                 keep_trace=keep_traces)
                  for t in range(n_trials)]
        env.close() if hasattr(env, "close") else None
        all_trials.extend(trials)
        per_maze.append(dict(maze_seed=ms, n_trials=n_trials,
                             n_goal=sum(1 for r in trials if r["no_contact_goal"]),
                             n_collision=sum(1 for r in trials
                                             if r["outcome"] == "collision"),
                             n_timeout=sum(1 for r in trials
                                           if r["outcome"] == "timeout"),
                             trials=trials))

    n = len(all_trials)
    ok = [r for r in all_trials if r["no_contact_goal"]]      # **成功走行のみ**

    # 代表走行の trace を決定的な規則で選び、本体は per_maze から落とす
    traces = {}
    if keep_traces:
        flat = [(pm["maze_seed"], i, t) for pm in per_maze
                for i, t in enumerate(pm["trials"])]
        good = sorted([r for r in flat if r[2]["no_contact_goal"]],
                      key=lambda r: 0.5 * (r[2]["sign_flip_rate_left"]
                                           + r[2]["sign_flip_rate_right"]))
        if good:
            for tag, rec in (("median_flip", good[len(good) // 2]),
                             ("max_flip", good[-1])):
                if rec[2].get("trace"):
                    traces[tag] = dict(maze_seed=rec[0], trial_index=rec[1],
                                       **rec[2]["trace"])
        bad = [r for r in flat if not r[2]["no_contact_goal"]]
        if bad and bad[0][2].get("trace"):
            traces["first_failure"] = dict(maze_seed=bad[0][0], trial_index=bad[0][1],
                                           **bad[0][2]["trace"])
    for pm in per_maze:
        for t in pm["trials"]:
            t.pop("trace", None)

    return dict(
        # --- 来歴（再実行で検算できるようにするため）---
        git_rev=_git_rev(), model_sha256=model_sha256, eval_schema_version=2,
        traces=traces,
        n_mazes=len(maze_seeds), n_trials_per_maze=n_trials, n_total_trials=n,
        goal_rate=sum(1 for r in all_trials if r["no_contact_goal"]) / n,
        collision_rate=sum(1 for r in all_trials if r["outcome"] == "collision") / n,
        timeout_rate=sum(1 for r in all_trials if r["outcome"] == "timeout") / n,
        n_success=len(ok),
        # --- 以下はすべて成功走行のみの集計 ---
        mean_goal_time_s=_mean([r["goal_time_s"] for r in ok]),
        mean_sec_per_cell=_mean([r["sec_per_cell"] for r in ok]),
        mean_n_visited=_mean([r["n_visited"] for r in ok]),
        mean_odom_error_m=_mean([r["odom_error_m"] for r in ok]),
        action_diff_rms_mean=_mean([r["action_diff_rms"] for r in ok]),
        sign_flip_rate_mean=_mean([0.5 * (r["sign_flip_rate_left"]
                                          + r["sign_flip_rate_right"]) for r in ok]),
        i_rms_mean=_mean([r["i_rms"] for r in ok]),
        i_dc_mean=_mean([r["i_dc"] for r in ok]),
        hf_ratio_mean=_mean([r["hf_ratio"] for r in ok]),
        hf_ratio_worst_norm=(_mean([r["hf_ratio"] for r in ok]) / HF_WORST) if ok else None,
        min_wall_clearance_min_m=(min(r["min_wall_clearance_m"] for r in ok)
                                  if ok else None),
        min_wall_clearance_mean_m=_mean([r["min_wall_clearance_m"] for r in ok]),
        failed_maze_seeds=[pm["maze_seed"] for pm in per_maze if pm["n_goal"] == 0],
        maze_dir=str(maze_dir), seed=seed, gamma=gamma, maze_mode=maze_mode,
        per_maze=per_maze,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="M2-0（6x6 迷路の単走）の評価")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--split", choices=["eval", "validation"], default="eval")
    ap.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    maze_dir = EVAL_MAZE_DIR if args.split == "eval" else VALIDATION_MAZE_DIR
    s = evaluate_maze6(lambda o: model.predict(o, deterministic=True)[0],
                       maze_dir=maze_dir, n_trials=args.n_trials, seed=args.seed)

    print(f"=== M2-0 評価: {args.model} / {args.split}帯 "
          f"({s['n_mazes']} 面 ×{s['n_trials_per_maze']} 試行 = {s['n_total_trials']} 試行) ===")
    print(f"  ゴール到達率     {s['goal_rate']:.2f}"
          f"  （衝突 {s['collision_rate']:.2f} / 時間切れ {s['timeout_rate']:.2f}）")
    print(f"  --- 以下はゴールした {s['n_success']} 走行のみの集計 ---")
    for label, key, fmt in [("到達時間 [s]", "mean_goal_time_s", ".2f"),
                            ("1 歩あたり [s]", "mean_sec_per_cell", ".3f"),
                            ("訪問区画数", "mean_n_visited", ".1f"),
                            ("オドメトリ誤差 [m]", "mean_odom_error_m", ".4f"),
                            ("符号反転 [回/s]", "sign_flip_rate_mean", ".1f"),
                            ("HF比", "hf_ratio_mean", ".3f"),
                            ("　（理論最悪 0.762 比）", "hf_ratio_worst_norm", ".1%"),
                            ("RMS 電流 [A]", "i_rms_mean", ".3f"),
                            ("直流成分 [A]", "i_dc_mean", ".3f"),
                            ("最小壁余裕 最小 [m]", "min_wall_clearance_min_m", ".4f"),
                            ("　　　　　 平均 [m]", "min_wall_clearance_mean_m", ".4f")]:
        v = s[key]
        print(f"  {label:<20}" + (format(v, fmt) if v is not None else "—"))
    i_cont = 1.16e-3 / RobotParams().motor_Kt
    if s["i_rms_mean"]:
        print(f"  （連続定格 {i_cont:.3f} A の {s['i_rms_mean']/i_cont:.2f} 倍）")
    if s["failed_maze_seeds"]:
        print(f"  全滅した迷路: {s['failed_maze_seeds']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
        print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
