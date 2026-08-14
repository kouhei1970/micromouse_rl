#!/usr/bin/env python3
"""exp_019 の P5 を判定する（裁定 R50 の判定量 ＋ 裁定 R51 の測定経路と 1 対 1 対応）。

Measure P5 for exp_019 (one-to-one with the pre-registered clauses in card.md §3).

🔴 **本スクリプトは判定の道具であり、学習の道具ではない。**
`mouse/` 配下（投入版の実行コード）には一切触れない。`_trial_seed` は
`mouse/maze6_eval.py` の既存関数を **import して使う**（再定義しない）。

条文（`card.md` §3「P5 の測定経路」）との対応:

| 条文 | 実装 |
|---|---|
| 各 seed の**最終方策**で rollout（`deterministic=True`） | `_load_policy()` ＋ `model.predict(obs, deterministic=True)` |
| **学習環境**（v2・リスポーン有り・学習時と同じ上限 2000 歩） | `_ENV_KWARGS`（`ENV_VERSION_FLAGS['v2']` と同値）＋ 条件 E |
| **検証帯 7000-7019 の 20 面 × 各 1 エピソード** | `VALIDATION_MAZE_DIR` の npz を全数・1 面 1 本 |
| **毎歩の `dist_to_goal` と `respawned` を記録** | `_run_episode()` の `d_hist` / `resp_hist` |
| 窓「**最後のリスポーン以降**」の境界は `respawned` で切る | `_judge_episode()` の `last` |
| reset seed = `_trial_seed(base=0, maze_seed, trial_idx=0)` | `_SEED_BASE` / `_TRIAL_IDX`（定数として固定） |
| 母集団 = **リスポーンを 1 回以上経験したエピソード** | `_judge_episode()` が `None` を返す面は分母から外れる |
| 判定 = `min D(t) <= D0 - 1` の割合 → **6 seed の中央値 >= 50%** | `summarize()` |
| **プール集計はしない**（§9-18） | seed ごとに割合を出し、**中央値だけを取る** |
| 分母 0 の seed は**欠測として除外し n を併記** | `summarize()` の `excluded_seeds` / `n_denominator` |
| 6 seed すべて分母 0 なら**対象消滅（判定不能）** | `verdict = "undecidable_no_respawn"` |
| 報告時要件（この n で判別できる差の大きさ） | `resolution`（= 1/分母。seed ごとに出力） |

使い方:
    .venv/bin/python experiments/exp_019_env_v2_baseline/measure_p5.py \
        --models models/exp_019_v2_seed1.zip ... models/exp_019_v2_seed6.zip \
        --out outputs/exp_019_p5.json
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stable_baselines3 import PPO  # noqa: E402

from common.seed_bands import assert_seeds_allowed, describe_seeds  # noqa: E402
from mouse.maze6_env import Maze6Env  # noqa: E402
# 🔴 `_trial_seed` は**既存の評価規約をそのまま使う**（条文の指定。再定義しない）
from mouse.maze6_eval import VALIDATION_MAZE_DIR, _trial_seed  # noqa: E402

GAMMA = 0.995

# reset seed の規則（条文で値まで固定されている。**判定時に選ぶ余地を残さない**）。
# base=0・trial_idx=0 は**定期評価と同じ値**なので、P5 の rollout は
# 定期評価の第 0 試行と**同一の初期条件**から始まる（照合可能性のため意図的）。
_SEED_BASE = 0
_TRIAL_IDX = 0

# 学習環境 v2（`train.py` の `ENV_VERSION_FLAGS['v2']` と同値）＋ 条件 E。
# 報酬側の係数は渡していない — **観測に報酬・Φ は入らない**ので、方策が決める軌道は
# 報酬パラメータに依存しない（`Maze6Env._make_observation()`）。
_ENV_KWARGS = dict(
    continuous_potential=True,       # 条件 E（Φ の実現方法。軌道には効かない）
    goal_rule_containment=True,      # v2: 規約終端（機体全体の内包）
    collision_respawn=True,          # v2: 衝突リスポーン（P5 の前提事象）
    episode_limit_steps=2000,        # v2: 学習時と同じエピソード上限
)


def _load_policy(model_path: Path):
    """最終方策を読み込み、obs -> action の関数にする。"""
    model = PPO.load(str(model_path), device="cpu")
    return lambda o: model.predict(o, deterministic=True)[0]


def _run_episode(maze_dir: Path, maze_seed: int, policy_fn) -> dict:
    """1 面 1 エピソードを回し、**毎歩の D(t) と respawned** を返す。

    Run one episode on one maze and record D(t) and the respawn flag every step.
    """
    env = Maze6Env(maze_dir=maze_dir, maze_seeds=[maze_seed], max_cache=2,
                   gamma=GAMMA, mode="fixed", maze_mode="loop", **_ENV_KWARGS)
    tseed = _trial_seed(_SEED_BASE, maze_seed, _TRIAL_IDX)
    obs, info = env.reset(seed=tseed)
    d0 = int(info["dist_to_goal"])          # D_0 = 開始区画のゴール距離［区画数］
    d_hist, resp_hist = [], []
    outcome = "timeout"
    while True:
        a = np.clip(np.asarray(policy_fn(obs), dtype=np.float64), -1.0, 1.0)
        obs, _r, terminated, truncated, info = env.step(a)
        d_hist.append(int(info["dist_to_goal"]))
        # `collision_respawn=True` なら **毎歩出る**（`maze6_env.py:1157-1159`）
        resp_hist.append(bool(info["respawned"]))
        if terminated:
            outcome = "goal" if info.get("goal") else "collision"
            break
        if truncated:
            outcome = "timeout"
            break
    if hasattr(env, "close"):
        env.close()
    return dict(maze_seed=maze_seed, trial_seed=int(tseed), d0=d0, outcome=outcome,
                n_steps=len(d_hist), n_respawn=int(info.get("n_respawn", 0)),
                d_hist=d_hist, resp_hist=resp_hist)


def _judge_episode(ep: dict) -> dict | None:
    """1 エピソードの P5 判定。**母集団に入らない面は None を返す**。

    判定量 = **最後のリスポーン以降**の min D(t)。成立 = `min D(t) <= D_0 - 1`。

    窓の左端は**リスポーンが起きた歩そのものを含む**。その歩の `dist_to_goal` は
    **開始区画の距離 ＝ D_0** である（`maze6_env.py` の step は
    `_respawn_to_start()` の後に `cell = self._cell` を取り直してから info を作る）ので、
    **含めても含めなくても min の判定は変わらない**（D_0 は「<= D_0 - 1」を満たさない）。
    """
    if not any(ep["resp_hist"]):
        return None                                  # 母集団外（リスポーン経験なし）
    last = max(i for i, r in enumerate(ep["resp_hist"]) if r)
    window = ep["d_hist"][last:]
    min_d = min(window)
    return dict(maze_seed=ep["maze_seed"], d0=ep["d0"], outcome=ep["outcome"],
                n_respawn=ep["n_respawn"], last_respawn_step=last + 1,
                window_len=len(window), min_d_after_last_respawn=min_d,
                advanced=bool(min_d <= ep["d0"] - 1))


def summarize(per_seed: dict) -> dict:
    """seed ごとの割合 → **6 seed の中央値**（プール集計はしない。§9-18）。"""
    rates, excluded = {}, []
    for name, judged in per_seed.items():
        denom = len(judged)
        if denom == 0:
            excluded.append(name)                    # 分母 0 は**欠測として除外**
            continue
        n_pass = sum(1 for j in judged if j["advanced"])
        rates[name] = dict(n_denominator=denom, n_pass=n_pass, rate=n_pass / denom,
                           # 報告時要件: この n で判別できる差の大きさ（割合の刻み）
                           resolution=1.0 / denom)
    if not rates:
        return dict(verdict="undecidable_no_respawn", median_rate=None,
                    per_seed=rates, excluded_seeds=excluded,
                    note="6 seed すべて分母 0 ＝ 前提事象（リスポーン）が不発生。対象消滅。")
    med = statistics.median([v["rate"] for v in rates.values()])
    return dict(verdict=("P5_holds" if med >= 0.50 else "P5_fails"),
                median_rate=med, threshold=0.50,
                n_seeds_used=len(rates), per_seed=rates, excluded_seeds=excluded,
                note=("判定は seed ごとの割合の中央値。プール集計はしない（§9-18）。"
                      "各 seed の割合の刻みは 1/分母（resolution）。"))


def main() -> None:
    p = argparse.ArgumentParser(description="exp_019 P5 の判定（R50 判定量 ＋ R51 測定経路）")
    p.add_argument("--models", nargs="+", required=True, help="最終方策の .zip（6 seed 分）")
    p.add_argument("--maze-dir", type=str, default=VALIDATION_MAZE_DIR)
    p.add_argument("--out", type=str, default="outputs/exp_019_p5.json")
    args = p.parse_args()

    maze_dir = Path(args.maze_dir)
    maze_seeds = sorted(int(np.load(q)["seed"]) for q in maze_dir.glob("maze6_*.npz"))
    if not maze_seeds:
        raise SystemExit(f"迷路が見つかりません: {maze_dir}")
    # 帯の明示と安全弁（§9-7・R11 バッチ項目 7）。P5 は**測定**なので purpose='validate'
    print(describe_seeds(maze_seeds, "maze6"))
    assert_seeds_allowed(maze_seeds, "maze6", "validate")

    per_seed, raw = {}, {}
    for m in args.models:
        name = Path(m).stem
        policy_fn = _load_policy(Path(m))
        episodes = [_run_episode(maze_dir, ms, policy_fn) for ms in maze_seeds]
        judged = [j for j in (_judge_episode(e) for e in episodes) if j is not None]
        per_seed[name] = judged
        raw[name] = [dict(maze_seed=e["maze_seed"], trial_seed=e["trial_seed"],
                          d0=e["d0"], outcome=e["outcome"], n_steps=e["n_steps"],
                          n_respawn=e["n_respawn"]) for e in episodes]
        print(f"[{name}] 母集団 {len(judged)}/{len(episodes)} 面"
              f"（リスポーン経験あり） 成立 {sum(1 for j in judged if j['advanced'])}")

    summary = summarize(per_seed)
    out = dict(
        clause="card.md §3「P5 の測定経路」（裁定 R50 判定量 ＋ R51 測定経路）",
        seed_rule=dict(fn="mouse.maze6_eval._trial_seed", base=_SEED_BASE,
                       trial_idx=_TRIAL_IDX,
                       note="定期評価の第 0 試行と同一の初期条件"),
        env_kwargs=_ENV_KWARGS, maze_dir=str(maze_dir), n_mazes=len(maze_seeds),
        summary=summary, detail={k: v for k, v in per_seed.items()}, episodes=raw,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
