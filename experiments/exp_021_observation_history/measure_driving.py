#!/usr/bin/env python3
"""exp_021 の走行制御の中間量を測る（カード §4-1 の条文と 1 対 1 対応）。

Measure the driving-control quantities for exp_021 (one-to-one with card.md §4-1).

🔴 **本スクリプトは判定の道具であり、学習の道具ではない。**
`mouse/` 配下（投入版の実行コード）には一切触れない。`_trial_seed` は
`mouse/maze6_eval.py` の既存関数を **import して使う**（再定義しない）。
実装は `experiments/exp_019_env_v2_baseline/measure_p5.py` を土台にしている。

## 条文（`card.md` §4-1・§4-2-bis・§5）との対応

| 条文 | 実装 |
|---|---|
| 各 seed の**最終方策**で rollout（`deterministic=True`） | `_load_policy()` ＋ `predict(..., deterministic=True)` |
| **学習環境 v2**（リスポーン有り・上限 2000 歩） | `_ENV_KWARGS` |
| **検証用の 20 迷路（seed 7000-7019）× 各 1 エピソード** | `VALIDATION_MAZE_DIR` の npz を全数・1 迷路 1 本 |
| **迷路ごとに env を作り直す**（`maze_seeds=[ms]`・`mode="fixed"`） | `_run_episode()` が毎回構築 |
| reset seed = `_trial_seed(base=0, maze_seed, trial_idx=0)` | `_SEED_BASE` / `_TRIAL_IDX`（定数で固定） |
| **毎歩の `dist_to_goal`・`respawned`・`cell_entries` を記録** | `_run_episode()` の `d_hist` / `resp_hist` / `ce_hist` |
| **Q1** = (D₀ − エピソード中の最小 D) ÷ 歩数 × 1000 | `_episode_metrics()` の `net_progress_per_1000` |
| **Q2** = `respawned` が真の歩数 ÷ 歩数 × 1000 | 同 `respawn_per_1000` |
| 監視 = 歩数 ÷ `cell_entries`・`cell_entries` ÷ `n_visited` | 同 `steps_per_entry` / `entries_per_distinct_cell` |
| **Q4**（立て直し・R51 確定仕様） | `_judge_p5()`（`measure_p5.py` の `_judge_episode()` と同一の論理） |
| 集約 = **迷路 20 本の中央値 → 6 seed の中央値**（プール集計はしない） | `summarize()` |
| **観測履歴のラッパを学習と同じ形で掛ける** | `--history-lags`（**対照群は渡さない**） |
| **D < 0 が出たら止める**（`-1` を最小値に採ると「ゴールした」ように見える） | `_run_episode()` の `ValueError` |

## 使い方

```bash
# 対照群（exp_019）— 最終方策 6 本（200 万歩。Q1・Q2・Q4 の対照）
.venv/bin/python experiments/exp_021_observation_history/measure_driving.py \
    --models models/exp_019_v2_seed{1,2,3,4,5,6}.zip \
    --label exp_019_final --out outputs/exp_021_driving_control_final.json

# 対照群（exp_019）— 80 万歩の退避重み 6 本（§4-2-bis の報告トリガーの対照）
.venv/bin/python experiments/exp_021_observation_history/measure_driving.py \
    --models logs/exp_019_v2_seed{1,2,3,4,5,6}/rl_model_800000_steps.zip \
    --label exp_019_800k --out outputs/exp_021_driving_control_800k.json

# 介入群（exp_021）— 履歴ラッパを学習と同じ遅れで掛ける
.venv/bin/python experiments/exp_021_observation_history/measure_driving.py \
    --models models/exp_021_seed{1,2,3,4,5,6}.zip \
    --history-lags "1,2,4,8,16,32,64,128" \
    --label exp_021_final --out outputs/exp_021_driving_treat_final.json
```
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
from mouse.obs_history import ObsHistoryWrapper, parse_lags  # noqa: E402
from mouse.recurrent import RecurrentPolicyFn  # noqa: E402

GAMMA = 0.995

# reset seed の規則（条文で値まで固定されている。**判定時に選ぶ余地を残さない**）。
_SEED_BASE = 0
_TRIAL_IDX = 0

# 学習環境 v2（`train.py` の `ENV_VERSION_FLAGS['v2']` と同値）＋ 条件 E。
# 🔴 **学習環境で測る理由**（カード §4-1）: 定期評価の環境にはリスポーンが無いので
# 「すぐ衝突する方策ほど歩数が短い」＝ 歩数を分母に持つ量に生き残りの偏りが入る。
# 学習環境 v2 はゴールするか 2000 歩に達するまで終わらないので、分母がほぼ全走行で揃う。
_ENV_KWARGS = dict(
    continuous_potential=True,       # 条件 E（Φ の実現方法。軌道には効かない）
    goal_rule_containment=True,      # v2: 規約終端（機体全体の内包）
    collision_respawn=True,          # v2: 衝突リスポーン
    episode_limit_steps=2000,        # v2: 学習時と同じエピソード上限
)


def _model_name(path: Path) -> str:
    """集計キーに使う一意な呼び名。

    🔴 **ファイル名だけでは一意にならない。**退避重みは 6 seed とも
    `logs/exp_019_v2_seed{N}/rl_model_800000_steps.zip` で**ファイル名が同一**なので、
    `Path.stem` を鍵にすると辞書が上書きされ、**6 seed の中央値が 1 本の値になる**
    （2026-08-14 に実際に踏んだ。ログに同じ呼び名が並んで気づいた）。
    親ディレクトリ名を前置して区別する。
    """
    return f"{path.parent.name}/{path.stem}"


def _load_policy(model_path: Path, recurrent: bool = False):
    """方策を読み込み、(呼び出し口, 実歩数) を返す。

    **前向きの方策は無状態**（`lambda o: ...`）。
    **再帰型方策（exp_023）は隠れ状態を持つ**ので `RecurrentPolicyFn` を返す —
    **エピソードの切り替わりで `reset()` を呼ばないと、前のエピソードの文脈を引き継ぐ**
    （**例外も出ず次元も合い、値だけが誤る**）。
    """
    if recurrent:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(str(model_path), device="cpu")
        return RecurrentPolicyFn(model, deterministic=True), int(model.num_timesteps)
    model = PPO.load(str(model_path), device="cpu")
    return (lambda o: model.predict(o, deterministic=True)[0],
            int(model.num_timesteps))


def _run_episode(maze_dir: Path, maze_seed: int, policy_fn, lags, sham=False) -> dict:
    """1 迷路 1 エピソードを回し、**毎歩の D(t)・respawned・cell_entries** を返す。

    Run one episode on one maze; record D(t), the respawn flag and cell entries.
    """
    env = Maze6Env(maze_dir=maze_dir, maze_seeds=[maze_seed], max_cache=2,
                   gamma=GAMMA, mode="fixed", maze_mode="loop", **_ENV_KWARGS)
    if lags:
        # 学習時と同じ形で掛ける（観測の形が学習と評価で食い違う事故を防ぐ）。
        # exp_022: sham=True で「にせ履歴」（遅れの位置に現在の観測を複製）
        env = ObsHistoryWrapper(env, lags, sham=sham)
    tseed = _trial_seed(_SEED_BASE, maze_seed, _TRIAL_IDX)
    # 🔴 再帰型方策は 1 エピソードごとに隠れ状態を捨てる（exp_023）
    if hasattr(policy_fn, "reset"):
        policy_fn.reset()
    obs, info = env.reset(seed=tseed)
    d0 = int(info["dist_to_goal"])          # D_0 = 開始区画のゴール距離［区画数］
    if d0 < 0:
        raise ValueError(f"maze_seed={maze_seed}: reset 直後の dist_to_goal が負（{d0}）")
    d_hist, resp_hist, ce_hist = [], [], []
    outcome = "timeout"
    while True:
        a = np.clip(np.asarray(policy_fn(obs), dtype=np.float64), -1.0, 1.0)
        obs, _r, terminated, truncated, info = env.step(a)
        d = int(info["dist_to_goal"])
        if d < 0:
            # 🔴 `-1` は「距離場にその区画が無い」ことを表す番兵。黙って捨てず止める
            # （そのまま min を取ると「ゴールに着いた」ように見えてしまう）。
            raise ValueError(f"maze_seed={maze_seed} 歩 {len(d_hist)+1}: "
                             f"dist_to_goal が負（{d}）。距離場の欠落を疑うこと")
        d_hist.append(d)
        resp_hist.append(bool(info["respawned"]))
        ce_hist.append(int(info["cell_entries"]))
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
                n_visited=int(info.get("n_visited", 0)),
                cell_entries=ce_hist[-1] if ce_hist else 0,
                d_hist=d_hist, resp_hist=resp_hist)


def _episode_metrics(ep: dict) -> dict:
    """1 エピソードの判定量と監視量（カード §4-1 の定義そのまま）。"""
    n = ep["n_steps"]
    if n <= 0:
        raise ValueError(f"maze_seed={ep['maze_seed']}: 歩数 0 のエピソード")
    min_d = min(ep["d_hist"])
    # Q1: エピソードを通じて**最も深く入り込めた地点** ÷ 歩数（前進の積算ではない）。
    # リスポーンは開始区画へ戻すので最小 D は下がらず、往復では分子を稼げない。
    net_progress = (ep["d0"] - min_d) / n * 1000.0
    # Q2: 衝突による立て直しの頻度
    respawn_rate = sum(1 for r in ep["resp_hist"] if r) / n * 1000.0
    ce = ep["cell_entries"]
    nv = ep["n_visited"]
    return dict(
        maze_seed=ep["maze_seed"], d0=ep["d0"], min_d=min_d, outcome=ep["outcome"],
        n_steps=n, n_respawn=ep["n_respawn"], cell_entries=ce, n_visited=nv,
        net_progress_per_1000=net_progress,          # Q1（大きいほど良い）
        respawn_per_1000=respawn_rate,               # Q2（小さいほど良い）
        # 監視のみ（§7-9。振動で稼げる抜け道があるので判定には使わない）
        steps_per_entry=(n / ce) if ce > 0 else None,
        entries_per_distinct_cell=(ce / nv) if nv > 0 else None,
    )


def _judge_p5(ep: dict) -> dict | None:
    """Q4（立て直し・R51 確定仕様）。**母集団に入らない迷路は None を返す**。

    `measure_p5.py:_judge_episode()` と**同一の論理**（窓 = 最後のリスポーン以降・
    成立 = 窓内の min D <= D0 - 1・母集団 = リスポーンを 1 回以上経験したエピソード）。
    """
    if not any(ep["resp_hist"]):
        return None                                  # 母集団外（リスポーン経験なし）
    last = max(i for i, r in enumerate(ep["resp_hist"]) if r)
    min_d = min(ep["d_hist"][last:])
    return dict(maze_seed=ep["maze_seed"], d0=ep["d0"],
                min_d_after_last_respawn=min_d,
                advanced=bool(min_d <= ep["d0"] - 1))


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def summarize(per_seed: dict) -> dict:
    """**迷路 20 本の中央値 → 6 seed の中央値**。プール集計はしない（研究計画書 §9-18）。"""
    keys = ("net_progress_per_1000", "respawn_per_1000",
            "steps_per_entry", "entries_per_distinct_cell")
    per_seed_medians, p5_rates, p5_excluded = {}, {}, []
    for name, d in per_seed.items():
        mets = d["metrics"]
        per_seed_medians[name] = {k: _median([m[k] for m in mets]) for k in keys}
        per_seed_medians[name]["n_mazes"] = len(mets)
        per_seed_medians[name]["n_goal"] = sum(1 for m in mets if m["outcome"] == "goal")
        judged = d["p5"]
        if len(judged) == 0:
            p5_excluded.append(name)                 # 分母 0 は**欠測として除外**
            continue
        n_pass = sum(1 for j in judged if j["advanced"])
        p5_rates[name] = dict(n_denominator=len(judged), n_pass=n_pass,
                              rate=n_pass / len(judged),
                              # 報告時要件: この n で判別できる差の大きさ（割合の刻み）
                              resolution=1.0 / len(judged))
    across = {k: _median([v[k] for v in per_seed_medians.values()]) for k in keys}
    # 🔴 exp_022 の P2・P3（**中央値では裾が見えないので 120 走行の件数で数える**）。
    # 定義: P2 = (D0 - エピソード中の最小 D) >= 5 の走行数／P3 = ゴールした走行数。
    n_runs = sum(len(d["metrics"]) for d in per_seed.values())
    n_reach_ge5 = sum(1 for d in per_seed.values() for m in d["metrics"]
                      if (m["d0"] - m["min_d"]) >= 5)
    n_goal_rollout = sum(1 for d in per_seed.values() for m in d["metrics"]
                         if m["outcome"] == "goal")
    reach_hist = {}
    for d in per_seed.values():
        for m in d["metrics"]:
            k = int(m["d0"] - m["min_d"])
            reach_hist[k] = reach_hist.get(k, 0) + 1
    if p5_rates:
        p5 = dict(median_rate=statistics.median([v["rate"] for v in p5_rates.values()]),
                  threshold=0.50, n_seeds_used=len(p5_rates),
                  per_seed=p5_rates, excluded_seeds=p5_excluded)
        p5["verdict"] = "P5_holds" if p5["median_rate"] >= 0.50 else "P5_fails"
    else:
        p5 = dict(verdict="undecidable_no_respawn", median_rate=None,
                  per_seed=p5_rates, excluded_seeds=p5_excluded,
                  note="6 seed すべて分母 0 ＝ 前提事象（リスポーン）が不発生。対象消滅。")
    return dict(across_seeds_median=across, per_seed_median=per_seed_medians, p5=p5,
                n_runs=n_runs, n_reach_ge5=n_reach_ge5, n_goal_rollout=n_goal_rollout,
                reach_hist={str(k): reach_hist[k] for k in sorted(reach_hist)},
                note=("集約は迷路 20 本の中央値 → seed の中央値。プール集計はしない。"
                      "Q1 = net_progress_per_1000（大きいほど良い）・"
                      "Q2 = respawn_per_1000（小さいほど良い）。"
                      "steps_per_entry と entries_per_distinct_cell は監視のみ。"))


def main() -> None:
    p = argparse.ArgumentParser(
        description="exp_021 の走行制御の中間量を測る（カード §4-1 の条文と 1 対 1）")
    p.add_argument("--models", nargs="+", required=True, help="方策の .zip（6 seed 分）")
    p.add_argument("--maze-dir", type=str, default=VALIDATION_MAZE_DIR,
                   help="測定に使う迷路のディレクトリ（既定 = 検証用の 20 迷路 7000-7019）")
    p.add_argument("--history-lags", type=str, default=None,
                   help="観測履歴の遅れ（例 '1,2,4,8,16,32,64,128'）。"
                        "**対照群（履歴なしの方策）では渡さない**")
    p.add_argument("--recurrent", action="store_true",
                   help="再帰型方策（exp_023）。RecurrentPPO として読み込み、"
                        "**エピソードごとに隠れ状態を捨てる**")
    p.add_argument("--history-sham", action="store_true",
                   help="にせ履歴（exp_022）。遅れの位置に現在の観測を複製する。"
                        "**学習時と同じ設定にすること**")
    p.add_argument("--label", type=str, required=True,
                   help="この測定の呼び名（出力に記録する。例 exp_019_final / exp_021_final）")
    p.add_argument("--purpose", type=str, default="validate",
                   choices=["validate", "diagnose"],
                   help="seed 帯の安全弁に渡す用途（既定 validate = 検証用の 20 迷路）")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    lags = parse_lags(args.history_lags)
    maze_dir = Path(args.maze_dir)
    maze_seeds = sorted(int(np.load(q)["seed"]) for q in maze_dir.glob("maze6_*.npz"))
    if not maze_seeds:
        raise SystemExit(f"迷路が見つかりません: {maze_dir}")
    # 🔴 帯の明示 (a) と拒否の安全弁 (b)（R40 条件 4 の両方）
    print(describe_seeds(maze_seeds, "maze6"))
    assert_seeds_allowed(maze_seeds, "maze6", args.purpose)
    print(f"[measure] 呼び名 = {args.label} / 観測履歴の遅れ = {lags or '（なし）'}"
          f" / 迷路 {len(maze_seeds)} 本 / 方策 {len(args.models)} 本", flush=True)

    # 🔴 呼び名が一意であることを**先に**確かめる（衝突すると黙って上書きされ、
    # 「6 seed の中央値」が 1 本の値になる。空振りを構成上検出する）。
    names = [_model_name(Path(m)) for m in args.models]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise SystemExit(f"モデルの呼び名が衝突している（集計が壊れる）: {dup}")

    per_seed, model_info = {}, []
    for m in args.models:
        name = _model_name(Path(m))
        policy_fn, n_ts = _load_policy(Path(m), recurrent=args.recurrent)
        eps = [_run_episode(maze_dir, ms, policy_fn, lags, sham=args.history_sham)
               for ms in maze_seeds]
        mets = [_episode_metrics(e) for e in eps]
        p5 = [j for j in (_judge_p5(e) for e in eps) if j is not None]
        # 🔴 **毎歩の生データを出力に残す**（准教授 AUDIT_041 §3・2026-08-14 採択）。
        # `min_d` と `min_d_after_last_respawn` は、D(t) と毎歩の `respawned` が無いと
        # **独立に再計算できない**。とくに Q4 の窓（最後のリスポーン以降）は**裁定 R50 で
        # 実際に誤りが見つかった箇所**で、窓の取り方が 1 箇所ずれても値だけ見ていては
        # 気づけない。AUDIT_039 §3-1（生データが残らず検算できなかった件）と同じ構造。
        raw = [dict(maze_seed=e["maze_seed"], trial_seed=e["trial_seed"], d0=e["d0"],
                    outcome=e["outcome"], d_hist=e["d_hist"],
                    resp_hist=[int(r) for r in e["resp_hist"]]) for e in eps]
        per_seed[name] = dict(metrics=mets, p5=p5, raw=raw)
        # 🔴 実歩数を記録する（学習量が揃っているかを後から確認できるように）
        model_info.append(dict(name=name, path=str(m), num_timesteps=n_ts))
        print(f"[{name}] 正味の前進(中央値) "
              f"{_median([x['net_progress_per_1000'] for x in mets]):.3f} 区画/1000歩 / "
              f"リスポーン {_median([x['respawn_per_1000'] for x in mets]):.3f} 回/1000歩 / "
              f"P5 母集団 {len(p5)}/{len(eps)}", flush=True)

    summary = summarize(per_seed)
    out = dict(
        label=args.label,
        clause="experiments/exp_021_observation_history/card.md §4-1・§5",
        # 🔴 測定条件を出力に残す（verify_bit_identity.py で「どのファイルを照合したか」が
        # 記録されず別実験の結果と見分けがつかなかった件の教訓。R11 候補として登録済み）
        models=model_info, history_lags=list(lags), history_sham=bool(args.history_sham),
        recurrent=bool(args.recurrent),
        seed_rule=dict(fn="mouse.maze6_eval._trial_seed", base=_SEED_BASE,
                       trial_idx=_TRIAL_IDX),
        env_kwargs=_ENV_KWARGS, maze_dir=str(maze_dir), maze_seeds=maze_seeds,
        summary=summary, detail=per_seed,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary["across_seeds_median"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["p5"].get("median_rate"), ensure_ascii=False))
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
