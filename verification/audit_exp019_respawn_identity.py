#!/usr/bin/env python3
"""監査: 環境 v2 の「リスポーン整形の恒等式」の独立判定（准教授セッション）

判定するもの
------------
`REVIEW_007` §4-1 で自前に導出した恒等式が、**実装で成り立つか**を直接検査する。

  Σ_{t=t0}^{t_r} γ^(t−t0) · F_t = 0        （F_t = 整形の項）

導出（実装を見ずに書いたもの。REVIEW_007 §4-1）:

  F_t = γ·Φ(s_{t+1}) − Φ(s_t)  なので
  Σ_{t=0}^{n-1} γ^t·F_t = Σ (γ^(t+1)·Φ_{t+1} − γ^t·Φ_t) = γ^n·Φ_n − Φ_0   （テレスコープ）

  区間の始まり（エピソード開始 or 直前のリスポーン直後）では Φ_0 = Φ(start) = 0、
  区間の終わり（リスポーンの歩）では**リスポーン後**の状態が start なので Φ_n = Φ(start) = 0。
  ⟹ **総和は厳密に 0**（＝ 稼いだ整形をポテンシャル自身が取り返す ＝ 没収と等価）。

**独立性の作り（ここを壊さないこと）**
---------------------------------------
`Φ` の実装（`_potential*()`）は**呼ばない**。整形の項は
**報酬契約（documented reward contract）から差し引きで復元する**:

  F_t = r_t + TIME_PENALTY − GOAL_BONUS·[goal] − COLLISION_PENALTY·[collision]
             − VISIT_BONUS·Δn_visited

つまり「実装の Φ と実装の Φ を突き合わせる」のではなく、
**報酬の実測値と、自分の導いた式**を突き合わせる。

**判定形（結果を見る前に確定させる。動かさない）**
--------------------------------------------------
- **合格**: リスポーンで終わる全区間で |総和| ≤ 1e-9
- **不合格**: 1 区間でも超えたら不合格。**閾値は後から緩めない**
- 区間が 0 個なら **判定不能**（前提事象が不発生）と記録する

使い方: `.venv/bin/python verification/audit_exp019_respawn_identity.py`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mouse.maze6_env import Maze6Env  # noqa: E402

# --- 報酬契約の定数（documented reward contract）。実装から読むが、値を assert する ---
from mouse.maze6_env import (  # noqa: E402
    _COLLISION_PENALTY,
    _GOAL_BONUS,
    _TIME_PENALTY,
    _VISIT_BONUS,
)

TOL = 1e-9                     # 判定の閾値（事前に確定・後から緩めない）
GAMMA = 0.995                  # train.py の既定
VALIDATION_DIR = "assets/maze6/loop/validation"
MAZE_SEEDS = [7000, 7001, 7002]   # 検証帯。**学習には使わない帯なので判定に使ってよい**
STEPS = 2000                   # 学習環境と同じエピソード上限
RNG_SEED = 20260814            # 乱数方策の種（再現用）


def _check_contract() -> None:
    """報酬契約の定数が、判定に使った前提と一致するか（合成値の腐り対策・作法 11）。"""
    assert _TIME_PENALTY == 0.001, _TIME_PENALTY
    assert _GOAL_BONUS == 1.0, _GOAL_BONUS
    assert _COLLISION_PENALTY == -1.0, _COLLISION_PENALTY
    assert _VISIT_BONUS == 0.02, _VISIT_BONUS


def run_one(maze_seed: int, rng: np.random.Generator) -> dict:
    """1 面を乱数方策で回し、区間ごとの恒等式の残差を返す。

    **方策は問わない**（恒等式は環境の性質であって方策の性質ではない）。
    乱数方策を使うのは、**衝突＝リスポーンを多く発生させるため**である。
    """
    env = Maze6Env(
        maze_dir=VALIDATION_DIR, maze_seeds=[maze_seed], max_cache=2,
        gamma=GAMMA, mode="fixed", maze_mode="loop",
        goal_rule_containment=True, collision_respawn=True,
        episode_limit_steps=STEPS,
        action_smooth_penalty=0.0, action_highpass_penalty=0.0,
    )
    # 平滑・高域の罰は 0 にしてある（入っていると差し引きの式に項が増える）
    assert env.action_smooth_penalty == 0.0
    assert env.action_highpass_penalty == 0.0
    assert env.visit_bonus == _VISIT_BONUS
    assert env.collision_penalty == _COLLISION_PENALTY

    obs, info = env.reset(seed=int(maze_seed))
    prev_visited = int(info["n_visited"])

    seg_sum = 0.0        # 区間の割引総和
    seg_k = 0            # 区間内の歩数（割引の指数）
    seg_start_step = 0
    residuals, segments = [], []
    n_steps = 0

    for t in range(STEPS):
        a = rng.uniform(-1.0, 1.0, size=2)
        obs, r, term, trunc, info = env.step(a)
        n_steps += 1

        n_visited = int(info["n_visited"])
        d_visit = n_visited - prev_visited
        prev_visited = n_visited

        # --- 整形の項を報酬契約から復元する（実装の Φ は呼ばない）---
        f = (float(r)
             + _TIME_PENALTY
             - (_GOAL_BONUS if bool(info["goal"]) else 0.0)
             - (_COLLISION_PENALTY if bool(info["collision"]) else 0.0)
             - _VISIT_BONUS * d_visit)

        seg_sum += (GAMMA ** seg_k) * f
        seg_k += 1

        if bool(info.get("respawned", False)):
            residuals.append(abs(seg_sum))
            segments.append({"maze_seed": maze_seed,
                             "start_step": seg_start_step, "end_step": t,
                             "n_steps": seg_k, "residual": seg_sum})
            seg_sum, seg_k, seg_start_step = 0.0, 0, t + 1

        if term or trunc:
            break

    return {"maze_seed": maze_seed, "n_steps": n_steps,
            "n_respawn": int(info.get("n_respawn", 0)),
            "n_segments": len(residuals),
            "max_residual": max(residuals) if residuals else None,
            "segments": segments}


def main() -> int:
    _check_contract()
    rng = np.random.default_rng(RNG_SEED)
    results = [run_one(ms, rng) for ms in MAZE_SEEDS]

    n_seg = sum(r["n_segments"] for r in results)
    resid = [s["residual"] for r in results for s in r["segments"]]
    max_abs = max(abs(x) for x in resid) if resid else None

    print("=" * 68)
    print("監査: 環境 v2 のリスポーン整形の恒等式（REVIEW_007 §4-1 の自前導出）")
    print("=" * 68)
    for r in results:
        mx = "n/a" if r["max_residual"] is None else f"{r['max_residual']:.3e}"
        print(f"  面 {r['maze_seed']}: {r['n_steps']:5d} 歩 / "
              f"リスポーン {r['n_respawn']:3d} 回 / 区間 {r['n_segments']:3d} / 最大残差 {mx}")

    print("-" * 68)
    if n_seg == 0:
        print("判定: **判定不能**（リスポーンで終わる区間が 0 個 = 前提事象が不発生）")
        verdict = "INCONCLUSIVE"
    elif max_abs <= TOL:
        print(f"判定: **合格**  区間 {n_seg} 本すべてで |総和| ≤ {TOL:.0e}"
              f"（最大 {max_abs:.3e}）")
        verdict = "PASS"
    else:
        print(f"判定: **不合格**  最大 |総和| = {max_abs:.3e} > {TOL:.0e}")
        verdict = "FAIL"
    print("=" * 68)

    out = Path(__file__).resolve().parent / "out" / "exp019_respawn_identity.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"verdict": verdict, "tol": TOL, "gamma": GAMMA,
         "n_segments": n_seg, "max_abs_residual": max_abs,
         "maze_seeds": MAZE_SEEDS, "rng_seed": RNG_SEED,
         "results": results}, ensure_ascii=False, indent=2))
    print(f"出力: {out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
