#!/usr/bin/env python3
"""監査: ⑥ 手順 ②（リスポーンの実走 regime での $T$・$\\alpha$・訪問の再測定）

背景
----
`AUDIT_025` が検査した机上模型は、衝突シナリオの入力に
**exp_011（v1・リスポーン無し）の実測値** $T$=608 歩・$\\alpha$=0.133・訪問 +0.055 を使っていた。
**⑥ の手順 ② は、これを v2 の実走（リスポーン有り・上限 2000 歩）で取り直すことを求めている。**

測るもの（模型の入力と同じ名前で）
----------------------------------
- **$T$** … 1 つのリスポーン区間の長さ［歩］（区間の始まり → リスポーンの歩）
- **$\\alpha$** … 区間で稼いだ進捗率。$\\alpha = \\Phi_c / (D_0 \\cdot \\text{CELL})$
- **訪問** … 1 エピソードで回収した訪問ボーナスの**割引後**総和

**独立性の作り（`AUDIT_024` と同じ）**
--------------------------------------
**実装の $\\Phi$（`_potential*()`）は呼ばない。**報酬契約から整形の項を差し引きで復元し、

    F_t = γ·Φ_{t+1} − Φ_t   ⟹   Φ_{t+1} = (F_t + Φ_t) / γ

を **Φ_0 = Φ(start) = 0 から前進積分**して $\\Phi(t)$ を作る。
**$D_0$ は npz の壁配列から自前の BFS。**

**組み込みの自己検査**: リスポーンの歩の**直後**の $\\Phi$ は構成上 0 のはずである
（`_respawn_to_start()` が開始点へ戻してから整形を計算するため）。
**これが 0 にならなければ、復元が間違っている**ので、その場合は測定値を採用しない。

使い方: `.venv/bin/python verification/audit_exp019_regime.py`
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mouse.maze6_env import (  # noqa: E402
    _COLLISION_PENALTY, _GOAL_BONUS, _TIME_PENALTY, _VISIT_BONUS, Maze6Env,
)
from mouse.maze6_eval import _trial_seed  # noqa: E402

VALID_DIR = "assets/maze6/loop/validation"
SEEDS = [1, 2, 3, 4, 5, 6]
LIMIT, GAMMA, CELL = 2000, 0.995, 0.18
MODEL_TMPL = "models/exp_019_v2_seed{n}.zip"
PHI_RESET_TOL = 1e-9        # リスポーン直後の Φ が 0 であることの許容誤差

# 模型（AUDIT_025 が検査した式）が使っていた入力 — 突き合わせ先
MODEL_T, MODEL_ALPHA, MODEL_VISIT = 608.0, 0.133, 0.055


def d0_of(path: str) -> tuple[int, int]:
    d = np.load(path)
    v, h = d["v_walls"], d["h_walls"]
    w, hh = int(d["width"]), int(d["height"])
    start = tuple(int(x) for x in d["start"])
    goals = {tuple(int(x) for x in g) for g in d["goal_cells"]}

    def conn(x, y, nx, ny):
        if nx == x + 1:
            return v[x + 1, y] == 0
        if nx == x - 1:
            return v[x, y] == 0
        if ny == y + 1:
            return h[x, y + 1] == 0
        return h[x, y] == 0

    dist, q = {start: 0}, deque([start])
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < hh and (nx, ny) not in dist \
                    and conn(x, y, nx, ny):
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))
    return int(d["seed"]), min(dist[g] for g in goals if g in dist)


def rollout(model, maze_seed: int, d0: int) -> dict:
    env = Maze6Env(maze_dir=VALID_DIR, maze_seeds=[maze_seed], max_cache=2,
                   gamma=GAMMA, mode="fixed", maze_mode="loop",
                   goal_rule_containment=True, collision_respawn=True,
                   episode_limit_steps=LIMIT,
                   action_smooth_penalty=0.0, action_highpass_penalty=0.0)
    obs, info = env.reset(seed=_trial_seed(0, maze_seed, 0))
    prev_visited = int(info["n_visited"])

    phi = 0.0                 # Φ(start) = 0
    seg_start, visit_disc, visit_raw = 0, 0.0, 0.0
    segs, phi_reset_max = [], 0.0
    D0_m = d0 * CELL

    for t in range(LIMIT):
        a = model.predict(obs, deterministic=True)[0]
        obs, r, term, trunc, info = env.step(a)

        n_v = int(info["n_visited"])
        dv = n_v - prev_visited
        prev_visited = n_v
        visit_disc += (GAMMA ** t) * _VISIT_BONUS * dv
        visit_raw += _VISIT_BONUS * dv          # 割引なし（定義の違いを切り分けるため）

        # 整形の項を報酬契約から復元（実装の Φ は呼ばない）
        f = (float(r) + _TIME_PENALTY
             - (_GOAL_BONUS if bool(info["goal"]) else 0.0)
             - (_COLLISION_PENALTY if bool(info["collision"]) else 0.0)
             - _VISIT_BONUS * dv)
        phi_pre = phi                      # この歩に入る前の Φ（= 衝突点の Φ）
        phi = (f + phi) / GAMMA            # 前進積分

        if bool(info.get("respawned", False)):
            # 自己検査: リスポーン直後の Φ は構成上 0
            phi_reset_max = max(phi_reset_max, abs(phi))
            segs.append(dict(T=t - seg_start + 1, phi_c=phi_pre,
                             alpha=(phi_pre / D0_m) if D0_m > 0 else float("nan")))
            phi, seg_start = 0.0, t + 1    # 次の区間へ（Φ は 0 から）
        if term or trunc:
            break

    return dict(maze_seed=maze_seed, d0=d0, n_seg=len(segs),
                visit_discounted=visit_disc, visit_raw=visit_raw,
                phi_reset_max=phi_reset_max, segments=segs)


def main() -> int:
    from stable_baselines3 import PPO
    faces = sorted(d0_of(p) for p in glob.glob(f"{VALID_DIR}/*.npz"))
    per_seed, all_T, all_a, all_v, all_vr, reset_max = [], [], [], [], [], 0.0

    for n in SEEDS:
        mp = Path(MODEL_TMPL.format(n=n))
        if not mp.exists():
            print(f"⚠️ seed{n}: 最終モデルが無い — 判定に使わない")
            continue
        model = PPO.load(str(mp), device="cpu")
        rows = [rollout(model, ms, d0) for ms, d0 in faces]
        T = [s["T"] for r in rows for s in r["segments"]]
        A = [s["alpha"] for r in rows for s in r["segments"]]
        V = [r["visit_discounted"] for r in rows]
        VR = [r["visit_raw"] for r in rows]
        reset_max = max(reset_max, max(r["phi_reset_max"] for r in rows))
        all_T += T; all_a += A; all_v += V; all_vr += VR
        per_seed.append(dict(seed=n, n_seg=len(T),
                             T_med=statistics.median(T) if T else None,
                             alpha_med=statistics.median(A) if A else None,
                             visit_med=statistics.median(V),
                             visit_raw_med=statistics.median(VR), rows=rows))

    print("=" * 84)
    print("監査: ⑥ 手順 ② — リスポーンの実走 regime での T・α・訪問の再測定")
    print("=" * 84)
    print(f"  🔎 自己検査: リスポーン直後の |Φ| の最大 = {reset_max:.3e}"
          f"  → {'合格' if reset_max <= PHI_RESET_TOL else '🔴 不合格（復元が誤り）'}")
    if reset_max > PHI_RESET_TOL:
        print("  復元が誤っているので測定値を採用しない。")
        return 1
    print(f"{'seed':>5}{'区間数':>7}{'T 中央値':>10}{'α 中央値':>10}{'訪問 中央値':>12}")
    for s in per_seed:
        print(f"{s['seed']:>5}{s['n_seg']:>7}{s['T_med']:>10.1f}"
              f"{s['alpha_med']:>10.4f}{s['visit_med']:>12.4f}")
    print("-" * 84)
    Tm, Am, Vm = (statistics.median(all_T), statistics.median(all_a),
                  statistics.median(all_v))
    print(f"  全体（区間 {len(all_T)} 本・エピソード {len(all_v)} 本）:")
    print(f"    T      実測中央値 {Tm:8.1f} 歩   対 模型 {MODEL_T:6.1f}  "
          f"（模型/実測 = {MODEL_T/Tm:.2f} 倍）")
    q = statistics.quantiles(all_a, n=4)
    frac_le0 = sum(1 for x in all_a if x <= 0) / len(all_a)
    print(f"    α      実測中央値 {Am:8.4f}      対 模型 {MODEL_ALPHA:6.3f}  "
          f"（25%={q[0]:.4f} 75%={q[2]:.4f}・**α<=0 の区間が {frac_le0*100:.1f}%**）")
    Vrm = statistics.median(all_vr)
    print(f"    訪問（割引後） 実測中央値 {Vm:8.4f}  対 模型 {MODEL_VISIT:6.3f}  "
          f"（模型/実測 = {MODEL_VISIT/Vm:.2f} 倍）")
    print(f"    訪問（割引なし）実測中央値 {Vrm:8.4f}  ← 学生B の 0.08〜0.10 との突き合わせ用")
    print("=" * 84)

    p = Path(__file__).resolve().parent / "out" / "exp019_regime.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps({"phi_reset_max": reset_max,
                             "T_median": Tm, "alpha_median": Am,
                             "visit_median": Vm,
                             "visit_raw_median": statistics.median(all_vr),
                             "alpha_q1": q[0], "alpha_q3": q[2],
                             "alpha_frac_le0": frac_le0,
                             "all_T": all_T, "all_alpha": all_a,
                             "model_inputs": {"T": MODEL_T, "alpha": MODEL_ALPHA,
                                              "visit": MODEL_VISIT},
                             "per_seed": per_seed}, ensure_ascii=False, indent=2))
    print(f"出力: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
