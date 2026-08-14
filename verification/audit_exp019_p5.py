#!/usr/bin/env python3
"""監査: exp_019 の予測 P5 の独立判定（准教授セッション・**結果を見る前に確定させた形**）

判定する条文（裁定 R50 ＋ R51 の是正後の形。カード `card.md` §3-4）
--------------------------------------------------------------------
- **判定量** = 「**最後のリスポーン以降の**」 min_t D(t)（D = ゴールまでの区画距離）
- **判定**   = min_t D(t) ≤ D₀ − 1
- **母集団** = リスポーンを 1 回以上経験したエピソード
- **集計**   = seed ごとに割合を出し、**6 seed の中央値 ≥ 50%** で P5 成立
             （**プール集計はしない** — 規約 §9-18）
- **分母 0** の seed は欠測として除外し n を併記。6 seed すべて分母 0 なら**判定不能**
- **rollout** = 各 seed の最終方策・学習環境 v2（リスポーン有り・上限 2000 歩）・
             検証帯 7000-7019 の 20 面 × 各 1 エピソード・`deterministic=True`
- **初期擾乱** = 面ごとに `reset(seed=_trial_seed(base=0, maze_seed, trial_idx=0))` で固定
             （`mouse/maze6_eval.py` の既存関数を import。**再定義しない** — 条文の指定）

独立性の作り（**次の担当はここを壊さないこと**）
------------------------------------------------
- **D₀ は `assets/maze6/loop/validation/*.npz` の壁配列から自前の BFS** で出す
  （`shortest_distances()` は呼ばない）。**`info["d_start"]` は照合にしか使わない**
- **窓の切り出し・min・割合・中央値は、条文から自分で書いた**
- 学生B の `measure_p5.py` の出力は**突き合わせ先**であって、こちらの入力ではない

**判定の規律**
--------------
- **閾値 50%・母集団の定義・窓の取り方は、結果を見てから動かさない**
- **完走していない seed（最終モデルが無い）があれば、その seed は判定に使わない**
  （**歩数を明記して報告する**。打ち切りで停止した場合も同じ）

使い方: `.venv/bin/python verification/audit_exp019_p5.py`
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

from mouse.maze6_env import Maze6Env  # noqa: E402
from mouse.maze6_eval import _trial_seed  # noqa: E402  （条文の指定: import して使う）

VALID_DIR = "assets/maze6/loop/validation"
SEEDS = [1, 2, 3, 4, 5, 6]
LIMIT = 2000
GAMMA = 0.995
THRESHOLD = 0.50          # 事前確定・動かさない
MODEL_TMPL = "models/exp_019_v2_seed{n}.zip"
# 第 1 引数でモデルの雛形を差し替えられる（exp_020 の Q4 で流用するため）
if len(sys.argv) > 1:
    MODEL_TMPL = sys.argv[1]


# ------------------------------------------------------------ 自前 BFS
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


def rollout(model, maze_seed: int, d0_mine: int) -> dict:
    """1 面 1 エピソード。窓は `respawned` で切る（$D$ の値からは切らない）。"""
    env = Maze6Env(maze_dir=VALID_DIR, maze_seeds=[maze_seed], max_cache=2,
                   gamma=GAMMA, mode="fixed", maze_mode="loop",
                   goal_rule_containment=True, collision_respawn=True,
                   episode_limit_steps=LIMIT)
    obs, info = env.reset(seed=_trial_seed(0, maze_seed, 0))
    # 照合のみ（判定には自前 BFS の d0_mine を使う）
    d_start_env = int(info.get("d_start", -1))

    n_respawn = 0
    win_min = None          # 「最後のリスポーン以降」の min D(t)
    seen_respawn = False
    for _ in range(LIMIT):
        a = model.predict(obs, deterministic=True)[0]
        obs, _r, term, trunc, info = env.step(a)
        if bool(info.get("respawned", False)):
            n_respawn += 1
            seen_respawn = True
            win_min = None            # 窓をリセット（ここから測り直す）
        if seen_respawn:
            d = int(info["dist_to_goal"])
            win_min = d if win_min is None else min(win_min, d)
        if term or trunc:
            break

    qualifies = seen_respawn
    passed = bool(qualifies and win_min is not None and win_min <= d0_mine - 1)
    return dict(maze_seed=maze_seed, d0=d0_mine, d0_env=d_start_env,
                n_respawn=n_respawn, min_D_after_last_respawn=win_min,
                qualifies=qualifies, passed=passed)


def main() -> int:
    from stable_baselines3 import PPO

    faces = sorted(d0_of(p) for p in glob.glob(f"{VALID_DIR}/*.npz"))
    per_seed, missing = [], []
    for n in SEEDS:
        mp = Path(MODEL_TMPL.format(n=n))
        if not mp.exists():
            missing.append(n)
            continue
        model = PPO.load(str(mp), device="cpu")
        rows = [rollout(model, ms, d0) for ms, d0 in faces]
        denom = [r for r in rows if r["qualifies"]]
        num = [r for r in denom if r["passed"]]
        per_seed.append(dict(seed=n, n_denom=len(denom), n_pass=len(num),
                             frac=(len(num) / len(denom)) if denom else None,
                             rows=rows))

    print("=" * 78)
    print("監査: exp_019 予測 P5 の独立判定（R50 の窓 ＋ R51 の測定経路）")
    print("=" * 78)
    if missing:
        print(f"  ⚠️ 最終モデルが無い seed: {missing} — **判定に使わない**")
    print(f"{'seed':>6}{'分母 n':>8}{'合格':>6}{'割合':>9}")
    fracs = []
    for s in per_seed:
        f = "n/a（分母 0）" if s["frac"] is None else f"{s['frac']:.3f}"
        print(f"{s['seed']:>6}{s['n_denom']:>8}{s['n_pass']:>6}{f:>9}")
        if s["frac"] is not None:
            fracs.append(s["frac"])

    print("-" * 78)
    if not fracs:
        verdict, med = "INCONCLUSIVE", None
        print("判定: **判定不能**（全 seed で分母 0 ＝ 前提事象が不発生）")
    else:
        med = statistics.median(fracs)
        verdict = "P5_HOLDS" if med >= THRESHOLD else "P5_FAILS"
        print(f"中央値 = {med:.3f}（有効 seed {len(fracs)} 本）"
              f"  → **P5 {'成立' if med >= THRESHOLD else '不成立（反証条件が発火）'}**")
        print(f"⚠️ 分母は seed あたり最大 20。**この n で判別できる差は粗い** — "
              f"「差が無い」ではなく「この n では判定できない」と書き分けること（§9-18）")
    print("=" * 78)

    tag = "exp020_q4" if "exp_020" in MODEL_TMPL else "exp019_p5"
    out = Path(__file__).resolve().parent / "out" / f"{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"verdict": verdict, "median": med,
                               "threshold": THRESHOLD, "missing_seeds": missing,
                               "per_seed": per_seed}, ensure_ascii=False, indent=2))
    print(f"出力: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
