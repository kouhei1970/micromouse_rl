#!/usr/bin/env python3
"""監査 028: 「届くが定着しない」の離脱過程を 200 歩刻みで追う

**事前登録**: `verification/AUDIT_028_PREREG_goal_decay.md`（`339c752`・**実施前にコミット済み**）
**判定形は凍結してある。結果を見てから動かさない。**

段階
----
- `T0` … **前提の反証テスト**。`first_goal` の重みで検証帯を回し直し、
        記録された `goal_rate` = 0.05 が**厳密に再現するか**。
        **再現しなければ以降は実施しない**（「評価は重みの決定的な関数」という前提が偽）
- `T1` … 退避された全点を評価し、**どの 200 歩区間で失われたか**を区間で名指しする
- `T2` … 隣接点間のパラメータ距離。**対照（失わなかった区間）と並べて**のみ大小を述べる

使い方:
    .venv/bin/python verification/audit_exp019_decay.py T0
    .venv/bin/python verification/audit_exp019_decay.py T1
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mouse.maze6_eval import VALIDATION_MAZE_DIR, evaluate_maze6  # noqa: E402

GAMMA = 0.995
EVAL_KWARGS = {"goal_rule_containment": True}   # train.py の EVAL_ENV_FLAGS["v2"]
RECORDED = {3: {2000000: 0.05}, 4: {1900000: 0.05}}   # 記録された値（照合先）

# 退避された点（歩数の昇順）。**歩数はファイル名から取る**
CKPT_GLOB = "models/exp_019_v2_seed{n}_*.zip"


def checkpoints(seed: int) -> list[tuple[int, str, Path]]:
    out = []
    for p in sorted(Path("models").glob(f"exp_019_v2_seed{seed}_*.zip")):
        m = re.search(r"_(\d+)\.zip$", p.name)
        if not m:
            continue
        kind = ("first_goal" if "first_goal" in p.name
                else "fine" if "fine" in p.name else "eval")
        out.append((int(m.group(1)), kind, p))
    return sorted(out)


def evaluate(path: Path) -> dict:
    from stable_baselines3 import PPO
    model = PPO.load(str(path), device="cpu")
    s = evaluate_maze6(lambda o: model.predict(o, deterministic=True)[0],
                       maze_dir=VALIDATION_MAZE_DIR, n_trials=1, seed=0,
                       gamma=GAMMA, maze_mode="loop", keep_traces=False,
                       env_kwargs=EVAL_KWARGS)
    goal_faces = sorted(int(pm["maze_seed"]) for pm in s["per_maze"]
                        if int(pm["n_goal"]) > 0)
    return dict(goal_rate=float(s["goal_rate"]),
                goal_rate_center_rule=s.get("goal_rate_center_rule"),
                goal_faces=goal_faces,
                n_failed=len(s["failed_maze_seeds"]))


def flat_params(path: Path) -> np.ndarray:
    from stable_baselines3 import PPO
    sd = PPO.load(str(path), device="cpu").policy.state_dict()
    return np.concatenate([v.detach().cpu().numpy().ravel()
                           for k, v in sorted(sd.items())])


def stage_T0() -> int:
    print("=" * 78)
    print("T0: 前提の反証テスト — first_goal の重みで goal_rate が厳密に再現するか")
    print("=" * 78)
    ok = True
    res = {}
    for seed, rec in RECORDED.items():
        step, expected = next(iter(rec.items()))
        p = Path(f"models/exp_019_v2_seed{seed}_first_goal_{step}.zip")
        r = evaluate(p)
        good = abs(r["goal_rate"] - expected) < 1e-12
        ok &= good
        res[seed] = dict(step=step, expected=expected, **r, reproduced=good)
        print(f"  seed{seed} @ {step:,}: 記録 {expected:.2f} / 再実行 "
              f"{r['goal_rate']:.2f}  ゴール面 {r['goal_faces']}  "
              f"→ {'✅ 再現' if good else '🔴 再現せず'}")
    print("-" * 78)
    if ok:
        print("判定: **前提は成立**。評価は重みの決定的な関数であり、")
        print("      H2（評価の擾乱の引きで落ちた）は構成上排除される。→ T1 へ進む")
    else:
        print("判定: **🔴 前提が偽**。離脱の議論は行わず、非決定性の同定へ切り替える。")
    Path("verification/out").mkdir(exist_ok=True)
    Path("verification/out/exp019_decay_T0.json").write_text(
        json.dumps({"premise_holds": ok, "results": res}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


def stage_T1() -> int:
    t0 = json.loads(Path("verification/out/exp019_decay_T0.json").read_text())
    if not t0["premise_holds"]:
        print("🔴 T0 が不合格なので T1 は実施しない（事前登録の規律）")
        return 1
    out = {}
    for seed in (3, 4):
        print("=" * 78)
        print(f"T1/T2: seed{seed} — 退避点の系列")
        print("=" * 78)
        cks = checkpoints(seed)
        rows, prev_vec, prev_step = [], None, None
        print(f"{'歩':>10}{'種別':>12}{'goal_rate':>11}{'ゴール面':>14}"
              f"{'‖Δθ‖':>12}{'変化要素':>10}")
        for step, kind, p in cks:
            r = evaluate(p)
            vec = flat_params(p)
            d = n_ch = None
            if prev_vec is not None:
                diff = vec - prev_vec
                d = float(np.linalg.norm(diff))
                n_ch = int(np.count_nonzero(diff))
            rows.append(dict(step=step, kind=kind, **r, dist=d, n_changed=n_ch,
                             prev_step=prev_step))
            print(f"{step:>10,}{kind:>12}{r['goal_rate']:>11.2f}"
                  f"{str(r['goal_faces']):>14}"
                  f"{(f'{d:.4f}' if d is not None else '-'):>12}"
                  f"{(str(n_ch) if n_ch is not None else '-'):>10}")
            prev_vec, prev_step = vec, step
        out[seed] = rows
        # 区間の名指し
        lost = [(a["step"], b["step"]) for a, b in zip(rows, rows[1:])
                if a["goal_rate"] > 0 and b["goal_rate"] == 0]
        print(f"\n  失われた区間: {lost if lost else '**200 歩の解像度では捉えられない**'}")
        tot = [r["n_changed"] for r in rows[1:] if r["n_changed"] is not None]
        if tot:
            print(f"  参考: 全パラメータ数 = {len(prev_vec):,} / "
                  f"区間ごとの変化要素数 = {tot}")
    Path("verification/out/exp019_decay_T1.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    print("\n出力: verification/out/exp019_decay_T1.json")
    return 0


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "T0"
    raise SystemExit(stage_T0() if stage == "T0" else stage_T1())
