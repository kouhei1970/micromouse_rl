"""
research_notes/scripts/export_all_traces.py
===========================================
**全走行の時系列を書き出し、記録の忠実性を確認する**（2026-08-11、准教授の提案）。

## なぜ全走行か

盲検の独立実装は代表 3 走行を 0.04% 以内で再現したが、**検証されたのは全体の 3% で、
しかも非無作為**である（保存規約が「中央値・最大・失敗の 1 本目」を選ぶので、
**統計的代表性は最初から捨てられている**）。**集計値まで独立検証するには全走行が要る。**

## 何を確認するか

**検証されているのは「時系列 → 指標」の写像であって、「方策 → 時系列」ではない。**
つまり**記録が方策の出力そのものか**は、どの検証項目も担っていない。そこで:

- **(a) 全走行**: 保存した時系列の `n_steps` と `sim_time` が `metrics.json` と一致するか。
  **記録の取りこぼし・重複・時刻ずれ**が捕まる
- **(b) 10 走行**: 保存した `action` を**評価器へ再入力して軌跡を再生**し、
  `progress_m` と `n_cells` が一致するか。
  「**記録された指令で走らせると、記録された結果になるか**」の確認。
  初期姿勢に擾乱が入るので**同じ試行 seed で reset する**。
  **再現できなければ「再現できなかった」と報告する**（無理に合わせにいかない）

## 出力先

**リポジトリには入れない**（検証用の使い捨て）。`--out-dir` の既定はセッションの
一時領域。`latest/metrics.json` は従来の形のまま触らない。

使い方:
    .venv/bin/python research_notes/scripts/export_all_traces.py --out-dir /tmp/traces
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO  # noqa: E402

from mouse.corridor_env import CorridorEnv  # noqa: E402
from mouse.corridor_eval import (  # noqa: E402
    DEFAULT_COURSE_DIR, _run_one_trial, _trial_seed, file_sha256)

MODELS = [
    ("hp_k8.7e-3_seed1", "models/exp_006d_hp_k8.7e-3_seed1.zip"),
    ("hp_k8.7e-3_seed3", "models/exp_006d_hp_k8.7e-3_seed3.zip"),
    ("control_k0_seed0", "models/exp_006_control_k0.zip"),
]
N_TRIALS = 5


def replay(env, actions, tseed):
    """記録した行動列を再入力して軌跡を再生する。**同じ試行 seed で reset する。**"""
    env.reset(seed=tseed)
    info = {}
    for a in actions:
        _obs, _r, term, trunc, info = env.step(np.asarray(a, dtype=np.float64))
        if term or trunc:
            break
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None,
                    help="書き出し先（既定: セッションの一時領域。リポジトリには入れない）")
    ap.add_argument("--n-replay", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        "/tmp/claude-501/-Users-kouhei-tmp-github-micromouse-rl/"
        "98ddc680-a89a-48f3-8f4d-f7e1dbb15e16/scratchpad/traces_all")
    out_dir.mkdir(parents=True, exist_ok=True)

    course_seeds = sorted(int(np.load(p)["seed"])
                          for p in Path(DEFAULT_COURSE_DIR).glob("corridor_*.npz"))

    manifest, mismatches, all_runs = [], [], []
    for tag, path in MODELS:
        model = PPO.load(str(REPO_ROOT / path), device="cpu")

        def policy(obs, _m=model):
            a, _ = _m.predict(obs, deterministic=True)
            return a

        for cs in course_seeds:
            env = CorridorEnv(course_dir=DEFAULT_COURSE_DIR, course_seeds=[cs],
                              max_cache=2, gamma=0.995, obs_dist_diff=True)
            for t in range(N_TRIALS):
                tseed = _trial_seed(0, cs, t)
                r = _run_one_trial(env, policy, tseed, keep_trace=True)
                tr = r["trace"]
                # --- (a) 記録の長さと時刻が指標と一致するか ---------------
                n_rec = len(tr["action"])
                t_rec = tr["t"][-1] if tr["t"] else None
                if n_rec != r["n_steps"]:
                    mismatches.append(dict(kind="n_steps", tag=tag, course_seed=cs,
                                           trial=t, recorded=n_rec,
                                           metrics=r["n_steps"]))
                if t_rec is None or abs(t_rec - r["sim_time"]) > 1e-9:
                    mismatches.append(dict(kind="sim_time", tag=tag, course_seed=cs,
                                           trial=t, recorded=t_rec,
                                           metrics=r["sim_time"]))
                fn = out_dir / f"{tag}__c{cs}_t{t}.npz"
                np.savez_compressed(
                    fn, t=np.array(tr["t"], dtype=np.float32),
                    # **float64 で保存する。**float32 でも 7 桁は保つが、再生の忠実性は
                    # 完全精度でしか成立しないことを実測したため余裕を取る
                    action=np.array(tr["action"], dtype=np.float64),
                    wheel_omega=np.array(tr["wheel_omega"], dtype=np.float64),
                    pose=np.array(tr["pose"], dtype=np.float64),
                    outcome=r["outcome"], course_seed=cs, trial_index=t,
                    trial_seed=tseed, n_steps=r["n_steps"], sim_time=r["sim_time"],
                    progress_m=r["progress_m"], n_cells=r["n_cells"])
                rec = {k: v for k, v in r.items() if k != "trace"}
                rec.update(tag=tag, course_seed=cs, trial_index=t, trial_seed=tseed,
                           trace_file=fn.name)
                all_runs.append(rec)
                manifest.append(dict(tag=tag, course_seed=cs, trial=t, file=fn.name,
                                     n_steps=r["n_steps"], outcome=r["outcome"]))
            env.close()
        print(f"[export] {tag}: {len(course_seeds) * N_TRIALS} 走行", flush=True)

    # --- (b) 再生による忠実性の確認 --------------------------------------
    # 条件をまたいで散らす: 各モデルから、コース seed を等間隔に選んで試行 0 を使う
    picks = []
    for tag, _ in MODELS:
        idxs = np.linspace(0, len(course_seeds) - 1,
                           max(args.n_replay // len(MODELS), 1)).astype(int)
        picks += [(tag, course_seeds[i], 0) for i in idxs]
    picks = picks[:args.n_replay]

    print("\n" + "=" * 92)
    print(f"(b) 再生による忠実性の確認（{len(picks)} 走行。"
          f"各モデルからコース seed を等間隔に選び試行 0 を使う）")
    print("=" * 92)
    print(f"{'条件':<20}{'コース':>8}{'記録 progress':>15}{'再生 progress':>15}"
          f"{'記録 n_cells':>13}{'再生 n_cells':>13}{'判定':>8}")
    replay_bad = []
    for tag, cs, t in picks:
        d = np.load(out_dir / f"{tag}__c{cs}_t{t}.npz", allow_pickle=True)
        env = CorridorEnv(course_dir=DEFAULT_COURSE_DIR, course_seeds=[cs],
                          max_cache=1, gamma=0.995, obs_dist_diff=True)
        info = replay(env, d["action"], int(d["trial_seed"]))
        env.close()
        p_rec, p_rep = float(d["progress_m"]), float(info.get("progress_m", float("nan")))
        c_rec, c_rep = int(d["n_cells"]), int(info.get("n_cells", -1))
        ok = abs(p_rec - p_rep) < 1e-6 and c_rec == c_rep
        if not ok:
            replay_bad.append(dict(tag=tag, course_seed=cs, trial=t,
                                   progress_recorded=p_rec, progress_replayed=p_rep,
                                   n_cells_recorded=c_rec, n_cells_replayed=c_rep))
        print(f"{tag:<20}{cs:>8}{p_rec:>15.6f}{p_rep:>15.6f}"
              f"{c_rec:>13}{c_rep:>13}{'一致' if ok else '**不一致**':>8}")

    summary = dict(
        n_runs=len(all_runs), out_dir=str(out_dir),
        models=[dict(tag=t, path=p, sha256=file_sha256(REPO_ROOT / p)) for t, p in MODELS],
        n_trials_per_course=N_TRIALS, course_seeds=course_seeds,
        fidelity_a_mismatches=mismatches,
        fidelity_b_picks=[dict(tag=a, course_seed=b, trial=c) for a, b, c in picks],
        fidelity_b_mismatches=replay_bad,
        runs=all_runs,
    )
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    total_mb = sum(p.stat().st_size for p in out_dir.glob("*.npz")) / 1e6
    print("\n" + "=" * 92)
    print(f"  書き出し: {len(all_runs)} 走行 / {total_mb:.1f} MB → {out_dir}")
    print(f"  (a) n_steps・sim_time の不一致: "
          f"**{len(mismatches)} 件**{'' if mismatches else '（全走行で一致）'}")
    for m in mismatches[:20]:
        print(f"      {m}")
    print(f"  (b) 再生の不一致: **{len(replay_bad)} 件** / {len(picks)} 走行")
    for m in replay_bad:
        print(f"      {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
