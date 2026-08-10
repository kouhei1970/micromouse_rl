"""
research_notes/scripts/check_random_policy_flips.py
===================================================
**乱数方策**の符号反転率の分布を測る（学習不要）。

背景（2026-08-11 教授指示）: exp_006 の出発点は「学習後の 75.6 回/s は学習前の乱数方策
53.3 回/s より悪い ＝ 学習すると振動が悪化する」だった。しかし**これは n=1 対 n=1 の
比較**である。基準線側（75.6）が n=1 だったことは既に判明した（k=0 の 3 seed で
28.1〜75.1 回/s とばらつく）。**対照側の 53.3 にも同じ疑いがかかる。**

もし乱数方策の反転が幅を持つなら、「学習すると振動が悪化する」という前提自体が
成り立たない可能性がある（k=0 seed 2 の 28.1 は乱数方策より**良い**かもしれない）。

**乱数方策の定義**: 学習前の PPO 方策と同じ、平均 0・標準偏差 σ のガウス分布から
毎ステップ独立に行動をサンプルする。SB3 の既定初期化では log_std=0 ＝ σ=1.0 だが、
行動空間が [-1, 1] にクリップされる点まで含めて同じ扱いにする。
σ は既定 1.0（初期化直後の値）。比較のため 0.83（10 万ステップ時点の実測値）も測る。

使い方:
    .venv/bin/python research_notes/scripts/check_random_policy_flips.py
    .venv/bin/python research_notes/scripts/check_random_policy_flips.py --n-seeds 10
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mouse.corridor_eval import DEFAULT_COURSE_DIR, evaluate_corridor  # noqa: E402

# 学習方策の実測（gate 帯・完走走行のみ。exp_006c、2026-08-11）を並べて比較する
LEARNED_REF = {"k=0 seed0": 75.1, "k=0 seed1": 58.2, "k=0 seed2": 28.1,
               "k=1e-4 seed1": 36.8, "k=1e-4 seed2": 63.0}
PREDECESSOR_RANDOM = 53.3   # 初代が記録した乱数方策の値（n=1）


def make_random_policy(rng: np.random.Generator, sigma: float):
    """毎ステップ独立にガウス雑音を出す方策（学習前の PPO 方策と同じ形）。"""
    def policy_fn(obs):
        a = rng.normal(0.0, sigma, size=2)
        return np.clip(a, -1.0, 1.0)
    return policy_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=6, help="乱数方策の seed 本数")
    ap.add_argument("--n-trials", type=int, default=1,
                    help="コースあたり試行数（既定 1。gate 帯 20 コース ×1）")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 0.83],
                    help="行動の標準偏差。既定 1.0（初期化直後）と 0.83（10 万時点の実測）")
    ap.add_argument("--out", type=str,
                    default="outputs/random_policy_flip_distribution.json")
    args = ap.parse_args()

    results = {}
    for sigma in args.sigmas:
        rows = []
        for s in range(args.n_seeds):
            rng = np.random.default_rng(1000 + s)
            summary = evaluate_corridor(
                make_random_policy(rng, sigma), course_dir=DEFAULT_COURSE_DIR,
                n_trials=args.n_trials, deterministic=False, seed=0, gamma=0.995,
                save_output=False, obs_dist_diff=True,
            )
            # 乱数方策は完走できないので、**全試行**での値を使うほかない。
            # 学習方策側（完走走行のみ）との比較には、この非対称を明記して扱う。
            flip = 0.5 * ((summary["sign_flip_rate_left_mean"] or 0.0)
                          + (summary["sign_flip_rate_right_mean"] or 0.0))
            rows.append(dict(seed=s, flip=flip,
                             completion=summary["no_contact_completion_rate"],
                             diff_rms=summary["action_diff_rms_mean"]))
            print(f"[random σ={sigma}] seed={s}: 反転 {flip:.1f} 回/s "
                  f"完走率 {summary['no_contact_completion_rate']:.2f}", flush=True)
        results[str(sigma)] = rows

    print("\n" + "=" * 78)
    print(f"乱数方策の符号反転率の分布（gate 帯 20 コース ×{args.n_trials} 試行／seed）")
    print("=" * 78)
    print(f"{'σ':>6}{'n':>4}{'平均':>9}{'最小':>9}{'最大':>9}{'標準偏差':>10}{'幅':>9}")
    for sigma, rows in results.items():
        fl = np.array([r["flip"] for r in rows])
        print(f"{sigma:>6}{len(fl):>4}{fl.mean():>9.1f}{fl.min():>9.1f}"
              f"{fl.max():>9.1f}{fl.std(ddof=1):>10.1f}{fl.max() - fl.min():>9.1f}")

    print(f"\n  初代が記録した乱数方策の値（n=1）: {PREDECESSOR_RANDOM} 回/s")
    print("\n  学習方策（gate 帯・完走走行のみ）:")
    for k, v in sorted(LEARNED_REF.items(), key=lambda kv: kv[1]):
        print(f"    {k:<16}{v:>8.1f} 回/s")

    print("\n  ⚠️ 比較の非対称: 乱数方策は完走できないので**全試行**の値。")
    print("     学習方策側は**完走走行のみ**の値。走行の質が違うので厳密な同列比較ではない。")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(results=results, learned_reference=LEARNED_REF,
                       predecessor_random_n1=PREDECESSOR_RANDOM,
                       n_trials_per_course=args.n_trials), f,
                  indent=2, ensure_ascii=False)
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
