#!/usr/bin/env python3
"""exp_021 の Q1〜Q5 を機械的に判定する（カード §5 の条文と 1 対 1 対応）。

Mechanically evaluate the pre-registered predictions Q1-Q5 for exp_021.

🔴 **判定は手で計算しない。**私の誤りの 4 件（#5〜#8）はすべて「結果を読むときの層」
（対照の取り方・群間比較の単位・条文の母集団・集約値の一致）で起きた。
**条文を先にコードへ落として、値を入れたら判定が出る形にする。**

## 条文（`card.md` §5）との対応

| 条文 | 実装 |
|---|---|
| **Q1** = `net_progress_per_1000` の 6 seed 中央値 ≥ 対照群の中央値 × 1.25 | `judge_q1()` |
| **Q2** = `respawn_per_1000` が対照の 0.80 倍以下 **かつ** 前進が対照の 0.90 倍以上 | `judge_q2()` |
| **Q3** = 最終評価（200 万歩・検証用の 20 迷路・決定的）の `goal_rate` の 6 seed 中央値 < 0.05 | `judge_q3()` |
| **Q4** = 立て直しの成立割合（R51 確定仕様）の 6 seed 中央値 ≥ 0.50 | `judge_q4()` |
| **Q5** = 打ち切り条文の発火が 6 seed すべて | `judge_q5()` |
| 集約は **迷路 20 本の中央値 → 6 seed の中央値**・プール集計はしない | `measure_driving.py` の `summarize()`（本script はその値を読むだけ） |
| **中央値が厳密に 0.05 の場合は Q3 は外れ**（境界を投入前に固定） | `judge_q3()` の `< 0.05` |
| Q4 の分母 0 の seed は欠測・全 seed 分母 0 なら判定不能 | `measure_driving.py` が `verdict` を出す。ここでは転記する |

## 使い方

```bash
.venv/bin/python experiments/exp_021_observation_history/judge.py \
    --control outputs/exp_021_driving_control_final.json \
    --treat   outputs/exp_021_driving_treat_final.json \
    --logs logs/exp_021_seed1 ... logs/exp_021_seed6 \
    --out outputs/exp_021_judgment.json
```
"""
import argparse
import json
import statistics
from pathlib import Path

#: 打ち切り条文の窓（カード §4-2。exp_019・exp_020 と同じ形）。
CUTOFF_POINTS = [100_000 * k for k in range(1, 11)]   # 10 万〜100 万歩の 10 点
CUTOFF_RATE = 0.05
#: 事前登録した係数（**投入前に固定・変更しない**）
Q1_FACTOR = 1.25          # 前進は対照の 1.25 倍以上
Q2_FACTOR = 0.80          # リスポーンは対照の 0.80 倍以下
Q2_GUARD_FACTOR = 0.90    # かつ前進が対照の 0.90 倍を下回らない
Q3_GOAL_RATE = 0.05
Q4_RATE = 0.50
Q5_MIN_FIRED = 6


def _medians(summary: dict, key: str) -> tuple:
    """seed ごとの中央値の一覧と、その 6 seed 中央値。"""
    vals = [v[key] for v in summary["per_seed_median"].values()]
    return sorted(vals), statistics.median(vals)


def judge_q1(c: dict, t: dict) -> dict:
    cv, cm = _medians(c["summary"], "net_progress_per_1000")
    tv, tm = _medians(t["summary"], "net_progress_per_1000")
    thr = cm * Q1_FACTOR
    return dict(name="Q1 正味の前進", control_per_seed=cv, control_median=cm,
                treat_per_seed=tv, treat_median=tm, threshold=thr,
                ratio=(tm / cm if cm else None),
                hit=bool(tm >= thr),
                clause=f"介入群の中央値 >= 対照群の中央値 × {Q1_FACTOR}")


def judge_q2(c: dict, t: dict) -> dict:
    cv, cm = _medians(c["summary"], "respawn_per_1000")
    tv, tm = _medians(t["summary"], "respawn_per_1000")
    _, cnp = _medians(c["summary"], "net_progress_per_1000")
    _, tnp = _medians(t["summary"], "net_progress_per_1000")
    thr, guard = cm * Q2_FACTOR, cnp * Q2_GUARD_FACTOR
    cond_a, cond_b = bool(tm <= thr), bool(tnp >= guard)
    return dict(name="Q2 衝突の頻度", control_per_seed=cv, control_median=cm,
                treat_per_seed=tv, treat_median=tm, threshold=thr,
                ratio=(tm / cm if cm else None),
                guard_threshold=guard, treat_net_progress=tnp,
                cond_respawn_ok=cond_a, cond_progress_not_degraded=cond_b,
                hit=bool(cond_a and cond_b),
                clause=(f"リスポーンが対照の {Q2_FACTOR} 倍以下 かつ "
                        f"前進が対照の {Q2_GUARD_FACTOR} 倍を下回らない"))


def _final_goal_rates(log_dirs) -> dict:
    out = {}
    for d in log_dirs:
        h = json.loads((Path(d) / "validation_history.json").read_text(encoding="utf-8"))
        last = max(h, key=lambda r: r["total_timesteps"])
        out[Path(d).name] = dict(total_timesteps=last["total_timesteps"],
                                 goal_rate=last["goal_rate"])
    return out


def judge_q3(log_dirs) -> dict:
    fr = _final_goal_rates(log_dirs)
    vals = sorted(v["goal_rate"] for v in fr.values())
    med = statistics.median(vals)
    return dict(name="Q3 ゴール率は床のまま", per_seed=fr, per_seed_sorted=vals,
                median=med, threshold=Q3_GOAL_RATE,
                hit=bool(med < Q3_GOAL_RATE),
                clause=(f"最終評価の goal_rate の 6 seed 中央値 < {Q3_GOAL_RATE}"
                        "（**厳密に 0.05 なら外れ**。境界は投入前に固定）"))


def judge_q4(t: dict) -> dict:
    p5 = t["summary"]["p5"]
    med = p5.get("median_rate")
    return dict(name="Q4 立て直しの維持", verdict=p5.get("verdict"),
                per_seed={k: v["rate"] for k, v in p5.get("per_seed", {}).items()},
                n_denominator={k: v["n_denominator"] for k, v in p5.get("per_seed", {}).items()},
                excluded_seeds=p5.get("excluded_seeds", []),
                median=med, threshold=Q4_RATE,
                hit=(None if med is None else bool(med >= Q4_RATE)),
                clause=f"立て直しの成立割合の 6 seed 中央値 >= {Q4_RATE}")


def judge_q5(log_dirs) -> dict:
    per_seed, n_fired = {}, 0
    for d in log_dirs:
        h = json.loads((Path(d) / "validation_history.json").read_text(encoding="utf-8"))
        by_step = {r["total_timesteps"]: r["goal_rate"] for r in h}
        pts = {s: by_step.get(s) for s in CUTOFF_POINTS}
        missing = [s for s, v in pts.items() if v is None]
        fired = (not missing) and all(v < CUTOFF_RATE for v in pts.values())
        n_fired += int(fired)
        per_seed[Path(d).name] = dict(points=pts, missing=missing, fired=bool(fired),
                                      max_rate=(max(v for v in pts.values() if v is not None)
                                                if any(v is not None for v in pts.values())
                                                else None))
    return dict(name="Q5 打ち切り条文の発火", per_seed=per_seed, n_fired=n_fired,
                threshold=Q5_MIN_FIRED, hit=bool(n_fired >= Q5_MIN_FIRED),
                clause=(f"10 万〜100 万歩の 10 点すべてでゴール率 < {CUTOFF_RATE} を満たす seed が "
                        f"{Q5_MIN_FIRED} 本（5 本以下なら外れ）"))


def main() -> None:
    p = argparse.ArgumentParser(description="exp_021 の Q1〜Q5 の機械的判定")
    p.add_argument("--control", required=True, help="対照群の最終方策の測定出力")
    p.add_argument("--treat", required=True, help="介入群の最終方策の測定出力")
    p.add_argument("--logs", nargs="+", required=True, help="介入群の log ディレクトリ 6 本")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    c = json.loads(Path(args.control).read_text(encoding="utf-8"))
    t = json.loads(Path(args.treat).read_text(encoding="utf-8"))

    # 🔴 前提の検査（**判定の前に落とす**）
    n_c, n_t = len(c["summary"]["per_seed_median"]), len(t["summary"]["per_seed_median"])
    if n_c != 6 or n_t != 6:
        raise SystemExit(f"seed 数が 6 でない（対照 {n_c} / 介入 {n_t}）。集計が壊れている")
    steps_t = sorted({m["num_timesteps"] for m in t["models"]})
    steps_c = sorted({m["num_timesteps"] for m in c["models"]})
    if len(steps_t) != 1 or len(steps_c) != 1:
        raise SystemExit(f"学習量が seed 間で揃っていない（対照 {steps_c} / 介入 {steps_t}）")
    if steps_t != steps_c:
        raise SystemExit(f"介入群と対照群の学習量が違う（対照 {steps_c} / 介入 {steps_t}）")
    if not t["history_lags"]:
        raise SystemExit("介入群の測定に観測履歴の遅れが記録されていない（--history-lags 忘れ）")
    if c["history_lags"]:
        raise SystemExit("対照群の測定に観測履歴の遅れが入っている（対照は履歴なしのはず）")

    res = [judge_q1(c, t), judge_q2(c, t), judge_q3(args.logs), judge_q4(t), judge_q5(args.logs)]
    out = dict(clause="experiments/exp_021_observation_history/card.md §5",
               num_timesteps=steps_t[0], history_lags=t["history_lags"],
               control=str(args.control), treat=str(args.treat),
               results=res)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"学習量: 対照・介入とも {steps_t[0]:,} 歩 / 遅れ {t['history_lags']}")
    print("=" * 78)
    for r in res:
        mark = "❓" if r["hit"] is None else ("✅ 当たり" if r["hit"] else "❌ 外れ")
        print(f"{mark}  {r['name']}")
        print(f"      条文: {r['clause']}")
        for k in ("control_median", "treat_median", "threshold", "ratio", "median",
                  "n_fired", "guard_threshold", "treat_net_progress", "verdict"):
            if k in r and r[k] is not None:
                print(f"      {k} = {r[k]}")
    print("=" * 78)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
