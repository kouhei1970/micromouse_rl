"""
research_notes/scripts/check_seed_robustness.py
===============================================
exp_006c（k × 学習 seed の要因計画）の**最終モデル**を、検証帯と gate 帯の両方で
全指標を揃えて測る。

**測定の作法（2026-08-11 教授指摘）**:
- **異なる方策の指標を足し合わせない。**符号反転は「完走できた方策そのもの」で測る。
  完走できない方策の反転が小さくても、それは滑らかさの達成ではない
  （完走するには曲がる必要があり、曲がれば舵を切り、舵を切れば反転が増える）
- 結論は **x / N seed** の形で書く。条件間の比較は同じ seed 数で揃える

判定基準（3 つ同時。exp_006 と同じ）:
  符号反転 10 回/s 未満・完走率 90% 以上・1 区画 0.205 s 以内（exp_005 の 0.171 s から 20%）
横偏差（最大・RMS）も併記する（exp_005 の基準値は 最大 52.0 mm・RMS 14.5 mm、
接触水準 54 mm にほぼ達している）。

使い方:
    .venv/bin/python research_notes/scripts/check_seed_robustness.py            # 検証帯のみ
    .venv/bin/python research_notes/scripts/check_seed_robustness.py --gate     # gate 帯も
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO  # noqa: E402

from mouse.corridor_eval import (  # noqa: E402
    DEFAULT_COURSE_DIR, VALIDATION_COURSE_DIR, evaluate_corridor)

# (表示名, k, seed, モデルのパス)。既存 exp_006 の 4 本＋今回の seed 違い。
RUNS = [
    ("k=0",     0.0,  0, "models/exp_006_control_k0.zip"),
    ("k=0",     0.0,  1, "models/exp_006c_seed1.zip"),
    ("k=0",     0.0,  2, "models/exp_006c_seed2.zip"),
    ("k=1e-4",  1e-4, 0, "models/exp_006_smooth_k1e-4.zip"),
    ("k=1e-4",  1e-4, 1, "models/exp_006c_k1e-4_seed1.zip"),
    ("k=1e-4",  1e-4, 2, "models/exp_006c_k1e-4_seed2.zip"),
    ("k=1e-3",  1e-3, 0, "models/exp_006_smooth_k1e-3.zip"),
    ("k=1e-3",  1e-3, 1, "models/exp_006c_k1e-3_seed1.zip"),
    ("k=1e-3",  1e-3, 2, "models/exp_006c_k1e-3_seed2.zip"),
    ("k=1e-2",  1e-2, 0, "models/exp_006_smooth_k1e-2.zip"),
    # 案 3（highpass 版）。モデルが出来次第ここへ足す
    ("案3 k=2e-3", None, 1, "models/exp_006d_hp_k2e-3_seed1.zip"),
    ("案3 k=2e-3", None, 2, "models/exp_006d_hp_k2e-3_seed2.zip"),
]

# 3 基準（exp_006 と同じ）
CRIT_FLIP_PER_S = 10.0
CRIT_COMPLETION = 0.90
CRIT_SEC_PER_CELL = 0.205


def measure(model_path: Path, course_dir: str, n_trials: int):
    model = PPO.load(str(model_path), device="cpu")

    def policy_fn(obs):
        a, _ = model.predict(obs, deterministic=True)
        return a

    return evaluate_corridor(
        policy_fn, course_dir=course_dir, n_trials=n_trials,
        deterministic=True, seed=0, gamma=0.995,
        save_output=False, obs_dist_diff=True,
    )


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else None


def successful_trial_stats(s):
    """**壁接触なし完走した試行だけ**に限った指標を per_course から再集計する。

    2026-08-11 教授指摘: `evaluate_corridor` の要約は失敗走行を含む全試行の平均で、
    横偏差の最大値は「失敗走行＝定義上壁に当たっている」を含むためほぼ自明になる。
    符号反転も同様に、完走できなかった方策の値が混ざる。
    **滑らかさと横偏差は、完走した走行そのもので見る。**
    """
    ok = [t for pc in s["per_course"] for t in pc["trials"]
          if t["no_contact_complete"]]
    if not ok:
        return dict(n=0, flip=None, speed=None, sec_per_cell=None,
                    lat_max_mm=None, lat_rms_mm=None)
    return dict(
        n=len(ok),
        flip=_mean([0.5 * (t["sign_flip_rate_left"] + t["sign_flip_rate_right"])
                    for t in ok]),
        speed=_mean([t["mean_speed"] for t in ok]),
        sec_per_cell=_mean([t["sec_per_cell"] for t in ok
                            if t["sec_per_cell"] is not None]),
        lat_max_mm=max(t["lateral_max_m"] for t in ok) * 1000.0,
        lat_rms_mm=_mean([t["lateral_rms_m"] for t in ok]) * 1000.0,
    )


def row(label, seed, s):
    comp = s["no_contact_completion_rate"]
    suc = successful_trial_stats(s)
    # 3 基準は**完走した走行の指標**で判定する（別の方策・別の走行の指標を混ぜない）
    ok = (comp >= CRIT_COMPLETION and suc["flip"] is not None
          and suc["flip"] < CRIT_FLIP_PER_S
          and suc["sec_per_cell"] is not None
          and suc["sec_per_cell"] <= CRIT_SEC_PER_CELL)
    return dict(
        label=label, seed=seed, completion=comp,
        collision=s["collision_rate"], timeout=s["timeout_rate"],
        n_success=suc["n"],
        # 完走走行のみ（主。これを判定と報告に使う）
        speed=suc["speed"], sec_per_cell=suc["sec_per_cell"], flip=suc["flip"],
        lat_max_mm=suc["lat_max_mm"], lat_rms_mm=suc["lat_rms_mm"],
        # 全試行（従。過去の記録との照合用に残す）
        flip_all=0.5 * ((s["sign_flip_rate_left_mean"] or 0.0)
                        + (s["sign_flip_rate_right_mean"] or 0.0)),
        lat_max_mm_all=(s["lateral_max_m_max"] or 0.0) * 1000.0,
        lat_rms_mm_all=(s["lateral_rms_m_mean"] or 0.0) * 1000.0,
        all_criteria=ok,
    )


def print_table(title, rows, n_desc):
    print("\n" + "=" * 112)
    print(f"{title}（{n_desc}）")
    print("**速度・反転・横偏差はすべて『壁接触なし完走した走行のみ』の集計**")
    print("=" * 112)
    print(f"{'条件':<12}{'seed':>5}{'完走率':>8}{'衝突':>7}{'時切':>7}{'成功n':>7}"
          f"{'速度[m/s]':>11}{'s/区画':>9}{'反転[回/s]':>12}"
          f"{'横偏差最大[mm]':>15}{'RMS[mm]':>10}{'3基準':>7}")
    for r in rows:
        def f(key, fmt):
            return format(r[key], fmt) if r[key] is not None else "—"
        print(f"{r['label']:<12}{r['seed']:>5}{r['completion']:>8.2f}{r['collision']:>7.2f}"
              f"{r['timeout']:>7.2f}{r['n_success']:>7}"
              f"{f('speed', '.3f'):>11}{f('sec_per_cell', '.3f'):>9}"
              f"{f('flip', '.1f'):>12}{f('lat_max_mm', '.1f'):>15}"
              f"{f('lat_rms_mm', '.1f'):>10}"
              f"{'✅' if r['all_criteria'] else '—':>7}")
    print("\n  参考: 全試行（失敗を含む）での値 — 過去の記録との照合用")
    print(f"{'条件':<12}{'seed':>5}{'反転[回/s]':>12}{'横偏差最大[mm]':>16}{'RMS[mm]':>10}")
    for r in rows:
        print(f"{r['label']:<12}{r['seed']:>5}{r['flip_all']:>12.1f}"
              f"{r['lat_max_mm_all']:>16.1f}{r['lat_rms_mm_all']:>10.1f}")


def print_success_rate(rows):
    """P(成功 | 条件) を x / N seed の形で出す（2026-08-11 教授裁定）。"""
    print("\n" + "=" * 104)
    print("P(成功 | 条件) — 完走率 0.90 以上を「成功」とする")
    print("=" * 104)
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    print(f"{'条件':<12}{'成功 / 試した seed':>20}{'完走率（seed 別）':>34}"
          f"{'反転（成功した方策のみ）':>28}")
    for label, rs in by_label.items():
        ok = [r for r in rs if r["completion"] >= CRIT_COMPLETION]
        comps = " ".join(f"{r['completion']:.2f}" for r in rs)
        flips = " ".join(f"{r['flip']:.1f}" for r in ok) if ok else "（成功なし）"
        print(f"{label:<12}{f'{len(ok)} / {len(rs)}':>20}{comps:>34}{flips:>28}")
    print("\n  ※ 反転は**完走できた方策の、完走した走行のみ**の値。"
          "別の方策・別の走行の指標と混ぜない")
    for label, rs in by_label.items():
        ok = [r for r in rs if r["completion"] >= CRIT_COMPLETION]
        if len(ok) >= 2:
            fl = [r["flip"] for r in ok]
            print(f"  {label}: 反転の範囲 {min(fl):.1f}〜{max(fl):.1f} 回/s"
                  f"（成功 {len(ok)} seed）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true",
                    help="gate 帯（seed 3000-3019 × 5 試行）でも測る")
    ap.add_argument("--out", type=str,
                    default="outputs/exp_006c_seed_robustness.json")
    args = ap.parse_args()

    present = [(lb, k, sd, p) for lb, k, sd, p in RUNS if (REPO_ROOT / p).exists()]
    missing = [p for _lb, _k, _sd, p in RUNS if not (REPO_ROOT / p).exists()]
    if missing:
        print("[注意] まだ存在しないモデル（学習中・未実施）:")
        for p in missing:
            print(f"  - {p}")

    val_rows, gate_rows = [], []
    for label, _k, seed, path in present:
        print(f"[eval] {label} seed={seed} 検証帯 ...", flush=True)
        val_rows.append(row(label, seed, measure(REPO_ROOT / path,
                                                 VALIDATION_COURSE_DIR, 1)))
        if args.gate:
            print(f"[eval] {label} seed={seed} gate 帯 ...", flush=True)
            gate_rows.append(row(label, seed, measure(REPO_ROOT / path,
                                                      DEFAULT_COURSE_DIR, 5)))

    print_table("検証帯（seed 5000-5019）", val_rows, "20 コース ×1 試行 = 20 試行／行")
    if gate_rows:
        print_table("gate 帯（seed 3000-3019）", gate_rows,
                    "20 コース ×5 試行 = 100 試行／行")
    print_success_rate(gate_rows if gate_rows else val_rows)

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(validation=val_rows, gate=gate_rows,
                       criteria=dict(flip_per_s=CRIT_FLIP_PER_S,
                                     completion=CRIT_COMPLETION,
                                     sec_per_cell=CRIT_SEC_PER_CELL),
                       missing_models=missing), f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
