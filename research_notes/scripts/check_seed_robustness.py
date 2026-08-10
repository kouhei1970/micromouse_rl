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


def row(label, seed, s):
    flip = 0.5 * ((s["sign_flip_rate_left_mean"] or 0.0)
                  + (s["sign_flip_rate_right_mean"] or 0.0))
    spc = s["mean_sec_per_cell"]
    comp = s["no_contact_completion_rate"]
    # 3 基準の同時達成（速度は完走試行がないと測れないので None は未達扱い）
    ok = (flip < CRIT_FLIP_PER_S and comp >= CRIT_COMPLETION
          and spc is not None and spc <= CRIT_SEC_PER_CELL)
    return dict(
        label=label, seed=seed, completion=comp,
        collision=s["collision_rate"], timeout=s["timeout_rate"],
        speed=s["mean_forward_speed_mps"], sec_per_cell=spc, flip=flip,
        lat_max_mm=(s["lateral_max_m_max"] or 0.0) * 1000.0,
        lat_rms_mm=(s["lateral_rms_m_mean"] or 0.0) * 1000.0,
        all_criteria=ok,
    )


def print_table(title, rows, n_desc):
    print("\n" + "=" * 104)
    print(f"{title}（{n_desc}）")
    print("=" * 104)
    print(f"{'条件':<12}{'seed':>5}{'完走率':>9}{'衝突':>7}{'時間切れ':>9}"
          f"{'速度[m/s]':>11}{'s/区画':>9}{'反転[回/s]':>12}"
          f"{'横偏差最大[mm]':>15}{'RMS[mm]':>10}{'3基準':>7}")
    for r in rows:
        spc = f"{r['sec_per_cell']:.3f}" if r["sec_per_cell"] is not None else "—"
        spd = f"{r['speed']:.3f}" if r["speed"] is not None else "—"
        print(f"{r['label']:<12}{r['seed']:>5}{r['completion']:>9.2f}{r['collision']:>7.2f}"
              f"{r['timeout']:>9.2f}{spd:>11}{spc:>9}{r['flip']:>12.1f}"
              f"{r['lat_max_mm']:>15.1f}{r['lat_rms_mm']:>10.1f}"
              f"{'✅' if r['all_criteria'] else '—':>7}")


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
    print("\n  ※ 反転は**完走できた方策そのもの**の値のみを並べている。"
          "完走できない方策の反転と混ぜない")


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
