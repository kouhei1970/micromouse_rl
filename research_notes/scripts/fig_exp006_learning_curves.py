"""
research_notes/scripts/fig_exp006_learning_curves.py
====================================================
タスク A: exp_006（行動差分への罰 −k‖Δa‖²）4 本の学習曲線を重ね描きする。

予測は `experiments/exp_006_action_smoothness/taskA_prediction.md` に**観測前に**登録済み。
本スクリプトは観測量を出すだけで、解釈は行わない。

描く 4 段（横軸はすべて総ステップ数）:
 1. 検証帯（seed 5000-5019、20 本 ×1 試行）の壁接触なし完走率
 2. 衝突率（同上）
 3. 時間切れ率（同上）
 4. 方策の標準偏差 train/std（progress.csv。探索の広さ）

使い方:
    .venv/bin/python research_notes/scripts/fig_exp006_learning_curves.py
"""
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 比較する 4 本（対照 k=0 と k>0 の 3 点）。追加分（タスク C の seed 違い）は
# EXTRA_RUNS で足せるようにしておく。
RUNS = [
    ("k=0（対照）",  "logs/exp_006_control_k0",   "#333333", "-"),
    ("k=1e-4",       "logs/exp_006_smooth_k1e-4", "#1f77b4", "-"),
    ("k=1e-3",       "logs/exp_006_smooth_k1e-3", "#ff7f0e", "-"),
    ("k=1e-2",       "logs/exp_006_smooth_k1e-2", "#d62728", "-"),
]
EXTRA_RUNS = [
    ("k=0 seed1", "logs/exp_006c_seed1", "#777777", "--"),
    ("k=0 seed2", "logs/exp_006c_seed2", "#aaaaaa", "--"),
]

OUT_FIG = REPO_ROOT / "outputs/figures/exp006_learning_curves.png"


def load_validation(log_dir: Path):
    """validation_history.json を読む。学習中の実行では途中まででも読めるようにする。"""
    path = log_dir / "validation_history.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        hist = json.load(f)
    if not hist:
        return None
    return dict(
        t=np.array([r["total_timesteps"] for r in hist], dtype=float),
        completion=np.array([r["no_contact_completion_rate"] for r in hist]),
        collision=np.array([r["collision_rate"] for r in hist]),
        timeout=np.array([r["timeout_rate"] for r in hist]),
        flip=np.array([0.5 * (r["sign_flip_rate_left_mean"]
                              + r["sign_flip_rate_right_mean"]) for r in hist]),
        speed=np.array([r["mean_forward_speed_mps"] for r in hist]),
    )


def load_std(log_dir: Path):
    """progress.csv から train/std の時系列を読む（PPO の状態非依存 log_std）。"""
    path = log_dir / "progress.csv"
    if not path.exists():
        return None
    t, std = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s, n = row.get("train/std", ""), row.get("time/total_timesteps", "")
            if s and n:
                std.append(float(s))
                t.append(float(n))
    if not t:
        return None
    return np.array(t), np.array(std)


def interp_at(t, y, t_query):
    """指定ステップ数での値（記録点が一致しない場合は線形補間）。"""
    if t_query < t[0] or t_query > t[-1]:
        return float("nan")
    return float(np.interp(t_query, t, y))


def main():
    runs = list(RUNS)
    for label, d, color, ls in EXTRA_RUNS:
        if (REPO_ROOT / d / "validation_history.json").exists():
            runs.append((label, d, color, ls))

    data = {}
    for label, d, color, ls in runs:
        log_dir = REPO_ROOT / d
        val = load_validation(log_dir)
        if val is None:
            print(f"[skip] {label}: {d} に validation_history.json がない")
            continue
        val["std"] = load_std(log_dir)
        data[label] = (val, color, ls)

    # ---- 数値表（判別基準の統計量） -------------------------------------
    print("=== 検証帯（seed 5000-5019、20 本 ×1 試行）の要約 ===")
    print(f"{'条件':<14}{'n点':>4}{'peak':>8}{'peak時':>10}{'final':>8}"
          f"{'cmin':>8}{'cmin時':>10}{'to_max':>8}")
    for label, (v, _, _) in data.items():
        i_peak = int(np.argmax(v["completion"]))
        i_cmin = int(np.argmin(v["collision"]))
        print(f"{label:<14}{len(v['t']):>4}{v['completion'][i_peak]:>8.2f}"
              f"{v['t'][i_peak]/1e4:>9.0f}万{v['completion'][-1]:>8.2f}"
              f"{v['collision'][i_cmin]:>8.2f}{v['t'][i_cmin]/1e4:>9.0f}万"
              f"{v['timeout'].max():>8.2f}")

    print("\n=== 完走率の時系列（5 万ステップ刻み） ===")
    header = "steps  " + "".join(f"{lb:>14}" for lb in data)
    print(header)
    all_t = sorted({int(x) for v, _, _ in data.values() for x in v["t"]})
    for tq in all_t:
        row = f"{tq/1e4:>5.0f}万"
        for label, (v, _, _) in data.items():
            j = np.where(v["t"] == tq)[0]
            row += f"{v['completion'][j[0]]:>14.2f}" if len(j) else f"{'-':>14}"
        print(row)

    print("\n=== 衝突率の時系列 ===")
    print(header)
    for tq in all_t:
        row = f"{tq/1e4:>5.0f}万"
        for label, (v, _, _) in data.items():
            j = np.where(v["t"] == tq)[0]
            row += f"{v['collision'][j[0]]:>14.2f}" if len(j) else f"{'-':>14}"
        print(row)

    print("\n=== 時間切れ率の時系列 ===")
    print(header)
    for tq in all_t:
        row = f"{tq/1e4:>5.0f}万"
        for label, (v, _, _) in data.items():
            j = np.where(v["t"] == tq)[0]
            row += f"{v['timeout'][j[0]]:>14.2f}" if len(j) else f"{'-':>14}"
        print(row)

    print("\n=== 方策の標準偏差 train/std（代表点で線形補間） ===")
    print(f"{'条件':<14}" + "".join(f"{f'{s//10000}万':>10}"
                                    for s in (100_000, 300_000, 500_000, 1_000_000)))
    for label, (v, _, _) in data.items():
        if v["std"] is None:
            print(f"{label:<14}{'（progress.csv なし）':>10}")
            continue
        ts, ss = v["std"]
        print(f"{label:<14}" + "".join(
            f"{interp_at(ts, ss, s):>10.3f}"
            for s in (100_000, 300_000, 500_000, 1_000_000)))

    # ---- 図 ------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    panels = [
        ("completion", "検証帯 完走率（壁接触なし）", (-0.03, 1.03)),
        ("collision", "衝突率", (-0.03, 1.03)),
        ("timeout", "時間切れ率", (-0.03, 1.03)),
    ]
    for ax, (key, title, ylim) in zip(axes, panels):
        for label, (v, color, ls) in data.items():
            ax.plot(v["t"] / 1e6, v[key], ls, color=color, marker="o",
                    markersize=3, label=label, linewidth=1.6)
        ax.set_ylabel(title)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9, ncol=2)

    ax = axes[3]
    for label, (v, color, ls) in data.items():
        if v["std"] is None:
            continue
        ts, ss = v["std"]
        ax.plot(ts / 1e6, ss, ls, color=color, label=label, linewidth=1.6)
    ax.set_ylabel("方策の標準偏差 train/std")
    ax.set_xlabel("総ステップ数 [×10⁶]")
    ax.grid(alpha=0.3)

    fig.suptitle("exp_006: 行動差分への罰 −k‖Δa‖² の学習曲線（検証帯 seed 5000-5019, n=20）",
                 fontsize=12)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140)
    print(f"\n[fig] saved: {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
