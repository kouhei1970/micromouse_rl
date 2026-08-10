# research_notes/scripts/fig_learning_curves.py
# M1 の学習曲線を 1 枚に重ねる図（2026-08-10、学生B）。
#
# 縦軸は **検証帯（seed 5000-5019）の壁接触なし完走率**。ep_rew_mean は
# ポテンシャル整形のせいで性能を映さない（exp_003 は報酬 31.2 まで上昇しながら
# 完走率 0.00 だった）ため、学習の進み具合はこの指標だけで見る。
#
# 出力: outputs/figures/learning_curves.png
# 実行: .venv/bin/python research_notes/scripts/fig_learning_curves.py
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "outputs" / "figures" / "learning_curves.png"

# 日本語フォント（macOS 標準。_video_l0_common.py と同じものを使う）
for cand in ["/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
             "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break

# 検証済みカテゴリカル配色（dataviz スキルの既定パレット、slot 1-5 を固定順で使用）
# validate_palette.js: 全項目 PASS（隣接 CVD ΔE 9.1 / 通常視 ΔE 19.6）。
# コントラストの警告に対しては直接ラベルを併記して補う（relief rule）。
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d8d7d2"

# 系列定義（label, 検証履歴のパス, 色, 直接ラベルの y オフセット [pt]）
# exp_003(0.00) と exp_004(0.05) は右端でほぼ重なるので、上下へ離して置く。
SERIES = [
    ("exp_003 差分あり・並列 6", "logs/exp_003_sensor_history", C_ORANGE, -13),
    ("exp_003b 並列 1", "logs/exp_003b_single_env", C_BLUE, 0),
    ("exp_004 Φ オフセット", "logs/exp_004_potential_offset", C_YELLOW, +13),
    ("exp_005 ＋衝突罰", "logs/exp_005_collision_penalty", C_AQUA, 0),
]

# exp_002（観測に差分なし）は当時 5 万ステップごとの記録を取っていないため、
# 残存チェックポイントを事後に測った 2 点のみ（本図では点で示す）。
EXP002_POINTS = [(500_000, 0.35), (1_000_000, 0.75)]

GATE = 0.90


def load_curve(log_dir):
    p = REPO_ROOT / log_dir / "validation_history.json"
    h = json.load(open(p, encoding="utf-8"))
    xs = [r["total_timesteps"] for r in h]
    ys = [r["no_contact_completion_rate"] for r in h]
    return xs, ys


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.0, 6.0), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    # gate の水平線（参照線は控えめに）
    ax.axhline(GATE, color=TEXT_SECONDARY, lw=1.2, ls=(0, (5, 4)), zorder=1)
    ax.text(12_000, GATE + 0.015, "M1 gate 0.90", color=TEXT_SECONDARY,
            fontsize=10, va="bottom", ha="left")

    for label, log_dir, color, dy in SERIES:
        try:
            xs, ys = load_curve(log_dir)
        except FileNotFoundError:
            print(f"  （記録なしのため省略: {log_dir}）")
            continue
        ax.plot(xs, ys, color=color, lw=2.0, label=label, zorder=3,
                solid_capstyle="round")
        # 直接ラベル（線の右端に添える。凡例と併用してコントラスト警告を補う）
        ax.annotate(label, xy=(xs[-1], ys[-1]), xytext=(10, dy),
                    textcoords="offset points", color=color, fontsize=10,
                    va="center", fontweight="bold")

    # exp_002 は 2 点のみ（事後測定）
    xs2 = [p[0] for p in EXP002_POINTS]
    ys2 = [p[1] for p in EXP002_POINTS]
    ax.plot(xs2, ys2, color=C_MAGENTA, lw=2.0, ls=(0, (2, 2)), marker="o",
            markersize=7, label="exp_002 差分なし（事後測定 2 点）", zorder=2)
    ax.annotate("exp_002 差分なし（事後測定 2 点）", xy=(xs2[-1], ys2[-1]),
                xytext=(10, 0), textcoords="offset points", color=C_MAGENTA,
                fontsize=10, va="center", fontweight="bold")

    ax.set_xlim(0, 1_420_000)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xticks([0, 200_000, 400_000, 600_000, 800_000, 1_000_000])
    ax.set_xticklabels(["0", "20 万", "40 万", "60 万", "80 万", "100 万"])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("学習ステップ数", color=TEXT_SECONDARY, fontsize=11)
    ax.set_ylabel("検証帯 壁接触なし完走率", color=TEXT_SECONDARY, fontsize=11)
    ax.set_title("M1 廊下追従の学習曲線 — 報酬設計が結果を分けた",
                 color=TEXT_PRIMARY, fontsize=15, fontweight="bold", pad=14)

    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)

    # 凡例は図の下（軸の外）に横 1 行。線に重ならないようにする。
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.13),
                    ncol=3, frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(TEXT_SECONDARY)

    note = ("検証帯 = seed 5000-2019 の 20 本 ×1 試行。gate 判定（seed 3000-3019）とは別帯。\n"
            "ep_rew_mean は性能を映さないため使わない（exp_003 は報酬 31.2 まで上昇しながら完走率 0.00）。")
    note = note.replace("5000-2019", "5000-5019")
    fig.text(0.075, 0.012, note, color=TEXT_SECONDARY, fontsize=9, va="bottom")

    fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.30)
    fig.savefig(OUT_PATH, facecolor=fig.get_facecolor())
    print(f"[fig] 保存: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
