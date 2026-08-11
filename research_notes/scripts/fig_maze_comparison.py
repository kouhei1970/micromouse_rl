#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""迷路は本当に治ったのか — 是正前 / 是正後 / 大会実迷路 を絵で比べる。

**伝えたいこと**: 是正で**経路長**は大会実迷路の水準になったが、**探索の遠回り**は
直っていない。それは 2 本の線の重なり具合で一目で分かる。

各パネルに重ねるもの:
  1. 迷路の壁・スタート・ゴール 2x2
  2. **真の最短経路**（太い実線）… 壁が完全に既知のときの最短経路
  3. **足立法の初回探索の実走経路**（細い線）… 未知の壁を「通れる」と仮定して
     走ったときに実際に通った区画列

  是正前   … 最短経路が**短い**
  是正後   … 最短経路は**長くなった**が、2 本の線が**ほぼ重なる**
  大会実迷路 … 最短経路が長く、2 本の線が**大きく食い違う**

**代表面の選び方（恣意的に選ばない）**:
  上段 = その帯で $D_{true}$ が**中央値に最も近い**面
  下段 = その帯で**経路比 R が中央値に最も近い**面
  （同点は maze_id の辞書順で若い方。帯ごとに独立に選ぶ）

物理シミュレーションは使わない（迷路の壁配列と区画単位の探索のみ）。

使い方:
    .venv/bin/python research_notes/scripts/fig_maze_comparison.py
出力:
    outputs/figures/maze_before_after_contest.png
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition.explore_cost import (  # noqa: E402
    detour_ratio, first_run_path, n_delta, shortest_cells, true_shortest,
    true_shortest_path)

CELL = 1.0
GOAL_CENTER = ((7, 7), (7, 8), (8, 7), (8, 8))

# 配色（research_notes/scripts/_video_l0_common.py と同じ検証済みパレット）
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
WALL = "#3b3a37"

BANDS = [
    ("是正前の評価迷路", "competition/mazes/eval_v2_short", "maze_*.npz", False),
    ("是正後の評価迷路（現行）", "competition/mazes/eval", "maze_*.npz", False),
    ("大会実迷路（本物）", "competition/reference_mazes/contest", "contest_*.npz", True),
]


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for cand in ["/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
                 "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"]:
        if os.path.exists(cand):
            font_manager.fontManager.addfont(cand)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
            break
    return plt


def load(path, is_contest):
    z = np.load(path)
    v = np.array(z["v_walls"], dtype=int)
    h = np.array(z["h_walls"], dtype=int)
    if is_contest:
        start = (int(z["start_x"]), int(z["start_y"]))
        goals = tuple((int(a), int(b)) for a, b in zip(z["goals_x"], z["goals_y"]))
    else:
        start, goals = (0, 0), GOAL_CENTER
    return v, h, start, goals


def collect(band):
    """帯の全面について指標を測る（大会は中央ゴールの面のみ）。"""
    label, d, pat, is_contest = band
    rows = []
    for f in sorted(glob.glob(str(REPO_ROOT / d / pat))):
        v, h, s, g = load(f, is_contest)
        if is_contest and frozenset(g) != frozenset(GOAL_CENTER):
            continue          # 評価器と同じ中央 2x2 ゴールの面だけを使う
        D = true_shortest(v, h, s, g)
        R = detour_ratio(v, h, s, g)
        if D <= 0 or not np.isfinite(R):
            continue
        open_edges = int((v[1:16, :] == 0).sum() + (h[:, 1:16] == 0).sum())
        rows.append(dict(path=f, name=Path(f).stem, v=v, h=h, start=s, goals=g,
                         D=D, R=R, beta=open_edges - 256 + 1,
                         N2=n_delta(v, h, (2,), s, g)["N2"] / D))
    return label, rows


def pick(rows, key):
    """key の中央値に最も近い面を選ぶ（同点は名前の辞書順で若い方）。"""
    med = float(np.median([r[key] for r in rows]))
    return min(rows, key=lambda r: (abs(r[key] - med), r["name"]))


def draw(ax, r, plt, title, subtitle):
    v, h, start, goals = r["v"], r["h"], r["start"], r["goals"]
    # --- ゴール 2x2 の塗り
    gx = [g[0] for g in goals]
    gy = [g[1] for g in goals]
    ax.add_patch(plt.Rectangle((min(gx) * CELL, min(gy) * CELL), 2 * CELL, 2 * CELL,
                               facecolor=C_YELLOW, alpha=0.22, edgecolor="none", zorder=0))
    # --- スタート区画の塗り
    ax.add_patch(plt.Rectangle((start[0] * CELL, start[1] * CELL), CELL, CELL,
                               facecolor=C_AQUA, alpha=0.28, edgecolor="none", zorder=0))
    # --- 「最短経路になりうる」区画の網掛け
    #     最短経路は複数あることが多い。1 本だけ描くと、探索が別の等長ルートを
    #     通っただけでも図の上では外れて見えるので、その範囲を薄く示す。
    for (cx, cy) in shortest_cells(v, h, start, goals):
        ax.add_patch(plt.Rectangle((cx * CELL, cy * CELL), CELL, CELL,
                                   facecolor=C_BLUE, alpha=0.13, edgecolor="none",
                                   zorder=1))
    # --- 壁
    for x in range(17):
        for y in range(16):
            if v[x, y]:
                ax.plot([x * CELL, x * CELL], [y * CELL, (y + 1) * CELL],
                        color=WALL, lw=1.6, solid_capstyle="round", zorder=3)
    for x in range(16):
        for y in range(17):
            if h[x, y]:
                ax.plot([x * CELL, (x + 1) * CELL], [y * CELL, y * CELL],
                        color=WALL, lw=1.6, solid_capstyle="round", zorder=3)
    # --- 探索の実走経路（細い線）: 同じ区画を何度も通るので薄く重ねる
    ex = first_run_path(v, h, start, goals)
    if ex:
        xs = [(c[0] + 0.5) * CELL for c in ex]
        ys = [(c[1] + 0.5) * CELL for c in ex]
        ax.plot(xs, ys, color=C_ORANGE, lw=1.6, alpha=0.85, zorder=4,
                solid_capstyle="round", solid_joinstyle="round")
    # --- 真の最短経路（太い線）
    sp = true_shortest_path(v, h, start, goals)
    if sp:
        xs = [(c[0] + 0.5) * CELL for c in sp]
        ys = [(c[1] + 0.5) * CELL for c in sp]
        ax.plot(xs, ys, color=C_BLUE, lw=3.4, alpha=0.55, zorder=2,
                solid_capstyle="round", solid_joinstyle="round")
    ax.set_xlim(-0.3, 16.3)
    ax.set_ylim(-1.9, 16.3)          # 下に注記を書く余白
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10, color=TEXT_PRIMARY, pad=4)
    ax.text(8.0, -0.9, subtitle, ha="center", va="top", fontsize=9.5,
            color=TEXT_SECONDARY)


def main():
    plt = _setup_mpl()
    bands = [collect(b) for b in BANDS]

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 11.2))
    for col, (label, rows) in enumerate(bands):
        for row, (key, what) in enumerate((("D", "最短距離が中央値の面"),
                                           ("R", "経路比が中央値の面"))):
            r = pick(rows, key)
            sub = (f"$D_{{true}}$ = {r['D']} 区画　／　経路比 = {r['R']:.3f}"
                   f"　／　β = {r['beta']}")
            draw(axes[row][col], r, plt, f"{r['name']}（{what}）", sub)

    fig.tight_layout(rect=(0, 0.135, 1, 0.885))
    fig.subplots_adjust(hspace=0.30)   # 上段の注記と下段のタイトルが重ならないように

    # 列見出し（帯の名前）— 各列の中央に、上段パネルの上へ
    for col, (label, rows) in enumerate(bands):
        box = axes[0][col].get_position()
        fig.text(box.x0 + box.width / 2, 0.905, label, ha="center", va="bottom",
                 fontsize=13, color=TEXT_PRIMARY, fontweight="bold")

    # 凡例（図全体に 1 つ）
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], color=C_BLUE, lw=3.4, alpha=0.55,
               label="真の最短経路（壁が全部分かっていれば通る道）"),
        Line2D([], [], color=C_ORANGE, lw=1.8,
               label="足立法の初回探索が実際に通った道"),
        Patch(facecolor=C_BLUE, alpha=0.13,
              label="最短経路になりうる区画（等長の別ルートを含む）"),
        Patch(facecolor=C_AQUA, alpha=0.28, label="スタート"),
        Patch(facecolor=C_YELLOW, alpha=0.22, label="ゴール 2x2"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, 0.080))

    # 帯ごとの中央値（キャプション。3 行に分けて切れないように）
    cells = []
    for label, rows in bands:
        cells.append((label, np.median([r["D"] for r in rows]),
                      np.median([r["R"] for r in rows]),
                      np.median([r["N2"] for r in rows]), len(rows)))
    fig.text(0.5, 0.062, "帯ごとの中央値（代表 2 面ではなく全面）", ha="center",
             va="top", fontsize=10, color=TEXT_PRIMARY)
    for i, (lab, d, rr, n2, n) in enumerate(cells):
        box = axes[1][i].get_position()
        fig.text(box.x0 + box.width / 2, 0.042,
                 f"{lab}（n={n}）\n$D_{{true}}$ {d:.0f} 区画　/　経路比 {rr:.3f}"
                 f"　/　$N_2/D_0$ {n2:.2f}",
                 ha="center", va="top", fontsize=9.5, color=TEXT_SECONDARY)
    fig.suptitle("経路長は直った。探索の遠回りは直っていない。\n"
                 "— 2 本の線が重なるほど「探索しても何も分からない迷路」",
                 fontsize=14, color=TEXT_PRIMARY, y=0.985)

    out = REPO_ROOT / "outputs" / "figures" / "maze_before_after_contest.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"図: {out}")
    for lab, d, rr, n2, n in cells:
        print(f"  {lab}（n={n}）: D_true 中央値 {d:.0f} / 経路比 中央値 {rr:.3f} / N2/D0 {n2:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
