# research_notes/scripts/video_kinematics/clip_01_opening.py
# クリップ1: 導入 — 16x16の未知の迷路と、「何と比べるか」という問い（約40.1秒）
#
# 迷路は competition/mazes/design_turn_v1/maze_41001.npz の真の壁配置（読み取り専用の
# import）をそのまま描く。最短経路は classic/ideal.py の true_shortest_path
# （読み取り専用の import）で計算する。手打ちの経路は使わない。
# 凝りすぎないこと（コーディネータ指示）: 迷路の図と、同じ経路を違う速さで走る
# 2つの点を並べる程度に留める。
#
# 🔴 映像に年数を出さない（ユーザ指摘）。導入で数字を出すなら「16×16」など
# 競技の規格だけにする。
#
# 台本（narration/clip_01_opening.txt）の切れ目で画面の要素を増やす:
#   文1: 迷路が現れる（16x16・未知・探索して最短経路） /
#   文2: 「どこまで速いのか」という問い /
#   文3: 同じ経路を2つの点が違う速さで走る（比べ方の問題） /
#   文4: 一方を物理限界（分母）と呼ぶ /
#   文5: η = T_ideal / T_measured という物差しが姿を見せる
#
# 実行: .venv/bin/python research_notes/scripts/video_kinematics/clip_01_opening.py
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classic.geometry import wall_obstacles  # noqa: E402 読み取り専用の import
from classic.ideal import true_shortest_path  # noqa: E402 読み取り専用の import
from classic.maze_map import Direction  # noqa: E402 読み取り専用の import

import matplotlib.patches as mpatches  # noqa: E402
import _common as C  # noqa: E402

MAZE_PATH = REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "maze_41001.npz"
CELL = 0.180
START = (0, 0)
GOALS = [(7, 7), (7, 8), (8, 7), (8, 8)]


def polyline_lengths(pts_mm):
    """折れ線 `pts_mm`（[(x,y),...]）の累積弧長（先頭0）を返す。"""
    cum = [0.0]
    for k in range(1, len(pts_mm)):
        x0, y0 = pts_mm[k - 1]
        x1, y1 = pts_mm[k]
        cum.append(cum[-1] + math.hypot(x1 - x0, y1 - y0))
    return cum


def pos_at(pts_mm, cum, t: float):
    """折れ線上を弧長比率 `t`(0..1) だけ進んだ点を返す（線形補間）。"""
    t = min(max(t, 0.0), 1.0)
    target = t * cum[-1]
    for k in range(1, len(cum)):
        if target <= cum[k] or k == len(cum) - 1:
            span = max(cum[k] - cum[k - 1], 1e-9)
            frac = (target - cum[k - 1]) / span
            x0, y0 = pts_mm[k - 1]
            x1, y1 = pts_mm[k]
            return (x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac)
    return pts_mm[-1]


def main() -> None:
    C.setup_style()

    data = np.load(MAZE_PATH)
    v_walls, h_walls = data["v_walls"], data["h_walls"]
    obstacles = wall_obstacles(v_walls, h_walls)
    path_cells = true_shortest_path(v_walls, h_walls, START, GOALS, Direction.N)
    path_mm = [((x + 0.5) * CELL * 1000.0, (y + 0.5) * CELL * 1000.0) for x, y in path_cells]
    cum = polyline_lengths(path_mm)

    # ---- 台本の切れ目（narration/clip_01_opening.txt。更新済みの台本） ----
    s1_text = "マイクロマウスは、16かける16の未知の迷路を、自分で探索して、最短経路を走る競技です。"
    s2_text = ("この競技には、長いあいだ、答えの出しにくい問いがありました。"
               "いま自分のマウスは、どこまで速いのか。")
    s3_text = "速くなった、というのは比べ方の問題です。何と比べるかで答えが変わる。"
    s4_text = "この機体が物理的に出せる最速は、実は計算できます。それを分母に置けば、"
    s5_text = "速くなったではなく、限界の何割まで来たかで語れます。"

    # ナレーション実測 39.072s（ffprobe, 2026-08-20。台本改訂で再合成済み） + 余韻 1.0s。
    total_seconds = 40.072
    n_active = C.seconds_to_active_frames(total_seconds)
    b = C.stage_bounds(
        [len(s1_text), len(s2_text), len(s3_text), len(s4_text), len(s5_text)], n_active)
    # b = [0, 文1末, 文2末, 文3末, 文4末, n_active(文5末)]

    xs_mm = [(obs.cx - obs.hx) * 1000.0 for obs in obstacles] + \
            [(obs.cx + obs.hx) * 1000.0 for obs in obstacles]
    ys_mm = [(obs.cy - obs.hy) * 1000.0 for obs in obstacles] + \
            [(obs.cy + obs.hy) * 1000.0 for obs in obstacles]
    margin = 60.0
    xlim = (min(xs_mm) - margin, max(xs_mm) + margin)
    ylim = (min(ys_mm) - margin, max(ys_mm) + margin)

    fig = C.new_figure()
    ax = fig.add_axes([0.30, 0.10, 0.42, 0.80])
    C.style_axes(ax)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    title = C.add_title(fig, "16×16 の未知の迷路を、自分で探索して、最短経路を走る", fontsize=27)
    caption = C.add_caption(fig, "「速くなった」は、何と比べるかで答えが変わる。", y=0.045)
    caption.set_visible(False)

    question_note = fig.text(0.5, 0.90, "いま、自分のマウスはどこまで速いのか？",
                              ha="center", va="top", color=C.FG, fontsize=24,
                              fontweight="bold")
    question_note.set_visible(False)

    formula_note = fig.text(
        0.74, 0.55, "", ha="left", va="center", color=C.FG, fontsize=22,
        linespacing=1.8,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#1A1D1F",
                  edgecolor=C.GRID, linewidth=1.2))
    formula_note.set_visible(False)

    label_limit = fig.text(0.74, 0.68, "", ha="left", va="center", color=C.C_LIMIT,
                            fontsize=19)
    label_measured = fig.text(0.74, 0.42, "", ha="left", va="center", color=C.C_MEASURED,
                               fontsize=19)
    label_limit.set_visible(False)
    label_measured.set_visible(False)

    # 壁・柱
    for obs in obstacles:
        rect = mpatches.Rectangle(
            ((obs.cx - obs.hx) * 1000.0, (obs.cy - obs.hy) * 1000.0),
            2 * obs.hx * 1000.0, 2 * obs.hy * 1000.0,
            facecolor=C.GRID, edgecolor=C.FG, linewidth=0.6, zorder=2)
        ax.add_patch(rect)

    start_mm = path_mm[0]
    goal_mm = path_mm[-1]
    ax.plot([start_mm[0]], [start_mm[1]], marker="s", markersize=10, color=C.FG, zorder=5)
    ax.plot([goal_mm[0]], [goal_mm[1]], marker="*", markersize=18, color=C.FG, zorder=5)

    path_line, = ax.plot([], [], color=C.GRID, linewidth=2.0, alpha=0.9, zorder=3)
    dot_limit, = ax.plot([], [], marker="o", markersize=13, color=C.C_LIMIT, zorder=6)
    dot_measured, = ax.plot([], [], marker="o", markersize=13, color=C.C_MEASURED, zorder=6)
    dot_limit.set_visible(False)
    dot_measured.set_visible(False)

    def draw_frame(i: int):
        # ---- 文1: 迷路と経路が現れる（タイトルは通し） ----
        if i < b[2]:
            reveal = min(i / max(b[1], 1), 1.0) if i < b[1] else 1.0
            n_show = max(int(reveal * (len(path_mm) - 1)), 0) + 1
            path_line.set_data([p[0] for p in path_mm[:n_show]],
                                [p[1] for p in path_mm[:n_show]])
        else:
            path_line.set_data([p[0] for p in path_mm], [p[1] for p in path_mm])

        # ---- 文2: 問い ----
        question_note.set_visible(b[1] <= i < b[2])

        # ---- 文3+文4+文5: 2つの点が同じ経路を違う速さで走る ----
        if i >= b[2]:
            motion_span = max(n_active - b[2] - 1, 1)
            local_t = (i - b[2]) / motion_span
            # 「物理限界」は速く進み、序盤でゴールに達して待つ。「実測」は最後まで走り続ける。
            t_limit = min(local_t / 0.42, 1.0)
            t_measured = local_t
            p_limit = pos_at(path_mm, cum, t_limit)
            p_measured = pos_at(path_mm, cum, t_measured)
            dot_limit.set_data([p_limit[0]], [p_limit[1]])
            dot_measured.set_data([p_measured[0]], [p_measured[1]])
            dot_limit.set_visible(True)
            dot_measured.set_visible(True)
        else:
            dot_limit.set_visible(False)
            dot_measured.set_visible(False)

        # ---- 文4: ラベルが付く（「分母」の伏線） ----
        if i >= b[3]:
            label_limit.set_text("物理的に出せる最速\n（この先の動画の分母）")
            label_measured.set_text("いまの実測")
            label_limit.set_visible(True)
            label_measured.set_visible(True)
        else:
            label_limit.set_visible(False)
            label_measured.set_visible(False)

        # ---- 文5: 物差し η が姿を見せる ----
        if i >= b[4]:
            formula_note.set_text("η = T_ideal / T_measured")
            formula_note.set_visible(True)
            caption.set_visible(True)
        else:
            formula_note.set_visible(False)
            caption.set_visible(False)
        return ()

    out_path = C.OUT_DIR / "clip_01_opening.mp4"
    C.render_clip(fig, draw_frame, n_active, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
