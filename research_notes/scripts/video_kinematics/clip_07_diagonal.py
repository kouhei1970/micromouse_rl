# research_notes/scripts/video_kinematics/clip_07_diagonal.py
# クリップ7: 斜めは得か — 通路幅の比較とターン回数の削減（約15秒）
#
# 左パネル: 直線通路と斜め通路の通路幅（機体中心の横方向の余地）を
# classic/geometry.py（読み取り専用の import）で計算して比較する。
# 右パネル: 階段 k 段を直交で抜く場合と斜めで抜く場合のターン回数を、
# k=1,2,3,5 と変えながらアニメで見せる（ターン回数はその場で数える。手打ちしない）。
#
# 実行: .venv/bin/python research_notes/scripts/video_kinematics/clip_07_diagonal.py
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classic.geometry import (  # noqa: E402 読み取り専用の import
    HALF_WIDTH, Pose, PathSegment, sweep_clearance, wall_obstacles, poses_along,
)
from mouse.params import RobotParams  # noqa: E402 読み取り専用の import

import matplotlib.patches as mpatches  # noqa: E402
import _common as C  # noqa: E402

CELL = RobotParams().cell_size  # 0.18 m


def point_rect_distance(px, py, rect):
    dx = max(0.0, abs(px - rect.cx) - rect.hx)
    dy = max(0.0, abs(py - rect.cy) - rect.hy)
    return math.hypot(dx, dy)


def corridor_room(start: Pose):
    """柱だけの迷路（壁なし）を `start` の向きに貫く直線通路の
    片側の空き幅（点モデル）と機体中心の横方向の余地（矩形モデル）を計算する。

    `tests/test_geometry.py::test_diagonal_corridor_width` と同じ手法
    （柱の総当たりで最大化して確認済みの経路）を直線（0°）にも適用する。
    """
    width = height = 4
    v_walls = np.zeros((width + 1, height), dtype=np.int8)
    h_walls = np.zeros((width, height + 1), dtype=np.int8)
    obstacles = wall_obstacles(v_walls, h_walls)  # 柱だけ
    segments = [PathSegment(kind="straight", length=0.30, curvature=0.0)]
    poses = poses_along(segments, start, ds=0.001)
    channel = min(point_rect_distance(p.x, p.y, o) for p in poses for o in obstacles)
    room, _ = sweep_clearance(poses, obstacles)
    return channel, room


def main() -> None:
    C.setup_style()

    # ---- 左パネル: 通路幅の比較（classic/geometry.py で計算） ----
    # 直線（0°）: 区画中心線 y=CELL/2 を貫く。
    # 斜め（45°）: y = x + CELL/2 が最も広い斜め経路（柱の総当たりで確認済み、
    # test_diagonal_corridor_width のコメント参照）。
    straight_channel, straight_room = corridor_room(Pose(-0.03, CELL / 2.0, 0.0))
    diag_channel, diag_room = corridor_room(
        Pose(-0.03, -0.03 + CELL / 2.0, math.radians(45.0)))
    ratio_pct = diag_room / straight_room * 100.0

    # ---- 右パネル: 階段 k 段のターン回数（直交 vs 斜め） ----
    k_values = [1, 2, 3, 5]

    def turn_counts(k: int):
        n_turns_ortho = 2 * k - 1   # 経路上の中間頂点の数（始点・終点を除く全頂点）
        n_turns_diag = 2            # 斜め区間への出入りの2回のみ
        length_ortho = 2 * k        # 区画数（マンハッタン距離）
        length_diag = k * math.sqrt(2.0)  # 対角の区画数換算
        return n_turns_ortho, n_turns_diag, length_ortho, length_diag

    total_seconds = 15.0
    n_active = C.seconds_to_active_frames(total_seconds)
    n_cases = len(k_values)
    bounds = [round(n_active * i / n_cases) for i in range(n_cases + 1)]
    build_in_frames = min(45, n_active)  # 左パネルのバーが伸びる時間

    fig = C.new_figure()
    ax_l = fig.add_axes([0.055, 0.20, 0.28, 0.58])
    ax_r = fig.add_axes([0.42, 0.14, 0.56, 0.68])
    C.style_axes(ax_l)
    C.style_axes(ax_r)

    C.add_title(fig, "斜めは得か — 通路幅とターン回数", y=0.965)
    C.add_caption(fig, "経路が29%短いことより、ターンが消えることのほうが大きい。",
                  y=0.045, fontsize=20)

    # 左パネルの下地
    ax_l.set_xlim(-0.6, 1.6)
    ax_l.set_ylim(0.0, max(straight_room, diag_room) * 1000.0 * 1.25)
    ax_l.set_xticks([0.0, 1.0])
    ax_l.set_xticklabels(["直線通路", "斜め通路"], fontsize=15)
    ax_l.set_ylabel("機体中心の横方向の余地 W_C [mm]", fontsize=13)
    ax_l.set_title("通路幅の比較（classic/geometry.py）", color=C.FG, fontsize=16, pad=10)

    bar_straight = mpatches.Rectangle((-0.30, 0.0), 0.60, 0.0, facecolor=C.C_LIMIT,
                                       edgecolor=C.C_LIMIT, linewidth=1.5, alpha=0.75)
    bar_diag = mpatches.Rectangle((0.70, 0.0), 0.60, 0.0, facecolor=C.C_CAP_COST,
                                   edgecolor=C.C_CAP_COST, linewidth=1.5, alpha=0.75)
    ax_l.add_patch(bar_straight)
    ax_l.add_patch(bar_diag)
    label_straight = ax_l.text(0.0, 0.0, "", ha="center", va="bottom", color=C.C_LIMIT,
                                fontsize=15)
    label_diag = ax_l.text(1.0, 0.0, "", ha="center", va="bottom", color=C.C_CAP_COST,
                            fontsize=15)
    ratio_label = ax_l.text(0.5, max(straight_room, diag_room) * 1000.0 * 1.12,
                             "", ha="center", va="center", color=C.FG, fontsize=16)

    # 右パネルの下地（k ごとに set_xlim/ylim を更新）
    ax_r.set_aspect("equal")
    ax_r.set_xlabel("区画（マンハッタン方向）", fontsize=13)

    ortho_line, = ax_r.plot([], [], color=C.C_MEASURED, linewidth=3.0, zorder=3,
                             label="直交（階段）")
    ortho_turns, = ax_r.plot([], [], marker="o", markersize=10, color=C.C_MEASURED,
                              linestyle="none", zorder=4)
    diag_line, = ax_r.plot([], [], color=C.C_LIMIT, linewidth=3.0, zorder=3,
                            label="斜め")
    diag_turns, = ax_r.plot([], [], marker="o", markersize=10, color=C.C_LIMIT,
                             linestyle="none", zorder=4)
    legend = ax_r.legend(loc="lower right", fontsize=14, framealpha=0.85,
                          facecolor="#1A1D1F", edgecolor=C.GRID, labelcolor=C.FG)

    # 右パネル内（軸のローカル座標）に置く。図全体レベルの右上に置くと、幅の
    # 広い文言がクリップ幅いっぱいのタイトルと衝突するため、軸内の空白
    # （階段・斜め線が届かない左上）に固定する。
    k_title = ax_r.text(0.5, 1.10, "", transform=ax_r.transAxes, ha="center",
                         va="bottom", color=C.FG, fontsize=26, fontweight="bold")
    stat_panel = ax_r.text(0.03, 0.97, "", transform=ax_r.transAxes, ha="left",
                            va="top", color=C.FG, fontsize=17, linespacing=1.7,
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1A1D1F",
                                      edgecolor=C.GRID, linewidth=1.0))

    grid_lines = []

    def set_grid(k: int):
        for ln in grid_lines:
            ln.remove()
        grid_lines.clear()
        for x in range(k + 1):
            ln, = ax_r.plot([x, x], [0, k], color=C.GRID, linewidth=0.8, zorder=1)
            grid_lines.append(ln)
        for y in range(k + 1):
            ln, = ax_r.plot([0, k], [y, y], color=C.GRID, linewidth=0.8, zorder=1)
            grid_lines.append(ln)
        pad = 0.6
        ax_r.set_xlim(-pad, k + pad)
        ax_r.set_ylim(-pad, k + pad)

    def draw_frame(i: int):
        # 左パネル: バーの伸び（最初の build_in_frames だけ）
        grow = min(i / build_in_frames, 1.0) if build_in_frames > 0 else 1.0
        h_s = straight_room * 1000.0 * grow
        h_d = diag_room * 1000.0 * grow
        bar_straight.set_height(h_s)
        bar_diag.set_height(h_d)
        label_straight.set_position((0.0, h_s + 6))
        label_diag.set_position((1.0, h_d + 6))
        label_straight.set_text(f"{straight_room * 1000.0:.2f} mm" if grow >= 1.0 else "")
        label_diag.set_text(f"{diag_room * 1000.0:.2f} mm" if grow >= 1.0 else "")
        ratio_label.set_text(
            f"斜めは直線の {ratio_pct:.0f}%" if grow >= 1.0 else "")

        # 右パネル: 現在の k を求める
        case_idx = n_cases - 1
        for c in range(n_cases):
            if bounds[c] <= i < bounds[c + 1]:
                case_idx = c
                break
        k = k_values[case_idx]
        set_grid(k)

        n_to, n_td, len_o, len_d = turn_counts(k)

        # 直交の階段経路: (0,0)->(1,0)->(1,1)->(2,1)->...->(k,k)
        xs, ys = [0.0], [0.0]
        cx, cy = 0, 0
        for step in range(k):
            cx += 1
            xs.append(cx)
            ys.append(cy)
            cy += 1
            xs.append(cx)
            ys.append(cy)
        ortho_line.set_data(xs, ys)
        turn_xs = xs[1:-1]
        turn_ys = ys[1:-1]
        ortho_turns.set_data(turn_xs, turn_ys)

        # 斜め経路: (0,0)->(k,k) を直線で結ぶ。ターンは始点・終点の2回のみ。
        diag_line.set_data([0.0, k], [0.0, k])
        diag_turns.set_data([0.0, k], [0.0, k])

        k_title.set_text(f"階段 k = {k} 段")

        stat_panel.set_text(
            f"マンハッタン距離 {2 * k} 区画\n"
            f"直交: ターン {n_to} 回・経路長 {len_o} 区画\n"
            f"斜め: ターン {n_td} 回・経路長 {len_d:.2f} 区画\n"
            f"経路長の比 = {len_d / len_o * 100.0:.1f}%\n"
            f"ターン回数の比 = {n_td / n_to * 100.0:.1f}%"
        )
        return ()

    out_path = C.OUT_DIR / "clip_07_diagonal.mp4"
    C.render_clip(fig, draw_frame, n_active, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
