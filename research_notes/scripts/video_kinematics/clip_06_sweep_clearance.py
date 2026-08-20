# research_notes/scripts/video_kinematics/clip_06_sweep_clearance.py
# クリップ6: 通れるかどうかは幾何が決める — 90° ターンの掃引余裕（約15秒）
#
# classic/geometry.py（読み取り専用の import）で、迷路の壁・柱と機体外形（100x80mm）を
# 掃引したときの最小余裕を計算する。半径を 90 -> 190 -> 250mm と上げていくと余裕が
# 減り、通れなくなったところで赤くする。局所迷路の配置（西区画→角区画→北区画の
# 90°左ターン）は tests/test_geometry.py の検査2〜4・6・7 と同じもの
# （geometry_anchor.py・note_031 の値との照合済みの配置）を再構成する。
#
# 実行: .venv/bin/python research_notes/scripts/video_kinematics/clip_06_sweep_clearance.py
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classic.geometry import (  # noqa: E402 読み取り専用の import
    HALF_WIDTH, HALF_LENGTH, Pose, clearance, sweep_clearance, wall_obstacles,
    turn_path, poses_along, robot_corners, max_feasible_radius,
)

import matplotlib.patches as mpatches  # noqa: E402
import _common as C  # noqa: E402

CELL = 0.180
ENTRY_LINE_Y = 0.09
EXIT_LINE_X = 0.27


def turn_obstacles():
    """西区画(0,0) → 角区画(1,0) → 北区画(1,1) の 90° 左ターン局所迷路（壁配置）。

    `tests/test_geometry.py::_turn_obstacles` と同一の壁配置（値の出所も同じ）。
    """
    v_walls = np.zeros((3, 2), dtype=np.int8)
    h_walls = np.zeros((2, 3), dtype=np.int8)
    v_walls[2, 0] = 1  # 角区画の東壁（外側）
    v_walls[1, 1] = 1  # 北区画の西壁
    v_walls[2, 1] = 1  # 北区画の東壁
    h_walls[0, 0] = 1  # 西区画の南壁
    h_walls[1, 0] = 1  # 角区画の南壁（外側）
    h_walls[0, 1] = 1  # 西区画の北壁
    return wall_obstacles(v_walls, h_walls, center_goal=False)


def corner_pose(offset: float = 0.0) -> Pose:
    return Pose(EXIT_LINE_X + offset, ENTRY_LINE_Y - offset, 0.0)


def quarter_turn_poses(radius: float):
    delta = math.pi / 2
    corner = corner_pose(0.0)
    segments, consumed = turn_path(delta, radius)
    lead = segments[0].length
    back_off = consumed + lead
    sweep_start = Pose(corner.x - back_off, corner.y, corner.theta)
    return poses_along(segments, sweep_start)


def main() -> None:
    C.setup_style()
    obstacles = turn_obstacles()

    radii_mm = [90.0, 190.0, 250.0]
    phase_colors = [C.C_LIMIT, C.C_CAP_COST, C.C_MEASURED]

    r_star_mm = max_feasible_radius(math.pi / 2, obstacles, corner_pose(0.0),
                                     margin=0.0) * 1000.0

    cases = []
    for r_mm in radii_mm:
        poses = quarter_turn_poses(r_mm / 1000.0)
        clears = [clearance(p, obstacles) * 1000.0 for p in poses]  # mm
        min_clear = min(clears)
        worst_idx = clears.index(min_clear)
        cases.append(dict(r_mm=r_mm, poses=poses, clears=clears,
                           min_clear=min_clear, worst_idx=worst_idx))

    # 描画範囲（すべての半径の掃引と障害物を含む bbox に余白を付ける。mm 単位）
    xs, ys = [], []
    for obs in obstacles:
        xs += [obs.cx - obs.hx, obs.cx + obs.hx]
        ys += [obs.cy - obs.hy, obs.cy + obs.hy]
    for case in cases:
        for p in case["poses"]:
            for cx, cy in robot_corners(p):
                xs.append(cx)
                ys.append(cy)
    xs = [v * 1000.0 for v in xs]
    ys = [v * 1000.0 for v in ys]
    margin = 20.0
    xlim = (min(xs) - margin, max(xs) + margin)
    ylim = (min(ys) - margin, max(ys) + margin)

    total_seconds = 15.0
    n_active = C.seconds_to_active_frames(total_seconds)
    n_cases = len(cases)
    bounds = [round(n_active * k / n_cases) for k in range(n_cases + 1)]
    sweep_frac = 0.82
    n_env = 26  # 掃引の残像（機体外形のコマ送り）の最大表示数

    fig = C.new_figure()
    ax = fig.add_axes([0.06, 0.15, 0.62, 0.68])
    C.style_axes(ax)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]", fontsize=14)
    ax.set_ylabel("y [mm]", fontsize=14)

    C.add_title(fig, "通れるかどうかは幾何が決める — 90°ターンの掃引余裕", y=0.965)
    C.add_caption(
        fig,
        f"中心線上で通れる最大半径は {r_star_mm:.0f} mm。"
        "通路幅いっぱいの300 mmは機体の長さのぶん通れない。",
        y=0.045, fontsize=20,
    )

    # 壁・柱（迷路の障害物）。GRID 色の塗り＋文字色の縁取り。
    for obs in obstacles:
        rect = mpatches.Rectangle(
            ((obs.cx - obs.hx) * 1000.0, (obs.cy - obs.hy) * 1000.0),
            2 * obs.hx * 1000.0, 2 * obs.hy * 1000.0,
            facecolor=C.GRID, edgecolor=C.FG, linewidth=0.8, zorder=2)
        ax.add_patch(rect)

    def main_robot_patch(color):
        poly = mpatches.Polygon([(0, 0)] * 4, closed=True, facecolor=color,
                                 alpha=0.55, edgecolor=color, linewidth=2.2, zorder=5)
        ax.add_patch(poly)
        return poly

    main_patch = main_robot_patch(C.C_LIMIT)
    env_patches = [mpatches.Polygon([(0, 0)] * 4, closed=True, facecolor="none",
                                     edgecolor=C.FG, alpha=0.0, linewidth=1.0, zorder=3)
                   for _ in range(n_env)]
    for p in env_patches:
        ax.add_patch(p)
    worst_marker, = ax.plot([], [], marker="x", markersize=13, mew=3,
                             color=C.FG, zorder=6)

    stat_panel = fig.text(0.985, 0.855, "", ha="right", va="top", color=C.FG,
                           fontsize=19, linespacing=1.7,
                           bbox=dict(boxstyle="round,pad=0.6", facecolor="#1A1D1F",
                                     edgecolor=C.GRID, linewidth=1.0))
    radius_title = fig.text(0.37, 0.885, "", ha="center", va="top", color=C.FG,
                             fontsize=26, fontweight="bold")

    def draw_frame(i: int):
        case_idx = n_cases - 1
        for k in range(n_cases):
            if bounds[k] <= i < bounds[k + 1]:
                case_idx = k
                break
        phase_len = max(bounds[case_idx + 1] - bounds[case_idx], 1)
        local_i = i - bounds[case_idx]
        progress = min(local_i / (phase_len * sweep_frac), 1.0)

        case = cases[case_idx]
        color = phase_colors[case_idx]
        poses = case["poses"]
        clears = case["clears"]
        n_poses = len(poses)
        idx = max(int(progress * (n_poses - 1)), 0)

        # 残像（機体外形のコマ送り）
        env_idxs = sorted(set(int(k * idx / max(n_env - 1, 1)) for k in range(n_env)))
        for p_artist, k in zip(env_patches, env_idxs + [None] * (n_env - len(env_idxs))):
            if k is None:
                p_artist.set_alpha(0.0)
                continue
            corners = [(cx * 1000.0, cy * 1000.0) for cx, cy in robot_corners(poses[k])]
            p_artist.set_xy(corners)
            p_artist.set_edgecolor(color)
            p_artist.set_alpha(0.22)

        # 現在の機体外形
        corners_now = [(cx * 1000.0, cy * 1000.0) for cx, cy in robot_corners(poses[idx])]
        main_patch.set_xy(corners_now)
        main_patch.set_facecolor(color)
        main_patch.set_edgecolor(color)

        # ここまでの最小余裕
        running_min = min(clears[:idx + 1])
        running_worst_idx = clears[:idx + 1].index(running_min)
        wp = poses[running_worst_idx]
        worst_marker.set_data([wp.x * 1000.0], [wp.y * 1000.0])
        worst_marker.set_color(color)

        radius_title.set_text(f"半径 R = {case['r_mm']:.0f} mm")
        radius_title.set_color(color)

        verdict = "通れる（余裕あり）" if case["min_clear"] >= 0.0 else "通れない（干渉）"
        stat_panel.set_text(
            "機体外形 100mm x 80mm を掃引\n"
            f"ここまでの最小余裕 = {running_min:.2f} mm\n"
            f"この半径の最小余裕 = {case['min_clear']:.2f} mm\n"
            f"判定: {verdict}\n"
            f"中心線上の通行可能な最大半径\n"
            f"  = {r_star_mm:.2f} mm（margin=0）"
        )
        return ()

    out_path = C.OUT_DIR / "clip_06_sweep_clearance.mp4"
    C.render_clip(fig, draw_frame, n_active, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
