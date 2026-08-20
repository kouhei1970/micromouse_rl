# research_notes/scripts/video_kinematics/clip_06_sweep_clearance.py
# クリップ6: 通れるかどうかは幾何が決める — 90° ターンの掃引余裕（約49.7秒）
#
# classic/geometry.py（読み取り専用の import）で、迷路の壁・柱と機体外形（100x80mm）を
# 掃引したときの最小余裕を計算する。半径を 90 -> 190 -> 250mm と上げていくと余裕が
# 減り、通れなくなったところで赤くする。局所迷路の配置（西区画→角区画→北区画の
# 90°左ターン）は tests/test_geometry.py の検査2〜4・6・7 と同じもの
# （geometry_anchor.py・note_031 の値との照合済みの配置）を再構成する。
#
# 台本（narration/clip_06_sweep_clearance.txt）の切れ目で画面の要素を増やす:
#   導入（幾何が決める・掃引の説明） / R=90mm / R=190mm / R=250mm /
#   仕上げ（通路幅いっぱいの300mmとの対比・機体前後端の張り出し）
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

    # ---- 台本の切れ目（narration/clip_06_sweep_clearance.txt。行4は2文を含むので分ける） ----
    intro_text = ("次に、その経路が通れるかどうかは幾何が決めます。"
                  "機体の外形を円弧に沿って掃引し、壁と柱にどれだけ余裕があるかを測ります。")
    r90_text = "半径90mmでは、内側の柱まで41.5mm、外側の壁まで34.7mmの余裕があります。"
    r190_text = "半径を上げていくと余裕は減り、190mmでほぼゼロになります。"
    r250_text = "250mmでは通れません。"
    caveat_text = ("通路幅から素朴に計算すると300mmまで取れそうに見えますが、"
                   "弧を描くとき機体の前後端は中心より外側を通ります。その分が入っていませんでした。")

    # ナレーション実測 48.720s（ffprobe, 2026-08-20） + 余韻 1.0s。
    total_seconds = 49.720
    n_active = C.seconds_to_active_frames(total_seconds)
    n_cases = len(cases)
    b = C.stage_bounds(
        [len(intro_text), len(r90_text), len(r190_text), len(r250_text), len(caveat_text)],
        n_active,
    )
    # b = [0, intro末, r90末, r190末, r250末, n_active(=caveat末)]
    case_bounds = b[1:5]  # 3ケースぶんの [開始,終了) 境界（intro分だけ後ろにずらす）
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
    caption = C.add_caption(
        fig,
        f"中心線上で通れる最大半径は {r_star_mm:.0f} mm。"
        "通路幅いっぱいの300 mmは機体の長さのぶん通れない。",
        y=0.045, fontsize=20,
    )
    caption.set_visible(False)  # 仕上げ段（caveat）でだけ出す（bboxが残るのでalphaでなくvisibleで切替）

    intro_note = fig.text(0.37, 0.885, "機体の外形を円弧に沿って掃引する", ha="center",
                           va="top", color=C.FG, fontsize=24, fontweight="bold")

    # 仕上げ段: 機体の前後端が中心線の外側を通ることを示す張り出しの注記
    # （HALF_LENGTH は classic/geometry.py の機体半長。手打ちの定数ではない）
    # 位置はプロット領域と重ならないよう右下の余白（プロット右・キャプション上）に置く。
    overhang_mm = HALF_LENGTH * 1000.0
    overhang_note = fig.text(
        0.74, 0.30,
        f"機体の半長 = {overhang_mm:.0f} mm ぶん、\n前後端は中心線より外側を通る",
        ha="left", va="top", color=C.C_MEASURED, fontsize=19, linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1A1D1F",
                  edgecolor=C.C_MEASURED, linewidth=1.2))
    overhang_note.set_visible(False)
    overhang_marker, = ax.plot([], [], marker="D", markersize=11, color=C.C_MEASURED,
                                zorder=7)
    overhang_marker.set_visible(False)

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
        # ---- 導入段（文1+文2）: まだ掃引を始めない。機体は最初の姿勢で静止 ----
        if i < b[1]:
            intro_note.set_visible(True)
            radius_title.set_visible(False)
            stat_panel.set_visible(False)
            idle_pose = cases[0]["poses"][0]
            corners_idle = [(cx * 1000.0, cy * 1000.0) for cx, cy in robot_corners(idle_pose)]
            main_patch.set_xy(corners_idle)
            main_patch.set_facecolor(C.GRID)
            main_patch.set_edgecolor(C.FG)
            for p_artist in env_patches:
                p_artist.set_alpha(0.0)
            worst_marker.set_data([], [])
            caption.set_visible(False)
            overhang_note.set_visible(False)
            overhang_marker.set_visible(False)
            return ()

        intro_note.set_visible(False)
        radius_title.set_visible(True)
        stat_panel.set_visible(True)

        # ---- どの半径のケースか（文3=R90 / 文4前半=R190 / 文4後半=R250） ----
        case_idx = n_cases - 1
        for k in range(n_cases):
            if case_bounds[k] <= i < case_bounds[k + 1]:
                case_idx = k
                break
        stage_start = case_bounds[case_idx]
        phase_len = max(case_bounds[case_idx + 1] - stage_start, 1)
        local_i = i - stage_start
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

        # ---- 仕上げ段（文5+文6）: R=250mm の最終姿勢を保持し、張り出しの注記を足す ----
        if i >= b[4]:
            caption.set_visible(True)
            overhang_note.set_visible(True)
            # 機体の最前端コーナー（進行方向ベクトルへの射影が最大の頂点）を
            # 張り出しの実例として指す（中心線＝pose.x,y は進行方向に沿って進むが、
            # 前端コーナーはその外側＝進行方向により先を通る）。
            p_final = poses[idx]
            heading = (math.cos(p_final.theta), math.sin(p_final.theta))
            cx_mm, cy_mm = p_final.x * 1000.0, p_final.y * 1000.0
            front_corner = max(
                corners_now,
                key=lambda c: (c[0] - cx_mm) * heading[0] + (c[1] - cy_mm) * heading[1],
            )
            overhang_marker.set_data([front_corner[0]], [front_corner[1]])
            overhang_marker.set_visible(True)
        else:
            caption.set_visible(False)
            overhang_note.set_visible(False)
            overhang_marker.set_visible(False)
        return ()

    out_path = C.OUT_DIR / "clip_06_sweep_clearance.mp4"
    C.render_clip(fig, draw_frame, n_active, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
