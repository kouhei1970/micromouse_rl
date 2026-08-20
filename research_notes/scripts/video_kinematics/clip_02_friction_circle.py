# research_notes/scripts/video_kinematics/clip_02_friction_circle.py
# クリップ2: 摩擦円（楕円）— この機体は何ができるのか（約12秒）
#
# 横軸 a_y（横方向・上限 A_LAT）、縦軸 a_x（前後方向・上限 A_TR）の楕円を描き、
# 加速度ベクトルが縁を一周するアニメーション。値はすべて classic/profile.py の
# vehicle_limits()（読み取り専用の import）から計算する。手打ちの定数は使わない。
#
# 実行: .venv/bin/python research_notes/scripts/video_kinematics/clip_02_friction_circle.py
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classic.profile import vehicle_limits  # noqa: E402  読み取り専用の import

import _common as C  # noqa: E402


def main() -> None:
    C.setup_style()
    lim = vehicle_limits()
    A_LAT = lim.A_LAT   # 横方向（a_y）の上限 [m/s^2]
    A_TR = lim.A_TR      # 前後方向（a_x）の上限 [m/s^2]
    V_TOP = lim.V_TOP
    alpha_yaw_max = lim.alpha_yaw_max

    total_seconds = 12.0
    n_active = C.seconds_to_active_frames(total_seconds)

    fig = C.new_figure()
    ax = fig.add_axes([0.09, 0.17, 0.80, 0.66])
    C.style_axes(ax)

    lim_axis = max(A_LAT, A_TR) * 1.30
    ax.set_xlim(-lim_axis, lim_axis)
    ax.set_ylim(-lim_axis, lim_axis)
    ax.set_aspect("equal")
    ax.set_xlabel("a_y  横方向加速度  [m/s²]", fontsize=16)
    ax.set_ylabel("a_x  前後方向加速度  [m/s²]", fontsize=16)
    ax.axhline(0.0, color=C.GRID, linewidth=1.0)
    ax.axvline(0.0, color=C.GRID, linewidth=1.0)

    # 楕円の境界と塗りつぶし
    theta_full = [2.0 * math.pi * k / 400 for k in range(401)]
    ex = [A_LAT * math.cos(t) for t in theta_full]
    ey = [A_TR * math.sin(t) for t in theta_full]
    ax.fill(ex, ey, color=C.C_LIMIT, alpha=0.12, zorder=1)
    ax.plot(ex, ey, color=C.C_LIMIT, linewidth=3.0, zorder=2)

    # 縁の極値（A_TR・A_LAT）は右上の数値パネルに数値があるため、ここでは軸へ
    # 破線を落として「軸の切片＝上限そのもの」であることだけ示す（文字は重ねない）。
    ax.plot([0.0, 0.0], [0.0, A_TR], color=C.C_LIMIT, linewidth=1.0,
            linestyle="--", alpha=0.5, zorder=1)
    ax.plot([0.0, A_LAT], [0.0, 0.0], color=C.C_LIMIT, linewidth=1.0,
            linestyle="--", alpha=0.5, zorder=1)

    # 動く加速度ベクトル（矢印）と軌跡（縁を一周した分だけ描く）
    arrow = ax.annotate("", xy=(A_LAT, 0.0), xytext=(0.0, 0.0),
                         arrowprops=dict(arrowstyle="-|>", color=C.FG,
                                          linewidth=3.0, mutation_scale=22),
                         zorder=4)
    tip_dot, = ax.plot([A_LAT], [0.0], marker="o", markersize=9,
                        color=C.FG, zorder=5)
    trail, = ax.plot([], [], color=C.FG, linewidth=1.6, alpha=0.55, zorder=3)

    C.add_title(fig, "この機体は何ができるのか — 摩擦円（楕円）")
    C.add_caption(fig, "楕円の内側なら滑らない。最小時間の走りは縁に張り付く。", y=0.045)

    stat_lines = [
        "車両の物理限界（vehicle_limits()）",
        f"V_TOP           = {V_TOP:.3f} m/s",
        f"A_TR            = {A_TR:.3f} m/s²",
        f"A_LAT           = {A_LAT:.3f} m/s²",
        f"alpha_yaw_max   = {alpha_yaw_max:.1f} rad/s²",
    ]
    C.add_stat_panel(fig, stat_lines, x=0.985, y=0.90)

    vector_label = fig.text(0.985, 0.60, "", ha="right", va="top", color=C.FG,
                             fontsize=18, linespacing=1.6,
                             bbox=dict(boxstyle="round,pad=0.6", facecolor="#1A1D1F",
                                       edgecolor=C.GRID, linewidth=1.0))

    trail_theta = []

    def draw_frame(i: int):
        theta = 2.0 * math.pi * i / n_active
        a_y_v = A_LAT * math.cos(theta)
        a_x_v = A_TR * math.sin(theta)
        arrow.xy = (a_y_v, a_x_v)
        tip_dot.set_data([a_y_v], [a_x_v])
        trail_theta.append(theta)
        tx = [A_LAT * math.cos(t) for t in trail_theta]
        ty = [A_TR * math.sin(t) for t in trail_theta]
        trail.set_data(tx, ty)
        usage = math.hypot(a_y_v / A_LAT, a_x_v / A_TR) * 100.0
        vector_label.set_text(
            "現在の加速度ベクトル\n"
            f"a_y = {a_y_v:+.3f} m/s²\n"
            f"a_x = {a_x_v:+.3f} m/s²\n"
            f"楕円の使用率 = {usage:.1f}%"
        )
        return ()

    out_path = C.OUT_DIR / "clip_02_friction_circle.mp4"
    C.render_clip(fig, draw_frame, n_active, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
