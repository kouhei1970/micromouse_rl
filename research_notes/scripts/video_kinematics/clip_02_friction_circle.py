# research_notes/scripts/video_kinematics/clip_02_friction_circle.py
# クリップ2: 摩擦円（楕円）— この機体は何ができるのか（約38.8秒）
#
# 横軸 a_y（横方向・上限 A_LAT）、縦軸 a_x（前後方向・上限 A_TR）の楕円を描き、
# 加速度ベクトルが縁を一周するアニメーション。値はすべて classic/profile.py の
# vehicle_limits()（読み取り専用の import）から計算する。手打ちの定数は使わない。
#
# 台本（narration/clip_02_friction_circle.txt）の4文の切れ目で画面の要素を増やす:
#   文1: 導入（軸だけ） / 文2: 楕円が育つ（前後と横の代償） /
#   文3: ベクトルが縁を回る（滑らない領域） / 文4: 4つの数値パネルが出る
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

# 台本の文（narration/clip_02_friction_circle.txt と同一）。文字数の比で
# 画面の要素が現れるタイミングを決める（C.stage_bounds）。
SENTENCES = [
    "まず、この機体に何ができるのかを決めておきます。",
    "タイヤが路面に伝えられる力には上限があり、前後に使った分だけ、横に使える分が減ります。",
    "この楕円の内側なら滑りません。最小時間の走りとは、縁に張り付いたまま経路をなぞることです。",
    "前後の上限は5.57m/s²、横は6.20m/s²、最高速は3.84m/s、ヨーの角加速度は210rad/s²です。",
]

# ナレーション実測 37.752s（ffprobe, 2026-08-20） + 余韻 1.0s。
# 🔴 尺は build_part1.py が環境変数で渡す（下の値はクリップ単体で走らせたときの既定）
_DEFAULT_TOTAL_SECONDS = 38.752
def main() -> None:
    C.setup_style()
    lim = vehicle_limits()
    A_LAT = lim.A_LAT   # 横方向（a_y）の上限 [m/s^2]
    A_TR = lim.A_TR      # 前後方向（a_x）の上限 [m/s^2]
    V_TOP = lim.V_TOP
    alpha_yaw_max = lim.alpha_yaw_max

    n_active = C.seconds_to_active_frames(C.target_seconds(_DEFAULT_TOTAL_SECONDS))
    b = C.stage_bounds([len(s) for s in SENTENCES], n_active)
    # b[0]=0（文1開始）, b[1]=文2開始, b[2]=文3開始, b[3]=文4開始, b[4]=n_active

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

    # 楕円の境界と塗りつぶし（文2で 0 -> 1 に育つ）
    theta_full = [2.0 * math.pi * k / 400 for k in range(401)]
    cos_full = [math.cos(t) for t in theta_full]
    sin_full = [math.sin(t) for t in theta_full]
    ellipse_fill, = ax.fill([0.0], [0.0], color=C.C_LIMIT, alpha=0.12, zorder=1)
    ellipse_line, = ax.plot([], [], color=C.C_LIMIT, linewidth=3.0, zorder=2)

    # 縁の極値（A_TR・A_LAT）は右上の数値パネルに数値があるため、ここでは軸へ
    # 破線を落として「軸の切片＝上限そのもの」であることだけ示す（文字は重ねない）。
    intercept_v, = ax.plot([], [], color=C.C_LIMIT, linewidth=1.0,
                            linestyle="--", alpha=0.5, zorder=1)
    intercept_h, = ax.plot([], [], color=C.C_LIMIT, linewidth=1.0,
                            linestyle="--", alpha=0.5, zorder=1)

    # 動く加速度ベクトル（矢印）と軌跡（文3から動き出す）
    arrow = ax.annotate("", xy=(A_LAT, 0.0), xytext=(0.0, 0.0),
                         arrowprops=dict(arrowstyle="-|>", color=C.FG,
                                          linewidth=3.0, mutation_scale=22),
                         zorder=4)
    arrow.set_visible(False)
    tip_dot, = ax.plot([], [], marker="o", markersize=9, color=C.FG, zorder=5)
    trail, = ax.plot([], [], color=C.FG, linewidth=1.6, alpha=0.55, zorder=3)

    C.add_title(fig, "この機体は何ができるのか — 摩擦円（楕円）")
    caption = C.add_caption(
        fig, "楕円の内側なら滑らない。最小時間の走りは縁に張り付く。", y=0.045)
    caption.set_visible(False)  # set_alpha(0)だとbboxが残るため可視性で切り替える

    stat_lines = [
        "車両の物理限界（vehicle_limits()）",
        f"V_TOP           = {V_TOP:.3f} m/s",
        f"A_TR            = {A_TR:.3f} m/s²",
        f"A_LAT           = {A_LAT:.3f} m/s²",
        f"alpha_yaw_max   = {alpha_yaw_max:.1f} rad/s²",
    ]
    stat_panel = C.add_stat_panel(fig, stat_lines, x=0.985, y=0.90)
    stat_panel.set_visible(False)

    vector_label = fig.text(0.985, 0.60, "", ha="right", va="top", color=C.FG,
                             fontsize=18, linespacing=1.6,
                             bbox=dict(boxstyle="round,pad=0.6", facecolor="#1A1D1F",
                                       edgecolor=C.GRID, linewidth=1.0))
    vector_label.set_visible(False)

    trail_theta = []
    # 文3+文4（ベクトルが動く区間）で 2 周する
    revolutions = 2.0
    motion_span = max(n_active - b[2] - 1, 1)

    def draw_frame(i: int):
        # ---- 楕円の生育（文1: 0 / 文2: 0->1 / 文3以降: 1） ----
        if i < b[1]:
            growth = 0.0
        elif i < b[2]:
            growth = (i - b[1]) / max(b[2] - b[1], 1)
        else:
            growth = 1.0

        if growth > 0.0:
            ex = [growth * A_LAT * c for c in cos_full]
            ey = [growth * A_TR * s for s in sin_full]
            ellipse_fill.set_xy(list(zip(ex, ey)))
            ellipse_line.set_data(ex, ey)
            intercept_v.set_data([0.0, 0.0], [0.0, growth * A_TR])
            intercept_h.set_data([0.0, growth * A_LAT], [0.0, 0.0])
        else:
            ellipse_fill.set_xy([(0.0, 0.0)])
            ellipse_line.set_data([], [])
            intercept_v.set_data([], [])
            intercept_h.set_data([], [])

        # ---- ベクトル（文3から動き出し、文3+文4で2周する） ----
        if i >= b[2]:
            theta = 2.0 * math.pi * revolutions * (i - b[2]) / motion_span
            a_y_v = A_LAT * math.cos(theta)
            a_x_v = A_TR * math.sin(theta)
            arrow.xy = (a_y_v, a_x_v)
            arrow.set_visible(True)
            tip_dot.set_data([a_y_v], [a_x_v])
            trail_theta.append(theta)
            tx = [A_LAT * math.cos(t) for t in trail_theta]
            ty = [A_TR * math.sin(t) for t in trail_theta]
            trail.set_data(tx, ty)
            usage = math.hypot(a_y_v / A_LAT, a_x_v / A_TR) * 100.0
        else:
            a_y_v = a_x_v = usage = 0.0

        caption.set_visible(i >= b[2])

        # ---- 数値（文4だけで見せる） ----
        if i >= b[3]:
            stat_panel.set_visible(True)
            vector_label.set_visible(True)
            vector_label.set_text(
                "現在の加速度ベクトル\n"
                f"a_y = {a_y_v:+.3f} m/s²\n"
                f"a_x = {a_x_v:+.3f} m/s²\n"
                f"楕円の使用率 = {usage:.1f}%"
            )
        else:
            stat_panel.set_visible(False)
            vector_label.set_visible(False)
        return ()

    out_path = C.OUT_DIR / "clip_02_friction_circle.mp4"
    C.render_clip(fig, draw_frame, n_active, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
