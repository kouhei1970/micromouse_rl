# research_notes/scripts/video_kinematics/clip_03_two_choices.py
# クリップ3: 2つの設計上の選択 — η=5.9%の中身を「見せる」（約58.4秒）
#
# 前作（clip_09_eta）は「ターン方式2.17倍・速度上限4.23倍」と数字を言うだけで、
# その2つが具体的に何なのかを一度も見せていなかった（ユーザ指摘）。本クリップは
# その2つの選択を実際の動き・実際の数値で見せる:
#   選択1（曲がり方）: 区画中心で止まり、その場で回り、また発進する現在の方式を、
#     物理限界（バンバン制御の軌道）と並べて動かす。時計（ウェッジ）で経過秒を見せる。
#   選択2（速度の上限）: 巡航速度 0.12 m/s と最高速 3.84 m/s を横棒で並べ、
#     同じ1区画（0.18m）を渡る速さの違いを2つの点の動きで見せる。
#
# 台本（narration/clip_03_two_choices.txt）の6文の切れ目で画面を切り替える:
#   文1: 導入 / 文2+文3: 選択1（曲がり方） / 文4+文5: 選択2（速度の上限） / 文6: 締め
#
# 数値の出所:
#   - 物理限界（その場90°旋回の最小時間・V_TOP・A_TR・A_LAT）は
#     classic/profile.py の vehicle_limits()・spin_turn_time()（読み取り専用の import）
#     からその場で計算する。手打ちしない。
#   - 巡航速度 0.12 m/s は classic/motion.py の DEFAULT_V_CRUISE（読み取り専用の
#     import。現行実装が実際に使っている値そのもの）。
#   - 現行方式の実測 90°旋回時間 2.000 s は classic/ の外・実験で得た値
#     （research_notes/note_031_profile_planner_and_eta.md §「その場旋回は物理限界の
#     8%でしか回っていない」/ README.md「現在地」表。clip_09_eta.py の
#     T_MEASURED_SPIN90 と同一の出典・同一の値）。
#
# 実行: .venv/bin/python research_notes/scripts/video_kinematics/clip_03_two_choices.py
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classic.profile import vehicle_limits, spin_turn_time  # noqa: E402 読み取り専用
from classic.motion import DEFAULT_V_CRUISE  # noqa: E402 読み取り専用の import
from classic.geometry import Pose, robot_corners  # noqa: E402 読み取り専用（描画のみに使う）
from mouse.params import RobotParams  # noqa: E402 読み取り専用の import

import matplotlib.patches as mpatches  # noqa: E402
import _common as C  # noqa: E402

CELL = RobotParams().cell_size  # 0.180 m

# ---- 実測（classic/ の外・実験で得た値。出典は上のモジュールdocstring参照） ----
T_MEASURED_SPIN90 = 2.000  # 現行方式（コマンド発行→停止待ち）の実測。clip_09_eta.py と同一出典。


def robot_polygon(ax, color):
    poly = mpatches.Polygon([(0.0, 0.0)] * 4, closed=True, facecolor=color,
                             alpha=0.65, edgecolor=color, linewidth=2.2, zorder=5)
    ax.add_patch(poly)
    return poly


def update_robot(poly, pose: Pose):
    poly.set_xy(robot_corners(pose))


def lerp(a, b, t):
    t = min(max(t, 0.0), 1.0)
    return a + (b - a) * t


def draw_clock(ax, center, radius, color):
    circle = mpatches.Circle(center, radius, facecolor="none", edgecolor=color,
                              linewidth=2.0, zorder=6)
    ax.add_patch(circle)
    wedge = mpatches.Wedge(center, radius, 90.0, 90.0, facecolor=color, alpha=0.45,
                            edgecolor="none", zorder=6)
    ax.add_patch(wedge)
    return circle, wedge


def set_clock(wedge, center, radius, frac):
    """`frac`（0..1）ぶん時計回りに扇形を満たす（12時位置から）。"""
    frac = min(max(frac, 0.0), 1.0)
    theta1 = 90.0 - 360.0 * frac
    wedge.set_center(center)
    wedge.set_radius(radius)
    wedge.theta1 = theta1
    wedge.theta2 = 90.0


def main() -> None:
    C.setup_style()
    limits = vehicle_limits()
    V_TOP = limits.V_TOP
    t_limit_spin90 = spin_turn_time(math.pi / 2.0, limits).time
    ratio_turn = T_MEASURED_SPIN90 / t_limit_spin90
    usage_pct = DEFAULT_V_CRUISE / V_TOP * 100.0
    t_cross_slow = CELL / DEFAULT_V_CRUISE
    t_cross_fast = CELL / V_TOP

    # ---- 台本の文（narration/clip_03_two_choices.txt と同一） ----
    s1 = "その5.9パーセントは、2つの設計上の選択で説明がつきます。どちらも、物理が禁じているわけではありません。"
    s2 = "1つめは、曲がり方です。いまのマウスは、区画の中心でいったん止まり、その場で回り、また発進します。"
    s3 = "その場で90°回るのに2.00秒かかります。物理限界は0.173秒ですから、11.6倍です。"
    s4 = "2つめは、速度の上限です。巡航速度を0.12m/sに決めてあります。"
    s5 = "この機体の最高速は3.84m/sですから、3パーセントしか使っていません。"
    s6 = "どちらも、確実に動かすために選んだ値でした。安全側に倒した結果が、限界の20分の1です。"

    # ナレーション実測 57.432s（ffprobe, 2026-08-20） + 余韻 1.0s。
    total_seconds = C.target_seconds(58.432)
    n_active = C.seconds_to_active_frames(total_seconds)
    b = C.stage_bounds([len(t) for t in (s1, s2, s3, s4, s5, s6)], n_active)
    # b = [0, s1末, s2末, s3末, s4末, s5末, n_active(s6末)]

    fig = C.new_figure()
    C.add_title(fig, "2つの設計上の選択 — η=5.9%の中身", y=0.95)

    # ==== View A: 導入（文1） ====
    intro_big = fig.text(0.5, 0.55, "", ha="center", va="center", color=C.FG,
                          fontsize=40, fontweight="bold")
    intro_sub = fig.text(0.5, 0.42, "", ha="center", va="center", color=C.FG, fontsize=22)

    # ==== View B: 選択1・曲がり方（文2+文3） ====
    ax_l = fig.add_axes([0.06, 0.24, 0.40, 0.62])
    ax_r = fig.add_axes([0.56, 0.24, 0.40, 0.62])
    for ax in (ax_l, ax_r):
        C.style_axes(ax)
        ax.set_xlim(-0.15 * CELL, 2.15 * CELL)
        ax.set_ylim(-0.15 * CELL, 2.15 * CELL)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    def draw_walls(ax):
        # L字（SW+SE+NE の3区画）の外周。入口(西)と出口(北)だけ開けてある。
        segs = [
            ((0.0, 0.0), (2 * CELL, 0.0)),
            ((2 * CELL, 0.0), (2 * CELL, 2 * CELL)),
            ((CELL, 2 * CELL), (CELL, CELL)),
            ((CELL, CELL), (0.0, CELL)),
        ]
        for (x0, y0), (x1, y1) in segs:
            ax.plot([x0, x1], [y0, y1], color=C.FG, linewidth=3.0, zorder=3)
        # 区画境界（薄い目印線）
        ax.plot([CELL, CELL], [0.0, CELL], color=C.GRID, linewidth=1.0, linestyle="--", zorder=1)
        ax.plot([CELL, 2 * CELL], [CELL, CELL], color=C.GRID, linewidth=1.0, linestyle="--", zorder=1)

    draw_walls(ax_l)
    draw_walls(ax_r)
    ax_l.set_title("現在の方式（コマンド）", color=C.C_MEASURED, fontsize=20, pad=8)
    ax_r.set_title("物理限界（バンバン）", color=C.C_LIMIT, fontsize=20, pad=8)

    robot_l = robot_polygon(ax_l, C.C_MEASURED)
    robot_r = robot_polygon(ax_r, C.C_LIMIT)

    clock_center = (0.5 * CELL, 1.64 * CELL)
    clock_radius = 0.26 * CELL
    circle_l, wedge_l = draw_clock(ax_l, clock_center, clock_radius, C.C_MEASURED)
    circle_r, wedge_r = draw_clock(ax_r, clock_center, clock_radius, C.C_LIMIT)
    for c, w in ((circle_l, wedge_l), (circle_r, wedge_r)):
        c.set_visible(False)
        w.set_visible(False)
    time_text_l = ax_l.text(clock_center[0], clock_center[1] - clock_radius - 0.10 * CELL,
                             "", ha="center", va="top", color=C.C_MEASURED, fontsize=17,
                             fontweight="bold")
    time_text_r = ax_r.text(clock_center[0], clock_center[1] - clock_radius - 0.10 * CELL,
                             "", ha="center", va="top", color=C.C_LIMIT, fontsize=17,
                             fontweight="bold")

    ratio_callout = fig.text(0.5, 0.10, "", ha="center", va="center", color=C.FG,
                              fontsize=26, fontweight="bold",
                              bbox=dict(boxstyle="round,pad=0.5", facecolor="#1A1D1F",
                                        edgecolor=C.GRID, linewidth=1.0))
    ratio_callout.set_visible(False)

    entry_pose = Pose(-0.10 * CELL, 0.5 * CELL, 0.0)
    center_pose = Pose(1.5 * CELL, 0.5 * CELL, 0.0)
    exit_pose = Pose(1.5 * CELL, 2.10 * CELL, math.pi / 2.0)

    def pose_schematic(progress: float) -> Pose:
        """文2段（概念）: 進入(0-35%)→回転(35-75%)→退出(75-100%)。数値なし。"""
        if progress < 0.35:
            t = progress / 0.35
            x = lerp(entry_pose.x, center_pose.x, t)
            return Pose(x, 0.5 * CELL, 0.0)
        if progress < 0.75:
            t = (progress - 0.35) / 0.40
            theta = lerp(0.0, math.pi / 2.0, t)
            return Pose(center_pose.x, center_pose.y, theta)
        t = (progress - 0.75) / 0.25
        y = lerp(center_pose.y, exit_pose.y, t)
        return Pose(center_pose.x, y, math.pi / 2.0)

    def draw_view_b(i: int):
        ax_l.set_visible(True)
        ax_r.set_visible(True)
        local = i - b[1]
        n_view = b[3] - b[1]
        sub2_end = b[2] - b[1]

        if local < sub2_end:
            # ---- 文2: 概念のみ（同じ動きを両パネルで見せる。数値・時計は無し） ----
            n_sub2 = max(sub2_end, 1)
            progress = min(local / max(n_sub2 - 1, 1), 1.0)
            pose = pose_schematic(progress)
            update_robot(robot_l, pose)
            update_robot(robot_r, pose)
            for c, w in ((circle_l, wedge_l), (circle_r, wedge_r)):
                c.set_visible(False)
                w.set_visible(False)
            time_text_l.set_text("")
            time_text_r.set_text("")
            ratio_callout.set_visible(False)
            return ()

        # ---- 文3: 実時間比較（区画中心で静止し、その場旋回だけをやり直す） ----
        local3 = local - sub2_end
        n_sub3 = max(n_view - sub2_end, 1)
        # 同じ「仮想時間の進み方」を両パネルに与える。左は2.00sで満了、
        # 右は0.173sで満了して待つ（この差そのものが見せたいもの）。
        virtual_t = min(local3 / max(n_sub3 - 1, 1), 1.0) * T_MEASURED_SPIN90
        elapsed_l = min(virtual_t, T_MEASURED_SPIN90)
        elapsed_r = min(virtual_t, t_limit_spin90)
        theta_l = elapsed_l / T_MEASURED_SPIN90 * (math.pi / 2.0)
        theta_r = elapsed_r / t_limit_spin90 * (math.pi / 2.0)
        update_robot(robot_l, Pose(center_pose.x, center_pose.y, theta_l))
        update_robot(robot_r, Pose(center_pose.x, center_pose.y, theta_r))

        for c, w in ((circle_l, wedge_l), (circle_r, wedge_r)):
            c.set_visible(True)
            w.set_visible(True)
        set_clock(wedge_l, clock_center, clock_radius, elapsed_l / T_MEASURED_SPIN90)
        set_clock(wedge_r, clock_center, clock_radius, elapsed_r / t_limit_spin90)
        time_text_l.set_text(f"{elapsed_l:.3f} s")
        done_r = "（完了）" if elapsed_r >= t_limit_spin90 - 1e-9 else ""
        time_text_r.set_text(f"{elapsed_r:.3f} s{done_r}")

        if local3 >= 0.82 * n_sub3:
            ratio_callout.set_visible(True)
            ratio_callout.set_text(
                f"{T_MEASURED_SPIN90:.3f} s ÷ {t_limit_spin90:.3f} s = {ratio_turn:.1f} 倍")
        else:
            ratio_callout.set_visible(False)
        return ()

    # ==== View C: 選択2・速度の上限（文4+文5） ====
    ax_speed = fig.add_axes([0.10, 0.44, 0.80, 0.26])
    C.style_axes(ax_speed)
    ax_speed.set_xlim(0.0, V_TOP * 1.15)
    ax_speed.set_ylim(-0.8, 1.8)
    ax_speed.set_yticks([0.0, 1.0])
    ax_speed.set_yticklabels(["巡航速度\n（現在）", "最高速\n（物理限界）"], fontsize=14)
    ax_speed.set_xlabel("速度 [m/s]", fontsize=14)

    bar_cap = mpatches.Rectangle((0.0, -0.30), 0.0, 0.60, facecolor=C.C_CAP_COST,
                                  edgecolor=C.C_CAP_COST, alpha=0.80, linewidth=1.5)
    bar_top = mpatches.Rectangle((0.0, 0.70), 0.0, 0.60, facecolor=C.C_LIMIT,
                                  edgecolor=C.C_LIMIT, alpha=0.80, linewidth=1.5)
    ax_speed.add_patch(bar_cap)
    ax_speed.add_patch(bar_top)
    label_cap = ax_speed.text(0.0, 0.0, "", va="center", ha="left", color=C.C_CAP_COST, fontsize=17)
    label_top = ax_speed.text(0.0, 1.0, "", va="center", ha="left", color=C.C_LIMIT, fontsize=17)
    usage_label = fig.text(0.5, 0.36, "", ha="center", va="center", color=C.FG, fontsize=18)

    ax_race = fig.add_axes([0.10, 0.18, 0.80, 0.16])
    C.style_axes(ax_race)
    ax_race.set_xlim(-0.02 * CELL, 1.15 * CELL)
    ax_race.set_ylim(-1.0, 1.0)
    ax_race.set_yticks([])
    ax_race.set_xlabel("1区画 = 0.18 m を渡る速さ", fontsize=14)
    ax_race.axvline(CELL, color=C.GRID, linewidth=1.2, linestyle="--")
    ax_race.plot([0.0, CELL], [0.35, 0.35], color=C.GRID, linewidth=1.5)
    ax_race.plot([0.0, CELL], [-0.35, -0.35], color=C.GRID, linewidth=1.5)
    dot_slow, = ax_race.plot([0.0], [0.35], marker="o", markersize=14, color=C.C_CAP_COST, zorder=5)
    dot_fast, = ax_race.plot([0.0], [-0.35], marker="o", markersize=14, color=C.C_LIMIT, zorder=5)
    slow_label = ax_race.text(0.0, 0.62, "", color=C.C_CAP_COST, fontsize=14, ha="left")
    fast_label = ax_race.text(0.0, -0.62, "", color=C.C_LIMIT, fontsize=14, ha="left")
    for artist in (dot_slow, dot_fast, slow_label, fast_label):
        artist.set_visible(False)

    def draw_view_c(i: int):
        ax_speed.set_visible(True)
        ax_race.set_visible(True)
        local = i - b[3]
        n_view = b[5] - b[3]
        sub4_end = b[4] - b[3]

        if local < sub4_end:
            # ---- 文4: 棒が育つ ----
            progress = min(local / max(sub4_end - 1, 1), 1.0)
            w_cap = DEFAULT_V_CRUISE * progress
            w_top = V_TOP * progress
            bar_cap.set_width(w_cap)
            bar_top.set_width(w_top)
            if progress >= 1.0:
                label_cap.set_position((w_cap + V_TOP * 0.01, 0.0))
                label_cap.set_text(f"{DEFAULT_V_CRUISE:.2f} m/s")
                label_top.set_position((w_top + V_TOP * 0.01, 1.0))
                label_top.set_text(f"{V_TOP:.3f} m/s")
            else:
                label_cap.set_text("")
                label_top.set_text("")
            usage_label.set_visible(False)
            for artist in (dot_slow, dot_fast, slow_label, fast_label):
                artist.set_visible(False)
            return ()

        # ---- 文5: 使用率の表示 + 1区画を渡るレース ----
        bar_cap.set_width(DEFAULT_V_CRUISE)
        bar_top.set_width(V_TOP)
        label_cap.set_position((DEFAULT_V_CRUISE + V_TOP * 0.01, 0.0))
        label_cap.set_text(f"{DEFAULT_V_CRUISE:.2f} m/s")
        label_top.set_position((V_TOP + V_TOP * 0.01, 1.0))
        label_top.set_text(f"{V_TOP:.3f} m/s")
        usage_label.set_visible(True)
        usage_label.set_text(
            f"{DEFAULT_V_CRUISE:.2f} / {V_TOP:.3f} = {usage_pct:.1f}%  しか使っていない")

        local5 = local - sub4_end
        n_sub5 = max(n_view - sub4_end, 1)
        progress5 = min(local5 / max(n_sub5 - 1, 1), 1.0)
        x_slow = min(CELL * progress5, CELL)
        # 速い方は同じ「仮想時間の進み方」で、1/32弱の時間で渡り終えて待つ
        virtual_t = progress5 * t_cross_slow
        x_fast = min(V_TOP * virtual_t, CELL)
        for artist in (dot_slow, dot_fast, slow_label, fast_label):
            artist.set_visible(True)
        dot_slow.set_data([x_slow], [0.35])
        dot_fast.set_data([x_fast], [-0.35])
        slow_label.set_text(f"{DEFAULT_V_CRUISE:.2f} m/s（{t_cross_slow:.2f} s で渡る）")
        fast_label.set_text(f"{V_TOP:.3f} m/s（{t_cross_fast:.3f} s で渡る）")
        return ()

    # ==== View D: 締め（文6） ====
    recap_line1 = fig.text(0.5, 0.62, "", ha="center", va="center", color=C.FG, fontsize=22)
    recap_line2 = fig.text(0.5, 0.53, "", ha="center", va="center", color=C.FG, fontsize=22)
    closing_text = fig.text(0.5, 0.36, "", ha="center", va="center", color=C.FG,
                             fontsize=26, fontweight="bold", linespacing=1.6)

    def draw_view_d(i: int):
        local = i - b[5]
        n_view = max(n_active - b[5], 1)
        progress = min(local / max(n_view - 1, 1), 1.0)
        recap_line1.set_visible(True)
        recap_line2.set_visible(True)
        recap_line1.set_text(
            f"曲がり方: {T_MEASURED_SPIN90:.2f} s → 物理限界 {t_limit_spin90:.3f} s"
            f"（{ratio_turn:.1f} 倍）")
        recap_line2.set_text(
            f"速度上限: {DEFAULT_V_CRUISE:.2f} m/s → 物理限界 {V_TOP:.3f} m/s"
            f"（使用率 {usage_pct:.1f}%）")
        if progress >= 0.35:
            closing_text.set_text(
                "どちらも、確実に動かすために選んだ値。\n安全側に倒した結果が、限界の20分の1。")
        else:
            closing_text.set_text("")
        return ()

    def hide_all():
        intro_big.set_visible(False)
        intro_sub.set_visible(False)
        ax_l.set_visible(False)
        ax_r.set_visible(False)
        ax_speed.set_visible(False)
        ax_race.set_visible(False)
        usage_label.set_visible(False)
        ratio_callout.set_visible(False)
        recap_line1.set_visible(False)
        recap_line2.set_visible(False)
        closing_text.set_text("")

    def draw_frame(i: int):
        hide_all()
        if i < b[1]:
            intro_big.set_visible(True)
            intro_sub.set_visible(True)
            intro_big.set_text("2つの設計上の選択")
            intro_sub.set_text("η=5.9%の中身。物理が禁じているわけではない。")
            return ()
        if i < b[3]:
            return draw_view_b(i)
        if i < b[5]:
            return draw_view_c(i)
        return draw_view_d(i)

    out_path = C.OUT_DIR / "clip_03_two_choices.mp4"
    C.render_clip(fig, draw_frame, n_active, out_path)
    print(f"saved: {out_path}")
    print(
        f"V_TOP={V_TOP:.6f} t_limit_spin90={t_limit_spin90:.6f} "
        f"ratio_turn={ratio_turn:.4f} DEFAULT_V_CRUISE={DEFAULT_V_CRUISE:.6f} "
        f"usage_pct={usage_pct:.4f} t_cross_slow={t_cross_slow:.4f} "
        f"t_cross_fast={t_cross_fast:.4f}"
    )


if __name__ == "__main__":
    main()
