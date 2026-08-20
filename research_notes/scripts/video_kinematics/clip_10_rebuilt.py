# research_notes/scripts/video_kinematics/clip_10_rebuilt.py
# クリップ10: 作り直した結果と、残っている壁（約65.3秒）
#
# 第1部の結末（ユーザ指摘: 診断（η=5.9%・原因は2つ）で終わっており結末が無い）。
# 「作り直した」の中身（コマンド発行→停止待ち／経路全体の速度プロファイル追従）を
# v(s) の形の違いで見せ、要素ごとの改善（旋回・弧・長い直線）を数字で示し、
# 最後に「まだ完走できていない」ことを正直に見せて第2部へ渡す。
#
# 台本（narration/clip_10_rebuilt.txt）の8文の切れ目で画面を切り替える:
#   文1: 導入 / 文2: v(s)の形（コマンド方式 vs 追従） / 文3: 終端だけがゼロという注記 /
#   文4: その場旋回の改善（バーが縮む） / 文5: 弧・長い直線の追従精度（表） /
#   文6: 完走 0/10 / 文7: 通路の余地44mmと衝突 / 文8: 締め（第2部へ）
#
# 数値の出所:
#   - 物理限界（その場旋回・v(s)台形）は classic/profile.py の
#     vehicle_limits()・spin_turn_time()・min_time()（読み取り専用の import）から
#     その場で計算する。手打ちしない。
#   - 直線→弧→直線の計画（半径は classic/motion.py の DEFAULT_ARC_RADIUS=0.09m。
#     ProfileTracker が実際に使っている値）は classic/geometry.py の turn_path
#     （読み取り専用の import）で作った区間列を min_time に渡して計算する。
#   - 通路の余地（片側）は classic/geometry.py の sweep_clearance
#     （clip_07_diagonal.py と同じ手法）でその場で計算する。44mm という数値を
#     直書きしない。
#   - 経路計画の総距離（16.90m）は、outputs/exp_027/u_0.30/wc_on/maze_41004.json
#     に一次記録として残る学習地図（maze_v_walls/maze_h_walls）から、
#     classic/fast_planner.py の plan_fast_run（読み取り専用の import。
#     experiments/exp_027_friction_sweep/judge.py と同じ手法）でその場で
#     計画を作り直して計算する。手打ちしない。
#   - 現行方式の実測 90°旋回 2.000 s は classic/ の外・実験で得た値
#     （clip_09_eta.py / clip_03_two_choices.py の T_MEASURED_SPIN90 と同一出典）。
#   - 追従（電圧前置き）の実測 90°旋回 0.180 s・弧接続 1.02 倍・
#     直線16区画(3.72m/s) 1.07 倍は research_notes/note_031_profile_planner_and_eta.md
#     §「段4の結果」実測（単体）表からの引用（classic/ の外・実験で得た値）。
#   - 完走 0/10・衝突距離 14.266 m は同ノート §「exp_027の途中結果」および
#     outputs/exp_027/u_0.30/wc_on/maze_41004.json の実測（u=0.30・壁センサ補正あり・
#     run index=4）。
#
# 実行: .venv/bin/python research_notes/scripts/video_kinematics/clip_10_rebuilt.py
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classic.profile import vehicle_limits, spin_turn_time, min_time, Segment  # noqa: E402
from classic.geometry import (  # noqa: E402 読み取り専用の import
    HALF_WIDTH, Pose, PathSegment, sweep_clearance, wall_obstacles, poses_along, turn_path,
)
from classic.motion import DEFAULT_V_CRUISE, DEFAULT_ARC_RADIUS  # noqa: E402
from classic.fast_planner import plan_fast_run, PathBlock  # noqa: E402 読み取り専用
from classic.maze_map import Direction, MazeMap  # noqa: E402 読み取り専用
from competition.evaluator import goal_cells  # noqa: E402 読み取り専用
from mouse.params import RobotParams  # noqa: E402 読み取り専用

import matplotlib.patches as mpatches  # noqa: E402
import _common as C  # noqa: E402

CELL = RobotParams().cell_size  # 0.180 m

# ---- 実測（classic/ の外・実験で得た値。出典は上のモジュールdocstring参照） ----
T_MEASURED_SPIN90_OLD = 2.000   # 現行方式（コマンド発行→停止待ち）。clip_03/clip_09と同一出典。
T_MEASURED_SPIN90_NEW = 0.180   # 追従（電圧前置き）。note_031 §段4「実測（単体）」表。
RATIO_ARC_CONNECTED = 1.02      # 直線→弧→直線（実測/計画）。同上。
RATIO_LONG_STRAIGHT = 1.07      # 直線16区画・上限なし・3.72m/s（実測/計画）。同上。
V_MEASURED_LONG_STRAIGHT = 3.72  # 同上に付随する実測最高速。

EXP027_RECORD = REPO_ROOT / "outputs" / "exp_027" / "u_0.30" / "wc_on" / "maze_41004.json"
N_MAZES_EXP027 = 10   # design_turn_v1・exp_027 の対象迷路数
N_GOAL_EXP027 = 0     # 完走した迷路数（note_031 §「exp_027の途中結果」: 完走はまだ0本）
U_FRICTION = 0.30


def compute_corridor_room() -> float:
    """直線通路の片側の余地 [m]（clip_07_diagonal.py の corridor_room と同一手法）。"""
    width = height = 4
    v_walls = np.zeros((width + 1, height), dtype=np.int8)
    h_walls = np.zeros((width, height + 1), dtype=np.int8)
    obstacles = wall_obstacles(v_walls, h_walls)
    segments = [PathSegment(kind="straight", length=0.30, curvature=0.0)]
    poses = poses_along(segments, Pose(-0.03, CELL / 2.0, 0.0), ds=0.001)
    room, _ = sweep_clearance(poses, obstacles)
    return room


def compute_exp027_reach():
    """exp_027（u=0.30・壁センサ補正あり）の一次記録から、衝突距離と計画総距離を
    その場で計算し直す（judge.py の compute_t_plan_from_saved_map と同じ手法）。"""
    record = json.loads(EXP027_RECORD.read_text(encoding="utf-8"))
    v = np.asarray(record["maze_v_walls"], dtype=np.int8)
    h = np.asarray(record["maze_h_walls"], dtype=np.int8)
    width = int(v.shape[0] - 1)
    height = int(v.shape[1])
    maze = MazeMap(width, height)
    maze.v_walls[:, :] = v
    maze.h_walls[:, :] = h
    goals = goal_cells(width, height)
    plan = plan_fast_run(maze, start=(0, 0), goals=goals, start_heading=Direction.N,
                          friction_use=U_FRICTION)
    plan_len = sum(st.s_grid[-1] for st in plan.steps if isinstance(st, PathBlock))
    collisions = [r["path_length_m"] for r in record["runs"] if r.get("outcome") == "collision"]
    reach = max(collisions)
    return reach, plan_len


def main() -> None:
    C.setup_style()
    limits = vehicle_limits()
    V_TOP = limits.V_TOP
    t_limit_spin90 = spin_turn_time(math.pi / 2.0, limits).time
    ratio_old = T_MEASURED_SPIN90_OLD / t_limit_spin90
    ratio_new = T_MEASURED_SPIN90_NEW / t_limit_spin90

    # ---- View A: v(s) の形（コマンド方式 vs 追従） ----
    cmd_s, cmd_v = [], []
    offset = 0.0
    for _ in range(4):
        r = min_time([Segment(length=CELL, curvature=0.0, kind="straight")], limits,
                      v_cap=DEFAULT_V_CRUISE)
        cmd_s += [offset + s for s in r.s_grid]
        cmd_v += list(r.v_grid)
        offset += CELL
    t_cmd_total = sum(
        min_time([Segment(length=CELL, curvature=0.0, kind="straight")], limits,
                  v_cap=DEFAULT_V_CRUISE).total
        for _ in range(4))
    r_cont = min_time([Segment(length=CELL * 4, curvature=0.0, kind="straight")], limits,
                       v_cap=DEFAULT_V_CRUISE)
    cont_s, cont_v = r_cont.s_grid, r_cont.v_grid
    t_cont_total = r_cont.total

    # ---- View B: 直線→弧→直線の計画（DEFAULT_ARC_RADIUS）と 2.88m 直線 ----
    turn_segs, _consumed = turn_path(math.pi / 2.0, DEFAULT_ARC_RADIUS)
    prof_segs = [Segment(length=s.length, curvature=s.curvature, kind=s.kind) for s in turn_segs]
    r_arc = min_time(prof_segs, limits, v_cap=DEFAULT_V_CRUISE)
    t_plan_arc = r_arc.total
    t_measured_arc = t_plan_arc * RATIO_ARC_CONNECTED

    r_long = min_time([Segment(length=2.88, curvature=0.0, kind="straight")], limits)
    t_plan_long = r_long.total
    t_measured_long = t_plan_long * RATIO_LONG_STRAIGHT

    # ---- View C: 通路の余地・完走率・衝突距離 ----
    room_m = compute_corridor_room()
    room_mm = room_m * 1000.0
    reach_m, plan_len_m = compute_exp027_reach()
    reach_pct = reach_m / plan_len_m * 100.0

    # ---- 台本の文（narration/clip_10_rebuilt.txt と同一） ----
    s1 = "そこで、走らせ方そのものを作り直しました。"
    s2 = ("コマンドを1つずつ発行して完了を待つのではなく、経路全体の速度プロファイルを"
          "あらかじめ作り、それを追従する。")
    s3 = "終端で速度がゼロになる軌道を設計するのですから、止まるのを待つ必要がありません。"
    s4 = "結果です。その場90°旋回は、物理限界の11.6倍から1.04倍になりました。"
    s5 = ("直線と円弧をつないだ区間は計画の1.02倍。迷路の端から端まで2.88mを、"
          "最高速3.72m/sで走らせても、計画の1.07倍で追従します。")
    s6 = "ただし、まだ迷路を1本も完走できていません。"
    s7 = "高速で走ると推測航法がずれ、通路の余地44mmを使い切って壁に当たります。"
    s8 = "どこまで肉薄できるのか。それは第2部で。"

    # ナレーション実測 64.320s（ffprobe, 2026-08-20） + 余韻 1.0s。
    total_seconds = C.target_seconds(65.320)
    n_active = C.seconds_to_active_frames(total_seconds)
    b = C.stage_bounds([len(t) for t in (s1, s2, s3, s4, s5, s6, s7, s8)], n_active)
    # b = [0, s1, s2, s3, s4, s5, s6, s7, n_active(s8)]

    fig = C.new_figure()
    C.add_title(fig, "作り直した結果と、残っている壁", y=0.95)

    # ==== View A: v(s)（文1+文2+文3） ====
    intro_big = fig.text(0.5, 0.55, "", ha="center", va="center", color=C.FG,
                          fontsize=38, fontweight="bold")
    intro_sub = fig.text(0.5, 0.42, "", ha="center", va="center", color=C.FG, fontsize=20)

    ax_vs = fig.add_axes([0.09, 0.20, 0.80, 0.60])
    C.style_axes(ax_vs)
    ax_vs.set_xlim(0.0, CELL * 4 * 1.02)
    ax_vs.set_ylim(0.0, DEFAULT_V_CRUISE * 1.35)
    ax_vs.set_xlabel("弧長 s [m]（区画 0.18m x 4）", fontsize=15)
    ax_vs.set_ylabel("速度 v [m/s]", fontsize=15)
    for k in (1, 2, 3):
        ax_vs.axvline(CELL * k, color=C.GRID, linewidth=1.0, linestyle="--", zorder=1)

    line_cmd, = ax_vs.plot([], [], color=C.C_MEASURED, linewidth=2.6, zorder=3,
                            label="現在: コマンド方式（区画ごとに完了待ち）")
    fill_cmd = ax_vs.fill_between([], [], color=C.C_MEASURED, alpha=0.18)
    line_cont, = ax_vs.plot([], [], color=C.C_LIMIT, linewidth=2.6, zorder=3,
                             label="作り直し: 経路全体を追従")
    legend_vs = ax_vs.legend(loc="upper right", fontsize=14, framealpha=0.85,
                              facecolor="#1A1D1F", edgecolor=C.GRID, labelcolor=C.FG)
    legend_vs.set_visible(False)

    stop_dots, = ax_vs.plot([], [], marker="o", markersize=10, color=C.C_MEASURED,
                             linestyle="none", zorder=5)
    keep_dots, = ax_vs.plot([], [], marker="o", markersize=10, color=C.C_LIMIT,
                             linestyle="none", zorder=5)
    stop_label = ax_vs.text(0.0, 0.0, "", color=C.C_MEASURED, fontsize=14, ha="center", va="bottom")
    keep_label = ax_vs.text(0.0, 0.0, "", color=C.C_LIMIT, fontsize=14, ha="center", va="top")
    time_note = fig.text(0.5, 0.095, "", ha="center", va="center", color=C.FG, fontsize=18)

    def draw_view_a(i: int):
        ax_vs.set_visible(True)
        if i < b[1]:
            intro_big.set_visible(True)
            intro_sub.set_visible(True)
            intro_big.set_text("走らせ方そのものを作り直した")
            intro_sub.set_text("")
            ax_vs.set_visible(False)
            return ()

        local = i - b[1]
        n_view = b[3] - b[1]
        sub2_end = b[2] - b[1]

        if local < sub2_end:
            local2 = local
            n_sub2 = max(sub2_end, 1)
            half = n_sub2 / 2.0
            if local2 < half:
                progress_cmd = min(local2 / max(half - 1, 1), 1.0)
                progress_cont = 0.0
            else:
                progress_cmd = 1.0
                progress_cont = min((local2 - half) / max(n_sub2 - half - 1, 1), 1.0)

            n_cmd = max(int(progress_cmd * (len(cmd_s) - 1)), 1)
            line_cmd.set_data(cmd_s[:n_cmd + 1], cmd_v[:n_cmd + 1])
            legend_vs.set_visible(progress_cmd > 0.0)

            if progress_cont > 0.0:
                n_cont = max(int(progress_cont * (len(cont_s) - 1)), 1)
                line_cont.set_data(cont_s[:n_cont + 1], cont_v[:n_cont + 1])
            else:
                line_cont.set_data([], [])
            stop_dots.set_data([], [])
            keep_dots.set_data([], [])
            stop_label.set_text("")
            keep_label.set_text("")
            time_note.set_text("")
            return ()

        # ---- 文3: 区画境界での違いを注記する ----
        line_cmd.set_data(cmd_s, cmd_v)
        line_cont.set_data(cont_s, cont_v)
        legend_vs.set_visible(True)
        bx = [CELL, 2 * CELL, 3 * CELL]
        stop_dots.set_data(bx, [0.0, 0.0, 0.0])
        keep_dots.set_data(bx, [DEFAULT_V_CRUISE] * 3)
        stop_label.set_text("")
        keep_label.set_text("")
        local3 = local - sub2_end
        n_sub3 = max(n_view - sub2_end, 1)
        if local3 >= 0.30 * n_sub3:
            stop_label.set_position((2 * CELL, 0.012))
            stop_label.set_text("区画境界でも一旦停止")
            keep_label.set_position((2 * CELL, DEFAULT_V_CRUISE + 0.008))
            keep_label.set_text(f"止まらない（{DEFAULT_V_CRUISE:.2f} m/s を保つ）")
        if local3 >= 0.65 * n_sub3:
            time_note.set_text(
                f"合計 {t_cmd_total:.3f} s → {t_cont_total:.3f} s"
                f"（終端でだけ v=0 に設計する）")
        return ()

    # ==== View B: 結果の表（文4+文5） ====
    ax_bar = fig.add_axes([0.10, 0.56, 0.80, 0.20])
    C.style_axes(ax_bar)
    ax_bar.set_xlim(0.0, T_MEASURED_SPIN90_OLD * 1.08)
    ax_bar.set_ylim(-0.8, 0.8)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("その場90°旋回の所要時間 [s]", fontsize=15)
    ax_bar.axvline(t_limit_spin90, color=C.C_LIMIT, linewidth=2.0, linestyle="--", zorder=2)
    ax_bar.text(t_limit_spin90, 0.62, f"物理限界 {t_limit_spin90:.3f} s", color=C.C_LIMIT,
                fontsize=14, ha="left", va="center")
    spin_bar = mpatches.Rectangle((0.0, -0.30), T_MEASURED_SPIN90_OLD, 0.60,
                                   facecolor=C.C_MEASURED, edgecolor=C.C_MEASURED,
                                   alpha=0.80, linewidth=1.5)
    ax_bar.add_patch(spin_bar)
    spin_label = ax_bar.text(0.0, 0.0, "", va="center", ha="left", color=C.FG, fontsize=17,
                              fontweight="bold")

    table_title = fig.text(0.5, 0.44, "計画への追従精度（実測 / 計画）", ha="center", va="center",
                            color=C.FG, fontsize=22, fontweight="bold")
    table_title.set_visible(False)
    col_x = [0.10, 0.55, 0.76, 0.90]
    header_y = 0.36
    headers = ["区間", "計画 T_plan", "実測", "倍率"]
    header_artists = []
    for x, htext, ha in zip(col_x, headers, ["left", "center", "center", "center"]):
        t = fig.text(x, header_y, htext, ha=ha, va="center", color=C.FG, fontsize=18,
                     fontweight="bold")
        t.set_visible(False)
        header_artists.append(t)
    rows_data = [
        ("直線 → 弧(R=90mm) → 直線", t_plan_arc, t_measured_arc, RATIO_ARC_CONNECTED),
        ("直線16区画 2.88m（3.72m/s）", t_plan_long, t_measured_long, RATIO_LONG_STRAIGHT),
    ]
    row_artists = []
    for k, (name, tp, tm, ratio) in enumerate(rows_data):
        y = header_y - 0.07 * (k + 1)
        lbl = fig.text(col_x[0], y, name, ha="left", va="center", color=C.FG, fontsize=18)
        v_plan = fig.text(col_x[1], y, f"{tp:.3f} s", ha="center", va="center", color=C.C_LIMIT,
                           fontsize=18)
        v_meas = fig.text(col_x[2], y, f"{tm:.3f} s", ha="center", va="center", color=C.C_MEASURED,
                           fontsize=18)
        v_ratio = fig.text(col_x[3], y, f"{ratio:.2f} 倍", ha="center", va="center", color=C.FG,
                            fontsize=18, fontweight="bold")
        for a in (lbl, v_plan, v_meas, v_ratio):
            a.set_visible(False)
        row_artists.append((lbl, v_plan, v_meas, v_ratio))

    def draw_view_b(i: int):
        ax_bar.set_visible(True)
        local = i - b[3]
        n_view = b[5] - b[3]
        sub4_end = b[4] - b[3]

        if local < sub4_end:
            progress = min(local / max(sub4_end - 1, 1), 1.0)
            width = T_MEASURED_SPIN90_OLD + (T_MEASURED_SPIN90_NEW - T_MEASURED_SPIN90_OLD) * progress
            spin_bar.set_width(width)
            spin_label.set_position((width + T_MEASURED_SPIN90_OLD * 0.02, 0.0))
            if progress >= 1.0:
                spin_label.set_text(
                    f"{T_MEASURED_SPIN90_NEW:.3f} s（限界の{ratio_new:.2f}倍。"
                    f"従来は{T_MEASURED_SPIN90_OLD:.2f} s = {ratio_old:.1f}倍）")
            else:
                spin_label.set_text(f"{width:.3f} s")
            table_title.set_visible(False)
            for t in header_artists:
                t.set_visible(False)
            for row in row_artists:
                for a in row:
                    a.set_visible(False)
            return ()

        spin_bar.set_width(T_MEASURED_SPIN90_NEW)
        spin_label.set_position((T_MEASURED_SPIN90_NEW + T_MEASURED_SPIN90_OLD * 0.02, 0.0))
        spin_label.set_text(
            f"{T_MEASURED_SPIN90_NEW:.3f} s（限界の{ratio_new:.2f}倍。"
            f"従来は{T_MEASURED_SPIN90_OLD:.2f} s = {ratio_old:.1f}倍）")

        table_title.set_visible(True)
        for t in header_artists:
            t.set_visible(True)
        local5 = local - sub4_end
        n_sub5 = max(n_view - sub4_end, 1)
        n_rows_show = 1 if local5 < 0.5 * n_sub5 else 2
        for k, row in enumerate(row_artists):
            show = k < n_rows_show
            for a in row:
                a.set_visible(show)
        return ()

    # ==== View C: 残っている壁（文6+文7） ====
    zero_stat = fig.text(0.5, 0.55, "", ha="center", va="center", color=C.C_MEASURED,
                          fontsize=64, fontweight="bold")
    zero_sub = fig.text(0.5, 0.42, "", ha="center", va="center", color=C.FG, fontsize=22)

    ax_reach = fig.add_axes([0.10, 0.56, 0.80, 0.18])
    C.style_axes(ax_reach)
    ax_reach.set_xlim(0.0, plan_len_m * 1.05)
    ax_reach.set_ylim(-0.8, 0.8)
    ax_reach.set_yticks([])
    ax_reach.set_xlabel("経路上の距離 [m]（maze_41004・摩擦円の使用率 u=0.30）", fontsize=14)
    reach_bar = mpatches.Rectangle((0.0, -0.30), 0.0, 0.60, facecolor=C.C_LIMIT,
                                    edgecolor=C.C_LIMIT, alpha=0.55, linewidth=1.5)
    ax_reach.add_patch(reach_bar)
    reach_full_label = ax_reach.text(plan_len_m, 0.55, f"計画 {plan_len_m:.2f} m",
                                      color=C.C_LIMIT, fontsize=14, ha="right", va="center")
    collide_mark, = ax_reach.plot([], [], marker="x", markersize=16, mew=3, color=C.C_MEASURED,
                                   zorder=5)
    collide_label = ax_reach.text(0.0, 0.0, "", color=C.C_MEASURED, fontsize=15, ha="left",
                                   va="bottom")
    collide_label.set_visible(False)
    reach_full_label.set_visible(False)

    corridor_w = 2.0 * HALF_WIDTH + 2.0 * room_m
    ax_cor = fig.add_axes([0.20, 0.20, 0.60, 0.26])
    C.style_axes(ax_cor)
    ax_cor.set_xlim(-corridor_w * 1000.0 * 0.6, corridor_w * 1000.0 * 0.6)
    ax_cor.set_ylim(-0.6, 0.6)
    ax_cor.set_yticks([])
    ax_cor.set_xlabel("通路の断面 [mm]（機体中心の左右のずれ）", fontsize=14)
    wall_left = mpatches.Rectangle((-corridor_w * 1000.0 / 2.0 - 8.0, -0.5), 8.0, 1.0,
                                    facecolor=C.GRID, edgecolor=C.FG, linewidth=1.0)
    wall_right = mpatches.Rectangle((corridor_w * 1000.0 / 2.0, -0.5), 8.0, 1.0,
                                     facecolor=C.GRID, edgecolor=C.FG, linewidth=1.0)
    ax_cor.add_patch(wall_left)
    ax_cor.add_patch(wall_right)
    robot_w_mm = HALF_WIDTH * 1000.0
    gap_center = (robot_w_mm + corridor_w * 1000.0 / 2.0) / 2.0
    room_left_txt = ax_cor.text(-gap_center, 0.46, "", color=C.FG, fontsize=14, ha="center")
    room_right_txt = ax_cor.text(gap_center, 0.46, "", color=C.FG, fontsize=14, ha="center")
    robot_patch = mpatches.Rectangle((-robot_w_mm, -0.32), 2 * robot_w_mm, 0.64,
                                      facecolor=C.C_LIMIT, edgecolor=C.C_LIMIT, alpha=0.75,
                                      linewidth=1.5)
    ax_cor.add_patch(robot_patch)
    drift_note = fig.text(0.5, 0.10, "", ha="center", va="center", color=C.FG, fontsize=20,
                           fontweight="bold")

    def draw_view_c(i: int):
        local = i - b[5]
        n_view = b[7] - b[5]
        sub6_end = b[6] - b[5]

        if local < sub6_end:
            zero_stat.set_visible(True)
            zero_sub.set_visible(True)
            zero_stat.set_text(f"{N_GOAL_EXP027} / {N_MAZES_EXP027}")
            zero_sub.set_text("完走できた迷路（design_turn_v1・exp_027・追従方式）")
            ax_reach.set_visible(False)
            ax_cor.set_visible(False)
            drift_note.set_visible(False)
            return ()

        zero_stat.set_visible(False)
        zero_sub.set_visible(False)
        ax_reach.set_visible(True)
        ax_cor.set_visible(True)
        drift_note.set_visible(True)

        local7 = local - sub6_end
        n_sub7 = max(n_view - sub6_end, 1)
        progress = min(local7 / max(n_sub7 - 1, 1), 1.0)

        # 前半: 計画に沿って進む → 衝突点で止まる。後半: 通路断面でずれて壁に当たる。
        if progress < 0.5:
            p = progress / 0.5
            reach_bar.set_width(reach_m * p)
            reach_full_label.set_visible(False)
            collide_label.set_visible(False)
            collide_mark.set_data([], [])
            room_left_txt.set_text("")
            room_right_txt.set_text("")
            robot_patch.set_x(-robot_w_mm)
            drift_note.set_text("")
        else:
            reach_bar.set_width(reach_m)
            reach_full_label.set_visible(True)
            collide_mark.set_data([reach_m], [0.0])
            collide_label.set_visible(True)
            collide_label.set_position((reach_m, -0.55))
            collide_label.set_text(f"衝突 {reach_m:.2f} m（計画の{reach_pct:.0f}%）")

            room_left_txt.set_text(f"{room_mm:.0f} mm")
            room_right_txt.set_text(f"{room_mm:.0f} mm")
            p2 = (progress - 0.5) / 0.5
            drift_mm = room_mm * p2
            robot_patch.set_x(-robot_w_mm + drift_mm)
            drift_note.set_text(
                f"推測航法のずれが {room_mm:.0f} mm の余地を使い切ると壁に当たる")
        return ()

    # ==== View D: 締め（文8） ====
    closing_recap = fig.text(0.5, 0.60, "", ha="center", va="center", color=C.FG, fontsize=22)
    closing_text = fig.text(0.5, 0.42, "", ha="center", va="center", color=C.FG, fontsize=30,
                             fontweight="bold", linespacing=1.6)

    def draw_view_d(i: int):
        local = i - b[7]
        n_view = max(n_active - b[7], 1)
        progress = min(local / max(n_view - 1, 1), 1.0)
        closing_recap.set_visible(True)
        closing_recap.set_text(
            f"要素はほぼ物理限界に肉薄（旋回{ratio_new:.2f}倍・弧{RATIO_ARC_CONNECTED:.2f}倍・"
            f"直線{RATIO_LONG_STRAIGHT:.2f}倍）。ただし通しでは{N_GOAL_EXP027}/{N_MAZES_EXP027}。")
        if progress >= 0.35:
            closing_text.set_text("どこまで肉薄できるのか。\nそれは第2部で。")
        else:
            closing_text.set_text("")
        return ()

    def hide_all():
        intro_big.set_visible(False)
        intro_sub.set_visible(False)
        ax_vs.set_visible(False)
        time_note.set_text("")
        ax_bar.set_visible(False)
        table_title.set_visible(False)
        for t in header_artists:
            t.set_visible(False)
        for row in row_artists:
            for a in row:
                a.set_visible(False)
        zero_stat.set_visible(False)
        zero_sub.set_visible(False)
        ax_reach.set_visible(False)
        ax_cor.set_visible(False)
        drift_note.set_visible(False)
        closing_recap.set_visible(False)
        closing_text.set_text("")

    def draw_frame(i: int):
        hide_all()
        if i < b[3]:
            return draw_view_a(i)
        if i < b[5]:
            return draw_view_b(i)
        if i < b[7]:
            return draw_view_c(i)
        return draw_view_d(i)

    out_path = C.OUT_DIR / "clip_10_rebuilt.mp4"
    C.render_clip(fig, draw_frame, n_active, out_path)
    print(f"saved: {out_path}")
    print(
        f"t_limit_spin90={t_limit_spin90:.6f} ratio_old={ratio_old:.4f} "
        f"ratio_new={ratio_new:.4f} t_cmd_total={t_cmd_total:.6f} "
        f"t_cont_total={t_cont_total:.6f} t_plan_arc={t_plan_arc:.6f} "
        f"t_measured_arc={t_measured_arc:.6f} t_plan_long={t_plan_long:.6f} "
        f"t_measured_long={t_measured_long:.6f} v_max_long={r_long.v_max:.6f} "
        f"room_mm={room_mm:.6f} reach_m={reach_m:.6f} plan_len_m={plan_len_m:.6f} "
        f"reach_pct={reach_pct:.4f}"
    )


if __name__ == "__main__":
    main()
