# research_notes/scripts/video_kinematics/clip_09_eta.py
# クリップ9: 物差し η — 物理限界からの肉薄度（約59.1秒）
#
# η = T_ideal / T_measured。T_ideal（物理限界の所要時間）は
# classic/profile.py・classic/ideal.py（読み取り専用の import）から計算する
# （直線1区画・その場90°旋回・maze_41001の最短走行の3通り）。
#
# T_measured（現行実装の実測）は classic/ を実際に走らせて得た値であり、本クリップ
# 単体では計算できない（`classic/` はモジュールとして純粋に物理限界を計算する層で、
# 実装の実測タイムは持たない）。README.md（正本の要約）・note_031（判定基準の
# 起草ノート）に記録済みの実測値をそのまま引用する（手打ちの物理定数ではなく、
# 実験で得た値の引用。出典はコメントに明記する）:
#   直線1区画       2.030 s  （README.md「現在地」表）
#   その場90°旋回   2.000 s  （同上）
#   最短走行1本      154.24 s （同上。maze_41001・exp_024の最短走行）
#
# 分解 154.2 = 9.17 x 2.17(ターン方式) x 4.23(速度上限) x 1.83(制御) の
# 2.17・4.23 倍率も、note_031 の値をそのまま使うのではなく、本クリップが
# classic/ideal.py の ideal_time_for_path（mode="spin"/"slalom"）と
# classic/profile.py の min_time（v_cap 付き）から改めて計算し直す
# （note_031 の値と一致することを実行時に確認できる。手打ちしない）。
#
# 台本（narration/clip_09_eta.txt）の切れ目で画面を3つの見せ方に切り替える:
#   文1+文2: η の定義（式） / 文3+文4: 要素ごとの表 / 文5+文6: 分解の帯グラフ
#
# 実行: .venv/bin/python research_notes/scripts/video_kinematics/clip_09_eta.py
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classic.ideal import ideal_time_for_path, true_shortest_path  # noqa: E402 読み取り専用
from classic.maze_map import Direction  # noqa: E402 読み取り専用の import
from classic.profile import (  # noqa: E402 読み取り専用の import
    Segment, min_time, spin_turn_time, vehicle_limits,
)

import matplotlib.patches as mpatches  # noqa: E402
import _common as C  # noqa: E402

MAZE_PATH = REPO_ROOT / "competition" / "mazes" / "design_turn_v1" / "maze_41001.npz"
START = (0, 0)
GOALS = [(7, 7), (7, 8), (8, 7), (8, 8)]

# ---- 実測（classic/ の外・実験で得た値。出典はモジュールdocstring参照） ----
T_MEASURED_STRAIGHT = 2.030
T_MEASURED_SPIN90 = 2.000
T_MEASURED_SHORTEST = 154.24


def compute_ideal_values():
    """T_ideal の3通りと、分解の3倍率を classic/profile.py・classic/ideal.py から計算する。"""
    limits = vehicle_limits()

    t_ideal_straight = min_time(
        [Segment(length=0.180, curvature=0.0, kind="straight")], limits).total
    t_ideal_spin90 = spin_turn_time(math.pi / 2.0, limits).time

    data = np.load(MAZE_PATH)
    v_walls, h_walls = data["v_walls"], data["h_walls"]
    path = true_shortest_path(v_walls, h_walls, START, GOALS, Direction.N)

    res_spin = ideal_time_for_path(path, v_walls, h_walls, Direction.N, mode="spin")
    res_greedy = ideal_time_for_path(path, v_walls, h_walls, Direction.N, mode="slalom",
                                      allocation="greedy")
    res_prop = ideal_time_for_path(path, v_walls, h_walls, Direction.N, mode="slalom",
                                    allocation="proportional")
    res_best = res_greedy if res_greedy.total <= res_prop.total else res_prop
    t_ideal_shortest = res_best.total

    turn_factor = res_spin.total / t_ideal_shortest

    vcap = 0.12
    straight_vcap_total = sum(
        min_time([seg], limits, v_cap=vcap).total for seg in res_spin.segments)
    spin_vcap_total = straight_vcap_total + res_spin.by_kind["spin"]
    vcap_factor = spin_vcap_total / res_spin.total

    control_factor = T_MEASURED_SHORTEST / spin_vcap_total

    return dict(
        t_ideal_straight=t_ideal_straight, t_ideal_spin90=t_ideal_spin90,
        t_ideal_shortest=t_ideal_shortest, turn_factor=turn_factor,
        vcap_factor=vcap_factor, control_factor=control_factor,
    )


def main() -> None:
    C.setup_style()
    vals = compute_ideal_values()
    t_i_straight = vals["t_ideal_straight"]
    t_i_spin = vals["t_ideal_spin90"]
    t_i_short = vals["t_ideal_shortest"]
    f_turn = vals["turn_factor"]
    f_vcap = vals["vcap_factor"]
    f_ctrl = vals["control_factor"]

    eta_straight = t_i_straight / T_MEASURED_STRAIGHT * 100.0
    eta_spin = t_i_spin / T_MEASURED_SPIN90 * 100.0
    eta_short = t_i_short / T_MEASURED_SHORTEST * 100.0

    # ---- 台本の切れ目（narration/clip_09_eta.txt。文3は2つの節に分ける） ----
    s1 = "これで物差しが作れます。物理限界の所要時間を、実測の所要時間で割る。これを肉薄度ηと呼びます。"
    s2 = "迷路が違えば直進距離もターンの回数も違うので、迷路ごとに計算し直します。"
    s3a = "現行の実装を測ると、直線1区画は0.36秒に対して2.03秒、"
    s3b = "その場90°旋回は0.173秒に対して2.00秒。"
    s4 = "最短走行1本で見ると、9.17秒に対して154.24秒。ηは5.9パーセントでした。"
    s5 = "そしてこの差は、2つの設計上の選択で説明がつきます。"
    s6 = "ターン方式が2.17倍、速度上限が4.23倍。制御そのものは1.83倍にすぎません。"

    # ナレーション実測 58.128s（ffprobe, 2026-08-20） + 余韻 1.0s。
    total_seconds = 59.128
    n_active = C.seconds_to_active_frames(total_seconds)
    b = C.stage_bounds([len(t) for t in (s1, s2, s3a, s3b, s4, s5, s6)], n_active)
    # b = [0, S1末, S2末, S3a末, S3b末, S4末, S5末, n_active(S6末)]

    fig = C.new_figure()
    C.add_title(fig, "物差し η — 物理限界からの肉薄度", y=0.95)

    # ==== View A: 定義式（文1+文2） ====
    formula_big = fig.text(0.5, 0.56, "", ha="center", va="center", color=C.FG,
                            fontsize=54, fontweight="bold")
    formula_sub = fig.text(0.5, 0.42, "", ha="center", va="center", color=C.FG,
                            fontsize=22)
    formula_note2 = fig.text(0.5, 0.30, "", ha="center", va="center", color=C.C_CAP_COST,
                              fontsize=20)

    # ==== View B: 要素ごとの表（文3+文4） ====
    table_rows = [
        ("直線 1区画（停止→停止）", t_i_straight, T_MEASURED_STRAIGHT, eta_straight),
        ("その場 90°旋回", t_i_spin, T_MEASURED_SPIN90, eta_spin),
        ("最短走行 1本（maze_41001）", t_i_short, T_MEASURED_SHORTEST, eta_short),
    ]
    col_x = [0.08, 0.47, 0.68, 0.90]
    header_y = 0.72
    row_dy = 0.11
    header_texts = ["要素", "T_ideal（物理限界）", "T_measured（実測）", "η"]
    header_colors = [C.FG, C.C_LIMIT, C.C_MEASURED, C.FG]
    table_header = []
    for x, htext, ha, col in zip(
        col_x, header_texts, ["left", "center", "center", "center"], header_colors,
    ):
        t = fig.text(x, header_y, htext, ha=ha, va="center", color=col,
                     fontsize=19, fontweight="bold")
        table_header.append(t)
    table_rule = mpatches.Rectangle((0.08, header_y - 0.045), 0.86, 0.003,
                                     transform=fig.transFigure, facecolor=C.GRID)
    fig.add_artist(table_rule)

    row_labels = []
    row_texts = []
    for k, (name, ti, tm, eta) in enumerate(table_rows):
        y = header_y - 0.045 - row_dy * (k + 1)
        lbl = fig.text(col_x[0], y, name, ha="left", va="center", color=C.FG, fontsize=21)
        v_ideal = fig.text(col_x[1], y, f"{ti:.3f} s", ha="center", va="center",
                            color=C.C_LIMIT, fontsize=21)
        v_measured = fig.text(col_x[2], y, f"{tm:.3f} s", ha="center", va="center",
                               color=C.C_MEASURED, fontsize=21)
        v_eta = fig.text(col_x[3], y, f"{eta:.1f}%", ha="center", va="center",
                          color=C.FG, fontsize=21, fontweight="bold")
        for artist in (lbl, v_ideal, v_measured, v_eta):
            artist.set_visible(False)
        row_labels.append(lbl)
        row_texts.append((v_ideal, v_measured, v_eta))
    for t in table_header:
        t.set_visible(False)
    table_rule.set_visible(False)

    # ==== View C: 分解の帯グラフ（文5+文6） ====
    lead_note = fig.text(0.5, 0.80, "", ha="center", va="center", color=C.FG,
                          fontsize=24, fontweight="bold")
    ax_bar = fig.add_axes([0.10, 0.20, 0.80, 0.40])
    C.style_axes(ax_bar)
    ax_bar.set_xlim(0.0, 1.0)
    total_len = t_i_short * f_turn * f_vcap * f_ctrl  # ≈ T_MEASURED_SHORTEST
    ax_bar.set_ylim(0.0, total_len * 1.05)
    ax_bar.set_xticks([])
    ax_bar.set_ylabel("所要時間 [s]", fontsize=15)

    bar_colors = [C.C_LIMIT, C.C_CAP_COST, C.C_MEASURED, C.FG]
    bar_labels = [
        f"T_ideal = {t_i_short:.2f} s",
        f"x {f_turn:.2f}（ターン方式）",
        f"x {f_vcap:.2f}（速度上限）",
        f"x {f_ctrl:.2f}（制御）",
    ]
    segs_top = [t_i_short, t_i_short * f_turn, t_i_short * f_turn * f_vcap, total_len]
    segs_bottom = [0.0] + segs_top[:-1]
    bars = []
    seg_texts = []
    for k in range(4):
        rect = mpatches.Rectangle((0.30, 0.0), 0.40, 0.0, facecolor=bar_colors[k],
                                   edgecolor=bar_colors[k], alpha=0.75, linewidth=1.5)
        ax_bar.add_patch(rect)
        rect.set_visible(False)
        bars.append(rect)
        txt = ax_bar.text(0.74, 0.0, "", ha="left", va="center", color=bar_colors[k],
                           fontsize=18)
        txt.set_visible(False)
        seg_texts.append(txt)
    total_label = fig.text(0.5, 0.145, "", ha="center", va="center", color=C.FG,
                            fontsize=24, fontweight="bold")

    def set_view(view: str):
        active = (view == "formula")
        formula_big.set_visible(active)
        formula_sub.set_visible(active)
        if not active:
            formula_note2.set_visible(False)

        active_b = (view == "table")
        for t in table_header:
            t.set_visible(active_b)
        table_rule.set_visible(active_b)

        active_c = (view == "bars")
        ax_bar.set_visible(active_c)
        lead_note.set_visible(active_c)
        if not active_c:
            total_label.set_visible(False)
        if not active_b:
            for lbl in row_labels:
                lbl.set_visible(False)
            for cols in row_texts:
                for a in cols:
                    a.set_visible(False)

    def draw_frame(i: int):
        if i < b[2]:
            set_view("formula")
            formula_big.set_text("η = T_ideal / T_measured")
            formula_sub.set_text("物理限界の所要時間を、実測の所要時間で割る")
            formula_note2.set_visible(i >= b[1])
            formula_note2.set_text("迷路ごとに計算し直す（直進距離もターン回数も迷路で変わる）")
            for lbl in row_labels:
                lbl.set_visible(False)
            for v in row_texts:
                for a in v:
                    a.set_visible(False)
            return ()

        if i < b[5]:
            set_view("table")
            formula_note2.set_visible(False)
            n_rows_show = 1 if i < b[3] else (2 if i < b[4] else 3)
            for k, (lbl, cols) in enumerate(zip(row_labels, row_texts)):
                show = k < n_rows_show
                lbl.set_visible(show)
                for a in cols:
                    a.set_visible(show)
            return ()

        set_view("bars")
        lead_note.set_text("この差は、2つの設計上の選択で説明がつきます")
        if i < b[6]:
            for rect in bars:
                rect.set_visible(False)
            for txt in seg_texts:
                txt.set_visible(False)
            total_label.set_visible(False)
            return ()

        # 文6: 帯グラフを1段ずつ積み上げる
        span = max(n_active - b[6] - 1, 1)
        local = i - b[6]
        n_seg_show = min(int(local / span * 4) + 1, 4)
        for k in range(4):
            rect = bars[k]
            txt = seg_texts[k]
            if k < n_seg_show:
                bottom = segs_bottom[k]
                top = segs_top[k]
                rect.set_y(bottom)
                rect.set_height(top - bottom)
                rect.set_visible(True)
                txt.set_position((0.74, (bottom + top) / 2.0))
                txt.set_text(bar_labels[k])
                txt.set_visible(True)
            else:
                rect.set_visible(False)
                txt.set_visible(False)
        if n_seg_show >= 4:
            total_label.set_text(
                f"{total_len:.2f} s = {t_i_short:.2f} x {f_turn:.2f} x "
                f"{f_vcap:.2f} x {f_ctrl:.2f}   (η = {eta_short:.1f}%)")
            total_label.set_visible(True)
        return ()

    out_path = C.OUT_DIR / "clip_09_eta.mp4"
    C.render_clip(fig, draw_frame, n_active, out_path)
    print(f"saved: {out_path}")
    print(
        f"t_ideal_straight={t_i_straight:.6f} t_ideal_spin90={t_i_spin:.6f} "
        f"t_ideal_shortest={t_i_short:.6f} turn_factor={f_turn:.6f} "
        f"vcap_factor={f_vcap:.6f} control_factor={f_ctrl:.6f} "
        f"eta_short={eta_short:.4f}%"
    )


if __name__ == "__main__":
    main()
