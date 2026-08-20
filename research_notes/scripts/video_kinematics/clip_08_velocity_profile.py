# research_notes/scripts/video_kinematics/clip_08_velocity_profile.py
# クリップ8: 速度プロファイル — 短い直線と長い直線（約20秒）
#
# classic/profile.py の min_time（読み取り専用の import）で 4 通りの速度プロファイルを
# その場で計算し、順に切り替えて見せる。横軸は弧長 s、縦軸は速度。
# 加速／定常／減速を塗り分ける（区間の分類は IdealTime.by_mode と同じ判定則を
# 表示用グリッドに適用する）。
#
# 実行: .venv/bin/python research_notes/scripts/video_kinematics/clip_08_velocity_profile.py
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classic.profile import (  # noqa: E402 読み取り専用の import
    vehicle_limits, min_time, Segment, _MODE_EPS,
)

import _common as C  # noqa: E402


CASES = [
    dict(label="1. 直交1区画（0.18 m・上限なし）", length=0.180, v_cap=None),
    dict(label="2. 直交1区画（0.18 m・上限 0.12 m/s）", length=0.180, v_cap=0.12),
    dict(label="3. 直交16区画（2.88 m・上限なし）", length=2.880, v_cap=None),
    dict(label="4. 斜め15段（3.818 m・規約上の最長）", length=3.818, v_cap=None),
]

def classify_modes(s_grid, v_grid):
    """`classic/profile.py` の `_summarize_segment` と同じ判定則（`_MODE_EPS` による
    隣接差の符号）で、格子の各区間を加速／定常／減速に分類する。

    表示用に間引くと近似的な平坦部（実際には定常ではない緩やかな肩）を「定常」と
    誤って塗ってしまい、`by_mode` の数値（定常=0.000 等）と矛盾する。ここでは
    間引かず、計算そのものに使った細かい格子（ds=0.0005m）に対して、
    `min_time` の内部と全く同じ判定則を適用する（数値と描画を一致させる）。
    """
    modes = []
    for i in range(len(v_grid) - 1):
        dv = v_grid[i + 1] - v_grid[i]
        if dv > _MODE_EPS:
            modes.append("accel")
        elif dv < -_MODE_EPS:
            modes.append("decel")
        else:
            modes.append("cruise")
    return modes


def build_case(label, length, v_cap, limits):
    res = min_time([Segment(length=length, curvature=0.0, kind="straight")],
                    limits, v_cap=v_cap)
    modes = classify_modes(res.s_grid, res.v_grid)

    thresh = 0.95 * limits.V_TOP
    frac95 = 0.0
    for i in range(len(res.s_grid) - 1):
        v0, v1 = res.v_grid[i], res.v_grid[i + 1]
        if v0 >= thresh and v1 >= thresh:
            frac95 += res.s_grid[i + 1] - res.s_grid[i]
    frac95_pct = frac95 / res.path_length * 100.0

    return dict(label=label, length=length, v_cap=v_cap, res=res,
                s_disp=res.s_grid, v_disp=res.v_grid, modes=modes,
                frac95_pct=frac95_pct, is_capped=(v_cap is not None))


def main() -> None:
    C.setup_style()
    limits = vehicle_limits()
    V_TOP = limits.V_TOP

    cases = [build_case(c["label"], c["length"], c["v_cap"], limits) for c in CASES]

    total_seconds = 20.0
    n_active = C.seconds_to_active_frames(total_seconds)
    n_cases = len(cases)
    # フェーズ境界（フレーム数を4分割。端数は前方のフェーズへ配る）
    bounds = [round(n_active * k / n_cases) for k in range(n_cases + 1)]
    sweep_frac = 0.78  # フェーズ内でマーカーが動く割合。残りは完成形を保持する

    fig = C.new_figure()
    ax = fig.add_axes([0.09, 0.17, 0.80, 0.64])
    C.style_axes(ax)

    C.add_title(fig, "短い直線と長い直線 — 速度プロファイル")
    C.add_caption(fig, "台形が出るかどうかは直線の長さで決まる。", y=0.045)

    y_top = V_TOP * 1.08

    def draw_frame(i: int):
        # 現在のフェーズと、フェーズ内の進み具合を求める
        case_idx = n_cases - 1
        for k in range(n_cases):
            if bounds[k] <= i < bounds[k + 1]:
                case_idx = k
                break
        phase_len = max(bounds[case_idx + 1] - bounds[case_idx], 1)
        local_i = i - bounds[case_idx]
        progress = min(local_i / (phase_len * sweep_frac), 1.0)

        case = cases[case_idx]
        res = case["res"]
        length = case["length"]
        base_color = C.C_CAP_COST if case["is_capped"] else C.C_LIMIT

        ax.clear()
        C.style_axes(ax)
        ax.set_xlim(0.0, length * 1.02)
        ax.set_ylim(0.0, y_top)
        ax.set_xlabel("弧長 s [m]", fontsize=16)
        ax.set_ylabel("速度 v [m/s]", fontsize=16)
        ax.axhline(V_TOP, color=C.C_LIMIT, linewidth=1.2, linestyle="--", alpha=0.5)
        ax.text(length * 1.005, V_TOP, f"V_TOP={V_TOP:.3f} m/s", color=C.C_LIMIT,
                fontsize=13, ha="left", va="center")

        # 計算に使った格子そのもの（間引きなし）のうち、進捗ぶんだけを塗り分けて描く
        n_grid = len(case["s_disp"]) - 1
        n_show = max(int(progress * n_grid), 1)
        s_show = case["s_disp"][:n_show + 1]
        v_show = case["v_disp"][:n_show + 1]
        modes_show = case["modes"][:n_show]

        alpha_map = {"accel": 0.35, "decel": 0.35, "cruise": 0.68}
        seg_start = 0
        for k in range(1, len(modes_show) + 1):
            if k == len(modes_show) or modes_show[k] != modes_show[seg_start]:
                sl = slice(seg_start, k + 1)
                ax.fill_between(s_show[sl], 0.0, v_show[sl], color=base_color,
                                 alpha=alpha_map[modes_show[seg_start]], linewidth=0)
                seg_start = k
        ax.plot(s_show, v_show, color=base_color, linewidth=2.6, zorder=3)

        # 現在位置のマーカー
        cur_s = s_show[-1]
        cur_v = v_show[-1]
        ax.plot([cur_s], [cur_v], marker="o", markersize=9, color=C.FG, zorder=4)

        ax.set_title(case["label"], color=C.FG, fontsize=22, pad=14)

        t_accel = res.by_mode.get("accel", (0.0, 0.0))
        t_cruise = res.by_mode.get("cruise", (0.0, 0.0))
        t_decel = res.by_mode.get("decel", (0.0, 0.0))
        stat_lines = [
            f"所要時間 = {res.total:.3f} s   経路長 = {res.path_length:.3f} m",
            f"最高速 v_peak = {res.v_max:.3f} m/s（V_TOPの{res.v_max / V_TOP * 100:.1f}%）",
            f"距離のうち v>=95%V_TOP: {case['frac95_pct']:.1f}%",
            f"加速: {t_accel[0]:.3f}s/{t_accel[1]:.3f}m   "
            f"定常: {t_cruise[0]:.3f}s/{t_cruise[1]:.3f}m   "
            f"減速: {t_decel[0]:.3f}s/{t_decel[1]:.3f}m",
            f"現在位置: s={cur_s:.3f} m   v={cur_v:.3f} m/s（{cur_v / V_TOP * 100:.1f}% V_TOP）",
        ]
        ax.text(0.02, 0.94, "\n".join(stat_lines), transform=ax.transAxes,
                color=C.FG, fontsize=15, va="top", ha="left", linespacing=1.7,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#1A1D1F",
                          edgecolor=C.GRID, linewidth=1.0))
        return ()

    out_path = C.OUT_DIR / "clip_08_velocity_profile.mp4"
    C.render_clip(fig, draw_frame, n_active, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
