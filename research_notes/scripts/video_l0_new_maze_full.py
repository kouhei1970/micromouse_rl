# L0-a（古典ベースライン AdachiPolicy・超信地旋回走行「区画ごと停止」方式）の
# 新16x16評価迷路（規定準拠版）フル録画
#
# Full-length, real-time (1x), no-skip recording of the classical L0-a baseline
# (AdachiPolicy, flood-fill + 超信地旋回走行「区画ごと停止」) navigating a new
# NTF-compliant 16x16 evaluation maze through all 5 runs (探索走行 -> 帰還 ->
# 2〜5走行目で最短走行が収束するまで) を省略なく等倍速で録画する。
#
# 画面レイアウト: 1920x1080（16:9）。左1080x1080 = 迷路俯瞰（重畳物なし）、
# 右840x1080 = 情報パネル（既知壁地図 / 走行状態 / 数値計器 / 直近10秒の速度
# 時系列グラフ）。詳細は各セクションの docstring 参照。
#
# 用語統一: 走行方式は「超信地旋回走行（区画ごと停止）」と呼ぶ（英語の
# "stop-and-go" は使わない）。画面には常時「L0-a 超信地旋回走行・区画ごと停止」
# を表示する（L0-b・L0-c との比較用）。
#
# 使い方 / Usage:
#   .venv/bin/python research_notes/scripts/video_l0_new_maze_full.py
#
# --------------------------------------------------------------------------
# タイミングを competition/evaluator.py と完全一致させる方法（重要）
# --------------------------------------------------------------------------
# 走行タイム（run_time）・走行境界（t_start/t_end）は、本スクリプト側で
# FREE/RUN_ACTIVE 状態機械を再実装するのではなく、competition/evaluator.py の
# CompetitionEvaluator.evaluate_maze() を実際にそのまま呼び出し、その内部で
# 呼ばれる方策フック（on_run_start/on_run_end）の発火タイミングで
# sim.sim_time を読み取ることで求める（RecordingPolicyWrapper）。
#
# evaluate_maze() 内部では、あるステップで run_time 確定 → 同一ステップ内で
# policy.on_run_end(outcome) 呼び出し、という順序が保証されており
# （sim.data.time はその間変化しない）、on_run_start/on_run_end 発火時に
# 読み取った sim.sim_time は評価器自身が使う t_start/t_end と厳密に同一になる。
# こうすることで、状態機械の再実装によるロジックの食い違い（実際に
# 2026-08-10 に発生: 自前再実装版は公式記録と 0.04〜0.19 秒ずれた）を
# 構造的に排除する。
#
# なお 2026-08-10 17:36 頃、competition/baseline_classical.py 側で観測ベクトル
# の距離センサ本数依存インデックス計算が修正され（本数を sim.params.sensors
# から動的導出）、mouse/params.py の既定センサ本数変更（r6, 6→4本）との
# 不整合は解消済み。よって本スクリプトは既定 RobotParams()（センサ4本）を
# そのまま使い、competition/mazes/eval/ の既存XMLをそのまま読む
# （XML再生成・センサ本数の回避策は不要になった）。
import argparse
import os
import sys
import time
from collections import deque

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_REPO)
sys.path.insert(0, _REPO)

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import mujoco  # noqa: E402
import imageio  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from competition.evaluator import CompetitionEvaluator, DEFAULT_CELL_SIZE  # noqa: E402
from competition.baseline_classical import AdachiPolicy  # noqa: E402
from competition.policy_interface import MousePolicy  # noqa: E402
from mouse.params import RobotParams  # noqa: E402


# ==========================================================================
# 設定
# ==========================================================================
# 面の選定: competition/mazes/eval/ の20面（規定準拠・maze_gen_v2生成）を
# L0-a + 420秒プロトコルで全面走査した結果、maze_1015（seed=1015）が
#   探索走行(第1走行) 82.73s -> 最短走行(第4走行) 25.29s（短縮率 69.4%）
# と20面中で最大の短縮率、かつ全5走行が衝突・スタック・転倒による係員回収
# なしで成立（incidents=[]）だったため選定（competition/results/
# adachi_classical_20260810_173850/maze_1015.json で追試確認済み）。
# 次点は maze_1017（55.3%）・maze_1004（55.0%）・maze_1000（45.3%）。
SELECTED_MAZE_ID = "maze_1015"
EVAL_MAZE_DIR = Path("competition/mazes/eval")  # 読み取り専用
MAX_RUNS = 5
TIME_BUDGET_S = 420.0

OUT_VIDEO_PATH = Path("outputs/videos/l0a_full.mp4")
METHOD_LABEL = "L0-a 超信地旋回走行・区画ごと停止"

# --- レイアウト: 1920x1080、左=迷路俯瞰（正方形）、右=情報パネル ---
MAZE_SIZE = 1080
RIGHT_WIDTH = 840
OUT_WIDTH = MAZE_SIZE + RIGHT_WIDTH   # 1920
OUT_HEIGHT = 1080
FPS = 30
FRAME_DT = 1.0 / FPS
CAMERA_DISTANCE = 4.6         # video_l0_run.py で実測検証済みの値を踏襲
HOLD_FRAMES_AT_END = 90       # 5走行終了後の静止保持フレーム数（3秒 @30fps）
GRAPH_WINDOW_S = 10.0         # 速度グラフの表示窓 [s]

# 右パネル内の縦方向セクション高さ配分（合計 OUT_HEIGHT=1080）
SEC1_H = 420   # 既知壁地図（330x330グリッド + 2行凡例）
SEC2_H = 260   # 走行状態・確定タイム
SEC3_H = 180   # 数値計器（v, omega, 電圧, 距離センサ）
SEC4_H = OUT_HEIGHT - SEC1_H - SEC2_H - SEC3_H  # 速度時系列グラフ（220）

# --- 色 ---
COLOR_EXPLORE = (66, 133, 244, 255)     # 第1走行（探索）の軌跡色（青）
COLOR_RETURN = (189, 189, 189, 255)     # 帰還中の軌跡色（灰）
COLOR_LATER_RUN = (255, 159, 10, 255)   # 第2走行以降（暫定）の軌跡色（橙）
COLOR_BEST_RUN = (52, 199, 89, 255)     # その時点の最速走行の軌跡色（緑）
COLOR_ROBOT_DOT = (255, 59, 48, 255)
COLOR_WALL_KNOWN = (10, 10, 10, 255)
COLOR_WALL_UNKNOWN = (190, 190, 190, 90)
COLOR_START = (52, 199, 89, 130)
COLOR_GOAL = (255, 149, 0, 130)
PANEL_BG = (24, 26, 30, 255)
SEC_BG_ALT = (32, 35, 40, 255)
TEXT_WHITE = (240, 240, 240, 255)
TEXT_DIM = (170, 170, 175, 255)
TEXT_ACCENT = (255, 214, 10, 255)
BAR_BG = (60, 63, 68, 255)
BAR_FG = (10, 132, 255, 255)


def _load_font(size: int):
    candidates = [
        ("/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", 0),
        ("/System/Library/Fonts/Menlo.ttc", 0),
        ("/System/Library/Fonts/SFNSMono.ttf", 0),
    ]
    for path, index in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size, index=index)
    return ImageFont.load_default()


def make_camera(width: int, height: int, cell_size: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [width * cell_size / 2, height * cell_size / 2, 0]
    cam.azimuth = 90
    cam.elevation = -90
    cam.distance = CAMERA_DISTANCE
    return cam


# ==========================================================================
# 既知壁地図パネル（未探索=薄い / 既知=濃い）+ 走行軌跡
# 座標系: world(0,0)=左下、world(max,max)=右上（実測検証済み。video_l0_run系
# と同じ真上俯瞰カメラの向きに合わせ、画像y軸のみ反転する）。
# ==========================================================================
class WallMapPanel:
    """走行軌跡は「その時点で最速と判明している走行が緑」という動的な色分けを
    行う（ある走行が最速かどうかはその走行が終わってみないと分からないため、
    走行中にリアルタイムで色を確定することはできない）。そのため座標点は
    走行番号（run_index）ごとに分けて保持し、実際の色は render() 呼び出し時に
    現在判明している best_run_index を見て毎回決め直す（走行後に判明した
    最速走行の軌跡を遡って緑に塗り直せる）。"""

    def __init__(self, width: int, height: int, cell_size: float, size_px: int = 330):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.extent = width * cell_size
        self.size_px = size_px
        self.legend_font = _load_font(15)
        self.traj_by_run = {}   # run_index(int) -> [(px,py), ...]
        self.traj_return = []

    def world_to_px(self, x, y):
        return (x / self.extent) * self.size_px, self.size_px - (y / self.extent) * self.size_px

    def add_point(self, phase, run_index, x, y):
        pt = self.world_to_px(x, y)
        if phase == "active":
            self.traj_by_run.setdefault(run_index, []).append(pt)
        elif phase == "return":
            self.traj_return.append(pt)

    def render(self, v_walls_known, h_walls_known, robot_xy, best_run_index=None) -> Image.Image:
        legend_h = 76  # 凡例2行分（1行あたり2項目、幅が狭いパネルでも見切れないように）
        img = Image.new("RGBA", (self.size_px, self.size_px + legend_h), (255, 255, 255, 235))
        draw = ImageDraw.Draw(img)

        gx0, gx1 = self.width // 2 - 1, self.width // 2
        gy0, gy1 = self.height // 2 - 1, self.height // 2
        x0, y0 = self.world_to_px(gx0 * self.cell_size, (gy1 + 1) * self.cell_size)
        x1, y1 = self.world_to_px((gx1 + 1) * self.cell_size, gy0 * self.cell_size)
        draw.rectangle([x0, y0, x1, y1], fill=COLOR_GOAL)
        sx0, sy0 = self.world_to_px(0.0, self.cell_size)
        sx1, sy1 = self.world_to_px(self.cell_size, 0.0)
        draw.rectangle([sx0, sy0, sx1, sy1], fill=COLOR_START)

        for gx in range(self.width + 1):
            for gy in range(self.height):
                val = int(v_walls_known[gx, gy])
                wx = gx * self.cell_size
                p0, p1 = self.world_to_px(wx, gy * self.cell_size), self.world_to_px(wx, (gy + 1) * self.cell_size)
                if val == 1:
                    draw.line([p0, p1], fill=COLOR_WALL_KNOWN, width=3)
                elif val == -1:
                    draw.line([p0, p1], fill=COLOR_WALL_UNKNOWN, width=1)
        for gx in range(self.width):
            for gy in range(self.height + 1):
                val = int(h_walls_known[gx, gy])
                wy = gy * self.cell_size
                p0 = self.world_to_px(gx * self.cell_size, wy)
                p1 = self.world_to_px((gx + 1) * self.cell_size, wy)
                if val == 1:
                    draw.line([p0, p1], fill=COLOR_WALL_KNOWN, width=3)
                elif val == -1:
                    draw.line([p0, p1], fill=COLOR_WALL_UNKNOWN, width=1)

        if len(self.traj_return) >= 2:
            draw.line(self.traj_return, fill=COLOR_RETURN, width=4, joint="curve")
        # 最速走行（分かっていれば）を最後に描いて最前面にする
        for run_index in sorted(self.traj_by_run.keys()):
            if run_index == best_run_index:
                continue
            pts = self.traj_by_run[run_index]
            if len(pts) < 2:
                continue
            color = COLOR_EXPLORE if run_index == 1 else COLOR_LATER_RUN
            draw.line(pts, fill=color, width=4, joint="curve")
        if best_run_index is not None and len(self.traj_by_run.get(best_run_index, [])) >= 2:
            draw.line(self.traj_by_run[best_run_index], fill=COLOR_BEST_RUN, width=5, joint="curve")

        rx, ry = self.world_to_px(robot_xy[0], robot_xy[1])
        r = 6
        draw.ellipse([rx - r, ry - r, rx + r, ry + r], fill=COLOR_ROBOT_DOT, outline=(255, 255, 255, 255), width=1)
        draw.rectangle([0, 0, self.size_px - 1, self.size_px - 1], outline=(60, 60, 60, 255), width=2)

        # 凡例: 2行×2項目（幅330pxでも「その時点の最速」まで見切れないように行を分ける）
        rows = [
            [(COLOR_EXPLORE, "第1走行"), (COLOR_RETURN, "帰還")],
            [(COLOR_LATER_RUN, "第2走行以降"), (COLOR_BEST_RUN, "その時点の最速")],
        ]
        for row_i, row in enumerate(rows):
            legend_y = self.size_px + 4 + row_i * 26
            lx = 4
            for color, label in row:
                draw.rectangle([lx, legend_y + 3, lx + 14, legend_y + 14], fill=color)
                draw.text((lx + 19, legend_y - 1), label, font=self.legend_font, fill=(20, 20, 20, 255))
                lx = draw.textbbox((lx + 19, legend_y - 1), label, font=self.legend_font)[2] + 16
        return img


# ==========================================================================
# 方策ラッパー: competition/evaluator.py の CompetitionEvaluator.evaluate_maze()
# を実際にそのまま駆動し、走行境界タイミングをフックから取得することで
# タイミング食い違いを構造的に排除する（本ファイル冒頭docstring参照）。
# ==========================================================================
class RecordingPolicyWrapper(MousePolicy):
    name = "adachi_classical"
    requires_privileged = True

    def __init__(self, inner, frame_cb, run_event_cb):
        self.inner = inner
        self.frame_cb = frame_cb
        self.run_event_cb = run_event_cb
        self.sim = None
        self.run_count = 0
        self.state = "FREE"

    def bind_sim(self, sim):
        self.sim = sim
        self.inner.bind_sim(sim)

    def bind_maze(self, v_walls, h_walls):
        self.inner.bind_maze(v_walls, h_walls)

    def on_maze_start(self, maze_info):
        self.run_count = 0
        self.state = "FREE"
        self.inner.on_maze_start(maze_info)
        self.run_event_cb("maze_start", 0, self.sim.sim_time, None)

    def on_run_start(self, run_index):
        self.run_count = run_index
        self.state = "RUN_ACTIVE"
        self.run_event_cb("run_start", run_index, self.sim.sim_time, None)
        self.inner.on_run_start(run_index)

    def on_run_end(self, outcome):
        self.run_event_cb("run_end", self.run_count, self.sim.sim_time, outcome)
        self.state = "FREE"
        self.inner.on_run_end(outcome)

    def on_retrieval(self):
        self.inner.on_retrieval()

    def act(self, obs):
        vl, vr = self.inner.act(obs)
        self.frame_cb(self.sim, obs, self.run_count, self.state, vl, vr,
                      self.inner.v_walls_known, self.inner.h_walls_known)
        return vl, vr


# ==========================================================================
# 数値計器パネル（v, omega, 左右モータ電圧, 距離センサ4本）
# ==========================================================================
def render_gauges_panel(width, height, v, omega, vl, vr, ranges_mm, sensor_names,
                         font, font_small) -> Image.Image:
    img = Image.new("RGBA", (width, height), SEC_BG_ALT)
    draw = ImageDraw.Draw(img)

    # --- 左半分: センサ配置図（機体を上から見た模式図。前方=上） ---
    diag_w = int(width * 0.36)
    cx, cy = diag_w // 2, height // 2
    body_w, body_h = 70, 100
    draw.rounded_rectangle([cx - body_w // 2, cy - body_h // 2, cx + body_w // 2, cy + body_h // 2],
                            radius=14, outline=(120, 120, 125, 255), width=2)
    draw.polygon([(cx, cy - body_h // 2 - 14), (cx - 12, cy - body_h // 2 + 6),
                  (cx + 12, cy - body_h // 2 + 6)], fill=(120, 120, 125, 255))  # 前方矢印

    positions = {
        "LF": (cx - 34, cy - 46), "RF": (cx + 34, cy - 46),
        "LS": (cx - 46, cy + 6), "RS": (cx + 46, cy + 6),
    }
    for name in sensor_names:
        if name not in positions:
            continue
        px, py = positions[name]
        val_mm = ranges_mm.get(name)
        txt = f"{name}\n{val_mm:4.0f}mm" if val_mm is not None else f"{name}\n---"
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(255, 214, 10, 255))
        for i, line in enumerate(txt.split("\n")):
            draw.text((px - 20, py + 10 + i * 16), line, font=font_small, fill=TEXT_WHITE)

    # --- 右半分: v / omega / 左右電圧 のバー表示 ---
    bx0 = diag_w + 16
    bar_w = width - bx0 - 20
    rows = [
        ("v", f"{v:+.2f} m/s", max(-1.0, min(1.0, v / 0.35)), True),
        ("ω", f"{omega:+.2f} rad/s", max(-1.0, min(1.0, omega / 8.0)), True),
        ("V_L", f"{vl:+.2f} V", max(-1.0, min(1.0, vl / 3.0)), True),
        ("V_R", f"{vr:+.2f} V", max(-1.0, min(1.0, vr / 3.0)), True),
    ]
    row_h = height // len(rows)
    for i, (label, txt, frac, bipolar) in enumerate(rows):
        ry = i * row_h + row_h // 2
        draw.text((bx0, ry - 26), f"{label}  {txt}", font=font_small, fill=TEXT_WHITE)
        bar_y0, bar_y1 = ry - 2, ry + 12
        draw.rectangle([bx0, bar_y0, bx0 + bar_w, bar_y1], fill=BAR_BG)
        mid = bx0 + bar_w // 2
        if bipolar:
            half = bar_w // 2
            fill_w = int(abs(frac) * half)
            if frac >= 0:
                draw.rectangle([mid, bar_y0, mid + fill_w, bar_y1], fill=BAR_FG)
            else:
                draw.rectangle([mid - fill_w, bar_y0, mid, bar_y1], fill=BAR_FG)
            draw.line([(mid, bar_y0 - 2), (mid, bar_y1 + 2)], fill=(200, 200, 200, 255), width=1)
        else:
            draw.rectangle([bx0, bar_y0, bx0 + int(frac * bar_w), bar_y1], fill=BAR_FG)

    return img


# ==========================================================================
# 速度時系列グラフ（直近 GRAPH_WINDOW_S 秒）
# --------------------------------------------------------------------------
# L0-a は区画ごとに停止するため加減速が繰り返され鋸歯状になる。L0-b/L0-c
# （後日制作）との走行方式比較で最も違いが分かりやすい図になる想定。
# ==========================================================================
def render_velocity_graph(width, height, history, v_max, font_small) -> Image.Image:
    img = Image.new("RGBA", (width, height), PANEL_BG)
    draw = ImageDraw.Draw(img)
    pad_l, pad_r, pad_t, pad_b = 54, 16, 28, 30
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    y_max = v_max * 1.3

    draw.text((16, 6), "速度 v の時系列（直近10秒）", font=font_small, fill=TEXT_WHITE)

    # 軸・グリッド
    draw.rectangle([pad_l, pad_t, pad_l + plot_w, pad_t + plot_h], outline=(90, 90, 95, 255), width=1)
    for frac, label in [(0.0, "0"), (0.5, f"{y_max/2:.2f}"), (1.0, f"{y_max:.2f}")]:
        gy = pad_t + plot_h - frac * plot_h
        draw.line([(pad_l, gy), (pad_l + plot_w, gy)], fill=(55, 58, 63, 255), width=1)
        draw.text((4, gy - 7), label, font=font_small, fill=TEXT_DIM)
    # v_max 基準線
    vmax_y = pad_t + plot_h - (v_max / y_max) * plot_h
    draw.line([(pad_l, vmax_y), (pad_l + plot_w, vmax_y)], fill=(255, 159, 10, 160), width=1)

    if len(history) >= 2:
        t_now = history[-1][0]
        pts = []
        for t, v in history:
            frac_x = 1.0 - min(max((t_now - t) / GRAPH_WINDOW_S, 0.0), 1.0)
            frac_y = min(max(v / y_max, 0.0), 1.0)
            pts.append((pad_l + frac_x * plot_w, pad_t + plot_h - frac_y * plot_h))
        draw.line(pts, fill=BAR_FG, width=2, joint="curve")

    draw.text((pad_l, height - 18), "-10s", font=font_small, fill=TEXT_DIM)
    draw.text((pad_l + plot_w - 16, height - 18), "now", font=font_small, fill=TEXT_DIM)
    return img


# ==========================================================================
# 右パネル全体（既知壁地図 / 走行状態 / 数値計器 / 速度グラフ）を合成
# ==========================================================================
def render_right_panel(ctx, wall_map, v, omega, vl, vr, ranges_mm) -> Image.Image:
    panel = Image.new("RGBA", (RIGHT_WIDTH, OUT_HEIGHT), PANEL_BG)
    draw = ImageDraw.Draw(panel)
    y = 0

    # --- セクション1: 既知壁地図 ---
    best_idx = ctx["best"][1] if ctx["best"] else None
    mm_img = wall_map.render(ctx["v_known"], ctx["h_known"], ctx["robot_xy"], best_run_index=best_idx)
    mx = (RIGHT_WIDTH - mm_img.width) // 2
    panel.alpha_composite(mm_img, dest=(mx, y + 8))
    y += SEC1_H
    draw.line([(0, y), (RIGHT_WIDTH, y)], fill=(70, 72, 78, 255), width=2)

    # --- セクション2: 走行状態・確定タイム ---
    ty = y + 14
    draw.text((16, ty), METHOD_LABEL, font=ctx["font_title"], fill=TEXT_ACCENT)
    ty += 32
    draw.text((16, ty), f"持ち時間  {ctx['t']:6.1f} / {TIME_BUDGET_S:.1f} s",
              font=ctx["font_body"], fill=TEXT_WHITE)
    ty += 28
    draw.text((16, ty), ctx["state_label"], font=ctx["font_body"], fill=TEXT_WHITE)
    ty += 30
    for label, rt in ctx["confirmed_times"]:
        draw.text((16, ty), f"{label}  {rt:.2f} s", font=ctx["font_small"], fill=TEXT_DIM)
        ty += 24
    if ctx["best"] is not None:
        best_t, best_idx = ctx["best"]
        ty += 4
        draw.text((16, ty), f"最速 {best_t:.2f} s（第{best_idx}走行）",
                  font=ctx["font_body"], fill=(52, 199, 89, 255))
    y += SEC2_H
    draw.line([(0, y), (RIGHT_WIDTH, y)], fill=(70, 72, 78, 255), width=2)

    # --- セクション3: 数値計器 ---
    gauges = render_gauges_panel(RIGHT_WIDTH, SEC3_H, v, omega, vl, vr, ranges_mm,
                                  ctx["sensor_names"], ctx["font_body"], ctx["font_small"])
    panel.alpha_composite(gauges, dest=(0, y))
    y += SEC3_H
    draw.line([(0, y), (RIGHT_WIDTH, y)], fill=(70, 72, 78, 255), width=2)

    # --- セクション4: 速度時系列グラフ ---
    graph = render_velocity_graph(RIGHT_WIDTH, SEC4_H, ctx["v_history"], ctx["v_max"], ctx["font_small"])
    panel.alpha_composite(graph, dest=(0, y))

    return panel


# ==========================================================================
# メイン録画
# ==========================================================================
def run_and_record(maze_id: str) -> dict:
    npz_path = EVAL_MAZE_DIR / f"{maze_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} が見つかりません")
    data_npz = np.load(npz_path)
    width, height = int(data_npz["width"]), int(data_npz["height"])
    cell_size = DEFAULT_CELL_SIZE

    OUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(OUT_VIDEO_PATH), fps=FPS, macro_block_size=1)

    ctx = {
        "font_title": _load_font(30), "font_body": _load_font(24), "font_small": _load_font(18),
        "next_frame_time": 0.0, "n_frames": 0,
        "confirmed_times": [], "best": None, "run_start_t": {},
        "v_history": deque(), "v_max": RobotParams().v_max if hasattr(RobotParams(), "v_max") else 0.3,
        "state_label": "スタート待機中", "t": 0.0,
        "v_known": None, "h_known": None, "robot_xy": (0.09, 0.09),
        "sensor_names": [s["name"] for s in RobotParams().sensors],
        "renderer": None, "cam": None, "wall_map": None,
        "timeline": [], "step_i": 0, "wall_clock_start": time.time(),
    }

    def state_text(run_count, state):
        if state == "FREE" and run_count == 0:
            return "スタート待機中"
        if state == "RUN_ACTIVE" and run_count == 1:
            return "第1走行（探索中）"
        if state == "RUN_ACTIVE":
            return f"第{run_count}走行"
        return "帰還中"

    def frame_cb(sim, obs, run_count, state, vl, vr, v_known, h_known):
        t = sim.sim_time
        x, y, _yaw = sim.privileged_pose()
        v_fwd, omega_z = sim.privileged_velocity()
        ctx["v_history"].append((t, abs(v_fwd)))
        while ctx["v_history"] and ctx["v_history"][0][0] < t - GRAPH_WINDOW_S - 0.5:
            ctx["v_history"].popleft()

        ctx["t"] = t
        ctx["v_known"], ctx["h_known"] = v_known, h_known
        ctx["robot_xy"] = (x, y)
        ctx["state_label"] = state_text(run_count, state)

        # 走行軌跡: 走行中は run_count 別に座標を積む（色は render() 側で
        # その時点の最速走行番号から動的に決める。WallMapPanel の docstring参照）
        if state == "RUN_ACTIVE":
            ctx["wall_map"].add_point("active", run_count, x, y)
        elif state == "FREE" and run_count >= 1:
            ctx["wall_map"].add_point("return", run_count, x, y)

        if t >= ctx["next_frame_time"]:
            n = ctx["sensor_names"]
            ranges_mm = {name: float(obs[i]) * 1000.0 for i, name in enumerate(n)}
            ctx["renderer"].update_scene(sim.data, camera=ctx["cam"])
            left = ctx["renderer"].render()
            right = render_right_panel(ctx, ctx["wall_map"], v_fwd, omega_z, vl, vr, ranges_mm)

            canvas = Image.new("RGB", (OUT_WIDTH, OUT_HEIGHT))
            canvas.paste(Image.fromarray(left), (0, 0))
            canvas.paste(right.convert("RGB"), (MAZE_SIZE, 0))
            writer.append_data(np.array(canvas))
            ctx["n_frames"] += 1
            ctx["next_frame_time"] += FRAME_DT

        ctx["step_i"] += 1
        if ctx["step_i"] % 1500 == 0:
            elapsed_wall = time.time() - ctx["wall_clock_start"]
            print(f"  ... sim_time={t:6.2f}s  run_count={run_count}  state={state:10s}  "
                  f"frames={ctx['n_frames']:5d}  経過(実時間)={elapsed_wall:6.1f}s")

    def run_event_cb(kind, run_index, sim_time, outcome):
        if kind == "maze_start":
            return
        if kind == "run_start":
            ctx["run_start_t"][run_index] = sim_time
            ctx["timeline"].append({"event": "run_start", "run": run_index, "t": sim_time})
            print(f"  [t={sim_time:7.2f}s] 第{run_index}走行 出発")
            return
        # run_end
        t_start = ctx["run_start_t"].get(run_index)
        run_time = (sim_time - t_start) if t_start is not None else float("nan")
        ctx["timeline"].append({"event": "run_end", "run": run_index, "outcome": outcome,
                                 "t": sim_time, "run_time": run_time})
        if outcome == "goal":
            ctx["confirmed_times"].append((f"第{run_index}走行", run_time))
            if run_index >= 2 and (ctx["best"] is None or run_time < ctx["best"][0]):
                ctx["best"] = (run_time, run_index)
            best_txt = f"（最速 {ctx['best'][0]:.2f}s@第{ctx['best'][1]}走行）" if ctx["best"] else ""
            print(f"  [t={sim_time:7.2f}s] 第{run_index}走行 ゴール到達: {run_time:.2f} s {best_txt}")
        else:
            print(f"  [t={sim_time:7.2f}s] 第{run_index}走行 {outcome}（係員回収）")

    inner_policy = AdachiPolicy()
    wrapped = RecordingPolicyWrapper(inner_policy, frame_cb, run_event_cb)

    # レンダラ・カメラ・壁地図パネルは bind_sim/on_maze_start 経由で幾何情報が
    # 揃った時点（evaluate_maze の最初の act() より前）で用意する。
    _orig_bind_sim = wrapped.bind_sim

    def _bind_sim_and_setup(sim):
        _orig_bind_sim(sim)
        sim.model.vis.global_.offwidth = MAZE_SIZE
        sim.model.vis.global_.offheight = MAZE_SIZE
        ctx["renderer"] = mujoco.Renderer(sim.model, height=MAZE_SIZE, width=MAZE_SIZE)
        ctx["cam"] = make_camera(width, height, cell_size)
        ctx["wall_map"] = WallMapPanel(width, height, cell_size)

    wrapped.bind_sim = _bind_sim_and_setup

    print(f"迷路 {maze_id} (width={width}, height={height}) の録画を開始します。")
    print(f"出力先: {OUT_VIDEO_PATH.resolve()}")
    print("competition/evaluator.py の CompetitionEvaluator.evaluate_maze() をそのまま駆動します"
          "（タイミング完全一致のため）。")

    evaluator = CompetitionEvaluator(maze_dir=str(EVAL_MAZE_DIR), time_budget=TIME_BUDGET_S,
                                      max_runs=MAX_RUNS)
    result = evaluator.evaluate_maze(npz_path, wrapped)

    # --- 5走行終了後の締めフレーム ---
    last_run = result["runs"][-1] if result["runs"] else None
    remaining = TIME_BUDGET_S - (last_run["t_end"] if last_run else ctx["t"])
    ctx["state_label"] = f"{len(result['runs'])}走行終了（残り時間 {remaining:.1f} s）"
    n = ctx["sensor_names"]
    sim = wrapped.sim
    obs_final = sim.observation()
    ranges_mm = {name: float(obs_final[i]) * 1000.0 for i, name in enumerate(n)}
    v_fwd, omega_z = sim.privileged_velocity()
    x, y, _yaw = sim.privileged_pose()
    ctx["robot_xy"] = (x, y)
    ctx["renderer"].update_scene(sim.data, camera=ctx["cam"])
    left = ctx["renderer"].render()
    right = render_right_panel(ctx, ctx["wall_map"], v_fwd, omega_z, 0.0, 0.0, ranges_mm)
    canvas = Image.new("RGB", (OUT_WIDTH, OUT_HEIGHT))
    canvas.paste(Image.fromarray(left), (0, 0))
    canvas.paste(right.convert("RGB"), (MAZE_SIZE, 0))
    final_frame = np.array(canvas)
    for _ in range(HOLD_FRAMES_AT_END):
        writer.append_data(final_frame)
        ctx["n_frames"] += 1

    writer.close()
    ctx["renderer"].close()
    wall_clock = time.time() - ctx["wall_clock_start"]

    return {
        "maze_id": maze_id, "official_runs": result["runs"], "official_best_time": result["best_time"],
        "confirmed_times": ctx["confirmed_times"], "best": ctx["best"],
        "timeline": ctx["timeline"], "n_frames_written": ctx["n_frames"], "wall_clock_s": wall_clock,
    }


def main():
    parser = argparse.ArgumentParser(description="L0-a 新16x16評価迷路 フル録画（5走行・省略なし・等倍速）")
    parser.add_argument("--maze-id", type=str, default=SELECTED_MAZE_ID,
                         help=f"録画対象の迷路ID（既定: {SELECTED_MAZE_ID}）")
    args = parser.parse_args()

    print("=" * 70)
    print(f"{METHOD_LABEL}（AdachiPolicy） 新16x16評価迷路 フル録画（全5走行）")
    print("=" * 70)
    result = run_and_record(args.maze_id)

    print("\n--- 公式 evaluate_maze() の走行記録（本動画のタイム表示の正） ---")
    for r in result["official_runs"]:
        print(f"  走行{r['index']}: {r['outcome']:9s} t_start={r['t_start']:7.2f}s "
              f"t_end={r['t_end']:7.2f}s run_time={r['run_time']:6.2f}s")
    print(f"  best_time = {result['official_best_time']:.2f} s")

    print("\n--- コールバック計測値との整合確認 ---")
    for label, rt in result["confirmed_times"]:
        print(f"  {label}: {rt:.2f} s")
    if result["best"]:
        print(f"  最速: {result['best'][0]:.2f} s（第{result['best'][1]}走行）")

    print(f"\n書き出しフレーム数: {result['n_frames_written']} "
          f"（{result['n_frames_written'] / FPS:.1f} s @ {FPS}fps）")
    print(f"出力: {OUT_VIDEO_PATH.resolve()}")
    print(f"実処理時間（wall clock）: {result['wall_clock_s']:.1f} s")


if __name__ == "__main__":
    main()
