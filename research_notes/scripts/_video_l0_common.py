# research_notes/scripts/_video_l0_common.py
# L0-a / L0-b / L0-c 共通のフル録画レンダリング基盤。
#
# 1920x1080（16:9）レイアウト: 左1080x1080=迷路俯瞰（重畳物なし）、
# 右840x1080=情報パネル（既知壁地図 / 走行状態・確定タイム / 数値計器 /
# 直近10秒の速度時系列グラフ）。方式（L0-a/b/c）ごとの差分は「どの方策を
# 渡すか」「画面上の方式名」「出力先」「補足キャプション（任意）」のみで、
# レンダリング・タイミング取得ロジックは完全に共有する（L0-a の検収で
# 見つかった表示崩れ3点の修正がL0-b以降にも自動的に反映されるように、
# 2026-08-10 に video_l0_new_maze_full.py から切り出した）。
#
# --------------------------------------------------------------------------
# タイミングを competition/evaluator.py と完全一致させる方法
# --------------------------------------------------------------------------
# 走行タイム（run_time）・走行境界（t_start/t_end）は、状態機械を自前で
# 再実装するのではなく、competition/evaluator.py の
# CompetitionEvaluator.evaluate_maze() を実際にそのまま呼び出し、その内部で
# 呼ばれる方策フック（on_run_start/on_run_end）の発火タイミングで
# sim.sim_time を読み取ることで求める（RecordingPolicyWrapper）。
# evaluate_maze() 内部では「run_time 確定 → 同一ステップ内で
# policy.on_run_end(outcome) 呼び出し」の順序が保証されており
# （sim.data.time はその間変化しない）、フック発火時に読み取った
# sim.sim_time は評価器自身が使う t_start/t_end と厳密に同一になる。
# 自前再実装は実際に公式記録と 0.04〜0.19 秒ずれる不具合を起こしたため
# （2026-08-10 検収時に発覚・本方式へ切替）、この方式を必須とする。
#
# 失敗走行（stuck/collision/tipover）の扱い: 隠さず記録する。確定タイム欄に
# 「第N走行（<理由>・係員回収）」として残し、既知壁地図の軌跡も専用の
# 赤色で描く（WallMapPanel.mark_failed）。
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont

from competition.evaluator import CompetitionEvaluator, DEFAULT_CELL_SIZE
from competition.policy_interface import MousePolicy
from mouse.params import RobotParams

EVAL_MAZE_DIR = Path("competition/mazes/eval")  # 読み取り専用
MAX_RUNS = 5
TIME_BUDGET_S = 420.0

# --- レイアウト: 1920x1080、左=迷路俯瞰（正方形）、右=情報パネル ---
MAZE_SIZE = 1080
RIGHT_WIDTH = 840
OUT_WIDTH = MAZE_SIZE + RIGHT_WIDTH   # 1920
OUT_HEIGHT = 1080
FPS = 30
FRAME_DT = 1.0 / FPS
CAMERA_DISTANCE = 4.6         # video_l0_run.py で実測検証済みの値を踏襲
HOLD_FRAMES_AT_END = 90       # 全走行終了後の静止保持フレーム数（3秒 @30fps）
GRAPH_WINDOW_S = 10.0         # 速度グラフの表示窓 [s]

# 右パネル内の縦方向セクション高さ配分（合計 OUT_HEIGHT=1080）
SEC1_H = 420   # 既知壁地図（330x330グリッド + 凡例）
SEC2_H = 260   # 走行状態・確定タイム
SEC3_H = 180   # 数値計器（v, omega, 電圧, 距離センサ）
SEC4_H = OUT_HEIGHT - SEC1_H - SEC2_H - SEC3_H  # 速度時系列グラフ（220）

# --- 色 ---
COLOR_EXPLORE = (66, 133, 244, 255)     # 探索走行（最初にゴールした走行）の軌跡色（青）
COLOR_RETURN = (189, 189, 189, 255)     # 帰還中の軌跡色（灰）
COLOR_LATER_RUN = (255, 159, 10, 255)   # 探索より後の走行（暫定）の軌跡色（橙）
COLOR_BEST_RUN = (52, 199, 89, 255)     # その時点の最速走行の軌跡色（緑）
COLOR_FAILED_RUN = (255, 69, 58, 255)   # 失敗（スタック等・係員回収）走行の軌跡色（赤）
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
TEXT_CAPTION = (255, 159, 10, 255)   # 補足キャプション（例: L0-bのv_max注記）
TEXT_FAIL = (255, 92, 82, 255)
BAR_BG = (60, 63, 68, 255)
BAR_FG = (10, 132, 255, 255)

_OUTCOME_JA = {"stuck": "スタック", "collision": "衝突", "tipover": "転倒", "timeout": "持ち時間切れ"}


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
# 座標系: world(0,0)=左下、world(max,max)=右上（実測検証済み）。
# ==========================================================================
class WallMapPanel:
    """走行軌跡の色は render() 呼び出し時にその時点で判明している
    explore_run_index（最初にゴールした走行）・best_run_index（それより後の
    走行のうち最速）・failed_runs（失敗して係員回収された走行）から動的に
    決める。ある走行が「最速」か「探索走行」かはその走行が終わるまで
    確定しないため、座標は走行番号ごとに分けて保持し、色は毎フレーム
    描き直す（過去の走行の軌跡も、後から判明した事実で塗り直せる）。"""

    def __init__(self, width: int, height: int, cell_size: float, size_px: int = 330):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.extent = width * cell_size
        self.size_px = size_px
        self.legend_font = _load_font(15)
        self.traj_by_run = {}   # run_index(int) -> [(px,py), ...]
        self.failed_runs = set()
        # 帰還（FREE）区間は複数回発生する。単一の平坦なリストに全部積むと、
        # ある帰還区間の終点から次の帰還区間の始点へ PIL が直線で
        # 「テレポート」して結んでしまう（実測確認済みのバグ）ため、
        # 区間ごとに別々のサブリストに分けて保持する。
        self.traj_return_segments = []  # [[(px,py), ...], ...]
        self._return_active = False

    def world_to_px(self, x, y):
        return (x / self.extent) * self.size_px, self.size_px - (y / self.extent) * self.size_px

    def add_point(self, phase, run_index, x, y):
        pt = self.world_to_px(x, y)
        if phase == "active":
            self.traj_by_run.setdefault(run_index, []).append(pt)
            self._return_active = False
        elif phase == "return":
            if not self._return_active:
                self.traj_return_segments.append([])
                self._return_active = True
            self.traj_return_segments[-1].append(pt)

    def mark_failed(self, run_index):
        self.failed_runs.add(run_index)

    def render(self, v_walls_known, h_walls_known, robot_xy,
               explore_run_index=None, best_run_index=None) -> Image.Image:
        legend_h = 96  # 凡例3行分（5色。幅330pxでも見切れないように行を分ける）
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

        for seg in self.traj_return_segments:
            if len(seg) >= 2:
                draw.line(seg, fill=COLOR_RETURN, width=4, joint="curve")

        # 最速走行（分かっていれば）を最後に描いて最前面にする
        for run_index in sorted(self.traj_by_run.keys()):
            if run_index == best_run_index:
                continue
            pts = self.traj_by_run[run_index]
            if len(pts) < 2:
                continue
            if run_index in self.failed_runs:
                color = COLOR_FAILED_RUN
            elif run_index == 1 or run_index == explore_run_index:
                color = COLOR_EXPLORE
            else:
                color = COLOR_LATER_RUN
            draw.line(pts, fill=color, width=4, joint="curve")
        if best_run_index is not None and len(self.traj_by_run.get(best_run_index, [])) >= 2:
            draw.line(self.traj_by_run[best_run_index], fill=COLOR_BEST_RUN, width=5, joint="curve")

        rx, ry = self.world_to_px(robot_xy[0], robot_xy[1])
        r = 6
        draw.ellipse([rx - r, ry - r, rx + r, ry + r], fill=COLOR_ROBOT_DOT, outline=(255, 255, 255, 255), width=1)
        draw.rectangle([0, 0, self.size_px - 1, self.size_px - 1], outline=(60, 60, 60, 255), width=2)

        # 凡例: 3行（幅330pxでも長いラベルが見切れないように行を分ける）
        rows = [
            [(COLOR_EXPLORE, "探索走行"), (COLOR_RETURN, "帰還")],
            [(COLOR_LATER_RUN, "探索より後"), (COLOR_BEST_RUN, "その時点の最速")],
            [(COLOR_FAILED_RUN, "失敗（係員回収）")],
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
# 方策ラッパー: CompetitionEvaluator.evaluate_maze() をそのまま駆動し、
# 走行境界タイミングをフックから取得する（本ファイル冒頭docstring参照）。
# ==========================================================================
class RecordingPolicyWrapper(MousePolicy):
    requires_privileged = True

    def __init__(self, inner, frame_cb, run_event_cb):
        self.inner = inner
        self.name = getattr(inner, "name", "unnamed")
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
# --------------------------------------------------------------------------
# 2026-08-10 検収で発覚した表示崩れ2点をここで修正:
#   (1) 最上段「v」の行がパネル上端で見切れていた → top_pad を確保
#   (2) LF/RF の値ラベルが接触して読みにくかった
#       → 各ラベルをテキスト幅を測って点の直下に中央寄せし、左右の点自体も
#         少し広げた
# ==========================================================================
def render_gauges_panel(width, height, v, omega, vl, vr, ranges_mm, sensor_names,
                         font, font_small) -> Image.Image:
    img = Image.new("RGBA", (width, height), SEC_BG_ALT)
    draw = ImageDraw.Draw(img)

    # --- 左側: センサ配置図（機体を上から見た模式図。前方=上） ---
    diag_w = int(width * 0.36)
    cx, cy = diag_w // 2, height // 2
    body_w, body_h = 70, 100
    draw.rounded_rectangle([cx - body_w // 2, cy - body_h // 2, cx + body_w // 2, cy + body_h // 2],
                            radius=14, outline=(120, 120, 125, 255), width=2)
    draw.polygon([(cx, cy - body_h // 2 - 14), (cx - 12, cy - body_h // 2 + 6),
                  (cx + 12, cy - body_h // 2 + 6)], fill=(120, 120, 125, 255))  # 前方矢印

    positions = {
        "LF": (cx - 40, cy - 46), "RF": (cx + 40, cy - 46),
        "LS": (cx - 52, cy + 6), "RS": (cx + 52, cy + 6),
    }
    for name in sensor_names:
        if name not in positions:
            continue
        px, py = positions[name]
        val_mm = ranges_mm.get(name)
        txt = f"{name} {val_mm:4.0f}mm" if val_mm is not None else f"{name} ---"
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(255, 214, 10, 255))
        bbox = draw.textbbox((0, 0), txt, font=font_small)
        tw = bbox[2] - bbox[0]
        draw.text((px - tw / 2, py + 10), txt, font=font_small, fill=TEXT_WHITE)

    # --- 右側: v / omega / 左右電圧 のバー表示 ---
    bx0 = diag_w + 16
    bar_w = width - bx0 - 20
    rows = [
        ("v", f"{v:+.2f} m/s", max(-1.0, min(1.0, v / 0.9)), True),
        ("ω", f"{omega:+.2f} rad/s", max(-1.0, min(1.0, omega / 8.0)), True),
        ("V_L", f"{vl:+.2f} V", max(-1.0, min(1.0, vl / 3.0)), True),
        ("V_R", f"{vr:+.2f} V", max(-1.0, min(1.0, vr / 3.0)), True),
    ]
    top_pad = 22  # (1) 最上段の見切れ修正: 上に余白を確保
    usable_h = height - top_pad
    row_h = usable_h // len(rows)
    for i, (label, txt, frac, bipolar) in enumerate(rows):
        ry = top_pad + i * row_h + row_h // 2
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
# との走行方式比較で違いが分かりやすい図になる。
# 2026-08-10 検収で発覚した表示崩れ1点をここで修正:
#   (3) 右下 "now" ラベルが下端で見切れていた → 下部余白を広げ、位置を
#       reserved な pad_b 領域の内側に収めた
# ==========================================================================
def render_velocity_graph(width, height, history, v_max, font_small) -> Image.Image:
    img = Image.new("RGBA", (width, height), PANEL_BG)
    draw = ImageDraw.Draw(img)
    pad_l, pad_r, pad_t, pad_b = 54, 16, 28, 40  # (3) pad_b を 30→40 に拡大
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    y_max = v_max * 1.3

    draw.text((16, 6), "速度 v の時系列（直近10秒）", font=font_small, fill=TEXT_WHITE)

    draw.rectangle([pad_l, pad_t, pad_l + plot_w, pad_t + plot_h], outline=(90, 90, 95, 255), width=1)
    for frac, label in [(0.0, "0"), (0.5, f"{y_max/2:.2f}"), (1.0, f"{y_max:.2f}")]:
        gy = pad_t + plot_h - frac * plot_h
        draw.line([(pad_l, gy), (pad_l + plot_w, gy)], fill=(55, 58, 63, 255), width=1)
        draw.text((4, gy - 7), label, font=font_small, fill=TEXT_DIM)
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

    label_y = pad_t + plot_h + 10  # (3) reserved pad_b(=40)の内側、軸のすぐ下
    draw.text((pad_l, label_y), "-10s", font=font_small, fill=TEXT_DIM)
    draw.text((pad_l + plot_w - 24, label_y), "now", font=font_small, fill=TEXT_DIM)
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
    mm_img = wall_map.render(ctx["v_known"], ctx["h_known"], ctx["robot_xy"],
                              explore_run_index=ctx["explore_run_index"], best_run_index=best_idx)
    mx = (RIGHT_WIDTH - mm_img.width) // 2
    panel.alpha_composite(mm_img, dest=(mx, y + 8))
    y += SEC1_H
    draw.line([(0, y), (RIGHT_WIDTH, y)], fill=(70, 72, 78, 255), width=2)

    # --- セクション2: 走行状態・確定タイム ---
    ty = y + 14
    draw.text((16, ty), ctx["method_label"], font=ctx["font_title"], fill=TEXT_ACCENT)
    ty += 32
    for cap in ctx.get("extra_caption", []):
        draw.text((16, ty), cap, font=ctx["font_small"], fill=TEXT_CAPTION)
        ty += 20
    draw.text((16, ty), f"持ち時間  {ctx['t']:6.1f} / {ctx['time_budget']:.1f} s",
              font=ctx["font_body"], fill=TEXT_WHITE)
    ty += 28
    draw.text((16, ty), ctx["state_label"], font=ctx["font_body"], fill=TEXT_WHITE)
    ty += 30
    for label, rt, is_fail in ctx["confirmed_times"]:
        color = TEXT_FAIL if is_fail else TEXT_DIM
        draw.text((16, ty), f"{label}  {rt:.2f} s", font=ctx["font_small"], fill=color)
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
# 汎用録画関数: L0-a/L0-b/L0-c いずれもこの1関数を policy・ラベル・出力先を
# 変えて呼ぶだけでよい。
# ==========================================================================
def record_run_video(policy, method_label: str, out_path: Path, maze_id: str,
                      extra_caption=None, max_runs: int = MAX_RUNS,
                      time_budget: float = TIME_BUDGET_S, v_max_for_graph: float = 0.3) -> dict:
    """policy（requires_privileged=True の方策インスタンス）で maze_id を
    5走行（既定）分フル録画する。タイミングは competition/evaluator.py の
    CompetitionEvaluator.evaluate_maze() をそのまま駆動して取得する
    （本ファイル冒頭docstring参照）。失敗走行（stuck等）も省略せず記録する。
    """
    npz_path = EVAL_MAZE_DIR / f"{maze_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} が見つかりません")
    data_npz = np.load(npz_path)
    width, height = int(data_npz["width"]), int(data_npz["height"])
    cell_size = DEFAULT_CELL_SIZE

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=FPS, macro_block_size=1)

    ctx = {
        "font_title": _load_font(28), "font_body": _load_font(24), "font_small": _load_font(18),
        "next_frame_time": 0.0, "n_frames": 0,
        "confirmed_times": [],   # [(label, run_time, is_fail), ...]
        "best": None, "explore_run_index": None, "run_start_t": {},
        "v_history": deque(), "v_max": v_max_for_graph,
        "state_label": "スタート待機中", "t": 0.0,
        "v_known": None, "h_known": None, "robot_xy": (0.09, 0.09),
        "sensor_names": [s["name"] for s in RobotParams().sensors],
        "renderer": None, "cam": None, "wall_map": None,
        "method_label": method_label, "time_budget": time_budget,
        "extra_caption": list(extra_caption) if extra_caption else [],
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
            if ctx["explore_run_index"] is None:
                ctx["explore_run_index"] = run_index
                ctx["confirmed_times"].append((f"第{run_index}走行", run_time, False))
                print(f"  [t={sim_time:7.2f}s] 第{run_index}走行（探索）ゴール到達: {run_time:.2f} s")
            else:
                ctx["confirmed_times"].append((f"第{run_index}走行", run_time, False))
                if ctx["best"] is None or run_time < ctx["best"][0]:
                    ctx["best"] = (run_time, run_index)
                print(f"  [t={sim_time:7.2f}s] 第{run_index}走行 ゴール到達: {run_time:.2f} s "
                      f"（最速 {ctx['best'][0]:.2f}s@第{ctx['best'][1]}走行）")
        else:
            outcome_ja = _OUTCOME_JA.get(outcome, outcome)
            ctx["confirmed_times"].append((f"第{run_index}走行（{outcome_ja}・係員回収）", run_time, True))
            ctx["wall_map"].mark_failed(run_index)
            print(f"  [t={sim_time:7.2f}s] 第{run_index}走行 {outcome_ja}（係員回収）: {run_time:.2f} s")

    wrapped = RecordingPolicyWrapper(policy, frame_cb, run_event_cb)

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
    print(f"出力先: {out_path.resolve()}")
    print("competition/evaluator.py の CompetitionEvaluator.evaluate_maze() をそのまま駆動します"
          "（タイミング完全一致のため）。")

    evaluator = CompetitionEvaluator(maze_dir=str(EVAL_MAZE_DIR), time_budget=time_budget,
                                      max_runs=max_runs)
    result = evaluator.evaluate_maze(npz_path, wrapped)

    # --- 全走行終了後の締めフレーム ---
    last_run = result["runs"][-1] if result["runs"] else None
    remaining = time_budget - (last_run["t_end"] if last_run else ctx["t"])
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
        "explore_run_index": ctx["explore_run_index"],
        "timeline": ctx["timeline"], "n_frames_written": ctx["n_frames"], "wall_clock_s": wall_clock,
        "out_path": out_path,
    }
