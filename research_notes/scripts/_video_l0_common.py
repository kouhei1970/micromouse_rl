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
import math
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
SEC1_H = 400   # 既知壁地図（正方形マップ + 凡例。地図辺長・凡例配置は動的計算）
# 注記行（extra_caption）や 5 走行ぶんの確定タイムが下のパネルへ被らないよう
# 余裕を持たせる（L0-b で注記 1 行が増えて最速行が隠れた不具合の是正）
SEC2_H = 290   # 走行状態・確定タイム
SEC3_H = 180   # 数値計器（v, omega, 電圧, 距離センサ）
SEC4_H = OUT_HEIGHT - SEC1_H - SEC2_H - SEC3_H  # 速度時系列グラフ（220）

# --- SEC1（既知壁地図パネル）レイアウト定数 ---
# 2026-08-10 是正: 凡例を「地図の下」から「地図横の余白」へ移設。
# 理由: SEC1 は幅840pxあるのに地図は330x330（横幅の半分未満）しか使って
# おらず、地図の左右に大きな余白を持て余していた（ユーザ指摘: 「マップの
# 両脇は空いているので、凡例などはそちらに配置してもいいのでは」）。加えて
# 「地図の下に凡例を積む」方式では、失敗走行が発生する方式（L0-b/L0-c）で
# 凡例が5項目に増えた際にSEC1の枠（高さ400px）をはみ出し、下のSEC2の方式名
# テキストと重なる不具合があった（L0-aは4項目だったのでたまたま収まって
# いただけ）。凡例を左右余白へ縦積みで逃がし、空いた縦方向の余地で地図を
# 正方形のまま拡大する。
WALLMAP_MARGIN_V = 20        # 地図の上下余白 [px]（これを残してSEC1の高さいっぱいまで地図を拡大）
WALLMAP_GAP = 16             # 地図と凡例の間の隙間 [px]
WALLMAP_OUTER_MARGIN = 16    # SEC1左右端の余白 [px]
# 凡例1行の (フォントサイズ, 行高, 色スウォッチの辺長) 候補。大きい順に試し、
# 項目数（3〜8個程度を想定するが、それ以外でも）が余白に収まる最初の候補を
# 採用する。これにより「項目数が増減しても崩れない」設計にする
# （必要高さは常に「行数×行高」から逆算し、座標を決め打ちしない）。
WALLMAP_LEGEND_ROW_CANDIDATES = [
    (15, 26, 14),
    (13, 23, 12),
    (12, 20, 11),
    (11, 18, 10),
    (10, 16, 9),
]

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
# 凡例レイアウト計算（地図横の余白へ縦積みで配置するための共通ロジック）
# --------------------------------------------------------------------------
# WallMapPanel.render() と、単体確認スクリプト
# _check_wallmap_layout.py の両方から同じ関数を使うことで、「項目数が
# 3〜8個のどれでも SEC1 の矩形からはみ出さない」ことを検証可能にする。
# ==========================================================================
def _text_w(draw, text, font):
    """指定フォントでのテキスト幅 [px] を返す（凡例の列幅計算に使う）。"""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _plan_legend_columns(draw, items, avail_h, avail_w):
    """凡例 items を幅avail_w x 高さavail_h の矩形へ収める配置を計算する。

    行高（≒フォントサイズ）を大きい候補から順に試し、1列で高さに収まる
    ものがあれば採用する。1列で幅が足りなければ列数を増やし、それでも幅が
    足りなければより小さいフォント候補へ切り替える。
    「行数×行高」から必要高さを逆算するだけで座標を決め打ちしないため、
    項目数が変わっても同じロジックで収まる配置を探索できる。
    収まる配置が見つからなければ None を返す（呼び出し側は別の余白を試すか
    最終手段の強制収容 _force_fit_legend へフォールバックする）。
    """
    n = len(items)
    if n == 0:
        return {"font": None, "font_size": 0, "row_h": 0, "sw": 0, "n_cols": 0,
                "max_rows": 0, "col_widths": [], "col_gap": 0, "label_gap": 0,
                "total_w": 0, "total_h": 0}
    col_gap, label_gap = 18, 6
    for font_size, row_h, sw in WALLMAP_LEGEND_ROW_CANDIDATES:
        if row_h > avail_h:
            continue
        font = _load_font(font_size)
        max_rows = max(1, int(avail_h // row_h))
        n_cols = math.ceil(n / max_rows)
        col_widths = []
        for c in range(n_cols):
            col_items = items[c * max_rows:(c + 1) * max_rows]
            w = max((_text_w(draw, label, font) for _, label in col_items), default=0)
            col_widths.append(sw + label_gap + w)
        total_w = sum(col_widths) + col_gap * (n_cols - 1)
        if total_w <= avail_w:
            rows_used = min(n, max_rows)
            return {"font": font, "font_size": font_size, "row_h": row_h, "sw": sw,
                    "n_cols": n_cols, "max_rows": max_rows, "col_widths": col_widths,
                    "col_gap": col_gap, "label_gap": label_gap,
                    "total_w": total_w, "total_h": rows_used * row_h}
    return None


def _force_fit_legend(draw, items, avail_h, avail_w):
    """想定を超えて項目数が多い等、通常のフォント候補では収まらない場合の
    最終手段。最小フォントで列数を増やし、それでも幅が足りないラベルは
    末尾を省略記号(…)で切り詰めてでも矩形内に強制収容する
    （「はみ出すくらいなら可読性を落とす」の方針。3〜8項目の想定範囲では
    _plan_legend_columns 側で必ず解決するため、通常は通らない経路）。
    """
    font_size, row_h, sw = WALLMAP_LEGEND_ROW_CANDIDATES[-1]
    font = _load_font(font_size)
    n = len(items)
    max_rows = max(1, int(avail_h // row_h))
    n_cols = max(1, math.ceil(n / max_rows))
    col_gap, label_gap = 8, 4
    col_w = max(sw + label_gap + 6, (avail_w - col_gap * (n_cols - 1)) // n_cols)
    text_budget = max(6, col_w - sw - label_gap)
    trimmed = []
    for color, label in items:
        s = label
        while s and _text_w(draw, s + "…", font) > text_budget:
            s = s[:-1]
        trimmed.append((color, (s + "…") if s and s != label else s))
    return {"font": font, "font_size": font_size, "row_h": row_h, "sw": sw,
            "n_cols": n_cols, "max_rows": max_rows, "col_widths": [col_w] * n_cols,
            "col_gap": col_gap, "label_gap": label_gap,
            "total_w": col_w * n_cols + col_gap * (n_cols - 1),
            "total_h": min(n, max_rows) * row_h, "items_override": trimmed}


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
    描き直す（過去の走行の軌跡も、後から判明した事実で塗り直せる）。

    地図は正方形を保ったまま SEC1 の高さいっぱいまで拡大し、凡例は地図横の
    余白（右優先・不足時は左）へ縦積みで配置する（2026-08-10 是正、詳細は
    WALLMAP_* 定数のコメントを参照）。"""

    def __init__(self, width: int, height: int, cell_size: float):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.extent = width * cell_size
        # 地図辺長は SEC1_H から動的に決める（旧: 固定330px）。
        # 上下に WALLMAP_MARGIN_V ずつ余白を残して正方形のまま拡大する。
        self.size_px = SEC1_H - 2 * WALLMAP_MARGIN_V
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
               explore_run_index=None, best_run_index=None, legend_items=None) -> Image.Image:
        """SEC1 の矩形（RIGHT_WIDTH x SEC1_H）とちょうど同じ大きさのキャンバス
        に描く。呼び出し側（render_right_panel）は dest=(0, y) に貼るだけで
        よく、これによりはみ出しの可能性をこの関数内の座標計算だけに
        閉じ込められる（呼び出し側の座標計算ミスでSEC1をはみ出す事故を防ぐ）。

        legend_items は通常 None（=標準の5項目）でよいが、単体確認スクリプト
        _check_wallmap_layout.py から任意の項目数を注入して、凡例レイアウトが
        3〜8項目のどれでも崩れないことを検証できるようにするための引数。
        """
        canvas_w, canvas_h = RIGHT_WIDTH, SEC1_H
        img = Image.new("RGBA", (canvas_w, canvas_h), PANEL_BG)
        draw = ImageDraw.Draw(img)

        # --- 地図: 正方形を保ったまま SEC1 の高さいっぱいまで拡大 ---
        # 凡例を横へ逃がして空いた分の余白を使う（是正の主眼）。
        map_x0 = (canvas_w - self.size_px) // 2
        map_y0 = WALLMAP_MARGIN_V

        def w2p(x, y):
            lx, ly = self.world_to_px(x, y)
            return lx + map_x0, ly + map_y0

        def shift(pts):
            return [(x + map_x0, y + map_y0) for x, y in pts]

        # 地図の背景（壁の黒線を見せるため明るい背景を維持。既存の配色を踏襲）
        draw.rectangle([map_x0, map_y0, map_x0 + self.size_px, map_y0 + self.size_px],
                        fill=(255, 255, 255, 235))

        gx0, gx1 = self.width // 2 - 1, self.width // 2
        gy0, gy1 = self.height // 2 - 1, self.height // 2
        x0, y0 = w2p(gx0 * self.cell_size, (gy1 + 1) * self.cell_size)
        x1, y1 = w2p((gx1 + 1) * self.cell_size, gy0 * self.cell_size)
        draw.rectangle([x0, y0, x1, y1], fill=COLOR_GOAL)
        sx0, sy0 = w2p(0.0, self.cell_size)
        sx1, sy1 = w2p(self.cell_size, 0.0)
        draw.rectangle([sx0, sy0, sx1, sy1], fill=COLOR_START)

        for gx in range(self.width + 1):
            for gy in range(self.height):
                val = int(v_walls_known[gx, gy])
                wx = gx * self.cell_size
                p0, p1 = w2p(wx, gy * self.cell_size), w2p(wx, (gy + 1) * self.cell_size)
                if val == 1:
                    draw.line([p0, p1], fill=COLOR_WALL_KNOWN, width=3)
                elif val == -1:
                    draw.line([p0, p1], fill=COLOR_WALL_UNKNOWN, width=1)
        for gx in range(self.width):
            for gy in range(self.height + 1):
                val = int(h_walls_known[gx, gy])
                wy = gy * self.cell_size
                p0 = w2p(gx * self.cell_size, wy)
                p1 = w2p((gx + 1) * self.cell_size, wy)
                if val == 1:
                    draw.line([p0, p1], fill=COLOR_WALL_KNOWN, width=3)
                elif val == -1:
                    draw.line([p0, p1], fill=COLOR_WALL_UNKNOWN, width=1)

        for seg in self.traj_return_segments:
            if len(seg) >= 2:
                draw.line(shift(seg), fill=COLOR_RETURN, width=4, joint="curve")

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
            draw.line(shift(pts), fill=color, width=4, joint="curve")
        if best_run_index is not None and len(self.traj_by_run.get(best_run_index, [])) >= 2:
            draw.line(shift(self.traj_by_run[best_run_index]), fill=COLOR_BEST_RUN, width=5, joint="curve")

        rx, ry = w2p(robot_xy[0], robot_xy[1])
        r = 6
        draw.ellipse([rx - r, ry - r, rx + r, ry + r], fill=COLOR_ROBOT_DOT, outline=(255, 255, 255, 255), width=1)
        draw.rectangle([map_x0, map_y0, map_x0 + self.size_px - 1, map_y0 + self.size_px - 1],
                        outline=(60, 60, 60, 255), width=2)

        # --- 凡例: 地図横の余白へ縦積みで配置（右優先、不足時は左） ---
        # （ユーザ指摘: 「マップの両脇は空いているので、凡例などはそちらに
        # 配置してもいいのでは」への対応。旧: 地図の下に配置し、5項目に増えた
        # 際にSEC1をはみ出してSEC2の方式名テキストと重なっていた）
        if legend_items is None:
            legend_items = [
                (COLOR_EXPLORE, "探索走行"),
                (COLOR_RETURN, "帰還"),
                (COLOR_LATER_RUN, "探索より後"),
                (COLOR_BEST_RUN, "その時点の最速"),
                (COLOR_FAILED_RUN, "失敗（係員回収）"),
            ]
        right_x0 = map_x0 + self.size_px + WALLMAP_GAP
        right_avail_w = canvas_w - WALLMAP_OUTER_MARGIN - right_x0
        left_avail_w = map_x0 - WALLMAP_GAP - WALLMAP_OUTER_MARGIN
        avail_h = self.size_px  # 地図と上下端を揃える

        plan = _plan_legend_columns(draw, legend_items, avail_h, right_avail_w)
        side = "right"
        if plan is None:
            plan = _plan_legend_columns(draw, legend_items, avail_h, left_avail_w)
            side = "left"
        if plan is None:
            # 通常の項目数（3〜8個程度）では上のいずれかで必ず収まる設計だが、
            # 想定外に項目数が多い場合の最終防波堤
            plan = _force_fit_legend(draw, legend_items, avail_h, right_avail_w)
            side = "right"

        if plan["n_cols"] > 0:
            lx0 = right_x0 if side == "right" else map_x0 - WALLMAP_GAP - plan["total_w"]
            ly0 = map_y0 + (self.size_px - plan["total_h"]) // 2  # 地図と縦方向中央を揃える

            draw_items = plan.get("items_override", legend_items)
            row_h, sw = plan["row_h"], plan["sw"]
            label_gap, max_rows = plan["label_gap"], plan["max_rows"]
            cx = lx0
            for c in range(plan["n_cols"]):
                col_items = draw_items[c * max_rows:(c + 1) * max_rows]
                for r, (color, label) in enumerate(col_items):
                    ry = ly0 + r * row_h
                    sy0 = ry + (row_h - sw) // 2
                    draw.rectangle([cx, sy0, cx + sw, sy0 + sw], fill=color)
                    ty = ry + (row_h - plan["font_size"]) // 2
                    draw.text((cx + sw + label_gap, ty), label, font=plan["font"], fill=TEXT_WHITE)
                cx += plan["col_widths"][c] + plan["col_gap"]

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
        # 🔴 描画のために**ラッパー自身**は sim を持つ（軌跡を描く「カメラ」であり、
        # 方策の入力ではない）。しかし **inner が特権を要求していないなら渡さない。**
        # 2026-08-19 是正: 従来は無条件に転送しており、requires_privileged=False の
        # 方策にも特権情報を差し出していた。実害は無かった（基底の受け口が空実装）が、
        # 「センサだけで走っている」という主張を**構成として**保証するために閉じる。
        self.sim = sim
        if getattr(self.inner, "requires_privileged", False):
            self.inner.bind_sim(sim)

    def bind_maze(self, v_walls, h_walls):
        # 同上。真の壁情報も、要求していない方策には渡さない。
        if getattr(self.inner, "requires_privileged", False):
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
    diag_w = int(width * 0.42)
    # 機体図は左端の値ラベルが切れないよう右へ寄せつつ、右側ラベルが
    # 状態量パネル（v/ω/V_L/V_R）へ食い込まない幅に収める
    # （L0-b で右ラベルがパネルと重なった不具合の是正）
    cx, cy = diag_w // 2 + 10, height // 2
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
        # 値ラベルは機体図の**外側**へ寄せる（中央揃えだと左右のラベルが接触して
        # 読めなくなる。L0-a 版で「235mmRF 231mm」と重なった不具合の是正）
        if px < cx:
            tx = px - tw - 12          # 左側センサ: ラベルを左外側へ
        else:
            tx = px + 12               # 右側センサ: ラベルを右外側へ
        draw.text((tx, py - 8), txt, font=font_small, fill=TEXT_WHITE)

    # --- 右側: v / omega / 左右電圧 のバー表示 ---
    bx0 = diag_w + 28
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
    # wall_map.render() は SEC1 と同じ (RIGHT_WIDTH x SEC1_H) を返すため、
    # dest=(0, y) にそのまま貼るだけでSEC1の矩形に厳密に一致する
    # （はみ出しの可能性は render() 内の座標計算に閉じ込められている）。
    best_idx = ctx["best"][1] if ctx["best"] else None
    mm_img = wall_map.render(ctx["v_known"], ctx["h_known"], ctx["robot_xy"],
                              explore_run_index=ctx["explore_run_index"], best_run_index=best_idx)
    panel.alpha_composite(mm_img, dest=(0, y))
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
                      time_budget: float = TIME_BUDGET_S, v_max_for_graph: float = 0.3,
                      maze_dir: Path | None = None) -> dict:
    """policy（requires_privileged=True の方策インスタンス）で maze_id を
    5走行（既定）分フル録画する。タイミングは competition/evaluator.py の
    CompetitionEvaluator.evaluate_maze() をそのまま駆動して取得する
    （本ファイル冒頭docstring参照）。失敗走行（stuck等）も省略せず記録する。
    """
    maze_dir = Path(maze_dir) if maze_dir is not None else EVAL_MAZE_DIR
    npz_path = maze_dir / f"{maze_id}.npz"
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

    evaluator = CompetitionEvaluator(maze_dir=str(maze_dir), time_budget=time_budget,
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
