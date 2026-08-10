# L0-a（古典ベースライン AdachiPolicy・超信地旋回走行「区画ごと停止」方式）の
# 新16x16評価迷路（規定準拠版）フル録画
# Full-length, real-time (1x), no-skip recording of the classical L0-a baseline
# (AdachiPolicy, flood-fill + 超信地旋回走行「区画ごと停止」/ stop-and-turn per
# cell) navigating a new NTF-compliant 16x16 evaluation maze from探索走行
# (exploration) -> 帰還 (autonomous return) -> 最短走行 (fastest run) までを
# 省略なく等倍速で録画する。
#
# 用語統一: 本ロボットの走行方式は「超信地旋回走行（区画ごと停止）」と呼ぶ
# （英語の "stop-and-go" は文中・画面表示のいずれでも使わない。2026-08-10
# 教授経由のユーザ要望）。画面には常時「L0-a 超信地旋回走行・区画ごと停止」を
# 表示し、他方式（L0-b・L0-c、後日制作予定）との比較時に何を見ているか
# 分かるようにする。
#
# 使い方 / Usage:
#   .venv/bin/python research_notes/scripts/video_l0_new_maze_full.py
#
# ベース: research_notes/scripts/video_l0_run.py のカメラ設定・PIL日本語重畳・
# 評価器 (competition/evaluator.py) と同じ FREE/RUN_ACTIVE 状態機械の駆動方法を
# そのまま踏襲する。座標判定・スタック検出は evaluator.py の関数をそのまま
# import して使い、ロジックの食い違いを防ぐ。
#
# --------------------------------------------------------------------------
# センサ本数の既知の不整合について（重要・re-run 時の注意）
# --------------------------------------------------------------------------
# 2026-08-10 17:05 のコミット e041bd8（mouse/params.py の既定センサ本数を
# 6→4 に変更）以降、competition/policy_interface.py が文書化する観測契約
# 「14次元 = 距離センサ6 + 加速度3 + 角速度3 + 車輪角速度2」と食い違い、
# competition/baseline_classical.py の AdachiPolicy がハードコードしている
# obs[11]（gyro_z）/obs[12]（左車輪角速度）/obs[13]（右車輪角速度）が
# 既定設定（センサ4本 → 観測12次元）では IndexError で即クラッシュする
# （実測確認済み）。competition/・mouse/ 配下は読み取り専用のため、
# この不整合はここでは修正しない。本スクリプトは mouse.params.SENSORS_6
# （params.py に「アブレーション再現用」として明示的に残されている、
# サポートされた構成オプション）を明示的に指定することで既存ファイルに
# 一切手を入れずに回避する。回避により観測次元が AdachiPolicy 設計時の
# 前提（14次元）に一致し、2026-08-10 12:39-13:54 に採取された既存の
# L0 公式結果（300秒プロトコル版）と同じ挙動が再現される
# （実測: maze_1004 で 探索64.28s / 最短28.92〜28.99s、既存3件の結果と整合）。
import argparse
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_REPO)
sys.path.insert(0, _REPO)

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import mujoco  # noqa: E402
import imageio  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from competition.evaluator import (  # noqa: E402
    in_start_region, in_goal_region, pos_to_cell, goal_cells,
    PositionRingBuffer,
    DEFAULT_STUCK_WINDOW_S, DEFAULT_STUCK_DISP_M, DEFAULT_CELL_SIZE,
)
from mouse.params import RobotParams, SENSORS_6  # noqa: E402
from mouse.sim import MouseSim  # noqa: E402
from mouse.mjcf import build_maze_robot_xml  # noqa: E402
from competition.baseline_classical import AdachiPolicy  # noqa: E402


# ==========================================================================
# 設定
# ==========================================================================
# 面の選定: competition/mazes/eval/ の20面（規定準拠・maze_gen_v2生成）を
# L0 + 420秒プロトコルでバックグラウンド走査（本スクリプト作成過程の一部として
# 別途実施。competition/results/ に420秒プロトコル版の既存結果が無かったため
# 自前で20面フル走査した）した結果、maze_1015（seed=1015）が
#   探索走行 82.92s -> 最短走行 25.38s（短縮率 69.4%）
# と、20面中で最大の短縮率かつ探索・最短ともに一発成功（衝突・スタック・転倒に
# よる係員回収なし、incidents=[]）だったため選定した。次点は maze_1017（55.3%）・
# maze_1004（55.0%）・maze_1000（45.3%）で、いずれも maze_1015 の 69.4% を下回る。
SELECTED_MAZE_ID = "maze_1015"
EVAL_MAZE_DIR = Path("competition/mazes/eval")  # 読み取り専用（npzのみ使用）

# センサ回避策（上のdocstring参照）で使う派生XMLの出力先。
# competition/ 配下は変更しないため、成果物は outputs/ 配下にキャッシュする。
XML_CACHE_DIR = Path("outputs/videos/_cache")
# 方式名を含む命名（l0a = L0-a）。同一迷路で L0-b・L0-c 版を後日制作する際に
# 混同しないため（2026-08-10 教授経由のユーザ要望で l0_new_maze_full.mp4 から改名）。
OUT_VIDEO_PATH = Path("outputs/videos/l0a_full.mp4")
METHOD_LABEL = "L0-a 超信地旋回走行・区画ごと停止"

FRAME_SIZE = 1280            # 出力フレーム一辺 [px]
FPS = 30
FRAME_DT = 1.0 / FPS
CAMERA_DISTANCE = 4.6        # video_l0_run.py で実測検証済みの値を踏襲（16x16全体が収まる）
TIME_BUDGET_S = 420.0        # NTF規定3-6準拠の持ち時間（competition/evaluator.py既定と同じ）
SAFETY_TIME_LIMIT_S = 420.0  # 安全弁: 持ち時間を超えたら異常とみなし打ち切る
HOLD_FRAMES_AT_END = 90      # 最短走行ゴール到達後の静止保持フレーム数（3秒 @30fps）

# --- 走行軌跡・地図ミニマップのパネル設定 ---
MINIMAP_SIZE = 380            # ミニマップ描画領域の一辺 [px]
MINIMAP_MARGIN = 28            # フレーム端からの余白 [px]
COLOR_EXPLORE = (66, 133, 244, 255)     # 探索走行の軌跡色（青）
COLOR_RETURN = (189, 189, 189, 255)     # 帰還中の軌跡色（灰）
COLOR_FASTEST = (52, 199, 89, 255)      # 最短走行の軌跡色（緑）
COLOR_ROBOT_DOT = (255, 59, 48, 255)    # 現在位置マーカー（赤）
COLOR_WALL_KNOWN = (20, 20, 20, 255)    # 既知の壁（濃い）
COLOR_WALL_UNKNOWN = (150, 150, 150, 110)  # 未知の壁（薄い）
COLOR_START = (52, 199, 89, 130)        # スタート区画ハイライト
COLOR_GOAL = (255, 149, 0, 130)         # ゴール区画ハイライト


def _load_font(size: int):
    """視認性重視のフォントを読み込む。等幅系（Menlo/SFNSMono）優先、
    無ければヒラギノ角ゴシック W4（index=0）にフォールバックする。"""
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
    """迷路全体が常に入る固定の真上俯瞰カメラを作る（video_l0_run.py と同一設定）。"""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cx = width * cell_size / 2
    cy = height * cell_size / 2
    cam.lookat[:] = [cx, cy, 0]
    cam.azimuth = 90
    cam.elevation = -90
    cam.distance = CAMERA_DISTANCE
    return cam


# ==========================================================================
# ミニマップ（既知壁マップ + 走行軌跡）
# --------------------------------------------------------------------------
# メインの俯瞰映像は物理的な迷路壁を常に完全な形でレンダリングしてしまう
# （MuJoCoモデルには真の壁形状がそのまま焼き込まれているため、方策が壁を
# 「知っているか」に関わらず視聴者には全壁が見えてしまう）。そこで「探索で
# 地図ができていく」様子を独立した2Dスキーマ・ミニマップパネルとして
# 別途描画し、そちらで 未知=薄い/既知=濃い を表現する。
#
# 座標系: カメラ実測検証（本スクリプト作成時に画像出力で確認）により、
# world (0,0) はフレーム左下、world (x_max,y_max) はフレーム右上に対応する
# （回転・反転なしの素直なxy平面）。ミニマップもこれに合わせ、
# 画像y軸のみ反転（world y増加 = 画像上方向）して描画する。
# ==========================================================================
class MiniMap:
    def __init__(self, width: int, height: int, cell_size: float, size_px: int = MINIMAP_SIZE):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.extent = width * cell_size  # 正方形前提（width == height）
        self.size_px = size_px
        self.legend_h = 40  # 凡例行の高さ [px]（グリッド下に確保。render()の画像高さに算入）
        self.cell_px = size_px / width
        self.font = _load_font(16)
        self.legend_font = _load_font(15)

        # 走行軌跡: フェーズ毎の点列（画像座標系, 各要素は (px, py)）
        self.traj_explore = []
        self.traj_return = []
        self.traj_fastest = []

    def world_to_px(self, x: float, y: float):
        px = (x / self.extent) * self.size_px
        py = self.size_px - (y / self.extent) * self.size_px  # y反転
        return px, py

    def add_point(self, phase: str, x: float, y: float):
        pt = self.world_to_px(x, y)
        if phase == "explore":
            self.traj_explore.append(pt)
        elif phase == "return":
            self.traj_return.append(pt)
        elif phase == "fastest":
            self.traj_fastest.append(pt)

    def render(self, v_walls_known, h_walls_known, robot_xy) -> Image.Image:
        img = Image.new("RGBA", (self.size_px, self.size_px), (255, 255, 255, 235))
        draw = ImageDraw.Draw(img)

        # スタート・ゴール区画ハイライト
        gx0, gx1 = self.width // 2 - 1, self.width // 2
        gy0, gy1 = self.height // 2 - 1, self.height // 2
        x0, y0 = self.world_to_px(gx0 * self.cell_size, (gy1 + 1) * self.cell_size)
        x1, y1 = self.world_to_px((gx1 + 1) * self.cell_size, gy0 * self.cell_size)
        draw.rectangle([x0, y0, x1, y1], fill=COLOR_GOAL)
        sx0, sy0 = self.world_to_px(0.0, self.cell_size)
        sx1, sy1 = self.world_to_px(self.cell_size, 0.0)
        draw.rectangle([sx0, sy0, sx1, sy1], fill=COLOR_START)

        # 壁グリッド（既知/未知の2段階濃淡）。まず全候補を薄く描き、既知の壁を
        # 上から濃く描く。既知の「壁なし」（開通路）は何も描かない＝背景のまま。
        # 縦壁 v_walls[x, y]: x=0..width, y=0..height-1
        for gx in range(self.width + 1):
            for gy in range(self.height):
                val = int(v_walls_known[gx, gy])
                wx = gx * self.cell_size
                p0 = self.world_to_px(wx, gy * self.cell_size)
                p1 = self.world_to_px(wx, (gy + 1) * self.cell_size)
                if val == 1:
                    draw.line([p0, p1], fill=COLOR_WALL_KNOWN, width=3)
                elif val == -1:
                    draw.line([p0, p1], fill=COLOR_WALL_UNKNOWN, width=1)
        # 横壁 h_walls[x, y]: x=0..width-1, y=0..height
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

        # 走行軌跡（帰還→探索→最短の順に描き、最短が一番上に来るように）
        if len(self.traj_return) >= 2:
            draw.line(self.traj_return, fill=COLOR_RETURN, width=4, joint="curve")
        if len(self.traj_explore) >= 2:
            draw.line(self.traj_explore, fill=COLOR_EXPLORE, width=4, joint="curve")
        if len(self.traj_fastest) >= 2:
            draw.line(self.traj_fastest, fill=COLOR_FASTEST, width=5, joint="curve")

        # 現在位置マーカー
        rx, ry = self.world_to_px(robot_xy[0], robot_xy[1])
        r = 6
        draw.ellipse([rx - r, ry - r, rx + r, ry + r], fill=COLOR_ROBOT_DOT, outline=(255, 255, 255, 255), width=1)

        # 外枠
        draw.rectangle([0, 0, self.size_px - 1, self.size_px - 1], outline=(60, 60, 60, 255), width=2)

        # 凡例
        legend_y = self.size_px + 6
        items = [
            (COLOR_EXPLORE, "探索走行"),
            (COLOR_RETURN, "帰還"),
            (COLOR_FASTEST, "最短走行"),
        ]
        lx = 4
        for color, label in items:
            draw.rectangle([lx, legend_y + 4, lx + 16, legend_y + 16], fill=color)
            draw.text((lx + 22, legend_y), label, font=self.legend_font, fill=(20, 20, 20, 255))
            bbox = draw.textbbox((lx + 22, legend_y), label, font=self.legend_font)
            lx = bbox[2] + 18

        return img


# ==========================================================================
# フレーム合成（メイン俯瞰映像 + 情報パネル + ミニマップ）
# ==========================================================================
def compose_frame(base_frame: np.ndarray, minimap_img: Image.Image, info_lines: list,
                   font, font_small) -> np.ndarray:
    img = Image.fromarray(base_frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # --- 情報パネル（左上、半透明黒地に白文字） ---
    pad = 16
    line_gap = 8
    line_heights = []
    max_w = 0
    for text, is_title in info_lines:
        f = font if is_title else font_small
        bbox = draw.textbbox((0, 0), text, font=f)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        line_heights.append(h)
        max_w = max(max_w, w)

    total_h = sum(line_heights) + line_gap * (len(info_lines) - 1) if info_lines else 0
    x0, y0 = 24, 24
    x1 = x0 + max_w + 2 * pad
    y1 = y0 + total_h + 2 * pad
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 175))

    cy = y0 + pad
    for (text, is_title), h in zip(info_lines, line_heights):
        f = font if is_title else font_small
        color = (255, 255, 255, 255) if is_title else (235, 235, 180, 255)
        draw.text((x0 + pad, cy), text, font=f, fill=color)
        cy += h + line_gap

    composited = Image.alpha_composite(img, overlay)

    # --- ミニマップ（右下、透過PNGをそのまま貼付。凡例分の余白込み） ---
    mm_w, mm_h = minimap_img.size
    mx = composited.width - MINIMAP_MARGIN - mm_w
    my = composited.height - MINIMAP_MARGIN - mm_h
    composited.alpha_composite(minimap_img, dest=(mx, my))

    return np.array(composited.convert("RGB"))


# ==========================================================================
# メイン録画ループ
# ==========================================================================
def run_and_record(maze_id: str) -> dict:
    npz_path = EVAL_MAZE_DIR / f"{maze_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} が見つかりません")

    data_npz = np.load(npz_path)
    v_walls = data_npz["v_walls"]
    h_walls = data_npz["h_walls"]
    seed = int(data_npz["seed"])
    width = int(data_npz["width"])
    height = int(data_npz["height"])

    # センサ回避策: SENSORS_6 で派生XMLをビルドする（competition/ 配下は不変）。
    # 派生XMLはロボット幾何形状には影響しない（センサはrangefinderサイトのみ）。
    XML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    xml_path = XML_CACHE_DIR / f"{maze_id}_sensors6.xml"
    params = RobotParams(sensors=SENSORS_6)
    build_maze_robot_xml(v_walls, h_walls, str(xml_path),
                          model_name=f"l0_new_maze_full_{maze_id}", params=params)

    sim = MouseSim(str(xml_path), params=params)
    sim.model.vis.global_.offwidth = FRAME_SIZE
    sim.model.vis.global_.offheight = FRAME_SIZE
    renderer = mujoco.Renderer(sim.model, height=FRAME_SIZE, width=FRAME_SIZE)
    cam = make_camera(width, height, DEFAULT_CELL_SIZE)
    font = _load_font(34)
    font_small = _load_font(26)

    sim.full_reset(cell=(0, 0), heading_deg=90.0)

    policy = AdachiPolicy()
    policy.bind_sim(sim)
    policy.bind_maze(v_walls, h_walls)
    policy.on_maze_start({
        "maze_id": maze_id, "seed": seed,
        "xml_path": str(xml_path), "width": width, "height": height,
    })

    cell_size = params.cell_size
    ring = PositionRingBuffer(window_s=DEFAULT_STUCK_WINDOW_S)
    minimap = MiniMap(width, height, cell_size)

    OUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(OUT_VIDEO_PATH), fps=FPS, macro_block_size=1)

    x, y, _yaw = sim.privileged_pose()
    prev_in_start = in_start_region(x, y, cell_size)
    segment_start_t = sim.sim_time
    ring.push(sim.sim_time, x, y)

    state = "FREE"
    run_count = 0
    timeline = []
    confirmed_times = []  # [(label, run_time), ...] 画面に残し続ける確定タイム
    next_frame_time = 0.0
    n_frames_written = 0
    step_i = 0
    stop_after_run2_goal = False  # run_count==2 が goal で終わったら録画終了
    fastest_run_time = None
    explore_run_time = None

    wall_clock_start = time.time()

    def current_phase():
        if state == "RUN_ACTIVE" and run_count == 1:
            return "explore"
        if state == "RUN_ACTIVE" and run_count >= 2:
            return "fastest"
        if state == "FREE" and run_count >= 1:
            return "return"
        return None

    def state_label():
        if state == "FREE" and run_count == 0:
            return "スタート待機中"
        if state == "RUN_ACTIVE" and run_count == 1:
            return "第1走行（探索中）"
        if state == "FREE" and run_count == 1:
            return "帰還中"
        if state == "RUN_ACTIVE" and run_count == 2:
            return "第2走行（最短走行）"
        if state == "RUN_ACTIVE":
            return f"第{run_count}走行"
        return "帰還中"

    print(f"迷路 {maze_id} (seed={seed}) の録画を開始します。")
    print(f"出力先: {OUT_VIDEO_PATH.resolve()}")

    while True:
        obs = sim.observation()
        vl, vr = policy.act(obs)
        result = sim.step_control(float(vl), float(vr))
        t = result["sim_time"]
        x, y, _yaw = sim.privileged_pose()
        ring.push(t, x, y)

        in_start = in_start_region(x, y, cell_size)
        in_goal = in_goal_region(x, y, width, height, cell_size)
        is_stuck, _disp = ring.check_stuck(t, x, y, segment_start_t, DEFAULT_STUCK_DISP_M)

        outcome = None
        if state == "RUN_ACTIVE":
            if in_goal:
                outcome = "goal"
            elif result["collision"]:
                outcome = "collision"
            elif result["tipped"]:
                outcome = "tipover"
            elif is_stuck:
                outcome = "stuck"

        # 実時間対応フレームキャプチャ
        while t >= next_frame_time and outcome != "goal":
            phase = current_phase()
            if phase is not None:
                minimap.add_point(phase, x, y)
            mm_img = minimap.render(policy.v_walls_known, policy.h_walls_known, (x, y))

            info_lines = [
                (METHOD_LABEL, True),
                (f"持ち時間  {t:6.1f} / {TIME_BUDGET_S:.1f} s", True),
                (state_label(), True),
            ]
            for label, rt in confirmed_times:
                info_lines.append((f"確定: {label}  {rt:.2f} s", False))

            base = render_scene(renderer, cam, sim.data)
            frame = compose_frame(base, mm_img, info_lines, font, font_small)
            writer.append_data(frame)
            n_frames_written += 1
            next_frame_time += FRAME_DT

        if state == "RUN_ACTIVE" and outcome is not None:
            policy.on_run_end(outcome)
            run_time = t - segment_start_t
            timeline.append({"event": "run_end", "run": run_count, "outcome": outcome, "t": t,
                              "run_time": run_time})

            if outcome == "goal":
                if run_count == 1:
                    explore_run_time = run_time
                    confirmed_times.append((f"第1走行（探索）", run_time))
                    print(f"  [t={t:7.2f}s] 第1走行（探索）ゴール到達: {run_time:.2f} s")
                elif run_count == 2:
                    fastest_run_time = run_time
                    confirmed_times.append((f"第2走行（最短）", run_time))
                    ratio = (1.0 - fastest_run_time / explore_run_time) * 100.0 \
                        if explore_run_time else None
                    print(f"  [t={t:7.2f}s] 第2走行（最短）ゴール到達: {run_time:.2f} s"
                          + (f"（短縮率 {ratio:.1f}%）" if ratio is not None else ""))
                    stop_after_run2_goal = True
                else:
                    confirmed_times.append((f"第{run_count}走行", run_time))
                    print(f"  [t={t:7.2f}s] 第{run_count}走行 ゴール到達: {run_time:.2f} s")

                state = "FREE"
                segment_start_t = t
                prev_in_start = in_start

                if stop_after_run2_goal:
                    if fastest_run_time is not None and explore_run_time:
                        ratio = 1.0 - fastest_run_time / explore_run_time
                        info_lines = [
                            (METHOD_LABEL, True),
                            (f"持ち時間  {t:6.1f} / {TIME_BUDGET_S:.1f} s", True),
                            ("最短走行 成立", True),
                        ]
                        for label, rt in confirmed_times:
                            info_lines.append((f"確定: {label}  {rt:.2f} s", False))
                        info_lines.append((f"短縮率 {ratio*100:.1f}%", False))
                    phase = "fastest"
                    minimap.add_point(phase, x, y)
                    mm_img = minimap.render(policy.v_walls_known, policy.h_walls_known, (x, y))
                    base = render_scene(renderer, cam, sim.data)
                    final_frame = compose_frame(base, mm_img, info_lines, font, font_small)
                    for _ in range(HOLD_FRAMES_AT_END):
                        writer.append_data(final_frame)
                        n_frames_written += 1
                    break
            else:
                print(f"  [t={t:7.2f}s] 第{run_count}走行 {outcome}（係員回収）")
                sim.reset_to_start(cell=(0, 0), heading_deg=90.0)
                policy.on_retrieval()
                x, y, _yaw = sim.privileged_pose()
                state = "FREE"
                prev_in_start = True
                segment_start_t = t
        else:
            if state == "FREE" and prev_in_start and not in_start:
                run_count += 1
                policy.on_run_start(run_count)
                state = "RUN_ACTIVE"
                segment_start_t = t
                timeline.append({"event": "run_start", "run": run_count, "t": t})
            prev_in_start = in_start

        if t > SAFETY_TIME_LIMIT_S:
            print(f"[WARNING] sim_time={t:.2f}s が安全弁 {SAFETY_TIME_LIMIT_S}s を超過。"
                  "異常とみなしループを打ち切ります。")
            break

        step_i += 1
        if step_i % 1000 == 0:
            elapsed_wall = time.time() - wall_clock_start
            print(f"  ... sim_time={t:6.2f}s  state={state:10s}  run_count={run_count}  "
                  f"frames={n_frames_written:5d} ({n_frames_written/FPS:6.1f}s分)  "
                  f"経過(実時間)={elapsed_wall:6.1f}s")

    writer.close()
    renderer.close()
    wall_clock = time.time() - wall_clock_start

    return {
        "maze_id": maze_id, "seed": seed,
        "explore_run_time": explore_run_time,
        "fastest_run_time": fastest_run_time,
        "timeline": timeline,
        "n_frames_written": n_frames_written,
        "wall_clock_s": wall_clock,
    }


def render_scene(renderer, cam, data) -> np.ndarray:
    renderer.update_scene(data, camera=cam)
    return renderer.render()


def main():
    parser = argparse.ArgumentParser(description="L0-a 新16x16評価迷路 フル録画（省略なし・等倍速）")
    parser.add_argument("--maze-id", type=str, default=SELECTED_MAZE_ID,
                         help=f"録画対象の迷路ID（既定: {SELECTED_MAZE_ID}）")
    args = parser.parse_args()

    print("=" * 70)
    print(f"{METHOD_LABEL}（AdachiPolicy） 新16x16評価迷路 フル録画（探索→帰還→最短走行）")
    print("=" * 70)
    result = run_and_record(args.maze_id)

    print("\n--- タイムライン（実測） ---")
    for ev in result["timeline"]:
        if ev["event"] == "run_start":
            print(f"  run {ev['run']}: 出発 t={ev['t']:.2f}s")
        else:
            print(f"  run {ev['run']}: 終了 outcome={ev['outcome']:10s} t={ev['t']:.2f}s "
                  f"run_time={ev['run_time']:.2f}s")

    print("\n--- 結果 ---")
    print(f"探索走行タイム: {result['explore_run_time']}")
    print(f"最短走行タイム: {result['fastest_run_time']}")
    if result["explore_run_time"] and result["fastest_run_time"]:
        ratio = 1.0 - result["fastest_run_time"] / result["explore_run_time"]
        print(f"短縮率: {ratio*100:.1f}%")
    print(f"書き出しフレーム数: {result['n_frames_written']} "
          f"（{result['n_frames_written'] / FPS:.1f} s @ {FPS}fps）")
    print(f"出力: {OUT_VIDEO_PATH.resolve()}")
    print(f"実処理時間（wall clock）: {result['wall_clock_s']:.1f} s")


if __name__ == "__main__":
    main()
