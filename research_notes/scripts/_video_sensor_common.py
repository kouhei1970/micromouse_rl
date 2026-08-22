"""
research_notes/scripts/_video_sensor_common.py
================
発表用動画（センサの読みの可視化）の共通部品。段階1（`video_sensor_stage1.py`）が
保存した走行記録（`outputs/video_sensor/run_data.npz`）を読み、各章の描画スクリプトが
共有する: 迷路俯瞰の描画・センサ計算・rangefinder（真の幾何距離）計算・
字幕/計器パネルの描画・ffmpeg への直接パイプ書き出し。

🔴 `mouse/`・`classic/`・`competition/` のコードは一切変更しない（読み取り・import のみ）。
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Hiragino Kaku Gothic ProN"  # 日本語グリフ用（既定のDejaVu Sansは非対応）
import matplotlib.pyplot as plt          # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402  3D投影（応答曲面パネルで使う）

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classic.geometry import Rect, wall_obstacles  # noqa: E402
from mouse.ir_sensor import (  # noqa: E402
    IrSensorSpec,
    SurfaceSpec,
    build_maze_cell_index,
    response,
    response_table,
)
from mouse.ir_table import interp_G, load_cumulative_table  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

FFMPEG = "/opt/homebrew/bin/ffmpeg"

FPS = 30
CANVAS_W, CANVAS_H = 1920, 1080

# 満量（AUDIT_050 §2-2 以来の固定値。`research_notes/scripts/video_sensor_stage1.py` と同じ）。
I_FULL = 0.8298934

DATA_PATH = REPO_ROOT / "outputs" / "video_sensor" / "run_data.npz"
META_PATH = REPO_ROOT / "outputs" / "video_sensor" / "run_meta.json"
MAZE_DIR = REPO_ROOT / "outputs" / "video_sensor" / "maze"
TABLE_PATH = REPO_ROOT / "mouse" / "data" / "ir_cumulative_table.npz"
OUT_DIR = REPO_ROOT / "outputs" / "video_sensor"
CHAPTERS_DIR = OUT_DIR / "chapters"
FRAMES_DIR = OUT_DIR / "frames"  # 代表コマ(png)置き場

# --------------------------------------------------------------------------
# ナレーション音声（`research_notes/scripts/video_sensor_narration/build_narration.py`
# が書き出す）。映像側は段落ごとの実測長を読んで尺を決める（ハードコードしない）。
# --------------------------------------------------------------------------
NARR_DIR = OUT_DIR / "narration"
NARR_PARA_DIR = NARR_DIR / "_paragraphs"
PARA_GAP_S = 0.6  # 段落間に置く無音（build_narration.py と共通の値。値の正本はこちら）
FFPROBE = "/opt/homebrew/bin/ffprobe"


def audio_duration(path: Path) -> float:
    """ffprobe で音声ファイルの長さ（秒）を測る。"""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def chapter_audio_duration(n: int) -> float:
    """章 n のナレーション mp3（連結済み）の長さ（秒）。"""
    p = NARR_DIR / f"ch{n}.mp3"
    if not p.exists():
        raise FileNotFoundError(f"ナレーションが無い: {p}（先に build_narration.py を実行）")
    return audio_duration(p)


def allocate_frames(stage_seconds: list, total_seconds: float, fps: int = FPS) -> list:
    """各段階の秒数（音声の実測長から出した目安）を、映像コマ数の割り当てに変える。
    合計コマ数は `ceil(total_seconds * fps)`（音声より短くならないようにする）に固定し、
    端数は最後の段階に寄せる（四捨五入の誤差を1箇所に集めるだけで、体感できるズレにはならない）。"""
    import math
    total_frames = int(math.ceil(total_seconds * fps))
    frames = [max(1, round(s * fps)) for s in stage_seconds]
    diff = total_frames - sum(frames)
    frames[-1] += diff
    if frames[-1] < 1:
        raise ValueError(f"最後の段階のコマ数が0以下になった: {frames}")
    return frames


def paragraph_durations(n: int) -> list:
    """章 n の段落ごとの mp3 の長さ（秒）のリスト（`build_narration.py` が書き出した
    `ch{n}_p{i}.mp3` を i=0,1,2,... と順に探す）。"""
    durs = []
    i = 0
    while True:
        p = NARR_PARA_DIR / f"ch{n}_p{i}.mp3"
        if not p.exists():
            break
        durs.append(audio_duration(p))
        i += 1
    if not durs:
        raise FileNotFoundError(f"段落音声が無い: {NARR_PARA_DIR}/ch{n}_p*.mp3"
                                 f"（先に build_narration.py を実行）")
    return durs

# --------------------------------------------------------------------------
# 配色（`_video_l0_common.py` のダーク配色を踏襲。プロジェクトの動画の作法に合わせる）
# --------------------------------------------------------------------------
BG = (18, 19, 22, 255)
PANEL_BG = (24, 26, 30, 255)
SEC_BG = (30, 32, 37, 255)
WALL_COLOR = (235, 235, 235, 255)
POST_COLOR = (150, 150, 155, 255)
FLOOR_COLOR = (46, 49, 54, 255)
ROBOT_COLOR = (10, 132, 255, 255)
TRAIL_COLOR = (255, 214, 10, 200)
RAY_COLORS = {
    "LF": (255, 99, 92, 220), "RF": (255, 99, 92, 220),
    "LS": (100, 210, 255, 220), "RS": (100, 210, 255, 220),
}
SENSOR_COLORS = {"LF": (255, 99, 92, 255), "LS": (100, 210, 255, 255),
                  "RF": (255, 149, 0, 255), "RS": (100, 255, 160, 255)}
TEXT_WHITE = (240, 240, 240, 255)
TEXT_DIM = (175, 178, 185, 255)
TEXT_ACCENT = (255, 214, 10, 255)
TEXT_CAPTION = (255, 214, 10, 255)
GRID_COLOR = (70, 73, 80, 255)
BAR_BG = (55, 58, 64, 255)

PHASE_JA = {"EXPLORE": "探索走行", "RETURN": "帰還（探索中）",
            "FAST": "最短走行", "RETURN2": "帰還（最短走行後）", "INIT": "-"}


# ==========================================================================
# フォント
# ==========================================================================
_FONT_CACHE: dict = {}


def font(size: int, bold: bool = False):
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidates = [
        ("/System/Library/Fonts/ヒラギノ角ゴシック W7.ttc" if bold else
         "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", 0),
        ("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", 0),
        ("/System/Library/Fonts/Menlo.ttc", 0),
    ]
    f = None
    for path, index in candidates:
        if os.path.exists(path):
            f = ImageFont.truetype(path, size, index=index)
            break
    if f is None:
        f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


# ==========================================================================
# 走行記録の読み込み
# ==========================================================================
class RunData:
    def __init__(self, path=DATA_PATH):
        d = np.load(path, allow_pickle=False)
        self.seed = int(d["seed"])
        self.cell_size = float(d["cell_size"])
        self.v_walls = d["v_walls"]
        self.h_walls = d["h_walls"]
        self.width = int(d["width"])
        self.height = int(d["height"])
        self.sim_time = d["sim_time"]
        self.x = d["x"]
        self.y = d["y"]
        self.yaw = d["yaw"]
        self.v_fwd = d["v_fwd"]
        self.omega_z = d["omega_z"]
        self.collision = d["collision"]
        self.tipped = d["tipped"]
        self.phase = d["phase"]
        self.run_index = d["run_index"]
        self.run_active = d["run_active"]

    def idx_range(self, t_lo: float, t_hi: float) -> np.ndarray:
        return np.where((self.sim_time >= t_lo) & (self.sim_time < t_hi))[0]

    def pose_at_time(self, t: float) -> tuple:
        """最寄りの記録点の (x,y,yaw,sim_time,phase,run_index,v_fwd) を返す
        （`np.searchsorted` による最近傍。記録は 100Hz 等間隔なので十分な精度）。"""
        i = int(np.searchsorted(self.sim_time, t))
        i = min(max(i, 0), len(self.sim_time) - 1)
        return (float(self.x[i]), float(self.y[i]), float(self.yaw[i]),
                float(self.sim_time[i]), str(self.phase[i]), int(self.run_index[i]),
                float(self.v_fwd[i]))


def maze_xml_path(seed: int) -> Path:
    p = MAZE_DIR / f"maze_{seed}.xml"
    if not p.exists():
        raise FileNotFoundError(f"迷路 XML が無い: {p}（先に video_sensor_stage1.py を実行）")
    return p


# ==========================================================================
# センサ計算（response_table。段階1と同じ手順）
# ==========================================================================
def build_ir_specs(p: RobotParams = None) -> list:
    p = p or RobotParams()
    specs = []
    for s in p.sensors:
        pos = tuple(float(v) for v in s["pos"].split())
        axis = tuple(float(v) for v in s["zaxis"].split())
        specs.append(IrSensorSpec(name=s["name"], pos=pos, axis=axis))
    return specs


class SensorContext:
    """response_table() 呼び出しに要る一式（迷路1枚につき1回だけ作る）。"""

    def __init__(self, v_walls, h_walls, cell_size: float, table_path: Path = TABLE_PATH):
        self.params = RobotParams()
        self.specs = build_ir_specs(self.params)
        self.surf = SurfaceSpec()
        self.rects = wall_obstacles(v_walls, h_walls, cell_size=cell_size)
        self.rects_arr = np.asarray(self.rects, dtype=np.float64)  # (M,4) cx,cy,hx,hy
        self.cell_index = build_maze_cell_index(self.rects, cell_size)
        self.cell_size = cell_size
        self.table = load_cumulative_table(table_path)

    def response_raw(self, x: float, y: float, yaw: float) -> np.ndarray:
        """4本ぶんの生の応答（任意単位）を返す。"""
        pose = (x, y, yaw)
        out = np.empty(len(self.specs), dtype=np.float64)
        for i, spec in enumerate(self.specs):
            out[i] = response_table(spec, pose, self.rects, self.surf,
                                     self.table, self.cell_index, self.cell_size)
        return out

    def response_ratio(self, x: float, y: float, yaw: float) -> np.ndarray:
        return self.response_raw(x, y, yaw) / I_FULL


# ==========================================================================
# rangefinder（MuJoCo純正・真の幾何距離）。理論曲線の x 軸（距離）に使う。
# センサモデルの計算（response_table）とは別に、「センサからその方向の障害物までの
# 幾何距離」を求めるためだけに、物理を進めず qpos を直接書いて mj_forward するだけの
# 軽い読み取りに使う（`mouse/sim.py::reset_to_start` と同じ qpos の書き方を踏襲）。
# ==========================================================================
class RangefinderProbe:
    def __init__(self, xml_path: Path):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self._root_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self._qadr = self.model.jnt_qposadr[self._root_jid]
        self._n = int(np.sum(self.model.sensor_type == mujoco.mjtSensor.mjSENS_RANGEFINDER))
        self.cutoff = float(RobotParams().sensor_cutoff)

    def ranges_m(self, x: float, y: float, yaw: float) -> np.ndarray:
        q = self._qadr
        self.data.qpos[q:q + 3] = [x, y, 0.002]
        self.data.qpos[q + 3] = np.cos(yaw / 2.0)
        self.data.qpos[q + 4] = 0.0
        self.data.qpos[q + 5] = 0.0
        self.data.qpos[q + 6] = np.sin(yaw / 2.0)
        mujoco.mj_forward(self.model, self.data)
        sd = np.array(self.data.sensordata[0:self._n], dtype=np.float64)
        sd = np.where(sd < 0, self.cutoff, sd)
        return np.clip(sd, 0.0, self.cutoff)


# ==========================================================================
# 幾何: センサ光が最初に当たる壁面の「垂直距離 d」と「入射角 θ」
# （教授セッション検収 2026-08-22。マイクロマウスの壁はすべて軸平行の矩形なので、
# 面の法線は必ず世界 x 軸か y 軸のどちらか——スラブ法（ray-AABB）で厳密に求まる。
# `mouse/ir_table.py` の d・θ の定義（面の外向き法線が θ=0 の基準）に厳密に合わせてある）。
# ==========================================================================
def sensor_world_ray(x: float, y: float, yaw: float, spec) -> tuple:
    """センサの取付位置（世界座標）と、光軸の世界方向（単位ベクトル）を返す。"""
    lx, ly = spec.pos[0], spec.pos[1]
    cyaw, syaw = np.cos(yaw), np.sin(yaw)
    wx = x + lx * cyaw - ly * syaw
    wy = y + lx * syaw + ly * cyaw
    ax, ay = spec.axis[0], spec.axis[1]
    n = max(1e-9, (ax * ax + ay * ay) ** 0.5)
    ax, ay = ax / n, ay / n
    dx = ax * cyaw - ay * syaw
    dy = ax * syaw + ay * cyaw
    return wx, wy, dx, dy


def ray_wall_hit(sx: float, sy: float, dxr: float, dyr: float, rects_arr: np.ndarray,
                  max_range: float):
    """光軸 `(sx,sy)→(dxr,dyr)` が最初に当たる壁面（矩形の1面）の垂直距離[m]・入射角[deg]を、
    スラブ法（ray-AABB intersection）で厳密に求める。`rects_arr` は `SensorContext.rects_arr`
    （cx,cy,hx,hy の列）。当たらなければ `None`（＝壁が視野に入っていないセンサ）。

    d・θ の定義は `mouse/ir_table.py` モジュール docstring と同じ（θ=0＝面に正対）。
    """
    cx, cy, hx, hy = rects_arr[:, 0], rects_arr[:, 1], rects_arr[:, 2], rects_arr[:, 3]
    eps = 1e-12
    if abs(dxr) > eps:
        tx1 = ((cx - hx) - sx) / dxr
        tx2 = ((cx + hx) - sx) / dxr
        tx_near = np.minimum(tx1, tx2)
        tx_far = np.maximum(tx1, tx2)
    else:
        inside_x = (sx >= cx - hx - 1e-9) & (sx <= cx + hx + 1e-9)
        tx_near = np.where(inside_x, -np.inf, np.inf)
        tx_far = np.where(inside_x, np.inf, -np.inf)
    if abs(dyr) > eps:
        ty1 = ((cy - hy) - sy) / dyr
        ty2 = ((cy + hy) - sy) / dyr
        ty_near = np.minimum(ty1, ty2)
        ty_far = np.maximum(ty1, ty2)
    else:
        inside_y = (sy >= cy - hy - 1e-9) & (sy <= cy + hy + 1e-9)
        ty_near = np.where(inside_y, -np.inf, np.inf)
        ty_far = np.where(inside_y, np.inf, -np.inf)

    t_near = np.maximum(tx_near, ty_near)
    t_far = np.minimum(tx_far, ty_far)
    valid = (t_near <= t_far) & (t_far >= 1e-6) & (t_near >= -1e-6)
    t_hit = np.where(valid, np.maximum(t_near, 0.0), np.inf)
    idx = int(np.argmin(t_hit))
    t_best = float(t_hit[idx])
    if not np.isfinite(t_best) or t_best > max_range:
        return None

    axis = "x" if tx_near[idx] >= ty_near[idx] else "y"
    if axis == "x":
        # 進行方向 +x ならまず左面(cx-hx)に、-x ならまず右面(cx+hx)に当たる
        plane = cx[idx] - hx[idx] if dxr > 0 else cx[idx] + hx[idx]
        d = abs(sx - plane)
        nx, ny = (1.0 if sx > plane else -1.0), 0.0
    else:
        plane = cy[idx] - hy[idx] if dyr > 0 else cy[idx] + hy[idx]
        d = abs(sy - plane)
        nx, ny = 0.0, (1.0 if sy > plane else -1.0)
    ux, uy = ny, -nx   # 法線を-90°回した向き（mouse/ir_table.py の u 軸と同じ規約）
    axis_local_x = -(dxr * nx + dyr * ny)
    axis_local_y = dxr * ux + dyr * uy
    theta_deg = float(np.degrees(np.arctan2(axis_local_y, axis_local_x)))
    return d, theta_deg


def sensor_d_theta(ctx: "SensorContext", x: float, y: float, yaw: float,
                    max_range: float = None) -> dict:
    """4本ぶんの (垂直距離[mm], 入射角[deg]) を返す（壁が視野に無いセンサは値 `None`）。"""
    max_range = ctx.params.sensor_cutoff if max_range is None else max_range
    out = {}
    for spec in ctx.specs:
        sx, sy, dxr, dyr = sensor_world_ray(x, y, yaw, spec)
        hit = ray_wall_hit(sx, sy, dxr, dyr, ctx.rects_arr, max_range)
        out[spec.name] = None if hit is None else (hit[0] * 1000.0, hit[1])
    return out


# ==========================================================================
# 描画: 迷路俯瞰パネル（壁・柱・機体・センサ光の向き・走行軌跡）
# 座標系: world (0,0)=左下 → 画像は y を反転して描く。
# ==========================================================================
class MazePanel:
    def __init__(self, x0: int, y0: int, size: int, width: int, height: int, cell_size: float,
                 margin_cells: float = 0.4):
        self.x0, self.y0, self.size = x0, y0, size
        self.extent = max(width, height) * cell_size
        self.margin = margin_cells * cell_size
        self.scale = size / (self.extent + 2 * self.margin)

    def w2p(self, x: float, y: float) -> tuple:
        px = self.x0 + (x + self.margin) * self.scale
        py = self.y0 + self.size - (y + self.margin) * self.scale
        return px, py


def draw_maze_panel_bg(draw: ImageDraw.ImageDraw, panel: MazePanel):
    draw.rectangle([panel.x0, panel.y0, panel.x0 + panel.size, panel.y0 + panel.size],
                    fill=FLOOR_COLOR)


def draw_maze_walls(draw: ImageDraw.ImageDraw, panel: MazePanel, rd: RunData,
                     wall_thickness: float = 0.012, post_size: float = 0.012):
    cs = rd.cell_size
    # 縦壁
    vw = rd.v_walls
    for ix in range(vw.shape[0]):
        for iy in range(vw.shape[1]):
            if vw[ix, iy]:
                cx, cy = ix * cs, iy * cs + cs / 2
                hx, hy = wall_thickness / 2, cs / 2 - post_size / 2
                p1 = panel.w2p(cx - hx, cy - hy)
                p2 = panel.w2p(cx + hx, cy + hy)
                draw.rectangle([min(p1[0], p2[0]), min(p1[1], p2[1]),
                                max(p1[0], p2[0]), max(p1[1], p2[1])], fill=WALL_COLOR)
    hw = rd.h_walls
    for ix in range(hw.shape[0]):
        for iy in range(hw.shape[1]):
            if hw[ix, iy]:
                cx, cy = ix * cs + cs / 2, iy * cs
                hx, hy = cs / 2 - post_size / 2, wall_thickness / 2
                p1 = panel.w2p(cx - hx, cy - hy)
                p2 = panel.w2p(cx + hx, cy + hy)
                draw.rectangle([min(p1[0], p2[0]), min(p1[1], p2[1]),
                                max(p1[0], p2[0]), max(p1[1], p2[1])], fill=WALL_COLOR)
    # 柱（格子点すべて）
    r_px = max(1.5, post_size / 2 * panel.scale)
    for ix in range(rd.width + 1):
        for iy in range(rd.height + 1):
            cx, cy = panel.w2p(ix * cs, iy * cs)
            draw.ellipse([cx - r_px, cy - r_px, cx + r_px, cy + r_px], fill=POST_COLOR)


def draw_trail(draw: ImageDraw.ImageDraw, panel: MazePanel, xs, ys, color=TRAIL_COLOR, width=4):
    if len(xs) < 2:
        return
    pts = [panel.w2p(x, y) for x, y in zip(xs, ys)]
    draw.line(pts, fill=color, width=width, joint="curve")


def draw_robot(draw: ImageDraw.ImageDraw, panel: MazePanel, x: float, y: float, yaw: float,
               body_half_m: float = 0.037, color=ROBOT_COLOR):
    """機体を進行方向が分かる三角形で描く（先端=前方）。"""
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    lat = np.array([-np.sin(yaw), np.cos(yaw)])
    p_front = (x + fwd[0] * body_half_m * 1.4, y + fwd[1] * body_half_m * 1.4)
    p_left = (x - fwd[0] * body_half_m * 0.8 + lat[0] * body_half_m,
              y - fwd[1] * body_half_m * 0.8 + lat[1] * body_half_m)
    p_right = (x - fwd[0] * body_half_m * 0.8 - lat[0] * body_half_m,
               y - fwd[1] * body_half_m * 0.8 - lat[1] * body_half_m)
    pts = [panel.w2p(*p_front), panel.w2p(*p_left), panel.w2p(*p_right)]
    draw.polygon(pts, fill=color, outline=(255, 255, 255, 255))


def draw_sensor_rays(draw: ImageDraw.ImageDraw, panel: MazePanel, x: float, y: float, yaw: float,
                      specs, ranges_m=None, default_len=0.12, colors=None, width=3, marker_r=4,
                      marker_outline=None):
    """センサ4本の光の向きを機体から線分で描く。`ranges_m` があれば実際に届いた距離まで
    描き（先端に丸＝当たっている点の印）、無ければ既定の短い線で「向き」だけ示す。
    `colors` を渡せば配色を差し替えられる（既定は薄い RAY_COLORS。拡大図では
    右の棒グラフと同じ SENSOR_COLORS を渡し、対応が分かるようにする）。"""
    colors = colors or RAY_COLORS
    cyaw, syaw = np.cos(yaw), np.sin(yaw)
    for i, spec in enumerate(specs):
        lx, ly = spec.pos[0], spec.pos[1]
        wx = x + lx * cyaw - ly * syaw
        wy = y + lx * syaw + ly * cyaw
        ax, ay = spec.axis[0], spec.axis[1]
        n = max(1e-9, (ax * ax + ay * ay) ** 0.5)
        ax, ay = ax / n, ay / n
        dx = ax * cyaw - ay * syaw
        dy = ax * syaw + ay * cyaw
        length = float(ranges_m[i]) if ranges_m is not None else default_len
        length = max(length, 0.01)
        ex, ey = wx + dx * length, wy + dy * length
        color = colors.get(spec.name, (255, 255, 255, 200))
        draw.line([panel.w2p(wx, wy), panel.w2p(ex, ey)], fill=color, width=width)
        r = marker_r
        p = panel.w2p(ex, ey)
        if marker_outline is not None:
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color,
                         outline=marker_outline, width=2)
        else:
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)


# ==========================================================================
# 描画: 機体まわりの拡大パネル（機体中心・固定物理窓）
# センサの光がどの壁に当たっているかを見せるのが目的（迷路全体図では機体が小さすぎて
# 読めないため、教授セッション検収 2026-08-22 で追加）。MazePanel と同じ w2p() を持つので
# draw_maze_walls / draw_trail / draw_robot / draw_sensor_rays をそのまま使い回せる。
# ==========================================================================
class ZoomPanel:
    def __init__(self, x0: int, y0: int, size: int, cx: float, cy: float, extent_m: float):
        self.x0, self.y0, self.size = x0, y0, size
        self.cx, self.cy = cx, cy
        self.extent = extent_m
        self.scale = size / extent_m

    def w2p(self, x: float, y: float) -> tuple:
        px = self.x0 + self.size / 2 + (x - self.cx) * self.scale
        py = self.y0 + self.size / 2 - (y - self.cy) * self.scale
        return px, py


def render_zoom_inset(rd: RunData, specs, x: float, y: float, yaw: float, ranges_m,
                       size: int, extent_m: float, trail_xs=None, trail_ys=None) -> Image.Image:
    """機体中心・`extent_m` 四方（既定は3区画=540mm）の拡大俯瞰を `size` 四方の画像で返す。
    独立した画像に描いてから貼り付ける作りなので、窓の外へはみ出す壁・光線は自動的に
    切れる（他パネルへ漏れない）。"""
    img = Image.new("RGB", (size, size), FLOOR_COLOR[:3])
    draw = ImageDraw.Draw(img, "RGBA")
    panel = ZoomPanel(0, 0, size, x, y, extent_m)
    draw_maze_walls(draw, panel, rd)
    if trail_xs is not None and len(trail_xs) >= 2:
        draw_trail(draw, panel, trail_xs, trail_ys, width=3)
    # センサ4本の光は右の棒グラフと同じ配色（SENSOR_COLORS）で描き、対応が分かるようにする。
    draw_sensor_rays(draw, panel, x, y, yaw, specs, ranges_m=ranges_m, colors=SENSOR_COLORS,
                      width=5, marker_r=9, marker_outline=(255, 255, 255, 255))
    draw_robot(draw, panel, x, y, yaw)
    draw.rectangle([0, 0, size - 1, size - 1], outline=(95, 98, 105, 255), width=2)
    return img


# ==========================================================================
# 描画: 右パネル（棒・時系列・応答曲線）
# ==========================================================================
def draw_panel_bg(draw, rect, bg=SEC_BG, title=None, title_color=TEXT_DIM):
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, fill=bg, outline=(0, 0, 0, 0))
    if title:
        draw.text((x0 + 14, y0 + 8), title, font=font(20, bold=True), fill=title_color)


SENSOR_ORDER = ["LF", "LS", "RF", "RS"]
# 棒グラフだけの左右の並び（ユーザ指示 2026-08-22）。機体から出る光は左から
# LS(左)・LF(左前)・RF(右前)・RS(右) の順に並んでいるので、棒もこの順にして
# 拡大図の光の配置とそのまま対応させる。色・時系列・応答曲線の並びは変えない
# （SENSOR_ORDER のまま。値は名前で引くので、この並びを変えても値の対応は崩れない）。
BAR_ORDER = ["LS", "LF", "RF", "RS"]


def draw_bars(draw, rect, ratios: dict, title="距離センサ 4 本（満量比）"):
    x0, y0, x1, y1 = rect
    draw_panel_bg(draw, rect, title=title)
    top = y0 + 40
    n = len(BAR_ORDER)
    gap = 18
    bw = (x1 - x0 - 28 - gap * (n - 1)) / n
    bar_top, bar_bot = top + 10, y1 - 34
    bar_h = bar_bot - bar_top
    for i, name in enumerate(BAR_ORDER):
        bx0 = x0 + 14 + i * (bw + gap)
        r = max(0.0, min(1.15, float(ratios.get(name, 0.0))))
        frac = min(1.0, r)
        fh = bar_h * frac
        draw.rectangle([bx0, bar_top, bx0 + bw, bar_bot], fill=BAR_BG)
        color = SENSOR_COLORS[name]
        draw.rectangle([bx0, bar_bot - fh, bx0 + bw, bar_bot], fill=color)
        draw.text((bx0 + bw / 2, bar_bot + 6), name, font=font(18, bold=True),
                   fill=TEXT_WHITE, anchor="ma")
        draw.text((bx0 + bw / 2, bar_top - 22), f"{r * 100:4.1f}%", font=font(16),
                   fill=TEXT_ACCENT, anchor="ma")


def draw_timeseries(draw, rect, hist_t, hist_ratios: dict, now_t: float, window_s: float = 2.0,
                     title="直近2秒の時系列（満量比）"):
    x0, y0, x1, y1 = rect
    draw_panel_bg(draw, rect, title=title)
    px0, py0, px1, py1 = x0 + 50, y0 + 40, x1 - 20, y1 - 30
    draw.rectangle([px0, py0, px1, py1], outline=GRID_COLOR, width=1)
    for frac in (0.25, 0.5, 0.75):
        yy = py1 - (py1 - py0) * frac
        draw.line([(px0, yy), (px1, yy)], fill=GRID_COLOR, width=1)
        draw.text((px0 - 6, yy), f"{frac:.2f}", font=font(13), fill=TEXT_DIM, anchor="rm")
    draw.text((px0, py1 + 6), f"-{window_s:.0f}s", font=font(13), fill=TEXT_DIM, anchor="la")
    draw.text((px1, py1 + 6), "今", font=font(13), fill=TEXT_DIM, anchor="ra")

    t_lo = now_t - window_s
    hist_t = np.asarray(hist_t)
    mask = hist_t >= t_lo
    if mask.sum() < 2:
        return
    ts = hist_t[mask]
    for name in SENSOR_ORDER:
        vals = np.asarray(hist_ratios[name])[mask]
        pts = []
        for t, v in zip(ts, vals):
            fx = px0 + (px1 - px0) * (t - t_lo) / window_s
            fy = py1 - (py1 - py0) * max(0.0, min(1.0, v))
            pts.append((fx, fy))
        if len(pts) >= 2:
            draw.line(pts, fill=SENSOR_COLORS[name], width=3, joint="curve")


def _rgb01(c) -> tuple:
    """(R,G,B[,A]) の 0-255 表現を matplotlib 用の 0-1 表現へ変える。"""
    return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)


# ==========================================================================
# 応答曲面（距離 d [mm] × 入射角 θ [deg] × 満量比）＋ 実機2種の実測点
# （教授セッション検収 2026-08-22。旧「応答曲線（正対だけ）」を置き換えた——迷路の中では
# 側方センサが常に十数度ずれて壁を見るため、正対の曲線に載せると点が下に来て見える、
# というユーザ指摘への対応。距離と入射角の両方を軸にした面なら、走行中の点は
# 常にその面の上に来る）。
#
# 🔴 `verification/audit_051_measured_vs_model.py::front_sensor_spec()` の既定引数
# （離隔6.5mm・半値角5°/5°）は使わない——これが「画面の曲線が実際の読みより高く出る」
# 不一致の主因だった（教授セッションの実測）。ここでは `IrSensorSpec` の**現在の既定値**
# （離隔6.0mm・LED3°/PT6°、note_034 追記14）で床なし・壁半長84mmの面を作る。
# 走行中の点の計算（`mouse.ir_sensor.response_table`）が使う累積表
# （`mouse/data/ir_cumulative_table.npz`）も同じ既定値で作られているので、
# `interp_G` で面を引けば走行中の点と完全に自己無矛盾になる。
# ==========================================================================
SURFACE_CACHE_PATH = OUT_DIR / "curve_surface_data.npz"
SURFACE_D_MM = np.arange(5.0, 221.0, 2.0)     # 面の距離軸 [mm]
SURFACE_TH_DEG = np.arange(0.0, 61.0, 2.0)    # 面の入射角軸 [deg]
SURFACE_HY_M = 0.084                          # 壁の半長（迷路の壁と同じ。hy=0.084）


def compute_surface_data(force: bool = False) -> dict:
    """曲面 `Z(d,θ)`（満量比）と、実機2種の実測点（面の上に乗せた絶対値）を作る。

    - 距離の実測（`mouse/data/ir_measured_front_20260821.csv`、正対＝面のθ=0の縁）:
      利得と距離のずれを、更新後の既定値のモデルで当てはめ直す
      （`verification/audit_055_datasheet_refit.py` と同じ当てはめ方。右センサ・帯80-200mm）。
    - 角度の実測（`assets/angle_vs_sensor.csv`、距離150/160/170/180mmで面を横切る）:
      `verification/audit_052_angle_vs_model.py` と同じ手順（センサ・距離ごとに角度0で
      規格化）で相対プロファイルを作り、面自身のθ=0の高さ（未知の利得を増やさない）を
      掛けて絶対値に戻す。
    """
    if SURFACE_CACHE_PATH.exists() and not force:
        d = np.load(SURFACE_CACHE_PATH)
        return {k: d[k] for k in d.files}

    table = load_cumulative_table(TABLE_PATH)
    Dg, THg = np.meshgrid(SURFACE_D_MM / 1000.0, SURFACE_TH_DEG, indexing="ij")
    g0 = interp_G(table, Dg, THg, np.full_like(Dg, -SURFACE_HY_M))
    g1 = interp_G(table, Dg, THg, np.full_like(Dg, SURFACE_HY_M))
    Z = (g0 - g1) / I_FULL

    def z_at(d_mm: float, theta_deg: float = 0.0) -> float:
        gg0 = interp_G(table, np.array([d_mm / 1000.0]), np.array([theta_deg]),
                        np.array([-SURFACE_HY_M]))
        gg1 = interp_G(table, np.array([d_mm / 1000.0]), np.array([theta_deg]),
                        np.array([SURFACE_HY_M]))
        return float((gg0 - gg1)[0] / I_FULL)

    # --- 距離の実測（面のθ=0の縁） ---
    from verification.audit_051_measured_vs_model import FIT_RANGE_RIGHT_MM, load_measurement

    def _lf_default_spec() -> IrSensorSpec:
        p = RobotParams()
        s = next(v for v in p.sensors if v["name"] == "LF")
        pos = tuple(float(v) for v in s["pos"].split())
        axis = tuple(float(v) for v in s["zaxis"].split())
        return IrSensorSpec(name="LF", pos=pos, axis=axis)   # 離隔・半値角は既定値のまま

    def _model_dist(dist_mm: float, spec: IrSensorSpec) -> float:
        x_face = spec.pos[0] + dist_mm / 1000.0
        rect = Rect(cx=x_face + 0.006, cy=0.0, hx=0.006, hy=0.30)
        return response(spec, (0.0, 0.0, 0.0), [rect], SurfaceSpec(),
                         include_floor=False, occlusion=True)

    d_mm, _left_ad, right_ad = load_measurement()
    spec = _lf_default_spec()
    lo, hi = FIT_RANGE_RIGHT_MM
    mask = (d_mm >= lo) & (d_mm <= hi)
    offsets = np.arange(-30.0, 30.5, 0.5)
    best = (None, None, np.inf)
    for off in offsets:
        mv = np.array([_model_dist(v + off, spec) for v in d_mm[mask]])
        gain = float(np.sum(mv * right_ad[mask]) / np.sum(mv * mv))
        rms = float(np.sqrt(np.mean(((gain * mv - right_ad[mask]) / right_ad[mask]) ** 2)))
        if rms < best[2]:
            best = (float(off), gain, rms)
    dist_off, dist_gain, dist_rms = best
    dist_meas_d_mm = d_mm + dist_off
    dist_meas_theta = np.zeros_like(dist_meas_d_mm)
    dist_meas_ratio = (right_ad / dist_gain) / I_FULL

    # --- 角度の実測（距離150/160/170/180mmで面を横切る） ---
    ang_path = REPO_ROOT / "assets" / "angle_vs_sensor.csv"
    table_ang: dict = {}
    with open(ang_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (int(row["range"]), int(row["angle"]))
            table_ang[key] = {n: int(row[n]) for n in ("LF", "LS", "RS", "RF")}
    ranges = sorted({k[0] for k in table_ang})
    angles = sorted({k[1] for k in table_ang})
    ang_d_mm, ang_theta, ang_ratio = [], [], []
    for rg in ranges:
        z0 = z_at(float(rg), 0.0)
        for a in angles:
            if (rg, a) not in table_ang or a == 0:
                continue
            vals = [table_ang[(rg, a)][n] / table_ang[(rg, 0)][n] for n in ("LF", "LS", "RS", "RF")]
            rel = float(np.mean(vals))
            ang_d_mm.append(float(rg))
            ang_theta.append(float(abs(a)))
            ang_ratio.append(z0 * rel)

    out = dict(
        d_axis_mm=SURFACE_D_MM, theta_axis_deg=SURFACE_TH_DEG, Z=Z,
        dist_meas_d_mm=dist_meas_d_mm, dist_meas_theta=dist_meas_theta,
        dist_meas_ratio=dist_meas_ratio,
        dist_fit_offset_mm=np.array([dist_off]), dist_fit_gain=np.array([dist_gain]),
        dist_fit_rms_pct=np.array([dist_rms * 100.0]),
        ang_d_mm=np.array(ang_d_mm), ang_theta=np.array(ang_theta), ang_ratio=np.array(ang_ratio),
    )
    SURFACE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SURFACE_CACHE_PATH, **out)
    return out


def format_dtheta_text(sensor_dtheta: dict) -> str:
    """「LS 64mm/15°｜RS 57mm/15°」のように、壁が視野に入っているセンサだけ数値で出す。"""
    parts = []
    for name in SENSOR_ORDER:
        v = sensor_dtheta.get(name)
        if v is None:
            continue
        d_mm, theta_deg = v
        parts.append(f"{name} {d_mm:.0f}mm/{abs(theta_deg):.0f}°")
    return "   ｜   ".join(parts) if parts else "（壁が視野に無い）"


class SurfacePanel:
    """応答曲面（距離×入射角×満量比）パネル。

    面は初期化時に matplotlib で 1 回だけ描いて画像として持ち、以後は使い回す
    （教授セッション指示: 毎コマ描き直さない＝ちらつかせない）。走行中の点は、
    面と同じ投影行列（`ax.get_proj()`）で 2 次元へ落とし、PIL で直描きする
    （軽い処理なので毎コマ呼べる）。
    """

    def __init__(self, rect: tuple, dpi: int = 100):
        self.rect = rect
        x0, y0, x1, y1 = rect
        self.w, self.h = x1 - x0, y1 - y0
        data = compute_surface_data()
        self.data = data

        # 教授セッションが 1020x500 で調整した値（レイアウト是正 2026-08-22）。
        # `fig.add_axes` で軸を図いっぱいに置き、表題・凡例は `fig.transFigure` で
        # 図の中に重ねる（軸の外に置くと余白ができ、面が枠の中で小さくなってしまう）。
        fig = plt.figure(figsize=(self.w / dpi, self.h / dpi), dpi=dpi)
        fig.patch.set_facecolor(_rgb01(SEC_BG))
        ax = fig.add_axes([-0.015, 0.015, 1.00, 0.955], projection="3d")
        ax.set_facecolor(_rgb01(SEC_BG))

        D = data["d_axis_mm"]; TH = data["theta_axis_deg"]; Z = data["Z"]
        Dg, THg = np.meshgrid(D, TH, indexing="ij")
        ax.plot_surface(Dg, THg, Z, cmap="viridis", alpha=0.88, linewidth=0,
                         antialiased=True, rcount=60, ccount=30)
        ax.scatter(data["dist_meas_d_mm"], data["dist_meas_theta"], data["dist_meas_ratio"],
                    color="white", s=10, depthshade=False, label="実機（距離）")
        ax.scatter(data["ang_d_mm"], data["ang_theta"], data["ang_ratio"],
                    color="white", s=10, depthshade=False, marker="^", label="実機（角度）")

        ax.set_xlabel("距離[mm]", color=_rgb01(TEXT_DIM), fontsize=11, labelpad=1)
        ax.set_ylabel("入射角[deg]", color=_rgb01(TEXT_DIM), fontsize=11, labelpad=1)
        # z軸ラベルは付けない（目盛の数字と重なる）。縦軸の意味は表題側に書く
        # （レイアウト是正 2026-08-22 その2。ラベルぶんの余白を空けずに済むので、
        # そのぶん box_aspect の横を伸ばして幅を使い切れる）。
        ax.tick_params(axis="both", colors=_rgb01(TEXT_DIM), labelsize=8, pad=0)
        try:
            for axis3d in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis3d.pane.set_facecolor(_rgb01(SEC_BG))
                axis3d.pane.set_alpha(1.0)
                axis3d.line.set_color(_rgb01(GRID_COLOR))
                axis3d._axinfo["grid"]["color"] = _rgb01(GRID_COLOR)
        except Exception:
            pass   # 見た目の微調整のみ。matplotlib版差で失敗しても致命的ではない

        rms_pct = float(data["dist_fit_rms_pct"][0])
        fig.text(0.008, 0.985,
                  f"応答曲面（距離×入射角、縦軸＝満量比）　白＝実機の実測"
                  f"（距離／角度、当てはめ残差{rms_pct:.2f}%）",
                  transform=fig.transFigure, ha="left", va="top",
                  fontsize=13.5, color=_rgb01(TEXT_WHITE))
        ax.legend(loc="upper left", bbox_to_anchor=(0.005, 0.945), bbox_transform=fig.transFigure,
                   fontsize=10, framealpha=0.0, labelcolor=_rgb01(TEXT_WHITE), handletextpad=0.3,
                   borderpad=0.2, borderaxespad=0.0)

        ax.view_init(elev=19, azim=-64)
        # 横(距離)を伸ばして枠の幅を使い切る（是正2026-08-22その2。
        # z軸ラベルを外した余白ぶんも使えるので、更に zoom を上げた。
        # 実測: outputs/video_sensor/frames/ch2_frame0060.png・ch2_frame0710.png で
        # 左右の空き計 120px 以内・はみ出し無しを確認済み）。
        try:
            ax.set_box_aspect((1.90, 1.0, 0.95), zoom=1.25)
        except TypeError:
            ax.set_box_aspect((2.6, 1.0, 0.82))   # 旧matplotlibはzoom引数が無い
        except Exception:
            pass
        fig.canvas.draw()

        self._proj = ax.get_proj()
        self._ax = ax
        self._fig = fig
        buf = np.asarray(fig.canvas.buffer_rgba())
        self._fig_h, self._fig_w = buf.shape[0], buf.shape[1]
        self._bg = Image.fromarray(buf[:, :, :3].copy(), "RGB").resize((self.w, self.h))

    def _project(self, d_mm: float, theta_deg: float, ratio: float) -> tuple:
        x2, y2, _ = proj3d.proj_transform(d_mm, theta_deg, ratio, self._proj)
        dxp, dyp = self._ax.transData.transform((x2, y2))
        px = dxp / self._fig_w * self.w
        py = self.h - dyp / self._fig_h * self.h
        return px, py

    def render(self, draw: ImageDraw.ImageDraw, img: Image.Image, points: dict, dtheta_text: str):
        """`points`: {sensor_name: (d_mm, theta_deg, ratio)}（壁が視野に無いセンサは含めない）。"""
        x0, y0, x1, y1 = self.rect
        img.paste(self._bg, (x0, y0))
        for name, (d_mm, theta_deg, ratio) in points.items():
            px, py = self._project(d_mm, abs(theta_deg), ratio)
            color = SENSOR_COLORS[name]
            draw.ellipse([x0 + px - 7, y0 + py - 7, x0 + px + 7, y0 + py + 7],
                         outline=(255, 255, 255, 255), width=2, fill=color)
        # 軸の目盛と重なって読みにくくならないよう、文字の下に薄い帯を敷く。
        txt_font = font(18, bold=True)
        tb = draw.textbbox((x0 + 14, y1 - 12), dtheta_text, font=txt_font, anchor="lb")
        draw.rectangle([tb[0] - 8, tb[1] - 4, tb[2] + 8, tb[3] + 4], fill=(20, 21, 24, 190))
        draw.text((x0 + 14, y1 - 12), dtheta_text, font=txt_font, fill=TEXT_WHITE, anchor="lb")


# ==========================================================================
# 字幕・帯
# ==========================================================================
def draw_status_bar(draw, rect, text: str):
    draw.rectangle(rect, fill=PANEL_BG)
    x0, y0, x1, y1 = rect
    draw.text((x0 + 20, (y0 + y1) / 2), text, font=font(24), fill=TEXT_WHITE, anchor="lm")


def draw_caption_bar(draw, rect, lines, color=TEXT_CAPTION, size=26):
    draw.rectangle(rect, fill=PANEL_BG)
    x0, y0, x1, y1 = rect
    if not lines:
        return
    fnt = font(size)
    line_h = size + 10
    total_h = line_h * len(lines)
    ty = (y0 + y1) / 2 - total_h / 2 + line_h / 2
    for ln in lines:
        draw.text(((x0 + x1) / 2, ty), ln, font=fnt, fill=color, anchor="mm")
        ty += line_h


def new_canvas() -> Image.Image:
    return Image.new("RGB", (CANVAS_W, CANVAS_H), BG[:3])


# ==========================================================================
# ffmpeg への直接書き出し（rawvideo パイプ）
# ==========================================================================
class FfmpegWriter:
    def __init__(self, out_path: Path, fps: int = FPS, w: int = CANVAS_W, h: int = CANVAS_H):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.w, self.h = w, h
        cmd = [
            FFMPEG, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{w}x{h}", "-framerate", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium",
            "-crf", "18",
            str(out_path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, img: Image.Image):
        assert img.size == (self.w, self.h), f"frame size {img.size} != {(self.w, self.h)}"
        self.proc.stdin.write(img.tobytes())

    def close(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg が失敗しました（code={ret}）")


# ==========================================================================
# 章1・章2共通: メインレイアウト（左=迷路俯瞰、右=棒・時系列・応答曲線）
# ==========================================================================
MAZE_RECT = (0, 60, 900, 960)   # 左列全体の外枠（章1の注釈ボックス用。中は全体像+拡大に分割）
# 右列の内訳（教授セッション指示 2026-08-22・レイアウト是正: 面が1020x340では小さすぎて
# 読めなかった。面を主役にし、棒と時系列を少し縮めて高さを面へ回す）。
BARS_RECT = (900, 60, 1920, 300)    # 240px（旧280px）
TS_RECT = (900, 300, 1920, 460)     # 160px（旧280px。少し高く）
CURVE_RECT = (900, 460, 1920, 960)  # 500px（旧340px。ここを主役に）
STATUS_RECT = (0, 0, 1920, 60)
CAPTION_RECT = (0, 960, 1920, 1080)

# 左列の内訳: 上＝迷路全体（小さく・位置の把握用）、下＝機体まわり拡大（主役）。
OVERVIEW_SIZE = 220
OVERVIEW_X0, OVERVIEW_Y0 = 20, 92
ZOOM_SIZE = 600
ZOOM_X0 = (900 - ZOOM_SIZE) // 2
ZOOM_Y0 = 348
ZOOM_EXTENT_CELLS = 3.0  # 機体を中心に3区画（cell_size×3。180mm規格なら540mm）四方を拡大


def render_main_frame(rd: RunData, specs, ratios: np.ndarray, ranges_m: np.ndarray,
                       x: float, y: float, yaw: float, sim_time: float, phase: str,
                       run_index: int, v_fwd: float, n_of_total: str,
                       trail_xs, trail_ys, hist_t, hist_ratios: dict,
                       surface_panel: "SurfacePanel", sensor_dtheta: dict, status_extra: str = "",
                       highlight_sensor: str = None) -> Image.Image:
    img = new_canvas()
    draw = ImageDraw.Draw(img, "RGBA")

    # 左上＝迷路全体（小さく残す。どこを走っているかの把握用。走った跡もそのまま描く）
    draw.text((OVERVIEW_X0, OVERVIEW_Y0 - 20), "全体（迷路16×16）", font=font(15),
               fill=TEXT_DIM, anchor="la")
    ov_panel = MazePanel(OVERVIEW_X0, OVERVIEW_Y0, OVERVIEW_SIZE, rd.width, rd.height, rd.cell_size)
    draw_maze_panel_bg(draw, ov_panel)
    draw_maze_walls(draw, ov_panel, rd)
    draw_trail(draw, ov_panel, trail_xs, trail_ys, width=2)
    draw_robot(draw, ov_panel, x, y, yaw)

    # 左下＝機体まわり拡大（主役）。センサ4本の光がどの壁に当たっているかを見せる。
    extent_m = ZOOM_EXTENT_CELLS * rd.cell_size
    draw.text((ZOOM_X0, ZOOM_Y0 - 24),
               f"機体まわり拡大（3×3区画 約{extent_m * 1000:.0f}mm角）", font=font(18, bold=True),
               fill=TEXT_WHITE, anchor="la")
    zoom_img = render_zoom_inset(rd, specs, x, y, yaw, ranges_m, ZOOM_SIZE, extent_m,
                                  trail_xs=trail_xs, trail_ys=trail_ys)
    img.paste(zoom_img, (ZOOM_X0, ZOOM_Y0))

    ratio_map = {name: float(ratios[i]) for i, name in enumerate(SENSOR_ORDER)}
    draw_bars(draw, BARS_RECT, ratio_map)
    draw_timeseries(draw, TS_RECT, hist_t, hist_ratios, sim_time)
    points = {name: (sensor_dtheta[name][0], sensor_dtheta[name][1], ratio_map[name])
              for name in SENSOR_ORDER if sensor_dtheta.get(name) is not None}
    surface_panel.render(draw, img, points, format_dtheta_text(sensor_dtheta))

    status = (f"模擬時刻 {sim_time:7.2f}s ｜ 走行段階: {PHASE_JA.get(phase, phase)} ｜ "
              f"走行回数: {run_index} ｜ 機体速度: {v_fwd:5.2f} m/s ｜ {n_of_total}")
    if status_extra:
        status += "  " + status_extra
    draw_status_bar(draw, STATUS_RECT, status)

    if highlight_sensor:
        color = SENSOR_COLORS[highlight_sensor]
        x0, y0, x1, y1 = BARS_RECT
        draw.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], outline=color, width=4)
    return img


def concat_videos(clip_paths, out_path: Path):
    out_path = Path(out_path)
    list_path = out_path.parent / "_concat_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{Path(p).resolve()}'\n")
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
           "-i", str(list_path), "-c", "copy", str(out_path)]
    subprocess.run(cmd, check=True)
