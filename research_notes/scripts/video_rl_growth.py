# research_notes/scripts/video_rl_growth.py
# 強化学習の成長過程を 1 本の動画にする（2026-08-10、学生B）。
#
# 同一コース・同一初期条件で、学習の各段階のモデルを走らせて 2x2 に並べる。
# 「センサ入力から左右モータ電圧を直接出す方策が、どこまで走れるようになったか」
# を数値ではなく動きで見せるのが目的（ユーザ要望「どうなっているのか追えていない」）。
#
# 段の構成（exp_003b の学習軌跡から採る）:
#   学習前（ランダム方策） / 10 万 / 20 万 / 60 万ステップ
# 20 万ステップは検証帯完走率が 0.00 まで落ちた時期（報酬設計の滞留の局所解に
# はまり「動かずに時間切れまで粘る」方策になっていた）。学習は一直線に進む
# わけではない、という事実もそのまま見せる。
#
# 出力: outputs/videos/rl_growth.mp4
# 実行: .venv/bin/python research_notes/scripts/video_rl_growth.py
import math
import os
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO  # noqa: E402

from mouse.corridor_env import CorridorEnv  # noqa: E402
from _video_l0_common import _load_font  # noqa: E402  レイアウトの一貫性のため流用

# --- 出力 ---
OUT_PATH = REPO_ROOT / "outputs" / "videos" / "rl_growth.mp4"
TILE = 540                  # 各段の走行ビュー（正方形）
OUT_SIZE = TILE * 2         # 1080x1080（正方形）
FPS = 20                    # 制御 100 Hz のデータを 20 fps で再生 ＝ 5 倍スロー
HOLD_FRAMES = 40            # 末尾の静止（2 秒）

# --- 走らせるコース（gate 帯の代表 1 本）---
COURSE_DIR = REPO_ROOT / "assets" / "corridor" / "eval"
COURSE_SEED = 3010          # 15 区画・6 ターン（直線・旋回・S 字が一通り入る）
TRIAL_SEED = 1000 * 1_000_000 + COURSE_SEED * 100  # corridor_eval と同じ導出（trial 0）

# --- 色（_video_l0_common の配色に合わせる）---
PANEL_BG = (24, 26, 30, 255)
TEXT_WHITE = (240, 240, 240, 255)
TEXT_DIM = (170, 170, 175, 255)
TEXT_ACCENT = (255, 214, 10, 255)
TEXT_GOOD = (52, 199, 89, 255)
TEXT_FAIL = (255, 92, 82, 255)
TEXT_STALL = (255, 159, 10, 255)

COLOR_PATH = (110, 118, 130, 255)        # 正解経路（走るべき道）
COLOR_TRAIL = (10, 132, 255, 255)        # 走行軌跡
COLOR_ROBOT = (255, 59, 48, 255)         # 現在位置

# note は「このコースでの実際の結果」を書く。検証帯完走率は 20 本平均なので、
# 個々のコースの成否と一致しないことがある（例: 10 万ステップは平均 0.80 だが
# このコースでは壁に当たる）。両方を並べて出し、混同しないようにする。
STAGES = [
    dict(key="pre",  label="学習前（ランダム方策）", steps=0,       model=None,
         val=None, note="0.5 秒で壁へ"),
    dict(key="100k", label="10 万ステップ",          steps=100_000,
         model="models/exp_003_sensor_history_smoke.zip", val=0.80,
         note="1.2 秒ぶん走れる"),
    dict(key="200k", label="20 万ステップ",          steps=200_000,
         model="logs/exp_003b_single_env/rl_model_200000_steps.zip", val=0.00,
         note="速度が落ち 2.1 秒"),
    dict(key="600k", label="60 万ステップ",          steps=600_000,
         model="logs/exp_003b_single_env/rl_model_600000_steps.zip", val=0.96,
         note="完走 2.9 秒（M1 合格）"),
]


def run_stage(stage, course_seed=COURSE_SEED):
    """1 段ぶんの走行を実行し、姿勢履歴と結果を返す。

    レンダリングは後段でまとめて行うため、ここでは qpos の写しだけ貯める。
    """
    env = CorridorEnv(course_dir=str(COURSE_DIR), course_seeds=[course_seed],
                      obs_dist_diff=True, max_cache=1)
    obs, _info = env.reset(seed=TRIAL_SEED)

    model = None
    if stage["model"] is not None:
        model = PPO.load(str(REPO_ROOT / stage["model"]))

    # 学習前は PPO の初期方策からサンプリングする（＝実際の学習開始時の挙動）。
    # 学習済みモデルは評価と同じ決定的方策で走らせる。
    rng = np.random.default_rng(0)

    root_jid = mujoco.mj_name2id(env.sim.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    qadr = env.sim.model.jnt_qposadr[root_jid]
    nq = env.sim.model.nq

    qpos_hist = [env.sim.data.qpos.copy()]
    action_hist = []
    outcome = "timeout"

    for _ in range(6000):
        if model is None:
            action = rng.uniform(-1.0, 1.0, size=2)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(action)
        qpos_hist.append(env.sim.data.qpos.copy())
        action_hist.append(np.clip(np.asarray(action, dtype=np.float64), -1, 1))
        if term or trunc:
            outcome = "goal" if info.get("goal") else ("collision" if info.get("collision") else "timeout")
            break

    course = env.course
    model_ref, cell = env.sim.model, env.params.cell_size
    env.close()
    return dict(stage=stage, qpos=qpos_hist, actions=action_hist, outcome=outcome,
                course=course, mj_model=model_ref, cell_size=cell, qadr=qadr, nq=nq)


def make_camera(course, cell_size):
    """コース全体が収まる俯瞰カメラ。"""
    w, h = course["width"], course["height"]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [w * cell_size / 2, h * cell_size / 2, 0]
    cam.azimuth = 90
    cam.elevation = -90
    # 正方形の画に長辺が収まる距離。上部のラベル帯にスタート地点が隠れないよう
    # 余白を広めに取る（1.28 倍は実測で決めた値）
    cam.distance = max(w, h) * cell_size * 1.28
    return cam


def world_to_screen(x, y, cam, fovy_deg, tile):
    """ワールド座標 [m] → 走行ビュー内のピクセル座標。

    カメラは真上からの俯瞰（elevation=−90, azimuth=90）なので、画面の右が +x、
    上が +y になる。透視投影だが視距離が十分に遠いため、注視面（z=0）上では
    一様な倍率で扱ってよい。
    """
    half_h = cam.distance * math.tan(math.radians(fovy_deg) / 2.0)
    scale = (tile / 2.0) / half_h
    sx = tile / 2.0 + (x - cam.lookat[0]) * scale
    sy = tile / 2.0 - (y - cam.lookat[1]) * scale
    return sx, sy


def draw_course_path(img, course, cell_size, cam, fovy_deg):
    """走るべき道（コースのセル中心を結んだ折れ線）を薄く敷く。"""
    draw = ImageDraw.Draw(img, "RGBA")
    pts = [world_to_screen((cx + 0.5) * cell_size, (cy + 0.5) * cell_size,
                           cam, fovy_deg, TILE) for cx, cy in course["path"]]
    if len(pts) >= 2:
        draw.line(pts, fill=COLOR_PATH, width=9, joint="curve")
    # 始点と終点
    sx, sy = pts[0]
    draw.ellipse([sx - 9, sy - 9, sx + 9, sy + 9], fill=(52, 199, 89, 200))
    gx, gy = pts[-1]
    draw.ellipse([gx - 11, gy - 11, gx + 11, gy + 11], fill=(255, 149, 0, 220))
    return img


def draw_trail(img, qpos_hist, idx, qadr, cam, fovy_deg):
    """走行軌跡（現在フレームまで）と現在位置を描く。"""
    draw = ImageDraw.Draw(img, "RGBA")
    pts = [world_to_screen(q[qadr], q[qadr + 1], cam, fovy_deg, TILE)
           for q in qpos_hist[:idx + 1]]
    if len(pts) >= 2:
        draw.line(pts, fill=COLOR_TRAIL, width=5, joint="curve")
    px, py = pts[-1]
    draw.ellipse([px - 8, py - 8, px + 8, py + 8], fill=COLOR_ROBOT)
    return img


def draw_tile_overlay(img, stage, outcome, frame_idx, n_frames, font, font_small, font_big):
    """1 段ぶんの走行ビューにラベル・完走率・状態を重ねる。"""
    draw = ImageDraw.Draw(img, "RGBA")
    # 上部の帯（文字が走行ビューに埋もれないように）
    draw.rectangle([0, 0, TILE, 88], fill=(18, 20, 24, 205))
    draw.text((16, 10), stage["label"], font=font_big, fill=TEXT_WHITE)

    if stage["val"] is None:
        val_txt = "検証帯 完走率 —"
        val_col = TEXT_DIM
    else:
        val_txt = f"検証帯 完走率 {stage['val']:.2f}"
        val_col = TEXT_GOOD if stage["val"] >= 0.8 else (
            TEXT_FAIL if stage["val"] < 0.3 else TEXT_ACCENT)
    draw.text((16, 56), val_txt, font=font_small, fill=val_col)
    draw.text((16 + 260, 56), stage["note"], font=font_small, fill=TEXT_DIM)

    # 走行が終わった段は、終了理由を画面中央下に出したまま静止する
    if frame_idx >= n_frames - 1:
        if outcome == "goal":
            txt, col = "ゴール", TEXT_GOOD
        elif outcome == "collision":
            txt, col = "壁に接触", TEXT_FAIL
        else:
            txt, col = "時間切れ（停止したまま）", TEXT_STALL
        bbox = draw.textbbox((0, 0), txt, font=font)
        tw = bbox[2] - bbox[0]
        x0 = (TILE - tw) // 2
        draw.rectangle([x0 - 14, TILE - 62, x0 + tw + 14, TILE - 18], fill=(18, 20, 24, 220))
        draw.text((x0, TILE - 56), txt, font=font, fill=col)
    return img


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    font_big = _load_font(30)
    font = _load_font(26)
    font_small = _load_font(19)

    print("[growth] 各段の走行を実行中…")
    runs = []
    for stage in STAGES:
        r = run_stage(stage)
        runs.append(r)
        n = len(r["qpos"]) - 1
        print(f"  {stage['label']}: {n} ステップ ({n*0.01:.2f} 秒) → {r['outcome']}")

    n_frames = max(len(r["qpos"]) for r in runs)
    total_frames = n_frames + HOLD_FRAMES
    print(f"[growth] 最長 {n_frames} フレーム → 出力 {total_frames} フレーム "
          f"({total_frames/FPS:.1f} 秒 @{FPS}fps)")

    # コースは全段同一なので Renderer は 1 つでよい（qpos を差し替えて描画する）
    base = runs[0]
    renderer = mujoco.Renderer(base["mj_model"], height=TILE, width=TILE)
    cam = make_camera(base["course"], base["cell_size"])
    data = mujoco.MjData(base["mj_model"])

    fovy = float(base["mj_model"].vis.global_.fovy)
    writer = imageio.get_writer(str(OUT_PATH), fps=FPS, macro_block_size=1,
                                quality=8)
    for f in range(total_frames):
        canvas = Image.new("RGBA", (OUT_SIZE, OUT_SIZE), PANEL_BG)
        for i, r in enumerate(runs):
            idx = min(f, len(r["qpos"]) - 1)
            data.qpos[:] = r["qpos"][idx]
            data.qvel[:] = 0.0
            mujoco.mj_forward(base["mj_model"], data)
            renderer.update_scene(data, camera=cam)
            tile = Image.fromarray(renderer.render()).convert("RGBA")
            tile = draw_course_path(tile, r["course"], r["cell_size"], cam, fovy)
            tile = draw_trail(tile, r["qpos"], idx, r["qadr"], cam, fovy)
            tile = draw_tile_overlay(tile, r["stage"], r["outcome"], idx,
                                     len(r["qpos"]), font, font_small, font_big)
            canvas.paste(tile, ((i % 2) * TILE, (i // 2) * TILE))

        draw = ImageDraw.Draw(canvas, "RGBA")
        for k in range(1, 2):  # 段の境界線
            draw.line([(k * TILE, 0), (k * TILE, OUT_SIZE)], fill=(70, 74, 80, 255), width=2)
            draw.line([(0, k * TILE), (OUT_SIZE, k * TILE)], fill=(70, 74, 80, 255), width=2)

        writer.append_data(np.asarray(canvas.convert("RGB")))
        if f % 50 == 0:
            print(f"  frame {f}/{total_frames}")

    writer.close()
    renderer.close()
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"[growth] 保存: {OUT_PATH} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
