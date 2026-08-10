# L0（古典ベースライン AdachiPolicy）のX投稿用デモ動画を生成する
# Render a short demo video (for X/Twitter) of the classical L0 baseline
# (AdachiPolicy, flood-fill) navigating maze_1004.xml, top-down offscreen.
#
# 使い方 / Usage:
#   .venv/bin/python research_notes/scripts/video_l0_run.py
#
# 前提: competition/mazes/eval/maze_1004.{npz,xml} が既に存在すること
# （本スクリプトは新規生成しない。存在確認のみ行う）。
# competition/evaluator.py の evaluate_maze（272-470行目付近）と同じ
# FREE/RUN_ACTIVE 状態機械を簡略再実装し、full_reset() の時刻0から
# 最初にゴールへ到達する瞬間まで（1走行目の探索・スタック・係員回収
# → 2走行目のゴール到達）を、真上固定カメラ・実時間対応（sim_time と
# 動画再生時間を1:1対応）で録画する。ロジックの食い違いを防ぐため、
# 座標判定・スタック検出は evaluator.py の関数をそのまま import して使う。
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

from competition.evaluator import (  # noqa: E402, F401
    in_start_region, in_goal_region, pos_to_cell, goal_cells,
    bfs_distances_from_targets, PositionRingBuffer,
    DEFAULT_STUCK_WINDOW_S, DEFAULT_STUCK_DISP_M, DEFAULT_CELL_SIZE,
)
from mouse.params import RobotParams  # noqa: E402
from mouse.sim import MouseSim  # noqa: E402
from competition.baseline_classical import AdachiPolicy  # noqa: E402


MAZE_NPZ_PATH = Path("competition/mazes/eval/maze_1004.npz")
OUT_VIDEO_PATH = Path("outputs/videos/l0_maze_run.mp4")

FRAME_SIZE = 1280            # 出力フレーム一辺 [px]（迷路が正方形なので正方形出力）
FPS = 30
FRAME_DT = 1.0 / FPS
CAMERA_DISTANCE = 4.6        # 実測検証済み: 16x16迷路全体が適度な余白付きで収まる値。変更しないこと
SAFETY_TIME_LIMIT_S = 120.0  # 安全弁: 想定(43.5s付近)を大きく超えたら異常とみなし打ち切る
HOLD_FRAMES_AT_GOAL = 30     # ゴール到達フレームを複製保持するフレーム数（1秒 @30fps）


def _load_font(size: int):
    """視認性重視のフォントを読み込む。等幅系（Menlo/SFNSMono）優先、
    無ければヒラギノ角ゴシック W4（index=0）にフォールバックする。"""
    candidates = [
        ("/System/Library/Fonts/Menlo.ttc", 0),
        ("/System/Library/Fonts/SFNSMono.ttf", 0),
        ("/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", 0),
    ]
    for path, index in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size, index=index)
    return ImageFont.load_default()


def make_camera(width: int, height: int, cell_size: float) -> mujoco.MjvCamera:
    """迷路全体が常に入る固定の真上俯瞰カメラを作る
    （docs/generate_demo.py の TopDownRenderer パターンを踏襲）。"""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cx = width * cell_size / 2
    cy = height * cell_size / 2
    cam.lookat[:] = [cx, cy, 0]
    cam.azimuth = 90
    cam.elevation = -90
    cam.distance = CAMERA_DISTANCE
    return cam


def render_frame_with_overlay(renderer, cam, data, sim_time: float, font) -> np.ndarray:
    """1フレームをレンダリングし、右下にシミュレーション時刻
    （t = 12.3 s 形式、小数点以下1桁）を半透明の黒背景付きで重畳する。"""
    renderer.update_scene(data, camera=cam)
    frame = renderer.render()  # (H, W, 3) uint8 RGB

    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    text = f"t = {sim_time:.1f} s"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    margin = 24
    pad = 14
    x2 = img.width - margin
    y2 = img.height - margin
    x1 = x2 - text_w - 2 * pad
    y1 = y2 - text_h - 2 * pad

    draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 170))
    draw.text((x1 + pad - bbox[0], y1 + pad - bbox[1]), text, font=font, fill=(255, 255, 255, 255))

    composited = Image.alpha_composite(img, overlay).convert("RGB")
    return np.array(composited)


def run_and_record() -> dict:
    """evaluate_maze の状態機械（FREE/RUN_ACTIVE）を簡略再実装しつつ走行させ、
    full_reset() の時刻0から最初のゴール到達までを実時間対応で録画する。
    戻り値: goal_sim_time・タイムライン・書き出しフレーム数等の記録 dict。"""
    xml_path = MAZE_NPZ_PATH.with_suffix(".xml")
    if not MAZE_NPZ_PATH.exists() or not xml_path.exists():
        raise FileNotFoundError(f"{MAZE_NPZ_PATH} または {xml_path} が見つかりません")

    data_npz = np.load(MAZE_NPZ_PATH)
    v_walls = data_npz["v_walls"]
    h_walls = data_npz["h_walls"]
    seed = int(data_npz["seed"])
    width = int(data_npz["width"])
    height = int(data_npz["height"])

    params = RobotParams()
    sim = MouseSim(str(xml_path), params=params)

    # オフスクリーンバッファを拡大（XML ファイル自体は変更せず、ロード後に
    # Python 側で model.vis.global_ を上書きする。Renderer 作成前に必須）。
    sim.model.vis.global_.offwidth = FRAME_SIZE
    sim.model.vis.global_.offheight = FRAME_SIZE
    renderer = mujoco.Renderer(sim.model, height=FRAME_SIZE, width=FRAME_SIZE)
    cam = make_camera(width, height, DEFAULT_CELL_SIZE)
    font = _load_font(40)

    sim.full_reset(cell=(0, 0), heading_deg=90.0)

    policy = AdachiPolicy()  # 引数無し・既定構成（13.44s を再現する構成）
    policy.bind_sim(sim)
    policy.bind_maze(v_walls, h_walls)
    policy.on_maze_start({
        "maze_id": MAZE_NPZ_PATH.stem, "seed": seed,
        "xml_path": str(xml_path), "width": width, "height": height,
    })

    cell_size = params.cell_size
    ring = PositionRingBuffer(window_s=DEFAULT_STUCK_WINDOW_S)

    OUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(OUT_VIDEO_PATH), fps=FPS, macro_block_size=1)

    x, y, _yaw = sim.privileged_pose()
    prev_in_start = in_start_region(x, y, cell_size)
    segment_start_t = sim.sim_time
    ring.push(sim.sim_time, x, y)

    state = "FREE"
    run_count = 0
    timeline = []  # 実測タイムライン記録（報告用。run_start/run_end イベント列）
    next_frame_time = 0.0
    n_frames_written = 0
    goal_sim_time = None
    step_i = 0

    wall_clock_start = time.time()

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

        # 実時間対応フレームキャプチャ: sim_time >= next_frame_time になるたびに1枚
        # （ゴール到達の瞬間は下の専用キャプチャに一本化し、二重キャプチャを避ける）
        while t >= next_frame_time and outcome != "goal":
            frame = render_frame_with_overlay(renderer, cam, sim.data, t, font)
            writer.append_data(frame)
            n_frames_written += 1
            next_frame_time += FRAME_DT

        if state == "RUN_ACTIVE" and outcome is not None:
            policy.on_run_end(outcome)  # AdachiPolicy では no-op だが評価器と同じ呼び出し順を踏襲
            if outcome == "goal":
                goal_sim_time = t
                timeline.append({"event": "run_end", "run": run_count, "outcome": "goal", "t": t})
                final_frame = render_frame_with_overlay(renderer, cam, sim.data, t, font)
                for _ in range(HOLD_FRAMES_AT_GOAL):
                    writer.append_data(final_frame)
                    n_frames_written += 1
                break
            else:
                timeline.append({"event": "run_end", "run": run_count, "outcome": outcome, "t": t})
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
        if step_i % 500 == 0:
            print(f"  ... sim_time={t:6.2f}s  state={state:10s}  run_count={run_count}  "
                  f"frames={n_frames_written}")

    writer.close()
    renderer.close()
    wall_clock = time.time() - wall_clock_start

    return {
        "goal_sim_time": goal_sim_time,
        "timeline": timeline,
        "n_frames_written": n_frames_written,
        "wall_clock_s": wall_clock,
    }


def main():
    print("=" * 60)
    print("L0 (AdachiPolicy) maze_1004 デモ動画生成")
    print("=" * 60)
    result = run_and_record()

    print("\n--- タイムライン（実測） ---")
    for ev in result["timeline"]:
        if ev["event"] == "run_start":
            print(f"  run {ev['run']}: 出発 t={ev['t']:.2f}s")
        else:
            print(f"  run {ev['run']}: 終了 outcome={ev['outcome']:10s} t={ev['t']:.2f}s")

    print("\n--- 結果 ---")
    if result["goal_sim_time"] is not None:
        print(f"ゴール到達時 sim_time = {result['goal_sim_time']:.2f} s （既知参考値: 43.52 s）")
    else:
        print("[WARNING] ゴールに到達しないままループが終了しました（安全弁作動の可能性）")
    print(f"書き出しフレーム数: {result['n_frames_written']} "
          f"（{result['n_frames_written'] / FPS:.1f} s @ {FPS}fps）")
    print(f"出力: {OUT_VIDEO_PATH.resolve()}")
    print(f"実処理時間（wall clock）: {result['wall_clock_s']:.1f} s")


if __name__ == "__main__":
    main()
