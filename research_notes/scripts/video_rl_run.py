# research_notes/scripts/video_rl_run.py
# 学習方策の走行動画（2026-08-10、学生B）。
#
# レイアウトは学生A の L0 系動画（_video_l0_common.py）に合わせる:
#   1920x1080、左 1080x1080 = 走行、右 840x1080 = 情報パネル。
# ただし右パネルの最下段は **モータ電圧の時系列** にした（教授指示）。学習方策の
# 行動振動（exp_005 最終モデルで符号反転 75.6 回/s）は数値で言われても実感しにくいが、
# 電圧波形を出すと一目で分かる。古典 L0-c の滑らかな波形と並べると差が際立つ。
#
# 出力: outputs/videos/rl_run_<tag>.mp4
# 実行: .venv/bin/python research_notes/scripts/video_rl_run.py \
#           --model models/exp_005_collision_penalty.zip --tag exp005
import argparse
import math
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
from mouse.params import RobotParams  # noqa: E402
from _video_l0_common import (  # noqa: E402  レイアウトの一貫性のため流用
    _load_font, render_gauges_panel, render_velocity_graph,
    PANEL_BG, SEC_BG_ALT, TEXT_WHITE, TEXT_DIM, TEXT_ACCENT, TEXT_FAIL,
)

VIEW = 1080
RIGHT_W = 840
OUT_W, OUT_H = VIEW + RIGHT_W, 1080
FPS = 20                 # 制御 100 Hz を 20 fps ＝ 5 倍スロー
HOLD_FRAMES = 50
GRAPH_WINDOW_S = 3.0     # 電圧グラフの表示窓 [s]（走行が数秒なので短め）

COURSE_DIR = REPO_ROOT / "assets" / "corridor" / "eval"
COURSE_SEED = 3010
TRIAL_SEED = 1000 * 1_000_000 + COURSE_SEED * 100

COLOR_PATH = (110, 118, 130, 255)
COLOR_TRAIL = (10, 132, 255, 255)
COLOR_ROBOT = (255, 59, 48, 255)
COLOR_VL = (10, 132, 255, 255)      # 左モータ電圧
COLOR_VR = (255, 149, 0, 255)       # 右モータ電圧


def collect_run(model_path, course_seed=COURSE_SEED):
    """1 走行ぶんの姿勢・速度・電圧・距離センサを記録する。"""
    params = RobotParams()
    env = CorridorEnv(course_dir=str(COURSE_DIR), course_seeds=[course_seed],
                      obs_dist_diff=True, max_cache=1)
    obs, _info = env.reset(seed=TRIAL_SEED)
    model = PPO.load(str(REPO_ROOT / model_path))

    root_jid = mujoco.mj_name2id(env.sim.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    qadr = env.sim.model.jnt_qposadr[root_jid]
    n_dist = len(params.sensors)
    names = [s["name"] for s in params.sensors]

    rec = dict(qpos=[env.sim.data.qpos.copy()], t=[0.0], v=[0.0], omega=[0.0],
               vl=[0.0], vr=[0.0], ranges=[env.sim.observation()[0:n_dist].copy()],
               flips=0)
    prev_action = np.zeros(2)
    outcome = "timeout"

    for _ in range(6000):
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        for k in (0, 1):
            if prev_action[k] * action[k] < 0.0:
                rec["flips"] += 1
        prev_action = action

        obs, _r, term, trunc, info = env.step(action)
        # MouseSim.privileged_velocity() は (機体前方向の速度 [m/s], ヨー角速度 [rad/s])
        v_fwd, wz = env.sim.privileged_velocity()

        rec["qpos"].append(env.sim.data.qpos.copy())
        rec["t"].append(info["sim_time"])
        rec["v"].append(v_fwd)
        rec["omega"].append(wz)
        rec["vl"].append(action[0] * params.voltage_limit)
        rec["vr"].append(action[1] * params.voltage_limit)
        rec["ranges"].append(env.sim.observation()[0:n_dist].copy())
        if term or trunc:
            outcome = "goal" if info.get("goal") else (
                "collision" if info.get("collision") else "timeout")
            break

    rec.update(outcome=outcome, course=env.course, mj_model=env.sim.model,
               cell_size=params.cell_size, qadr=qadr, sensor_names=names,
               n_cells=env.course["n_cells"])
    env.close()
    return rec


def make_camera(course, cell_size):
    w, h = course["width"], course["height"]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [w * cell_size / 2, h * cell_size / 2, 0]
    cam.azimuth, cam.elevation = 90, -90
    cam.distance = max(w, h) * cell_size * 1.18
    return cam


def world_to_screen(x, y, cam, fovy_deg, size):
    half_h = cam.distance * math.tan(math.radians(fovy_deg) / 2.0)
    scale = (size / 2.0) / half_h
    return (size / 2.0 + (x - cam.lookat[0]) * scale,
            size / 2.0 - (y - cam.lookat[1]) * scale)


def render_voltage_graph(width, height, t_hist, vl_hist, vr_hist, flips_per_s,
                          font, font_small):
    """左右モータ電圧の時系列。行動振動を目で見るための図。"""
    img = Image.new("RGBA", (width, height), PANEL_BG)
    draw = ImageDraw.Draw(img)
    pad_l, pad_r, pad_t, pad_b = 58, 18, 46, 44
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    v_lim = 3.2

    draw.text((16, 8), "モータ電圧の時系列（直近3秒）", font=font_small, fill=TEXT_WHITE)
    draw.text((16, 26), f"符号反転 {flips_per_s:.0f} 回/s", font=font_small,
              fill=TEXT_FAIL if flips_per_s >= 20 else TEXT_ACCENT)
    # 凡例（色だけに頼らないよう名前を併記）
    draw.line([(width - 210, 16), (width - 180, 16)], fill=COLOR_VL, width=3)
    draw.text((width - 174, 8), "V_L", font=font_small, fill=COLOR_VL)
    draw.line([(width - 120, 16), (width - 90, 16)], fill=COLOR_VR, width=3)
    draw.text((width - 84, 8), "V_R", font=font_small, fill=COLOR_VR)

    draw.rectangle([pad_l, pad_t, pad_l + pw, pad_t + ph], outline=(90, 90, 95, 255), width=1)
    for val in (-3.0, -1.5, 0.0, 1.5, 3.0):
        gy = pad_t + ph / 2 - (val / v_lim) * (ph / 2)
        col = (120, 122, 128, 255) if val == 0.0 else (55, 58, 63, 255)
        draw.line([(pad_l, gy), (pad_l + pw, gy)], fill=col, width=1)
        draw.text((6, gy - 8), f"{val:+.1f}", font=font_small, fill=TEXT_DIM)

    if len(t_hist) >= 2:
        t_now = t_hist[-1]
        for hist, color in ((vl_hist, COLOR_VL), (vr_hist, COLOR_VR)):
            pts = []
            for t, v in zip(t_hist, hist):
                if t_now - t > GRAPH_WINDOW_S:
                    continue
                fx = 1.0 - (t_now - t) / GRAPH_WINDOW_S
                fy = max(-1.0, min(1.0, v / v_lim))
                pts.append((pad_l + fx * pw, pad_t + ph / 2 - fy * (ph / 2)))
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=2)

    draw.text((pad_l, pad_t + ph + 10), f"-{GRAPH_WINDOW_S:.0f}s", font=font_small, fill=TEXT_DIM)
    draw.text((pad_l + pw - 26, pad_t + ph + 10), "now", font=font_small, fill=TEXT_DIM)
    return img


def render_status(width, height, rec, idx, title, font_big, font, font_small):
    img = Image.new("RGBA", (width, height), SEC_BG_ALT)
    draw = ImageDraw.Draw(img)
    draw.text((20, 14), title, font=font_big, fill=TEXT_WHITE)
    t = rec["t"][idx]
    draw.text((20, 62), f"経過 {t:5.2f} s", font=font, fill=TEXT_WHITE)
    draw.text((20, 100), f"コース seed {rec['course']['seed']}（{rec['n_cells']} 区画）",
              font=font_small, fill=TEXT_DIM)
    # 「1 区画あたり」は走行が終わってからでないと意味を持たない（途中で出すと、
    # まだ通っていない区画まで分母に入るため不当に小さく見える）。完走時のみ出す。
    finished = idx >= len(rec["qpos"]) - 1
    if finished and rec["outcome"] == "goal" and rec["n_cells"]:
        sec_per_cell = t / rec["n_cells"]
        draw.text((330, 62), f"1 区画 {sec_per_cell:.3f} s", font=font, fill=TEXT_ACCENT)
        draw.text((330, 100), "L0-a は 1.24 s/区画", font=font_small, fill=TEXT_DIM)

    if finished:
        if rec["outcome"] == "goal":
            draw.text((20, 140), "ゴール（完走）", font=font, fill=(52, 199, 89, 255))
        elif rec["outcome"] == "collision":
            draw.text((20, 140), "壁に接触", font=font, fill=TEXT_FAIL)
        else:
            draw.text((20, 140), "時間切れ", font=font, fill=TEXT_ACCENT)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description="学習方策の走行動画")
    ap.add_argument("--model", default="models/exp_005_collision_penalty.zip")
    ap.add_argument("--tag", default="exp005")
    ap.add_argument("--title", default="学習方策 exp_005（センサ→モータ電圧を直接出力）")
    args = ap.parse_args(argv)

    out_path = REPO_ROOT / "outputs" / "videos" / f"rl_run_{args.tag}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    font_big, font, font_small = _load_font(30), _load_font(26), _load_font(19)
    print(f"[run] 走行を記録中… model={args.model}")
    rec = collect_run(args.model)
    n = len(rec["qpos"])
    total_t = rec["t"][-1]
    flips_per_s = rec["flips"] / (2 * total_t) if total_t > 0 else 0.0
    print(f"  {n-1} ステップ ({total_t:.2f} s) → {rec['outcome']}、"
          f"符号反転 {flips_per_s:.1f} 回/s（左右平均）")

    cam = make_camera(rec["course"], rec["cell_size"])
    fovy = float(rec["mj_model"].vis.global_.fovy)
    # オフスクリーンのフレームバッファは XML 既定（800x800）のままだと 1080 を描けない。
    # Renderer 生成の**前に** Python 側で上書きする（video_l0_run.py と同じ手当て）。
    rec["mj_model"].vis.global_.offwidth = VIEW
    rec["mj_model"].vis.global_.offheight = VIEW
    renderer = mujoco.Renderer(rec["mj_model"], height=VIEW, width=VIEW)
    data = mujoco.MjData(rec["mj_model"])

    path_pts = [world_to_screen((cx + 0.5) * rec["cell_size"], (cy + 0.5) * rec["cell_size"],
                                cam, fovy, VIEW) for cx, cy in rec["course"]["path"]]

    writer = imageio.get_writer(str(out_path), fps=FPS, macro_block_size=1, quality=8)
    for f in range(n + HOLD_FRAMES):
        idx = min(f, n - 1)
        data.qpos[:] = rec["qpos"][idx]
        data.qvel[:] = 0.0
        mujoco.mj_forward(rec["mj_model"], data)
        renderer.update_scene(data, camera=cam)
        view = Image.fromarray(renderer.render()).convert("RGBA")

        d = ImageDraw.Draw(view, "RGBA")
        d.line(path_pts, fill=COLOR_PATH, width=12, joint="curve")
        sx, sy = path_pts[0]
        d.ellipse([sx - 11, sy - 11, sx + 11, sy + 11], fill=(52, 199, 89, 200))
        gx, gy = path_pts[-1]
        d.ellipse([gx - 13, gy - 13, gx + 13, gy + 13], fill=(255, 149, 0, 220))
        trail = [world_to_screen(q[rec["qadr"]], q[rec["qadr"] + 1], cam, fovy, VIEW)
                 for q in rec["qpos"][:idx + 1]]
        if len(trail) >= 2:
            d.line(trail, fill=COLOR_TRAIL, width=6, joint="curve")
        px, py = trail[-1]
        d.ellipse([px - 9, py - 9, px + 9, py + 9], fill=COLOR_ROBOT)

        panel = Image.new("RGBA", (RIGHT_W, OUT_H), PANEL_BG)
        panel.paste(render_status(RIGHT_W, 200, rec, idx, args.title,
                                  font_big, font, font_small), (0, 0))
        ranges_mm = {nm: float(rec["ranges"][idx][i]) * 1000.0
                     for i, nm in enumerate(rec["sensor_names"])}
        panel.paste(render_gauges_panel(RIGHT_W, 300, rec["v"][idx], rec["omega"][idx],
                                        rec["vl"][idx], rec["vr"][idx], ranges_mm,
                                        rec["sensor_names"], font, font_small), (0, 200))
        panel.paste(render_velocity_graph(RIGHT_W, 280,
                                          list(zip(rec["t"][:idx + 1], rec["v"][:idx + 1])),
                                          1.0, font_small), (0, 500))
        panel.paste(render_voltage_graph(RIGHT_W, 300, rec["t"][:idx + 1],
                                         rec["vl"][:idx + 1], rec["vr"][:idx + 1],
                                         flips_per_s, font, font_small), (0, 780))

        canvas = Image.new("RGBA", (OUT_W, OUT_H), PANEL_BG)
        canvas.paste(view, (0, 0))
        canvas.paste(panel, (VIEW, 0))
        writer.append_data(np.asarray(canvas.convert("RGB")))
        if f % 60 == 0:
            print(f"  frame {f}/{n + HOLD_FRAMES}")

    writer.close()
    renderer.close()
    print(f"[run] 保存: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
