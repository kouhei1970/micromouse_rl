# research_notes/scripts/video_rl_smooth_beforeafter.py
# 行動振動の改善前後を左右に並べる比較動画（2026-08-11、学生B）。
#
# 教授指示（ユーザ要望）: exp_005（改善前）と 案3 k=8.7e-3（改善後）を、
# **同一コース・同一時刻**で同時再生し、モータ電圧と電流の時系列を並べる。
#
# なぜ電圧と電流を出すか:
#   符号反転［回/s］は「指令の形」の指標であって害の実体ではない。害の実体は
#   **熱と機械的打撃**であり、その限界は仕様書にある（連続定格 0.586 A）。
#   電流波形を出すと「**捨てている電流**」が減る様子がそのまま見える。
#   I = (V − K_e·N·ω_w)/R（docs/MODEL_VERIFICATION_PLAN.md §4.1）。
#
# 改善後の seed 選定について（隠さず明記する）:
#   案3 k=8.7e-3 は **2/3 seed** が成功し、そのうち **seed3 を選んだ**。
#   理由は符号反転が最小（11.6 回/s）で目標値 10 に最も近いため。
#   **最良の 1 本を選んでいる**ことを動画中にも表示する。
#
# 出力: outputs/videos/rl_smooth_before_after.mp4
# 実行: .venv/bin/python research_notes/scripts/video_rl_smooth_beforeafter.py
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
    _load_font, PANEL_BG, SEC_BG_ALT, TEXT_WHITE, TEXT_DIM, TEXT_ACCENT, TEXT_FAIL,
)

# --- レイアウト -------------------------------------------------------------
VIEW = 700                      # 走行ビュー（正方形）×2
GRAPH_H = 210                   # 電圧・電流グラフの高さ
HEAD_H = 92                     # 各側の見出し
FOOT_H = 186                    # 下部の総括バー（注記 2 行を含む）
OUT_W = VIEW * 2
OUT_H = HEAD_H + VIEW + GRAPH_H * 2 + FOOT_H
FPS = 20                        # 制御 100 Hz を 20 fps ＝ 5 倍スロー
HOLD_FRAMES = 60
GRAPH_WINDOW_S = 2.0

COURSE_DIR = REPO_ROOT / "assets" / "corridor" / "eval"
COURSE_SEED = 3010
TRIAL_SEED = 1000 * 1_000_000 + COURSE_SEED * 100

COLOR_PATH = (110, 118, 130, 255)
COLOR_TRAIL = (10, 132, 255, 255)
COLOR_ROBOT = (255, 59, 48, 255)
COLOR_L = (10, 132, 255, 255)       # 左モータ
COLOR_R = (255, 149, 0, 255)        # 右モータ
COLOR_LIMIT = (255, 92, 82, 200)    # 連続定格の線
DIVIDER = (70, 74, 80, 255)

# (見出し, モデル, 副題)
BEFORE = ("改善前", "models/exp_006_control_k0.zip",
          "exp_005 相当（罰なし）")
AFTER = ("改善後", "models/exp_006d_hp_k8.7e-3_seed3.zip",
         "案3: 高周波成分への罰 k=8.7e-3, α=0.5")


def collect_run(model_path, course_seed=COURSE_SEED):
    """1 走行ぶんの姿勢・電圧・電流を記録する。

    電流は I = (V − K_e·N·ω_w)/R。制御周期内で V は一定、ω_w はステップ前後の平均で
    代表させる（周期 10 ms に対し車輪の時定数 20.6 ms なので誤差は小さい）。
    """
    p = RobotParams()
    env = CorridorEnv(course_dir=str(COURSE_DIR), course_seeds=[course_seed],
                      obs_dist_diff=True, max_cache=1)
    obs, _ = env.reset(seed=TRIAL_SEED)
    model = PPO.load(str(REPO_ROOT / model_path), device="cpu")

    root_jid = mujoco.mj_name2id(env.sim.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    qadr = env.sim.model.jnt_qposadr[root_jid]
    s = env.sim
    wheel_adr = (s._left_wheel_qvel_adr, s._right_wheel_qvel_adr)

    rec = dict(qpos=[s.data.qpos.copy()], t=[0.0], v=[0.0],
               vl=[0.0], vr=[0.0], il=[0.0], ir=[0.0], flips=0)
    prev_action, outcome = np.zeros(2), "timeout"

    for _ in range(6000):
        a, _ = model.predict(obs, deterministic=True)
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        rec["flips"] += int((prev_action[0] * a[0] < 0) + (prev_action[1] * a[1] < 0))
        prev_action = a

        v = a * p.voltage_limit
        w0 = np.array([s.data.qvel[wheel_adr[0]], s.data.qvel[wheel_adr[1]]])
        obs, _r, term, trunc, info = env.step(a)
        w1 = np.array([s.data.qvel[wheel_adr[0]], s.data.qvel[wheel_adr[1]]])
        cur = (v - p.motor_Ke * p.gear_ratio * 0.5 * (w0 + w1)) / p.motor_R

        rec["qpos"].append(s.data.qpos.copy())
        rec["t"].append(info["sim_time"])
        rec["v"].append(s.privileged_velocity()[0])
        rec["vl"].append(v[0]); rec["vr"].append(v[1])
        rec["il"].append(cur[0]); rec["ir"].append(cur[1])
        if term or trunc:
            outcome = ("goal" if info.get("goal") else
                       "collision" if info.get("collision") else "timeout")
            break

    n_steps = len(rec["t"]) - 1
    total_t = rec["t"][-1]
    cur_arr = np.array([rec["il"][1:], rec["ir"][1:]])
    rec.update(outcome=outcome, course=env.course, mj_model=s.model,
               cell_size=p.cell_size, qadr=qadr, n_cells=env.course["n_cells"],
               n_steps=n_steps, total_t=total_t,
               flips_per_s=rec["flips"] / (2 * total_t) if total_t > 0 else 0.0,
               i_rms=float(np.sqrt((cur_arr ** 2).mean())),
               sec_per_cell=(total_t / env.course["n_cells"]) if outcome == "goal" else None)
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


def render_series(width, height, t_hist, a_hist, b_hist, title, unit, lim,
                  font, font_small, limit_val=None, limit_label=None):
    """左右 2 本の時系列。電圧にも電流にも使う。"""
    img = Image.new("RGBA", (width, height), PANEL_BG)
    d = ImageDraw.Draw(img)
    pad_l, pad_r, pad_t, pad_b = 62, 16, 34, 26
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    d.text((14, 7), title, font=font_small, fill=TEXT_WHITE)
    d.line([(width - 196, 14), (width - 168, 14)], fill=COLOR_L, width=3)
    d.text((width - 162, 6), "左", font=font_small, fill=COLOR_L)
    d.line([(width - 118, 14), (width - 90, 14)], fill=COLOR_R, width=3)
    d.text((width - 84, 6), "右", font=font_small, fill=COLOR_R)

    d.rectangle([pad_l, pad_t, pad_l + pw, pad_t + ph], outline=(90, 90, 95, 255), width=1)
    for val in (-lim, -lim / 2, 0.0, lim / 2, lim):
        gy = pad_t + ph / 2 - (val / lim) * (ph / 2)
        col = (120, 122, 128, 255) if val == 0.0 else (55, 58, 63, 255)
        d.line([(pad_l, gy), (pad_l + pw, gy)], fill=col, width=1)
        d.text((6, gy - 8), f"{val:+.1f}", font=font_small, fill=TEXT_DIM)
    d.text((6, pad_t + ph + 6), unit, font=font_small, fill=TEXT_DIM)

    # 連続定格などの基準線（電流のときだけ）
    if limit_val is not None:
        for sgn in (1, -1):
            gy = pad_t + ph / 2 - (sgn * limit_val / lim) * (ph / 2)
            if pad_t <= gy <= pad_t + ph:
                for x0 in range(int(pad_l), int(pad_l + pw), 14):
                    d.line([(x0, gy), (x0 + 7, gy)], fill=COLOR_LIMIT, width=2)
        if limit_label:
            d.text((pad_l + 8, pad_t + 4), limit_label, font=font_small, fill=COLOR_LIMIT)

    if len(t_hist) >= 2:
        t_now = t_hist[-1]
        for hist, color in ((a_hist, COLOR_L), (b_hist, COLOR_R)):
            pts = []
            for t, y in zip(t_hist, hist):
                if t_now - t > GRAPH_WINDOW_S:
                    continue
                fx = 1.0 - (t_now - t) / GRAPH_WINDOW_S
                fy = max(-1.0, min(1.0, y / lim))
                pts.append((pad_l + fx * pw, pad_t + ph / 2 - fy * (ph / 2)))
            if len(pts) >= 2:
                d.line(pts, fill=color, width=2)
    return img


def render_head(width, height, label, model_sub, rec, idx, font_big, font, font_small,
                accent):
    img = Image.new("RGBA", (width, height), SEC_BG_ALT)
    d = ImageDraw.Draw(img)
    d.text((18, 10), label, font=font_big, fill=accent)
    d.text((18, 52), model_sub, font=font_small, fill=TEXT_DIM)
    t = rec["t"][min(idx, len(rec["t"]) - 1)]
    d.text((width - 250, 14), f"経過 {t:5.2f} s", font=font, fill=TEXT_WHITE)
    finished = idx >= rec["n_steps"]
    if finished:
        if rec["outcome"] == "goal":
            d.text((width - 250, 52), "ゴール", font=font_small, fill=(52, 199, 89, 255))
        else:
            d.text((width - 250, 52), "失敗", font=font_small, fill=TEXT_FAIL)
    return img


def render_footer(width, height, before, after, i_cont, font_big, font, font_small):
    """改善前後の数値を同じ画面に並べる総括バー。"""
    img = Image.new("RGBA", (width, height), SEC_BG_ALT)
    d = ImageDraw.Draw(img)
    cols = [
        ("符号反転", f"{before['flips_per_s']:.1f} 回/s", f"{after['flips_per_s']:.1f} 回/s",
         before["flips_per_s"] / max(after["flips_per_s"], 1e-9)),
        ("RMS 電流", f"{before['i_rms']:.2f} A（定格 {before['i_rms']/i_cont:.2f} 倍）",
         f"{after['i_rms']:.2f} A（定格 {after['i_rms']/i_cont:.2f} 倍）",
         before["i_rms"] / max(after["i_rms"], 1e-9)),
        ("銅損 I²R", f"{before['i_rms']**2*1.07*2:.2f} W", f"{after['i_rms']**2*1.07*2:.2f} W",
         (before["i_rms"] / max(after["i_rms"], 1e-9)) ** 2),
        ("1 区画所要",
         f"{before['sec_per_cell']:.3f} s" if before["sec_per_cell"] else "—",
         f"{after['sec_per_cell']:.3f} s" if after["sec_per_cell"] else "—",
         (before["sec_per_cell"] / after["sec_per_cell"])
         if (before["sec_per_cell"] and after["sec_per_cell"]) else None),
    ]
    cw = width // len(cols)
    top = 46
    # この画面が「1 走行」であることと、正式な集計値を先に明示する。
    # 単一走行の値を成果の代表値と読み違えないようにするため（コースごとに振れる）。
    d.text((18, 8),
           "この画面はコース seed 3010 の 1 走行。"
           "正式な成績は gate 帯 20 コース×5 試行 = 100 試行の平均:",
           font=font_small, fill=TEXT_WHITE)
    d.text((18, 26),
           "　符号反転 75.6 → 11.6 回/s ／ RMS 電流 2.09 → 0.72 A（定格 3.56 → 1.23 倍）"
           " ／ 1 区画 0.171 → 0.142 s ／ gate 完走率 0.93 → 0.94",
           font=font_small, fill=TEXT_DIM)
    for i, (name, bv, av, ratio) in enumerate(cols):
        x = i * cw
        if i:
            d.line([(x, top + 2), (x, height - 26)], fill=DIVIDER, width=1)
        d.text((x + 18, top), name, font=font_small, fill=TEXT_DIM)
        d.text((x + 18, top + 24), bv, font=font_small, fill=TEXT_FAIL)
        d.text((x + 18, top + 50), av, font=font, fill=(52, 199, 89, 255))
        if ratio is not None:
            d.text((x + 18, top + 86), f"この走行で {ratio:.1f} 倍",
                   font=font_small, fill=TEXT_ACCENT)
    d.text((18, height - 24),
           "改善後は 3 seed 中 2 seed が成功。うち符号反転が最小の seed3 を表示している"
           "（最良の 1 本を選んでいる）",
           font=font_small, fill=TEXT_DIM)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description="行動振動の改善前後の比較動画")
    ap.add_argument("--before", default=BEFORE[1])
    ap.add_argument("--after", default=AFTER[1])
    ap.add_argument("--out", default="outputs/videos/rl_smooth_before_after.mp4")
    args = ap.parse_args(argv)

    p = RobotParams()
    i_cont = 1.16e-3 / p.motor_Kt          # 連続定格電流 0.586 A（ROBOT_SPEC §3）
    f_big, f_mid, f_sm = _load_font(30), _load_font(23), _load_font(17)

    print("[run] 改善前を記録中…")
    rb = collect_run(args.before)
    print(f"  {rb['n_steps']} ステップ ({rb['total_t']:.2f} s) → {rb['outcome']}、"
          f"反転 {rb['flips_per_s']:.1f} 回/s、RMS 電流 {rb['i_rms']:.3f} A")
    print("[run] 改善後を記録中…")
    ra = collect_run(args.after)
    print(f"  {ra['n_steps']} ステップ ({ra['total_t']:.2f} s) → {ra['outcome']}、"
          f"反転 {ra['flips_per_s']:.1f} 回/s、RMS 電流 {ra['i_rms']:.3f} A")

    cam = make_camera(rb["course"], rb["cell_size"])
    fovy = float(rb["mj_model"].vis.global_.fovy)
    # オフスクリーンのフレームバッファは XML 既定（800x800）。Renderer 生成の**前に**上書きする
    for r in (rb, ra):
        r["mj_model"].vis.global_.offwidth = VIEW
        r["mj_model"].vis.global_.offheight = VIEW
    renderers = [mujoco.Renderer(r["mj_model"], height=VIEW, width=VIEW) for r in (rb, ra)]
    datas = [mujoco.MjData(r["mj_model"]) for r in (rb, ra)]

    path_pts = [world_to_screen((cx + .5) * rb["cell_size"], (cy + .5) * rb["cell_size"],
                                cam, fovy, VIEW) for cx, cy in rb["course"]["path"]]

    n = max(rb["n_steps"], ra["n_steps"]) + 1
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=FPS, macro_block_size=1, quality=8)

    i_lim = max(3.0, np.ceil(max(rb["i_rms"], ra["i_rms"]) * 2.2))
    for f in range(n + HOLD_FRAMES):
        canvas = Image.new("RGBA", (OUT_W, OUT_H), PANEL_BG)
        for side, (rec, rend, dat, meta, accent) in enumerate(
                ((rb, renderers[0], datas[0], BEFORE, TEXT_FAIL),
                 (ra, renderers[1], datas[1], AFTER, (52, 199, 89, 255)))):
            idx = min(f, rec["n_steps"])
            x0 = side * VIEW

            canvas.paste(render_head(VIEW, HEAD_H, meta[0], meta[2], rec, idx,
                                     f_big, f_mid, f_sm, accent), (x0, 0))

            dat.qpos[:] = rec["qpos"][idx]
            dat.qvel[:] = 0.0
            mujoco.mj_forward(rec["mj_model"], dat)
            rend.update_scene(dat, camera=cam)
            view = Image.fromarray(rend.render()).convert("RGBA")
            dv = ImageDraw.Draw(view, "RGBA")
            dv.line(path_pts, fill=COLOR_PATH, width=9, joint="curve")
            sx, sy = path_pts[0]
            dv.ellipse([sx - 8, sy - 8, sx + 8, sy + 8], fill=(52, 199, 89, 200))
            gx, gy = path_pts[-1]
            dv.ellipse([gx - 9, gy - 9, gx + 9, gy + 9], fill=(255, 149, 0, 220))
            trail = [world_to_screen(q[rec["qadr"]], q[rec["qadr"] + 1], cam, fovy, VIEW)
                     for q in rec["qpos"][:idx + 1]]
            if len(trail) >= 2:
                dv.line(trail, fill=COLOR_TRAIL, width=4, joint="curve")
            px, py = trail[-1]
            dv.ellipse([px - 7, py - 7, px + 7, py + 7], fill=COLOR_ROBOT)
            canvas.paste(view, (x0, HEAD_H))

            ts = rec["t"][:idx + 1]
            canvas.paste(render_series(
                VIEW, GRAPH_H, ts, rec["vl"][:idx + 1], rec["vr"][:idx + 1],
                f"モータ電圧（直近 {GRAPH_WINDOW_S:.0f} 秒）　"
                f"符号反転 {rec['flips_per_s']:.1f} 回/s",
                "V", 3.2, f_mid, f_sm), (x0, HEAD_H + VIEW))
            canvas.paste(render_series(
                VIEW, GRAPH_H, ts, rec["il"][:idx + 1], rec["ir"][:idx + 1],
                f"モータ電流　RMS {rec['i_rms']:.2f} A"
                f"（連続定格の {rec['i_rms']/i_cont:.2f} 倍）",
                "A", i_lim, f_mid, f_sm,
                limit_val=i_cont, limit_label=f"連続定格 ±{i_cont:.2f} A"),
                (x0, HEAD_H + VIEW + GRAPH_H))

        canvas.paste(render_footer(OUT_W, FOOT_H, rb, ra, i_cont, f_big, f_mid, f_sm),
                     (0, OUT_H - FOOT_H))
        d = ImageDraw.Draw(canvas)
        d.line([(VIEW, 0), (VIEW, OUT_H - FOOT_H)], fill=DIVIDER, width=2)
        writer.append_data(np.asarray(canvas.convert("RGB")))
        if f % 60 == 0:
            print(f"  frame {f}/{n + HOLD_FRAMES}", flush=True)

    writer.close()
    for r in renderers:
        r.close()
    print(f"[run] 保存: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
