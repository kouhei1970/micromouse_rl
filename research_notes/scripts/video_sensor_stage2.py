"""
research_notes/scripts/video_sensor_stage2.py
================
発表用動画（センサの読みの可視化）段階2: 動画を作る。

`outputs/video_sensor/run_data.npz`（段階1の記録・seed 41201）から、教授セッションが
確定した区間だけを使って章ごとに書き出し、最後に連結する。

章立て（教授セッション確定・2026-08-22。ナレーション追加により尺は音声の実測長へ
差し替えた。「作り直しは映像の側を音声に合わせる」— 詳細は
`research_notes/scripts/video_sensor_narration/script.md` と
`build_narration.py`。各章の尺は `outputs/video_sensor/narration/ch{N}.mp3` の
実測長から決まる。ハードコードした秒数ではなく、`vc.paragraph_durations()` /
`vc.chapter_audio_duration()` で毎回測り直す）:
  第0章 表題と前提（音声にあわせた静止スライド。P0=導入、P1=前提4つ）
  第1章 画面の見方（音声にあわせて4段階の注釈。P0=俯瞰、P1前半=棒、P1後半=時系列、P2=応答曲面）
  第2章 センサの読みを見る（1ステップ=1コマ・全ステップぶん再計算。
    模擬794.0〜816.12秒・2212ステップに伸ばした。伸ばした区間も衝突・転倒なしを確認済み）
  第3章 計算の速さ（音声にあわせて「説明の静止画→4レーン並走(実時計14秒・変えない)→
    結果の静止画」の3部構成。並走中は無音、その前後に音声を割り当てる）
  第4章 まとめ（音声にあわせた静止スライド。P0=表、P1=結論の一文をハイライト）

使い方（前景で、章ごとに分割して実行。1回の呼び出しは10分以内。
先にナレーションを作っておくこと: `video_sensor_narration/build_narration.py`）:
  .venv/bin/python research_notes/scripts/video_sensor_stage2.py --chapter 0
  .venv/bin/python research_notes/scripts/video_sensor_stage2.py --chapter 1
  .venv/bin/python research_notes/scripts/video_sensor_stage2.py --chapter 2
  .venv/bin/python research_notes/scripts/video_sensor_stage2.py --chapter 3
  .venv/bin/python research_notes/scripts/video_sensor_stage2.py --chapter 4
  .venv/bin/python research_notes/scripts/video_sensor_stage2.py --concat

🔴 `mouse/`・`classic/`・`competition/` のコードは一切変更しない。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _video_sensor_common as vc  # noqa: E402

FPS = vc.FPS
W, H = vc.CANVAS_W, vc.CANVAS_H

# ---- 章2・章3で使う区間 ----
# CH2_T0 は教授セッション確定（変えない）。CH2_T1 はナレーション追加にともない、
# 音声に合わせて 806.0→816.12 秒へ伸ばした
# （2026-08-22。1ステップ=1コマ・30fpsなので 2212 ステップ ≒ 73.73秒 ≧ 音声長。
# 応答曲面パネルへの作り直し（同日）で ch2.mp3 は実測 73.21秒に変わったが、
# これでもまだ映像(73.73秒)以下なので CH2_T1 はそのまま。
# 伸ばした区間 794.0-816.12 秒は run_index=3（最短走行）のまま・衝突/転倒なし・
# 走行終了 816.97 秒より前に収まることを `run_data.npz` で確認済み）。
CH2_T0, CH2_T1 = 794.0, 816.12
CH3_T0, CH3_T1 = 793.9, 817.0     # 23.1秒（第3走行=最短走行1回まるごと・変えない）

# ---- 第3章: 4レーンの進む速さ（教授セッション実測値。固定値として使う） ----
LANES = [
    ("基準（機体の実際の速さ）", 1.00, (100, 210, 255, 255)),
    ("従来 response()", 0.041, (255, 99, 92, 255)),
    ("面積分の高速版 response_fast()", 0.84, (255, 149, 0, 255)),
    ("表引き response_table()", 1.75, (52, 199, 89, 255)),
]
CH3_REAL_MOVE_S = 14.0  # 並走そのものの長さ（実時計。変えない）。ナレーション側の
                        # 無音（build_narration.py の SPECIAL_GAP）もこの値に合わせてある。


def _save_repr_png(img: Image.Image, name: str):
    vc.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    img.save(vc.FRAMES_DIR / f"{name}.png")


# ==========================================================================
# 第0章: 表題と前提（静止スライド。音声の長さにあわせる）
# P0（導入の一文）の間はタイトルだけ、P1（「先に、前提を四つだけ」以降）で
# 前提4つをクロスフェードで出す（ナレーション追加にともなう変更・2026-08-22）。
# ==========================================================================
def chapter0():
    def base_slide():
        img = vc.new_canvas()
        draw = ImageDraw.Draw(img, "RGBA")
        draw.rectangle([0, 0, W, H], fill=vc.BG[:3])
        draw.text((W / 2, 320), "センサモデルを迷路で走らせる", font=vc.font(56, bold=True),
                   fill=vc.TEXT_WHITE, anchor="mm")
        draw.text((W / 2, 400), "— 距離センサ4本の読みと、計算の速さ —", font=vc.font(28),
                   fill=vc.TEXT_DIM, anchor="mm")
        return img

    img_title = base_slide()
    img_full = base_slide()
    draw = ImageDraw.Draw(img_full, "RGBA")
    premises = [
        "① 機体の真の位置は模擬から与えている（方策が推定した位置ではない）",
        "② 壁の反射率 0.8・床の反射は無視（決定どおり）",
        "③ 本動画はセンサの読みと計算の速さを見るもので、走行や探索の方式を見せるものではない",
        "④ 使ったのは最短走行の区間で、衝突・転倒・立ちすくみが無いことを事前に確認してある",
    ]
    # 内容全体を縦方向に中央へ寄せる（教授セッション検収 2026-08-22。以前は上40%に寄っていた）。
    y = 570
    for line in premises:
        draw.text((W / 2, y), line, font=vc.font(30), fill=vc.TEXT_ACCENT, anchor="mm")
        y += 66

    p0_s, p1_s = vc.paragraph_durations(0)
    total_s = vc.chapter_audio_duration(0)
    n_total = int(np.ceil(total_s * FPS))
    fade_in_n = 15          # 開幕フェードイン
    trans_n = 15            # タイトルのみ→前提4つのクロスフェード
    trans_start = int(round((p0_s + vc.PARA_GAP_S) * FPS))

    writer = vc.FfmpegWriter(vc.CHAPTERS_DIR / "ch0.mp4")
    black = Image.new("RGB", (W, H), vc.BG[:3])
    for k in range(n_total):
        if k < fade_in_n:
            frame = Image.blend(black, img_title, (k + 1) / fade_in_n)
        elif k < trans_start:
            frame = img_title
        elif k < trans_start + trans_n:
            frame = Image.blend(img_title, img_full, (k - trans_start + 1) / trans_n)
        else:
            frame = img_full
        writer.write(frame)
        if k in (fade_in_n, n_total - 1):
            _save_repr_png(frame, f"ch0_frame{k:04d}")
    writer.close()
    print(f"第0章 完了（音声{total_s:.2f}s → 映像{n_total/FPS:.2f}s・{n_total}コマ）:",
          vc.CHAPTERS_DIR / "ch0.mp4")


# ==========================================================================
# 第1章: 画面の見方（20秒・4段階の注釈を静止画で見せる）
# ==========================================================================
def chapter1():
    rd = vc.RunData()
    ctx = vc.SensorContext(rd.v_walls, rd.h_walls, rd.cell_size)
    probe = vc.RangefinderProbe(vc.maze_xml_path(rd.seed))
    surface_panel = vc.SurfacePanel(vc.CURVE_RECT)

    t_show = CH2_T0
    x, y, yaw, st, phase, run_idx, v_fwd = rd.pose_at_time(t_show)
    ratios = ctx.response_ratio(x, y, yaw)
    ranges_m = probe.ranges_m(x, y, yaw)
    sensor_dtheta = vc.sensor_d_theta(ctx, x, y, yaw)

    # 直近2秒の時系列（間引きせず短区間だけ計算し直す。安い）
    idx = rd.idx_range(t_show - 2.0, t_show + 0.001)
    hist_t = rd.sim_time[idx]
    hist_ratios = {name: np.empty(len(idx)) for name in vc.SENSOR_ORDER}
    for k, i in enumerate(idx):
        r = ctx.response_ratio(float(rd.x[i]), float(rd.y[i]), float(rd.yaw[i]))
        for j, name in enumerate(vc.SENSOR_ORDER):
            hist_ratios[name][k] = r[j]

    trail_idx = rd.idx_range(793.94, t_show + 0.001)  # 第3走行の開始から

    base = vc.render_main_frame(
        rd, ctx.specs, ratios, ranges_m, x, y, yaw, st, phase, run_idx, v_fwd,
        "1コマ=1ステップ(10ms)", rd.x[trail_idx], rd.y[trail_idx], hist_t, hist_ratios,
        surface_panel, sensor_dtheta,
    )

    def with_box(rect, label, color=(255, 214, 10, 255)):
        im = base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        x0, y0, x1, y1 = rect
        d.rectangle([x0 + 3, y0 + 3, x1 - 3, y1 - 3], outline=color, width=6)
        return im

    def with_caption(im, lines):
        im = im.copy()
        d = ImageDraw.Draw(im, "RGBA")
        vc.draw_caption_bar(d, vc.CAPTION_RECT, lines)
        return im

    # 4段階の注釈。ナレーションの段落3つ(P0=俯瞰、P1=棒→時系列、P2=応答曲面)にあわせる
    # （2026-08-22）。P1は「棒の話」と「時系列の話」の2文を含むので、文字数の比
    # （48字:27字 ≒ 0.64）で内部分割する（Azureの単語単位の時刻は取っていないので
    # 文字数比による近似。数秒のズレは許容範囲）。
    BARS_FRAC = 0.64
    p0_s, p1_s, p2_s = vc.paragraph_durations(1)
    gap = vc.PARA_GAP_S
    stage_defs = [
        (vc.MAZE_RECT, ["左＝迷路の俯瞰。壁・機体・センサ4本の光の向き・走行軌跡（黄）を描く。"],
         p0_s + gap),
        (vc.BARS_RECT, ["右上＝距離センサ4本の読み（満量比。棒と数値）。"],
         p1_s * BARS_FRAC),
        (vc.TS_RECT, ["右中＝直近2秒の時系列。4本の読みの変化を追える。"],
         p1_s * (1 - BARS_FRAC) + gap),
        (vc.CURVE_RECT, ["右下＝距離と入射角で応答が変わる「面」と、いまの読み。",
                          "実機の実測点（白）も、この面の上に重ねてある。",
                          "このセンサは距離を測っていない。明るさを測って距離を推している。"],
         p2_s),
    ]
    total_s = vc.chapter_audio_duration(1)
    frame_counts = vc.allocate_frames([d for _, _, d in stage_defs], total_s, FPS)

    writer = vc.FfmpegWriter(vc.CHAPTERS_DIR / "ch1.mp4")
    for si, ((rect, lines, _), n_frames) in enumerate(zip(stage_defs, frame_counts)):
        frame = with_caption(with_box(rect, lines), lines)
        for k in range(n_frames):
            writer.write(frame)
        _save_repr_png(frame, f"ch1_stage{si}")
    writer.close()
    n_total = sum(frame_counts)
    print(f"第1章 完了（音声{total_s:.2f}s → 映像{n_total/FPS:.2f}s・{n_total}コマ、"
          f"段階ごとのコマ数={frame_counts}）:", vc.CHAPTERS_DIR / "ch1.mp4")


# ==========================================================================
# 第2章: センサの読みを見る（40秒・1ステップ=1コマ・全ステップぶん再計算）
# ==========================================================================
def chapter2():
    rd = vc.RunData()
    ctx = vc.SensorContext(rd.v_walls, rd.h_walls, rd.cell_size)
    probe = vc.RangefinderProbe(vc.maze_xml_path(rd.seed))
    surface_panel = vc.SurfacePanel(vc.CURVE_RECT)

    # 表示区間の前に2秒ぶんの助走を計算し、時系列パネルの立ち上がりを埋める。
    lead = 2.0
    idx = rd.idx_range(CH2_T0 - lead, CH2_T1 + 1e-6)
    n = len(idx)
    print(f"[ch2] 全{n}点ぶんセンサを計算し直す（記録は間引かれている）", flush=True)
    t0 = time.time()
    ratios_all = np.empty((n, 4))
    ranges_all = np.empty((n, 4))
    dtheta_all = [None] * n   # k -> {sensor_name: (d_mm, theta_deg) or None}
    for k, i in enumerate(idx):
        xk, yk, yawk = float(rd.x[i]), float(rd.y[i]), float(rd.yaw[i])
        ratios_all[k] = ctx.response_ratio(xk, yk, yawk)
        ranges_all[k] = probe.ranges_m(xk, yk, yawk)
        dtheta_all[k] = vc.sensor_d_theta(ctx, xk, yk, yawk)
        if (k + 1) % 300 == 0:
            print(f"  {k+1}/{n} ({time.time()-t0:.1f}s)", flush=True)
    print(f"[ch2] センサ計算 完了 {time.time()-t0:.1f}s", flush=True)

    t_all = rd.sim_time[idx]
    disp_mask = t_all >= CH2_T0 - 1e-9
    disp_idx_local = np.where(disp_mask)[0]
    n_disp = len(disp_idx_local)
    # コマ数はハードコードせず区間の長さから出す（ナレーション追加で CH2_T1 を伸ばした
    # ため、以前のように 1200 に固定できない。自己無矛盾になっているかだけ確認する）。
    expected_n = len(rd.idx_range(CH2_T0, CH2_T1))
    assert n_disp == expected_n, (n_disp, expected_n)
    slowdown = (n_disp / FPS) / (CH2_T1 - CH2_T0)  # 常に (1/FPS)/0.01 ≒ 3.33 になるはず

    # 字幕スケジュール（実測に基づく。研究ノート/報告参照。イベント自体の時刻は区間を
    # 伸ばしても動かない — 794.0-816.12秒はどれも run_index=3・最短走行1回の中）:
    #   794.0-796.0: ②壁切れ（LSがt=794.65付近で急落）
    #   796.0-799.0: つなぎ（余白）
    #   799.0-802.5: ①山越え（RSがt=801.0付近で山43mmを越える）
    #   802.5-CH2_T1: ③行き止まり（本区間には無いことを明記した一般論。伸ばした
    #                 区間も同じ最短走行の中＝行き止まりを通らないので変わらない）
    def caption_for(t):
        if t < 796.0:
            return ["② 横の壁が切れると側方センサが落ちる（壁切れ）",
                    "左側方センサ(LS)が t≈794.65s 付近で急に落ちている"]
        if t < 799.0:
            return ["4本の棒・時系列・応答曲面の動きを見比べてみてください"]
        if t < 802.5:
            return ["① 壁に近づくと上がり、近づきすぎると下がる（山＝43mm）",
                    "右側方センサ(RS)が t≈801.0s 付近で山を越えている（丸で強調）"]
        return ["③ 行き止まりでは4本とも上がる",
                "（この区間は最短走行中で行き止まりを通らないため、ここでは見られない）"]

    writer = vc.FfmpegWriter(vc.CHAPTERS_DIR / "ch2.mp4")
    trail_t0 = 793.94  # 第3走行の開始（run_index=3のt_start）
    for fi, li in enumerate(disp_idx_local):
        i = idx[li]
        x, y, yaw = float(rd.x[i]), float(rd.y[i]), float(rd.yaw[i])
        st, phase, run_idx, v_fwd = (float(rd.sim_time[i]), str(rd.phase[i]),
                                      int(rd.run_index[i]), float(rd.v_fwd[i]))
        ratios = ratios_all[li]
        ranges_m = ranges_all[li]
        sensor_dtheta = dtheta_all[li]

        hmask = (t_all >= st - 2.0) & (t_all <= st + 1e-9)
        hist_t = t_all[hmask]
        hist_ratios = {name: ratios_all[hmask, j] for j, name in enumerate(vc.SENSOR_ORDER)}

        trail_idx = rd.idx_range(trail_t0, st + 1e-9)

        highlight = "RS" if 800.9 <= st <= 801.15 else ("LS" if 794.55 <= st <= 794.75 else None)

        frame = vc.render_main_frame(
            rd, ctx.specs, ratios, ranges_m, x, y, yaw, st, phase, run_idx, v_fwd,
            f"{fi+1}/{n_disp}コマ（{slowdown:.1f}倍スロー）", rd.x[trail_idx], rd.y[trail_idx],
            hist_t, hist_ratios, surface_panel, sensor_dtheta, highlight_sensor=highlight,
        )
        d = ImageDraw.Draw(frame, "RGBA")
        vc.draw_caption_bar(d, vc.CAPTION_RECT, caption_for(st))
        if highlight == "RS":
            d.text((vc.CURVE_RECT[2] - 20, vc.CURVE_RECT[1] + 20), "← 山を越える瞬間",
                   font=vc.font(18, bold=True), fill=vc.SENSOR_COLORS[highlight], anchor="ra")
        elif highlight == "LS":
            d.text((vc.BARS_RECT[2] - 20, vc.BARS_RECT[1] + 20), "← 壁切れ",
                   font=vc.font(18, bold=True), fill=vc.SENSOR_COLORS[highlight], anchor="ra")

        writer.write(frame)
        if fi == 0 or fi == n_disp - 1 or highlight is not None:
            _save_repr_png(frame, f"ch2_frame{fi:04d}")
    writer.close()
    print(f"第2章 完了（{n_disp}コマ・{n_disp/FPS:.2f}s）:", vc.CHAPTERS_DIR / "ch2.mp4")


# ==========================================================================
# 第3章: 計算の速さ（4レーン並走・実時計1倍速）
# ==========================================================================
def _lane_panel_rect(row: int, col: int):
    top, bottom = 70, 960
    x0 = col * (W // 2)
    x1 = x0 + W // 2
    y0 = top + row * ((bottom - top) // 2)
    y1 = y0 + (bottom - top) // 2
    return (x0, y0, x1, y1)


def _draw_lane(draw, rd, rect, lane_name, rate, color, sim_t, real_elapsed, trail_mask):
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, fill=vc.SEC_BG)
    header_h = 34
    footer_h = 30  # 「模擬時間/経過した実時間」の帯ぶんを確保（以前は迷路図の下端に文字がかぶっていた）
    draw.text((x0 + 12, y0 + 6), f"{lane_name}（{rate:.2f}倍）", font=vc.font(20, bold=True),
               fill=color, anchor="la")
    size = min(x1 - x0, (y1 - y0) - header_h - footer_h) - 10
    panel = vc.MazePanel(x0 + (x1 - x0 - size) // 2, y0 + header_h, size, rd.width, rd.height,
                          rd.cell_size)
    vc.draw_maze_panel_bg(draw, panel)
    vc.draw_maze_walls(draw, panel, rd)
    vc.draw_trail(draw, panel, rd.x[trail_mask], rd.y[trail_mask], color=color, width=3)
    xi, yi, yawi = rd.x[trail_mask][-1], rd.y[trail_mask][-1], rd.yaw[trail_mask][-1]
    vc.draw_robot(draw, panel, float(xi), float(yi), float(yawi), color=color)
    draw.text((x0 + 12, y1 - 8),
              f"模擬時間 {sim_t - CH3_T0:5.1f}s / 経過した実時間 {real_elapsed:5.1f}s",
              font=vc.font(20, bold=True), fill=vc.TEXT_WHITE, anchor="lb")


CH3_TITLE = "第3走行（最短走行）を4つの計算方式で並走 — 実時計1倍速で再生"


def _lane_mask_at(rd, li, real_elapsed):
    name, rate, color = LANES[li]
    sim_t = min(CH3_T0 + rate * real_elapsed, CH3_T1)
    mask = rd.idx_range(CH3_T0, sim_t + 1e-6)
    if len(mask) == 0:
        mask = np.array([rd.idx_range(CH3_T0, CH3_T0 + 0.02)[0]])
    return sim_t, mask


def _render_lanes_frame(rd, real_elapsed, cap_lines, highlight_li=None):
    img = vc.new_canvas()
    draw = ImageDraw.Draw(img, "RGBA")
    vc.draw_status_bar(draw, vc.STATUS_RECT, CH3_TITLE)
    for li, (name, rate, color) in enumerate(LANES):
        sim_t, mask = _lane_mask_at(rd, li, real_elapsed)
        row, col = divmod(li, 2)
        rect = _lane_panel_rect(row, col)
        _draw_lane(draw, rd, rect, name, rate, color, sim_t, real_elapsed, mask)
        if highlight_li == li:
            x0, y0, x1, y1 = rect
            draw.rectangle([x0 + 3, y0 + 3, x1 - 3, y1 - 3], outline=(255, 214, 10, 255), width=6)
    vc.draw_caption_bar(draw, vc.CAPTION_RECT, cap_lines)
    return img


def chapter3():
    """3部構成（ナレーション追加にともなう変更・2026-08-22）:
    導入（静止画。音声のP0〜P2にあわせる）→ 並走本体（実時計14秒・変えない。音声は無音）
    → 結果（静止画。音声のP3にあわせる）。"""
    rd = vc.RunData()
    p0_s, p1_s, p2_s, p3_s = vc.paragraph_durations(3)
    gap = vc.PARA_GAP_S

    # ---- 導入: 開始状態（模擬時間0.0s）で静止させ、字幕だけ音声にあわせて切り替える ----
    sub_dur = (p1_s + gap) / 4  # P1は4レーンの説明。文で触れる順（左上→右上→左下→右下）に4分割
    intro_defs = [
        (None, ["同じ最短走行・同じ区間を、4つのセンサ計算方式で並べて走らせます。",
                "ここからは実時計の一倍速で再生します。"], p0_s + gap),
        (0, ["左上＝基準。機体の実際の速さ。"], sub_dur),
        (1, ["右上＝はじめに作った計算のしかた（従来 response()）。"], sub_dur),
        (2, ["左下＝それを速くしたやつ（response_fast()）。"], sub_dur),
        (3, ["右下＝あらかじめ計算しておいた表を引くしかた（response_table()）。"], sub_dur),
        (None, ["各レーンの下に、模擬時間と、実際に経った時間を出しています。"], p2_s),
    ]
    intro_durations = [d for _, _, d in intro_defs]
    intro_frames = vc.allocate_frames(intro_durations, sum(intro_durations), FPS)

    # ---- 並走本体: 実時計14秒（CH3_REAL_MOVE_S・変えない）。この間、音声は無音 ----
    race_frames = round(CH3_REAL_MOVE_S * FPS)

    # ---- 結果: 音声の全体長からの残りを結果スライドに割り当てる（P3の実測より少し長め）----
    total_frames = int(np.ceil(vc.chapter_audio_duration(3) * FPS))
    outro_frames = total_frames - sum(intro_frames) - race_frames
    if outro_frames < int(1.0 * FPS):
        raise RuntimeError(f"結果スライドのコマ数が小さすぎる（{outro_frames}コマ）。"
                            f"導入{sum(intro_frames)}+並走{race_frames}が音声全体"
                            f"{total_frames}コマに対して長すぎないか確認すること。")

    writer = vc.FfmpegWriter(vc.CHAPTERS_DIR / "ch3.mp4")

    for si, ((highlight_li, lines, _), n_frames) in enumerate(zip(intro_defs, intro_frames)):
        frame = _render_lanes_frame(rd, 0.0, lines, highlight_li=highlight_li)
        for _ in range(n_frames):
            writer.write(frame)
        _save_repr_png(frame, f"ch3_intro{si}")

    cap_race = ["同じ最短走行・同じ区間を、4つのセンサ計算方式で走らせている",
                "各レーンは表の速さぶんだけ模擬時間を進める（表引きが最も速い）"]
    for fi in range(race_frames):
        real_elapsed = min(fi / FPS, CH3_REAL_MOVE_S)
        frame = _render_lanes_frame(rd, real_elapsed, cap_race)
        writer.write(frame)
        if fi in (0, race_frames - 1):
            _save_repr_png(frame, f"ch3_race{fi:04d}")

    cap_outro = ["表引きだけが最短走行を走り切り、従来のモデルは3区画目（290mm）までしか進めない",
                 "実時間14秒でどこまで進んだか＝計算の速さの差そのもの"]
    frame = _render_lanes_frame(rd, CH3_REAL_MOVE_S, cap_outro)
    for _ in range(outro_frames):
        writer.write(frame)
    _save_repr_png(frame, "ch3_outro")

    writer.close()
    n_total = sum(intro_frames) + race_frames + outro_frames
    print(f"第3章 完了（導入{sum(intro_frames)/FPS:.2f}s + 並走{race_frames/FPS:.2f}s + "
          f"結果{outro_frames/FPS:.2f}s = 映像{n_total/FPS:.2f}s・{n_total}コマ、"
          f"音声{vc.chapter_audio_duration(3):.2f}s）:", vc.CHAPTERS_DIR / "ch3.mp4")


# ==========================================================================
# 第4章: まとめ（静止スライド。音声の長さにあわせる）
# P0（表の説明）の間は表だけ、P1（結論の一文）でその一文をクロスフェードで足す
# （ナレーション追加にともなう変更・2026-08-22）。
# ==========================================================================
def _chapter4_slide(with_conclusion: bool) -> Image.Image:
    img = vc.new_canvas()
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, W, H], fill=vc.BG[:3])
    # 内容全体を縦方向に中央へ寄せる（教授セッション検収 2026-08-22。以前は上40%に寄っていた）。
    draw.text((W / 2, 315), "まとめ", font=vc.font(50, bold=True), fill=vc.TEXT_WHITE, anchor="mm")

    headers = ["センサモデル", "模擬1秒に要する実時間", "進む速さ"]
    rows = [
        ["基準（機体の実際の速さ）", "—", "1.00 倍"],
        ["従来 response()", "24.2 秒", "0.041 倍"],
        ["面積分の高速版 response_fast()", "1.19 秒", "0.84 倍"],
        ["表引き response_table()", "0.57 秒", "1.75 倍"],
    ]
    col_x = [W / 2 - 560, W / 2 - 40, W / 2 + 470]
    y = 445
    for cx, htxt in zip(col_x, headers):
        draw.text((cx, y), htxt, font=vc.font(24, bold=True), fill=vc.TEXT_DIM, anchor="lm")
    # 見出し行(24pt)の下端と1行目(26pt)の上端の間に横線を引く（以前は1行目の文字にかぶっていた）。
    line_y = y + 34
    draw.line([(W / 2 - 580, line_y), (W / 2 + 700, line_y)], fill=vc.GRID_COLOR, width=2)
    y = line_y + 40
    for row in rows:
        for cx, val in zip(col_x, row):
            draw.text((cx, y), val, font=vc.font(26), fill=vc.TEXT_WHITE, anchor="lm")
        y += 54

    if with_conclusion:
        y += 40
        draw.text((W / 2, y), "実機と 0.6% で合う正しさと、実時間より速い速さが両立した",
                   font=vc.font(30, bold=True), fill=vc.TEXT_ACCENT, anchor="mm")
    return img


def chapter4():
    img_table = _chapter4_slide(with_conclusion=False)
    img_full = _chapter4_slide(with_conclusion=True)

    p0_s, p1_s = vc.paragraph_durations(4)
    total_s = vc.chapter_audio_duration(4)
    n_total = int(np.ceil(total_s * FPS))
    fade_in_n = 15
    trans_n = 15
    trans_start = int(round((p0_s + vc.PARA_GAP_S) * FPS))

    writer = vc.FfmpegWriter(vc.CHAPTERS_DIR / "ch4.mp4")
    black = Image.new("RGB", (W, H), vc.BG[:3])
    for k in range(n_total):
        if k < fade_in_n:
            frame = Image.blend(black, img_table, (k + 1) / fade_in_n)
        elif k < trans_start:
            frame = img_table
        elif k < trans_start + trans_n:
            frame = Image.blend(img_table, img_full, (k - trans_start + 1) / trans_n)
        else:
            frame = img_full
        writer.write(frame)
        if k in (fade_in_n, n_total - 1):
            _save_repr_png(frame, f"ch4_frame{k:04d}")
    writer.close()
    print(f"第4章 完了（音声{total_s:.2f}s → 映像{n_total/FPS:.2f}s・{n_total}コマ）:",
          vc.CHAPTERS_DIR / "ch4.mp4")


# ==========================================================================
def do_concat():
    """章ごとの無音映像（ch{i}.mp4）だけを連結する（QA用。ナレーション版の生成には
    使わない — 章ごとに音声を載せてから連結する `do_mux_and_concat()` を使うこと）。
    🔴 出力先は元の無音版 `outputs/video_sensor/sensor_visualization.mp4`
    （尺を伸ばす前の104秒版・タスク指示で保持と指定）を上書きしないよう別名にしてある。"""
    clips = [vc.CHAPTERS_DIR / f"ch{i}.mp4" for i in range(5)]
    for c in clips:
        if not c.exists():
            raise FileNotFoundError(f"{c} が無い（先に各章を書き出すこと）")
    out = vc.OUT_DIR / "sensor_visualization_v2_silent.mp4"
    vc.concat_videos(clips, out)
    print("連結（無音・QA用） 完了:", out)


def do_mux_and_concat():
    """章ごとに映像(ch{i}.mp4)とナレーション(narration/ch{i}.mp3)を合わせてから連結し、
    `sensor_visualization_narrated.mp4` を書き出す（`build_part1.py` と同じ手順:
    音声を映像長ぴったりまで無音で伸ばしてから多重化し、`-c copy` で連結する）。"""
    import subprocess as sp
    av_clips = []
    for i in range(5):
        v = vc.CHAPTERS_DIR / f"ch{i}.mp4"
        a = vc.NARR_DIR / f"ch{i}.mp3"
        if not v.exists():
            raise FileNotFoundError(f"{v} が無い（先に該当章を書き出すこと）")
        if not a.exists():
            raise FileNotFoundError(f"{a} が無い（先に build_narration.py を実行）")
        vd = vc.audio_duration(v)
        ad = vc.audio_duration(a)
        if vd + 1e-3 < ad:
            raise RuntimeError(f"ch{i}: 映像({vd:.3f}s)が音声({ad:.3f}s)より短い"
                                f"（音声が切れる。尺の割り当てを見直すこと）")
        out = vc.CHAPTERS_DIR / f"ch{i}_av.mp4"
        sp.run([
            vc.FFMPEG, "-y", "-loglevel", "error", "-i", str(v), "-i", str(a),
            "-filter_complex", f"[1:a]apad=whole_dur={vd:.3f}[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out),
        ], check=True)
        print(f"  ch{i}: 映像{vd:.3f}s / 音声{ad:.3f}s -> {out}")
        av_clips.append(out)

    out_final = vc.OUT_DIR / "sensor_visualization_narrated.mp4"
    lst = vc.CHAPTERS_DIR / "_concat_av_list.txt"
    with open(lst, "w", encoding="utf-8") as f:
        for p in av_clips:
            f.write(f"file '{p.resolve()}'\n")
    sp.run([
        vc.FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(lst), "-c", "copy", str(out_final),
    ], check=True)
    total = vc.audio_duration(out_final)
    print(f"ナレーション版 完了: {out_final}（{total:.3f}秒）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter", type=int, choices=[0, 1, 2, 3, 4], default=None)
    ap.add_argument("--concat", action="store_true", help="無音版チャプターの連結（QA用）")
    ap.add_argument("--mux", action="store_true", help="章ごとに音声を載せて連結（最終出力）")
    args = ap.parse_args()
    if args.mux:
        do_mux_and_concat()
        return
    if args.concat:
        do_concat()
        return
    fn = {0: chapter0, 1: chapter1, 2: chapter2, 3: chapter3, 4: chapter4}[args.chapter]
    fn()


if __name__ == "__main__":
    main()
