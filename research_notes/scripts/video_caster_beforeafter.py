# キャスタ摩擦バグ before/after 比較動画生成 (X投稿用)
# Generate a before/after comparison video of the caster-friction bug for X (Twitter) posts.
#
# 使い方 / Usage:
#   .venv/bin/python research_notes/scripts/video_caster_beforeafter.py
#
# 前提: assets/mouse_v2.xml は <contact> 内に caster_front/caster_back と floor の
# <pair friction="0.08 0.08 1e-4 1e-4 1e-4"/> を2行持つ「修正後」モデル。
# MuJoCoの結合則は要素ごとの最大値を取るため、キャスタ自身のfriction="0 0 0"（無摩擦のつもり）
# と床のfriction="1.0 0.005 1e-4"（μ=1.0）が組み合わさると combined friction が
# element-wise max で (1.0, 0.005, 1e-4) になり、キャスタが床に貼り付いて引きずる欠陥になる
# （v1由来の既知の不具合）。<pair> はこの結合則より優先されるため、上記2行を取り除くと
# 「修正前」の欠陥ある挙動を再現できる。本スクリプトは assets/mouse_v2.xml をコピーして
# <pair> 2行を取り除いた一時ファイルを「修正前」モデルとして使い、修正版と並べてレンダリングする。
import math
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_REPO)
sys.path.insert(0, _REPO)

import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from mouse.params import RobotParams  # noqa: E402
from mouse.sim import MouseSim  # noqa: E402

XML_PATH_AFTER = "assets/mouse_v2.xml"
OUT_PATH = "outputs/videos/caster_before_after.mp4"

PANEL_W, PANEL_H = 640, 720
FPS = 30
FRAME_DT = 1.0 / FPS
DRIVE_SECONDS = 3.0
HOLD_SECONDS = 3.5

FONT_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc"
FONT_SIZE = 30
LABEL_MARGIN_TOP = 16

LABEL_BEFORE = "修正前: キャスタが床を引きずる"
LABEL_AFTER = "修正後"


def make_before_xml(tmp_dir: str) -> str:
    """assets/mouse_v2.xml から <contact> 直下の <pair> 2要素を取り除いた
    「修正前（キャスタ摩擦バグ再現）」用XMLを一時ファイルに書き出し、そのパスを返す。"""
    tree = ET.parse(XML_PATH_AFTER)
    root = tree.getroot()
    contact = root.find("contact")
    if contact is None:
        raise RuntimeError("assets/mouse_v2.xml に <contact> 要素が見つかりません。")

    pairs = contact.findall("pair")
    if len(pairs) != 2:
        raise RuntimeError(
            f"<contact> 直下の <pair> 要素数が想定(2)と異なります: {len(pairs)}個。"
            " assets/mouse_v2.xml の構造が変わっていないか確認してください。"
        )
    for pair in pairs:
        contact.remove(pair)

    out_path = os.path.join(tmp_dir, "mouse_v2_before.xml")
    tree.write(out_path, encoding="unicode")
    return out_path


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """ヒラギノ角ゴシックを読み込む。"""
    return ImageFont.truetype(FONT_PATH, size, index=0)


def draw_label(frame: np.ndarray, text: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """パネル最上部に半透明の黒帯＋白文字で日本語ラベルを重畳する。"""
    h, w = frame.shape[:2]
    img = Image.fromarray(frame).convert("RGBA")

    # テキストサイズを計測して帯の高さを決める
    tmp_draw = ImageDraw.Draw(img)
    bbox = tmp_draw.textbbox((0, 0), text, font=font)
    text_h = bbox[3] - bbox[1]
    band_h = LABEL_MARGIN_TOP * 2 + text_h + (LABEL_MARGIN_TOP // 2)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([(0, 0), (w, band_h)], fill=(0, 0, 0, 160))
    text_x = (w - (bbox[2] - bbox[0])) // 2
    text_y = LABEL_MARGIN_TOP
    odraw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))

    composited = Image.alpha_composite(img, overlay)
    return np.array(composited.convert("RGB"))


def run_and_capture(xml_path: str, params: RobotParams, font: ImageFont.FreeTypeFont, label: str):
    """指定モデルで両輪+1.5Vを3秒間印加しつつ、30fps相当でフレームを間引きキャプチャする。

    Returns: (frames: list[np.ndarray HxWx3], distance_cm: float)
    """
    sim = MouseSim(xml_path, params=params)

    # オフスクリーンバッファ上限をXMLの800pxからパネルサイズへ拡張（レンダラ作成前に設定）
    sim.model.vis.global_.offwidth = PANEL_W
    sim.model.vis.global_.offheight = PANEL_H
    renderer = mujoco.Renderer(sim.model, height=PANEL_H, width=PANEL_W)

    x0, y0, _ = sim.privileged_pose()

    frames = []
    next_frame_time = 0.0
    n_steps = round(DRIVE_SECONDS / params.control_dt)

    for _ in range(n_steps):
        sim.step_control(1.5, 1.5)
        while sim.sim_time >= next_frame_time:
            renderer.update_scene(sim.data, camera="track")
            frame = renderer.render()
            frames.append(draw_label(frame, label, font))
            next_frame_time += FRAME_DT

    x1, y1, _ = sim.privileged_pose()
    distance_cm = math.hypot(x1 - x0, y1 - y0) * 100.0

    renderer.close()
    return frames, distance_cm


def main():
    params = RobotParams()
    font = load_font(FONT_SIZE)

    os.makedirs("outputs/videos", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        before_xml = make_before_xml(tmp_dir)

        print("[実行] 修正前モデルをシミュレーション中...")
        frames_before, dist_before = run_and_capture(before_xml, params, font, LABEL_BEFORE)

        print("[実行] 修正後モデルをシミュレーション中...")
        frames_after, dist_after = run_and_capture(XML_PATH_AFTER, params, font, LABEL_AFTER)

    print(f"[結果] 3秒後の移動距離: 修正前={dist_before:.1f}cm, 修正後={dist_after:.1f}cm")

    # フレーム数を揃える（アクティブ区間 ~90フレーム想定）
    n_active = min(len(frames_before), len(frames_after))
    frames_before = frames_before[:n_active]
    frames_after = frames_after[:n_active]

    hold_frames = round(HOLD_SECONDS * FPS)
    last_before = frames_before[-1]
    last_after = frames_after[-1]

    writer = imageio.get_writer(OUT_PATH, fps=FPS, macro_block_size=1)
    try:
        for fb, fa in zip(frames_before, frames_after):
            combined = np.hstack([fb, fa])
            writer.append_data(combined)
        held = np.hstack([last_before, last_after])
        for _ in range(hold_frames):
            writer.append_data(held)
    finally:
        writer.close()

    total_seconds = (n_active + hold_frames) / FPS
    print(f"[完了] {OUT_PATH} を書き出しました（フレーム数={n_active + hold_frames}, "
          f"長さ={total_seconds:.1f}秒）")


if __name__ == "__main__":
    main()
