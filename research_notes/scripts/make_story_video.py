"""research_notes/scripts/make_story_video.py — 発表用動画を組み立てる

研究プログラムの歩みを 1 本の動画にする。
**「速くなりました」ではなく「何を作り、何が壊れていて、どう建て直したか」**を伝える。

構成:
    1. 表題
    2. 問い（ミッション）
    3. 新実装の探索走行（早送り・キャプション付き）
    4. 何が起きたか（旧実装の破棄）
    5. 建て直し
    6. 次へ

出力: 1920x1080 / 30fps / H.264。X と YouTube のどちらにもそのまま出せる。
音声は無し（X の自動再生は無音のため、説明はすべて画面に焼き込む）。

使い方:
    .venv/bin/python research_notes/scripts/make_story_video.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent.parent
RUN_MP4 = REPO / "outputs/video_rebuild/maze_42134_classic.mp4"
OUT_MP4 = REPO / "outputs/video_rebuild/story.mp4"
W, H, FPS = 1920, 1080, 30

FONT_PATH = "/System/Library/Fonts/Hiragino Sans GB.ttc"
BG = (14, 16, 20)
FG = (238, 240, 244)
DIM = (150, 158, 170)
ACC = (255, 176, 59)
RED = (232, 92, 92)
GRN = (98, 200, 140)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


def wrap(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    """日本語は単語境界が無いので 1 文字ずつ詰めて折り返す。"""
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur); cur = ""; continue
        if draw.textlength(cur + ch, font=f) > max_w and cur:
            lines.append(cur); cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def card(title: str, body: str = "", kicker: str = "", accent=ACC,
         rows: list[tuple[str, str, tuple]] | None = None) -> np.ndarray:
    """静止カードを 1 枚描く。"""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    x, y = 150, 250

    if kicker:
        fk = font(30)
        d.text((x, y), kicker, font=fk, fill=accent)
        y += 60

    ft = font(74)
    for ln in wrap(d, title, ft, W - 2 * x):
        d.text((x, y), ln, font=ft, fill=FG)
        y += 96
    y += 26

    if body:
        fb = font(38)
        for ln in wrap(d, body, fb, W - 2 * x):
            d.text((x, y), ln, font=fb, fill=DIM)
            y += 56

    if rows:
        y += 30
        fl, fv = font(36), font(52)
        for label, value, col in rows:
            d.text((x, y + 14), label, font=fl, fill=DIM)
            d.text((x + 640, y), value, font=fv, fill=col)
            y += 84

    # 下端の細いアクセント
    d.rectangle([0, H - 8, W, H], fill=accent)
    return np.asarray(img)


def caption_band(frame: np.ndarray, text: str, sub: str = "", alpha: float = 1.0) -> np.ndarray:
    """走行映像の下端に半透明の帯とキャプションを焼き込む。"""
    if alpha <= 0.01 or not text:
        return frame
    img = Image.fromarray(frame).convert("RGB")
    band_h = 190 if sub else 140
    band = Image.new("RGB", (W, band_h), (0, 0, 0))
    base = img.crop((0, H - band_h, W, H))
    img.paste(Image.blend(base, band, 0.72 * alpha), (0, H - band_h))

    d = ImageDraw.Draw(img)
    ft = font(46)
    ty = H - band_h + 30
    for ln in wrap(d, text, ft, W - 220)[:2]:
        d.text((110, ty), ln, font=ft, fill=tuple(int(c * alpha + BG[i] * (1 - alpha))
                                                  for i, c in enumerate(FG)))
        ty += 58
    if sub:
        fs = font(32)
        d.text((110, ty + 4), sub, font=fs, fill=tuple(int(c * alpha) for c in ACC))
    return np.asarray(img)


def hold(writer, frame: np.ndarray, seconds: float, fade_in: float = 0.0) -> None:
    n = int(seconds * FPS)
    nf = int(fade_in * FPS)
    for i in range(n):
        if i < nf:
            a = (i + 1) / max(nf, 1)
            writer.append_data((frame.astype(np.float32) * a + np.array(BG, np.float32) * (1 - a)).astype(np.uint8))
        else:
            writer.append_data(frame)


def main() -> None:
    if not RUN_MP4.exists():
        raise SystemExit(f"走行動画が無い: {RUN_MP4}")
    OUT_MP4.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(OUT_MP4), fps=FPS, macro_block_size=1, quality=8)

    # ---- 1. 表題 -----------------------------------------------------
    hold(w, card(
        "マイクロマウスを、\nセンサだけで走らせる",
        "強化学習でセンサ入力から左右モータ電圧を直接出す — その前に、"
        "実機がやっていることを PC 上で再現する必要があった。",
        kicker="研究プログラムの記録  2026-08",
    ), 5.0, fade_in=0.7)

    # ---- 2. 問い -----------------------------------------------------
    hold(w, card(
        "まず古典手法を再現する",
        "実機のマウスは、区画ごとに壁をセンサで読んで地図を作り、"
        "歩数マップで最短経路を求め、ターンの列として走る。真の位置は知らない。",
        kicker="なぜ古典から",
    ), 5.5, fade_in=0.5)

    # ---- 3. 走行（早送り + キャプション） -----------------------------
    # 実時間 297 秒の探索を約 34 秒に詰める。
    reader = imageio.get_reader(str(RUN_MP4))
    n_total = reader.count_frames()
    step = 9                                   # 9 フレームに 1 枚 = 9 倍速
    picked = list(range(0, n_total, step))
    # キャプション（詰めたあとの通し番号で指定する）
    marks = [
        (0.00, "16×16 の迷路。地図は持っていない", "距離センサ・ジャイロ・車輪の回転だけ"),
        (0.18, "壁を読みながら、自分で地図を作っていく", "右上が、マウス自身が作った地図"),
        (0.45, "行き止まりに入れば、引き返して塗り直す", ""),
        (0.72, "真の位置は一度も見ていない", "推測航法 ＋ 壁センサによる区画ごとの補正"),
        (0.90, "ゴール到達", "297.7 秒・通過 68 区画"),
    ]
    m_idx = 0
    cur_text, cur_sub, since = "", "", -999
    for i, fi in enumerate(picked):
        p = i / max(len(picked) - 1, 1)
        if m_idx < len(marks) and p >= marks[m_idx][0]:
            _, cur_text, cur_sub = marks[m_idx]
            since = i
            m_idx += 1
        frame = reader.get_data(fi)
        age = (i - since) / FPS
        alpha = min(1.0, age / 0.35) if age >= 0 else 0.0
        if age > 6.5:                          # 6.5 秒でキャプションを引く
            alpha = max(0.0, 1.0 - (age - 6.5) / 0.5)
        w.append_data(caption_band(frame, cur_text, cur_sub, alpha))
    reader.close()

    # ---- 4. 何が起きたか ---------------------------------------------
    hold(w, card(
        "だが、その前に一度すべて捨てた",
        "半月かけて積み上げた古典実装は、計画した経路がほとんど走行に使われていなかった。"
        "しかもそれは 5 日前に一度発見され、記録され、忘れられていた。",
        kicker="2026-08-19 の決断",
        accent=RED,
        rows=[
            ("斜め経路に乗っていた割合", "0.23 %", RED),
            ("破棄した行数", "61,086 行", RED),
            ("残したもの", "物理モデル・評価器・記録", GRN),
        ],
    ), 7.5, fade_in=0.5)

    # ---- 5. 建て直し --------------------------------------------------
    hold(w, card(
        "教訓は、作法ではなく検査にした",
        "「計画した経路が本当に走行に使われたか」を毎回測る。"
        "真値を壊しても走りが変わらないことを確かめる。失効した数値が別の文書へ流れていないか探す。",
        kicker="同じ穴を掘らないために",
        accent=GRN,
        rows=[
            ("再発防止の検査", "4 種類", GRN),
            ("新実装のテスト", "87 件", GRN),
            ("地図の一致率", "95.8 〜 98.9 %", GRN),
        ],
    ), 7.5, fade_in=0.5)

    # ---- 6. 次へ ------------------------------------------------------
    hold(w, card(
        "次は、最短走行へ",
        "探索で作った地図から最短経路を引き、ターンの列として速く走る。"
        "そのあとが、本題の強化学習。",
        kicker="つづく",
    ), 5.5, fade_in=0.5)

    w.close()
    size_mb = OUT_MP4.stat().st_size / 1e6
    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(OUT_MP4)],
        capture_output=True, text=True).stdout.strip()
    print(f"完成: {OUT_MP4}  {size_mb:.1f} MB  {dur} 秒")


if __name__ == "__main__":
    main()
