#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""センサ動画（sensor_visualization）のナレーションを章ごとに合成する。

`script.md` の `>` で始まる行を台本本文として読み、章 (## 第 N 章) ごとに、
空行（`>` だけの行）で区切られた段落を別々に合成してから、段落の間に無音を
挟んで連結する（`build_part1.py` と同じ「手作業のコマンドを残さない」方針。
台本を直したらこれを1回走らせれば作り直せる）。

読みの固定は `research_notes/scripts/video_kinematics/tts_azure.py::READINGS` を
そのまま使う（このファイルに書き足す）。

使い方（前景で。全章まとめても数十秒程度で終わる軽い処理）:
    set -a; . .secrets/azure_speech.env; set +a
    .venv/bin/python research_notes/scripts/video_sensor_narration/build_narration.py
    .venv/bin/python research_notes/scripts/video_sensor_narration/build_narration.py --chapter 2
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "research_notes" / "scripts" / "video_kinematics"))
sys.path.insert(0, str(REPO_ROOT / "research_notes" / "scripts"))
import tts_azure  # noqa: E402
from _video_sensor_common import PARA_GAP_S  # noqa: E402  段落間の無音（正本はこちら）

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
SCRIPT_MD = Path(__file__).resolve().parent / "script.md"
OUT_DIR = REPO_ROOT / "outputs" / "video_sensor" / "narration"
TMP_DIR = OUT_DIR / "_paragraphs"

VOICE = "ja-JP-NanamiNeural"

# 第3章だけ特別扱い: 段落2(P2「模擬時間と実際に経った時間」)と段落3(結論)の間は、
# 通常の無音(PARA_GAP_S)ではなく、4レーン並走そのもの（`video_sensor_stage2.py` の
# CH3_REAL_MOVE_S=14.0、実時計14秒・変えない）とぴったり同じ長さの無音を空ける。
# ここに音のない14秒ができ、映像側はその間だけ並走を再生する（教授セッション設計
# 2026-08-22）。CH3_REAL_MOVE_S を変えたらここも合わせて直すこと。
SPECIAL_GAP = {(3, 2): 14.0}


def parse_chapters(md_text: str) -> list[tuple[int, str, list[str]]]:
    """`## 第 N 章 ...` で章を割り、各章の `>` 行を段落（空の `>` 行区切り）に分ける。"""
    parts = re.split(r"^## +第 (\d+) 章\s*(.*)$", md_text, flags=re.M)
    chapters = []
    for i in range(1, len(parts), 3):
        num = int(parts[i])
        title = parts[i + 1].strip()
        body = parts[i + 2]
        paras: list[str] = []
        cur: list[str] = []
        for raw in body.splitlines():
            line = raw.strip()
            if not line.startswith(">"):
                continue
            content = line[1:].strip()
            if content == "":
                if cur:
                    paras.append("".join(cur))
                    cur = []
            else:
                cur.append(content)
        if cur:
            paras.append("".join(cur))
        chapters.append((num, title, paras))
    return chapters


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def make_silence(path: Path, seconds: float):
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
        "-t", f"{seconds:.3f}",
        "-c:a", "libmp3lame", "-b:a", "192k",
        str(path),
    ], check=True)


def silence_file(seconds: float) -> Path:
    """指定秒数の無音 mp3 をキャッシュして返す（同じ長さは使い回す）。"""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    p = TMP_DIR / f"silence_{seconds:.3f}s.mp3"
    if not p.exists():
        make_silence(p, seconds)
    return p


def build_chapter(num: int, paras: list[str]) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seg_paths = []
    for pi, text in enumerate(paras):
        seg = TMP_DIR / f"ch{num}_p{pi}.mp3"
        tts_azure.synthesize(text, seg, voice=VOICE)
        seg_paths.append(seg)

    out = OUT_DIR / f"ch{num}.mp3"
    lst = TMP_DIR / f"ch{num}_concat.txt"
    gap_sum = 0.0
    with open(lst, "w", encoding="utf-8") as f:
        for i, seg in enumerate(seg_paths):
            f.write(f"file '{seg.resolve()}'\n")
            if i < len(seg_paths) - 1:
                gap_s = SPECIAL_GAP.get((num, i), PARA_GAP_S)
                gap_sum += gap_s
                f.write(f"file '{silence_file(gap_s).resolve()}'\n")
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(lst), "-c", "copy", str(out),
    ], check=True)
    dur = ffprobe_duration(out)
    para_sum = sum(ffprobe_duration(s) for s in seg_paths)
    print(f"第{num}章: 段落{len(seg_paths)}個 音声本体={para_sum:.2f}s "
          f"+ 段落間無音={gap_sum:.2f}s ≒ 合計{dur:.2f}s  -> {out}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chapter", type=int, choices=[0, 1, 2, 3, 4], default=None)
    args = ap.parse_args(argv)

    chapters = parse_chapters(SCRIPT_MD.read_text(encoding="utf-8"))
    targets = [c for c in chapters if args.chapter is None or c[0] == args.chapter]
    if not targets:
        raise SystemExit(f"章 {args.chapter} が script.md に見つかりません")

    for num, title, paras in targets:
        if not paras:
            print(f"第{num}章（{title}）: 台本の段落が空のためスキップ")
            continue
        build_chapter(num, paras)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
