#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解説動画 第 1 部を組み立てる（映像の生成 → 音声の多重化 → 連結）。

台本を直したら、これを 1 回走らせれば全部やり直せる。手作業のコマンドを残さないこと
（🔴 台本の読みを直したのに古い音声が残る、という事故を防ぐため）。

    .venv/bin/python research_notes/scripts/video_kinematics/build_part1.py
    .venv/bin/python research_notes/scripts/video_kinematics/build_part1.py --only clip_06_sweep_clearance
    .venv/bin/python research_notes/scripts/video_kinematics/build_part1.py --concat-only

【尺の決め方】各クリップの映像長は **ナレーションの実測長 ＋ 余韻 1.0 秒**。
ナレーション mp3 を ffprobe で測り、その値を環境変数
`VIDEO_KINEMATICS_TARGET_SECONDS` でクリップのスクリプトへ渡す
（各スクリプトはこの環境変数があればそれを使い、無ければ自分の既定値を使う）。

【音声の多重化】映像のほうが約 1 秒長いので、`apad` で音声を映像長ぴったりまで
無音で伸ばしてから多重化する。これをやらないと連結時に音声が前へ詰まる
（実際に起きた。`2ea0284` の報告）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parent
NARRATION = ROOT / "outputs" / "video_kinematics" / "narration"
OUT = ROOT / "outputs" / "video_kinematics"

# 話の順（番号順ではない）。連結もこの順で行う。
CLIPS = [
    "clip_01_opening",
    "clip_02_friction_circle",
    "clip_03_two_choices",
    "clip_06_sweep_clearance",
    "clip_07_diagonal",
    "clip_08_velocity_profile",
    "clip_09_eta",
    "clip_10_rebuilt",
]
TAIL_HOLD_S = 1.0   # ナレーションが終わってからの余韻


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def render(name: str) -> Path:
    """クリップ 1 本の映像を作る。尺はナレーションの実測長から決める。"""
    mp3 = NARRATION / f"{name}_r0.mp3"
    if not mp3.exists():
        raise SystemExit(f"ナレーションがありません: {mp3}\n"
                         f"  先に tts_azure.py で合成してください。")
    target = duration(mp3) + TAIL_HOLD_S
    env = dict(os.environ, VIDEO_KINEMATICS_TARGET_SECONDS=f"{target:.3f}")
    print(f"  [{name}] ナレーション {target - TAIL_HOLD_S:.2f}s → 映像 {target:.2f}s")
    subprocess.run([sys.executable, str(SCRIPTS / f"{name}.py")],
                   check=True, cwd=str(ROOT), env=env)
    return OUT / f"{name}.mp4"


def mux(name: str) -> Path:
    """映像にナレーションを載せる。音声は映像長まで無音で伸ばす。"""
    v, a = OUT / f"{name}.mp4", NARRATION / f"{name}_r0.mp3"
    out = OUT / f"{name}_av.mp4"
    vd = duration(v)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(v), "-i", str(a),
        "-filter_complex", f"[1:a]apad=whole_dur={vd:.3f}[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out),
    ], check=True)
    return out


def concat() -> Path:
    lst = OUT / "_concat_list.txt"
    lst.write_text("".join(f"file '{OUT / (n + '_av.mp4')}'\n" for n in CLIPS), encoding="utf-8")
    out = OUT / "part1.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(lst), "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", str(out),
    ], check=True)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--only", action="append", help="このクリップだけ作り直す（複数可）")
    p.add_argument("--concat-only", action="store_true", help="連結だけやり直す")
    a = p.parse_args(argv)

    if not a.concat_only:
        targets = a.only or CLIPS
        for name in targets:
            if name not in CLIPS:
                raise SystemExit(f"知らないクリップ: {name}（{CLIPS}）")
            render(name)
            mux(name)

    out = concat()
    total = duration(out)
    parts = sum(duration(OUT / f"{n}_av.mp4") for n in CLIPS)
    print(f"\n{out}  {total:.3f} 秒（各クリップの合計 {parts:.3f} 秒・差 {abs(total-parts):.3f} 秒）")
    if abs(total - parts) > 0.5:
        raise SystemExit("🔴 連結後の長さが合計と 0.5 秒以上ずれています")
    for n in CLIPS:
        v = duration(OUT / f"{n}.mp4"); s = duration(NARRATION / f"{n}_r0.mp3")
        mark = "OK" if v >= s else "🔴 音声が映像より長い"
        print(f"  {n:28s} 映像 {v:6.2f}s / 音声 {s:6.2f}s  {mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
