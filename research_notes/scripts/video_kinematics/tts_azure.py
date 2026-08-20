#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Azure Speech（REST API）で日本語のナレーションを合成する。

SDK を入れずに標準ライブラリだけで叩く（依存を増やさないため）。

【鍵の渡し方】環境変数のみ。**鍵をコードにも版管理にも置かない。**
    export AZURE_SPEECH_KEY="..."
    export AZURE_SPEECH_REGION="japaneast"      # 例: japaneast / japanwest / eastus

【なぜ Azure を選んだか】
技術文書の読み上げでは、声の自然さより**読みの正確さ**が効く。
`0.12 m/s`・`90°`・`η`・`5.57` のような表記は、どのエンジンでも高い確率で読み間違える。
Azure は SSML の `<sub alias="...">` で読みを 1 語ずつ指定できるので、
台本に「読み」を書けば必ずそのとおりに読む（`docs/JA_ENGINEERING_TERMS.md` の用語も同様に指定できる）。

    .venv/bin/python research_notes/scripts/video_kinematics/tts_azure.py --list-voices
    .venv/bin/python research_notes/scripts/video_kinematics/tts_azure.py \
        --text "こんにちは" --voice ja-JP-NanamiNeural --out out.mp3
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

# 日本語のニューラル音声（2026-08 時点で確認したもの。--list-voices で実際の一覧を取れる）
DEFAULT_VOICE = "ja-JP-NanamiNeural"

# 読みの置き換え表。台本にこの表記が出たら SSML の <sub> に変換する。
# 🔴 ここに書けば全クリップで一貫した読みになる。増えたら追記すること。
READINGS: list[tuple[str, str]] = [
    (r"η",            "イータ"),
    (r"m/s²",         "メートル毎秒毎秒"),
    (r"m/s",          "メートル毎秒"),
    (r"rad/s²",       "ラジアン毎秒毎秒"),
    (r"rad/s",        "ラジアン毎秒"),
    (r"mm",           "ミリメートル"),
    (r"(?<![A-Za-z])m(?![A-Za-z])", "メートル"),
    (r"°",            "度"),
    (r"摩擦円",        "まさつえん"),
    (r"弧長",          "こちょう"),
    (r"肉薄度",        "にくはくど"),
    (r"物理限界",      "ぶつりげんかい"),
    (r"超信地旋回",    "ちょうしんちせんかい"),
    (r"足立法",        "あだちほう"),
    (r"迷路",          "めいろ"),
    (r"区画",          "くかく"),
]


# READINGS を 1 本の正規表現にまとめる。
# 🔴 1 回の走査で当てること。順に re.sub を掛けると、置換した中身へ別の規則が
#    もう一度当たり `<sub><sub>...</sub></sub>` の入れ子になる（実際に起きた）。
#    表の並び順がそのまま優先順位になるので、長い表記を先に置くこと（"m/s²" → "m/s" → "m"）。
_READING_RE = re.compile("|".join(f"(?P<g{i}>{pat})" for i, (pat, _) in enumerate(READINGS)))


def to_ssml(text: str, voice: str, rate: str = "0%", pitch: str = "0%") -> str:
    """台本を SSML にする。数字と単位は <sub> で読みを固定する。"""
    def _sub(m: "re.Match") -> str:
        i = int(m.lastgroup[1:])          # 当たった規則の番号
        return f'<sub alias="{READINGS[i][1]}">{html.escape(m.group(0))}</sub>'

    # 先に置換してから、置換されなかった部分だけを escape する
    out, last = [], 0
    for m in _READING_RE.finditer(text):
        out.append(html.escape(text[last:m.start()]))
        out.append(_sub(m))
        last = m.end()
    out.append(html.escape(text[last:]))
    body = "".join(out)
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="ja-JP">'
        f'<voice name="{voice}">'
        f'<prosody rate="{rate}" pitch="{pitch}">{body}</prosody>'
        "</voice></speak>"
    )


def _creds() -> tuple[str, str]:
    key = os.environ.get("AZURE_SPEECH_KEY")
    region = os.environ.get("AZURE_SPEECH_REGION")
    if not key or not region:
        raise SystemExit(
            "AZURE_SPEECH_KEY と AZURE_SPEECH_REGION を環境変数で渡してください。\n"
            "  Azure ポータル → Speech service を作成 → 「キーとエンドポイント」から取得\n"
            "  例: export AZURE_SPEECH_KEY='...' ; export AZURE_SPEECH_REGION='japaneast'\n"
            "  🔴 鍵は版管理に入れないこと。"
        )
    return key, region


def list_voices() -> list[dict]:
    key, region = _creds()
    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/voices/list"
    req = urllib.request.Request(url, headers={"Ocp-Apim-Subscription-Key": key})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def synthesize(text: str, out: Path, voice: str = DEFAULT_VOICE,
               rate: str = "0%", pitch: str = "0%") -> Path:
    key, region = _creds()
    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
    ssml = to_ssml(text, voice, rate, pitch).encode("utf-8")
    req = urllib.request.Request(url, data=ssml, headers={
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/ssml+xml",
        # 48kHz・モノラル・192kbps。動画に載せるので余裕を持たせる
        "X-Microsoft-OutputFormat": "audio-48khz-192kbitrate-mono-mp3",
        "User-Agent": "micromouse_rl-narration",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--list-voices", action="store_true", help="日本語の音声一覧を出す")
    p.add_argument("--text", help="読み上げる台本")
    p.add_argument("--text-file", type=Path, help="台本のファイル")
    p.add_argument("--voice", default=DEFAULT_VOICE)
    p.add_argument("--rate", default="0%", help="速度（例 -10% / +15%）")
    p.add_argument("--pitch", default="0%")
    p.add_argument("--out", type=Path, default=Path("outputs/video_kinematics/narration.mp3"))
    p.add_argument("--show-ssml", action="store_true", help="生成した SSML を印字して終わる")
    a = p.parse_args(argv)

    if a.list_voices:
        for v in list_voices():
            if v.get("Locale", "").startswith("ja-"):
                print(f"  {v['ShortName']:28s} {v.get('Gender','?'):6s} {v.get('LocalName','')}")
        return 0

    text = a.text or (a.text_file.read_text(encoding="utf-8").strip() if a.text_file else None)
    if not text:
        p.error("--text か --text-file が要ります")

    if a.show_ssml:
        print(to_ssml(text, a.voice, a.rate, a.pitch))
        return 0

    out = synthesize(text, a.out, a.voice, a.rate, a.pitch)
    print(f"書き出した: {out}  ({out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
