#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`competition/mazes/contest_historical/`（npz・102 面）を `mazes/contest/*.maze` へ変換する。

設計は `research_notes/note_036_maze_database.md` §3 段 2。形式・読み書き器は
`common/maze_db.py`。**変換元は読むだけで、1 行も変更しない。**

年・区分は npz に埋め込まれた `source_file`（ヒストリーアーカイブの BMP ファイル名）
から復元する。ファイル名の書式は `research_notes/scripts/decode_ntf_maze_bitmaps.py`
が最初に読み取ったときの正規表現と同じ（本スクリプトの `_FILENAME_RE`）。

実行方法:
    .venv/bin/python research_notes/scripts/build_mazes_contest.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.maze_db import MazeRecord, WallState, compute_content_sha256, dump  # noqa: E402

SRC_DIR = ROOT / "competition" / "mazes" / "contest_historical"
OUT_DIR = ROOT / "mazes" / "contest"

SOURCE_URL = "https://www.ntf.or.jp/mouse/history/index.html"
RETRIEVED = "2026-08-23"  # manifest.json の provenance.retrieved と同じ
CONFIDENCE = "single-source"  # note_035 の食い違い 5 面の扱いは段 4（今回は触らない）
SOURCE_TYPE = "bmp"

_CLASS_MAP = {"": None, "exp": "expert", "frsh": "freshman"}
_STAGE_MAP = {"": None, "fin": "final", "pre": "preliminary"}

# decode_ntf_maze_bitmaps.py の meta() と同じ正規表現。
_FILENAME_RE = re.compile(
    r"(.+?)_(\d*)_?(\d{4})_classic_(\w*)_(\w*)_(\d+)x(\d+)(_maybe)?$"
)
_SOURCE_FILE_RE = re.compile(r"MazeImage/(.+)\.bmp$")


def _parse_source_file(source_file: str):
    m = _SOURCE_FILE_RE.search(source_file)
    if not m:
        raise ValueError(f"source_file の形式が想定外: {source_file!r}")
    basename = m.group(1)
    mm = _FILENAME_RE.match(basename)
    if not mm:
        raise ValueError(f"ファイル名を解析できない: {basename!r}")
    series, edition_s, year_s, cls, stage, w, h, maybe = mm.groups()
    edition = int(edition_s) if edition_s else None
    return dict(
        series=series,
        edition=edition,
        year=int(year_s),
        maze_class=_CLASS_MAP[cls],
        stage=_STAGE_MAP[stage],
        maybe=bool(maybe),
    )


def convert_one(npz_path: Path, maze_id: str) -> MazeRecord:
    d = np.load(npz_path, allow_pickle=True)
    width = int(d["width"])
    height = int(d["height"])
    start = (int(d["start_x"]), int(d["start_y"]))
    goal = sorted(zip(d["goals_x"].tolist(), d["goals_y"].tolist()), key=lambda p: (p[1], p[0]))
    source_file = str(d["source_file"])
    meta = _parse_source_file(source_file)

    # npz の 0/1（0=壁なし・1=壁あり）を WallState（0=未知・1=壁あり・2=壁なし）へ変換。
    # 実測がすべて 0/1 のみであることは manifest.json の構造検査で確認済み
    # （tests/test_maze_db.py 側でも独立に確認する）。
    v01 = np.asarray(d["v_walls"])
    h01 = np.asarray(d["h_walls"])
    v_walls = np.where(v01 == 1, WallState.WALL, WallState.OPEN).astype(np.int8)
    h_walls = np.where(h01 == 1, WallState.WALL, WallState.OPEN).astype(np.int8)

    content_sha256 = compute_content_sha256(width, height, v_walls, h_walls)

    notes = None
    if meta["maybe"]:
        notes = (
            "出典（NTF ヒストリーアーカイブ）側のファイル名が「_maybe」と"
            "留保を付けている面。留保の理由の詳細は出典に明示が無い。"
        )

    return MazeRecord(
        id=maze_id,
        width=width,
        height=height,
        start=start,
        start_heading="N",  # 発進は必ず北（docs/COORDINATE_SYSTEM.md §1）
        goal=goal,
        series=meta["series"],
        edition=meta["edition"],
        year=meta["year"],
        maze_class=meta["maze_class"],
        stage=meta["stage"],
        source_type=SOURCE_TYPE,
        source=source_file,
        source_url=SOURCE_URL,
        retrieved=RETRIEVED,
        confidence=CONFIDENCE,
        content_sha256=content_sha256,
        v_walls=v_walls,
        h_walls=h_walls,
        notes=notes,
    )


def main() -> None:
    manifest = json.loads((SRC_DIR / "manifest.json").read_text(encoding="utf-8"))
    names = manifest["mazes"]
    assert len(names) == 102, f"manifest の面数が想定と違う: {len(names)}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for name in names:
        assert name.startswith("maze_"), name
        maze_id = name[len("maze_"):]
        npz_path = SRC_DIR / f"{name}.npz"
        rec = convert_one(npz_path, maze_id)
        out_path = OUT_DIR / f"{maze_id}.maze"
        dump(rec, out_path)
        written.append(rec)

    print(f"変換完了: {len(written)} 面 → {OUT_DIR}")
    shas = [r.content_sha256 for r in written]
    dupes = {s for s in shas if shas.count(s) > 1}
    if dupes:
        print(f"content_sha256 の重複あり: {len(dupes)} 種類")
        for s in dupes:
            ids = [r.id for r in written if r.content_sha256 == s]
            print(f"  {s}: {ids}")
    else:
        print("content_sha256 の重複なし（102 面すべて別配置）")


if __name__ == "__main__":
    main()
