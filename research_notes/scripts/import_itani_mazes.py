#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""井谷氏の Wiki の `MAZ.zip`（256 バイト二進形式・127 ファイル）を `mazes/` へ取り込む。

設計は `research_notes/note_036_maze_database.md` §10 を基準文書とする。
経緯は `research_notes/note_035_historical_contest_mazes.md` 追記 6・追記 7。

## 出所

- 取得元ページ: https://w.atwiki.jp/mm3sakusya/pages/38.html（`mouse` ページ）
- 取得日: 2026-08-23（この PC の Chrome で取得。`curl` は Cloudflare に弾かれた）
- `MAZ.zip`（29,536 バイト・127 ファイル）そのものは版管理へ入れない（CLAUDE.md の禁止事項）。
  本スクリプトは `--src` で指定した展開先ディレクトリを読むだけで、リポジトリには
  変換結果の `.maze`（アスキー）だけを残す
- 権利についての判断（ユーザ決定・note_036 §10-1）:

  > 迷路探索シミュレータは著作権を放棄しています。どなた様もご自由に使用していただいて
  > 構いません。使用される時に私への連絡は必要ありません。

  この宣言は「迷路探索シミュレータ」についての文であり、データ集に及ぶかは文面から
  一意に決まらないが、**ユーザの決定によりデータ集も含まれると読んで取り込む**。

## 形式（教授セッションが確定済み。本スクリプトでは再検証しない）

256 バイトの二進形式:

    バイトの添字 = x * 16 + y     （x は東向き、y は北向き。区画 (0,0) が南西）
    ビット 0 = 北の壁、ビット 1 = 東の壁、ビット 2 = 南の壁、ビット 3 = 西の壁

1992 年決勝（`AllJapan_013_1992_exp_fin`）と 544/544 の壁が完全一致することで確定済み
（`note_035` 追記 6）。`x`・`y` の向きは `docs/COORDINATE_SYSTEM.md` と同一であり、
座標変換は不要（本データベースの `v_walls`/`h_walls` へ直接写像できる）。

432 バイトの 3 ファイル（`jpn2002ef.maz`・`jpn2002ep.maz`・`jpn2002f.maz`）は上と異なる、
32bit 16 進数を並べたテキスト形式（`0x00000000,0x00000000,` の並び。Wiki の `mouse`
ページに書式の例がある）。**別の担当が「向きが一意に決まらない」と報告しており**、
かつ 2002 年の全日本は本データベースの既存 157 面のどれとも一致しないことが分かっている
（`note_035` 追記 7・§7-2）ため、突き合わせによる裏取りができない。
**本スクリプトはこの 3 ファイルの取り込みを見送る**（720 バイト形式へ手を広げず、
`confidence` を下げてまで低い確度のまま載せるより、次に一次資料が見つかったときに
改めて取り込む方を選んだ）。

## 種別（`note_036` §10-3）

- `mazes/contest/` — 実在した競技の迷路
- `mazes/handmade/`（本取り込みで新設） — 手で作った検証用の迷路
- ファイル名から区分の判断が付かないものは `handmade/` に入れ、`notes` にその旨を書く

## id の付け方

- 年と回が分かるもの（`JPN-*`・`Jpn20xx`）: 既存の規則
  （`AllJapan_<回>_<年>[_<区分>][_<段階>]`）で組み、**末尾に `__itani` を付ける**
  （同じ迷路が別の出所から入っていることを指紋で照合できるようにするため。
  衝突の有無によらず一律に付ける — 出所がひと目で分かるようにするため）
- 年や回が分からないもの: 元のファイル名を小文字化 + `__itani`
  （例 `KILLER.MAZ` → `killer__itani`）
- 回は不明だが年だけ読み取れるもの（`Usa2001` 等、ファイル名に西暦がそのまま出ている）は
  `year` に入れるが、id は小文字化ルールに従う（`AllJapan` 以外に確立した
  「年→回」の変換式が無いため。**推測で回を埋めない**）
- ファイル名末尾の 2 桁の数字を西暦 19XX の下 2 桁として読む慣行は、本データベース既存の
  `Other(World85Fin)_1985` 等と同じ扱いを踏襲した（`IEE-88`→1988 等）。1〜2 桁でも
  年として読めない番号（`EURO-58` の `58`・`HOKU4` の `4` 等）は年を空欄のままにした

## 実行方法

    .venv/bin/python research_notes/scripts/import_itani_mazes.py --src <MAZ ディレクトリ> [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.maze_db import (  # noqa: E402
    MazeDB,
    MazeRecord,
    WallState,
    compute_content_sha256,
    dump,
    dumps,
    load,
)

CONTEST_DIR = ROOT / "mazes" / "contest"
HANDMADE_DIR = ROOT / "mazes" / "handmade"

SOURCE_URL = "https://w.atwiki.jp/mm3sakusya/pages/38.html"
RETRIEVED = "2026-08-23"
RIGHTS_NOTE = (
    "Wikiの『迷路探索シミュレータは著作権を放棄しています。どなた様もご自由に使用して"
    "いただいて構いません。使用される時に私への連絡は必要ありません。』という宣言に"
    "データ集も含まれると読んで取り込んだ（ユーザの判断。根拠は解釈に依る。note_036 §10-1）。"
)

EDITION_OFFSET_ALLJAPAN = 1979  # 既存102+53面から確立済み（kerikun11取り込みスクリプトと同一）

WIDTH = HEIGHT = 16
GOAL = [(7, 7), (8, 7), (7, 8), (8, 8)]
START = (0, 0)

_SKIPPED_TEXT_FORMAT = ("jpn2002ef", "jpn2002ep", "jpn2002f")


# ============================================================================
# 256 バイト二進形式のデコーダ
# ============================================================================
def decode_binary_256(data: bytes) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """256 バイトの二進形式を v_walls/h_walls へ変換する。

    戻り値の 3 つ目は、隣接する 2 区画が同じ壁について食い違う値を報告した場合の
    一覧（本来は起きないはずだが、壊れたファイルを検出するため記録する）。
    """
    if len(data) != 256:
        raise ValueError(f"256バイトでない: {len(data)}バイト")

    v = np.full((WIDTH + 1, HEIGHT), -1, dtype=np.int16)
    h = np.full((WIDTH, HEIGHT + 1), -1, dtype=np.int16)
    conflicts: List[str] = []

    def _set(arr: np.ndarray, i: int, j: int, is_wall: bool, label: str) -> None:
        val = int(WallState.WALL if is_wall else WallState.OPEN)
        cur = int(arr[i, j])
        if cur == -1:
            arr[i, j] = val
        elif cur != val:
            conflicts.append(f"{label}: 既存={cur} 新={val}")
            arr[i, j] = int(WallState.WALL)  # 食い違いは保守的に壁ありとする

    for x in range(WIDTH):
        for y in range(HEIGHT):
            b = data[x * 16 + y]
            north = bool(b & 0b0001)
            east = bool(b & 0b0010)
            south = bool(b & 0b0100)
            west = bool(b & 0b1000)
            _set(h, x, y + 1, north, f"h[{x},{y + 1}](区画{x},{y}の北壁)")
            _set(h, x, y, south, f"h[{x},{y}](区画{x},{y}の南壁)")
            _set(v, x + 1, y, east, f"v[{x + 1},{y}](区画{x},{y}の東壁)")
            _set(v, x, y, west, f"v[{x},{y}](区画{x},{y}の西壁)")

    if np.any(v == -1) or np.any(h == -1):
        raise ValueError("壁配列に未設定の要素が残っている（デコード漏れ）")
    return v.astype(np.int8), h.astype(np.int8), conflicts


# ============================================================================
# 構造検査（tests/test_maze_db.py の検査と同じ規約。書き換え禁止のため独立に実装）
# ============================================================================
def _open_sides(v: np.ndarray, h: np.ndarray, x: int, y: int) -> List[Tuple[str, Tuple[int, int]]]:
    sides = []
    if v[x + 1, y] == WallState.OPEN:
        sides.append(("E", (x + 1, y)))
    if v[x, y] == WallState.OPEN:
        sides.append(("W", (x - 1, y)))
    if h[x, y + 1] == WallState.OPEN:
        sides.append(("N", (x, y + 1)))
    if h[x, y] == WallState.OPEN:
        sides.append(("S", (x, y - 1)))
    return sides


def check_structure(v: np.ndarray, h: np.ndarray) -> Optional[str]:
    """問題が無ければ None、あれば理由の文字列を返す。"""
    if not np.all(v[0, :] == WallState.WALL):
        return "西の外周に穴"
    if not np.all(v[WIDTH, :] == WallState.WALL):
        return "東の外周に穴"
    if not np.all(h[:, 0] == WallState.WALL):
        return "南の外周に穴"
    if not np.all(h[:, HEIGHT] == WallState.WALL):
        return "北の外周に穴"

    goals = set(GOAL)
    for (x, y) in goals:
        for (nx, ny) in ((x + 1, y), (x, y + 1)):
            if (nx, ny) not in goals:
                continue
            state = v[x + 1, y] if nx == x + 1 else h[x, y + 1]
            if state != WallState.OPEN:
                return f"ゴール区画間 ({x},{y})-({nx},{ny}) に壁"

    sides = _open_sides(v, h, *START)
    if len(sides) != 1:
        return f"スタート区画の開口が{len(sides)}個（1個のはず）"

    seen = {START}
    dq = deque([START])
    reached = START in goals
    while dq and not reached:
        x, y = dq.popleft()
        for _s, n in _open_sides(v, h, x, y):
            if n in seen:
                continue
            seen.add(n)
            if n in goals:
                reached = True
                break
            dq.append(n)
    if not reached:
        return "スタートからゴールへ到達不能"
    return None


# ============================================================================
# ファイルごとのメタデータ（種別・系列・年・区分・段階）
# ============================================================================
@dataclass
class FileMeta:
    stem: str  # zip内の元ファイル名（拡張子抜き、大文字小文字は元のまま）
    kind: str  # "contest" | "handmade"
    series: str = ""
    year: Optional[int] = None
    maze_class: Optional[str] = None
    stage: Optional[str] = None
    edition: Optional[int] = None
    id_override: Optional[str] = None  # AllJapan系はここに完成形のidの前半を入れる
    notes_extra: str = ""
    ambiguous_kind: bool = False


NOTE_HANDMADE = "MAZ.zip収録。手で作った検証用の迷路と判断した（note_036 §10-3）。実在の競技結果ではない。"
NOTE_AMBIGUOUS_KIND = "区分（contest/handmade）の判断が付かない。実在の大会かどうかファイル名から確認できないため、指示に従いhandmadeへ分類した。"
NOTE_YEAR_2DIGIT = "ファイル名末尾の2桁の数字を西暦19XXの下2桁として読み取った（本データベース既存の『Other(World85Fin)』などと同じ慣行）。回・区分は一次資料で確認できていない。"
NOTE_NUMBER_UNCLEAR = "ファイル名の数字の意味（年・回・通し番号のいずれか）が確認できないため、yearは空欄とした。"


def _alljapan_plain(stem: str, edition: int, year: int, extra: str = "") -> FileMeta:
    return FileMeta(
        stem=stem, kind="contest", series="AllJapan", year=year, edition=edition,
        id_override=f"AllJapan_{edition:03d}_{year}",
        notes_extra=extra,
    )


# 前書きの `stage` は "final"/"preliminary" の完全語（既存102+53面の慣行。stage:final 等）。
# id の接尾（1990年より前は class が無いため `_fin`/`_pre`）はそこから機械的に作る。
STAGE_TO_ID_SUFFIX = {"final": "fin", "preliminary": "pre"}


def _alljapan_fp(stem: str, edition: int, year: int, stage: str) -> FileMeta:
    suffix = STAGE_TO_ID_SUFFIX[stage]
    return FileMeta(
        stem=stem, kind="contest", series="AllJapan", year=year, edition=edition,
        stage=stage, id_override=f"AllJapan_{edition:03d}_{year}_{suffix}",
    )


def _alljapan_exp_fin(stem: str, edition: int, year: int) -> FileMeta:
    return FileMeta(
        stem=stem, kind="contest", series="AllJapan", year=year, edition=edition,
        maze_class="expert", stage="final",
        id_override=f"AllJapan_{edition:03d}_{year}_exp_fin",
    )


_ENTRIES: List[FileMeta] = [
    # ---- 全日本（JPN-N: 回がファイル名に直接出ている。年 = 1979+回） ----
    _alljapan_plain("JPN-1", 1, 1980),
    _alljapan_plain("JPN-2", 2, 1981),
    _alljapan_plain("JPN-3", 3, 1982),
    _alljapan_plain("JPN-4", 4, 1983),
    _alljapan_plain("JPN-5", 5, 1984),
    # JPN-6/7: 1985・1986は既存DBに_fin/_preの2本があるが、ファイル名に区別が無い。
    # 段階はメイン処理で指紋突き合わせにより後から埋める（一致すれば採用、しなければ空欄）。
    FileMeta(stem="JPN-6", kind="contest", series="AllJapan", year=1985, edition=6,
             id_override="AllJapan_006_1985"),
    FileMeta(stem="JPN-7", kind="contest", series="AllJapan", year=1986, edition=7,
             id_override="AllJapan_007_1986"),
    _alljapan_fp("JPN-8F", 8, 1987, "final"),
    _alljapan_fp("JPN-8P", 8, 1987, "preliminary"),
    _alljapan_fp("JPN-9F", 9, 1988, "final"),
    _alljapan_fp("JPN-9P", 9, 1988, "preliminary"),
    _alljapan_fp("JPN-10F", 10, 1989, "final"),
    _alljapan_fp("JPN-10P", 10, 1989, "preliminary"),
    _alljapan_exp_fin("JPN-11EF", 11, 1990),
    _alljapan_exp_fin("JPN-12EF", 12, 1991),
    _alljapan_exp_fin("JPN-13EF", 13, 1992),
    _alljapan_exp_fin("JPN-14EF", 14, 1993),
    # ---- 全日本（Jpn20xx: 年が直接ファイル名。回 = 年-1979 は確立済みの変換式） ----
    _alljapan_plain("Jpn2000", 2000 - EDITION_OFFSET_ALLJAPAN, 2000,
                     extra="決勝・予選の区別はファイル名から不明。1面のみ。"),
    _alljapan_plain("Jpn2001", 2001 - EDITION_OFFSET_ALLJAPAN, 2001,
                     extra="決勝・予選の区別はファイル名から不明。1面のみ。"),
    _alljapan_plain("Jpn2002", 2002 - EDITION_OFFSET_ALLJAPAN, 2002,
                     extra="決勝・予選の区別はファイル名から不明。1面のみ。"
                           "note_035 §7-2: 我々の既存155面のどれとも一致しない（APEC2002/2018決勝いずれとも不一致）。"),

    # ---- 海外・地区大会（年や回の確立した変換式が無いため、id は小文字化フォールバック） ----
    FileMeta("2003kantou", "contest", series="Kantou", year=2003,
             notes_extra="関東地区大会と見られる。回・区分は不明。"),
    FileMeta("Boston", "contest", series="Boston",
             notes_extra="年・回とも不明。"),
    FileMeta("EURO-58", "contest", series="Euro", notes_extra=NOTE_NUMBER_UNCLEAR),
    FileMeta("EURO-V58", "contest", series="Euro", notes_extra=NOTE_NUMBER_UNCLEAR),
    FileMeta("EURO-V59", "contest", series="Euro", notes_extra=NOTE_NUMBER_UNCLEAR),
    FileMeta("HOKU4", "contest", series="Hokuriku", notes_extra=NOTE_NUMBER_UNCLEAR + "既存の『Hokuriku』地区大会（回=年-1982）とは別途の確認が要る。"),
    FileMeta("IEE-88", "contest", series="IEE", year=1988, notes_extra=NOTE_YEAR_2DIGIT),
    FileMeta("IEE-89F", "contest", series="IEE", year=1989, stage="final", notes_extra=NOTE_YEAR_2DIGIT),
    FileMeta("IEE-89P", "contest", series="IEE", year=1989, stage="preliminary", notes_extra=NOTE_YEAR_2DIGIT),
    FileMeta("J_STU-87", "contest", series="JStudent", year=1987, notes_extra=NOTE_YEAR_2DIGIT + "既存の『Student』（学生大会・回=年-1985）とは別系列として扱った（統合の裏取りができていない）。"),
    FileMeta("KYOT-89", "contest", series="Kyoto", year=1989, notes_extra=NOTE_YEAR_2DIGIT),
    FileMeta("MONT-88", "contest", series="Montreal", year=1988, notes_extra=NOTE_YEAR_2DIGIT),
    FileMeta("nagoya2002", "contest", series="Nagoya", year=2002, notes_extra="地区大会と見られる。回・区分は不明。"),
    FileMeta("niigata03", "contest", series="Niigata", year=2003, notes_extra="地区大会と見られる。回・区分は不明。"),
    FileMeta("SING87", "contest", series="Singapore", year=1987, notes_extra=NOTE_YEAR_2DIGIT),
    FileMeta("SING88", "contest", series="Singapore", year=1988, notes_extra=NOTE_YEAR_2DIGIT),
    FileMeta("SING89", "contest", series="Singapore", year=1989, notes_extra=NOTE_YEAR_2DIGIT),
    FileMeta("SING89SU", "contest", series="Singapore", year=1989, notes_extra=NOTE_YEAR_2DIGIT + "末尾『SU』の意味（段階等）は未確認。"),
    FileMeta("SOULU3", "contest", series="Soulu", notes_extra="『ソウル』の音訳と見られる。年・回とも不明。"),
    FileMeta("US-87", "contest", series="US", year=1987, notes_extra=NOTE_YEAR_2DIGIT),
    FileMeta("US-88", "contest", series="US", year=1988, notes_extra=NOTE_YEAR_2DIGIT),
    FileMeta("USA-1", "contest", series="USA", notes_extra=NOTE_NUMBER_UNCLEAR),
    FileMeta("USA-2", "contest", series="USA", notes_extra=NOTE_NUMBER_UNCLEAR),
    FileMeta("Usa2001", "contest", series="USA", year=2001, notes_extra="年・回とも既存DBに『USA』系列が無いため確立した変換式が無い。"),
    FileMeta("Usa2002", "contest", series="USA", year=2002, notes_extra="年・回とも既存DBに『USA』系列が無いため確立した変換式が無い。"),
    # WMMC-V1 は指紋が既存Other(World85Fin)_1985と、WMMC-Y2は既存Other(World85Pre2)_1985と
    # 完全一致した（main()の突き合わせで判明）。WMMC = World MicroMouse Contest と見て、
    # V1/V2/Y1/Y2の4本を1985年世界大会の一群と推定した。V2・Y1は指紋の裏付けが無いため
    # yearのみ揃え、確度は上げない（single-sourceのまま）。
    FileMeta("WMMC-V1", "contest", series="World85", year=1985, notes_extra="『WMMC』=World MicroMouse Contestと見た。V1の意味（決勝?）は未確認。"),
    FileMeta("WMMC-V2", "contest", series="World85", year=1985, notes_extra="『WMMC』=World MicroMouse Contestと見た。V2の意味は未確認。指紋は既存のいずれとも一致せず、裏付けは無い。"),
    FileMeta("WMMC-Y1", "contest", series="World85", year=1985, notes_extra="『WMMC』=World MicroMouse Contestと見た。Y1の意味は未確認。指紋は既存のいずれとも一致せず、裏付けは無い。"),
    FileMeta("WMMC-Y2", "contest", series="World85", year=1985, notes_extra="『WMMC』=World MicroMouse Contestと見た。Y2の意味は未確認。"),
    FileMeta("WORLD1", "contest", series="World85", year=1985, notes_extra="指紋がWMMC-V1・既存Other(World85Fin)_1985と完全一致。1985年世界大会決勝と見られる。"),
    FileMeta("Yama2002", "contest", series="Yamagata", year=2002, notes_extra="山形地区大会と見られる。回・区分は不明。"),

    # ---- 手で作った検証用の迷路（MAP-*・KILLER 等。すべて id は小文字化フォールバック） ----
    FileMeta("CUT", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("DAME", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("KILKAI", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("KILLER", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("KILNEW", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("KIREI", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("KIREI2", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-1", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-2", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-3", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-4", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-5", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-6", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-E5", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-E7", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-G1", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-G2", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-G3", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-G4", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-K5", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-N3", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-S3", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-S4", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-T1", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-T2", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-V1", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-V2", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-V3", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-V4", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-V5", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-V6", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-V7", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-V8", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-V9", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-Y10", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-Y4-1", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-Y4-2", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-Y4-3", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-Y4-4", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-Y5-1", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-Y5-2", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-Y5-3", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-Y5-4", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-Y6", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-Y7", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-Y77", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP-Y9", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MAP1", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MR1", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MR6", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("MTEST", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("NAMAE", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("QUO4", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("SHI", "handmade", notes_extra=NOTE_HANDMADE + NOTE_AMBIGUOUS_KIND, ambiguous_kind=True),
    FileMeta("SHI2", "handmade", notes_extra=NOTE_HANDMADE + NOTE_AMBIGUOUS_KIND, ambiguous_kind=True),
    FileMeta("SHIBA", "handmade", notes_extra=NOTE_HANDMADE + NOTE_AMBIGUOUS_KIND, ambiguous_kind=True),
    FileMeta("SHIKAI", "handmade", notes_extra=NOTE_HANDMADE + NOTE_AMBIGUOUS_KIND, ambiguous_kind=True),
    FileMeta("SHIMA1", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("SIZUKA", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("SUNADA", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("SUNDO-1", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("SUNKAI", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("SUPO1", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("SUPO2", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("SUPO3", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("SUPO4", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("TAISHO", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("V-SUPO", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("VM1", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("VM2", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("VM3", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("YAMA7", "handmade", notes_extra=NOTE_HANDMADE),
    FileMeta("YAMA89", "handmade", notes_extra=NOTE_HANDMADE),
]

assert len({e.stem for e in _ENTRIES}) == len(_ENTRIES), "stem が重複している"


def resolve_id(meta: FileMeta) -> str:
    base = meta.id_override if meta.id_override else meta.stem.lower()
    return base + "__itani"


# ============================================================================
# 変換本体
# ============================================================================
def build_record(meta: FileMeta, src_dir: Path, extra_notes: str = "") -> Tuple[MazeRecord, List[str]]:
    path = src_dir / f"{meta.stem}.MAZ"
    if not path.exists():
        path = src_dir / f"{meta.stem}.maz"
    data = path.read_bytes()
    v, h, conflicts = decode_binary_256(data)

    sha = compute_content_sha256(WIDTH, HEIGHT, v, h)
    start_heading = None
    problem = check_structure(v, h)
    if problem is None:
        sides = _open_sides(v, h, *START)
        start_heading = sides[0][0]

    notes_parts = [meta.notes_extra.strip(), extra_notes.strip(), RIGHTS_NOTE]
    notes = " ".join(p for p in notes_parts if p)

    rec = MazeRecord(
        id=resolve_id(meta),
        width=WIDTH,
        height=HEIGHT,
        start=START,
        start_heading=start_heading or "N",
        goal=GOAL,
        series=meta.series,
        edition=meta.edition,
        year=meta.year,
        maze_class=meta.maze_class,
        stage=meta.stage,
        source_type="binary",
        source=f"MAZ/{path.name}",
        source_url=SOURCE_URL,
        retrieved=RETRIEVED,
        confidence="single-source",
        content_sha256=sha,
        v_walls=v,
        h_walls=h,
        notes=notes,
        kind=meta.kind,
    )
    diagnostics = []
    if conflicts:
        diagnostics.append(f"{meta.stem}: 壁の食い違いを保守的に解決した ({len(conflicts)}件): {conflicts[:3]}")
    if problem is not None:
        diagnostics.append(f"{meta.stem}: 構造検査に落ちた: {problem}")
    return rec, diagnostics


def resolve_ambiguous_stage(meta: FileMeta, rec: MazeRecord, db: MazeDB) -> None:
    """JPN-6/JPN-7 のように fin/pre の区別がファイル名に無いものを、指紋突き合わせで埋める。"""
    if meta.stem not in ("JPN-6", "JPN-7"):
        return
    candidates = db.query(series="AllJapan", year=meta.year)
    matches = [c for c in candidates if c.content_sha256 == rec.content_sha256]
    if len(matches) == 1:
        stage = matches[0].stage
        rec.stage = stage
        suffix = STAGE_TO_ID_SUFFIX.get(stage, stage) if stage else None
        if suffix:
            assert rec.id.endswith("__itani")
            rec.id = rec.id[: -len("__itani")] + f"_{suffix}__itani"
        rec.notes = (rec.notes or "") + f" 指紋が既存{matches[0].id}と完全一致したためstage={stage}と判定した。"
    else:
        rec.notes = (rec.notes or "") + (
            f" fin/preの区別がファイル名に無く、既存の{meta.year}年AllJapan"
            f"（{[c.id for c in candidates]}）のいずれとも指紋が一致しなかったためstageは空欄のまま。"
        )


def build_fingerprint_maps(records: List[MazeRecord], db: MazeDB) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    existing_by_sha: Dict[str, List[str]] = {}
    for rid in db.ids():
        existing_by_sha.setdefault(db.get(rid).content_sha256, []).append(rid)
    new_by_sha: Dict[str, List[str]] = {}
    for rec in records:
        new_by_sha.setdefault(rec.content_sha256, []).append(rec.id)
    return existing_by_sha, new_by_sha


def cross_check_fingerprints(
    records: List[MazeRecord], existing_by_sha: Dict[str, List[str]], new_by_sha: Dict[str, List[str]]
) -> List[str]:
    """新規に作る全レコードを既存DBおよび新規どうしで指紋照合し、報告用の行を返す。"""
    report: List[str] = []
    for rec in records:
        matches = existing_by_sha.get(rec.content_sha256, [])
        if matches:
            report.append(f"一致(既存): {rec.id}  ==  {matches}")
    for sha, ids in new_by_sha.items():
        if len(ids) > 1:
            report.append(f"一致(新規どうし): {ids}")
    return report


# 既知の「同じ収集の中で系統が食い違う」組（note_035 追記6）。両方ともこの取り込みで作る面なので
# 直接 disputes を張れる。JPN-5/JPN-9F 側は既存DBとも一致しないことを確認済み（cross_check_fingerprints
# の既存一致リストに出ないことで裏取りする）。
_KNOWN_INTERNAL_DISPUTES = [
    ("AllJapan_005_1984__itani", "map-v5__itani"),
    ("AllJapan_009_1988_fin__itani", "map-v9__itani"),
]


def annotate_cross_references(
    records: List[MazeRecord], existing_by_sha: Dict[str, List[str]], new_by_sha: Dict[str, List[str]]
) -> None:
    """指紋の一致・食い違いを各レコードの notes/disputes へ書き込む（このスクリプトが作る分だけ）。"""
    for rec in records:
        extra = []
        existing_matches = existing_by_sha.get(rec.content_sha256, [])
        if existing_matches:
            extra.append(f"指紋が既存{existing_matches}と完全一致（独立した出所での裏付け）。")
        siblings = [i for i in new_by_sha.get(rec.content_sha256, []) if i != rec.id]
        if siblings:
            extra.append(f"指紋が本取り込みの他面{siblings}とも一致。")
        if extra:
            rec.notes = ((rec.notes or "") + " " + " ".join(extra)).strip()

    by_id = {r.id: r for r in records}
    for id_a, id_b in _KNOWN_INTERNAL_DISPUTES:
        a, b = by_id.get(id_a), by_id.get(id_b)
        if a is None or b is None:
            continue
        if a.content_sha256 == b.content_sha256:
            continue  # 食い違いが解消していれば何もしない（想定外だが安全側に倒す）
        a.disputes = list(dict.fromkeys((a.disputes or []) + [id_b]))
        b.disputes = list(dict.fromkeys((b.disputes or []) + [id_a]))
        a.notes = ((a.notes or "") + f" {id_b}と壁配置が食い違う（同じMAZ.zip内の別系統。note_035 追記6）。").strip()
        b.notes = ((b.notes or "") + f" {id_a}と壁配置が食い違う（同じMAZ.zip内の別系統。note_035 追記6）。").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="MAZ.zip を展開したディレクトリ")
    ap.add_argument("--dry-run", action="store_true", help="書き出さず一覧・突き合わせ結果だけ表示する")
    args = ap.parse_args()

    db = MazeDB()  # 既存 mazes/ 全部（contest・handmade・reserved）を読み込む

    records: List[MazeRecord] = []
    diagnostics_all: List[str] = []
    for meta in _ENTRIES:
        rec, diagnostics = build_record(meta, args.src)
        diagnostics_all.extend(diagnostics)
        resolve_ambiguous_stage(meta, rec, db)
        problem = check_structure(rec.v_walls, rec.h_walls)
        if problem is not None:
            print(f"[除外] {meta.stem}: {problem}")
            continue
        records.append(rec)

    print(f"\nデコード対象 {len(_ENTRIES)} 件のうち、構造検査を通過: {len(records)} 件")
    print(f"見送った 432 バイトのテキスト形式: {list(_SKIPPED_TEXT_FORMAT)}")

    if diagnostics_all:
        print("\n診断:")
        for d in diagnostics_all:
            print(f"  {d}")

    existing_by_sha, new_by_sha = build_fingerprint_maps(records, db)
    print("\n指紋の突き合わせ:")
    for line in cross_check_fingerprints(records, existing_by_sha, new_by_sha):
        print(f"  {line}")
    annotate_cross_references(records, existing_by_sha, new_by_sha)

    # 既知の食い違いを個別に確かめる
    print("\n既知の食い違いの確認:")
    id_to_rec = {r.id: r for r in records}
    jpn14 = id_to_rec.get("AllJapan_014_1993_exp_fin__itani")
    if jpn14 is not None:
        all_existing_sha = {db.get(i).content_sha256 for i in db.ids()}
        hit = jpn14.content_sha256 in all_existing_sha
        print(f"  JPN-14EF (1993決勝): 既存157面のいずれかと一致するか -> {hit}")
    jpn5 = id_to_rec.get("AllJapan_005_1984__itani")
    mapv5 = id_to_rec.get("map-v5__itani")
    existing_1984 = db.query(series="AllJapan", year=1984)
    if jpn5 is not None and existing_1984:
        print(f"  JPN-5 vs 既存1984: {'一致' if jpn5.content_sha256 == existing_1984[0].content_sha256 else '不一致'}")
    if mapv5 is not None and existing_1984:
        print(f"  MAP-V5 vs 既存1984: {'一致' if mapv5.content_sha256 == existing_1984[0].content_sha256 else '不一致'}")
    iee89p = id_to_rec.get("iee-89p__itani")
    iee1993 = db.query(series="IEE1993")
    if iee89p is not None and iee1993:
        print(f"  IEE-89P vs 既存IEE1993_1993: {'一致' if iee89p.content_sha256 == iee1993[0].content_sha256 else '不一致'}")

    if args.dry_run:
        for rec in records:
            print(f"(dry-run) {rec.id:55s} kind={rec.kind:9s} series={rec.series:12s} year={rec.year} sha={rec.content_sha256[:12]}")
        return

    CONTEST_DIR.mkdir(parents=True, exist_ok=True)
    HANDMADE_DIR.mkdir(parents=True, exist_ok=True)
    n_contest = n_handmade = 0
    for rec in records:
        out_dir = CONTEST_DIR if rec.kind == "contest" else HANDMADE_DIR
        out_path = out_dir / f"{rec.id}.maze"
        dump(rec, out_path)
        # 往復検査（書いた直後にその場で確かめる）
        reloaded_text = out_path.read_text(encoding="utf-8")
        rebuilt = dumps(load(out_path))
        if rebuilt != reloaded_text:
            raise AssertionError(f"{rec.id}: 往復検査に失敗した")
        if rec.kind == "contest":
            n_contest += 1
        else:
            n_handmade += 1

    print(f"\n書き出し: contest {n_contest} 面 / handmade {n_handmade} 面")


if __name__ == "__main__":
    main()
