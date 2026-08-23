#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`kerikun11/micromouse-maze-data`（MIT）を `mazes/contest/` へ取り込む（段 3）。

設計は `research_notes/note_036_maze_database.md` §9 を基準文書とする。

## 出所

- 上流: https://github.com/kerikun11/micromouse-maze-data
- ライセンス: MIT License / Copyright (c) 2020 Ryotaro Onuki
- 取得コミット: `762ed2b68735ea29148c6a1251a90ed0651ff26b`（2024-02-22, "add MM2023HX maze"）。
  2026-08-23 に GitHub の `master` の HEAD と突き合わせ、**同一コミットであることを確認済み**
  （上流に新しい面は増えていない）
- 本リポジトリでは `competition/reference_mazes/micromouse-maze-data/data/*.maze`（80 本、
  2026-08-10 に前任セッションが取得・無改変で同梱）に既にベンダリングされている。
  本スクリプトはそこを読み取るだけで、`competition/` 配下へは一切書き込まない
  （CLAUDE.md の禁止事項）。

## 取り込みの絞り込み（80 本 → 53 本）

除外 27 本の内訳:

| 区分 | 面数 | 理由 |
|---|---|---|
| ハーフサイズ（32×32・区画 90mm、実競技） `32MM*HX` | 15 | note_036 §6-2 未解決（本データベースの形式にハーフサイズをどう載せるか未確定）のため、今回は見送る |
| ハーフサイズのテスト・デモ迷路 `32_*` | 6 | 上と同じ理由に加え、実在の競技ではない |
| その他のテスト・デモ迷路 `04_test_01`／`09_test_*`／`64DFS01` | 6 | ファイル名が示すとおり試験・デモ用で実在の競技ではない（note_036 §4「学習中に作る迷路は入れない」と同じ考え方） |

残り 53 本を取り込む。内訳は `_ENTRIES` を参照。

## 🔴 id の付け方（既存 102 面の規則に揃えられない面がある）

既存 102 面は `<series>_<edition>_<year>[_<class>][_<stage>]` で、`class`/`stage` は
`expert`/`freshman`・`final`/`preliminary` という**意味の分かる語**である。

kerikun11 のファイル名接尾（`CX`・`C`・`H`・`HX`・`_pre`・`_semi`・`_cand` 等）のうち、
**意味が確認できたものだけ**既存語彙へ翻訳し、**確認できないものは翻訳せず原表記のまま**
id の末尾に残す（憶測で `expert`/`final` 等を割り当てない）。

| kerikun11 の表記 | 確からしさ | 扱い |
|---|---|---|
| 単独の `CX`（地区名なし） | 高（日本のマウサー界隈で定着した略称。X=エキスパート。2012 年面で NTF 側 `AllJapan_033_2012_exp_fin` と指紋が一致し裏が取れた） | `class: expert` `stage: final` へ翻訳 |
| `CX_pre` | 同上＋「pre」は素直に予選 | `class: expert` `stage: preliminary` |
| `CF_pre`（`08MM2016CF_pre`） | 中（X=エキスパートに倣うなら F=フレッシュマン。8×8 という小さい規格は初心者クラスと整合する。**未確認の推測**） | `class: freshman` `stage: preliminary`、`notes` に推測である旨を明記 |
| `H`・`HX`・地区名付き `C`・`_semi` 等 | 低。前任セッションの調査（`docs/MAZE_DIFFICULTY_REPORT.md` §7 限界1）でも「H=本戦？」止まりで確認できていない。同じ地区・同じ年に `C` 版と `H` 版の両方がある例が複数あり、両者が別物であることは分かるが、どちらが予選でどちらが本戦かは分からない | `class`/`stage` は空欄のまま、**原表記を小文字化して id 末尾にそのまま残す**（例 `_h`・`_c`・`_hx_pre`・`_h_semi`） |
| 台湾 `HX_Taiwan`（21×21） | 低。数字接頭辞は他面と違い 21（32 でも 16 でもない）。ハーフサイズ規格（32×32・区画90mm）と同じ表記 `HX` を使っているが、実グリッドが違うため区画寸法は未確認 | `series: Taiwan`・`class`/`stage`/`edition` は空欄、`notes` に寸法未確認を明記。**除外はしない**（32×32 ではないため今回のハーフサイズ除外規則の対象外という判断。要確認点として報告する） |
| `Cheese`／`Cheese_cand`（16×16・9×9） | 低。ゴールが中央でない・未知壁を大量に含む特別な迷路で、通常のクラシック規格の大会結果ではなさそうだが確証はない | `series: Cheese`（16×16）／`Cheese9`（9×9）。`class`/`stage`/`edition` は空欄 |

`edition` は、同じ `series` で既存 102 面から**年→回の関係式が 3 例以上の等差数列として確認できる
場合に限り**、その式を延長して埋める（`_EDITION_OFFSET` 参照）。新規の地区名
（Kansai・Tashiro・Kanazawa）や特別枠（Cheese・Cheese9・Taiwan）には前例が無いため空欄のまま。
延長で埋めた `edition` はすべて `notes` に「年からの外挿」である旨を記す。

## 未知壁（`.`）

kerikun11 の形式は横壁の未知を ` . `（空白・ドット・空白の 3 文字）、縦壁の未知を `.`
（1 文字）で表す。本データベースの `?`／`???` に置き換えるだけで、壁の意味（未知）はそのまま
1 対 1 で対応する。未知壁を含むのは `Cheese`（16×16）系 3 面のみで、外周・スタート開口・
ゴール区画内部はすべて既知（`check_cheese_boundary.py` で確認済み、本データベースの構造検査は
これらの面でも成立する）。

## 実行方法

    .venv/bin/python research_notes/scripts/import_kerikun11_mazes.py           # 変換して書き出す
    .venv/bin/python research_notes/scripts/import_kerikun11_mazes.py --dry-run # 書き出さず一覧だけ
"""
from __future__ import annotations

import argparse
import sys
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
)

SRC_DIR = ROOT / "competition" / "reference_mazes" / "micromouse-maze-data" / "data"
OUT_DIR = ROOT / "mazes" / "contest"

SOURCE_URL = "https://github.com/kerikun11/micromouse-maze-data"
RETRIEVED = "2026-08-23"

# 既存 102 面から確認できる「年 → 回」の関係式（edition = year - offset）。
# 3 例以上の等差数列で確認できたものだけを載せる（本スクリプトの docstring 参照）。
_EDITION_OFFSET = {
    "AllJapan": 1979,
    "Chubu": 1981,
    "Kyushu": 1981,
    "EastJapan": 1982,
    "Hokuriku": 1982,
    "Student": 1985,
}

_EXTRAPOLATED_NOTE = "edition は既存面の年→回の関係式（{series}: year-{offset}）を延長した推定値。"


# ============================================================================
# kerikun11 形式のアスキー図パーサ（未知壁 '.' に対応。他は本データベースと同じ記号）
# ============================================================================
def parse_kerikun11(text: str) -> Tuple[int, int, np.ndarray, np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    n_rows = len(lines)
    if n_rows % 2 != 1:
        raise ValueError(f"行数が奇数でない: {n_rows}")
    height = (n_rows - 1) // 2
    width = (len(lines[0]) - 1) // 4

    v = np.zeros((width + 1, height), dtype=np.int8)
    h = np.zeros((width, height + 1), dtype=np.int8)
    starts: List[Tuple[int, int]] = []
    goals: List[Tuple[int, int]] = []

    for row in range(height + 1):
        line = lines[2 * row]
        if len(line) != 4 * width + 1 or line[0] != "+" or line[-1] != "+":
            raise ValueError(f"{row} 行目（柱の行）が想定外: {line!r}")
        h_level = height - row
        for x in range(width):
            seg = line[1 + 4 * x: 1 + 4 * x + 3]
            if seg == "---":
                h[x, h_level] = int(WallState.WALL)
            elif seg == "   ":
                h[x, h_level] = int(WallState.OPEN)
            elif seg == " . ":
                h[x, h_level] = int(WallState.UNKNOWN)
            else:
                raise ValueError(f"想定外の横区間: {seg!r}（行 {line!r}）")
        if row < height:
            y = height - 1 - row
            cell_line = lines[2 * row + 1]
            for x in range(width + 1):
                ch = cell_line[4 * x]
                if ch == "|":
                    v[x, y] = int(WallState.WALL)
                elif ch == " ":
                    v[x, y] = int(WallState.OPEN)
                elif ch == ".":
                    v[x, y] = int(WallState.UNKNOWN)
                else:
                    raise ValueError(f"想定外の縦文字: {ch!r}（行 {cell_line!r}）")
            for x in range(width):
                label = cell_line[4 * x + 1: 4 * x + 4].strip()
                if label == "S":
                    starts.append((x, y))
                elif label == "G":
                    goals.append((x, y))
                elif label != "":
                    raise ValueError(f"想定外のラベル: {label!r}")
    return width, height, v, h, starts, goals


# ============================================================================
# 変換対象 53 本のメタデータ（stem -> 属性）
# ============================================================================
@dataclass
class Entry:
    stem: str  # data/<stem>.maze
    id: str
    series: str
    year: int
    maze_class: Optional[str] = None
    stage: Optional[str] = None
    edition: Optional[int] = None
    edition_extrapolated: bool = False
    notes: Optional[str] = None
    confidence: str = "single-source"


def _all_japan_final(year: int) -> Entry:
    ed = year - _EDITION_OFFSET["AllJapan"]
    maze_id = f"AllJapan_{ed:03d}_{year}_exp_fin"
    notes = None if year == 2012 else _EXTRAPOLATED_NOTE.format(series="AllJapan", offset=1979)
    if year == 2012:
        # 🔴 指紋の突き合わせで NTF 側の AllJapan_033_2012_exp_fin と壁配置が完全一致
        # （content_sha256 が一致、v_walls/h_walls も要素ごとに一致を確認済み）。
        # 同じ id は使えないので __kerikun11 を付す（note_036 §2-3 の別出所サフィックス規則）。
        maze_id += "__kerikun11"
        notes = (
            "NTF ヒストリーアーカイブの AllJapan_033_2012_exp_fin と壁配置が完全一致（指紋照合で確認）。"
            "独立した2出所が一致したため、この面と AllJapan_033_2012_exp_fin の両方を "
            "confidence: confirmed に更新した。"
        )
    entry = Entry(
        stem=f"16MM{year}CX",
        id=maze_id,
        series="AllJapan",
        year=year,
        maze_class="expert",
        stage="final",
        edition=ed,
        edition_extrapolated=(year != 2012),  # 2012 は NTF 側と指紋一致で裏取り済み
        notes=notes,
    )
    if year == 2012:
        entry.confidence = "confirmed"
    if year == 2018:
        # 🔴 指紋の突き合わせで NTF 側の APEC2002_2002（2002年APECの迷路）と壁配置が完全一致。
        # 「全日本2018年エキスパート決勝」と「APEC2002」は別の大会・別の年のはずであり、
        # これは note_036 §2-4 の「食い違い」とは逆に「一致してはいけないはずが一致した」事例。
        # どちらのラベルが誤っているか（あるいは意図的な同一迷路の再利用か）は未確認のため、
        # confidence は single-source のまま変えず、notes で相互参照するに留める。
        entry.notes = (
            (entry.notes + " " if entry.notes else "")
            + "🔴 壁配置が NTF ヒストリーアーカイブの APEC2002_2002（2002年APEC）と完全一致（指紋照合）。"
            "全日本2018年エキスパート決勝とAPEC2002は別大会・別年のはずで、一致の理由は未確認"
            "（同一迷路の意図的な再利用か、どちらかのラベル誤りかは判断できない）。"
        )
    return entry


_ENTRIES: List[Entry] = []

# --- AllJapan（地区名なし）: CX / CX_pre / HX_pre / H_semi ---
for _y in (2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020):
    _ENTRIES.append(_all_japan_final(_y))

_ed2017 = 2017 - _EDITION_OFFSET["AllJapan"]
_ENTRIES.append(Entry(
    stem="16MM2017CX_pre", id=f"AllJapan_{_ed2017:03d}_2017_exp_pre",
    series="AllJapan", year=2017, maze_class="expert", stage="preliminary",
    edition=_ed2017, edition_extrapolated=True,
    notes=_EXTRAPOLATED_NOTE.format(series="AllJapan", offset=1979),
))
_ENTRIES.append(Entry(
    stem="16MM2017HX_pre", id=f"AllJapan_{_ed2017:03d}_2017_k11hx_pre",
    series="AllJapan", year=2017, edition=_ed2017, edition_extrapolated=True,
    notes=(
        "kerikun11 のファイル名接尾 'HX_pre' は 'CX_pre'（エキスパート予選）と同年に別途存在し、"
        "両者が別物であることは分かるが意味は未確認のため class/stage は空欄。"
        + _EXTRAPOLATED_NOTE.format(series="AllJapan", offset=1979)
    ),
))
for _y in (2018, 2019, 2021):
    _ed = _y - _EDITION_OFFSET["AllJapan"]
    _ENTRIES.append(Entry(
        stem=f"16MM{_y}H_semi", id=f"AllJapan_{_ed:03d}_{_y}_k11h_semi",
        series="AllJapan", year=_y, edition=_ed, edition_extrapolated=True,
        notes=(
            "地区名が付かない全国規模の 'semi'（準決勝?）と見られるが未確認。class/stage は空欄。"
            + _EXTRAPOLATED_NOTE.format(series="AllJapan", offset=1979)
        ),
    ))

# --- 08MM2016CF_pre（8×8・フレッシュマン予選と推測） ---
_ed_cf = 2016 - _EDITION_OFFSET["AllJapan"]
_ENTRIES.append(Entry(
    stem="08MM2016CF_pre", id=f"AllJapan_{_ed_cf:03d}_2016_frsh_pre",
    series="AllJapan", year=2016, maze_class="freshman", stage="preliminary",
    edition=_ed_cf, edition_extrapolated=True,
    notes=(
        "ファイル名 'CF' を X=エキスパートに倣い F=フレッシュマンと解釈した（未確認の推測）。"
        "8×8 という小さい区画数は初心者クラスと整合するが一次資料での裏取りはできていない。"
        + _EXTRAPOLATED_NOTE.format(series="AllJapan", offset=1979)
    ),
))

# --- 地区名付き C_<region> / H_<region>（意味不明の原表記を id 末尾へ残す） ---
_REGION_SERIES = {
    "Chubu": "Chubu",
    "Kyushu": "Kyushu",
    "East": "EastJapan",
    "Hokuriku": "Hokuriku",
    "student": "Student",
    # 既存 102 面に前例が無い新規地区・企画名
    "Kansai": "Kansai",
    "Tashiro": "Tashiro",
    "Kanazawa": "Kanazawa",
}

_REGIONAL_RAW = [
    # (stem 接尾の地区名, 年, 原表記の文字コード)
    ("Chubu", 2015, "c"), ("Chubu", 2016, "c"), ("Kyushu", 2016, "c"),
    ("Kansai", 2016, "h"),
    ("Chubu", 2017, "c"), ("East", 2017, "c"), ("student", 2017, "c"),
    ("Chubu", 2017, "h"), ("Kansai", 2017, "h"), ("Tashiro", 2017, "h"), ("student", 2017, "h"),
    ("student", 2018, "c"),
    ("Chubu", 2018, "h"), ("Kansai", 2018, "h"), ("Tashiro", 2018, "h"), ("student", 2018, "h"),
    ("student", 2019, "c"),
    ("Chubu", 2019, "h"), ("East", 2019, "h"), ("Hokuriku", 2019, "h"), ("Kanazawa", 2019, "h"),
    ("Kansai", 2019, "h"), ("Kyushu", 2019, "h"), ("Tashiro", 2019, "h"), ("student", 2019, "h"),
    ("student", 2020, "c"), ("student", 2020, "h"),
    ("Kansai", 2021, "h"),
]

_NOTE_REGIONAL = (
    "kerikun11 のファイル名接尾のクラス文字（'{code}'）の意味（予選/本戦・クラス区分）は"
    "確認できていないため class/stage は空欄とし、原表記を小文字化して id 末尾に残した"
    "（`docs/MAZE_DIFFICULTY_REPORT.md` §7 限界1と同じ未確認事項）。"
)

for region_key, year, code in _REGIONAL_RAW:
    series = _REGION_SERIES[region_key]
    stem = f"16MM{year}{code.upper()}_{region_key}"
    ed = None
    extrapolated = False
    notes = _NOTE_REGIONAL.format(code=code)
    if series in _EDITION_OFFSET:
        ed = year - _EDITION_OFFSET[series]
        extrapolated = True
        notes += " " + _EXTRAPOLATED_NOTE.format(series=series, offset=_EDITION_OFFSET[series])
    id_parts = [series]
    if ed is not None:
        id_parts.append(f"{ed:03d}")
    id_parts.append(str(year))
    id_parts.append(f"k11{code}")
    _ENTRIES.append(Entry(
        stem=stem, id="_".join(id_parts), series=series, year=year,
        edition=ed, edition_extrapolated=extrapolated, notes=notes,
    ))

# --- Cheese（16×16・未知壁あり・ゴール非中央） ---
_NOTE_CHEESE16 = (
    "ゴールが中央 2x2 でなく未知壁（'.'）を多数含む特別な迷路。実際に開催された競技結果か、"
    "候補として作成された案かは未確認。series/class/stage は不明のため独立の series 'Cheese' とした。"
)
_ENTRIES.append(Entry(stem="16MM2017H_Cheese", id="Cheese_2017_k11h", series="Cheese", year=2017, notes=_NOTE_CHEESE16))
_ENTRIES.append(Entry(stem="16MM2019H_Cheese", id="Cheese_2019_k11h", series="Cheese", year=2019, notes=_NOTE_CHEESE16))
_ENTRIES.append(Entry(stem="16MM2019H_Cheese_cand", id="Cheese_2019_k11h_cand", series="Cheese", year=2019,
                       notes=_NOTE_CHEESE16 + " ファイル名の '_cand' は候補案（candidate）を示すとみられる。"))

# --- Cheese9（9×9・未知壁なし） ---
_NOTE_CHEESE9 = (
    "9x9 の小型迷路。ゴール 2x2 は中央ではない。'Cheese' 系だが未知壁は含まない。"
    "実競技かどうか未確認のため独立の series 'Cheese9' とした。"
)
_ENTRIES.append(Entry(stem="09MM2019C_Cheese", id="Cheese9_2019_k11c", series="Cheese9", year=2019, notes=_NOTE_CHEESE9))
_ENTRIES.append(Entry(stem="09MM2019C_Cheese_cand", id="Cheese9_2019_k11c_cand", series="Cheese9", year=2019,
                       notes=_NOTE_CHEESE9 + " ファイル名の '_cand' は候補案（candidate）を示すとみられる。"))

# --- Taiwan（21×21・区画寸法未確認） ---
_NOTE_TAIWAN = (
    "21x21 という本データベースの他面に無いサイズ。kerikun11 側のファイル名接尾 'HX' は"
    "ハーフサイズ規格（32×32・区画90mm）と同じ表記だが実グリッドが 21x21 であり、"
    "区画寸法（180mm/90mm 等）が一次資料で確認できていない。ハーフサイズ（32×32）の"
    "除外規則はサイズを名指ししたものであり本面はその対象外と判断したが、要確認点として報告する。"
)
for _y in (2014, 2015, 2016, 2017, 2018):
    _ENTRIES.append(Entry(stem=f"21MM{_y}HX_Taiwan", id=f"Taiwan_{_y}_k11hx", series="Taiwan", year=_y, notes=_NOTE_TAIWAN))

assert len(_ENTRIES) == 53, f"想定は53本、実際は{len(_ENTRIES)}本"
assert len({e.id for e in _ENTRIES}) == 53, "id が重複している"
assert len({e.stem for e in _ENTRIES}) == 53, "stem が重複している"


# ============================================================================
# 変換本体
# ============================================================================
def build_record(entry: Entry) -> MazeRecord:
    path = SRC_DIR / f"{entry.stem}.maze"
    text = path.read_text(encoding="utf-8")
    width, height, v, h, starts, goals = parse_kerikun11(text)
    if len(starts) != 1:
        raise ValueError(f"{entry.stem}: S の数が {len(starts)}（1 のはず）")
    sha = compute_content_sha256(width, height, v, h)
    return MazeRecord(
        id=entry.id,
        width=width,
        height=height,
        start=starts[0],
        start_heading="N",
        goal=goals,
        series=entry.series,
        edition=entry.edition,
        year=entry.year,
        maze_class=entry.maze_class,
        stage=entry.stage,
        source_type="ascii",
        source=f"kerikun11/micromouse-maze-data data/{entry.stem}.maze",
        source_url=SOURCE_URL,
        retrieved=RETRIEVED,
        confidence=entry.confidence,
        content_sha256=sha,
        v_walls=v,
        h_walls=h,
        notes=entry.notes,
        kind="contest",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="書き出さず一覧だけ表示する")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for entry in _ENTRIES:
        rec = build_record(entry)
        if args.dry_run:
            print(f"{rec.id:45s} {rec.width}x{rec.height} series={rec.series} year={rec.year} sha={rec.content_sha256[:12]}")
        else:
            dump(rec, OUT_DIR / f"{rec.id}.maze")
    print(f"{'(dry-run) ' if args.dry_run else ''}変換: {len(_ENTRIES)} 面")


if __name__ == "__main__":
    main()
