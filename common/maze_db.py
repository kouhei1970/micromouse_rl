"""
common/maze_db.py
==================
迷路データベース（`mazes/`）の読み書き器。

設計は `research_notes/note_036_maze_database.md` を基準文書とする。座標系は
`docs/COORDINATE_SYSTEM.md` を正とする。要点だけここに書く（数値・規約そのものは
基準文書を見ること）。

- 1 迷路 1 ファイル。拡張子 `.maze`。前書き（簡易 YAML 風）＋ アスキー図の 2 段構成
- アスキー図は `+---+` 形式（柱 `+`・横壁 `---`・縦壁 `|`・開通は空白・未知は `?`）。
  詰めた書き方（`|` と `_`）は採らない（`note_035` で実際に踏んだ落とし穴のため）
- **アスキー図は上が北。**図の行番号と迷路の y 座標の変換は、本ファイルでは
  `_ascii_h_level_of_row` と `_ascii_cell_y_of_row` の 2 関数にだけ書く
- 前書きの `content_sha256` は**壁の並びだけ**から決まる指紋（年・出所・確度は含めない）

前書きに置く `source_type` は、資料そのものの形式を表す（`confidence` とは別物。
確度は「確からしさの評価」、`source_type` は「資料の見た目」）:
    bmp       画像（ビットマップ）から壁配置を読み取ったもの
    ascii     出所側がアスキー図で公開しているもの
    pdf       PDF から読み取ったもの
    binary    `.maz` などの二進形式から取り込んだもの
    generated 生成器で作ったもの

本モジュールが返す `walls()` の配列は `competition/evaluator.py` の
v_walls/h_walls と同じ 0/1 の約束（0=壁なし・1=壁あり、非 0/1 は無い）に合わせてある。
内部では 3 値（未知/壁あり/壁なし）を持つため、未知の壁が 1 枚でも残っている迷路に
対して `walls()` を呼ぶとエラーになる（0/1 表現には未知を表す枠が無く、黙って
どちらかに丸めると情報が失われるため）。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------
# 壁の 3 値状態
# ----------------------------------------------------------------------
# classic/maze_map.py の WallState と数値の意味を合わせてあるが、common/ は
# classic/ に依存させたくないので独立に定義する（値を変えるときは両方直すこと）。


class WallState(IntEnum):
    """1 枚の壁がとりうる 3 状態。"""

    UNKNOWN = 0
    WALL = 1
    OPEN = 2


_H_SEG_TO_STATE = {"---": WallState.WALL, "   ": WallState.OPEN, "???": WallState.UNKNOWN}
_STATE_TO_H_SEG = {WallState.WALL: "---", WallState.OPEN: "   ", WallState.UNKNOWN: "???"}
_V_CHAR_TO_STATE = {"|": WallState.WALL, " ": WallState.OPEN, "?": WallState.UNKNOWN}
_STATE_TO_V_CHAR = {WallState.WALL: "|", WallState.OPEN: " ", WallState.UNKNOWN: "?"}

_HEADING_CHARS = ("N", "E", "S", "W")
_SOURCE_TYPES = ("bmp", "ascii", "pdf", "binary", "generated")

# 前書きのキーの並び順（書き出し・読み込みの両方でこの順を使う。§2-3）。
_FRONT_MATTER_KEYS = (
    "id",
    "size",
    "start",
    "start_heading",
    "goal",
    "series",
    "edition",
    "year",
    "class",
    "stage",
    "source_type",
    "source",
    "source_url",
    "retrieved",
    "confidence",
    "disputes",
    "notes",
    "content_sha256",
)


@dataclass
class MazeRecord:
    """迷路 1 面ぶんの前書き＋壁配置。"""

    id: str
    width: int
    height: int
    start: Tuple[int, int]
    start_heading: str
    goal: List[Tuple[int, int]]
    series: str
    edition: Optional[int]
    year: Optional[int]
    maze_class: Optional[str]  # 前書きの `class`（Python の予約語のため属性名を変えてある）
    stage: Optional[str]
    source_type: str
    source: str
    source_url: str
    retrieved: str
    confidence: str
    content_sha256: str
    v_walls: np.ndarray  # WallState コード。形状 (width+1, height)
    h_walls: np.ndarray  # WallState コード。形状 (width, height+1)
    disputes: Optional[List[str]] = None
    notes: Optional[str] = None
    kind: Optional[str] = field(default=None, compare=False)  # 置き場（contest/reserved）。前書きには無い
    path: Optional[Path] = field(default=None, compare=False)  # 読み込み元ファイル（診断用）


# ============================================================================
# アスキー図 ⇄ 壁配列（座標変換はここにだけ書く）
# ============================================================================
def _ascii_h_level_of_row(row_from_top: int, height: int) -> int:
    """柱の行（上から row_from_top 番目、0 始まり）が指す h_walls の y 添字。

    アスキー図は上が北（docs/COORDINATE_SYSTEM.md）。h_walls[:, height] が
    迷路最北の外周、h_walls[:, 0] が最南の外周なので、上から 0 行目が
    h_level=height に対応する。
    """
    return height - row_from_top


def _ascii_cell_y_of_row(row_from_top: int, height: int) -> int:
    """区画の行（上から row_from_top 番目、0 始まり）が指す区画の y 座標。"""
    return height - 1 - row_from_top


def render_ascii_grid(
    width: int,
    height: int,
    v_walls: np.ndarray,
    h_walls: np.ndarray,
    start: Tuple[int, int],
    goal: Iterable[Tuple[int, int]],
) -> List[str]:
    """壁配列からアスキー図（`+---+` 形式）を組み立てる。

    行数 2*height+1、各行の文字数 4*width+1（note_036 §2-2 の実測どおり）。
    """
    goal_set = set(goal)
    lines: List[str] = []
    for row in range(height + 1):
        h_level = _ascii_h_level_of_row(row, height)
        segs = ["+"]
        for x in range(width):
            segs.append(_STATE_TO_H_SEG[WallState(int(h_walls[x, h_level]))])
            segs.append("+")
        lines.append("".join(segs))

        if row < height:
            y = _ascii_cell_y_of_row(row, height)
            segs2 = [_STATE_TO_V_CHAR[WallState(int(v_walls[0, y]))]]
            for x in range(width):
                if (x, y) == tuple(start):
                    label = " S "
                elif (x, y) in goal_set:
                    label = " G "
                else:
                    label = "   "
                segs2.append(label)
                segs2.append(_STATE_TO_V_CHAR[WallState(int(v_walls[x + 1, y]))])
            lines.append("".join(segs2))
    return lines


def parse_ascii_grid(
    lines: List[str], width: int, height: int
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int]], List[Tuple[int, int]]]:
    """アスキー図（`+---+` 形式）を壁配列へ戻す。

    戻り値: (v_walls, h_walls, 図から読めた S の位置, 図から読めた G の位置一覧)。
    S・G は前書きの `start`/`goal` の裏取り用（本体は前書きが正）。
    """
    expected_rows = 2 * height + 1
    if len(lines) != expected_rows:
        raise ValueError(f"アスキー図の行数が {expected_rows} でない（実際 {len(lines)}）")
    expected_len = 4 * width + 1
    for i, ln in enumerate(lines):
        if len(ln) != expected_len:
            raise ValueError(
                f"アスキー図 {i} 行目の文字数が {expected_len} でない（実際 {len(ln)}: {ln!r}）"
            )

    v = np.zeros((width + 1, height), dtype=np.int8)
    h = np.zeros((width, height + 1), dtype=np.int8)
    start_found: Optional[Tuple[int, int]] = None
    goals_found: List[Tuple[int, int]] = []

    for row in range(height + 1):
        h_level = _ascii_h_level_of_row(row, height)
        line = lines[2 * row]
        if line[0] != "+" or line[-1] != "+":
            raise ValueError(f"柱の行の両端が '+' でない: {line!r}")
        for x in range(width):
            seg = line[1 + 4 * x : 1 + 4 * x + 3]
            if seg not in _H_SEG_TO_STATE:
                raise ValueError(f"横壁の区間が想定外: {seg!r}（行 {line!r}）")
            h[x, h_level] = int(_H_SEG_TO_STATE[seg])

        if row < height:
            y = _ascii_cell_y_of_row(row, height)
            cell_line = lines[2 * row + 1]
            for x in range(width + 1):
                ch = cell_line[4 * x]
                if ch not in _V_CHAR_TO_STATE:
                    raise ValueError(f"縦壁の文字が想定外: {ch!r}（行 {cell_line!r}）")
                v[x, y] = int(_V_CHAR_TO_STATE[ch])
            for x in range(width):
                label = cell_line[4 * x + 1 : 4 * x + 4].strip()
                if label == "S":
                    start_found = (x, y)
                elif label == "G":
                    goals_found.append((x, y))
                elif label != "":
                    raise ValueError(f"区画内の表示が想定外: {label!r}（行 {cell_line!r}）")

    return v, h, start_found, goals_found


# ============================================================================
# content_sha256（壁の並びだけから決まる指紋）
# ============================================================================
def compute_content_sha256(width: int, height: int, v_walls: np.ndarray, h_walls: np.ndarray) -> str:
    """壁配列だけから指紋を計算する。前書きの他の項目（年・出所・確度）は含めない。

    3 値（未知/壁あり/壁なし）をそのまま使う。0/1 表現に丸めてしまうと未知を
    表せなくなり、未知の壁を含む迷路と含まない迷路が同じ指紋になりかねない
    ため、丸める前の状態でハッシュを取る。
    """
    v = np.ascontiguousarray(v_walls, dtype=np.uint8)
    h = np.ascontiguousarray(h_walls, dtype=np.uint8)
    header = f"{width}x{height}\n".encode("ascii")
    digest = hashlib.sha256(header + v.tobytes() + h.tobytes())
    return digest.hexdigest()


# ============================================================================
# 前書き（簡易 YAML 風）の読み書き
# ============================================================================
def _fmt_scalar(value) -> str:
    return "" if value is None else str(value)


def _fmt_pair(pair: Tuple[int, int]) -> str:
    return f"[{pair[0]}, {pair[1]}]"


def dumps(rec: MazeRecord) -> str:
    """MazeRecord を `.maze` ファイルの中身（文字列）へ直列化する。

    ここで決めた書式だけを基準にしており、`load`/`loads` はこれの逆変換になる
    ように作ってある（往復検査が効くのはそのため）。
    """
    goal_sorted = sorted(rec.goal, key=lambda p: (p[1], p[0]))
    front: Dict[str, str] = {
        "id": rec.id,
        "size": f"[{rec.width}, {rec.height}]",
        "start": _fmt_pair(rec.start),
        "start_heading": rec.start_heading,
        "goal": "[" + ", ".join(_fmt_pair(p) for p in goal_sorted) + "]",
        "series": rec.series,
        "edition": _fmt_scalar(rec.edition),
        "year": _fmt_scalar(rec.year),
        "class": _fmt_scalar(rec.maze_class),
        "stage": _fmt_scalar(rec.stage),
        "source_type": rec.source_type,
        "source": f'"{rec.source}"',
        "source_url": f'"{rec.source_url}"',
        "retrieved": rec.retrieved,
        "confidence": rec.confidence,
        "content_sha256": rec.content_sha256,
    }
    if rec.disputes:
        front["disputes"] = "[" + ", ".join(rec.disputes) + "]"
    if rec.notes:
        front["notes"] = f'"{rec.notes}"'

    lines = ["---"]
    for key in _FRONT_MATTER_KEYS:
        if key not in front:
            continue  # disputes/notes は無ければ出さない
        lines.append(f"{key}: {front[key]}")
    lines.append("---")
    lines.extend(
        render_ascii_grid(rec.width, rec.height, rec.v_walls, rec.h_walls, rec.start, rec.goal)
    )
    return "\n".join(lines) + "\n"


_INT_LIST_RE = re.compile(r"-?\d+")


def _parse_int_pairs(s: str) -> List[Tuple[int, int]]:
    """`"[[7, 7], [8, 7]]"` のような文字列から (int, int) の並びを取り出す。"""
    nums = [int(m) for m in _INT_LIST_RE.findall(s)]
    if len(nums) % 2 != 0:
        raise ValueError(f"座標の並びが偶数個の整数になっていない: {s!r}")
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def _parse_optional_int(s: str) -> Optional[int]:
    s = s.strip()
    return int(s) if s else None


def _parse_optional_str(s: str) -> Optional[str]:
    s = s.strip()
    return s if s else None


def loads(text: str, path: Optional[Path] = None, kind: Optional[str] = None) -> MazeRecord:
    """`.maze` ファイルの中身（文字列）を MazeRecord へ変換する。"""
    lines = text.split("\n")
    if not lines or lines[0] != "---":
        raise ValueError("前書きの開始 '---' が無い")
    try:
        end_idx = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError("前書きの終端 '---' が無い") from exc

    front_lines = lines[1:end_idx]
    body_lines = lines[end_idx + 1 :]
    # 書き出し側は必ず末尾に改行を 1 つ付けるので、split("\n") すると最後に
    # 空文字列の要素が 1 つ残る。アスキー図の行数として数えないよう取り除く。
    if body_lines and body_lines[-1] == "":
        body_lines = body_lines[:-1]

    fields: Dict[str, str] = {}
    for ln in front_lines:
        if not ln.strip():
            continue
        key, sep, val = ln.partition(":")
        if not sep:
            raise ValueError(f"前書きの行が 'key: value' の形でない: {ln!r}")
        fields[key.strip()] = val.strip()

    missing = [k for k in ("id", "size", "start", "start_heading", "goal") if k not in fields]
    if missing:
        raise ValueError(f"前書きに必須項目が無い: {missing}")

    width, height = _parse_int_pairs(fields["size"])[0]
    start = _parse_int_pairs(fields["start"])[0]
    goal = _parse_int_pairs(fields["goal"])
    start_heading = fields["start_heading"]
    if start_heading not in _HEADING_CHARS:
        raise ValueError(f"start_heading が想定外: {start_heading!r}")

    source_type = fields.get("source_type", "")
    if source_type and source_type not in _SOURCE_TYPES:
        raise ValueError(f"source_type が想定外: {source_type!r}（{_SOURCE_TYPES} のいずれか）")

    disputes = None
    if "disputes" in fields and fields["disputes"].strip("[] ") != "":
        disputes = [s.strip() for s in fields["disputes"].strip("[]").split(",") if s.strip()]

    v_walls, h_walls, start_in_grid, goals_in_grid = parse_ascii_grid(body_lines, width, height)

    if start_in_grid is not None and tuple(start_in_grid) != tuple(start):
        raise ValueError(
            f"アスキー図の S と前書きの start が食い違う: 図={start_in_grid} 前書き={start}"
        )
    if goals_in_grid and set(goals_in_grid) != set(tuple(p) for p in goal):
        raise ValueError(
            f"アスキー図の G と前書きの goal が食い違う: 図={sorted(goals_in_grid)} "
            f"前書き={sorted(goal)}"
        )

    rec = MazeRecord(
        id=fields["id"],
        width=width,
        height=height,
        start=start,
        start_heading=start_heading,
        goal=goal,
        series=fields.get("series", ""),
        edition=_parse_optional_int(fields.get("edition", "")),
        year=_parse_optional_int(fields.get("year", "")),
        maze_class=_parse_optional_str(fields.get("class", "")),
        stage=_parse_optional_str(fields.get("stage", "")),
        source_type=source_type,
        source=_strip_quotes(fields.get("source", "")),
        source_url=_strip_quotes(fields.get("source_url", "")),
        retrieved=fields.get("retrieved", ""),
        confidence=fields.get("confidence", ""),
        content_sha256=fields.get("content_sha256", ""),
        v_walls=v_walls,
        h_walls=h_walls,
        disputes=disputes,
        notes=_parse_optional_str(_strip_quotes(fields.get("notes", ""))) if "notes" in fields else None,
        kind=kind,
        path=path,
    )
    return rec


def load(path: Path, kind: Optional[str] = None) -> MazeRecord:
    """`.maze` ファイルを読み込む。`kind` を省くと親ディレクトリ名を使う。"""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if kind is None:
        kind = path.parent.name
    return loads(text, path=path, kind=kind)


def dump(rec: MazeRecord, path: Path) -> None:
    """MazeRecord を `.maze` ファイルへ書き出す。"""
    Path(path).write_text(dumps(rec), encoding="utf-8")


# ============================================================================
# MazeDB — 属性で引く本体
# ============================================================================
_QUERY_KEY_ALIASES = {"class": "maze_class", "cls": "maze_class"}


class MazeDB:
    """`mazes/` を読み込み、属性で絞って引くための入口。

    使い方（note_036 §2-5c）:
        db = MazeDB()
        db.get("AllJapan_015_1994_exp_fin")
        db.query(kind="contest", stage="final", confidence="confirmed")
        db.query(year=range(1980, 1995), series="AllJapan")
        db.walls(rec)
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        if root is None:
            root = Path(__file__).resolve().parents[1] / "mazes"
        self.root = Path(root)
        self._records: Dict[str, MazeRecord] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.root.is_dir():
            return
        for path in sorted(self.root.glob("*/*.maze")):
            rec = load(path)
            if rec.id in self._records:
                other = self._records[rec.id].path
                raise ValueError(f"id が重複している: {rec.id}（{other} と {path}）")
            self._records[rec.id] = rec

    def __len__(self) -> int:
        return len(self._records)

    def ids(self) -> List[str]:
        return sorted(self._records.keys())

    def get(self, maze_id: str) -> MazeRecord:
        try:
            return self._records[maze_id]
        except KeyError as exc:
            raise KeyError(f"迷路データベースに id={maze_id!r} が無い") from exc

    def query(self, **attrs) -> List[MazeRecord]:
        """属性が一致する迷路を全部返す。

        値がそのまま渡された場合は等価判定、`range`/`set`/`list`/`tuple` の
        ような「in で判定できるもの」が渡された場合はその判定を使う
        （`year=range(1980, 1995)` のような使い方ができるようにするため）。
        """
        results = []
        for rec in self._records.values():
            ok = True
            for key, want in attrs.items():
                attr_name = _QUERY_KEY_ALIASES.get(key, key)
                have = getattr(rec, attr_name)
                if isinstance(want, (range, set, list, tuple)) and not isinstance(want, str):
                    if have not in want:
                        ok = False
                        break
                else:
                    if have != want:
                        ok = False
                        break
            if ok:
                results.append(rec)
        return sorted(results, key=lambda r: r.id)

    def walls(self, rec: MazeRecord) -> Tuple[np.ndarray, np.ndarray]:
        """既存コード（`competition/evaluator.py` 等）へそのまま渡せる形式で返す。

        0=壁なし・1=壁あり の uint8 配列。未知の壁が残っている迷路には使えない
        （3 値を 2 値へ丸めると情報が失われるため、その場合は例外にする）。
        """
        if np.any(rec.v_walls == WallState.UNKNOWN) or np.any(rec.h_walls == WallState.UNKNOWN):
            raise ValueError(
                f"{rec.id}: 未知の壁が残っており 0/1 表現へ変換できない"
            )
        v = np.where(rec.v_walls == WallState.WALL, 1, 0).astype(np.uint8)
        h = np.where(rec.h_walls == WallState.WALL, 1, 0).astype(np.uint8)
        return v, h
