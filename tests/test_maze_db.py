"""迷路データベース（`mazes/`）の検査。

設計は `research_notes/note_036_maze_database.md` を基準文書とする（§2-6 が検査の 3 段を定める）。
読み書き器は `common/maze_db.py`。

検査の構成:
    1. 往復      -- test_roundtrip_exact
    2. 構造      -- test_outer_boundary_closed 以下
    3. 索引の同期 -- test_index_tsv_matches_generated
    4. npz とのバイト単位の一致（変換元との突き合わせ） -- test_contest_maze_matches_source_npz_exactly
    5. content_sha256 が壁だけから決まること -- test_content_sha256_*
    6. 同じ壁配置なら別の出所でも同じ指紋になること -- test_identical_walls_from_different_years_share_fingerprint
    7. アスキーの文字位置の取り違えを捕まえる検査 -- test_ascii_char_positions_are_fixed

    .venv/bin/python -m pytest tests/test_maze_db.py -v
"""
import dataclasses
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_notes" / "scripts"))

from common.maze_db import (  # noqa: E402
    MazeDB,
    MazeRecord,
    WallState,
    compute_content_sha256,
    dumps,
    load,
    loads,
    parse_ascii_grid,
    render_ascii_grid,
)
from build_maze_index import build_index_tsv  # noqa: E402

MAZE_DIR = ROOT / "mazes"
CONTEST_HISTORICAL_DIR = ROOT / "competition" / "mazes" / "contest_historical"

ALL_MAZE_FILES = sorted(MAZE_DIR.glob("*/*.maze"))
_DB = MazeDB()

_MANIFEST = json.loads((CONTEST_HISTORICAL_DIR / "manifest.json").read_text(encoding="utf-8"))
_CONTEST_NPZ_NAMES = _MANIFEST["mazes"]


# ============================================================================
# 0. 前提（データベースが空でないこと）
# ============================================================================
def test_database_is_not_empty():
    assert len(_DB) == 102, "contest/ の 102 面が読み込めていない"
    assert len(ALL_MAZE_FILES) == len(_DB)


# ============================================================================
# 1. 往復検査 -- dumps(load(f)) が元のファイルと 1 文字単位で一致する
# ============================================================================
@pytest.mark.parametrize("path", ALL_MAZE_FILES, ids=[p.stem for p in ALL_MAZE_FILES])
def test_roundtrip_exact(path):
    original = path.read_text(encoding="utf-8")
    rec = loads(original, path=path, kind=path.parent.name)
    rebuilt = dumps(rec)
    assert rebuilt == original, f"{path.name}: 往復変換が元のファイルと不一致"


# ============================================================================
# 2. 構造検査
# ============================================================================
def _open_sides(v_walls, h_walls, x, y):
    """(x,y) の 4 辺のうち開いている辺。要素は ('N'|'E'|'S'|'W', (nx,ny))。"""
    sides = []
    if v_walls[x + 1, y] == WallState.OPEN:
        sides.append(("E", (x + 1, y)))
    if v_walls[x, y] == WallState.OPEN:
        sides.append(("W", (x - 1, y)))
    if h_walls[x, y + 1] == WallState.OPEN:
        sides.append(("N", (x, y + 1)))
    if h_walls[x, y] == WallState.OPEN:
        sides.append(("S", (x, y - 1)))
    return sides


_ALL_IDS = _DB.ids()


@pytest.mark.parametrize("rec_id", _ALL_IDS)
def test_outer_boundary_closed(rec_id):
    rec = _DB.get(rec_id)
    v, h = rec.v_walls, rec.h_walls
    assert np.all(v[0, :] == WallState.WALL), f"{rec_id}: 西の外周に穴"
    assert np.all(v[rec.width, :] == WallState.WALL), f"{rec_id}: 東の外周に穴"
    assert np.all(h[:, 0] == WallState.WALL), f"{rec_id}: 南の外周に穴"
    assert np.all(h[:, rec.height] == WallState.WALL), f"{rec_id}: 北の外周に穴"


@pytest.mark.parametrize("rec_id", _ALL_IDS)
def test_goal_block_has_no_internal_walls(rec_id):
    rec = _DB.get(rec_id)
    v, h = rec.v_walls, rec.h_walls
    goals = set(tuple(g) for g in rec.goal)
    for (x, y) in goals:
        for (nx, ny) in ((x + 1, y), (x, y + 1)):  # 片方向だけ見れば十分（対は 2 回検査しない）
            if (nx, ny) not in goals:
                continue
            state = v[x + 1, y] if nx == x + 1 else h[x, y + 1]
            assert state == WallState.OPEN, f"{rec_id}: ゴール区画間 ({x},{y})-({nx},{ny}) に壁"


@pytest.mark.parametrize("rec_id", _ALL_IDS)
def test_start_reaches_a_goal(rec_id):
    rec = _DB.get(rec_id)
    v, h = rec.v_walls, rec.h_walls
    start = tuple(rec.start)
    goals = set(tuple(g) for g in rec.goal)
    seen = {start}
    dq = deque([start])
    reached = start in goals
    while dq and not reached:
        x, y = dq.popleft()
        for _side, n in _open_sides(v, h, x, y):
            if n in seen:
                continue
            seen.add(n)
            if n in goals:
                reached = True
                break
            dq.append(n)
    assert reached, f"{rec_id}: スタートからゴールへ到達できない"


@pytest.mark.parametrize("rec_id", _ALL_IDS)
def test_start_cell_has_exactly_one_opening(rec_id):
    rec = _DB.get(rec_id)
    x, y = rec.start
    sides = _open_sides(rec.v_walls, rec.h_walls, x, y)
    assert len(sides) == 1, f"{rec_id}: スタート区画の開口が {len(sides)} 個（1 個のはず）"
    assert sides[0][0] == rec.start_heading, (
        f"{rec_id}: 開口方向 {sides[0][0]} が start_heading={rec.start_heading} と食い違う"
    )


# ============================================================================
# 3. 索引の同期
# ============================================================================
def test_index_tsv_matches_generated():
    generated = build_index_tsv(_DB)
    on_disk = (MAZE_DIR / "INDEX.tsv").read_text(encoding="utf-8")
    assert generated == on_disk, "INDEX.tsv が生成し直した内容と食い違う（生成し直すのを忘れていないか）"


# ============================================================================
# 4. 変換元 npz とのバイト単位の一致（102 面すべて）
# ============================================================================
@pytest.mark.parametrize("npz_name", _CONTEST_NPZ_NAMES)
def test_contest_maze_matches_source_npz_exactly(npz_name):
    assert npz_name.startswith("maze_")
    maze_id = npz_name[len("maze_"):]
    d = np.load(CONTEST_HISTORICAL_DIR / f"{npz_name}.npz", allow_pickle=True)
    rec = _DB.get(maze_id)
    v, h = _DB.walls(rec)

    assert v.dtype == np.uint8 and d["v_walls"].dtype == np.uint8
    assert h.dtype == np.uint8 and d["h_walls"].dtype == np.uint8
    assert v.tobytes() == np.ascontiguousarray(d["v_walls"]).tobytes(), (
        f"{maze_id}: v_walls が変換元 npz とバイト単位で不一致"
    )
    assert h.tobytes() == np.ascontiguousarray(d["h_walls"]).tobytes(), (
        f"{maze_id}: h_walls が変換元 npz とバイト単位で不一致"
    )
    assert (int(d["width"]), int(d["height"])) == (rec.width, rec.height)
    assert (int(d["start_x"]), int(d["start_y"])) == tuple(rec.start)
    goals_npz = set(zip(d["goals_x"].tolist(), d["goals_y"].tolist()))
    assert goals_npz == set(tuple(g) for g in rec.goal)


# ============================================================================
# 5. content_sha256 は壁だけから決まる
# ============================================================================
def test_content_sha256_matches_recomputation():
    for rec_id in _ALL_IDS:
        rec = _DB.get(rec_id)
        recomputed = compute_content_sha256(rec.width, rec.height, rec.v_walls, rec.h_walls)
        assert recomputed == rec.content_sha256, f"{rec_id}: 前書きの指紋が再計算と不一致"


def test_content_sha256_is_unaffected_by_metadata_changes():
    rec = _DB.get("AllJapan_015_1994_exp_fin")
    baseline = compute_content_sha256(rec.width, rec.height, rec.v_walls, rec.h_walls)

    # 年・出所・確度・系列など、壁以外の項目を総取り替えしても指紋は変わらない。
    changed = dataclasses.replace(
        rec,
        year=1900,
        edition=999,
        series="Dummy",
        maze_class="freshman",
        stage="preliminary",
        source_type="ascii",
        source="別の出所",
        source_url="http://example.com",
        retrieved="2000-01-01",
        confidence="disputed",
        notes="無関係な注記",
    )
    recomputed = compute_content_sha256(changed.width, changed.height, changed.v_walls, changed.h_walls)
    assert recomputed == baseline


def test_compute_content_sha256_changes_when_a_wall_changes():
    """壁を 1 枚変えれば指紋も変わる（壁を無視して定数を返す実装を弾く）。"""
    rec = _DB.get("AllJapan_015_1994_exp_fin")
    v2 = rec.v_walls.copy()
    # 外周ではない 1 枚を反転させる。
    flipped = WallState.OPEN if v2[1, 0] == WallState.WALL else WallState.WALL
    v2[1, 0] = flipped
    sha2 = compute_content_sha256(rec.width, rec.height, v2, rec.h_walls)
    assert sha2 != rec.content_sha256


# ============================================================================
# 6. 同じ壁配置なら別の出所でも同じ指紋になる
# ============================================================================
def test_identical_walls_from_different_years_share_fingerprint():
    """AllJapan_013_1992_exp_fin と AllJapan_016_1995_exp_fin は、変換元 npz の時点で
    壁配置が完全に同一（本作業で見つかった実例）。年・出所は異なるのに指紋は一致する。"""
    a = _DB.get("AllJapan_013_1992_exp_fin")
    b = _DB.get("AllJapan_016_1995_exp_fin")
    assert a.year != b.year
    assert a.source != b.source
    assert np.array_equal(a.v_walls, b.v_walls)
    assert np.array_equal(a.h_walls, b.h_walls)
    assert a.content_sha256 == b.content_sha256


def test_compute_content_sha256_is_a_pure_function_of_walls_and_size():
    width = height = 4
    v = np.full((width + 1, height), WallState.OPEN, dtype=np.int8)
    v[0, :] = WallState.WALL
    v[width, :] = WallState.WALL
    h = np.full((width, height + 1), WallState.OPEN, dtype=np.int8)
    h[:, 0] = WallState.WALL
    h[:, height] = WallState.WALL

    sha_a = compute_content_sha256(width, height, v, h)
    sha_b = compute_content_sha256(width, height, v.copy(), h.copy())
    assert sha_a == sha_b, "同じ壁配置なのに指紋が違う"

    v_changed = v.copy()
    v_changed[1, 0] = WallState.WALL
    sha_c = compute_content_sha256(width, height, v_changed, h)
    assert sha_c != sha_a, "壁を変えたのに指紋が同じ"


# ============================================================================
# 7. アスキーの文字位置を実測値で固定する（取り違えを捕まえる検査）
# ============================================================================
def test_ascii_char_positions_are_fixed():
    """`+---+` 形式の各文字がどの壁に対応するかを、実際に走らせた具体値で固定する。"""
    width = height = 2
    v = np.array(
        [
            [WallState.WALL, WallState.WALL],  # x=0（西外周）
            [WallState.WALL, WallState.OPEN],  # x=1（中の縦壁。南=壁、北=開通）
            [WallState.WALL, WallState.WALL],  # x=2（東外周）
        ],
        dtype=np.int8,
    )
    h = np.array(
        [
            [WallState.WALL, WallState.WALL, WallState.WALL],  # x=0 列（南外周・中・北外周）
            [WallState.WALL, WallState.OPEN, WallState.WALL],  # x=1 列（中だけ開通）
        ],
        dtype=np.int8,
    )
    lines = render_ascii_grid(width, height, v, h, start=(0, 0), goal=[(1, 1)])
    assert lines == [
        "+---+---+",
        "|     G |",
        "+---+   +",
        "| S |   |",
        "+---+---+",
    ]

    # 上が北であることの確認: 北寄りの区画 (1,1) の G が、南寄りの区画 (0,0) の S より
    # 前の行（上）に出る。
    s_row = next(i for i, ln in enumerate(lines) if "S" in ln)
    g_row = next(i for i, ln in enumerate(lines) if "G" in ln)
    assert g_row < s_row

    v2, h2, start2, goals2 = parse_ascii_grid(lines, width, height)
    assert np.array_equal(v2, v)
    assert np.array_equal(h2, h)
    assert start2 == (0, 0)
    assert goals2 == [(1, 1)]


def test_unknown_wall_round_trips_and_is_rejected_by_walls():
    width = height = 2
    v = np.full((width + 1, height), WallState.OPEN, dtype=np.int8)
    v[0, :] = WallState.WALL
    v[width, :] = WallState.WALL
    h = np.full((width, height + 1), WallState.OPEN, dtype=np.int8)
    h[:, 0] = WallState.WALL
    h[:, height] = WallState.WALL
    h[0, 1] = WallState.UNKNOWN  # 未知の壁を 1 枚だけ混ぜる

    sha = compute_content_sha256(width, height, v, h)
    rec = MazeRecord(
        id="unit_unknown_wall",
        width=width,
        height=height,
        start=(0, 0),
        start_heading="N",
        goal=[(1, 1)],
        series="Unit",
        edition=None,
        year=None,
        maze_class=None,
        stage=None,
        source_type="generated",
        source="unit test",
        source_url="",
        retrieved="2026-08-23",
        confidence="single-source",
        content_sha256=sha,
        v_walls=v,
        h_walls=h,
    )
    text = dumps(rec)
    assert "???" in text, "未知の壁 3 文字表現 '???' がアスキー図に出ていない"

    rec2 = loads(text)
    assert np.array_equal(rec2.v_walls, v)
    assert np.array_equal(rec2.h_walls, h)
    assert dumps(rec2) == text  # 未知を含んでいても往復は一致する

    with pytest.raises(ValueError):
        _DB.walls(rec2)  # 未知が残ったまま 0/1 表現へは変換できない


def test_start_goal_mismatch_between_front_matter_and_ascii_is_rejected():
    """図の S/G と前書きの start/goal がずれていたら読み込み時に弾く。"""
    width = height = 2
    v = np.full((width + 1, height), WallState.OPEN, dtype=np.int8)
    v[0, :] = WallState.WALL
    v[width, :] = WallState.WALL
    h = np.full((width, height + 1), WallState.OPEN, dtype=np.int8)
    h[:, 0] = WallState.WALL
    h[:, height] = WallState.WALL
    sha = compute_content_sha256(width, height, v, h)
    rec = MazeRecord(
        id="unit_mismatch",
        width=width,
        height=height,
        start=(0, 0),
        start_heading="N",
        goal=[(1, 1)],
        series="Unit",
        edition=None,
        year=None,
        maze_class=None,
        stage=None,
        source_type="generated",
        source="unit test",
        source_url="",
        retrieved="2026-08-23",
        confidence="single-source",
        content_sha256=sha,
        v_walls=v,
        h_walls=h,
    )
    text = dumps(rec)
    broken = text.replace("goal: [[1, 1]]", "goal: [[0, 1]]")
    with pytest.raises(ValueError):
        loads(broken)


# ============================================================================
# query() / get() / walls() の基本的な使い勝手
# ============================================================================
def test_get_raises_for_unknown_id():
    with pytest.raises(KeyError):
        _DB.get("no_such_maze_id")


def test_query_filters_by_kind_stage_confidence():
    finals = _DB.query(kind="contest", stage="final", confidence="single-source")
    assert len(finals) == 22  # 実測値（決勝 20 ＋ フレッシュマン決勝 2。note_035 の内訳と整合）
    assert all(r.kind == "contest" and r.stage == "final" for r in finals)


def test_query_filters_by_year_range_and_series():
    old_all_japan = _DB.query(year=range(1980, 1995), series="AllJapan")
    assert len(old_all_japan) > 0
    assert all(1980 <= r.year < 1995 and r.series == "AllJapan" for r in old_all_japan)


def test_query_by_class_accepts_keyword_and_alias():
    via_dict_key = _DB.query(**{"class": "expert"})
    via_alias = _DB.query(maze_class="expert")
    assert {r.id for r in via_dict_key} == {r.id for r in via_alias}
    assert len(via_dict_key) == 28  # 実測値
    assert all(r.maze_class == "expert" for r in via_dict_key)


def test_query_by_source_type():
    bmp_records = _DB.query(source_type="bmp")
    assert len(bmp_records) == len(_DB), "今回変換した 102 面はすべて source_type=bmp のはず"


def test_walls_returns_evaluator_style_uint8_arrays():
    rec = _DB.get("AllJapan_015_1994_exp_fin")
    v, h = _DB.walls(rec)
    assert v.dtype == np.uint8 and h.dtype == np.uint8
    assert set(np.unique(v).tolist()) <= {0, 1}
    assert set(np.unique(h).tolist()) <= {0, 1}
    assert v.shape == (rec.width + 1, rec.height)
    assert h.shape == (rec.width, rec.height + 1)
