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
    8. kerikun11 取り込み（段3）固有の検査 -- test_kerikun11_* 以下
       - 未知壁を含む実面の往復が1文字単位で一致すること
       - walls(rec, unknown=...) の3通りの振る舞い
       - disputes が相互に指していること
       - 指紋が一致する組を機械的に洗い出せること（既知の3組を含む）
    9. 井谷氏Wiki由来（2004・2006年、note_035 続報）固有の検査 -- test_itani_2004_* 以下
       - 最短経路の区画数・ターン数が、出典が独立な優勝記録の「〇区〇折」と一致すること
       - 2004年の元画像2枚（cose1.gif/cose2.gif、版管理外）が読み取れる場合だけ、
         両者の壁配置が完全一致することを検査する（無ければスキップ）

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
    # 102 面（NTF・段2）＋ 53 面（kerikun11・段3）＋ 2 面（2004・2006年 井谷氏Wiki日記の
    # 写真/シミュレータ画面からの読み取り、note_035 続報）。
    # 2026-08-23 の一括取り込みで 291 面（contest 221 + handmade 70）になった。
    # 内訳: NTF の BMP 102・kerikun11 53・井谷氏の日記の画像 2・井谷氏の MAZ.zip 109・
    #       NTF の年次画像 25。詳しくは note_035 の追記 6〜10。
    assert len(_DB) == 291, "291 面が読み込めていない"
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
    # 実測値。2026-08-23 の一括取り込み後（note_035 の追記 6〜10）。
    # 出所は NTF の BMP・kerikun11・井谷氏の日記の画像・井谷氏の MAZ.zip・NTF の年次画像。
    # confidence が confirmed へ格上げされた面はこの数から外れる。
    assert len(finals) == 52
    assert all(r.kind == "contest" and r.stage == "final" for r in finals)


def test_query_filters_by_year_range_and_series():
    old_all_japan = _DB.query(year=range(1980, 1995), series="AllJapan")
    assert len(old_all_japan) > 0
    assert all(1980 <= r.year < 1995 and r.series == "AllJapan" for r in old_all_japan)


def test_query_by_class_accepts_keyword_and_alias():
    via_dict_key = _DB.query(**{"class": "expert"})
    via_alias = _DB.query(maze_class="expert")
    assert {r.id for r in via_dict_key} == {r.id for r in via_alias}
    # 実測値。NTF 側 28 に、kerikun11 の全日本エキスパート（決勝9＋予選1）10 が加わって 38。
    # さらに井谷氏Wiki由来の 2004・2006年エキスパート決勝 2 が加わって 40。
    assert len(via_dict_key) == 61
    assert all(r.maze_class == "expert" for r in via_dict_key)


def test_query_by_source_type():
    # 2026-08-23 の一括取り込み後の実測値。
    #   bmp    = NTF の BMP 102 ＋ 井谷氏の日記の画像 2 ＋ NTF の年次画像 22
    #   ascii  = kerikun11 53 ＋ NTF の taikai の全角アスキー 3
    #   binary = 井谷氏の MAZ.zip 109
    bmp_records = _DB.query(source_type="bmp")
    ascii_records = _DB.query(source_type="ascii")
    binary_records = _DB.query(source_type="binary")
    assert len(bmp_records) == 126
    assert len(ascii_records) == 56
    assert len(binary_records) == 109
    assert len(bmp_records) + len(ascii_records) + len(binary_records) == len(_DB)


def test_walls_returns_evaluator_style_uint8_arrays():
    rec = _DB.get("AllJapan_015_1994_exp_fin")
    v, h = _DB.walls(rec)
    assert v.dtype == np.uint8 and h.dtype == np.uint8
    assert set(np.unique(v).tolist()) <= {0, 1}
    assert set(np.unique(h).tolist()) <= {0, 1}
    assert v.shape == (rec.width + 1, rec.height)
    assert h.shape == (rec.width, rec.height + 1)


# ============================================================================
# 8. kerikun11 取り込み（段3）固有の検査
# ============================================================================
# 未知壁（'.' → '?'）を実際に含む面。kerikun11 の Cheese 系 3 面のみ
# （note_036 §6-1 の未解決点はこの3面で答えが出た）。
_UNKNOWN_WALL_IDS = ("Cheese_2017_k11h", "Cheese_2019_k11h", "Cheese_2019_k11h_cand")


def test_kerikun11_unknown_wall_ids_are_exactly_the_cheese_series():
    """未知壁を含む面を機械的に洗い出すと、上記の3面と一致する（決め打ちの一覧が古くならないことの検査）。"""
    found = []
    for rec_id in _ALL_IDS:
        rec = _DB.get(rec_id)
        if np.any(rec.v_walls == WallState.UNKNOWN) or np.any(rec.h_walls == WallState.UNKNOWN):
            found.append(rec_id)
    assert sorted(found) == sorted(_UNKNOWN_WALL_IDS)


@pytest.mark.parametrize("rec_id", _UNKNOWN_WALL_IDS)
def test_kerikun11_unknown_wall_real_maze_roundtrips_exactly(rec_id):
    """未知壁を含む実在の面（Cheese 系）も、他の152面と同じく1文字単位で往復すること。"""
    rec = _DB.get(rec_id)
    original = rec.path.read_text(encoding="utf-8")
    rebuilt = dumps(rec)
    assert rebuilt == original
    assert "???" in original and "?" in original  # 未知壁の記号（横???・縦?）が実際に出ている


@pytest.mark.parametrize("rec_id", _UNKNOWN_WALL_IDS)
def test_kerikun11_walls_unknown_argument_three_behaviors(rec_id):
    """walls(rec, unknown=...) の3通りの振る舞いを固定する。"""
    rec = _DB.get(rec_id)

    # 既定（省略）: 例外（既存の振る舞いを変えていないこと）
    with pytest.raises(ValueError):
        _DB.walls(rec)

    # 'wall': 未知を壁ありとみなす（悲観）
    v_wall, h_wall = _DB.walls(rec, unknown="wall")
    assert v_wall.dtype == np.uint8 and h_wall.dtype == np.uint8
    unknown_v = rec.v_walls == WallState.UNKNOWN
    unknown_h = rec.h_walls == WallState.UNKNOWN
    assert np.all(v_wall[unknown_v] == 1)
    assert np.all(h_wall[unknown_h] == 1)

    # 'open': 未知を壁なしとみなす（楽観）
    v_open, h_open = _DB.walls(rec, unknown="open")
    assert np.all(v_open[unknown_v] == 0)
    assert np.all(h_open[unknown_h] == 0)

    # 既知の壁（未知でない部分）は unknown の指定によらず一致する
    known_v = ~unknown_v
    known_h = ~unknown_h
    assert np.array_equal(v_wall[known_v], v_open[known_v])
    assert np.array_equal(h_wall[known_h], h_open[known_h])

    # 想定外の値は弾く
    with pytest.raises(ValueError):
        _DB.walls(rec, unknown="not-a-real-option")


def test_kerikun11_walls_unknown_argument_is_noop_when_no_unknown_walls():
    """未知壁が無い面では、unknown='wall'/'open'/省略のいずれでも同じ結果になる。"""
    rec = _DB.get("AllJapan_015_1994_exp_fin")
    v0, h0 = _DB.walls(rec)
    v_wall, h_wall = _DB.walls(rec, unknown="wall")
    v_open, h_open = _DB.walls(rec, unknown="open")
    assert np.array_equal(v0, v_wall) and np.array_equal(h0, h_wall)
    assert np.array_equal(v0, v_open) and np.array_equal(h0, h_open)


def test_disputes_point_at_each_other_bidirectionally():
    """disputes を使う面があれば、必ず相互に指し合っていること（片方向だけは不正）。

    2026-08-23 時点では disputes を使う面はまだ無い（NTF と kerikun11 の指紋一致は
    confirmed への格上げか notes での相互参照で扱い、disputes は「食い違い」専用に
    温存した）。このテストは disputes の仕組み自体の検査であり、使用例が増える
    段4（map.html の食い違う5面）以降も壊れないことを保証する。
    """
    all_recs = {r.id: r for r in _DB.query()}
    any_disputes = False
    for rec in all_recs.values():
        if not rec.disputes:
            continue
        any_disputes = True
        for other_id in rec.disputes:
            assert other_id in all_recs, f"{rec.id}: disputes が指す {other_id!r} が存在しない"
            other = all_recs[other_id]
            assert other.disputes and rec.id in other.disputes, (
                f"{rec.id} -> {other_id} は片方向（{other_id} 側が {rec.id} を指し返していない）"
            )
    # 現状は使用例が無いことも合わせて明示する（今後 disputes が実際に使われたら
    # このアサーションだけ外せばよい）。
    # 2026-08-23: disputes が実際に使われ始めた（井谷氏の MAZ.zip と既存の面の食い違い。
    # note_036 §2-4「食い違う資料は両方残し、相互に指す」）。上の相互参照の検査が本体になった。
    assert any_disputes is True, "disputes が 1 件も無い。食い違いの記録が失われていないか確かめること"


def test_fingerprints_can_be_grouped_mechanically_and_find_known_matches():
    """content_sha256 で全面を機械的に束ねると、既知の3組が見つかること（note_036 §2-5b の検査）。"""
    groups: dict = {}
    for rec in _DB.query():
        groups.setdefault(rec.content_sha256, []).append(rec.id)
    duplicate_groups = {sha: sorted(ids) for sha, ids in groups.items() if len(ids) > 1}

    # 1) NTF 内部の重複（note_035 の追記で見つかった1992/1995）
    # 1992 と 1995 は NTF のアーカイブ側の重複（note_035 追記 3）。井谷氏の MAZ.zip の
    # 同じ迷路が加わって組が 3 つ以上になることがあるので、「同じ組にいる」ことを見る。
    _pair_group = [g for g in duplicate_groups.values()
                   if "AllJapan_013_1992_exp_fin" in g and "AllJapan_016_1995_exp_fin" in g]
    assert _pair_group, "1992 と 1995 が同じ指紋の組に入っていない"

    # 2) NTF と kerikun11 が同じ大会・同じ年で一致した組（段3で確認）
    _k11_group = [g for g in duplicate_groups.values()
                  if "AllJapan_033_2012_exp_fin" in g and "AllJapan_033_2012_exp_fin__kerikun11" in g]
    assert _k11_group, "2012 の NTF 版と kerikun11 版が同じ指紋の組に入っていない"

    # 3) 🔴 別大会・別年のはずなのに一致した組（段3で発見。未解決のまま報告済み）
    assert ["APEC2002_2002", "AllJapan_039_2018_exp_fin"] in duplicate_groups.values()

    # 一致した2組はいずれも confidence: confirmed への格上げか notes の相互参照が
    # されていること（サイレントな重複のまま放置していないこと）。
    for a_id, b_id in (
        ("AllJapan_033_2012_exp_fin", "AllJapan_033_2012_exp_fin__kerikun11"),
        ("APEC2002_2002", "AllJapan_039_2018_exp_fin"),
    ):
        a, b = _DB.get(a_id), _DB.get(b_id)
        assert (a.confidence == "confirmed" and b.confidence == "confirmed") or (a.notes and b.notes), (
            f"{a_id} / {b_id}: 指紋一致が記録に反映されていない"
        )


# ============================================================================
# 9. 井谷氏Wiki由来（2004・2006年、note_035 続報）固有の検査
# ============================================================================
# 出典: https://w.atwiki.jp/mm3sakusya/pages/25.html
#   （NTF発行冊子「マイクロマウス2000」を出典とする優勝記録の表。最短コース長
#   欄が「〇区〇折」の形で載っている。読み取った壁配置とは完全に独立な一次資料）。
_ITANI_RECORD_COURSE = {
    # (id): (最短区画数, 最小ターン数)
    "AllJapan_025_2004_exp_fin": (86, 49),
    "AllJapan_027_2006_exp_fin": (69, 48),
}


@pytest.mark.parametrize("rec_id, expected", sorted(_ITANI_RECORD_COURSE.items()))
def test_itani_maze_shortest_path_matches_published_record(rec_id, expected):
    """最短経路の区画数・ターン数が、優勝記録表の「〇区〇折」と一致すること。

    区画数は歩数マップの距離（移動回数）、ターン数は同じ歩数の経路が複数ある
    場合にターン数最小のものを採る `classic/route.py` の規約に従う（実機が
    速い方を選ぶのと同じ理由）。両方とも読み取った壁配置だけから計算しており、
    比較先の「〇区〇折」は読み取りに一切使っていない独立な数字である。
    """
    from classic.flood import FloodMode
    from classic.maze_map import Direction, MazeMap
    from classic.route import CommandType, path_to_commands, shortest_path

    rec = _DB.get(rec_id)
    maze = MazeMap(rec.width, rec.height)
    maze.v_walls = rec.v_walls.copy()
    maze.h_walls = rec.h_walls.copy()

    path = shortest_path(maze, rec.start, rec.goal, FloodMode.PESSIMISTIC)
    commands = path_to_commands(path, start_heading=Direction[rec.start_heading])

    cells = len(path) - 1
    turns = sum(
        1
        for c in commands
        if c.type in (CommandType.TURN_LEFT90, CommandType.TURN_RIGHT90, CommandType.TURN_180)
    )
    expected_cells, expected_turns = expected
    assert (cells, turns) == (expected_cells, expected_turns), (
        f"{rec_id}: 最短経路が {cells}区画{turns}折 — 記録表の "
        f"{expected_cells}区{expected_turns}折と食い違う"
    )


def test_itani_2004_cose_images_agree_when_available():
    """2004年の元画像2枚（斜め優先/直線優先の別経路を解析したスクリーンショット）
    が手元にある場合だけ、両者から読み取った壁配置が完全一致することを確かめる。

    画像そのものは一次資料の複製可否が未確認のため版管理に含めていない
    （note_036 §6-3）。そのため既定ではスキップされる。手元に置く場合は
    次のいずれかに `itani_2004_cose1.gif` / `itani_2004_cose2.gif` を置く。
    """
    import os

    from decode_itani_2004_cose import decode

    candidates = []
    env_dir = os.environ.get("ITANI_2004_COSE_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(ROOT / "research_notes" / "scripts" / "testdata" / "itani_2004")

    cose_dir = next((d for d in candidates if (d / "itani_2004_cose1.gif").exists()), None)
    if cose_dir is None:
        pytest.skip(
            "元画像（itani_2004_cose1.gif/cose2.gif）が手元に無いためスキップ"
            "（版管理には含めていない。note_036 §6-3）"
        )

    v1, h1 = decode(cose_dir / "itani_2004_cose1.gif")
    v2, h2 = decode(cose_dir / "itani_2004_cose2.gif")
    assert np.array_equal(v1, v2) and np.array_equal(h1, h2), (
        "cose1.gif と cose2.gif から読み取った壁配置が食い違う（読み取りが誤っている疑い）"
    )

    # 読み取り結果は版管理下の .maze 本体とも一致するはず。
    rec = _DB.get("AllJapan_025_2004_exp_fin")
    v_db, h_db = _DB.walls(rec)
    assert np.array_equal(v1, v_db) and np.array_equal(h1, h_db), (
        "cose1.gif の読み取り結果が mazes/contest/AllJapan_025_2004_exp_fin.maze と食い違う"
    )
