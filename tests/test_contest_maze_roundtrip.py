#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大会実迷路（`competition/reference_mazes/contest/`）の往復変換を **全数** 検証する。

背景: `competition/reference_mazes/validate.py` は往復検証を行うが、
**manifest の先頭 1 面と中央 1 面の計 2 面しか見ていない**（同ファイル L126）。
一方 `competition/reference_mazes/README.md` は「往復検証が**全 42 面 OK**」と書いており、
**記述と実装が食い違っていた**（2026-08-15 学生A が実行して発見）。
本テストが全数を実際に検査することで、その主張を成り立たせる。

検査内容（1 面につき）:
  1. 往復変換 — npz から `.maze` テキストを再構成し、原本と 1 行 1 文字まで一致するか
  2. 外周壁がすべて閉じているか
  3. スタート区画・ゴール区画の個数と位置が妥当か

    .venv/bin/python -m pytest tests/test_contest_maze_roundtrip.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "competition" / "reference_mazes"
sys.path.insert(0, str(REF))

from parse_maze import parse_maze_file          # noqa: E402
from validate import render_from_npz            # noqa: E402

SRC_DIR = REF / "micromouse-maze-data" / "data"
OUT_DIR = REF / "contest"


def _entries():
    manifest = json.load(open(OUT_DIR / "manifest.json", encoding="utf-8"))
    return manifest["converted"]


ENTRIES = _entries()


def test_manifest_covers_all_npz():
    """manifest の件数と、実際に置かれている npz の件数が一致すること。"""
    on_disk = sorted(p.name for p in OUT_DIR.glob("*.npz"))
    listed = sorted(e["npz"] for e in ENTRIES)
    assert listed == on_disk, f"manifest {len(listed)} 件 対 実体 {len(on_disk)} 件"


@pytest.mark.parametrize("entry", ENTRIES, ids=[e["npz"][:-4] for e in ENTRIES])
def test_roundtrip_exact(entry):
    """npz → テキストの再構成が原本と完全一致すること（1 行 1 文字単位）。"""
    pm = parse_maze_file(SRC_DIR / entry["source"])
    d = np.load(OUT_DIR / entry["npz"])
    M = pm.size
    assert d["v_walls"].shape == (M + 1, M)
    assert d["h_walls"].shape == (M, M + 1)

    start_xy = (int(d["start_x"]), int(d["start_y"]))
    goals_xy = set(zip(d["goals_x"].tolist(), d["goals_y"].tolist()))
    rebuilt = render_from_npz(d["v_walls"], d["h_walls"], start_xy, goals_xy, M=M)

    assert len(rebuilt) == len(pm.raw_lines), (
        f"{entry['npz']}: 行数 再構成 {len(rebuilt)} 対 原本 {len(pm.raw_lines)}"
    )
    diff = [i for i, (r, o) in enumerate(zip(rebuilt, pm.raw_lines)) if r != o]
    assert not diff, f"{entry['npz']}: {len(diff)} 行が不一致（先頭 {diff[:5]}）"


@pytest.mark.parametrize("entry", ENTRIES, ids=[e["npz"][:-4] for e in ENTRIES])
def test_border_closed(entry):
    """外周の壁がすべて閉じていること。"""
    d = np.load(OUT_DIR / entry["npz"])
    v, h = d["v_walls"], d["h_walls"]
    M = v.shape[1]
    assert (v[0, :] == 1).all() and (v[M, :] == 1).all(), f"{entry['npz']}: 左右の外周に穴"
    assert (h[:, 0] == 1).all() and (h[:, M] == 1).all(), f"{entry['npz']}: 上下の外周に穴"


@pytest.mark.parametrize("entry", ENTRIES, ids=[e["npz"][:-4] for e in ENTRIES])
def test_start_and_goals_sane(entry):
    """スタートが 1 区画で盤内、ゴールが 1 区画以上で盤内にあること。"""
    d = np.load(OUT_DIR / entry["npz"])
    M = d["v_walls"].shape[1]
    sx, sy = int(d["start_x"]), int(d["start_y"])
    assert 0 <= sx < M and 0 <= sy < M, f"{entry['npz']}: スタートが盤外 ({sx},{sy})"
    gx, gy = d["goals_x"], d["goals_y"]
    assert len(gx) >= 1 and len(gx) == len(gy), f"{entry['npz']}: ゴールの個数が不正"
    assert all(0 <= x < M for x in gx) and all(0 <= y < M for y in gy), \
        f"{entry['npz']}: ゴールが盤外"
