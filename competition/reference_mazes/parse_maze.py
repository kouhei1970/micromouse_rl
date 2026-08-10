#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.maze 形式（micromouse-maze-data リポジトリの README.md 記載形式）のパーサ。

フォーマット仕様（README.md より抜粋・必ず一次資料に基づく）:
  パーツ         表現形式    注記
  柱             +           隣接壁がないときも記載
  既知壁（横）   ---         ハイフン3つ
  既知壁（縦）   |           縦棒1つ
  未知壁（横）    .          半角スペースでドットを挟む（3文字: " . "）
  未知壁（縦）   .           ドット1つ
  スタート区画    S          半角スペースでSを挟む（3文字: " S "）
  ゴール区画      G          半角スペースでGを挟む（3文字: " G "）
  その他の区画              半角スペース3つ（3文字: "   "）

  迷路は正方形。ファイルは (2*M+1) 行、各行 (4*M+1) 文字（+終端の改行）。
  偶数行番号（0-indexed, テキスト先頭から）: 柱+横壁の行 "+---+   +...+"
  奇数行番号: 縦壁+区画の行 "|   |   |...|"

  テキストの最上行がセルの最上端（y=M-1側）、最下行がセルの最下端（y=0, スタート側）
  に対応する（README添付例で S が最下行左端、G が最上行右上付近にあることから確認）。

壁の値は 3 値で保持する: 1=既知壁あり, 0=既知で壁なし(通行可), -1=未知（探索中データ用。
正式な大会迷路では出現しないはず）。
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from dataclasses import dataclass


class MazeFormatError(ValueError):
    pass


@dataclass
class ParsedMaze:
    size: int  # M（迷路の1辺の区画数）
    # h_wall[k, x] : 横壁。k=0..M（テキスト柱行のインデックス, 上から下へ）
    #   k=0 は最上端の外周壁（セル y=M-1 の上壁）
    #   k=M は最下端の外周壁（セル y=0 の下壁）
    #   一般に h_wall[k, x] はセル (x, M-k) と (x, M-k-1) の間の壁
    h_wall: np.ndarray  # shape (M+1, M), values in {-1,0,1}
    # v_wall[j, x] : 縦壁。j=0..M（テキスト壁行内の左右仕切りインデックス, 左から右へ）
    #   j=0 は左端の外周壁（セル x=0 の左壁）
    #   j=M は右端の外周壁（セル x=M-1 の右壁）
    #   一般に v_wall[j, x_row] はセル (j-1, y) と (j, y) の間の壁 (y はそのテキスト行のセルy)
    # ただし実装の都合上、行(y)方向にまとめて (M, M+1) の配列として保持する:
    #   v_wall[y, j]
    v_wall: np.ndarray  # shape (M, M+1), values in {-1,0,1}
    start: tuple  # (x, y)
    goals: list  # [(x, y), ...]
    raw_lines: list  # 元のテキスト行（検証用）


def _classify_h_token(tok: str, lineno: int, col: int) -> int:
    if tok == "---":
        return 1
    if tok == "   ":
        return 0
    if tok == " . ":
        return -1
    raise MazeFormatError(
        f"行{lineno} 列{col}: 不明な横壁トークン {tok!r}"
    )


def _classify_v_token(ch: str, lineno: int, col: int) -> int:
    if ch == "|":
        return 1
    if ch == " ":
        return 0
    if ch == ".":
        return -1
    raise MazeFormatError(
        f"行{lineno} 列{col}: 不明な縦壁文字 {ch!r}"
    )


def parse_maze_file(path: str) -> ParsedMaze:
    text = Path(path).read_text(encoding="utf-8")
    # ファイル終端の改行規則を許容しつつ行分割
    lines = text.split("\n")
    # 末尾の空行（終端改行由来）を除去
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise MazeFormatError(f"{path}: 空ファイル")

    n_lines = len(lines)
    if n_lines % 2 != 1:
        raise MazeFormatError(
            f"{path}: 行数が奇数でない (行数={n_lines})。正方形の迷路テキストではない可能性。"
        )
    M = (n_lines - 1) // 2
    if M <= 0:
        raise MazeFormatError(f"{path}: 迷路サイズが不正 (M={M})")

    expected_len = 4 * M + 1
    for i, line in enumerate(lines):
        if len(line) != expected_len:
            raise MazeFormatError(
                f"{path}: 行{i} の長さが {len(line)} (期待値 {expected_len})。"
                f" 内容: {line!r}"
            )

    h_wall = np.full((M + 1, M), -2, dtype=int)  # -2 = 未設定(バグ検出用)
    v_wall = np.full((M, M + 1), -2, dtype=int)
    start = None
    goals = []

    for lineno, line in enumerate(lines):
        if lineno % 2 == 0:
            # 柱+横壁行
            k = lineno // 2
            # 柱の位置チェック
            for x in range(M + 1):
                pos = 4 * x
                if line[pos] != "+":
                    raise MazeFormatError(
                        f"{path}: 行{lineno} 位置{pos} に柱 '+' がない: {line[pos]!r}"
                    )
            for x in range(M):
                tok = line[4 * x + 1: 4 * x + 4]
                h_wall[k, x] = _classify_h_token(tok, lineno, x)
        else:
            # 壁+区画行
            row = (lineno - 1) // 2  # 0=最上段のセル行
            y = M - 1 - row
            for j in range(M + 1):
                pos = 4 * j
                v_wall[row, j] = _classify_v_token(line[pos], lineno, j)
            for x in range(M):
                cell = line[4 * x + 1: 4 * x + 4]
                if cell == "   ":
                    pass
                elif cell == " S ":
                    if start is not None:
                        raise MazeFormatError(f"{path}: スタート区画が複数存在")
                    start = (x, y)
                elif cell == " G ":
                    goals.append((x, y))
                else:
                    raise MazeFormatError(
                        f"{path}: 行{lineno} 列{x} の区画表記が不明 {cell!r}"
                    )

    assert (h_wall != -2).all(), "h_wall 未設定セルが残っている(パーサのバグ)"
    assert (v_wall != -2).all(), "v_wall 未設定セルが残っている(パーサのバグ)"

    return ParsedMaze(
        size=M, h_wall=h_wall, v_wall=v_wall,
        start=start, goals=goals, raw_lines=lines,
    )


def to_npz_format(pm: ParsedMaze):
    """
    現行評価迷路の npz 形式に変換する。
      v_walls[x, y] : 区画(x-1,y)と(x,y)の間の縦壁。 shape (M+1, M)
      h_walls[x, y] : 区画(x,y-1)と(x,y)の間の横壁。 shape (M, M+1)
    x: 0..M (列方向、0が左端外周、Mが右端外周)
    y: 0..M-1 (v_walls) / 0..M (h_walls)、0が下端(スタート側)

    未知壁(-1)が含まれる場合は ValueError を送出する（正式な大会迷路には
    存在しないはずであり、混入は推測で埋めず明示的にエラーとする）。
    """
    M = pm.size
    if (pm.h_wall == -1).any() or (pm.v_wall == -1).any():
        raise ValueError("未知壁(-1)が含まれる。正式な大会迷路データとして不適格。")

    v_walls = np.zeros((M + 1, M), dtype=int)
    for row in range(M):  # row: 0=最上段セル行
        y = M - 1 - row
        for j in range(M + 1):
            v_walls[j, y] = pm.v_wall[row, j]

    h_walls = np.zeros((M, M + 1), dtype=int)
    for k in range(M + 1):
        # h_wall[k, x] はセル(x, M-k) と (x, M-k-1) の間の壁 -> h_walls[x, y] で y=M-k
        y = M - k
        for x in range(M):
            h_walls[x, y] = pm.h_wall[k, x]

    return v_walls, h_walls


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "micromouse-maze-data/data/16MM2019H_Kansai.maze"
    pm = parse_maze_file(p)
    print(f"size={pm.size} start={pm.start} goals={sorted(pm.goals)}")
    v_walls, h_walls = to_npz_format(pm)
    print("v_walls shape", v_walls.shape, "h_walls shape", h_walls.shape)
