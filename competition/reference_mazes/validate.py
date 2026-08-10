#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
変換の正しさを検証する。

方法: npz (v_walls, h_walls, start, goals) から .maze テキストを逆生成し、
元のテキストファイルと完全一致するかを1行1文字単位で突き合わせる（往復検証）。
一致すれば、パーサ→変換の対応関係にバグがないことの強い証拠になる。
加えて、外周壁がすべて1であること、スタート区画・ゴール区画周辺の壁が
テキストの直接目視と一致することを個別にも確認する。
"""
import sys, glob
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_maze import parse_maze_file

SRC_DIR = Path(__file__).resolve().parent / "micromouse-maze-data" / "data"
OUT_DIR = Path(__file__).resolve().parent / "contest"


def render_from_npz(v_walls, h_walls, start_xy, goals_xy, M=16):
    """npz 配列(現行フォーマット)から .maze テキストを再構成する。
    parse_maze_file の変換と完全に逆の対応を用いる。"""
    lines = []
    for lineno in range(2 * M + 1):
        if lineno % 2 == 0:
            k = lineno // 2
            y = M - k  # h_walls の y 座標（k=0 は上端 y=M）
            chars = ["+"]
            for x in range(M):
                w = h_walls[x, y]
                chars.append("---" if w == 1 else "   ")
                chars.append("+")
            lines.append("".join(chars))
        else:
            row = (lineno - 1) // 2
            y = M - 1 - row
            chars = []
            for j in range(M + 1):
                w = v_walls[j, y]
                chars.append("|" if w == 1 else " ")
                if j < M:
                    x = j
                    if (x, y) == tuple(start_xy):
                        chars.append(" S ")
                    elif (x, y) in goals_xy:
                        chars.append(" G ")
                    else:
                        chars.append("   ")
            lines.append("".join(chars))
    return lines


def validate_file(maze_path: Path, npz_path: Path):
    print(f"\n=== 検証対象: {maze_path.name} <-> {npz_path.name} ===")
    pm = parse_maze_file(str(maze_path))
    d = np.load(npz_path, allow_pickle=True)
    v_walls = d["v_walls"]
    h_walls = d["h_walls"]
    start_xy = (int(d["start_x"]), int(d["start_y"]))
    goals_xy = set(zip(d["goals_x"].tolist(), d["goals_y"].tolist()))

    M = pm.size
    assert v_walls.shape == (M + 1, M), f"v_walls shape mismatch: {v_walls.shape}"
    assert h_walls.shape == (M, M + 1), f"h_walls shape mismatch: {h_walls.shape}"

    # --- 1. 往復検証: npz から .maze テキストを再構成して元テキストと完全一致するか ---
    reconstructed = render_from_npz(v_walls, h_walls, start_xy, goals_xy, M=M)
    orig = pm.raw_lines
    assert len(reconstructed) == len(orig), (
        f"行数不一致: 再構成{len(reconstructed)} vs 元{len(orig)}"
    )
    n_diff = 0
    for i, (r, o) in enumerate(zip(reconstructed, orig)):
        if r != o:
            n_diff += 1
            print(f"  [不一致] 行{i}:")
            print(f"    元      : {o}")
            print(f"    再構成  : {r}")
    if n_diff == 0:
        print(f"  [OK] 往復検証: 全{len(orig)}行が完全一致 (テキスト <-> npz 相互変換で情報欠損・誤りなし)")
    else:
        print(f"  [NG] 往復検証: {n_diff}/{len(orig)} 行が不一致")

    # --- 2. 外周壁がすべて1か ---
    border_ok = (
        (v_walls[0, :] == 1).all() and (v_walls[M, :] == 1).all()
        and (h_walls[:, 0] == 1).all() and (h_walls[:, M] == 1).all()
    )
    print(f"  外周壁(すべて1か): {'OK' if border_ok else 'NG'}")
    if not border_ok:
        print(f"    v_walls[0,:]={v_walls[0,:]}")
        print(f"    v_walls[{M},:]={v_walls[M,:]}")
        print(f"    h_walls[:,0]={h_walls[:,0]}")
        print(f"    h_walls[:,{M}]={h_walls[:,M]}")

    # --- 3. スタート区画の壁構成を、元テキストの目視位置と突き合わせ ---
    sx, sy = start_xy
    print(f"  スタート区画: ({sx},{sy})")
    print(f"    周囲の壁: 左v_walls[{sx},{sy}]={v_walls[sx,sy]} "
          f"右v_walls[{sx+1},{sy}]={v_walls[sx+1,sy]} "
          f"下h_walls[{sx},{sy}]={h_walls[sx,sy]} "
          f"上h_walls[{sx},{sy+1}]={h_walls[sx,sy+1]}")

    # --- 4. ゴール区画群の内壁(隣接ゴール間)が正しく無壁(0)かを確認 ---
    print(f"  ゴール区画: {sorted(goals_xy)}")
    for (gx, gy) in sorted(goals_xy):
        for (dx, dy, kind) in [(1, 0, "v"), (0, 1, "h")]:
            nb = (gx + dx, gy + dy)
            if nb in goals_xy:
                if kind == "v":
                    w = v_walls[gx + 1, gy]
                else:
                    w = h_walls[gx, gy + 1]
                mark = "OK(無壁)" if w == 0 else "??壁あり"
                print(f"    ゴール({gx},{gy})-({nb[0]},{nb[1]}) 間: {w} {mark}")

    return n_diff == 0 and border_ok


if __name__ == "__main__":
    import json
    manifest = json.load(open(OUT_DIR / "manifest.json", encoding="utf-8"))
    targets = manifest["converted"][:1] + [manifest["converted"][len(manifest["converted"]) // 2]]
    all_ok = True
    for entry in targets:
        maze_path = SRC_DIR / entry["source"]
        npz_path = OUT_DIR / entry["npz"]
        ok = validate_file(maze_path, npz_path)
        all_ok = all_ok and ok
    print(f"\n=== 総合結果: {'全項目OK' if all_ok else '不一致あり'} ===")
