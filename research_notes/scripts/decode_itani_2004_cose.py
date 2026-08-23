"""井谷氏Wiki（mm3sakusya）2004-11日記の迷路探索シミュレータ画面 2枚から壁配置を読み取る。

対象: `itani_2004_cose1.gif`・`itani_2004_cose2.gif`（330x351/330x352）。
2004年全日本大会決勝の同一迷路を、斜め優先・直線優先の異なる経路で解析した
2枚のスクリーンショット（`research_notes/note_035_historical_contest_mazes.md`
2026-08-23 追記参照）。**画像そのものは版管理に含めない**（井谷氏の一次資料の複製の
可否が未確認のため、note_036 §6-3）。

書式（実測）: 黒地に赤い壁線・青い経路線・白文字（S/G）。柱の格子間隔 20px、
壁の太さ約3px、迷路領域は画像内オフセット (x0=4, y0=24) から 16x16。
ウィンドウのタイトルバー（Windows のシミュレータのスクリーンショット）を
除いた領域だけを見る。

このモジュールは `tests/test_maze_db.py` の
`test_itani_2004_cose_images_agree_when_available` から呼ばれる。元画像が
手元にある場合だけ検査が走り、無い場合はスキップされる（note_036 §6-3・
「画像は複製しないが、検査の枠組みは残す」）。
"""
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

X0, Y0, PITCH, N = 4, 24, 20, 16


def decode(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """画像1枚から (v_walls, h_walls) を 0/1 で読み取る。

    戻り値の形状・添字の向きは `common.maze_db` の v_walls/h_walls と同じ約束
    （0番目が西端、v[x,y] は区画(x,y)の西側の壁）。
    """
    im = Image.open(path).convert("RGB")
    a = np.array(im)
    red = (a[:, :, 0] > 150) & (a[:, :, 1] < 100) & (a[:, :, 2] < 100)

    v = np.zeros((N + 1, N), dtype=np.uint8)
    h = np.zeros((N, N + 1), dtype=np.uint8)
    for k in range(N + 1):  # 縦壁の線番号（西=0 → 東=N）
        xline = X0 + PITCH * k + 1
        for j in range(N):  # 上から j 番目の行帯
            yline = Y0 + PITCH * j + PITCH // 2
            y_maze = N - 1 - j
            v[k, y_maze] = 1 if red[yline, xline] else 0
    for k in range(N + 1):  # 横壁の線番号（北=0 → 南=N、画像の上から数える）
        yline = Y0 + PITCH * k + 1
        for i in range(N):  # 左から i 番目の列帯
            xline = X0 + PITCH * i + PITCH // 2
            h_level = N - k
            h[i, h_level] = 1 if red[yline, xline] else 0
    return v, h


if __name__ == "__main__":
    import sys

    v1, h1 = decode(Path(sys.argv[1]))
    v2, h2 = decode(Path(sys.argv[2]))
    print("v差分:", int(np.sum(v1 != v2)), "/", v1.size)
    print("h差分:", int(np.sum(h1 != h2)), "/", h1.size)
