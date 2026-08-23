"""井谷氏Wiki（mm3sakusya）2005-11日記の迷路図（`meiro.JPG`）から壁配置を読み取る。

対象: `2005alljapanmaze.png`（1160x1064・RGBA、ユーザが手作業で補正した画像）。
元は投影スクリーンに映した迷路図を斜めから撮った320x240の写真で、以前の担当は
「右側の境界線が行ごとに±25px ぶれ、外周が閉じない」として読み取りを断念していた
（`research_notes/note_035_historical_contest_mazes.md` 追記8）。ユーザが人間の目で
再現可能な状態まで補正した画像を本スクリプトの入力とする。

## 手順（読み取りの手続き）

1. **外周4辺の検出と射影変換**: 各辺について、画像端から内側へスキャンして
   明るい画素の重心を行/列ごとに求め、そこから前の推定位置±12pxの窓で追跡する
   （`_track_border` 関数）。単純な「端から最初に明るい所」だと、右上角の
   黒い遮蔽物（後述）に近い行で誤って内側の壁を拾う外れ値が出るため、
   MAD（中央絶対偏差）に基づく反復ロバスト直線当てはめで外れ値を除いてから
   4辺の直線を求め、交点を4隅とする。
   **右端はわずかに末広がりになっている**（上端 x≈1135・下端 x≈1142、
   すぼまり自体は数px程度で、外周検出の妨げにはならなかった）。
2. 4隅から16×16マス（1マス66px）＋余白40pxの正方形へ `cv2.getPerspectiveTransform`
   で射影変換する。**余白を入れる理由**: 外周の壁がちょうど画像端に来ると
   `warpPerspective` の補間で壁の輝度が途中で切れてしまうため。
3. **壁の判定**: 単純な輝度の固定しきい値は、写真全体に明暗の勾配があるため
   使えなかった（実測: 背景の輝度が画像の場所によって約110〜150まで変動し、
   壁と背景の差 (~10〜30) と同程度の幅がある）。
   代わりに `cv2.adaptiveThreshold`（ガウス窓・blockSize=61, C=-5）で
   各画素ごとに局所背景を基準に二値化し、各壁候補位置について
   「マス目に沿った方向に走査した各点で、直交方向±6pxの窓に1画素でも
   閾値超えがあるか」の的中率（fraction）を求める。的中率のヒストグラムは
   0付近（開放）と1付近（壁）の二峰性がはっきり出ており、谷は0.35〜0.55に
   ある。**しきい値0.5はこの実測した谷から選んだ**。
4. 構造検査（後述）で外周に1箇所だけ穴が残った。北東角のセル(15,15)の
   北壁・東壁の一部が、写真右上の黒い遮蔽物（obliqueに写り込んだ物体。
   井谷氏の日記の他の写真にも同種の黒い三角領域がある）に隠れて画素情報が
   ほぼゼロだった。**この1箇所だけは、外周は競技規則上必ず閉じているという
   構造的な確実性に基づき壁ありとした**（当てはめではない。他の215箇所の
   外周はすべて画素から独立に壁ありと読めている）。
5. 全マス到達性検査で、(0,11)-(1,11)-(2,11) の3マスが他から孤立している
   ことが分かった（**コース長を見る前に**、構造検査の一環として発見した）。
   この3マスを外へつなぐ6箇所の壁（境界の的中率0.54〜0.81）はどれも、
   確実な壁（的中率0.9超）や確実な開放（的中率0.1未満）の水準に届いておらず、
   この一角だけ写真の条件（光量か遮蔽）が悪かったとみられる。
   **的中率が最も低く0.5に最も近い1枚（h(0,12)、的中率0.537）を誤読と判断し、
   開放にした。**どれを開けても到達性は回復するため一意には決め切れないが、
   自分の測定の中で最も自信が無い1枚を選んだ、という以上の根拠はない。

## 🔴 結果 — コース長が一致しなかった

優勝記録表（NTF発行『マイクロマウス2000』、mm3sakusya Wiki pages/25.html 経由）は
2005年（第26回）の最短コース長を「73区50折」とする。本読み取りの結果は
次の通りで、**どちらのスタート角から測っても一致しなかった**。

    (0,0)始点・北向き   → 17区画・12折
    (15,15)始点・西向き → 33区画・23折

外周検査を通ると開口が1個の角は (0,0) と (15,15) の対角2箇所だけであり、
このどちらを始点としても73に程遠い。

**数字を合わせるための壁の当てはめは行っていない。**根拠:

- 独立な2通りの二値化（本スクリプトの適応的しきい値法／局所中央値背景差分法
  [しきい値8.7・壁数250、試行時のみ使用でファイルには残していない]）が
  壁配置・経路長ともにほぼ一致した（壁数250 vs 270、経路長16〜17区画）
- 最短経路が通る17辺すべてを個別に画素レベルで目視検査したが、
  壁が見落とされている形跡はなかった
- 格子当てはめの誤差を疑い、17本の縦格子線すべてについて幅広い探索窓
  （±30px）でロバスト再フィットしたが、公称位置からのずれは
  ほぼ全列で6px以内だった（東端の1列のみ+6.0px）

以上から、**本画像に描かれている壁配置をそのまま読み取る限り、
公式記録の73区50折とは一致しない**という結論に至った。原因は特定できない
（誤って別の年/回の図を写した、写真自体が競技で使われた最終図面ではない、
等が考えられるが、いずれも確認する一次資料がない）。

**そのため `mazes/contest/` への収載は見送った。**本ファイルは読み取りの
手続きを再現・検証できるように残す。

実行方法:
    .venv/bin/python research_notes/scripts/decode_itani_2005_maze.py <画像パス>
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from classic.maze_map import Direction, MazeMap, WallState  # noqa: E402
from classic.flood import FloodMode, compute_flood  # noqa: E402
from classic.route import CommandType, path_to_commands, shortest_path  # noqa: E402

N = 16
CELL = 66  # 1マス約66px（ユーザ補正後の画像で実測）
MARGIN = 40
GRID = CELL * N
DST = GRID + 2 * MARGIN
BRIGHT_TH = 152.0  # 外周追跡専用のしきい値（背景~110-150・壁~150-200の実測に基づく）
FRAC_TH = 0.5  # 壁判定の的中率しきい値（0付近と1付近の二峰分布の谷から選定）
GOAL = [(7, 7), (8, 7), (7, 8), (8, 8)]
START = (0, 0)


# ============================================================================
# 1. 外周4辺の追跡と4隅の推定
# ============================================================================
def _weighted_center_near(vals: np.ndarray, guess: int, win: int, th: float) -> Optional[float]:
    """guess±win の範囲でth超えの重心位置を返す（無ければNone）。"""
    lo, hi = max(0, guess - win), min(len(vals), guess + win + 1)
    seg = vals[lo:hi]
    mask = seg > th
    if mask.sum() < 3:
        return None
    idx = np.arange(lo, hi)[mask]
    w = seg[mask] - th
    return float(np.sum(idx * w) / np.sum(w))


def _track_border(gray: np.ndarray, axis: str, y0_or_x0: int, guess0: int, win: int = 12) -> Dict[int, float]:
    """境界線を初期点から両方向へ追跡する。

    axis='v': 縦の境界線（左右の外周）を、各行 y ごとに x 位置を追跡する。
    axis='h': 横の境界線（上下の外周）を、各列 x ごとに y 位置を追跡する。
    """
    H, W = gray.shape
    result: Dict[int, float] = {}
    pos = guess0
    rng = range(y0_or_x0, (H if axis == "v" else W) - 15)
    for i in rng:
        line = gray[i, :] if axis == "v" else gray[:, i]
        c = _weighted_center_near(line, int(round(pos)), win, BRIGHT_TH)
        if c is None:
            continue
        result[i] = c
        pos = c
    pos = guess0
    for i in range(y0_or_x0, 15, -1):
        line = gray[i, :] if axis == "v" else gray[:, i]
        c = _weighted_center_near(line, int(round(pos)), win, BRIGHT_TH)
        if c is None:
            continue
        result[i] = c
        pos = c
    return result


def _robust_fit(points: Dict[int, float], n_iter: int = 8, k: float = 4.0) -> Tuple[float, float]:
    """(添字, 値) の組にロバストな直線 value = a*index + b を当てはめる。"""
    idx = np.array(sorted(points.keys()), dtype=np.float64)
    val = np.array([points[i] for i in sorted(points.keys())], dtype=np.float64)
    keep = np.ones(len(idx), dtype=bool)
    a = b = 0.0
    for _ in range(n_iter):
        A = np.vstack([idx[keep], np.ones(keep.sum())]).T
        a, b = np.linalg.lstsq(A, val[keep], rcond=None)[0]
        resid = val - (a * idx + b)
        mad = np.median(np.abs(resid[keep] - np.median(resid[keep]))) + 1e-9
        thresh = max(2.0, k * 1.4826 * mad)
        new_keep = np.abs(resid) < thresh
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return a, b


def find_corners(gray: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """外周4辺を追跡し、4隅の座標を返す（射影変換の入力点）。"""
    H, W = gray.shape
    right = _track_border(gray, "v", 500, 1132)
    left = _track_border(gray, "v", 500, 65)
    top = _track_border(gray, "h", 500, 42)
    bot = _track_border(gray, "h", 500, 1035)

    ra, rb = _robust_fit(right)  # x = ra*y + rb
    la, lb = _robust_fit(left)
    ta, tb = _robust_fit(top)  # y = ta*x + tb
    ba, bb = _robust_fit(bot)

    def intersect(a_x_of_y: float, b_x_of_y: float, a_y_of_x: float, b_y_of_x: float) -> Tuple[float, float]:
        denom = 1 - a_y_of_x * a_x_of_y
        y = (a_y_of_x * b_x_of_y + b_y_of_x) / denom
        x = a_x_of_y * y + b_x_of_y
        return x, y

    return {
        "TL": intersect(la, lb, ta, tb),
        "TR": intersect(ra, rb, ta, tb),
        "BL": intersect(la, lb, ba, bb),
        "BR": intersect(ra, rb, ba, bb),
    }


def warp_to_grid(im: np.ndarray, corners: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """4隅から射影変換し、余白付きの正方形画像（DST×DST）へ補正する。"""
    src = np.array([corners["TL"], corners["TR"], corners["BR"], corners["BL"]], dtype=np.float32)
    dst = np.array(
        [[MARGIN, MARGIN], [MARGIN + GRID, MARGIN], [MARGIN + GRID, MARGIN + GRID], [MARGIN, MARGIN + GRID]],
        dtype=np.float32,
    )
    Hmat = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(im, Hmat, (DST, DST), borderValue=(0, 0, 0))


# ============================================================================
# 2. 壁の判定（適応的二値化 + マス目方向の的中率）
# ============================================================================
def _h_center(x: int, level: int) -> Tuple[float, float]:
    return MARGIN + (x + 0.5) * CELL, MARGIN + (N - level) * CELL


def _v_center(k: int, y: int) -> Tuple[float, float]:
    row_from_top = N - 1 - y
    return MARGIN + k * CELL, MARGIN + (row_from_top + 0.5) * CELL


def _hit_fraction(maskb: np.ndarray, xc: float, yc: float, along: str, span_frac: float = 0.6, perp_search: int = 6) -> float:
    """マス目方向(along)に±span_frac*CELL/2 だけ走査し、直交方向±perp_search
    の窓に1画素でも壁マスクの陽性があれば「的中」として、的中率を返す。"""
    half = int(round(span_frac * CELL / 2))
    xc, yc = int(round(xc)), int(round(yc))
    hits = total = 0
    if along == "h":
        for dx in range(-half, half + 1):
            x = xc + dx
            col = maskb[yc - perp_search : yc + perp_search + 1, x]
            total += 1
            hits += int(col.any())
    else:
        for dy in range(-half, half + 1):
            y = yc + dy
            row = maskb[y, xc - perp_search : xc + perp_search + 1]
            total += 1
            hits += int(row.any())
    return hits / total


def decode(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """画像1枚から (v_walls, h_walls) を0/1で読み取る。

    戻り値の3つ目は warp 後の画像（診断・目視検査用）。
    """
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"画像を読み込めない: {path}")
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float64)

    corners = find_corners(gray)
    warped = warp_to_grid(im, corners)
    wgray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(wgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, -5)
    maskb = mask > 0

    v = np.zeros((N + 1, N), dtype=np.uint8)
    h = np.zeros((N, N + 1), dtype=np.uint8)
    for x in range(N):
        for level in range(N + 1):
            xc, yc = _h_center(x, level)
            h[x, level] = 1 if _hit_fraction(maskb, xc, yc, "h") > FRAC_TH else 0
    for k in range(N + 1):
        for y in range(N):
            xc, yc = _v_center(k, y)
            v[k, y] = 1 if _hit_fraction(maskb, xc, yc, "v") > FRAC_TH else 0

    # 北東角セル(15,15)の北壁・東壁は、写真右上の黒い遮蔽物のため画素情報が
    # ほぼ得られない（東壁は下半分だけ部分的に見えるが的中率は閾値未満、
    # 北壁は全域が遮蔽されゼロ）。外周は競技規則上必ず閉じているという
    # 構造的な確実性に基づき、この1箇所だけ壁ありとする（他の外周215箇所は
    # すべて独立に画素から壁ありと読めている）。
    if h[15, 16] == 0:
        h[15, 16] = 1
    if v[16, 15] == 0:
        v[16, 15] = 1

    # 到達性検査で見つかった孤立ポケット(0,11)-(1,11)-(2,11)の是正。
    # この3マスは他と隔絶しており（西=外周・その他5面はすべて的中率0.53〜0.80と
    # 際どく、v(3,11)だけ1.0で確実に壁）、盤面としてありえない（全マス到達可能の
    # 大前提に反する）。的中率が最も低く閾値0.5に最も近いh(0,12)（0.537）を
    # 誤読と判断して開放にした——**コース長を見る前に、到達性検査だけを根拠に
    # 決めた選択**。h(0,11)/(1,11)/(1,12)/(2,11)/(2,12)のどれを開けても
    # 到達性は回復するため一意には決め切れないが、最も確信度が低い1枚を選んだ。
    if h[0, 12] == 1:
        h[0, 12] = 0

    return v, h, warped


# ============================================================================
# 3. 構造検査
# ============================================================================
def check_structure(v: np.ndarray, h: np.ndarray) -> Optional[str]:
    if not np.all(v[0, :] == 1):
        return "西の外周に穴"
    if not np.all(v[N, :] == 1):
        return "東の外周に穴"
    if not np.all(h[:, 0] == 1):
        return "南の外周に穴"
    if not np.all(h[:, N] == 1):
        return "北の外周に穴"

    goals = set(GOAL)
    for (x, y) in goals:
        for (nx, ny) in ((x + 1, y), (x, y + 1)):
            if (nx, ny) not in goals:
                continue
            state = v[x + 1, y] if nx == x + 1 else h[x, y + 1]
            if state != 0:
                return f"ゴール区画間 ({x},{y})-({nx},{ny}) に壁"

    from collections import deque

    def open_sides(x: int, y: int) -> List[Tuple[int, int]]:
        sides = []
        if v[x + 1, y] == 0:
            sides.append((x + 1, y))
        if v[x, y] == 0:
            sides.append((x - 1, y))
        if h[x, y + 1] == 0:
            sides.append((x, y + 1))
        if h[x, y] == 0:
            sides.append((x, y - 1))
        return sides

    if len(open_sides(*START)) != 1:
        return f"スタート区画の開口が{len(open_sides(*START))}個（1個のはず）"

    seen = {START}
    dq = deque([START])
    while dq:
        cell = dq.popleft()
        for n in open_sides(*cell):
            if n not in seen:
                seen.add(n)
                dq.append(n)
    if len(seen) != N * N:
        return f"到達可能な区画が{len(seen)}/{N * N}"
    return None


# ============================================================================
# 4. コース長の照合
# ============================================================================
def route_length(v: np.ndarray, h: np.ndarray, start: Tuple[int, int], heading: Direction) -> Tuple[int, int]:
    maze = MazeMap(N, N)
    maze.v_walls[:, :] = np.where(v == 1, int(WallState.WALL), int(WallState.OPEN))
    maze.h_walls[:, :] = np.where(h == 1, int(WallState.WALL), int(WallState.OPEN))
    dist = compute_flood(maze, GOAL, FloodMode.PESSIMISTIC)
    path = shortest_path(maze, start, GOAL, FloodMode.PESSIMISTIC)
    cmds = path_to_commands(path, start_heading=heading)
    turns = sum(1 for c in cmds if c.type in (CommandType.TURN_LEFT90, CommandType.TURN_RIGHT90, CommandType.TURN_180))
    return int(dist[start]), turns


def main() -> None:
    if len(sys.argv) < 2:
        print(f"使い方: {sys.argv[0]} <画像パス>")
        sys.exit(1)
    path = Path(sys.argv[1])
    v, h, warped = decode(path)

    print(f"壁の総数: 縦{int(v.sum())}枚 + 横{int(h.sum())}枚 = {int(v.sum() + h.sum())}枚")

    problem = check_structure(v, h)
    print(f"構造検査: {'OK' if problem is None else 'NG - ' + problem}")

    for start, heading, label in [(START, Direction.N, "(0,0)始点・北向き"), ((15, 15), Direction.W, "(15,15)始点・西向き")]:
        cells, turns = route_length(v, h, start, heading)
        print(f"{label}: {cells}区画・{turns}折  (記録表は 73区50折)")

    print("\n🔴 いずれも記録表の 73区50折 と一致しない。docstring の説明を参照。")


if __name__ == "__main__":
    main()
