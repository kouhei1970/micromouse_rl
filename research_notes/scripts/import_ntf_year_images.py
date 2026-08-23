"""NTF の年次ページ・taikai ページの画像／HTML から、抜けている年
（1999〜2003・2013〜2015・2022〜2025）の競技迷路を読み取り、
`mazes/contest/` へ `.maze` として書き出す。

前提となる読み方（実測して確かめたもの）:

1. **1999 年（第20回）**: `taikai/20-exp01.html` 等の `<pre>` 内に、全角文字
   （柱 `・`・横壁 `－`・縦壁 `｜`・開通は全角空白）でアスキー迷路が直接
   埋め込まれている。柱行に `GOAL` の文字列が割り込むことがあり、これは
   ゴール2×2の中央柱を隠すもの（柱1本ぶん＝2ステップぶん、両側とも開通）
   として扱う。`parse_html_ascii_maze()`。

2. **2000 年（第21回）・2003 年（第24回）**: NTF の `taikai/` 直下に
   258×258 の GIF（黒背景・赤壁、格子間隔16画素・壁太さ2画素）が個別に
   置かれている（`21-exp01.gif` 等）。`decode_red_bitmap()`
   （`decode_ntf_maze_bitmaps.py` と同じ考え方、pitch/thickness を自動推定）。

3. **2001 年（第22回）・2002 年（第23回）**: 同じく `taikai/` 直下に
   258×258 の GIF が個別に存在する（`22-exp01.gif` 等）ことが分かったので、
   2000・2003 と同じ手法で読み取れる。**別途、年次ページに置かれていた
   ベクター調の合成図**（`2303.jpg`＝1998〜2000年の参考図、
   `24-cousemaze02.jpg`＝2002年の図）からも独立に読み取り、両者を
   壁単位で突き合わせて裏取りした（`decode_vector_crop.py` 系）。

4. **2013〜2015・2022〜2025 年**: NTF 年次ページの PNG は、軸に 0〜15 の
   ラベルが振られた綺麗な図（外周・壁が太い実線、区画境界が細い点線）。
   `decode_labeled_png()`。**この関数は 2013 年決勝で正解
   （kerikun11 由来の既存面）と壁単位で完全一致することを確認済み**。

いずれの画像も、**始点開口・ゴール2×2内壁なしの構造検査**と、
**最短経路（区画数・折数）を一次資料の記載値と突き合わせる検算**を行った上で
`mazes/contest/` へ書き出す。詳細は `research_notes/note_037_*.md`
（または該当セッションの報告）を参照。
"""
from __future__ import annotations

import hashlib
import re
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from common import maze_db  # noqa: E402


# ============================================================================
# 1. HTML 全角アスキー（1999年）
# ============================================================================
ROW_LABELS_FULL = "０１２３４５６７８９ＡＢＣＤＥＦ"


def _extract_pre_body(html_path: Path) -> str:
    html = html_path.read_text(encoding="utf-8-sig")
    idx = html.find("<pre>")
    if idx < 0:
        raise ValueError(f"<pre> が無い: {html_path}")
    body = html[idx + len("<pre>"):]
    end = body.find("<hr>")
    if end >= 0:
        body = body[:end]
    return body.replace("&nbsp;", "　").replace("&amp;", "&")


def _find_pillar_rows(lines: List[str], N: int) -> List[int]:
    cand = []
    for i, l in enumerate(lines):
        stripped = l.strip("　")
        n_pillar = stripped.count("・")
        if n_pillar >= N - 2 and set(stripped) <= set("・－　GOALgoal "):
            cand.append(i)
    return cand


def _parse_pillar_row(line: str, N: int) -> List[int]:
    parts = line.split("・")
    gaps = parts[1:-1] if len(parts) >= 2 else []
    states: List[int] = []
    x = 0
    gi = 0
    while x < N:
        if gi >= len(gaps):
            raise ValueError(f"柱行の解析が壁数に届かない: {line!r}")
        gap = gaps[gi]
        gi += 1
        if gap == "－":
            states.append(1)
            x += 1
        elif gap == "　" or gap.strip("　 ") == "":
            states.append(0)
            x += 1
        else:
            # GOAL 等の割り込み: 中央の柱1本ぶんを隠す＝2ステップぶん両方 open
            states.append(0)
            states.append(0)
            x += 2
    return states[:N]


def _find_cell_rows(lines: List[str], pillar_idxs: List[int], N: int) -> Dict[int, str]:
    lens = [len(lines[i]) for i in pillar_idxs if "GOAL" not in lines[i].upper()]
    expected_len = max(set(lens), key=lens.count)
    rows: Dict[int, str] = {}
    for i, l in enumerate(lines):
        if len(l) != expected_len:
            continue
        s = l.strip("　")
        if not s or len(s) < 2:
            continue
        label = s[0]
        if label in ROW_LABELS_FULL and s[1] in "｜　":
            y = ROW_LABELS_FULL.index(label)
            if y in rows:
                raise ValueError(f"行ラベル {label!r} が重複した")
            rows[y] = l
    return rows


def _parse_cell_row(line: str, N: int) -> Tuple[List[int], Optional[int]]:
    idx = 0
    while line[idx] == "　":
        idx += 1
    idx += 1
    rest = line[idx:]
    v_states = []
    start_x = None
    x = 0
    pos = 0
    while x <= N:
        ch = rest[pos]
        if ch == "｜":
            v_states.append(1)
        elif ch == "　":
            v_states.append(0)
        else:
            raise ValueError(f"縦壁位置に想定外の文字: {ch!r}")
        pos += 1
        if x < N:
            if rest[pos] == "↑":
                start_x = x
            pos += 1
        x += 1
    return v_states, start_x


def parse_html_ascii_maze(html_path: Path, N: int = 16) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """NTF taikai の全角アスキー迷路ページを読み取る。戻り値: (v_walls, h_walls, start)。"""
    body = _extract_pre_body(html_path)
    lines = body.split("\n")
    pillar_idxs = _find_pillar_rows(lines, N)
    if len(pillar_idxs) != N + 1:
        raise ValueError(f"柱行が{N+1}本見つからない（実際{len(pillar_idxs)}）: {html_path}")
    cell_rows = _find_cell_rows(lines, pillar_idxs, N)
    if len(cell_rows) != N:
        raise ValueError(f"区画行が{N}本見つからない（実際{len(cell_rows)}）: {html_path}")

    h = np.zeros((N, N + 1), dtype=np.uint8)
    for k, li in enumerate(pillar_idxs):
        level = N - k
        states = _parse_pillar_row(lines[li], N)
        for x in range(N):
            h[x, level] = states[x]

    v = np.zeros((N + 1, N), dtype=np.uint8)
    start = None
    for y, line in cell_rows.items():
        v_states, start_x = _parse_cell_row(line, N)
        for x in range(N + 1):
            v[x, y] = v_states[x]
        if start_x is not None:
            start = (start_x, y)
    return v, h, start


# ============================================================================
# 2. 赤/黒 2値ビットマップ（GIF・BMP 共通。decode_ntf_maze_bitmaps.py と同じ考え方）
# ============================================================================
def _detect_pitch(size: int, N: int = 16, thickness_candidates=(2, 3, 4)) -> Tuple[int, int]:
    for T in thickness_candidates:
        if (size - T) % N == 0:
            return (size - T) // N, T
    raise ValueError(f"pitch/thickness を特定できない: size={size}")


def decode_red_bitmap(path: Path, N: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """黒背景・赤壁の2値ビットマップ（NTF の GIF/BMP 共通書式）を読み取る。"""
    a = np.array(Image.open(path).convert("RGB"))
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"正方形でない: {a.shape}")
    size = a.shape[0]
    P, T = _detect_pitch(size, N)
    red = (a[:, :, 0] > 128) & (a[:, :, 1] < 128) & (a[:, :, 2] < 128)

    v = np.zeros((N + 1, N), dtype=np.uint8)
    h = np.zeros((N, N + 1), dtype=np.uint8)
    for y in range(N):
        r_mid = P * (N - 1 - y) + P // 2
        for x in range(N + 1):
            v[x, y] = red[r_mid, P * x]
    for x in range(N):
        c_mid = P * x + P // 2
        for level in range(N + 1):
            h[x, level] = red[P * (N - level), c_mid]
    return v, h


# ============================================================================
# 3. 軸ラベル付きの綺麗な PNG（2013〜2015・2022〜2025）
#    外周・壁=太い実線、区画境界=細い点線。2013年決勝で正解と完全一致を確認済み。
# ============================================================================
def _load_binary(path: Path, dark_thresh: int = 150) -> np.ndarray:
    a = np.array(Image.open(path).convert("L"))
    return a < dark_thresh


def _find_border_groups(sums: np.ndarray, total: float, frac: float = 0.5) -> List[List[int]]:
    thresh = total * frac
    idx = [i for i, val in enumerate(sums) if val > thresh]
    if not idx:
        raise ValueError("外周が検出できない")
    groups: List[List[int]] = []
    cur = [idx[0]]
    for x in idx[1:]:
        if x - cur[-1] <= 3:
            cur.append(x)
        else:
            groups.append(cur)
            cur = [x]
    groups.append(cur)
    return groups


def _detect_grid(dark: np.ndarray, N: int = 16):
    H, W = dark.shape
    rowsum = dark.sum(axis=1)
    colsum = dark.sum(axis=0)
    row_groups = _find_border_groups(rowsum, W, frac=0.5)
    top = (row_groups[0][0] + row_groups[0][-1]) / 2
    bot = (row_groups[-1][0] + row_groups[-1][-1]) / 2

    sub = dark[int(top):int(bot) + 1, :]
    colsum2 = sub.sum(axis=0)
    col_groups = _find_border_groups(colsum2, bot - top, frac=0.5)
    left = (col_groups[0][0] + col_groups[0][-1]) / 2
    right = (col_groups[-1][0] + col_groups[-1][-1]) / 2

    ys_img = [top + (bot - top) * k / N for k in range(N + 1)]  # 上から k 番目（北→南）
    xs = [left + (right - left) * k / N for k in range(N + 1)]
    return xs, ys_img


def _longest_run(bool_arr) -> int:
    best = cur = 0
    for b in bool_arr:
        if b:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def decode_labeled_png(path: Path, N: int = 16, margin: float = 0.30, run_thresh_frac: float = 0.5):
    """点線グリッド付きの軸ラベル PNG を読み取る。戻り値: (v_walls, h_walls)。"""
    dark = _load_binary(path)
    xs, ys_img = _detect_grid(dark, N)
    v = np.zeros((N + 1, N), dtype=np.uint8)
    h = np.zeros((N, N + 1), dtype=np.uint8)

    for x in range(N + 1):
        c = int(round(xs[x]))
        for y in range(N):
            y0, y1 = ys_img[N - 1 - y], ys_img[N - y]
            hgt = y1 - y0
            r0 = int(round(y0 + hgt * margin))
            r1 = int(round(y1 - hgt * margin))
            col = dark[r0:r1 + 1, c]
            run = _longest_run(col)
            v[x, y] = 1 if run / (r1 - r0 + 1) > run_thresh_frac else 0

    for x in range(N):
        x0, x1 = xs[x], xs[x + 1]
        wid = x1 - x0
        c0 = int(round(x0 + wid * margin))
        c1 = int(round(x1 - wid * margin))
        for level in range(N + 1):
            r = int(round(ys_img[N - level]))
            row = dark[r, c0:c1 + 1]
            run = _longest_run(row)
            h[x, level] = 1 if run / (c1 - c0 + 1) > run_thresh_frac else 0

    return v, h


# ============================================================================
# 4. ベクター調コンポジート画像（2001・2002 の裏取り用。既知の bbox を渡す）
# ============================================================================
def decode_vector_crop_run(dark: np.ndarray, bbox, N: int = 16, margin: float = 0.12, run_thresh_frac: float = 0.75):
    """輪郭線のみ（点線グリッド無し）の画像から、既知 bbox で1枚読み取る（run長方式）。"""
    x0, y0, x1, y1 = bbox
    xs = [x0 + (x1 - x0) * k / N for k in range(N + 1)]
    ys_img = [y0 + (y1 - y0) * k / N for k in range(N + 1)]
    v = np.zeros((N + 1, N), dtype=np.uint8)
    h = np.zeros((N, N + 1), dtype=np.uint8)
    for x in range(N + 1):
        c = int(round(xs[x]))
        for y in range(N):
            yy0, yy1 = ys_img[N - 1 - y], ys_img[N - y]
            hgt = yy1 - yy0
            r0 = int(round(yy0 + hgt * margin))
            r1 = max(int(round(yy1 - hgt * margin)), r0)
            col = dark[r0:r1 + 1, max(c - 1, 0):c + 2]
            run = max((_longest_run(col[:, k]) for k in range(col.shape[1])), default=0)
            v[x, y] = 1 if run / max(r1 - r0 + 1, 1) > run_thresh_frac else 0
    for x in range(N):
        xx0, xx1 = xs[x], xs[x + 1]
        wid = xx1 - xx0
        c0 = int(round(xx0 + wid * margin))
        c1 = max(int(round(xx1 - wid * margin)), c0)
        for level in range(N + 1):
            r = int(round(ys_img[N - level]))
            row = dark[max(r - 1, 0):r + 2, c0:c1 + 1]
            run = max((_longest_run(row[k, :]) for k in range(row.shape[0])), default=0)
            h[x, level] = 1 if run / max(c1 - c0 + 1, 1) > run_thresh_frac else 0
    return v, h


def decode_vector_crop_density(dark: np.ndarray, bbox, N: int = 16, margin: float = 0.15, wall_thresh_frac: float = 0.2):
    """破線（ダッシュ）で壁を描く画像向け（密度方式）。24-cousemaze02.jpg で使用。"""
    x0, y0, x1, y1 = bbox
    xs = [x0 + (x1 - x0) * k / N for k in range(N + 1)]
    ys_img = [y0 + (y1 - y0) * k / N for k in range(N + 1)]
    v = np.zeros((N + 1, N), dtype=np.uint8)
    h = np.zeros((N, N + 1), dtype=np.uint8)
    for x in range(N + 1):
        c = int(round(xs[x]))
        for y in range(N):
            yy0, yy1 = ys_img[N - 1 - y], ys_img[N - y]
            hgt = yy1 - yy0
            r0 = int(round(yy0 + hgt * margin))
            r1 = max(int(round(yy1 - hgt * margin)), r0)
            window = dark[r0:r1 + 1, max(c - 1, 0):c + 2]
            v[x, y] = 1 if (window.mean() if window.size else 0.0) > wall_thresh_frac else 0
    for x in range(N):
        xx0, xx1 = xs[x], xs[x + 1]
        wid = xx1 - xx0
        c0 = int(round(xx0 + wid * margin))
        c1 = max(int(round(xx1 - wid * margin)), c0)
        for level in range(N + 1):
            r = int(round(ys_img[N - level]))
            window = dark[max(r - 1, 0):r + 2, c0:c1 + 1]
            h[x, level] = 1 if (window.mean() if window.size else 0.0) > wall_thresh_frac else 0
    return v, h


# ============================================================================
# 5. 構造検査・コース長（区画数・折数）
# ============================================================================
def check_structure(v, h, N=16, start=(0, 0), goal_cells=None):
    problems = []
    if goal_cells is None:
        goal_cells = [(7, 7), (8, 7), (7, 8), (8, 8)]
    if not np.all(v[0, :] == 1):
        problems.append("西端の外周に開口がある")
    if not np.all(v[N, :] == 1):
        problems.append("東端の外周に開口がある")
    if not np.all(h[:, 0] == 1):
        problems.append("南端の外周に開口がある")
    if not np.all(h[:, N] == 1):
        problems.append("北端の外周に開口がある")

    gx = [c[0] for c in goal_cells]
    gy = [c[1] for c in goal_cells]
    x0, x1 = min(gx), max(gx)
    y0, y1 = min(gy), max(gy)
    if v[x1, y0] != 0:
        problems.append(f"ゴール内側の縦壁が残っている: v[{x1},{y0}]")
    if v[x1, y1] != 0:
        problems.append(f"ゴール内側の縦壁が残っている: v[{x1},{y1}]")
    if h[x0, y1] != 0:
        problems.append(f"ゴール内側の横壁が残っている: h[{x0},{y1}]")
    if h[x1, y1] != 0:
        problems.append(f"ゴール内側の横壁が残っている: h[{x1},{y1}]")

    sx, sy = start
    openings = 0
    if v[sx, sy] == 0:
        openings += 1
    if v[sx + 1, sy] == 0:
        openings += 1
    if h[sx, sy] == 0:
        openings += 1
    if h[sx, sy + 1] == 0:
        openings += 1
    if openings != 1:
        problems.append(f"スタート区画の開口が{openings}個（1個であるべき）")

    reach = _reachable_mask(v, h, N, start)
    if not all(reach[c] for c in goal_cells):
        problems.append("スタートからゴールへ到達できない")
    return problems, int(reach.sum())


def _reachable_mask(v, h, N, start):
    visited = np.zeros((N, N), dtype=bool)
    q = deque([start])
    visited[start] = True
    while q:
        x, y = q.popleft()
        if x + 1 < N and v[x + 1, y] == 0 and not visited[x + 1, y]:
            visited[x + 1, y] = True
            q.append((x + 1, y))
        if x - 1 >= 0 and v[x, y] == 0 and not visited[x - 1, y]:
            visited[x - 1, y] = True
            q.append((x - 1, y))
        if y + 1 < N and h[x, y + 1] == 0 and not visited[x, y + 1]:
            visited[x, y + 1] = True
            q.append((x, y + 1))
        if y - 1 >= 0 and h[x, y] == 0 and not visited[x, y - 1]:
            visited[x, y - 1] = True
            q.append((x, y - 1))
    return visited


def _neighbors(v, h, N, x, y):
    out = []
    if x + 1 < N and v[x + 1, y] == 0:
        out.append(("E", x + 1, y))
    if x - 1 >= 0 and v[x, y] == 0:
        out.append(("W", x - 1, y))
    if y + 1 < N and h[x, y + 1] == 0:
        out.append(("N", x, y + 1))
    if y - 1 >= 0 and h[x, y] == 0:
        out.append(("S", x, y - 1))
    return out


def shortest_path_to_goals(v, h, N, start, goal_cells, start_heading="N"):
    """スタートから各ゴール区画への最短(区画数, 折数)を求める（辞書式最小）。

    ゴール区画に着いたら打ち切る（ゴール内部を通り抜けるルートは数えない）。
    """
    import heapq

    goal_set = set(goal_cells)
    start_state = (start[0], start[1], start_heading)
    dist = {start_state: (0, 0)}
    pq = [(0, 0, start_state)]
    best_to_goal: Dict[Tuple[int, int], Tuple[int, int]] = {}
    while pq:
        steps, turns, (x, y, heading) = heapq.heappop(pq)
        if (steps, turns) != dist.get((x, y, heading), (None, None)):
            continue
        if (x, y) in goal_set:
            if (x, y) not in best_to_goal or (steps, turns) < best_to_goal[(x, y)]:
                best_to_goal[(x, y)] = (steps, turns)
            continue
        for d, nx, ny in _neighbors(v, h, N, x, y):
            nsteps = steps + 1
            nturns = turns + (0 if d == heading else 1)
            nstate = (nx, ny, d)
            if nstate not in dist or (nsteps, nturns) < dist[nstate]:
                dist[nstate] = (nsteps, nturns)
                heapq.heappush(pq, (nsteps, nturns, nstate))
    return best_to_goal


# ============================================================================
# 6. .maze への書き出し
# ============================================================================
def to_wallstate(arr01: np.ndarray) -> np.ndarray:
    return np.where(arr01 == 1, maze_db.WallState.WALL, maze_db.WallState.OPEN)


def build_record(*, maze_id, edition, year, cls, stage, source_type, source, source_url,
                  v01, h01, notes=None, start=(0, 0), start_heading="N",
                  goal=((7, 7), (8, 7), (7, 8), (8, 8))) -> maze_db.MazeRecord:
    v_state = to_wallstate(v01)
    h_state = to_wallstate(h01)
    sha = maze_db.compute_content_sha256(16, 16, v_state, h_state)
    return maze_db.MazeRecord(
        id=maze_id, width=16, height=16, start=start, start_heading=start_heading,
        goal=list(goal), series="AllJapan", edition=edition, year=year,
        maze_class=cls, stage=stage, source_type=source_type, source=source,
        source_url=source_url, retrieved="2026-08-23", confidence="single-source",
        content_sha256=sha, v_walls=v_state, h_walls=h_state, notes=notes,
    )


def write_and_verify(rec: maze_db.MazeRecord, out_dir: Path) -> Path:
    path = out_dir / f"{rec.id}.maze"
    text = maze_db.dumps(rec)
    path.write_text(text, encoding="utf-8")
    # 往復検査: 書いたものを読み直して、直列化結果が1文字単位で一致すること
    reloaded = maze_db.load(path)
    text2 = maze_db.dumps(reloaded)
    if text != text2:
        raise AssertionError(f"{rec.id}: 往復検査に失敗した")
    return path


if __name__ == "__main__":
    print(__doc__)
    print("この場ではメイン処理は実行しない。作業ログはセッションの報告を参照。")
