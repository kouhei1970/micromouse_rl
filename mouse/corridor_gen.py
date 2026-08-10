"""
mouse/corridor_gen.py
================
M1（廊下追従・exp_001_corridor）で使う「分岐のない自己回避パス」コース生成器。

表現方法（spec_m1_corridor.md §1 確定）: コースは格子上の自己回避パス
（self-avoiding path）として表す。壁配列はパス上の連続するセル間だけを開通させ、
それ以外は全て壁とする。これにより分岐が構造的に存在しえない
（非連続で隣接するセル間には壁が残るので抜け道もできない）。

生成手順は決定的（random.Random(seed) のみを使い、global random は使わない）。
MuJoCo XML は mouse.mjcf.build_maze_robot_xml で生成する（独自の XML 組立をしない。
凍結ロボット仕様からの派生物という憲章の規約を守るため）。
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from mouse.mjcf import build_maze_robot_xml
from mouse.params import RobotParams

# 方向インデックス: 0=N(+y) 1=E(+x) 2=S(-y) 3=W(-x)
_DIR_VECS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
_DIR_HEADING_DEG = [90.0, 0.0, 270.0, 180.0]

_MAX_RETRIES_PER_COUNT = 200
_MIN_SEGMENTS_FLOOR = 1
_MIN_GRID = 3
_MAX_GRID = 24
_STRAIGHT_LEN_RANGE = (2, 6)
_SCURVE_MID_LEN_RANGE = (1, 2)
# セグメント総数。教授指示（指示書#2 補足4）の「総セル数 10〜20 セル程度」に合わせて
# (6,12) から縮小した（(6,12) では中央値 27 セルと推奨の倍近くになり、
# 壁接触なし完走の gate が意図より難しくなるため）。(3,6) で中央値 15 セル・
# 74% が 10〜20 セルに収まる（残りは 5〜9 セルの短いコースと 21〜26 セルの長いコース）。
_N_SEGMENTS_RANGE = (3, 6)


# ==========================================================================
# パス構築（内部ヘルパー）
# ==========================================================================
def _turn(direction: int, way: str) -> int:
    """direction（0=N,1=E,2=S,3=W）を左/右に90°回した方向インデックスを返す。
    右回り(時計回り)が index+1、左回り(反時計回り)が index-1。"""
    if way == 'left':
        return (direction - 1) % 4
    if way == 'right':
        return (direction + 1) % 4
    raise ValueError(f"way は 'left'/'right': {way!r}")


def _try_build_path(rng: random.Random, n_segments: int):
    """1回分のパス構築試行。既訪問セルへの衝突、または最終外接矩形が
    最大格子サイズを超える場合は None を返す（呼び出し側で再試行する）。

    戻り値: (path, segment_types) または None。
      path: [(cx,cy), ...]（(0,0) 起点、オフセット前の生座標）
      segment_types: 各セグメントの種別（'straight'/'turn'/'s_curve'）のリスト
    """
    path = [(0, 0)]
    visited = {(0, 0)}
    direction = rng.randrange(4)
    segment_types = []

    def _step(d):
        cx, cy = path[-1]
        dx, dy = _DIR_VECS[d]
        nxt = (cx + dx, cy + dy)
        if nxt in visited:
            return False
        path.append(nxt)
        visited.add(nxt)
        return True

    for seg_idx in range(n_segments):
        # 最初は必ず直線セグメントから始める（初期姿勢の擾乱を吸収する助走）
        seg_type = 'straight' if seg_idx == 0 else rng.choice(['straight', 'turn', 's_curve'])
        segment_types.append(seg_type)

        if seg_type == 'straight':
            length = rng.randint(*_STRAIGHT_LEN_RANGE)
            for _ in range(length):
                if not _step(direction):
                    return None

        elif seg_type == 'turn':
            way = rng.choice(['left', 'right'])
            direction = _turn(direction, way)
            if not _step(direction):
                return None

        else:  # s_curve: 右→左 または 左→右、間に直線1〜2セル
            first, second = rng.choice([('right', 'left'), ('left', 'right')])
            direction = _turn(direction, first)
            if not _step(direction):
                return None
            length = rng.randint(*_SCURVE_MID_LEN_RANGE)
            for _ in range(length):
                if not _step(direction):
                    return None
            direction = _turn(direction, second)
            if not _step(direction):
                return None

    xs = [c[0] for c in path]
    ys = [c[1] for c in path]
    bbox_w = max(xs) - min(xs) + 1
    bbox_h = max(ys) - min(ys) + 1
    if bbox_w + 2 > _MAX_GRID or bbox_h + 2 > _MAX_GRID:
        return None

    return path, segment_types


def _build_wall_arrays(path, width, height, offset):
    """path（オフセット前の生座標）から v_walls/h_walls を作る。offset 分だけ
    平行移動した座標列 offset_path も返す。

    壁のインデックス規約は competition/evaluator.py の _cell_connects と同一
    （v_walls[x,y] shape (W+1,H): セル(x-1,y)-(x,y)間、h_walls[x,y] shape
    (W,H+1): セル(x,y-1)-(x,y)間）。"""
    v_walls = np.ones((width + 1, height), dtype=int)
    h_walls = np.ones((width, height + 1), dtype=int)
    ox, oy = offset
    offset_path = [(x + ox, y + oy) for (x, y) in path]

    for (x0, y0), (x1, y1) in zip(offset_path, offset_path[1:]):
        dx, dy = x1 - x0, y1 - y0
        if dx == 1 and dy == 0:
            v_walls[x1, y0] = 0
        elif dx == -1 and dy == 0:
            v_walls[x0, y0] = 0
        elif dy == 1 and dx == 0:
            h_walls[x0, y1] = 0
        elif dy == -1 and dx == 0:
            h_walls[x0, y0] = 0
        else:
            raise AssertionError(f"非隣接な連続セル: {(x0, y0)} -> {(x1, y1)}")

    return v_walls, h_walls, offset_path


def _count_turns(path):
    """パス中間セルのうち、進入方向と退出方向が異なるセルの数を返す。"""
    n_turns = 0
    for i in range(1, len(path) - 1):
        d_in = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        d_out = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if d_in != d_out:
            n_turns += 1
    return n_turns


def _validate_course(course: dict) -> None:
    """spec_m1_corridor.md §1 手順5の生成器内 assert:
      (a) パスに重複セルがない
      (b) パス上の各セルの開通方向数が端点で1・中間で2（分岐なし・行き止まりなし）
      (c) 外周が全て壁
    """
    v_walls = course['v_walls']
    h_walls = course['h_walls']
    path = course['path']
    width = course['width']
    height = course['height']

    assert len(set(path)) == len(path), f"パスに重複セルがある: seed={course['seed']}"

    for i, (cx, cy) in enumerate(path):
        open_count = 0
        if cx + 1 <= width and v_walls[cx + 1, cy] == 0:
            open_count += 1
        if cx - 1 >= 0 and v_walls[cx, cy] == 0:
            open_count += 1
        if cy + 1 <= height and h_walls[cx, cy + 1] == 0:
            open_count += 1
        if cy - 1 >= 0 and h_walls[cx, cy] == 0:
            open_count += 1
        expected = 1 if (i == 0 or i == len(path) - 1) else 2
        assert open_count == expected, (
            f"seed={course['seed']}: セル{(cx, cy)}(index{i})の開通数={open_count}, "
            f"期待={expected}"
        )

    assert np.all(v_walls[0, :] == 1) and np.all(v_walls[width, :] == 1), (
        f"seed={course['seed']}: 縦方向外周に壁がない")
    assert np.all(h_walls[:, 0] == 1) and np.all(h_walls[:, height] == 1), (
        f"seed={course['seed']}: 横方向外周に壁がない")


# ==========================================================================
# 公開 API
# ==========================================================================
def generate_course(seed: int) -> dict:
    """seed から決定的に「分岐のない自己回避パス」コースを1本生成する。

    返り値 dict:
      v_walls: (W+1, H) int, h_walls: (W, H+1) int   # 1=壁あり
      path:    [(cx,cy), ...]  スタートセルからゴールセルまでのセル列
      width, height, seed, n_cells, n_turns
      segment_types: 生成に使われたセグメント種別列（'straight'/'turn'/'s_curve'）
                      （観測用の追加情報。spec の必須フィールドではない）
    """
    rng = random.Random(seed)
    n_segments = rng.randint(*_N_SEGMENTS_RANGE)

    result = None
    cur_n_segments = n_segments
    while cur_n_segments >= _MIN_SEGMENTS_FLOOR:
        for _ in range(_MAX_RETRIES_PER_COUNT):
            candidate = _try_build_path(rng, cur_n_segments)
            if candidate is not None:
                result = candidate
                break
        if result is not None:
            break
        cur_n_segments -= 1

    if result is None:
        raise RuntimeError(f"generate_course(seed={seed}): パス構築に失敗しました")

    path, segment_types = result

    xs = [c[0] for c in path]
    ys = [c[1] for c in path]
    bbox_w = max(xs) - min(xs) + 1
    bbox_h = max(ys) - min(ys) + 1
    # _try_build_path 内で bbox_w+2<=MAX_GRID を検証済みのため、ここでの clip は
    # 実質 MIN_GRID 側の安全弁のみ（最短パスでも bbox+2>=3 になるため理論上は無発火）。
    width = min(max(bbox_w + 2, _MIN_GRID), _MAX_GRID)
    height = min(max(bbox_h + 2, _MIN_GRID), _MAX_GRID)

    offset_x = 1 - min(xs)
    offset_y = 1 - min(ys)
    v_walls, h_walls, offset_path = _build_wall_arrays(path, width, height, (offset_x, offset_y))

    n_turns = _count_turns(offset_path)

    course = dict(
        v_walls=v_walls, h_walls=h_walls, path=offset_path,
        width=width, height=height, seed=seed,
        n_cells=len(offset_path), n_turns=n_turns,
        segment_types=segment_types,
    )
    _validate_course(course)
    return course


def initial_heading_deg(path) -> float:
    """path[0] から path[1] へ向かう方位 [deg]（mouse_euler の z 成分と同じ規約:
    90=+y(北), 0=+x(東), 270=-y(南), 180=-x(西)）を返す。"""
    (x0, y0), (x1, y1) = path[0], path[1]
    dx, dy = x1 - x0, y1 - y0
    idx = _DIR_VECS.index((dx, dy))
    return _DIR_HEADING_DEG[idx]


def path_cell_centers(path, cell_size: float):
    """パスのセル中心座標列 [(x,y), ...] を返す（メートル）。"""
    return [((cx + 0.5) * cell_size, (cy + 0.5) * cell_size) for cx, cy in path]


def path_cumulative_lengths(centers):
    """centers[i] までの累積弧長 [m] のリスト（cum[0]=0、cum[-1]=総経路長）。"""
    cum = [0.0]
    for (x0, y0), (x1, y1) in zip(centers, centers[1:]):
        cum.append(cum[-1] + math.hypot(x1 - x0, y1 - y0))
    return cum


def remaining_path_length(centers, cum_lengths, x: float, y: float) -> float:
    """現在位置 (x,y) から終点までの残り経路長 [m] を返す。

    パスのセル中心を結んだ折れ線に (x,y) を射影し（各線分への垂線の足、
    ただし線分の外には出ない = t を [0,1] にクランプ）、最も近い線分への
    射影点から終点までの弧長を返す（spec_m1_corridor.md §2 の Φ(s) の定義）。
    """
    total = cum_lengths[-1]
    best_remaining = None
    best_dist_perp = None
    for i in range(len(centers) - 1):
        x0, y0 = centers[i]
        x1, y1 = centers[i + 1]
        seg_dx, seg_dy = x1 - x0, y1 - y0
        seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
        if seg_len_sq < 1e-12:
            t = 0.0
        else:
            t = ((x - x0) * seg_dx + (y - y0) * seg_dy) / seg_len_sq
            t = min(max(t, 0.0), 1.0)
        proj_x = x0 + t * seg_dx
        proj_y = y0 + t * seg_dy
        dist_perp = math.hypot(x - proj_x, y - proj_y)
        arc = cum_lengths[i] + t * math.hypot(seg_dx, seg_dy)
        remaining = total - arc
        if best_dist_perp is None or dist_perp < best_dist_perp:
            best_dist_perp = dist_perp
            best_remaining = remaining
    return best_remaining


def lateral_deviation(centers, x: float, y: float) -> float:
    """現在位置 (x,y) の、経路中心線からの横偏差（垂直距離）[m] を返す。

    経路のセル中心を結んだ折れ線に (x,y) を射影し、最も近い線分への垂線の長さを
    返す（線分の外へは出ないよう t を [0,1] にクランプする）。符号は持たせない。

    用途: exp_006 以降の評価指標（2026-08-10 教授指示）。行動を滑らかにしたぶん
    機体が壁へ寄っていないかを見る。区画は 180 mm、機体幅は 72 mm なので、
    中心線から片側 54 mm で壁に接触する計算になる（壁厚 12 mm を差し引くと 48 mm）。

    注意: competition/baseline_*.py の kp_lateral が使う「横偏差」は制御用の
    符号つき誤差であり、こちらは評価用の絶対値。定義が異なる。
    """
    best = None
    for i in range(len(centers) - 1):
        x0, y0 = centers[i]
        x1, y1 = centers[i + 1]
        seg_dx, seg_dy = x1 - x0, y1 - y0
        seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
        if seg_len_sq < 1e-12:
            t = 0.0
        else:
            t = ((x - x0) * seg_dx + (y - y0) * seg_dy) / seg_len_sq
            t = min(max(t, 0.0), 1.0)
        d = math.hypot(x - (x0 + t * seg_dx), y - (y0 + t * seg_dy))
        if best is None or d < best:
            best = d
    return best


def save_course(course: dict, out_dir, params: RobotParams = None):
    """course dict を npz + XML として out_dir に保存する。
    戻り値: (npz_path, xml_path)"""
    params = params or RobotParams()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = course['seed']
    stem = f"corridor_{seed}"
    npz_path = out_dir / f"{stem}.npz"
    xml_path = out_dir / f"{stem}.xml"

    path_arr = np.array(course['path'], dtype=np.int64)
    np.savez(
        npz_path,
        v_walls=course['v_walls'], h_walls=course['h_walls'],
        path=path_arr, width=course['width'], height=course['height'],
        seed=seed, n_cells=course['n_cells'], n_turns=course['n_turns'],
    )

    cell_size = params.cell_size
    sx, sy = course['path'][0]
    mouse_pos = f"{sx * cell_size + cell_size / 2} {sy * cell_size + cell_size / 2} 0.002"
    heading = initial_heading_deg(course['path'])
    mouse_euler = f"0 0 {heading}"

    build_maze_robot_xml(
        course['v_walls'], course['h_walls'], str(xml_path),
        model_name=f"corridor_{seed}", mouse_pos=mouse_pos, mouse_euler=mouse_euler,
        center_goal=False, params=params,
    )
    return npz_path, xml_path


# ==========================================================================
# CLI
# ==========================================================================
def _parse_seed_range(spec: str):
    start, _, end = spec.partition('-')
    return list(range(int(start), int(end) + 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description="M1 廊下コース生成器（CLI）")
    # 研究計画書 §9-7（データ三分割）: train=学習用（毎エピソード生成のため通常は
    # 事前生成不要）、validation=日常の実験判断用（seed 5000-5019）、
    # eval=M1 gate 判定専用（seed 3000-3019。これを見て設定を調整しない）
    parser.add_argument('--split', choices=['train', 'eval', 'validation'], required=True)
    parser.add_argument('--seeds', type=str, required=True, help='例: 2000-2199')
    parser.add_argument('--out-root', type=str, default='assets/corridor')
    args = parser.parse_args(argv)

    seeds = _parse_seed_range(args.seeds)
    out_dir = Path(args.out_root) / args.split

    records = []
    for seed in seeds:
        course = generate_course(seed)
        npz_path, xml_path = save_course(course, out_dir)
        records.append(dict(
            seed=seed, n_cells=course['n_cells'], n_turns=course['n_turns'],
            width=course['width'], height=course['height'],
            npz=str(npz_path), xml=str(xml_path),
        ))
        print(f"[corridor_gen] split={args.split} seed={seed} "
              f"n_cells={course['n_cells']} n_turns={course['n_turns']} "
              f"grid={course['width']}x{course['height']}")

    manifest = dict(
        split=args.split, seeds=seeds, count=len(seeds),
        generator='mouse/corridor_gen.py: generate_course',
        courses=records,
    )
    manifest_path = out_dir / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[corridor_gen] saved {manifest_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
