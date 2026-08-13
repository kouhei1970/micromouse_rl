"""裁定 R34 — 1/ρ の面ごと回答表（現行マスク・R34 是正後マスクの 2 通り）。

盲検のため、本スクリプトと出力はリポジトリ外（セッション作業領域）に置き、
出力 JSON の SHA-256 だけを先行コミットする（COMMIT_001 方式）。
学生B の再測定値の提出後に、両方をリポジトリへコミットして開示・照合する。

2 通りのマスク:
  A. 現行マスク  = 実装の `Maze6Env._geo_allowed_mask()` をそのまま呼ぶ
  B. R34 是正後  = 物理モデル（mouse/mjcf.py が実際に生成する壁の箱＋柱）の
                   表面からの距離 ≥ w_lat = 0.0400 m（円板モデル）
どちらも**実装の格子・グラフ位相・Dijkstra**（`_compute_geodesic_field`）を使い、
**マスクだけを差し替える**。したがって B は「R34 を実装したときに出るはずの値」の予測である。

1/ρ = g(start) / (0.18 · D_0)。始点は**スタート区画の中心**（実装は擾乱つきの
reset 位置で取るので、面ごとの比較では ±の揺れが乗ることに注意）。
"""
import hashlib
import json
import sys
from types import SimpleNamespace

REPO_ROOT = "/Users/kouhei/tmp/github/micromouse_rl"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, f"{REPO_ROOT}/verification")

import numpy as np

from audit_r33_config_space import CS, MODE, SEEDS, W_LAT, erode_mask, obstacle_rects
from mouse.maze6_env import _GEO_GRID_H, _GEO_GRID_N, Maze6Env
from mouse.maze6_gen import generate_maze, shortest_distances

OUT = ("/private/tmp/claude-501/-Users-kouhei-tmp-github-micromouse-rl/"
       "0e53cd17-454e-4726-bd78-45bc2638dc76/scratchpad/r34_answers.json")


def field_with_mask(maze, mask=None):
    """実装の _compute_geodesic_field() を、マスクだけ差し替えて呼ぶ。"""
    shim = SimpleNamespace(maze=maze, params=SimpleNamespace(cell_size=CS))
    if mask is None:
        shim._geo_allowed_mask = lambda s=shim: Maze6Env._geo_allowed_mask(s)
    else:
        shim._geo_allowed_mask = lambda m=mask: m
    return Maze6Env._compute_geodesic_field(shim)


def main():
    h = _GEO_GRID_H
    xs = np.arange(_GEO_GRID_N) * h
    rows = []
    for seed in SEEDS:
        m = generate_maze(seed, mode=MODE)
        dmap = shortest_distances(m["v_walls"], m["h_walls"])
        start = tuple(int(v) for v in m["start"])
        d0 = int(dmap[start])
        i = int(round((start[0] + 0.5) * CS / h))
        j = int(round((start[1] + 0.5) * CS / h))

        f_cur = field_with_mask(m, None)
        rects = obstacle_rects(m["v_walls"], m["h_walls"], with_posts=True, full_length=False)
        mask_r34 = erode_mask(xs, xs, rects, W_LAT)
        f_r34 = field_with_mask(m, mask_r34)

        rows.append({
            "seed": int(seed),
            "start": list(start),
            "D0": d0,
            "g_start_current": round(float(f_cur[i, j]), 9),
            "inv_rho_current": round(float(f_cur[i, j]) / (CS * d0), 9),
            "g_start_r34": round(float(f_r34[i, j]), 9),
            "inv_rho_r34": round(float(f_r34[i, j]) / (CS * d0), 9),
            "n_allowed_current": int(Maze6Env._geo_allowed_mask(
                SimpleNamespace(maze=m, params=SimpleNamespace(cell_size=CS))).sum()),
            "n_allowed_r34": int(mask_r34.sum()),
        })
        print(f"seed {seed:>3}: D0={d0:>2}  1/rho 現行={rows[-1]['inv_rho_current']:.4f}  "
              f"R34={rows[-1]['inv_rho_r34']:.4f}")

    cur = [r["inv_rho_current"] for r in rows]
    r34 = [r["inv_rho_r34"] for r in rows]
    doc = {
        "purpose": "裁定 R34 の面ごと回答表（准教授の事前予測）。SHA-256 を先行コミットする",
        "band": "学習迷路 6x6 loop, seed 1-20",
        "grid_h_m": h, "grid_n": _GEO_GRID_N,
        "start_point": "スタート区画の中心（実装は擾乱つき reset 位置なので面ごとに揺れる）",
        "definition": "inv_rho = g(start) / (0.18 * D0)",
        "mask_A_current": "実装 Maze6Env._geo_allowed_mask() をそのまま",
        "mask_B_r34": "mouse/mjcf.py が生成する壁の箱＋柱の表面からの距離 >= 0.0400 m",
        "median_current": round(float(np.median(cur)), 9),
        "median_r34": round(float(np.median(r34)), 9),
        "rows": rows,
    }
    blob = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True).encode("utf-8")
    with open(OUT, "wb") as f:
        f.write(blob)
    print()
    print(f"中央値: 現行 {doc['median_current']:.4f}  R34 {doc['median_r34']:.4f}")
    print(f"出力: {OUT}")
    print(f"SHA-256: {hashlib.sha256(blob).hexdigest()}")
    print(f"バイト数: {len(blob)}")


if __name__ == "__main__":
    main()
