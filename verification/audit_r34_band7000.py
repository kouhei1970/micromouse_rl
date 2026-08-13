"""裁定 R34 の照合 — 学生B の提出帯（seed 7000〜7019）で同じ計算を回す。

`COMMIT_002` で先行コミットした回答表は**学習迷路 seed 1〜20**（`AUDIT_011` §5-quinquies §5 と
同じ帯）だが、学生B の提出は **seed 7000〜7019** だった。**帯が違うので面ごとの直接比較が
できない。**本スクリプトは提出帯で同じ計算をやり直し、差が帯によるものかを判別する。

`COMMIT_002` の 2 ファイル（`r34_answers.json` / `predict_r34_inv_rho.py`）は
**ハッシュを先行コミットしてあるので改変しない。**本スクリプトは別ファイルとして追加する。

⚠️ **本スクリプトの実行は、学生B の集計値（中央値 0.6923・範囲 0.5012〜0.7847）を
受け取った後に行った。**したがって**集計値の水準については盲検ではない。**
面ごとの値は受け取っていない（受け取ったのは表テキストの SHA-256 のみ）。
"""
import json
import sys
from types import SimpleNamespace

REPO_ROOT = "/Users/kouhei/tmp/github/micromouse_rl"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, f"{REPO_ROOT}/verification")

import numpy as np

from audit_r33_config_space import CS, MODE, W_LAT, erode_mask, obstacle_rects
from mouse.maze6_env import _GEO_GRID_H, _GEO_GRID_N, Maze6Env
from mouse.maze6_gen import generate_maze, shortest_distances

BAND = list(range(7000, 7020))


def field_with_mask(maze, mask=None):
    """実装の `_compute_geodesic_field()` を、マスクだけ差し替えて呼ぶ。"""
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
    for seed in BAND:
        m = generate_maze(seed, mode=MODE)
        dmap = shortest_distances(m["v_walls"], m["h_walls"])
        start = tuple(int(v) for v in m["start"])
        d0 = int(dmap[start])
        i = int(round((start[0] + 0.5) * CS / h))
        j = int(round((start[1] + 0.5) * CS / h))
        f_cur = field_with_mask(m, None)
        mask_r34 = erode_mask(xs, xs,
                              obstacle_rects(m["v_walls"], m["h_walls"], True, False), W_LAT)
        f_r34 = field_with_mask(m, mask_r34)
        rows.append({
            "seed": seed, "start": list(start), "D0": d0,
            "g_start_current": round(float(f_cur[i, j]), 9),
            "inv_rho_current": round(float(f_cur[i, j]) / (CS * d0), 9),
            "g_start_r34": round(float(f_r34[i, j]), 9),
            "inv_rho_r34": round(float(f_r34[i, j]) / (CS * d0), 9),
        })
        print(f"seed {seed}: D0={d0:>2}  現行={rows[-1]['inv_rho_current']:.4f}  "
              f"R34={rows[-1]['inv_rho_r34']:.4f}")
    cur = [r["inv_rho_current"] for r in rows]
    r34 = [r["inv_rho_r34"] for r in rows]
    print()
    print(f"現行マスク: 中央値 {np.median(cur):.4f}  [{min(cur):.4f}–{max(cur):.4f}]")
    print(f"R34 マスク: 中央値 {np.median(r34):.4f}  [{min(r34):.4f}–{max(r34):.4f}]")
    print("学生B の報告: 中央値 0.6923  [0.5012–0.7847]")
    doc = {"band": "seed 7000-7019", "grid_h_m": h, "grid_n": _GEO_GRID_N,
           "median_current": float(np.median(cur)), "median_r34": float(np.median(r34)),
           "rows": rows}
    path = f"{REPO_ROOT}/verification/out/r34_band7000.json"
    with open(path, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"書き出し: {path}")


if __name__ == "__main__":
    main()
