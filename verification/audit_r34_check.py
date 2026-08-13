"""裁定 R34 の実装に対する自動検査（R34-3・R34-4 と予測の照合）。

⚠️ **本スクリプトは学生B の R34 実装を読む前に書いた**（`AUDIT_006` §4-octies の
確認項目と `COMMIT_002` §11 の予測を、実行可能な形にしただけである）。
**実装を見てから合格条件を作っていない**ことを、コミットの時系列で担保する。

自動で検査できるのは 3 件:
  - **R34-3（距離の厳密さ）**: 実装のマスクが、私が独立に構成した
    「物理モデルの箱の表面からの厳密なユークリッド距離 ≥ w_lat」マスクと**一致するか**。
    画像処理的な膨張・チェビシェフ距離・隅を四角く切る実装なら**隅で食い違う**
  - **R34-4（連結性）**: `_compute_geodesic_field()` が到達不能検査を発火させずに完走するか
  - **食い込みゼロ**: 実装が許可する格子点が、物理の壁・柱から w_lat 未満に無いこと
    （是正前は **9.17%・最悪 0.00 mm** だった。`AUDIT_011` §5-quinquies §4）
  - **予測の照合**: 提出帯 seed 7000〜7019 の 1/ρ 中央値が **0.7344**（面ごとは
    `verification/out/r34_band7000.json`）と一致するか

R34-1（障害物集合の導出元）・R34-2（円板近似の限界の条文化）・R34-5（ρ の再導出と
C' 連動）は**文書と実装の読解が要るので自動化しない。**別途 `AUDIT_013` で判定する。
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
PRED = f"{REPO_ROOT}/verification/out/r34_band7000.json"


def impl_env(seed):
    """実装のマスクを得るために**本物の環境**を建てる。

    ⚠️ **配管だけの変更**（2026-08-14。検査の中身は事前作成時のまま）:
    提出版 `b5dd960` の `_geo_obstacle_boxes()` は **MuJoCo モデル（`self.sim.model`）
    から障害物を読む**ので、器（SimpleNamespace）では呼べない。**これは R34-1 が
    要求した「モデル本体から導出」そのものなので、合格方向の帰結である。**
    **合格条件は 1 つも変えていない。**
    """
    env = Maze6Env(maze_dir=REPO_ROOT, maze_seeds=[seed], mode="fixed",
                   maze_mode=MODE, geodesic_potential=True)
    env.reset(seed=0)          # sim（MuJoCo モデル）はここで建つ
    return env


def main():
    h = _GEO_GRID_H
    xs = np.arange(_GEO_GRID_N) * h
    pred = {r["seed"]: r for r in json.load(open(PRED))["rows"]} if __import__("os").path.exists(PRED) else {}

    n_mismatch = n_total = 0
    n_bite = n_allowed = 0
    worst_bite = (9.9, None)
    field_fail = []
    inv_rho = []
    print(f"{'seed':>6} {'マスク不一致':>12} {'食い込み点':>10} {'1/ρ':>8} {'予測':>8} {'差':>9}")
    for seed in BAND:
        m = generate_maze(seed, mode=MODE)
        env = impl_env(seed)
        assert tuple(env.maze["start"]) == tuple(m["start"]), "迷路が一致しない"
        im = env._geo_allowed_mask()          # 仕様マスク（表面から w_lat）= 1/ρ_spec の側
        rects = obstacle_rects(m["v_walls"], m["h_walls"], with_posts=True, full_length=False)
        mine = erode_mask(xs, xs, rects, W_LAT)
        mism = int(np.sum(im != mine))
        n_mismatch += mism
        n_total += im.size

        # 食い込み検査: 許可点から物理障害物までの最小距離
        X, Y = xs[:, None], xs[None, :]
        d = np.full((len(xs), len(xs)), 9.9)
        for (x0, x1, y0, y1) in rects:
            dx = np.maximum(np.maximum(x0 - X, X - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - Y, Y - y1), 0.0)
            d = np.minimum(d, np.hypot(dx, dy))
        n_allowed += int(im.sum())
        n_bite += int(np.sum(im & (d < W_LAT)))
        if im.any():
            w = float(np.min(np.where(im, d, 9.9)))
            if w < worst_bite[0]:
                worst_bite = (w, seed)

        # 場が完走するか ＋ 1/ρ
        try:
            env._geo_allowed_mask = lambda *a, mm=im, **kw: mm
            field = env._compute_geodesic_field()
            dmap = shortest_distances(m["v_walls"], m["h_walls"])
            start = tuple(int(v) for v in m["start"])
            d0 = int(dmap[start])
            i = int(round((start[0] + 0.5) * CS / h))
            j = int(round((start[1] + 0.5) * CS / h))
            r = float(field[i, j]) / (CS * d0)
            env.close()
        except AssertionError as e:
            field_fail.append((seed, str(e)[:60]))
            r = float("nan")
        inv_rho.append(r)
        p = pred.get(seed, {}).get("inv_rho_r34")
        print(f"{seed:>6} {mism:>12} {int(np.sum(im & (d < W_LAT))):>10} {r:>8.4f} "
              + (f"{p:>8.4f} {r-p:>+9.5f}" if p is not None else f"{'—':>8} {'—':>9}"))

    print()
    print("=" * 78)
    print(f"R34-3 マスクの一致     : 不一致 {n_mismatch} / {n_total} 点 "
          f"→ {'合格' if n_mismatch == 0 else '🔴 不合格（隅の扱いを見ること）'}")
    print(f"食い込みゼロ           : {n_bite} / {n_allowed} 点  最小距離 {worst_bite[0]*1000:.2f} mm "
          f"(seed {worst_bite[1]}) → {'合格' if n_bite == 0 else '🔴 不合格'}")
    print(f"R34-4 場の完走         : 失敗 {len(field_fail)} 面 "
          f"→ {'合格' if not field_fail else '🔴 不合格 ' + str(field_fail)}")
    good = [v for v in inv_rho if v == v]
    if good:
        med = float(np.median(good))
        print(f"予測の照合             : 1/ρ 中央値 実測 {med:.4f} 対 予測 0.7344 "
              f"(差 {med-0.7344:+.5f})")


if __name__ == "__main__":
    main()
