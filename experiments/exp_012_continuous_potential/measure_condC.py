"""
experiments/exp_012_continuous_potential/measure_condC.py
=========================================================
条件 C（配置空間の測地距離場）と条件 C'（ρ 倍で総整形量を E に揃えたもの）の
**登録値を測る**スクリプト。裁定 R32（場を配置空間の測地へ）・R33（収縮の基準点）
の後の再測定に使う。

⚠️ **基準点の併記**（裁定 R33。**数値を書くときは必ず基準を添える**）:
    許可 = **障害物（壁・柱）の表面**から w_lat = **0.0400 m** 以上（裁定 R34。一次の書き方）
         = 壁の**直線部**でのみ「壁**中心線**から t_w/2 + w_lat = 0.0460 m」と等価
  **柱・壁の端では等価にならない**（表面までの最短距離が斜めに測られるため）。
  実装（`mouse.maze6_env._geo_allowed_mask()`）は**物理モデルに実際に生成されている
  障害物 geom** から箱を読み、表面距離で判定する。

測る内容:
  I.   C' の ρ 配管の検証（実装が定義どおりか。走行を通した実測）
  II.  1/ρ = g(start)/(cs·D₀) の分布（裁定 R24-1 の記録の義務）
  III. C-1: 区画中心の g と d·cs の差の分布（裁定 R21-C。(a) の代わり）
  IV.  (L) 刻み不変性 — **裁定 R28-3 の 2 本立て**
         (1) 超過 |Δg| − |Δp| の**有界性**
         (2) 刻みを細かくしたときの**解像度スケーリング**
       有界性だけでは真の跳びと離散化誤差を判別できない（判別はスケーリングが担う）。
       あわせて、最大比を与える点対が**マスク境界に接しているか**で機構を判別する
       （帯へ延長した値が双線形に混ざる実装上の欠陥か、場そのものの性質か）。

使い方:
    .venv/bin/python experiments/exp_012_continuous_potential/measure_condC.py
    .venv/bin/python experiments/exp_012_continuous_potential/measure_condC.py --quick
    （--quick は (L) の最細水準と走行検証を省く。既定は全部）

出力: 標準出力への表 ＋ outputs/exp_012_condC/measure_condC.json
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mouse.maze6_env import (  # noqa: E402
    _GEO_CLEARANCE, _GEO_GRID_H, _GEO_GRID_N, _GEO_STEPS_PER_CELL,
    _ROBOT_LAT_HALF_WIDTH, Maze6Env,
)
from mouse.maze6_gen import GOAL_CELLS, SIZE  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

VALID_SEEDS = list(range(7000, 7020))     # 検証帯（研究計画書 §9-7）。学習には使わない帯
MAZE_MODE = "loop"
_P = RobotParams()
CS = _P.cell_size

# (L) の刻み（区画一辺 0.18 m を N 分割 → 6.0 / 1.5 / 0.375 mm）。テストと同じ 3 水準。
GRID_LEVELS = (30, 120, 480)
L_SEEDS = VALID_SEEDS[:5]                 # (L) は 5 面（全区画を舐めるので重い）
LATERAL_PERTURB_M = 0.020                 # reset の横擾乱（Maze6Env の _LATERAL_PERTURB_M）


# ======================================================================
# 共通ヘルパ
# ======================================================================
def geo_env(seed: int, rho_scale: bool = False) -> Maze6Env:
    """測地距離場を検査するための環境。

    **裁定 R34 以降、マスクは物理モデルの障害物 geom から作る**ので、MuJoCo モデルが
    要る。したがって `reset()` を通す（迷路の読み込み・sim の構築・場の前計算まで
    実装と同じ経路を通る）。検査対象は実装の実物である。
    """
    env = Maze6Env(maze_dir=REPO_ROOT, maze_seeds=[seed], mode="fixed",
                   maze_mode=MAZE_MODE, geodesic_potential=True,
                   geodesic_rho_scale=rho_scale)
    env.reset(seed=0)
    return env


def bilinear_vec(field: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """`Maze6Env._geodesic_value()` のベクトル化版（**同一の式**であることを検算する）。

    実装を写しているので、必ず `check_bilinear_matches()` で実物との一致を確認して
    から使う（写し間違いを検査に持ち込まないため）。
    """
    h, N = _GEO_GRID_H, _GEO_GRID_N
    u, v = xs / h, ys / h
    i0 = np.clip(np.floor(u).astype(np.int64), 0, N - 2)
    j0 = np.clip(np.floor(v).astype(np.int64), 0, N - 2)
    a, b = u - i0, v - j0
    return ((1.0 - a) * (1.0 - b) * field[i0, j0]
            + a * (1.0 - b) * field[i0 + 1, j0]
            + (1.0 - a) * b * field[i0, j0 + 1]
            + a * b * field[i0 + 1, j0 + 1])


def stencil_all_allowed(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """双線形が使う 4 隅がすべて到達可能な格子点か（実装欠陥の判別に使う）。"""
    h, N = _GEO_GRID_H, _GEO_GRID_N
    i0 = np.clip(np.floor(xs / h).astype(np.int64), 0, N - 2)
    j0 = np.clip(np.floor(ys / h).astype(np.int64), 0, N - 2)
    return (mask[i0, j0] & mask[i0 + 1, j0] & mask[i0, j0 + 1] & mask[i0 + 1, j0 + 1])


def blocked_flags(env: Maze6Env):
    """障害物の箱（cx, cy, hx, hy）。`_geo_obstacle_boxes()` の実物をそのまま使う。"""
    return env._geo_obstacle_boxes()


def reachable_point(env, boxes, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """**格子点ではなく任意の点**について「機体中心が居られるか」を判定する。

    判定規則は `_geo_allowed_mask()` と同じ（裁定 R34）: **障害物（壁・柱）の表面から
    $w_\\text{lat}$ = 0.0400 m 以上**離れていることを要求する（機体 = 半径 w_lat の円板）。
    ⚠️ 基準点: **壁面から 0.0400 m**。壁の**直線部**でのみ「壁中心線から 0.0460 m」と等価で、
    柱・壁の端では等価にならない。
    """
    dmin = np.full(np.shape(xs), np.inf, dtype=np.float64)
    for cx, cy, hx, hy in boxes:
        dx = np.maximum(np.abs(xs - cx) - hx, 0.0)
        dy = np.maximum(np.abs(ys - cy) - hy, 0.0)
        np.minimum(dmin, np.hypot(dx, dy), out=dmin)
    return dmin >= _ROBOT_LAT_HALF_WIDTH


def pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=np.float64), q))


def effective_clearance(env: Maze6Env) -> dict:
    """**実効離隔** = 許可格子点が実際に確保している最小の障害物表面距離（C-6 の要求）。

    格子の量子化のため、実効離隔は公称 $w_\\text{lat}$ = 0.0400 m より必ず大きくなる
    （壁面から k·h_g − t_w/2 の位置にしか格子点が無いため）。**解像度掃引の結果を
    読むときは、これを併記しないと「解像度の効果」と「実効離隔の効果」が交絡する**
    （准教授の C-6 精緻化。7.5 mm で +16.3%・6.0 mm で +5.0% になる）。
    """
    N, H = _GEO_GRID_N, _GEO_GRID_H
    pos = np.arange(N) * H
    X = pos[:, None] * np.ones((1, N))
    Y = np.ones((N, 1)) * pos[None, :]
    dmin = np.full((N, N), np.inf)
    for cx, cy, hx, hy in env._geo_obstacle_boxes():
        dx = np.maximum(np.abs(X - cx) - hx, 0.0)
        dy = np.maximum(np.abs(Y - cy) - hy, 0.0)
        np.minimum(dmin, np.hypot(dx, dy), out=dmin)
    mask = env._geo_allowed_mask()
    d_allowed = dmin[mask]
    return dict(h_g_mm=H * 1000.0, nominal_m=_ROBOT_LAT_HALF_WIDTH,
                effective_min_m=float(d_allowed.min()),
                excess_pct=float((d_allowed.min() / _ROBOT_LAT_HALF_WIDTH - 1.0) * 100.0),
                n_allowed=int(mask.sum()), n_total=int(mask.size),
                allowed_frac=float(mask.mean()))


def measure_reset_distribution(n_draws: int = 10) -> dict:
    """擾乱つき reset を n_draws 回引いたときの g(start) と 1/ρ の分布（裁定 R35-2）。

    実装は `reset()` で**擾乱後の真の位置**の g をエピソード定数にするので、学習中に
    記録される `geo_inv_rho` はこの分布で揺れる。区画中心での決定的な値（照合用）と
    区別するために別立てで測る。
    """
    rows = []
    for seed in VALID_SEEDS:
        env = Maze6Env(maze_dir=REPO_ROOT, maze_seeds=[seed], mode="fixed",
                       maze_mode=MAZE_MODE, geodesic_potential=True)
        gs, irs = [], []
        for k in range(n_draws):
            _obs, info = env.reset(seed=1000 + k)
            gs.append(float(env._geo_start))
            irs.append(float(info["geo_inv_rho"]))
        d0 = int(env._d_start)
        # 同じ面の区画中心での決定的な値（対照）
        sx, sy = env.maze["start"]
        g_center = float(env._geodesic_value(sx * CS + CS / 2, sy * CS + CS / 2))
        rows.append(dict(
            seed=seed, d0=d0, n_draws=n_draws,
            g_center=g_center, inv_rho_center=g_center / (CS * d0),
            g_median=float(np.median(gs)), g_min=float(np.min(gs)), g_max=float(np.max(gs)),
            inv_rho_median=float(np.median(irs)), inv_rho_min=float(np.min(irs)),
            inv_rho_max=float(np.max(irs))))
        print(f"  面{seed} D₀={d0:>2}  g(中心)={g_center:.6f}  "
              f"g(reset×{n_draws}) 中央値 {rows[-1]['g_median']:.6f} "
              f"[{rows[-1]['g_min']:.6f}, {rows[-1]['g_max']:.6f}]  "
              f"1/ρ 中央値 {rows[-1]['inv_rho_median']:.6f} "
              f"[{rows[-1]['inv_rho_min']:.6f}, {rows[-1]['inv_rho_max']:.6f}]", flush=True)
    med = [r["inv_rho_median"] for r in rows]
    spread = [r["inv_rho_max"] - r["inv_rho_min"] for r in rows]
    return dict(n_draws=n_draws, per_seed=rows,
                inv_rho_median_of_medians=float(np.median(med)),
                max_within_maze_spread=float(np.max(spread)),
                median_within_maze_spread=float(np.median(spread)))


# ======================================================================
# 0. ベクトル化の検算（写し間違いを検査に持ち込まない）
# ======================================================================
def check_bilinear_matches(rng) -> dict:
    env = geo_env(VALID_SEEDS[0])
    xs = rng.uniform(0.0, SIZE * CS, size=2000)
    ys = rng.uniform(0.0, SIZE * CS, size=2000)
    mine = bilinear_vec(env._geo_field, xs, ys)
    theirs = np.array([env._geodesic_value(float(x), float(y)) for x, y in zip(xs, ys)])
    d = float(np.max(np.abs(mine - theirs)))
    print(f"[0] ベクトル化した双線形 vs 実装 `_geodesic_value()`: "
          f"2000 点で最大差 {d:.3e}（0 でなければ検査側の写し間違い）")
    return dict(max_abs_diff=d, n=2000)


# ======================================================================
# I. C' の ρ 配管の検証
# ======================================================================
def verify_rho_plumbing(run_rollout: bool = True) -> dict:
    """条件 C' の実装が定義 Φ_C' = ρ·Φ_C（ρ = cs·D₀/g(start)）どおりかを確認する。"""
    out = {}
    # --- (i) 単独指定を弾く ------------------------------------------------
    try:
        Maze6Env(maze_dir=REPO_ROOT, maze_seeds=[VALID_SEEDS[0]], mode="fixed",
                 geodesic_rho_scale=True)
        out["rejects_lone_flag"] = False
    except ValueError:
        out["rejects_lone_flag"] = True

    # --- (ii) 幾何だけでの ρ の一致 ----------------------------------------
    rows = []
    for seed in VALID_SEEDS:
        env = geo_env(seed)
        m = env.maze
        sx, sy = m["start"]
        cx_m, cy_m = sx * CS + CS / 2, sy * CS + CS / 2
        g_start = env._geodesic_value(cx_m, cy_m)
        d0 = env._d_start
        rows.append(dict(seed=seed, d0=int(d0), g_start=float(g_start),
                         rho=float(CS * d0 / g_start),
                         inv_rho=float(g_start / (CS * d0))))
    out["rho_at_start_center"] = rows

    # --- (iii) 走行を通した検証（Φ 列が厳密に ρ 倍か・Φ₀=0 か）-------------
    if run_rollout:
        details = []
        for seed in VALID_SEEDS[:3]:
            phis = {}
            rho_seen = None
            info0 = None
            for scale in (False, True):
                env = Maze6Env(maze_dir=REPO_ROOT, maze_seeds=[seed], mode="fixed",
                               maze_mode=MAZE_MODE, geodesic_potential=True,
                               geodesic_rho_scale=scale)
                _obs, info = env.reset(seed=0)
                if scale:
                    rho_seen = env._geo_rho
                else:
                    info0 = info
                series = [env._potential(env._cell, env._prev_cell,
                                         *env.sim.privileged_pose()[:2])]
                for _t in range(60):
                    env.step(np.array([0.6, 0.6], dtype=np.float64))
                    x, y, _ = env.sim.privileged_pose()
                    series.append(env._potential(env._cell, env._prev_cell, x, y))
                phis[scale] = np.array(series, dtype=np.float64)
            base, scaled = phis[False], phis[True]
            resid = float(np.max(np.abs(scaled - rho_seen * base)))
            details.append(dict(
                seed=seed, rho=float(rho_seen), n_steps=len(base),
                max_abs_resid_scaled_minus_rho_times_base=resid,
                phi0_base=float(base[0]), phi0_scaled=float(scaled[0]),
                info_inv_rho=float(info0["geo_inv_rho"]),
                info_rho_times_inv_rho=float(rho_seen * info0["geo_inv_rho"]),
                max_abs_phi_base=float(np.max(np.abs(base))),
            ))
        out["rollout"] = details

    # --- (iv) 総整形量が条件 E に一致するか（ゴール格子点での Φ）-----------
    tot = []
    for seed in VALID_SEEDS:
        env = geo_env(seed, rho_scale=True)
        m = env.maze
        sx, sy = m["start"]
        cx_m, cy_m = sx * CS + CS / 2, sy * CS + CS / 2
        env._geo_start = env._geodesic_value(cx_m, cy_m)
        env._geo_rho = (CS * env._d_start) / env._geo_start
        # ゴール区画の**区画中心**で Φ を測る（g = 0 の点）。
        # ⚠️ ゴール 2x2 の**中心**は使えない — そこには柱 post_3_3 が実際に立っており
        # （design.md「既知の環境特性」）、R34 のマスクでは禁止帯に入るため g = 0 でない。
        gc = sorted(GOAL_CELLS)[0]
        gxm, gym = gc[0] * CS + CS / 2, gc[1] * CS + CS / 2
        phi_goal = env._potential_geodesic(gxm, gym)
        tot.append(dict(seed=seed, d0=int(env._d_start),
                        phi_goal=float(phi_goal), target=float(CS * env._d_start),
                        err=float(phi_goal - CS * env._d_start)))
    out["total_shaping_at_goal"] = tot
    return out


# ======================================================================
# II & III. 1/ρ と C-1
# ======================================================================
def measure_inv_rho_and_c1() -> dict:
    inv_rhos, rhos, c1, per_seed = [], [], [], []
    inv_rho_perturb = []
    for seed in VALID_SEEDS:
        env = geo_env(seed)
        flags = blocked_flags(env)
        m = env.maze
        sx, sy = m["start"]
        cx_m, cy_m = sx * CS + CS / 2, sy * CS + CS / 2
        g_start = env._geodesic_value(cx_m, cy_m)
        d0 = env._d_start
        inv_rho = g_start / (CS * d0)
        inv_rhos.append(inv_rho)
        rhos.append(1.0 / inv_rho)

        # 横擾乱 ±20 mm ぶんの振れ（reset は方位に垂直な向きへずらす。ここでは
        # x・y 両方向を舐めて上下界を取る＝実際の擾乱の**超集合**で保守的に見る）
        offs = np.linspace(-LATERAL_PERTURB_M, LATERAL_PERTURB_M, 41)
        cand = []
        for dx in offs:
            for dy in (0.0,):
                if reachable_point(env, flags, np.array([cx_m + dx]), np.array([cy_m + dy]))[0]:
                    cand.append(env._geodesic_value(cx_m + dx, cy_m + dy) / (CS * d0))
        for dy in offs:
            if reachable_point(env, flags, np.array([cx_m]), np.array([cy_m + dy]))[0]:
                cand.append(env._geodesic_value(cx_m, cy_m + dy) / (CS * d0))
        inv_rho_perturb.append((min(cand), max(cand)))

        # C-1: 非ゴール区画の中心での g − d·cs
        diffs = []
        for cx in range(SIZE):
            for cy in range(SIZE):
                if (cx, cy) in GOAL_CELLS:
                    continue
                x = cx * CS + CS / 2
                y = cy * CS + CS / 2
                diffs.append(env._geodesic_value(x, y) - env._dist_map[(cx, cy)] * CS)
        c1 += diffs
        per_seed.append(dict(seed=seed, d0=int(d0), g_start=float(g_start),
                             inv_rho=float(inv_rho), rho=float(1.0 / inv_rho),
                             c1_median=float(np.median(diffs))))
    lo = [p[0] for p in inv_rho_perturb]
    hi = [p[1] for p in inv_rho_perturb]
    return dict(
        n_mazes=len(VALID_SEEDS),
        inv_rho=dict(median=float(np.median(inv_rhos)), min=float(np.min(inv_rhos)),
                     max=float(np.max(inv_rhos)), mean=float(np.mean(inv_rhos))),
        rho=dict(median=float(np.median(rhos)), min=float(np.min(rhos)),
                 max=float(np.max(rhos)), mean=float(np.mean(rhos))),
        inv_rho_under_lateral_perturbation=dict(
            min_over_mazes=float(np.min(lo)), max_over_mazes=float(np.max(hi)),
            max_spread_within_maze=float(np.max(np.array(hi) - np.array(lo)))),
        c1=dict(n=len(c1), median=float(np.median(c1)), mean=float(np.mean(c1)),
                min=float(np.min(c1)), max=float(np.max(c1)),
                p05=pct(c1, 5), p95=pct(c1, 95)),
        per_seed=per_seed,
    )


# ======================================================================
# IV. (L) 刻み不変性（R28 の 2 本立て）＋ 機構の判別
# ======================================================================
def measure_L(levels=GRID_LEVELS, seeds=L_SEEDS) -> dict:
    """全面を格子で走査し、|Δg|/|Δp| の最大と超過 |Δg| − |Δp| の最大を測る。

    **到達可能域に限る**（両端とも機体中心が居られる位置）。閉じた壁の中心線から
    0.0460 m 以内は両端とも除かれるので、壁をまたぐ点対は構成上入らない
    （旧実装で必要だった「壁を挟む点対の除外」がここでは不要になる）。
    """
    results = []
    for N in levels:
        step = CS / N
        diag = step * math.sqrt(2.0)
        t0 = time.time()
        worst = dict(ratio=0.0, at=None, pure=None)
        worst_pure = dict(ratio=0.0, at=None)
        worst_exc = dict(exc=-1e9, at=None, pure=None)
        worst_exc_pure = dict(exc=-1e9, at=None)
        n_pairs = 0
        n_pairs_pure = 0
        for seed in seeds:
            env = geo_env(seed)
            field = env._geo_field
            mask = env._geo_allowed_mask()
            flags = blocked_flags(env)
            n_side = SIZE * N + 1
            # 列方向にチャンクして走る（最細水準でも記憶容量を抑える）
            chunk = max(1, int(2_000_000 / n_side))
            for i_start in range(0, n_side - 1, chunk):
                i_end = min(i_start + chunk + 1, n_side)      # 1 列ぶん重ねる
                xi = np.arange(i_start, i_end)
                yj = np.arange(n_side)
                X = (xi[:, None] * step) * np.ones((1, n_side))
                Y = np.ones((len(xi), 1)) * (yj[None, :] * step)
                G = bilinear_vec(field, X, Y)
                R = reachable_point(env, flags, X, Y)
                S = stencil_all_allowed(mask, X, Y)
                for (di, dj, dist) in ((1, 0, step), (0, 1, step),
                                       (1, 1, diag), (1, -1, diag)):
                    a = (slice(0, len(xi) - di if di else None),
                         slice(max(0, -dj), (len(yj) - dj) if dj > 0 else None))
                    b = (slice(di, None) if di else slice(0, None),
                         slice(max(0, dj), (len(yj) + dj) if dj < 0 else None))
                    ga, gb = G[a], G[b]
                    ok = R[a] & R[b]
                    if not ok.any():
                        continue
                    d = np.abs(gb - ga)
                    ratio = np.where(ok, d / dist, 0.0)
                    exc = np.where(ok, d - dist, -1e9)
                    pure = S[a] & S[b] & ok
                    n_pairs += int(ok.sum())
                    n_pairs_pure += int(pure.sum())
                    k = int(np.argmax(ratio))
                    if ratio.flat[k] > worst["ratio"]:
                        idx = np.unravel_index(k, ratio.shape)
                        worst = dict(ratio=float(ratio[idx]),
                                     at=dict(seed=seed, x=float(X[a][idx]), y=float(Y[a][idx]),
                                             dir=(di, dj)),
                                     pure=bool(pure[idx]))
                    k = int(np.argmax(exc))
                    if exc.flat[k] > worst_exc["exc"]:
                        idx = np.unravel_index(k, exc.shape)
                        worst_exc = dict(exc=float(exc[idx]),
                                         at=dict(seed=seed, x=float(X[a][idx]),
                                                 y=float(Y[a][idx]), dir=(di, dj)),
                                         pure=bool(pure[idx]))
                    if pure.any():
                        rp = np.where(pure, ratio, 0.0)
                        k = int(np.argmax(rp))
                        if rp.flat[k] > worst_pure["ratio"]:
                            idx = np.unravel_index(k, rp.shape)
                            worst_pure = dict(ratio=float(rp[idx]),
                                              at=dict(seed=seed, x=float(X[a][idx]),
                                                      y=float(Y[a][idx]), dir=(di, dj)))
                        ep = np.where(pure, exc, -1e9)
                        k = int(np.argmax(ep))
                        if ep.flat[k] > worst_exc_pure["exc"]:
                            idx = np.unravel_index(k, ep.shape)
                            worst_exc_pure = dict(exc=float(ep[idx]),
                                                  at=dict(seed=seed, x=float(X[a][idx]),
                                                          y=float(Y[a][idx]), dir=(di, dj)))
        results.append(dict(
            n_div=N, step_mm=step * 1000.0, n_pairs=n_pairs, n_pairs_pure=n_pairs_pure,
            max_ratio=worst["ratio"], max_ratio_at=worst["at"],
            max_ratio_stencil_pure=worst["pure"],
            max_ratio_pure_only=worst_pure["ratio"], max_ratio_pure_at=worst_pure["at"],
            max_excess_m=worst_exc["exc"], max_excess_at=worst_exc["at"],
            max_excess_stencil_pure=worst_exc["pure"],
            max_excess_pure_only_m=worst_exc_pure["exc"],
            wall_time_s=time.time() - t0))
        r = results[-1]
        print(f"[IV] 刻み {r['step_mm']:.3f} mm: 最大比 {r['max_ratio']:.4f} "
              f"（帯に接しない点対に限ると {r['max_ratio_pure_only']:.4f}）／"
              f"最大超過 {r['max_excess_m']*1000:.4f} mm "
              f"（同 {r['max_excess_pure_only_m']*1000:.4f} mm）／"
              f"点対 {r['n_pairs']:,}（うち帯に接しない {r['n_pairs_pure']:,}）"
              f" [{r['wall_time_s']:.1f} s]", flush=True)
    return dict(levels=results, seeds=list(seeds))


# ======================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description="条件 C・C' の登録値の測定（R32/R33 後）")
    p.add_argument("--quick", action="store_true",
                   help="(L) の最細水準と走行検証を省く")
    p.add_argument("--no-rollout", action="store_true",
                   help="I の走行検証（MuJoCo を建てる）を省く")
    p.add_argument("--levels", default=None,
                   help="(L) の刻み（区画の分割数をカンマ区切り。例 30,120,480。"
                        "'none' で (L) を省く）")
    p.add_argument("--reset-dist", action="store_true",
                   help="擾乱つき reset を引いた g(start)・1/ρ の分布も測る（R35-2）")
    p.add_argument("--reset-draws", type=int, default=10)
    p.add_argument("--out", default="outputs/exp_012_condC/measure_condC.json")
    args = p.parse_args(argv)

    rng = np.random.default_rng(0)
    print("=" * 78)
    print("条件 C・C' の測定（配置空間の測地距離場。裁定 R32・R33）")
    print(f"  収縮の基準（R34）: 障害物**表面**から w_lat = {_ROBOT_LAT_HALF_WIDTH:.4f} m 以上"
          f"（壁の直線部でのみ壁中心線から {_GEO_CLEARANCE:.4f} m と等価）")
    print(f"  格子: {_GEO_GRID_N}×{_GEO_GRID_N}（h_g = {_GEO_GRID_H*1000:.1f} mm・"
          f"1 区画 {_GEO_STEPS_PER_CELL} 分割）")
    print("=" * 78)

    res = {}
    res["bilinear_check"] = check_bilinear_matches(rng)
    res["effective_clearance"] = effective_clearance(geo_env(VALID_SEEDS[0]))
    ec = res["effective_clearance"]
    print(f"[0] 実効離隔（許可格子点が確保する最小の障害物表面距離）: "
          f"{ec['effective_min_m']*1000:.2f} mm（公称 {ec['nominal_m']*1000:.1f} mm の "
          f"{ec['excess_pct']:+.1f}%・h_g = {ec['h_g_mm']:.1f} mm）／"
          f"許可格子点 {ec['n_allowed']:,}/{ec['n_total']:,}（{ec['allowed_frac']*100:.1f}%）")

    print("\n[I] C' の ρ 配管の検証")
    res["rho_plumbing"] = verify_rho_plumbing(
        run_rollout=not (args.quick or args.no_rollout))
    rp = res["rho_plumbing"]
    print(f"  geodesic_rho_scale 単独指定を ValueError で弾く: {rp['rejects_lone_flag']}")
    if "rollout" in rp:
        for d in rp["rollout"]:
            print(f"  面{d['seed']} ρ={d['rho']:.4f}  "
                  f"max|Φ_C' − ρ·Φ_C| = {d['max_abs_resid_scaled_minus_rho_times_base']:.3e} m "
                  f"（Φ 列 {d['n_steps']} 点・max|Φ_C| = {d['max_abs_phi_base']:.4f}）  "
                  f"Φ₀: {d['phi0_base']:.1e} / {d['phi0_scaled']:.1e}  "
                  f"ρ·(1/ρ)={d['info_rho_times_inv_rho']:.12f}")
    errs = [abs(t["err"]) for t in rp["total_shaping_at_goal"]]
    print(f"  総整形量（ゴールでの Φ_C'）が cs·D₀ に一致: 20 面で最大差 {max(errs):.3e} m")

    print("\n[II/III] 1/ρ と C-1")
    res["inv_rho_c1"] = measure_inv_rho_and_c1()
    ir = res["inv_rho_c1"]
    print(f"  1/ρ : 中央値 {ir['inv_rho']['median']:.4f}  "
          f"最小 {ir['inv_rho']['min']:.4f}  最大 {ir['inv_rho']['max']:.4f}")
    print(f"  ρ   : 中央値 {ir['rho']['median']:.4f}  "
          f"最小 {ir['rho']['min']:.4f}  最大 {ir['rho']['max']:.4f}")
    print(f"  横擾乱 ±20 mm での 1/ρ の振れ: 面内の最大幅 "
          f"{ir['inv_rho_under_lateral_perturbation']['max_spread_within_maze']:.4f}")
    print(f"  C-1（区画中心の g − d·cs）: n={ir['c1']['n']}  中央値 {ir['c1']['median']:.5f} m  "
          f"[{ir['c1']['min']:.5f}, {ir['c1']['max']:.5f}]")

    if args.reset_dist:
        print("\n[II-b] 擾乱つき reset での g(start)・1/ρ の分布（裁定 R35-2）")
        res["reset_distribution"] = measure_reset_distribution(n_draws=args.reset_draws)
        rd = res["reset_distribution"]
        print(f"  面ごと中央値の中央値 = {rd['inv_rho_median_of_medians']:.6f}／"
              f"面内の振れ幅: 中央値 {rd['median_within_maze_spread']:.6f}・"
              f"最大 {rd['max_within_maze_spread']:.6f}")

    if args.levels is not None and args.levels.lower() == "none":
        print("\n[IV] (L) は省略（--levels none）")
    else:
        print("\n[IV] (L) 刻み不変性（R28 の 2 本立て）")
        if args.levels:
            levels = tuple(int(s) for s in args.levels.split(","))
        else:
            levels = GRID_LEVELS[:2] if args.quick else GRID_LEVELS
        res["L"] = measure_L(levels=levels)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
