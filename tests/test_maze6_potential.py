"""
tests/test_maze6_potential.py
================
exp_012（ポテンシャル Φ の連続化）の単体テスト。

**このテストが全項目 PASS することが、exp_012 の学習を回す条件である**
（裁定 R5-1。仕様は `experiments/exp_012_continuous_potential/design.md` §4 末尾を正とする）。

pytest は使わない plain Python スクリプト（tests/test_corridor.py と同じ流儀）。
実行方法（リポジトリルートで）:
    .venv/bin/python tests/test_maze6_potential.py
    .venv/bin/python tests/test_maze6_potential.py --skip-corridor   # (e) の廊下側だけ省略

検査項目（design.md §4 末尾の (a)〜(e) ＋ (b-2) ＋ (b-3)。裁定 R8・R12 を反映）:
  (a)   区画中心で Φ が旧実装（階段版）と一致する（検証帯 20 面 × 全区画。ゴール区画は除く）
  (b)   跳びが無いこと。閾値は式で定義する（裁定 R7-6）:
            |ΔΦ_t| ≤ max(|v_t|, |v_(t−1)|)·Δt·1.25 + 0.001 m
        かつ絶対上限 v_max·Δt = 0.0409 m。直進・90°旋回・S 字の 3 種で全ステップを検査。
        同じ走行を階段版でも検査し、そちらでは 0.18 に近い跳びが出ることを確認する
        （＝この検査が連続化を検出できることの証拠）
  (b-2) 降下方向が曲がる区画での境界一致（裁定 R8。当初案の欠陥の回帰検査）
  (b-3) step() 経路と直接呼び出しの全ステップ一致（裁定 R12 の確認事項）
  (c)   Φ が横方向のずれに不感（折れ線が一直線の区画に限る。飛ばした区画数を報告）
  (d)   Φ₀ = 0（reset 直後、横 ±20 mm・方位 ±10° の擾乱あり）
  (e)   予約 seed の読み飛ばしと、廊下側（tests/test_corridor.py）の全項目 PASS
  (f)   【追加】既定 continuous_potential=False で既存挙動が変わらないこと
        （design.md §4 末尾の「既定は False とし、既存挙動が bit 単位で変わらない」）

いずれかのテストで例外/assert が起きても他のテストは継続実行し、最後に全テストの
実測値をまとめて表として print する。**閾値は design.md の式そのものであり、
テスト側で緩めてはならない。**
"""
import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mouse.maze6_env import Maze6Env, _RESERVED_MAZE_SEEDS  # noqa: E402
from mouse.maze6_gen import (  # noqa: E402
    GOAL_CELLS, SIZE, cells_open, generate_maze, shortest_distances,
)
from mouse.params import RobotParams  # noqa: E402

# ----------------------------------------------------------------------
# 定数（design.md §4・§6 と研究計画書 §9-7 に対応させる。ここで値を作らない）
# ----------------------------------------------------------------------
VALID_SEEDS = list(range(7000, 7020))     # 検証帯（§9-7）。学習には使わない帯
MAZE_MODE = "loop"                        # M2-0 の設定（exp_010/011/012 共通）

_P = RobotParams()
DT = _P.control_dt                        # 制御周期 [s]
CS = _P.cell_size                         # 区画寸法 [m]
WHEEL_R = _P.wheel_radius                 # 車輪半径 [m]
# 無負荷速度 v_max = r·V/(K_e·G_r)（design.md §4 (b)。3.0 V・G_r=5 で 4.09 m/s）
VMAX = WHEEL_R * _P.voltage_limit / (_P.motor_Ke * _P.gear_ratio)

# (b) の閾値の係数（design.md §4 (b) の式。**緩めてはならない**）
JUMP_SLOPE = 1.25                         # 離散化・滑り・旋回ぶんの余裕
JUMP_OFFSET = 0.001                       # [m]
JUMP_ABS_CAP = VMAX * DT                  # [m] 絶対上限

TOL_EXACT = 1e-12                         # (a)(b-3)(c)(f) の一致判定。(b-3) のみ design.md
                                          # が「絶対誤差 1e-12 未満」と明示。他は厳密一致が
                                          # 数学的に期待される量なので同じ水準を当てている
TOL_BOUNDARY = 1e-9                       # (b-2) は design.md が「絶対誤差 1e-9 未満」と明示。
                                          # (d) は design.md に数値の指定が無く、本テストの判断
                                          # で同じ水準を当てている（Φ₀ は厳密に 0 が期待値）

MAX_SCRIPT_STEPS = 600                    # 走行台本 1 本あたりの上限ステップ


# ======================================================================
# 【条文】テスト (b) の判定対象からの除外規則（design.md §4 (b) / 裁定 R12）
# ----------------------------------------------------------------------
# 除外してよいのは次の 2 種**のみ**である。これ以外は理由の如何を問わず除外しない。
#
#   条文 1「ゴール到達の最終ステップ」
#       info["goal"] が True のステップ。d=0 の区画で remaining=0 と特別扱いする
#       ため cs/2 = 0.09 m 跳ぶ。根拠: ゴール到達はエピソードの終了なので、この
#       跳びを含む閉路は作れない（＝報酬ポンプにならない）。
#
#   条文 2「方向反転ステップ」
#       d(新区画) = d(旧区画) + 1 のステップ。降下隣接の tie-break 起因で
#       最大 cs = 0.18 m 跳ぶ。根拠: 境界の一価性は全境界で機械確認済み
#       （2032 通り・最大差 4.4e-16 m）であり、跳びは別々の状態間の値差であって
#       報酬ポンプではない（閉路列挙で整形総和 = 0 を機械確認済み）。
#
# **「区画内」（区画が変わらない）と「前進」（d が 1 減る）で閾値を超えたら、
#   それは実装の欠陥であり除外しない。**上記 2 種のどちらでもない遷移
#   （区画が変わったが d の差が ±1 でない等）も除外しない。
#
# 除外したステップは黙って捨てず、種別ごとに件数と最大 |ΔΦ| を報告する。
# ======================================================================
KIND_IN_CELL = "区画内"
KIND_FORWARD = "前進"
KIND_GOAL_FINAL = "ゴール到達の最終ステップ"       # 条文 1（除外）
KIND_REVERSAL = "方向反転"                         # 条文 2（除外）
KIND_OTHER = "その他の遷移"
_EXCLUDED_KINDS = frozenset({KIND_GOAL_FINAL, KIND_REVERSAL})


def classify_step(cell_prev, cell_now, d_prev, d_now, goal_flag) -> str:
    """1 ステップを上の条文に従って分類する（除外の可否は _EXCLUDED_KINDS が決める）。"""
    if goal_flag:
        return KIND_GOAL_FINAL              # 条文 1
    if cell_now == cell_prev:
        return KIND_IN_CELL
    if d_now == d_prev - 1:
        return KIND_FORWARD
    if d_now == d_prev + 1:
        return KIND_REVERSAL                # 条文 2
    return KIND_OTHER


def jump_threshold(v_now: float, v_prev: float) -> float:
    """design.md §4 (b) の閾値 [m]。速度依存の式と絶対上限の**厳しい方**を採る。"""
    return min(max(abs(v_now), abs(v_prev)) * DT * JUMP_SLOPE + JUMP_OFFSET, JUMP_ABS_CAP)


# ======================================================================
# 共通ヘルパ
# ======================================================================
def geom_env(maze_seed: int, continuous: bool = True) -> Maze6Env:
    """幾何だけを検査するための環境（MuJoCo を構築しない）。

    reset() を呼ばずに maze / _dist_map / _d_start だけを reset() と同じ手順で
    埋める。検査対象は `_potential_stair` / `_potential_continuous` の実物である。
    """
    env = Maze6Env(maze_dir=REPO_ROOT, maze_seeds=[maze_seed], mode="fixed",
                   maze_mode=MAZE_MODE, continuous_potential=continuous)
    m = generate_maze(maze_seed, mode=MAZE_MODE)
    env.maze = m
    env._dist_map = shortest_distances(m["v_walls"], m["h_walls"])
    env._d_start = env._dist_map[tuple(m["start"])]
    return env


def run_env(maze_seed: int, continuous: bool, reset_seed: int = 0, **kwargs) -> Maze6Env:
    """実際に走らせる環境（MuJoCo を構築する）。"""
    env = Maze6Env(maze_dir=REPO_ROOT, maze_seeds=[maze_seed], mode="fixed",
                   maze_mode=MAZE_MODE, continuous_potential=continuous, **kwargs)
    env.reset(seed=reset_seed)
    return env


def forward_speed(env: Maze6Env) -> float:
    """車輪角速度から求めた前進速度 v = r(ω_L + ω_R)/2 [m/s]（design.md §4 (b)）。"""
    raw = env.sim.observation()
    n = env._n_dist
    return WHEEL_R * (float(raw[n + 6]) + float(raw[n + 7])) / 2.0


# 走行台本 3 種（design.md §4 (b): 直進・90°旋回・S 字）。開ループなので、
# 壁に当たるか上限に達するまで走らせて、そこまでの全ステップを検査する。
def script_straight(t: int) -> np.ndarray:
    return np.array([0.6, 0.6], dtype=np.float64)


def script_turn90(t: int) -> np.ndarray:
    if t < 40:
        return np.array([0.6, 0.6], dtype=np.float64)
    if t < 100:
        return np.array([0.55, 0.10], dtype=np.float64)   # 右へ大きく舵を切る
    return np.array([0.6, 0.6], dtype=np.float64)


def script_scurve(t: int) -> np.ndarray:
    d = 0.25 * math.sin(2.0 * math.pi * t / 100.0)        # 周期 1.0 s
    return np.array([0.5 + d, 0.5 - d], dtype=np.float64)


SCRIPTS = [("直進", script_straight), ("90°旋回", script_turn90), ("S字", script_scurve)]


def rollout(maze_seed: int, script, continuous: bool, reset_seed: int = 0,
            max_steps: int = MAX_SCRIPT_STEPS):
    """台本どおりに走らせ、各ステップの Φ・v・区画・d・ゴール判定を記録する。

    戻り値: (records, env)。records[0] は reset 直後（Φ₀・v₀）。
    """
    env = run_env(maze_seed, continuous, reset_seed=reset_seed)
    x, y, _ = env.sim.privileged_pose()
    recs = [dict(phi=float(env._prev_potential), v=forward_speed(env),
                 cell=env._cell, d=int(env._dist_map.get(env._cell, -1)),
                 goal=False, x=x, y=y)]
    for t in range(max_steps):
        _obs, _r, terminated, truncated, info = env.step(script(t))
        x, y, _ = env.sim.privileged_pose()
        recs.append(dict(phi=float(env._prev_potential), v=forward_speed(env),
                         cell=tuple(info["cell"]), d=int(info["dist_to_goal"]),
                         goal=bool(info["goal"]), x=x, y=y))
        if terminated or truncated:
            break
    return recs, env


def is_bend(w_in, center, w_out) -> bool:
    """折れ線 [w_in, center, w_out] が一直線でない（＝降下方向が曲がる）か。"""
    ax, ay = center[0] - w_in[0], center[1] - w_in[1]
    bx, by = w_out[0] - center[0], w_out[1] - center[1]
    if math.hypot(ax, ay) < 1e-12:      # w_in == center は起こらないが安全側に
        return False
    return abs(ax * by - ay * bx) > 1e-12


def prev_cell_candidates(env: Maze6Env, cell):
    """c_(−1) として起こりうるもの: None（reset 直後）と、開通した全隣接区画。"""
    return [None] + list(env._open_neighbors(cell))


# ======================================================================
# (a) 区画中心で階段版と一致する
# ======================================================================
def test_a_center_matches_stair():
    max_diff = 0.0
    max_at = None
    n_conf = 0
    n_cells = 0
    goal_max_diff = 0.0
    for seed in VALID_SEEDS:
        env = geom_env(seed)
        for cx in range(SIZE):
            for cy in range(SIZE):
                cell = (cx, cy)
                center = env._cell_center(cell)
                phi_s = env._potential_stair(cell)
                if cell in GOAL_CELLS:
                    # 仕様どおり判定対象から外す（参考値としてだけ測る）
                    phi_c = env._potential_continuous(cell, None, *center)
                    goal_max_diff = max(goal_max_diff, abs(phi_c - phi_s))
                    continue
                n_cells += 1
                for prev in prev_cell_candidates(env, cell):
                    phi_c = env._potential_continuous(cell, prev, *center)
                    diff = abs(phi_c - phi_s)
                    n_conf += 1
                    if diff > max_diff:
                        max_diff, max_at = diff, (seed, cell, prev)
    ok = max_diff < TOL_EXACT
    detail = [
        f"検査した (面, 区画, c_prev) の組: {n_conf} 通り（区画 {n_cells} 個 / {len(VALID_SEEDS)} 面）",
        f"最大 |Φ_連続 − Φ_階段| = {max_diff:.3e} m（許容 {TOL_EXACT:.0e}）",
        f"最大値の位置: {max_at}",
        f"参考: ゴール区画（判定対象外）の最大差 = {goal_max_diff:.3e} m",
    ]
    return ok, detail


# ======================================================================
# (b) 跳びが無いこと（＋ 階段版では跳びが出ることの確認）
# ======================================================================
def _scan_jumps(continuous: bool):
    """全台本 × 全検証面を走らせ、種別ごとに |ΔΦ| と閾値超過を集計する。"""
    stats = {}          # kind -> dict(n, max_jump, n_over, worst)
    n_steps = 0
    n_transitions = 0
    violations = []
    for seed in VALID_SEEDS:
        for sname, script in SCRIPTS:
            recs, env = rollout(seed, script, continuous)
            env.close()
            for t in range(1, len(recs)):
                a, b = recs[t - 1], recs[t]
                kind = classify_step(a["cell"], b["cell"], a["d"], b["d"], b["goal"])
                dphi = abs(b["phi"] - a["phi"])
                thr = jump_threshold(b["v"], a["v"])
                st = stats.setdefault(kind, dict(n=0, max_jump=0.0, n_over=0, worst=None))
                st["n"] += 1
                n_steps += 1
                if b["cell"] != a["cell"]:
                    n_transitions += 1
                if dphi > st["max_jump"]:
                    st["max_jump"] = dphi
                    st["worst"] = (seed, sname, t, a["cell"], b["cell"], thr)
                if dphi > thr:
                    st["n_over"] += 1
                    if kind not in _EXCLUDED_KINDS and len(violations) < 20:
                        violations.append(dict(seed=seed, script=sname, step=t, kind=kind,
                                               dphi=dphi, thr=thr,
                                               cell_from=a["cell"], cell_to=b["cell"],
                                               v_prev=a["v"], v_now=b["v"],
                                               xy_from=(a["x"], a["y"]),
                                               xy_to=(b["x"], b["y"])))
    return stats, n_steps, n_transitions, violations


def test_b_no_jump():
    stats, n_steps, n_trans, violations = _scan_jumps(continuous=True)
    n_over_judged = sum(v["n_over"] for k, v in stats.items() if k not in _EXCLUDED_KINDS)
    ok = (n_over_judged == 0) and (n_trans > 0)

    detail = [
        f"走行: {len(VALID_SEEDS)} 面 × 台本 {len(SCRIPTS)} 種、判定したステップ {n_steps}、"
        f"区画遷移 {n_trans} 回",
        f"閾値: min(max(|v_t|,|v_(t−1)|)·{DT}·{JUMP_SLOPE} + {JUMP_OFFSET}, "
        f"v_max·Δt={JUMP_ABS_CAP:.4f}) m　（v_max = {VMAX:.3f} m/s）",
        "",
        f"{'種別':<28}{'判定':<6}{'件数':>7}{'最大|ΔΦ|[m]':>14}{'閾値超過':>10}",
    ]
    for kind in (KIND_IN_CELL, KIND_FORWARD, KIND_OTHER, KIND_GOAL_FINAL, KIND_REVERSAL):
        st = stats.get(kind)
        if st is None:
            detail.append(f"{kind:<28}{'—':<6}{0:>7}{'—':>14}{'—':>10}")
            continue
        judged = "除外" if kind in _EXCLUDED_KINDS else "判定"
        detail.append(f"{kind:<28}{judged:<6}{st['n']:>7}{st['max_jump']:>14.5f}"
                      f"{st['n_over']:>10}")
    detail.append("")
    if n_trans == 0:
        detail.append("🔴 区画遷移が 1 度も起きていない。検査が空振りしている（台本を見直すこと）")
    if violations:
        detail.append(f"🔴 除外対象外での閾値超過 {n_over_judged} 件（先頭 {len(violations)} 件）:")
        for v in violations:
            detail.append(
                f"   面{v['seed']} {v['script']} step{v['step']} [{v['kind']}] "
                f"|ΔΦ|={v['dphi']:.5f} > 閾値 {v['thr']:.5f} "
                f"区画 {v['cell_from']}→{v['cell_to']} "
                f"v=({v['v_prev']:.3f},{v['v_now']:.3f}) "
                f"xy=({v['xy_from'][0]:.4f},{v['xy_from'][1]:.4f})→"
                f"({v['xy_to'][0]:.4f},{v['xy_to'][1]:.4f})")
    return ok, detail


def test_b_stair_detects_jump():
    """同じ走行を階段版で検査し、0.18 に近い跳びが実際に出ることを確認する。

    これが出なければ (b) の検査自体が連続化を検出できていない証拠になる。
    """
    stats, n_steps, n_trans, _ = _scan_jumps(continuous=False)
    trans_max = max((st["max_jump"] for k, st in stats.items() if k != KIND_IN_CELL),
                    default=0.0)
    in_cell_max = stats.get(KIND_IN_CELL, dict(max_jump=0.0))["max_jump"]
    n_over_all = sum(st["n_over"] for st in stats.values())
    ok = (n_trans > 0) and (trans_max >= 0.17)
    detail = [
        f"階段版の同一走行: 判定ステップ {n_steps}、区画遷移 {n_trans} 回",
        f"区画遷移ステップの最大 |ΔΦ| = {trans_max:.5f} m（cs = {CS} m に近いこと）",
        f"区画内ステップの最大 |ΔΦ| = {in_cell_max:.3e} m（階段版なので 0 のはず）",
        f"閾値を超えたステップ（種別問わず）= {n_over_all} 件"
        f"（＝この検査が跳びを検出できている証拠）",
    ]
    return ok, detail


# ======================================================================
# (b-2) 降下方向が曲がる区画での境界一致（裁定 R8 の回帰検査）
# ======================================================================
def test_b2_bend_boundary():
    max_diff = 0.0
    max_at = None
    n_conf = 0
    n_bend = 0
    n_skip_goal_side = 0
    bend_cells = set()
    for seed in VALID_SEEDS:
        env = geom_env(seed)
        for cx in range(SIZE):
            for cy in range(SIZE):
                c0 = (cx, cy)
                if c0 in GOAL_CELLS:
                    continue
                n = env._descending_neighbor(c0)
                w_out = env._edge_midpoint(c0, n)
                if env._dist_map[n] == 0:
                    # n がゴール区画だと remaining=0 の特別扱い（条文 1 と同じ理由）で
                    # 境界は一致しない。これは仕様であり検査対象外。件数を報告する。
                    n_skip_goal_side += 1
                    continue
                for prev in prev_cell_candidates(env, c0):
                    w_in = w_out if prev is None else env._edge_midpoint(prev, c0)
                    bend_c0 = is_bend(w_in, env._cell_center(c0), w_out)
                    # n 側の折れ線は [w_out(c0), center(n), w_out(n)]
                    n2 = env._descending_neighbor(n)
                    bend_n = is_bend(w_out, env._cell_center(n), env._edge_midpoint(n, n2))
                    if bend_c0 or bend_n:
                        n_bend += 1
                        bend_cells.add((seed, c0 if bend_c0 else n))
                    n_conf += 1
                    phi_from_c0 = env._potential_continuous(c0, prev, *w_out)
                    phi_from_n = env._potential_continuous(n, c0, *w_out)
                    diff = abs(phi_from_c0 - phi_from_n)
                    if diff > max_diff:
                        max_diff, max_at = diff, (seed, c0, n, prev, bend_c0, bend_n)
    ok = (max_diff < TOL_BOUNDARY) and (n_bend > 0)
    detail = [
        f"検査した境界の組 (面, c₀, c_prev): {n_conf} 通り / {len(VALID_SEEDS)} 面",
        f"うち**降下方向が曲がる**構成: {n_bend} 通り（曲がる区画 {len(bend_cells)} 個）",
        f"最大 |Φ(c₀ 側) − Φ(n 側)| = {max_diff:.3e} m（許容 {TOL_BOUNDARY:.0e}）",
        f"最大値の位置 (面, c₀, n, c_prev, bend_c₀, bend_n): {max_at}",
        f"n がゴール区画のため対象外にした c₀: {n_skip_goal_side} 個"
        f"（remaining=0 の特別扱い。条文 1 と同じ理由）",
    ]
    if n_bend == 0:
        detail.append("🔴 曲がる区画が 0 個 = 検査が空振りしている（回帰検査として無意味）")
    return ok, detail


# ======================================================================
# (b-3) step() 経路と直接呼び出しの一致（裁定 R12 の確認事項）
# ======================================================================
def test_b3_step_path_consistency():
    """`step()` 内での `_prev_cell` の更新タイミングが仕様どおりかを検査する。

    仕様（design.md §4 / maze6_env.py の docstring）:
      - reset 直後: c_prev = None、c = start
      - step: 区画が変わったら c_prev ← 直前の区画、c ← 新しい区画。
        **その更新の後**の (c, c_prev) で Φ を計算する
      - 同じ区画に留まっている間は c_prev を更新しない

    テスト側で同じ規則を並走させ（shadow）、`_potential_continuous` を直接呼んだ値と
    `step()` が使った値（`env._prev_potential`）が全ステップで一致することを見る。
    """
    max_diff = 0.0
    max_at = None
    n_steps = 0
    n_prev_updates = 0
    for seed in VALID_SEEDS:
        for sname, script in SCRIPTS:
            env = run_env(seed, continuous=True, reset_seed=0)
            # --- reset 直後 ---
            shadow_cell = tuple(env.maze["start"])
            shadow_prev = None
            x, y, _ = env.sim.privileged_pose()
            direct = env._potential_continuous(shadow_cell, shadow_prev, x, y)
            diff = abs(direct - float(env._prev_potential))
            n_steps += 1
            if diff > max_diff:
                max_diff, max_at = diff, (seed, sname, "reset")
            # --- step 経路 ---
            for t in range(MAX_SCRIPT_STEPS):
                _o, _r, terminated, truncated, info = env.step(script(t))
                cell = tuple(info["cell"])
                if cell != shadow_cell:
                    shadow_prev = shadow_cell
                    shadow_cell = cell
                    n_prev_updates += 1
                x, y, _ = env.sim.privileged_pose()
                direct = env._potential_continuous(shadow_cell, shadow_prev, x, y)
                diff = abs(direct - float(env._prev_potential))
                n_steps += 1
                if diff > max_diff:
                    max_diff, max_at = diff, (seed, sname, t, shadow_cell, shadow_prev)
                if terminated or truncated:
                    break
            env.close()
    ok = (max_diff < TOL_EXACT) and (n_prev_updates > 0)
    detail = [
        f"検査したステップ: {n_steps}（{len(VALID_SEEDS)} 面 × 台本 {len(SCRIPTS)} 種、"
        f"reset 直後を含む）",
        f"c_prev が更新された回数: {n_prev_updates}"
        f"（0 なら区画遷移が起きておらず検査が空振り）",
        f"最大 |Φ(step 経由) − Φ(直接呼び出し)| = {max_diff:.3e} m（許容 {TOL_EXACT:.0e}）",
        f"最大値の位置: {max_at}",
    ]
    if n_prev_updates == 0:
        detail.append("🔴 区画遷移が 1 度も起きていない = 検査が空振りしている")
    return ok, detail


# ======================================================================
# (c) 横方向のずれに不感（折れ線が一直線の区画に限る）
# ======================================================================
def test_c_lateral_invariance():
    offsets = (-0.020, -0.010, 0.010, 0.020)   # [m]
    max_diff = 0.0
    max_at = None
    n_checked = 0
    n_skipped_bend = 0
    for seed in VALID_SEEDS:
        env = geom_env(seed)
        for cx in range(SIZE):
            for cy in range(SIZE):
                cell = (cx, cy)
                if cell in GOAL_CELLS:
                    continue
                center = env._cell_center(cell)
                n = env._descending_neighbor(cell)
                w_out = env._edge_midpoint(cell, n)
                for prev in prev_cell_candidates(env, cell):
                    w_in = w_out if prev is None else env._edge_midpoint(prev, cell)
                    if is_bend(w_in, center, w_out):
                        n_skipped_bend += 1        # 折れ点付近はクランプで変わりうる
                        continue
                    ux, uy = w_out[0] - center[0], w_out[1] - center[1]
                    norm = math.hypot(ux, uy)
                    px, py = -uy / norm, ux / norm      # 進行方向に垂直な単位ベクトル
                    base = env._potential_continuous(cell, prev, *center)
                    for off in offsets:
                        phi = env._potential_continuous(
                            cell, prev, center[0] + off * px, center[1] + off * py)
                        diff = abs(phi - base)
                        n_checked += 1
                        if diff > max_diff:
                            max_diff, max_at = diff, (seed, cell, prev, off)
    ok = (max_diff < TOL_EXACT) and (n_checked > 0)
    detail = [
        f"横ずれ {offsets} m を与えた検査: {n_checked} 通り",
        f"折れ線が一直線でないため飛ばした構成: {n_skipped_bend} 通り",
        f"最大 |ΔΦ| = {max_diff:.3e} m（許容 {TOL_EXACT:.0e}）",
        f"最大値の位置 (面, 区画, c_prev, 横ずれ[m]): {max_at}",
    ]
    return ok, detail


# ======================================================================
# (d) Φ₀ = 0（reset 直後、擾乱あり）
# ======================================================================
def test_d_phi_zero_at_reset():
    n_reset_per_maze = 5
    max_abs = 0.0
    max_at = None
    offsets = []
    for seed in VALID_SEEDS:
        for k in range(n_reset_per_maze):
            env = run_env(seed, continuous=True, reset_seed=1000 + k)
            phi0 = abs(float(env._prev_potential))
            # 擾乱が実際に入っていることを確認する（区画中心・規定方位のままなら
            # このテストは何も検査していないことになる）
            x, y, _ = env.sim.privileged_pose()
            cx, cy = env._cell_center(env._cell)
            offsets.append(math.hypot(x - cx, y - cy))
            env.close()
            if phi0 > max_abs:
                max_abs, max_at = phi0, (seed, 1000 + k, offsets[-1])
    n = len(offsets)
    # 擾乱が実際に効いていること（≒ ±20 mm の端まで使われていること）を確認する。
    # ここが 0 に近いままだと (d) は何も検査していないことになる。
    ok = (max_abs < TOL_BOUNDARY) and (max(offsets) > 0.015)
    detail = [
        f"reset 回数: {n}（{len(VALID_SEEDS)} 面 × {n_reset_per_maze} 通りの擾乱）",
        f"最大 |Φ₀| = {max_abs:.3e} m（許容 {TOL_BOUNDARY:.0e}）",
        f"最大値の位置 (面, reset seed, 区画中心からのずれ[m]): {max_at}",
        f"擾乱の実効範囲（区画中心からのずれ）: 最小 {min(offsets) * 1000:.2f} mm / "
        f"最大 {max(offsets) * 1000:.2f} mm / 平均 {sum(offsets) / n * 1000:.2f} mm"
        f"（規定は横 ±20 mm・方位 ±10°）",
    ]
    if max(offsets) <= 0.015:
        detail.append("🔴 擾乱が小さすぎる = (d) が実質的に無擾乱の検査になっている")
    return ok, detail


# ======================================================================
# (e) 予約 seed の読み飛ばし ＋ 廊下側テスト
# ======================================================================
def test_e_reserved_seed_skip():
    hits = []
    checked = 0
    seqs = {}
    for base in (5990, 6990, 6000, 7000):
        env = Maze6Env(mode="generate", base_seed=base, continuous_potential=True)
        seq = [env._next_maze_seed() for _ in range(60)]
        seqs[base] = seq
        checked += len(seq)
        hits += [s for s in seq if s in _RESERVED_MAZE_SEEDS]
        if len(seq) != len(set(seq)):
            hits.append(f"重複あり(base={base})")
    ok = len(hits) == 0
    detail = [
        f"検査した学習 seed: {checked} 個（base_seed = 5990 / 6990 / 6000 / 7000）",
        f"予約帯（6000-6019, 7000-7019）に踏み込んだ数: {len(hits)}",
        f"base=5990 の先頭 15 個: {seqs[5990][:15]}",
        f"base=6990 の先頭 15 個: {seqs[6990][:15]}",
        f"base=6000 の先頭 5 個: {seqs[6000][:5]}（6020 から始まること）",
    ]
    return ok, detail


def test_e_corridor_suite():
    """廊下側 tests/test_corridor.py を別プロセスで走らせ、全項目 PASS を確認する。"""
    t0 = time.time()
    proc = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "tests", "test_corridor.py")],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800)
    ok = proc.returncode == 0
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-8:]
    detail = [f"コマンド: {sys.executable} tests/test_corridor.py",
              f"終了コード {proc.returncode}（{time.time() - t0:.1f} s）"]
    detail += ["  | " + ln for ln in tail]
    if proc.returncode != 0 and proc.stderr.strip():
        detail += ["  ! " + ln for ln in proc.stderr.strip().splitlines()[-5:]]
    return ok, detail


# ======================================================================
# (f) 既定 continuous_potential=False で既存挙動が変わらない
# ======================================================================
def test_f_default_unchanged():
    """既定 False では (1) Φ が x, y, c_prev を一切見ず階段版そのものであること、
    (2) 報酬が階段版 Φ から独立に組み立てた式と bit 単位で一致すること、
    (3) 同一条件の 2 回の走行が bit 単位で一致すること を確認する。
    """
    problems = []

    # (1) x, y, c_prev を無視する
    max_diff = 0.0
    for seed in VALID_SEEDS[:5]:
        env = geom_env(seed, continuous=False)
        for cx in range(SIZE):
            for cy in range(SIZE):
                cell = (cx, cy)
                ref = env._potential_stair(cell)
                for prev in prev_cell_candidates(env, cell):
                    for (px, py) in ((0.0, 0.0), (1.234, -5.678), env._cell_center(cell)):
                        v = env._potential(cell, prev, px, py)
                        max_diff = max(max_diff, abs(v - ref))
    if max_diff != 0.0:
        problems.append(f"既定 False なのに x/y/c_prev で Φ が変わった（最大差 {max_diff:.3e}）")

    # (2) 報酬の独立な再構成（運転点 k=8.7e-3, α=0.5 で組む）
    gamma, k, alpha = 0.995, 8.7e-3, 0.5
    env = run_env(VALID_SEEDS[0], continuous=False, reset_seed=7,
                  gamma=gamma, action_highpass_penalty=k, action_highpass_alpha=alpha)
    phi_prev = env._potential_stair(env._cell)
    lowpass = np.zeros(2, dtype=np.float64)
    visited = set(env._visited)
    max_rew_diff = 0.0
    n_rew = 0
    rewards_a = []
    for t in range(300):
        a = np.clip(script_scurve(t), -1.0, 1.0)
        _o, r, terminated, truncated, info = env.step(a)
        cell = tuple(info["cell"])
        phi = env._potential_stair(cell)
        expect = gamma * phi - phi_prev - 0.001
        if info["goal"]:
            expect += 1.0
        elif info["collision"]:
            expect += -1.0
        if cell not in visited:
            visited.add(cell)
            expect += 0.02
        lowpass = alpha * lowpass + (1.0 - alpha) * a
        hp = a - lowpass
        expect -= k * float(np.dot(hp, hp))
        max_rew_diff = max(max_rew_diff, abs(r - expect))
        rewards_a.append(r)
        phi_prev = phi
        n_rew += 1
        if terminated or truncated:
            break
    env.close()
    if max_rew_diff >= TOL_EXACT:
        problems.append(f"報酬が階段版の式と一致しない（最大差 {max_rew_diff:.3e}, n={n_rew}）")

    # (3) 決定性（同一条件の 2 回目が bit 単位で一致）
    env = run_env(VALID_SEEDS[0], continuous=False, reset_seed=7,
                  gamma=gamma, action_highpass_penalty=k, action_highpass_alpha=alpha)
    rewards_b = []
    for t in range(300):
        _o, r, terminated, truncated, _i = env.step(np.clip(script_scurve(t), -1.0, 1.0))
        rewards_b.append(r)
        if terminated or truncated:
            break
    env.close()
    same = (len(rewards_a) == len(rewards_b)
            and all(x == y for x, y in zip(rewards_a, rewards_b)))
    if not same:
        problems.append("同一条件の 2 回の走行が bit 単位で一致しない")

    ok = not problems
    detail = [
        f"(1) x/y/c_prev を無視: 最大差 {max_diff:.3e}（0.0 であること）",
        f"(2) 報酬の独立再構成: {n_rew} ステップ、最大差 {max_rew_diff:.3e}（許容 {TOL_EXACT:.0e}）",
        f"(3) 決定性: 2 回の報酬列が bit 一致 = {same}（{len(rewards_a)} / {len(rewards_b)} ステップ）",
    ]
    detail += ["🔴 " + p for p in problems]
    return ok, detail


# ======================================================================
TESTS = [
    ("(a) 区画中心で階段版と一致", test_a_center_matches_stair),
    ("(b) 跳びが無い（連続版）", test_b_no_jump),
    ("(b) 階段版では跳びが出る（検査の有効性）", test_b_stair_detects_jump),
    ("(b-2) 曲がる区画での境界一致", test_b2_bend_boundary),
    ("(b-3) step() 経路と直接呼び出しの一致", test_b3_step_path_consistency),
    ("(c) 横方向のずれに不感", test_c_lateral_invariance),
    ("(d) Φ₀ = 0（擾乱あり）", test_d_phi_zero_at_reset),
    ("(e) 予約 seed の読み飛ばし", test_e_reserved_seed_skip),
    ("(f) 既定 False で既存挙動が不変", test_f_default_unchanged),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-corridor", action="store_true",
                    help="(e) の廊下側スイート（tests/test_corridor.py）を省略する")
    ap.add_argument("--only", default=None, help="名前に含まれる文字列で絞り込む")
    args = ap.parse_args(argv)

    tests = list(TESTS)
    if not args.skip_corridor:
        tests.append(("(e) 廊下側 tests/test_corridor.py 全項目", test_e_corridor_suite))
    if args.only:
        tests = [t for t in tests if args.only in t[0]]

    results = []
    for name, fn in tests:
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}", flush=True)
        t0 = time.time()
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001 — 1 項目の失敗で全体を止めない
            import traceback
            ok, detail = False, ["例外: " + repr(exc)] + traceback.format_exc().splitlines()[-6:]
        dt = time.time() - t0
        for ln in detail:
            print("  " + ln)
        print(f"  → {'PASS' if ok else 'FAIL'}（{dt:.1f} s）", flush=True)
        results.append((name, ok, dt))

    print(f"\n{'=' * 78}\n結果まとめ（exp_012 の学習投入条件: 全項目 PASS）\n{'=' * 78}")
    print(f"{'項目':<46}{'判定':<8}{'時間[s]':>9}")
    for name, ok, dt in results:
        print(f"{name:<46}{'PASS' if ok else 'FAIL':<8}{dt:>9.1f}")
    n_fail = sum(1 for _n, ok, _d in results if not ok)
    print(f"\n{len(results) - n_fail} / {len(results)} PASS"
          + ("" if n_fail == 0 else f"　🔴 FAIL {n_fail} 件 → **学習は投入しない**"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
