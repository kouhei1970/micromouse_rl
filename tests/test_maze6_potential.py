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
  (b)   跳びが無いこと。閾値は式で定義する（裁定 R7-6・R18-2・R21-E）:
            |ΔΦ_t| ≤ 2.7972 · 1.10 · |Δp_t| + 1e-9 m
        |Δp| は真の位置から求めた機体中心の 1 ステップ変位。直進・90°旋回・S 字・急停止の
        4 種で全ステップを検査。同じ走行を階段版でも検査し、そちらでは 0.18 に近い跳びが
        出ることを確認する（＝この検査が連続化を検出できることの証拠。階段版の比は 8.08
        以上あるので、定数を 2.7972 にしても検出力は残る）
  (L)   刻み不変性（裁定 R18-4 で新設）。区画内を 3 水準の格子で走査し、最大
        |ΔΦ|/|Δp| が**定義の Lipschitz 定数**（条件 E は √2）近傍で一定であること。
        真の不連続なら刻み数に比例して発散する。**(b) の 2.7972 とは用途が違う定数**
  (b-2) **全**降下開口部での境界一致（裁定 R8・R13 条件 3。D1/D2 の回帰検査）
  (b-3) step() 経路と直接呼び出しの全ステップ一致（裁定 R12 の確認事項）
  (c)   Φ が横方向のずれに不感（**降下隣接がちょうど 1 つ、かつ折れ線が一直線**の構成に
        限る。裁定 R18-3。飛ばした構成の数を報告）
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

# (b) の閾値（design.md §4 (b) の式。**緩めてはならない**）
#   |ΔΦ_t| ≤ L · κ · |Δp_t| + ε
# κ は「区画をまたぐ歩で境界の交点を経由して評価する」ぶんの余裕で、1 歩の弧長／弦長
# (θ/2)/sin(θ/2) を押さえる。物理的な最大は両輪を逆向きに全電圧で回した
# ω_max = 2·v_max/tread で θ_max = ω_max·Δt = 1.136 rad → 1.0553。κ=1.10 は 4% の余裕。
# ε は数値誤差ぶん（Φ は O(1 m) の倍精度和なので実際の誤差は 1e-15 程度）。
# 🔴 **2 つの定数を混同しないこと**（用途が違う）:
#   JUMP_L_E   … (b) 用。**区画をまたぐ歩**を含めた実効の上限。条件 E は境界一致が
#                開口部の中点でしか成立しないので、中点から外れて通ると差が出る。
#                辺全体での一致を課したときの下限。無次元形は（裁定 R24-3）
#                    L_E = s / (√2·(t_w/2 + w_r)) = 1/(√2·ŵ)、ŵ = (t_w/2 + w_r)/s
#                s=区画寸法 0.18、t_w=壁厚 0.012（mouse/mjcf.py）、w_r=機体最外半径 0.0395。
#                上限に到達する 2 点対は「進入辺・退出辺のそれぞれで、2 辺が共有する
#                内側の角へ最も寄れる点」（中点から ±μ_max、μ_max=(s−t_w)/2−w_r=0.0445）。
#                🔴 初出時に 3.2223 と書いたのは t_w/2=6mm を落とした誤り。是正は
#                定数を小さくする＝**検査を厳しくする**向き。裁定 R21-E・R24-3
#   LIPSCHITZ_DEF … (L) 用。**同一区画・同一 c_prev の中**での定義そのものの
#                Lipschitz 定数。条件 E では折れの内側で ∇ℓ = a − b（a ⊥ b・単位）なので √2
JUMP_L_E = 0.18 / (math.sqrt(2.0) * (0.012 / 2 + 0.0395))   # = 2.7972
LIPSCHITZ_DEF = math.sqrt(2.0)
JUMP_KAPPA = 1.10
JUMP_EPS = 1e-9                           # [m]
DISP_ABS_CAP = VMAX * DT                  # [m] |Δp| 自体の物理的な上限（照合用）

# (L) 刻み不変性検査の刻み数（区画一辺 0.18 m を N 分割）と、許容する最大比
GRID_LEVELS = (30, 120, 480)              # → 6.0 / 1.5 / 0.375 mm
GRID_TOL = 1e-6                           # 最大比が定義の Lipschitz 定数を超えてよい幅

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


def jump_threshold(disp: float) -> float:
    """design.md §4 (b) の閾値 [m]。**機体中心の実変位 |Δp| から直接決める。**

    旧式は車輪角速度から求めた前進速度 v を使っていたが、v は惰行・制動時に実変位を
    最大 2.414 倍まで過小評価する（実測。10311 歩）。v 基準を保つには余裕係数 2.5 が
    必要で検査が 2.4 倍甘くなるため、真の位置から |Δp| を直接測る形へ改めた。
    """
    return JUMP_L_E * JUMP_KAPPA * disp + JUMP_EPS


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


def script_brake(t: int) -> np.ndarray:
    """40 歩走ってから指令を 0 にして惰行・停止する（低速域を通す台本）。

    exp_010 が落ち込んだ停止層（|v| < 0.05 m/s）は上の 3 台本ではほとんど通らない
    （実測 2286 歩中 1 歩）。惰行中は車輪角速度が実変位を過小評価するので、
    閾値を |Δp| 基準へ改めたことの妥当性もここで効く。
    """
    return (np.array([0.6, 0.6], dtype=np.float64) if t < 40
            else np.array([0.0, 0.0], dtype=np.float64))


SCRIPTS = [("直進", script_straight), ("90°旋回", script_turn90),
           ("S字", script_scurve), ("急停止", script_brake)]


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


def descending_neighbors(env: Maze6Env, cell):
    """降下隣接の**全一覧**（d が cell よりちょうど 1 小さい開通隣接）。

    環境側の実装に依存せずテスト側で独立に列挙する（実装の tie-break の有無に
    関わらず、すべての降下開口部を検査対象にするため。裁定 R13 条件 3）。
    """
    d0 = env._dist_map[cell]
    return [nb for nb in env._open_neighbors(cell)
            if env._dist_map.get(nb, -1) == d0 - 1]


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
    n_disp_over_cap = 0     # |Δp| が物理上限 v_max·Δt を超えた歩（超えたら物理側の異常）
    max_disp = 0.0
    for seed in VALID_SEEDS:
        for sname, script in SCRIPTS:
            recs, env = rollout(seed, script, continuous)
            env.close()
            for t in range(1, len(recs)):
                a, b = recs[t - 1], recs[t]
                kind = classify_step(a["cell"], b["cell"], a["d"], b["d"], b["goal"])
                dphi = abs(b["phi"] - a["phi"])
                disp = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
                max_disp = max(max_disp, disp)
                if disp > DISP_ABS_CAP:
                    n_disp_over_cap += 1
                thr = jump_threshold(disp)
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
                                               dphi=dphi, thr=thr, disp=disp,
                                               cell_from=a["cell"], cell_to=b["cell"],
                                               ratio=dphi / disp if disp > 0 else float("inf"),
                                               xy_from=(a["x"], a["y"]),
                                               xy_to=(b["x"], b["y"])))
    return stats, n_steps, n_transitions, violations, n_disp_over_cap, max_disp


def test_b_no_jump():
    stats, n_steps, n_trans, violations, n_cap, max_disp = _scan_jumps(continuous=True)
    n_over_judged = sum(v["n_over"] for k, v in stats.items() if k not in _EXCLUDED_KINDS)
    ok = (n_over_judged == 0) and (n_trans > 0) and (n_cap == 0)

    detail = [
        f"走行: {len(VALID_SEEDS)} 面 × 台本 {len(SCRIPTS)} 種、判定したステップ {n_steps}、"
        f"区画遷移 {n_trans} 回",
        f"閾値: {JUMP_L_E} · {JUMP_KAPPA} · |Δp| + {JUMP_EPS:.0e} m"
        f"（L_E = s/(√2·(t_w/2+w_r)) = {JUMP_L_E:.4f}。辺全体一致と機体到達域から。裁定 R24-3）",
        f"|Δp| の最大 = {max_disp:.5f} m（物理上限 v_max·Δt = {DISP_ABS_CAP:.4f} m、"
        f"超過 {n_cap} 歩。v_max = {VMAX:.3f} m/s）",
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
    if n_cap:
        detail.append(f"🔴 |Δp| が物理上限 {DISP_ABS_CAP:.4f} m を超えた歩が {n_cap} 件ある"
                      f"（Φ ではなく物理側の異常）")
    if violations:
        detail.append(f"🔴 除外対象外での閾値超過 {n_over_judged} 件（先頭 {len(violations)} 件）:")
        for v in violations:
            detail.append(
                f"   面{v['seed']} {v['script']} step{v['step']} [{v['kind']}] "
                f"|ΔΦ|={v['dphi']:.5f} > 閾値 {v['thr']:.5f} "
                f"（|Δp|={v['disp']:.5f}、比 {v['ratio']:.3f}／許容 "
                f"{JUMP_L_E * JUMP_KAPPA:.3f}）区画 {v['cell_from']}→{v['cell_to']} "
                f"xy=({v['xy_from'][0]:.4f},{v['xy_from'][1]:.4f})→"
                f"({v['xy_to'][0]:.4f},{v['xy_to'][1]:.4f})")
    return ok, detail


def test_b_stair_detects_jump():
    """同じ走行を階段版で検査し、0.18 に近い跳びが実際に出ることを確認する。

    これが出なければ (b) の検査自体が連続化を検出できていない証拠になる。
    """
    stats, n_steps, n_trans, _v, _c, _m = _scan_jumps(continuous=False)
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
    """**すべての**降下開口部で境界の値が一致するか（裁定 R13 条件 3 で拡張）。

    旧版は tie-break が選んだ n との境界だけを見ていたため、D2（tie-break が選ばな
    かった開口部を通ると Φ が跳ぶ）を検出できなかった。全降下隣接を対象にする。

    【条文】除外は 1 種のみ（裁定 R18-3 で明文化）:
      n がゴール区画（d(n)=0）である c₀ は対象外。remaining=0 と特別扱いするため、
      (b) の条文 1 と同じ理由で境界は一致しない。**件数を必ず報告する。**
    """
    max_diff = 0.0
    max_at = None
    n_conf = 0
    n_bend = 0
    n_skip_goal_side = 0
    n_multi_desc = 0
    bend_cells = set()
    for seed in VALID_SEEDS:
        env = geom_env(seed)
        for cx in range(SIZE):
            for cy in range(SIZE):
                c0 = (cx, cy)
                if c0 in GOAL_CELLS:
                    continue
                desc = descending_neighbors(env, c0)
                if len(desc) >= 2:
                    n_multi_desc += 1
                for n in desc:                       # ← tie-break 分だけでなく全部
                    w_out = env._edge_midpoint(c0, n)
                    if env._dist_map[n] == 0:
                        n_skip_goal_side += 1        # 条文（上記）による唯一の除外
                        continue
                    for prev in prev_cell_candidates(env, c0):
                        w_in = w_out if prev is None else env._edge_midpoint(prev, c0)
                        bend_c0 = is_bend(w_in, env._cell_center(c0), w_out)
                        # n 側の折れ線は [w_out(c0), center(n), w_out(n)]。n の降下隣接も
                        # 複数ありうるので、どれか 1 つでも曲がれば「曲がる構成」と数える
                        bend_n = any(
                            is_bend(w_out, env._cell_center(n), env._edge_midpoint(n, m))
                            for m in descending_neighbors(env, n))
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
        f"検査した境界の組 (面, c₀, n, c_prev): {n_conf} 通り / {len(VALID_SEEDS)} 面"
        f"（**全降下開口部**。tie-break 選択分だけではない）",
        f"うち**降下方向が曲がる**構成: {n_bend} 通り（曲がる区画 {len(bend_cells)} 個）",
        f"降下隣接が 2 つ以上ある区画: {n_multi_desc} 個"
        f"（D2 が起きうる区画。0 なら D2 の回帰検査として空振り）",
        f"最大 |Φ(c₀ 側) − Φ(n 側)| = {max_diff:.3e} m（許容 {TOL_BOUNDARY:.0e}）",
        f"最大値の位置 (面, c₀, n, c_prev, bend_c₀, bend_n): {max_at}",
        f"【条文による除外】n がゴール区画のため対象外にした (c₀, n): {n_skip_goal_side} 組"
        f"（remaining=0 の特別扱い。(b) の条文 1 と同じ理由）",
    ]
    if n_bend == 0:
        detail.append("🔴 曲がる区画が 0 個 = 検査が空振りしている（回帰検査として無意味）")
    if n_multi_desc == 0:
        detail.append("🔴 降下隣接が 2 つ以上ある区画が 0 個 = D2 の回帰検査が空振り")
    return ok, detail


# ======================================================================
# (L) 刻み不変性検査（裁定 R18-4 で新設。D1 型＝真の不連続の回帰検査）
# ======================================================================
def test_L_grid_invariance():
    """区画内を格子走査し、刻みを変えても最大 |ΔΦ|/|Δp| が √2 近傍で一定かを見る。

    真の不連続があれば最大比は刻み数に比例して発散する（旧実装では刻み 30/120/480 に
    対し比 30/120/480 だった）。走行台本に依存しないのが (b) に対する利点。

    ⚠️ **軸方向だけを見ると折れ区画の最大を取り逃がす**（実際に取り逃がした）。
       斜め 2 方向を必ず含めること。
    """
    # 最も細かい刻みは 481×481 点になるので、全区画を舐めると単体テストとして重すぎる。
    # **決定的に選んだ標本**に限り、代わりに 3 つの型（折れ／直線／退化）を必ず含める。
    # 型が 1 つでも欠けたら検査が空振りなので FAIL にする。
    configs = []            # (seed, cell, prev, 型)
    per_kind = {"折れ": [], "直線": [], "退化": [], "ゴール": []}
    for seed in VALID_SEEDS[:5]:
        env = geom_env(seed)
        for cx in range(SIZE):
            for cy in range(SIZE):
                cell = (cx, cy)
                if cell in GOAL_CELLS:
                    # 裁定 R21: ゴール 2x2 も走査対象に含める。d=0 で remaining=0 と
                    # 短絡するので Φ は一様のはずで、最大比 0 が期待値。0 でなければ
                    # 短絡の実装に欠陥がある。
                    for prev in prev_cell_candidates(env, cell):
                        if len(per_kind["ゴール"]) < 4:
                            per_kind["ゴール"].append((seed, cell, prev, "ゴール"))
                    continue
                C = env._cell_center(cell)
                for prev in prev_cell_candidates(env, cell):
                    for n in descending_neighbors(env, cell):
                        w_out = env._edge_midpoint(cell, n)
                        w_in = w_out if prev is None else env._edge_midpoint(prev, cell)
                        kind = ("退化" if w_in == w_out
                                else "折れ" if is_bend(w_in, C, w_out) else "直線")
                        if len(per_kind[kind]) < 8:
                            per_kind[kind].append((seed, cell, prev, kind))
                        break
    for kind in ("折れ", "直線", "退化", "ゴール"):
        configs += per_kind[kind]

    envs = {s: geom_env(s) for s in {c[0] for c in configs}}
    rows = []
    ok = all(len(per_kind[k]) > 0 for k in per_kind)
    for N in GRID_LEVELS:
        worst, worst_at = 0.0, None
        step = CS / N
        diag = step * math.sqrt(2.0)
        for (seed, cell, prev, kind) in configs:
            env = envs[seed]
            C = env._cell_center(cell)
            x0, y0 = C[0] - CS / 2, C[1] - CS / 2
            g = [[env._potential_continuous(cell, prev, x0 + i * step, y0 + j * step)
                  for j in range(N + 1)] for i in range(N + 1)]
            for i in range(N + 1):
                for j in range(N + 1):
                    cand = []
                    if i < N:
                        cand.append((abs(g[i + 1][j] - g[i][j]) / step, "軸"))
                    if j < N:
                        cand.append((abs(g[i][j + 1] - g[i][j]) / step, "軸"))
                    if i < N and j < N:
                        cand.append((abs(g[i + 1][j + 1] - g[i][j]) / diag, "斜め"))
                        cand.append((abs(g[i + 1][j] - g[i][j + 1]) / diag, "斜め"))
                    for r, d in cand:
                        if r > worst:
                            worst, worst_at = r, (seed, cell, prev, kind, d)
        rows.append((N, step * 1000, worst, worst_at))
        if worst > LIPSCHITZ_DEF + GRID_TOL:
            ok = False
    detail = [
        f"標本 {len(configs)} 構成（折れ {len(per_kind['折れ'])} / 直線 "
        f"{len(per_kind['直線'])} / 退化 {len(per_kind['退化'])} / "
        f"ゴール2x2 {len(per_kind['ゴール'])}）× 刻み 3 水準。"
        f"許容（定義の Lipschitz 定数）= {LIPSCHITZ_DEF:.6f} + {GRID_TOL:.0e}",
        f"{'刻み':>6}{'1 刻み[mm]':>12}{'最大 |ΔΦ|/|Δp|':>18}   最大値の位置",
    ]
    for N, mm, worst, at in rows:
        detail.append(f"{N:>6}{mm:>12.3f}{worst:>18.6f}   {at}")
    spread = max(r[2] for r in rows) - min(r[2] for r in rows)
    detail.append(f"3 水準の最大比のばらつき = {spread:.3e}"
                  f"（刻みに比例して増えていれば真の不連続。旧実装では 30 → 120 → 480 だった）")
    for kind in ("折れ", "直線", "退化", "ゴール"):
        if not per_kind[kind]:
            detail.append(f"🔴 型「{kind}」の標本が 0 件 = 検査が空振りしている")
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
                desc = descending_neighbors(env, cell)
                # 🔴 対象の限定（裁定 R18-3）: 降下隣接が直角に 2 つある区画では、
                #    横へずれることが本当にゴールへ近づくことなので Φ が動くのが正しい
                #    挙動である（全降下隣接の min を採るため）。よって「降下隣接が
                #    ちょうど 1 つ、かつ折れ線が一直線」の構成に限る。
                if len(desc) != 1:
                    n_skipped_bend += len(prev_cell_candidates(env, cell))
                    continue
                w_out = env._edge_midpoint(cell, desc[0])
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
        f"横ずれ {offsets} m を与えた検査: {n_checked} 通り"
        f"（対象は「降下隣接がちょうど 1 つ、かつ折れ線が一直線」の構成に限る）",
        f"降下隣接が 2 つ以上／折れ線が一直線でないため飛ばした構成: {n_skipped_bend} 通り",
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
    ("(L) 刻み不変性（真の不連続の回帰検査）", test_L_grid_invariance),
    ("(b-2) 全降下開口部での境界一致", test_b2_bend_boundary),
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
