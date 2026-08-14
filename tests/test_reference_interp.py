"""
tests/test_reference_interp.py
==============================
参照の弧長内挿（`competition/reference_interp.py`・exp_016 段階 F0-b）の単体テスト。

**最重要の検証: 基準スナップショットが壊れていないこと。**
**既定パラメータ（`ref_interp = False`）では現行と 1 ビットも変わらない**ことを
反証形式で確認する。

| # | 何を反証するか | 偽ならどう見えるか |
|---|---|---|
| 1 | 既定では走行がビット単位で不変 | 電圧・車輪角速度・姿勢のどれかが 1 ビットでも違う |
| 2 | `ref_interp` が実際に配線されている | 有効にしても電圧が変わらない |
| 3 | **内挿がカード §2-1 の定義どおり**（$v^2$ の線形内挿・$s_\\text{proj}$ の式） | 独立に計算した値と一致しない |
| 4 | **格子点の上では内挿値が計画値と一致する**（内挿が計画を壊していない） | 格子点で値がずれる |
| 5 | **等減速区間で $v^2$ 内挿が厳密**（$v$ 内挿との違いが実在する） | 両者が同じ ＝ 選択に意味が無い |
| 6 | F0 の混ぜ込みと**併用できる**（メソッドが衝突しない） | 例外・どちらかが効かない |

実行: .venv/bin/python tests/test_reference_interp.py
"""
import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from competition.baseline_slalom import (A_LAT_MEASURED,  # noqa: E402
                                          A_MAX_MEASURED, SlalomPolicy,
                                          build_speed_profile)
from competition.reference_interp import ReferenceInterpMixin  # noqa: E402
from competition.velocity_loop import VelocityLoopMixin  # noqa: E402
from mouse.params import RobotParams  # noqa: E402
from mouse.sim import MouseSim  # noqa: E402

RESULTS = []
FACE = os.path.join(REPO_ROOT, "competition", "mazes", "design_v4", "maze_41003")


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    RESULTS.append((name, ok))
    return ok


class Interp(ReferenceInterpMixin, SlalomPolicy):
    pass


class Both(ReferenceInterpMixin, VelocityLoopMixin, SlalomPolicy):
    """F0（速度ループ）と F0-b（参照の内挿）の併用。**別のメソッドを差し替えている。**"""


def _drive(cls, n_ticks=1200, **kw):
    params = RobotParams()
    sim = MouseSim(FACE + ".xml", params=params)
    z = np.load(FACE + ".npz")
    sim.full_reset(cell=(0, 0), heading_deg=90)
    pol = cls(**kw)
    pol.bind_sim(sim)
    pol.bind_maze(z["v_walls"], z["h_walls"])
    pol.on_maze_start(dict(width=16, height=16))
    rec = []
    for _ in range(n_ticks):
        obs = sim.observation()
        vl, vr = pol.act(obs)
        x, y, yaw = sim.privileged_pose()
        rec.append((vl, vr, float(obs[pol._i_wheel]), float(obs[pol._i_wheel + 1]), x, y, yaw))
        sim.step_control(vl, vr)
    return np.array(rec, dtype=np.float64), pol


# ==========================================================================
def test1_default_bit_identical():
    print("\n[test1] ref_interp=False の混ぜ込みが、素の方策と 1 ビットも違わないか")
    base, _ = _drive(SlalomPolicy)
    mixed, _ = _drive(Interp, ref_interp=False)
    check("記録に NaN が無い", not np.isnan(base).any(), f"{base.shape[0]} ティック")
    check("電圧・車輪角速度・姿勢が全ティックでビット一致",
          base.shape == mixed.shape and np.array_equal(base, mixed),
          f"最大差 {np.max(np.abs(base - mixed)) if base.shape == mixed.shape else 'shape 不一致'}")


def test2_wired():
    print("\n[test2] ref_interp を有効にすると電圧が変わるか（＝配線されているか）")
    off, _ = _drive(Interp, ref_interp=False)
    on, _ = _drive(Interp, ref_interp=True)
    d = np.max(np.abs(off[:, :2] - on[:, :2]))
    check("ref_interp=True で電圧が変わる", d > 1e-6, f"電圧の最大差 {d:.6f} V")


def test3_definition_matches_card():
    """カード §2-1 の定義（s_proj の式・v² の線形内挿）と厳密に一致するか。"""
    print("\n[test3] 内挿がカード §2-1 の定義どおりか（独立に計算して照合）")
    params = RobotParams()
    sim = MouseSim(FACE + ".xml", params=params)
    z = np.load(FACE + ".npz")
    sim.full_reset(cell=(0, 0), heading_deg=90)
    pol = Interp(ref_interp=True)
    pol.bind_sim(sim)
    pol.bind_maze(z["v_walls"], z["h_walls"])
    pol.on_maze_start(dict(width=16, height=16))
    for _ in range(200):     # 経路が張られるまで進める
        pol.act(sim.observation())
        sim.step_control(0.0, 0.0)
        if pol._path is not None:
            break
    if pol._path is None:
        return check("経路が張られた", False)

    path, rng = pol._path, np.random.default_rng(20260814)
    worst = 0.0
    for _ in range(500):
        idx = int(rng.integers(0, len(path.s)))
        x = float(path.x[idx]) + float(rng.uniform(-0.02, 0.02))
        y = float(path.y[idx]) + float(rng.uniform(-0.02, 0.02))
        got = pol._speed_at(idx, x, y)
        # --- 独立実装（カード §2-1 の式をそのまま書き下す） ---
        psi = float(path.heading[idx])
        sp = (float(path.s[idx]) + (x - float(path.x[idx])) * math.cos(psi)
              + (y - float(path.y[idx])) * math.sin(psi))
        sp = min(max(sp, float(path.s[0])), float(path.s[-1]))
        want = math.sqrt(max(0.0, float(np.interp(sp, path.s, np.asarray(path.speed) ** 2))))
        worst = max(worst, abs(got - want))
    check("独立実装と一致（無作為 500 点）", worst < 1e-12, f"最大差 {worst:.3e}")

    # 格子点の上では計画値そのものに戻ること
    worst2 = 0.0
    for idx in range(0, len(path.s), 7):
        got = pol._speed_at(idx, float(path.x[idx]), float(path.y[idx]))
        worst2 = max(worst2, abs(got - float(path.speed[idx])))
    check("格子点の上では計画値と一致（内挿が計画を壊していない）", worst2 < 1e-9,
          f"最大差 {worst2:.3e}")


def test4_vsq_vs_v_interp_differs():
    """等減速区間で v² 内挿と v 内挿が実際に別物であること（選択に意味があること）。"""
    print("\n[test4] 等減速区間で v² 内挿と v 内挿が別物か（選択に意味があるか）")
    p = RobotParams()
    a_max = A_MAX_MEASURED * 0.7
    s = np.arange(0.0, 0.5, 0.02)                    # わざと粗い刻み
    curv = np.zeros_like(s)
    v = build_speed_profile(s, curv, 1.0, A_LAT_MEASURED * 0.7, a_max, 0.15, True)
    mid = (s[:-1] + s[1:]) / 2.0
    v_sq = np.sqrt(np.interp(mid, s, v ** 2))
    v_lin = np.interp(mid, s, v)
    # ⚠️ **母集団を絞る**（作法 9）。「v が落ちている区間」ではなく
    #    「**a_max 律速がちょうど効いている区間**」だけを残す。落ち始めの 1 区間は
    #    上限 v_ceil で頭打ちになっており、a_max の等式が成り立たない
    ds_seg = np.diff(s)
    tight = np.abs(v[:-1] ** 2 - (v[1:] ** 2 + 2.0 * a_max * ds_seg)) < 1e-9
    dec = (np.diff(v) < -1e-6) & tight
    if not dec.any():
        return check("a_max 律速の減速区間が存在する", False)
    d = float(np.max(np.abs(v_sq[dec] - v_lin[dec])))
    check("v² 内挿と v 内挿は別物", d > 1e-4, f"最大差 {d*1000:.3f} mm/s（n={int(dec.sum())} 区間）")
    # v² 内挿は等減速区間で厳密（真値 = sqrt(v_next² + 2·a_max·Δs)）
    err, err_lin = [], []
    for i in np.where(dec)[0]:
        ds_half = float(s[i + 1] - mid[i])
        truth = math.sqrt(max(0.0, v[i + 1] ** 2 + 2.0 * a_max * ds_half))
        err.append(abs(v_sq[i] - truth))
        err_lin.append(abs(v_lin[i] - truth))
    check("v² 内挿は a_max 律速の区間で厳密", max(err) < 1e-9,
          f"最大誤差 {max(err):.3e}（同じ点での v 内挿の誤差 {max(err_lin):.3e}）")
    del p


def test5_composes_with_f0():
    """F0 の混ぜ込みと併用できること（差し替えるメソッドが違う）。"""
    print("\n[test5] F0（速度ループ）と併用できるか")
    a, _ = _drive(Both, ref_interp=False, k_acc_ff=0.0)
    base, _ = _drive(SlalomPolicy)
    check("両方 0 なら素の方策とビット一致",
          a.shape == base.shape and np.array_equal(a, base),
          f"最大差 {np.max(np.abs(a - base)) if a.shape == base.shape else 'shape 不一致'}")
    b, _ = _drive(Both, ref_interp=True, k_acc_ff=1.0)
    check("両方有効でも走り切る（例外なし・NaN なし）",
          (not np.isnan(b).any()) and b.shape[0] == base.shape[0],
          f"{b.shape[0]} ティック")


# ==========================================================================
def main():
    print("=" * 78)
    print("reference_interp（F0-b 参照の弧長内挿）単体テスト")
    print("=" * 78)
    for fn in (test1_default_bit_identical, test2_wired, test3_definition_matches_card,
               test4_vsq_vs_v_interp_differs, test5_composes_with_f0):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] {fn.__name__}: {type(e).__name__}: {e}")
            RESULTS.append((fn.__name__, False))
    n_ok = sum(1 for _, ok in RESULTS if ok)
    print("\n" + "=" * 78)
    print(f"合計: {n_ok}/{len(RESULTS)} PASS")
    print("=" * 78)
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
