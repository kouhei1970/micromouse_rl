"""
tests/test_velocity_loop.py
===========================
速度ループの是正（`competition/velocity_loop.py`・exp_016 段階 F0）の単体テスト。

**最重要の検証: 基準スナップショットが壊れていないこと。**
速度ループは**全方策が共有する最内ループ**なので、
**既定パラメータ（`k_acc_ff = 0`）では現行と 1 ビットも変わらない**ことを、
反証形式で確認する（1 ビットでも違えば `baseline_slalom.py` を凍結した意味が消える）。

| # | 何を反証するか | 偽ならどう見えるか |
|---|---|---|
| 1 | `WheelPIAccelFF` の写しが親からずれていない | `v_ff_extra=0` で親と 1 つでも値が違う |
| 2 | 既定では走行がビット単位で不変 | 軌跡・電圧・車輪角速度のどれかが 1 ビットでも違う |
| 3 | `k_acc_ff` が実際に効いている | 係数を上げても電圧が変わらない（＝配線されていない） |
| 4 | `J_eff` が基準文書のパラメータから導出されている | ハードコード値と一致する／モデルを変えても動かない |

実行: .venv/bin/python tests/test_velocity_loop.py
"""
import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT,
          os.path.join(REPO_ROOT, "experiments", "exp_016_diagonal"),
          os.path.join(REPO_ROOT, "experiments", "exp_015_time_optimal_route")):
    if p not in sys.path:
        sys.path.insert(0, p)

from competition.baseline_slalom import WheelPI  # noqa: E402
from competition.velocity_loop import (VelocityLoopMixin,  # noqa: E402
                                        WheelPIAccelFF, subtree_mass)
from mouse.params import RobotParams  # noqa: E402
from mouse.sim import MouseSim  # noqa: E402

RESULTS = []
DESIGN_FACE = os.path.join(REPO_ROOT, "competition", "mazes", "design_v4", "maze_41003")


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    RESULTS.append((name, ok))
    return ok


# ==========================================================================
def test1_wheelpi_copy_matches_parent():
    """写しが親からずれていないこと（**親を書き換えたらここが落ちる**）。"""
    print("\n[test1] WheelPIAccelFF（v_ff_extra=0）が WheelPI と完全一致するか")
    p = RobotParams()
    args = (p.motor_Ke * p.gear_ratio, 1.0 / p.gainprm0, p.wheel_damping,
            p.wheel_frictionloss, 0.05, 0.3, p.voltage_limit, 5.0)
    a, b = WheelPI(*args), WheelPIAccelFF(*args)
    rng = np.random.default_rng(20260814)
    # **飽和とアンチワインドアップを必ず通る**ように、指令を大きく振る入力列にする
    n_sat, worst = 0, 0.0
    for _ in range(20000):
        w_ref = float(rng.uniform(-400.0, 400.0))
        w_act = float(rng.uniform(-400.0, 400.0))
        va = a.step(w_ref, w_act, p.control_dt)
        vb = b.step(w_ref, w_act, p.control_dt)
        if abs(va) >= p.voltage_limit - 1e-12:
            n_sat += 1
        worst = max(worst, abs(va - vb), abs(a.integral - b.integral))
    check("v_ff_extra=0 で親と完全一致（電圧・積分状態とも）", worst == 0.0,
          f"最大差 {worst:.3e}／飽和したティック {n_sat}/20000")
    check("飽和を実際に通っている（アンチワインドアップ経路の被覆）", n_sat > 100,
          f"{n_sat} 件")

    # v_ff_extra を入れたら、その分だけ（飽和していない範囲で）ずれること
    a2, b2 = WheelPI(*args), WheelPIAccelFF(*args)
    b2.v_ff_extra = 0.2
    va = a2.step(50.0, 50.0, p.control_dt)
    vb = b2.step(50.0, 50.0, p.control_dt)
    check("v_ff_extra がそのまま電圧に足される", math.isclose(vb - va, 0.2, abs_tol=1e-12),
          f"差 {vb - va:.6f} V（期待 0.2）")


# ==========================================================================
def _drive(policy_cls, n_ticks=1200, **kw):
    """1 面を決定的に走らせ、電圧・車輪角速度・姿勢の全記録を返す。"""
    from competition.baseline_slalom import SlalomPolicy  # noqa: F401
    params = RobotParams()
    sim = MouseSim(DESIGN_FACE + ".xml", params=params)
    z = np.load(DESIGN_FACE + ".npz")
    sim.full_reset(cell=(0, 0), heading_deg=90)
    pol = policy_cls(**kw)
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
    return np.array(rec, dtype=np.float64)


def test2_default_is_bit_identical():
    """**既定パラメータで走行がビット単位で不変**であること。"""
    print("\n[test2] k_acc_ff=0 の混ぜ込みが、素の方策と 1 ビットも違わないか")
    from competition.baseline_slalom import SlalomPolicy

    class Mixed(VelocityLoopMixin, SlalomPolicy):
        pass

    base = _drive(SlalomPolicy)
    mixed = _drive(Mixed, k_acc_ff=0.0)
    same = base.shape == mixed.shape and np.array_equal(base, mixed)
    # array_equal は NaN を等しく扱わないので、NaN が無いことも確かめる
    check("記録に NaN が無い", not np.isnan(base).any(), f"{base.shape[0]} ティック")
    check("電圧・車輪角速度・姿勢が全ティックでビット一致", same,
          f"最大差 {np.max(np.abs(base - mixed)) if base.shape == mixed.shape else 'shape 不一致'}")


# ==========================================================================
def test3_coefficient_actually_wired():
    """**係数を上げたら実際に電圧が変わる**こと（配線されていない不具合の検出）。"""
    print("\n[test3] k_acc_ff を上げたら電圧が変わるか（＝配線されているか）")
    from competition.baseline_slalom import SlalomPolicy

    class Mixed(VelocityLoopMixin, SlalomPolicy):
        pass

    base = _drive(Mixed, k_acc_ff=0.0)
    on = _drive(Mixed, k_acc_ff=1.0)
    d = np.max(np.abs(base[:, :2] - on[:, :2]))
    check("k_acc_ff=1.0 で電圧が変わる", d > 1e-3, f"電圧の最大差 {d:.4f} V")


# ==========================================================================
def test4_j_eff_from_source():
    """`J_eff` が params とモデルから導出されていること（ハードコードでないこと）。"""
    print("\n[test4] J_eff が正しい出所から導出されているか")
    from competition.baseline_slalom import SlalomPolicy

    class Mixed(VelocityLoopMixin, SlalomPolicy):
        pass

    params = RobotParams()
    sim = MouseSim(DESIGN_FACE + ".xml", params=params)
    pol = Mixed(k_acc_ff=1.0)
    pol.bind_sim(sim)
    r = params.wheel_radius
    m_tot = subtree_mass(sim.model, "mouse")
    expect = params.armature + 0.5 * params.mass_wheel * r * r + 0.5 * m_tot * r * r
    check("J_eff = armature + (1/2)m_w r² + (1/2)m_tot r²",
          math.isclose(pol._J_eff, expect, rel_tol=0, abs_tol=0),
          f"{pol._J_eff:.6e} kg·m²（m_tot = {m_tot:.6f} kg）")
    # サブツリー質量が「機体だけ」を数えていること（迷路の壁を巻き込んでいない）
    check("m_total が機体サブツリーのみ（0.05〜0.30 kg の範囲）",
          0.05 < m_tot < 0.30, f"{m_tot:.6f} kg")
    # 機体の並進慣性が支配項であること（＝第 3 項を落とすと桁が変わる）
    share = (0.5 * m_tot * r * r) / expect
    check("機体の並進慣性が支配項", share > 0.5, f"寄与 {share * 100:.1f} %")

    # 前置補償の大きさが、前向きに計算した値と一致すること
    a_max = params.a_max if hasattr(params, "a_max") else pol.a_max_measured
    dv = pol.a_max_measured
    volt = pol.k_acc_ff * pol.inv_gain * pol._J_eff * (dv / r)
    check("a_max 相当での前置補償電圧が電圧上限の内側", 0.0 < volt < params.voltage_limit,
          f"{volt:.3f} V（上限 {params.voltage_limit} V・dv/dt = {dv} m/s²）")


# ==========================================================================
def main():
    print("=" * 78)
    print("velocity_loop（F0 速度ループの是正）単体テスト")
    print("=" * 78)
    for fn in (test1_wheelpi_copy_matches_parent, test2_default_is_bit_identical,
               test3_coefficient_actually_wired, test4_j_eff_from_source):
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
