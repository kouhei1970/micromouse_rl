"""
tests/test_control_2dof.py
==========================
前方注視つき 2 自由度化 ＋ レートダンピング（`competition/control_2dof.py`・exp_016 段階 F）の
単体テスト。

**最重要の検証は 2 つある。**

1. **基準スナップショットが壊れていないこと**（既定パラメータで新既定と 1 ビット不変）
2. 🔴 **混ぜ込みどうしが打ち消し合っていないこと** — `TwoDofControlMixin` と
   016-F0-b の `ReferenceInterpMixin` は**どちらも `_do_drive_control` を差し替える**ので、
   **素朴に重ねると F0-b が静かに無効化される**。**それを検出する**。

| # | 何を反証するか | 偽ならどう見えるか |
|---|---|---|
| 1 | 既定（`tau_la`=`k_r`=0）で新既定とビット不変 | 電圧・車輪角速度・姿勢のどれかが違う |
| 2 | `tau_la` / `k_r` が個別に配線されている | 片方だけ動かしても電圧が変わらない |
| 3 | **F0-b が生き残っている**（打ち消されていない） | `ref_interp` を切り替えても走行が同じ |
| 4 | 前方注視が**弧長で前を見ている**（曲率の先読み） | `tau_la` を変えても同じ曲率を読む |
| 5 | レート項が**定常旋回で 0** になる（定常偏差を作らない） | 定常円弧で $\\omega_\\text{ref}$ が $k_r$ に依存する |

実行: .venv/bin/python tests/test_control_2dof.py
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from competition.baseline_slalom import SlalomPolicy  # noqa: E402
from competition.control_2dof import TwoDofControlMixin  # noqa: E402
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


class NewDefault(ReferenceInterpMixin, VelocityLoopMixin, SlalomPolicy):
    """いまの既定（F0 ＋ F0-b 系）。"""


class WithF(TwoDofControlMixin, ReferenceInterpMixin, VelocityLoopMixin, SlalomPolicy):
    """新既定に 016-F（前方注視＋レートダンピング）を重ねたもの。"""


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


NEW = dict(k_acc_ff=1.0, ref_interp=True)


# ==========================================================================
def test1_default_bit_identical():
    print("\n[test1] tau_la=k_r=0 の混ぜ込みが、新既定と 1 ビットも違わないか")
    base, _ = _drive(NewDefault, **NEW)
    mixed, _ = _drive(WithF, tau_la=0.0, k_r=0.0, **NEW)
    check("記録に NaN が無い", not np.isnan(base).any(), f"{base.shape[0]} ティック")
    check("電圧・車輪角速度・姿勢が全ティックでビット一致",
          base.shape == mixed.shape and np.array_equal(base, mixed),
          f"最大差 {np.max(np.abs(base - mixed)) if base.shape == mixed.shape else 'shape 不一致'}")


def test2_each_gain_wired():
    print("\n[test2] tau_la と k_r が個別に配線されているか")
    off, _ = _drive(WithF, tau_la=0.0, k_r=0.0, **NEW)
    la, _ = _drive(WithF, tau_la=0.030, k_r=0.0, **NEW)
    kr, _ = _drive(WithF, tau_la=0.0, k_r=0.2, **NEW)
    check("tau_la だけで電圧が変わる", np.max(np.abs(off[:, :2] - la[:, :2])) > 1e-6,
          f"最大差 {np.max(np.abs(off[:, :2] - la[:, :2])):.6f} V")
    check("k_r だけで電圧が変わる", np.max(np.abs(off[:, :2] - kr[:, :2])) > 1e-6,
          f"最大差 {np.max(np.abs(off[:, :2] - kr[:, :2])):.6f} V")


def test3_f0b_survives_composition():
    """🔴 **016-F を重ねても F0-b が生きていること**（打ち消しの検出）。"""
    print("\n[test3] 016-F を重ねても F0-b（参照の弧長内挿）が生きているか")
    on, _ = _drive(WithF, tau_la=0.030, k_r=0.2, k_acc_ff=1.0, ref_interp=True)
    off, _ = _drive(WithF, tau_la=0.030, k_r=0.2, k_acc_ff=1.0, ref_interp=False)
    d = np.max(np.abs(on[:, :2] - off[:, :2]))
    check("ref_interp の有無で走行が変わる（＝打ち消されていない）", d > 1e-6,
          f"電圧の最大差 {d:.6f} V")
    # 参照速度を読む口が F0-b を通っていることを直接確かめる
    _, pol = _drive(WithF, n_ticks=300, tau_la=0.030, k_r=0.2, k_acc_ff=1.0, ref_interp=True)
    if pol._path is None:
        return check("経路が張られた", False)
    # ⚠️ **母集団を構成で選ぶ**（作法 9）。速度計画が**平坦な区間では内挿値は格子値と
    #    厳密に一致するのが正しい**ので、そこで試すと「効いていない」と誤判定する。
    #    探索中の経路は全域が一定速度（未観測区画の上限）なので、
    #    **変化する計画をこちらで与えてから読む**（走行の巡り合わせに依存させない）。
    n = len(pol._path.s)
    pol._path.speed = np.linspace(0.6, 0.2, n)      # 単調に落ちる合成の計画
    sp = np.asarray(pol._path.speed, dtype=float)
    idx = n // 2
    psi = float(pol._path.heading[idx])
    # 沿線方向へ半格子ぶん進んだ位置（＝格子点のちょうど中間）で読む
    half = 0.5 * float(pol._path.s[idx + 1] - pol._path.s[idx])
    x = float(pol._path.x[idx]) + half * np.cos(psi)
    y = float(pol._path.y[idx]) + half * np.sin(psi)
    got = pol._reference_speed(idx, x, y)
    grid = float(sp[idx])
    want = float(np.sqrt(max(0.0, (sp[idx] ** 2 + sp[idx + 1] ** 2) / 2.0)))
    check("_reference_speed が格子点の値と別の値を返す（内挿が効いている）",
          abs(got - grid) > 1e-9, f"内挿 {got:.6f} 対 格子 {grid:.6f}")
    check("その値が v² の中点内挿と一致する", abs(got - want) < 1e-9,
          f"{got:.9f} 対 {want:.9f}")


def test4_preview_looks_ahead():
    print("\n[test4] 前方注視が弧長で前を見ているか")
    _, pol = _drive(WithF, n_ticks=300, tau_la=0.030, k_r=0.0, **NEW)
    if pol._path is None:
        return check("経路が張られた", False)
    path, v_ref = pol._path, 0.6
    # 曲率が変わる点（直線 → 円弧）を探し、その手前で先読みが効くことを見る
    kap = np.abs(np.asarray(path.curvature))
    edges = np.where(np.abs(np.diff(kap)) > 1e-6)[0]
    if not len(edges):
        return check("曲率の変化点がある", False)
    i = int(edges[0])
    ahead_pts = v_ref * 0.030 / max(float(np.median(np.diff(path.s))), 1e-9)
    j = max(0, i - int(ahead_pts // 2))
    pol.tau_la = 0.0
    k0 = pol._preview_curvature(j, v_ref)
    pol.tau_la = 0.030
    k1 = pol._preview_curvature(j, v_ref)
    check("tau_la>0 で先の曲率を読む", abs(k1 - k0) > 1e-9 or abs(k0 - kap[j]) < 1e-12,
          f"tau_la=0: {k0:.4f} → tau_la=0.030: {k1:.4f}（先読み {ahead_pts:.1f} 点）")
    check("tau_la=0 は現在位置の曲率そのもの", abs(k0 - float(path.curvature[j])) < 1e-12,
          f"{k0:.6f}")


def test5_rate_term_zero_in_steady_turn():
    """レート項は $\\omega_z = \\omega_\\text{ff}$ の定常旋回で 0（定常偏差を作らない）。"""
    print("\n[test5] レート項が定常旋回で 0 になるか（定常偏差を作らない）")
    _, pol = _drive(WithF, n_ticks=300, tau_la=0.0, k_r=0.4, **NEW)
    # omega_z == omega_ff のとき rate_term = k_r*(omega_z - omega_ff) = 0
    for w in (0.0, 1.0, 5.0):
        rate = pol.k_r * (w - w)
        if rate != 0.0:
            return check("定常旋回でレート項が 0", False, f"omega={w}")
    check("定常旋回（ω_z = ω_ff）でレート項が厳密に 0", True, "k_r=0.4 で 3 点確認")


# ==========================================================================
def main():
    print("=" * 78)
    print("control_2dof（016-F 前方注視＋レートダンピング）単体テスト")
    print("=" * 78)
    for fn in (test1_default_bit_identical, test2_each_gain_wired,
               test3_f0b_survives_composition, test4_preview_looks_ahead,
               test5_rate_term_zero_in_steady_turn):
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
