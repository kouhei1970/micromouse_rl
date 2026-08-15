#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""角速度の遅れの推定量が、**遅れの情報を持っているか**を合成データで検査する。

背景（教授裁定 2026-08-15）: 016-H1 の診断 D1 で、同じ「遅れ」に対して 2 通りの答えが出た。

  - **相互相関**（1 ティック刻みの伝達遅れ）… 遅れ 0 ms → 予測 H4 不成立
  - **一次遅れの時定数**（最小二乗）… 20.16 ms → 予測 H4 成立（20/20 迷路）

私は「円弧の中で $\\omega_\\text{des}$ がほぼ一定なので、相互相関はこの信号では遅れの情報を
持たない」と判断して時定数を採ったが、**それが後知恵でないことは結果の値と独立に示せる**。
本テストは**既知の答えを仕込んだ合成データ**で次の 3 つを検査する。

  1. 一次遅れ応答に時定数 $\\tau$ を仕込むと、`d1_tau` が $\\tau$ を回復する
  2. **同じ信号**では相互相関が判別力を持たない（遅れを変えても相関がほとんど動かない）
  3. 🔴 **否定対照** — 入力が動く信号に**本物の伝達遅れ**を仕込めば、相互相関はそれを
     見つける。**つまり相互相関の実装が壊れているのではなく、円弧の信号が
     遅れの情報を持っていない**（この対照が無いと 2 は「実装が壊れている」でも説明がつく）

    .venv/bin/python -m pytest tests/test_omega_lag_estimator.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "exp_016_diagonal"))
sys.path.insert(0, str(ROOT))

from run_016h1_diag import KIND, d1_lag, d1_tau      # noqa: E402

DT = 0.01          # 制御周期 [s]（正本 = mouse/params.py の control_dt）


def _arc_like_command(n, dt=DT, omega=3.0, ramp=12):
    """円弧の指令角速度に似た形: 短い立ち上がり → **長い一定** → 短い立ち下がり。

    実測の円弧区間がこの形（ほぼ一定）であり、相互相関が効かない原因そのものである。
    """
    w = np.full(n, omega)
    w[:ramp] = np.linspace(0.0, omega, ramp)
    w[-ramp:] = np.linspace(omega, 0.0, ramp)
    return w


def _first_order(u, tau, dt=DT):
    """一次遅れ y[k+1] = y[k] + (dt/tau)(u[k] − y[k])。"""
    y = np.zeros_like(u)
    for k in range(len(u) - 1):
        y[k + 1] = y[k] + (dt / tau) * (u[k] - y[k])
    return y


def _rec(omega_des, omega_act, kappa=None):
    n = len(omega_des)
    return {"t": np.arange(n) * DT, "omega_des": omega_des, "omega_act": omega_act,
            "e_y": np.zeros(n), "kind": np.full(n, KIND["arc"], dtype=float),
            "s": np.arange(n) * 0.005,
            "kappa": np.full(n, 1.0) if kappa is None else kappa}


@pytest.mark.parametrize("tau_true", [0.010, 0.020, 0.030, 0.040])
def test_tau_estimator_recovers_known_tau(tau_true):
    """1. 仕込んだ時定数を、推定が 5 % 以内で取り戻すこと。"""
    u = _arc_like_command(400)
    y = _first_order(u, tau_true)
    est = d1_tau(_rec(u, y), DT)["円弧ぜんぶ"]
    assert est is not None, "標本が足りないと判定された"
    rel = abs(est["tau_s"] - tau_true) / tau_true
    assert rel < 0.05, f"tau {est['tau_s']*1000:.2f} ms 対 真値 {tau_true*1000:.2f} ms（誤差 {rel:.1%}）"


def test_crosscorr_has_no_power_on_arc_like_signal():
    """2. **同じ信号**で相互相関は判別力を持たない（遅れを変えても相関が動かない）。"""
    u = _arc_like_command(400)
    y = _first_order(u, 0.020)
    est = d1_lag(_rec(u, y), DT)
    assert est is not None
    # 最大点の相関と、遅れ 0 での相関がほとんど同じ ＝ どの遅れでも同じくらい合う
    assert est["peak_corr"] - est["corr_at_zero"] < 0.02, (
        f"相互相関が判別力を持ってしまっている（差 {est['peak_corr']-est['corr_at_zero']:.4f}）"
    )


def test_crosscorr_does_find_a_real_transport_delay():
    """3. 🔴 否定対照 — 入力が動く信号に本物の伝達遅れを入れれば、相互相関は見つける。

    これが通ることで、2 の結果が「実装が壊れている」ではなく
    「円弧の信号が遅れの情報を持っていない」であることが示される。
    """
    rng = np.random.default_rng(0)
    n, lag = 600, 4
    u = np.cumsum(rng.normal(0.0, 0.3, n))       # ゆっくり動き続ける入力
    y = np.concatenate([np.zeros(lag), u[:-lag]])  # 純粋な伝達遅れ 4 ティック
    est = d1_lag(_rec(u, y), DT)
    assert est is not None
    assert est["lag_ticks"] == lag, f"見つけた遅れ {est['lag_ticks']} ティック 対 仕込み {lag}"
    assert est["peak_corr"] - est["corr_at_zero"] > 0.05, "判別力が出ていない"


def test_tau_stratification_splits_clothoid_and_constant_curvature():
    """曲率が変化する標本（クロソイド）と一定の標本（定曲率）を分けられること。"""
    n = 400
    u = _arc_like_command(n)
    y = _first_order(u, 0.020)
    kappa = np.concatenate([np.linspace(0.0, 1.0, 150), np.full(n - 150, 1.0)])
    out = d1_tau(_rec(u, y, kappa), DT)
    assert out["クロソイド"] is not None and out["定曲率"] is not None
    assert out["クロソイド"]["n"] + out["定曲率"]["n"] == out["円弧ぜんぶ"]["n"]
