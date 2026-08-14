#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**是正 (D) の写しが実際に効いていることの検査**
（`card_016diag_fixD.md` §3 の手順 2）。

**防御 3 点は親（`baseline_slalom.py:1205-1241`）からの写しである。**
**写しは「書いたつもり」で効いていないことがあるので、
生成された速度計画の配列そのもので確かめる。**

| # | 防御 | ここで確かめること |
|---|---|---|
| **①** | 打ち切られた経路は `v_uncertain` で頭打ち | **打ち切りのとき、計画速度の最大が `v_uncertain` 以下** |
| **②** | 終端速度 | **打ち切りのとき、終端手前 `cell_size/2` 以降が `v_end` 以下** |
| **③** | 落としきれないなら全体を固定 | **短い経路（`s_term < d_brake`）で、計画速度の最大が保守的な上限以下** |

**加えて「防御が斜め・円弧の上限を引き上げていないこと」も確かめる**
（`np.minimum` で頭打ちにしているだけであることの確認）。

    .venv/bin/python -m pytest tests/test_terminal_speed_guard.py -q
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "experiments" / "exp_016_diagonal",
          REPO_ROOT / "experiments" / "exp_015_time_optimal_route"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from competition.baseline_slalom_diag_cal_fixab import (  # noqa: E402
    SlalomDiagCalFixABPolicy)
from competition.baseline_slalom_diag_cal_fixd import (  # noqa: E402
    SlalomDiagCalFixDPolicy)

V_DIAG = 0.45


def _policy(**kw):
    """`bind_sim` を経ずに、検査に要る量だけを埋めた方策を作る。"""
    p = SlalomDiagCalFixDPolicy(**kw)
    p.v_uncertain = 0.5510701394626467      # 実測値（安全率 0.75）
    return p


def _flat_path(length_m, ds=0.005):
    """直線の経路（曲率 0）の弧長配列。"""
    n = int(round(length_m / ds)) + 1
    return np.linspace(0.0, length_m, n), np.zeros(n)


# ==========================================================================
def test_guard1_truncated_path_capped_to_v_uncertain():
    """🔴 **防御 ①**: 打ち切られた経路は `v_uncertain` で頭打ちになる。"""
    p = _policy()
    s, k = _flat_path(0.19)                      # 37 点相当（衝突時と同じ長さ）
    v_ceil = np.full(len(s), p.v_cap)            # 斜めの分岐が作る「直進は v_cap」
    captured = {}

    def fake(s_arr, curv, v, a_lat, a_max, v_creep, stop_at_end):
        captured["v"] = np.asarray(v, dtype=float).copy()
        return np.asarray(v, dtype=float)

    p._orig_speed_profile = fake
    p._guarded_speed_profile(s, k, v_ceil, p.a_lat, p.a_max, p.v_creep,
                             stop_at_end=False)
    assert captured["v"].max() <= p.v_uncertain + 1e-12, \
        f"打ち切られた経路の上限が v_uncertain を超えている: {captured['v'].max()}"
    assert p.n_guard_capped == 1, "防御 ① が働いた記録が残っていない"


def test_guard3_short_path_clamped():
    """🔴 **防御 ③**: 落としきれない短い経路は、全体が保守的な上限に固定される。"""
    p = _policy()
    # 終端で止まる（stop_at_end=True）短い経路 → d_brake = v_cap^2/(2 a_max) ≈ 0.998 m
    s, k = _flat_path(0.30)
    v_ceil = np.full(len(s), p.v_cap)
    captured = {}

    def fake(s_arr, curv, v, *a, **kw):
        captured["v"] = np.asarray(v, dtype=float).copy()
        return np.asarray(v, dtype=float)

    p._orig_speed_profile = fake
    p._guarded_speed_profile(s, k, v_ceil, p.a_lat, p.a_max, p.v_creep,
                             stop_at_end=True)
    d_brake = p.v_cap ** 2 / (2.0 * p.a_max)
    assert 0.30 < d_brake, "この検査の前提（経路が d_brake より短い）が崩れている"
    assert captured["v"].max() <= p.v_uncertain + 1e-12, \
        f"短い経路が固定されていない: {captured['v'].max()}"
    assert p.n_guard_clamped == 1, "防御 ③ が働いた記録が残っていない"


def test_long_path_is_untouched():
    """**十分に長い経路では、防御は上限を下げない**（効きすぎていないこと）。"""
    p = _policy()
    s, k = _flat_path(3.0)                       # d_brake ≈ 0.998 m より十分長い
    v_ceil = np.full(len(s), p.v_cap)
    captured = {}

    def fake(s_arr, curv, v, *a, **kw):
        captured["v"] = np.asarray(v, dtype=float).copy()
        return np.asarray(v, dtype=float)

    p._orig_speed_profile = fake
    p._guarded_speed_profile(s, k, v_ceil, p.a_lat, p.a_max, p.v_creep,
                             stop_at_end=True)
    assert np.allclose(captured["v"], p.v_cap), "長い経路で上限が下がっている"
    assert p.n_guard_capped == 0 and p.n_guard_clamped == 0


def test_diagonal_ceiling_is_never_raised():
    """**防御は斜め・円弧の上限（0.45）を引き上げない**（頭打ちにするだけ）。"""
    p = _policy()
    s, k = _flat_path(0.19)
    v_ceil = np.full(len(s), V_DIAG)             # すべて斜め扱い
    captured = {}

    def fake(s_arr, curv, v, *a, **kw):
        captured["v"] = np.asarray(v, dtype=float).copy()
        return np.asarray(v, dtype=float)

    p._orig_speed_profile = fake
    p._guarded_speed_profile(s, k, v_ceil, p.a_lat, p.a_max, p.v_creep,
                             stop_at_end=False)
    assert captured["v"].max() <= V_DIAG + 1e-12, "斜めの上限が引き上げられた"


def test_guard_can_be_disabled():
    """**`terminal_guard=False` なら混ぜ込む前と同じ**（無害性 D3 の土台）。"""
    off = SlalomDiagCalFixDPolicy(terminal_guard=False)
    assert off.terminal_guard is False
    on = SlalomDiagCalFixDPolicy()
    assert on.terminal_guard is True
    # (A) 版には属性そのものが無い（混ぜ込みが漏れていないこと）
    assert not hasattr(SlalomDiagCalFixABPolicy(), "terminal_guard")


def test_class_composition():
    """**最終形が意図した組み合わせになっている**（(A)+(B)+(D)）。

    🔴 **是正 (C) は 2026-08-15 の教授裁定 (乙) で撤去した**
    （実効が無いことが実測で分かったため。`card_016diag_fixC.md` §6-2-1）。
    **ここで「(C) が混ざっていないこと」も確かめる。**
    """
    d = SlalomDiagCalFixDPolicy()
    assert d.terminal_guard, "(D) が欠けている"
    assert not hasattr(d, "keep_horizon"), "撤去したはずの (C) が混ざっている"
    assert d.align_check and d.use_maze_flag, "(A) か (B) が欠けている"
    assert d.safety_factor == 0.75 and d.ref_interp and d.k_acc_ff == 1.0
    # 混ぜ込みは一番外側（防御が最後に掛かる）
    mro = [c.__name__ for c in type(d).__mro__]
    assert mro.index("TerminalSpeedGuardMixin") < mro.index("SlalomDiagCalFixABPolicy")
