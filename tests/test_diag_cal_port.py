#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**斜め方策の載せ替えの構造検査**（`card_016diag_switch.md` §2-2 の安価な側）。

走らせずに確かめられることをここで押さえる。**走行のビット一致**は
`experiments/exp_016_diagonal/check_diag_cal_port.py` が別に測る（シミュレータが要るため）。

押さえる 4 点:

1. **載せた積み上げが新既定と同じ**（安全率 0.75・F0・F0-b・混ぜ込みの順序）
2. **016-D 版（`SlalomDiagPolicy`）の既定が汚れていない**（大域の差し替えが漏れていない）
3. **参照経路を作る関数の差し替えが、例外が出ても必ず戻る**
4. **`L_c` = 0 なら参照経路が 016-D 版とビット単位で同じ**（方策の口から通して確かめる）

    .venv/bin/python -m pytest tests/test_diag_cal_port.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "experiments" / "exp_016_diagonal",
          REPO_ROOT / "experiments" / "exp_015_time_optimal_route"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from competition.baseline_slalom_diag import SlalomDiagPolicy  # noqa: E402
from competition.baseline_slalom_diag_cal import (L_C_CLOTHOID_M,  # noqa: E402
                                                  SlalomDiagCalPolicy)
from competition.baseline_slalom_e1t_tr_f0b_cal import (  # noqa: E402
    SAFETY_FACTOR_CALIBRATED, SlalomE1TTRF0bCalPolicy)
from diag_path import build_diagonal_path  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

CELL = RobotParams().cell_size
R_ARC_M = 0.060


# ==========================================================================
def test_stack_matches_new_default():
    """**載せた積み上げが、斜めなしの新既定と同じ**であること。"""
    a = SlalomDiagCalPolicy()
    b = SlalomE1TTRF0bCalPolicy()
    for fld in ("safety_factor", "v_cap", "a_max", "a_lat", "k_acc_ff", "ref_interp"):
        assert getattr(a, fld) == getattr(b, fld), f"{fld} が新既定と違う"
    assert a.safety_factor == SAFETY_FACTOR_CALIBRATED == 0.75
    assert a.k_acc_ff == 1.0 and a.ref_interp is True

    # 混ぜ込みの順序が新既定と同じ（F0-b → F0 → 本体）
    mro = [c.__name__ for c in SlalomDiagCalPolicy.__mro__]
    i_ri, i_vl = mro.index("ReferenceInterpMixin"), mro.index("VelocityLoopMixin")
    assert i_ri < i_vl < mro.index("SlalomDiagPolicy"), f"混ぜ込みの順序が違う: {mro[:5]}"
    mro_b = [c.__name__ for c in SlalomE1TTRF0bCalPolicy.__mro__]
    assert mro_b.index("ReferenceInterpMixin") < mro_b.index("VelocityLoopMixin")


def test_016d_defaults_untouched():
    """**016-D 版の既定が汚れていない**（差し替えが大域へ漏れていない）。"""
    SlalomDiagCalPolicy()                      # 作るだけで汚れないこと
    q = SlalomDiagPolicy()
    assert q.safety_factor == 0.7, "016-D 版の安全率が動いている"
    assert not hasattr(q, "ref_interp") or q.ref_interp is False
    import competition.baseline_slalom_diag as mod
    assert mod.build_diagonal_path is build_diagonal_path, \
        "モジュール大域の build_diagonal_path が差し替わったまま残っている"


def test_swap_is_restored_even_on_exception():
    """**差し替えが、例外が出ても必ず戻る**（`finally` が効いていること）。"""
    import competition.baseline_slalom_diag as mod
    original = mod.build_diagonal_path
    pol = SlalomDiagCalPolicy()

    class Boom(Exception):
        pass

    def explode(self, x, y, yaw):
        assert mod.build_diagonal_path is not original, "差し替えが効いていない"
        raise Boom()

    # 親の `_replan` の代わりに爆発するものを置く
    SlalomDiagPolicy._replan, saved = explode, SlalomDiagPolicy._replan
    try:
        with pytest.raises(Boom):
            pol._replan(0.0, 0.0, 0.0)
    finally:
        SlalomDiagPolicy._replan = saved
    assert mod.build_diagonal_path is original, "例外の後に差し替えが戻っていない"


def _sample_segment():
    """調整用迷路 1 件から、斜めを含む節点列・方位列を取り出す。"""
    from competition.baseline_slalom_e1_tr import load_time_model
    from competition.route_planner import value_field
    from diagonal_model import DiagonalGridModel, cell_center_node, descend
    from route_model import connects_true, load_maze
    from run_016b import cut_segment, longest_diagonal_run

    a, b = load_time_model()
    f = sorted((REPO_ROOT / "competition" / "mazes" / "design_v4").glob("maze_*.npz"))[0]
    v, h, start, goals = load_maze(str(f))
    conn = connects_true(v, h)
    model = DiagonalGridModel(a, b, r=1.0)
    field = value_field([tuple(g) for g in goals], 16, 16, conn, model)
    p = descend(field, model, cell_center_node(tuple(start)), "N", 16, 16, conn)
    s0, e0 = longest_diagonal_run(p["dirs"])
    i, j = cut_segment(p["nodes"], p["dirs"], s0, e0)
    return p["nodes"][i:j + 1], p["dirs"][i:j]


def test_lc_zero_reproduces_016d_path_through_policy():
    """**`L_c` = 0 なら参照経路が 016-D 版とビット単位で同じ**（方策の口から通す）。"""
    nodes, dirs = _sample_segment()
    pol0 = SlalomDiagCalPolicy(L_c=0.0)
    pa, ka, ia = build_diagonal_path(nodes, dirs, CELL, R_ARC_M)
    pb, kb, ib = pol0._build_path(nodes, dirs, CELL, R_ARC_M)
    assert np.array_equal(ka, kb) and np.array_equal(ia, ib)
    for fld in ("s", "x", "y", "heading", "curvature"):
        assert np.array_equal(np.asarray(getattr(pa, fld)), np.asarray(getattr(pb, fld))), \
            f"{fld} がビット一致しない"

    # 既定（L_c = 47.1239 mm）では**違う経路になる**（検査が空振りしていないこと）
    pol = SlalomDiagCalPolicy()
    assert pol.L_c == L_C_CLOTHOID_M
    pc, _kc, _ic = pol._build_path(nodes, dirs, CELL, R_ARC_M)
    assert len(pc.s) != len(pa.s) or not np.array_equal(np.asarray(pc.x), np.asarray(pa.x)), \
        "既定の L_c でも経路が変わっていない — 差し替えが効いていない"
