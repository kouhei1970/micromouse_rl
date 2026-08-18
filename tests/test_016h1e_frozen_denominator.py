#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""016-H1e の否定対照（分母の凍結）が **発火と空振りの両方を実測して初めて
判別力を持つ** ことを確かめる（`PREREG_016h1e.md` §4 W2、教授指示 2026-08-18）。

**是正2件目（2026-08-18）**: 検査対象を、`run_016h1e.py` が本測定で実際に
使う凍結クラス `run_016h1e.make_frozen` そのものへ揃えた。
かつては本ファイルが `competition.baseline_slalom.SlalomPolicy` を親にした
**第 3 の写し**（`_make_frozen`）を自前で持ち、それを検査していたが、
**MRO 実測では `SlalomPolicy._do_drive_control` は走行中に一度も呼ばれない**
（測定条件 = F0 + F0-b + 安全率 0.75 では `TwoDofControlMixin` が
`ReferenceInterpMixin._do_drive_control` — `competition/reference_interp.py:139`
付近 — へ委譲する。`run_016h1e.py` 冒頭の注記を見よ）。処置の写しが 2 本に
分かれていると片方だけ直しても検査が通ってしまうため、自前の写しは削除し、
**`run_016h1e.make_frozen` を import して唯一の処置として検査する。**

検査対象の実装は `competition/reference_interp.py` の
`ReferenceInterpMixin._do_drive_control`（`ref_interp=True` の枝）:

    lateral_term = math.atan2(self.k_y * e_y, v_for_gain + self.v_eps)
    omega_ref    = pcurv * v_ref + k_psi * e_psi - lateral_term

方策は `run_016f0_ladder.make_policy_class(k_acc_ff=1.0, ref_interp=True, safety=0.75)`
（`card_016h1e.md` §3 の測定条件そのもの）で作り、`run_016h1e.make_frozen` で包む。

経路は曲率 0（直線）の 1 点だけのダミーとし、`pcurv * v_ref` 項を 0 にして
omega_ref を (e_y, e_psi, v) だけの関数にする。これにより速度計画・経路生成
一式を回さずに制御則の計算だけを直接検査できる（(a)(b) の 2 検査）。

さらに **新 W2**（PREREG §4 が 2026-08-18 に差し替え）を足す:
凍結クラスを `d0=None`（分母を凍結しない）で走らせると、健常と
**軌跡・電圧が bit 一致する**こと。これは「分母の 1 箇所以外は健常の
本体と同一である」という凍結クラスの構造上の主張そのものを検査する
（`tests/test_h1d_probe_harmless.py` の `_trajectory` と同型・許容差なし）。
この検査が空振りしないことも、$D_0$ に実際の値を入れれば落ちることで確認する。

    .venv/bin/python -m pytest tests/test_016h1e_frozen_denominator.py -q

--------------------------------------------------------------------------
tests/test_h1d_probe_harmless.py はまだ意味を持つか（card_016h1e.md §7 の判断）
--------------------------------------------------------------------------
**持つ。無改修のままでよい。** `test_h1d_probe_harmless.py` は
`run_016h1_diag.make_probed`（`_wheel_targets_to_voltage` を包んで記録するだけ・
返り値は親のものを素通しする「計装」）が走行を1ビットも変えないことを保証する
検査であり、本テストが検査する `run_016h1e.make_frozen`（`_do_drive_control` の
分母を意図的に書き換える「処置」）とは対象がそもそも別物である。
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "exp_016_diagonal"))
sys.path.insert(0, str(ROOT))

from competition.baseline_slalom_diag_cal import L_C_CLOTHOID_M      # noqa: E402
from competition.baseline_slalom_e1_tr import load_time_model        # noqa: E402
from competition.route_planner import value_field                    # noqa: E402
from mouse.mjcf import build_maze_robot_xml                          # noqa: E402
from mouse.params import RobotParams                                 # noqa: E402
from mouse.sim import MouseSim                                       # noqa: E402

import run_016f0_ladder                                              # noqa: E402
import run_016g_ladder                                               # noqa: E402
from diagonal_model import (DELTA8, DiagonalGridModel,               # noqa: E402
                            cell_center_node, descend)
from route_model import connects_true, load_maze                     # noqa: E402
from run_016b import cut_segment, longest_diagonal_run               # noqa: E402
from run_016c import R_ARC_M                                         # noqa: E402
from run_016h1e import D0_FROZEN_DEFAULT as D0_FROZEN, make_frozen   # noqa: E402

SAFETY = 0.75   # card_016h1e.md §3 の測定条件


def _base_policy_cls():
    """本測定で実際に使う方策（`run_016h1e.py` の `main()` と同一の作り方）。"""
    return run_016f0_ladder.make_policy_class(k_acc_ff=1.0, ref_interp=True, safety=SAFETY)


# --------------------------------------------------------------------------
# (a)(b): 制御則の計算だけを直接検査する（経路を曲率0の1点ダミーにする）
# --------------------------------------------------------------------------

class _FakeSim:
    """`privileged_velocity()` だけを持つ最小のダミー sim。"""

    def __init__(self, v_fwd: float):
        self._v_fwd = v_fwd

    def privileged_velocity(self):
        return self._v_fwd, 0.0


class _FakePath:
    """曲率0（直線）の1点だけのダミー経路。heading=0・位置=(0,0)に固定し、
    pcurv=0 なので `pcurv * v_ref` 項が消え、omega_ref は e_y, e_psi, v の
    3変数だけの関数になる。"""

    def __init__(self):
        self.x = np.array([0.0])
        self.y = np.array([0.0])
        self.heading = np.array([0.0])
        self.curvature = np.array([0.0])
        self.speed = np.array([0.0])   # pcurv=0 で v_ref 項は消えるため値は無関係
        self.s = np.array([0.0])
        self.stop_at_end = True


def _omega_ref(e_y: float, e_psi: float, v: float, *, frozen_d0=D0_FROZEN) -> float:
    """`run_016h1e.make_frozen` で包んだ本番方策の `_do_drive_control` を直接呼び、
    計算された omega_ref を返す（`_wheel_targets_to_voltage` を横取りする計装は
    `run_016h1_diag.make_probed` と同型）。

    経路 heading=0・位置=(0,0) に固定した上で、e_y = y、e_psi = -yaw となる
    (x, y, yaw) を逆算して渡す（`_do_drive_control` 内の e_y・e_psi の定義どおり）。
    """
    base_cls = make_frozen(_base_policy_cls(), d0=frozen_d0)

    class Probed(base_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.recorded_omega = float("nan")

        def _wheel_targets_to_voltage(self, v_cmd, omega_cmd, obs):
            self.recorded_omega = float(omega_cmd)   # 記録のみ。実装への書き戻しはしない
            return 0.0, 0.0

    pol = Probed(None, np.array([1.0]))
    pol._path = _FakePath()
    pol._cursor = 0
    pol._v_setpoint = 0.0
    pol._sim = _FakeSim(v)

    x, y, yaw = 0.0, e_y, -e_psi
    pol._do_drive_control(np.zeros(1), x, y, yaw)
    assert math.isfinite(pol.recorded_omega), "omega_ref が計算されなかった"
    return pol.recorded_omega


def _omega_ref_healthy(e_y: float, e_psi: float, v: float) -> float:
    """健常（分母を凍結しない、写しですらない本番クラスそのもの）の omega_ref。"""
    base_cls = _base_policy_cls()

    class Probed(base_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.recorded_omega = float("nan")

        def _wheel_targets_to_voltage(self, v_cmd, omega_cmd, obs):
            self.recorded_omega = float(omega_cmd)
            return 0.0, 0.0

    pol = Probed(None, np.array([1.0]))
    pol._path = _FakePath()
    pol._cursor = 0
    pol._v_setpoint = 0.0
    pol._sim = _FakeSim(v)

    x, y, yaw = 0.0, e_y, -e_psi
    pol._do_drive_control(np.zeros(1), x, y, yaw)
    assert math.isfinite(pol.recorded_omega), "omega_ref が計算されなかった"
    return pol.recorded_omega


# (e_y[m], e_psi[rad]) の代表的な組。事前登録が要求する「少なくとも5通り」。
# 円弧入口付近で観測される桁（|e_y| 数mm〜十数mm、|e_psi| 数度〜十数度）を覆う。
CASES = [
    (0.000, 0.000),
    (0.005, 0.000),
    (0.010, 0.05),
    (-0.008, -0.03),
    (0.015, 0.10),
    (0.003, -0.12),
]


@pytest.mark.parametrize("e_y,e_psi", CASES)
def test_w2_miss_at_v045(e_y, e_psi):
    """(a) 空振り側: v + v_eps = D0 となる v=0.45 では、健常と凍結の omega_ref が一致する。

    D0=0.60 は v=0.45 水準の公称値 0.45+0.15 そのものなので、構成上ここは
    一致しなければならない（PREREG §4 W2 冒頭）。"""
    v = 0.45
    healthy = _omega_ref_healthy(e_y, e_psi, v)
    frozen = _omega_ref(e_y, e_psi, v, frozen_d0=D0_FROZEN)
    assert healthy == pytest.approx(frozen, abs=1e-12), (
        f"空振りするはずの v=0.45 で健常と凍結が食い違った "
        f"(healthy={healthy!r}, frozen={frozen!r}, e_y={e_y}, e_psi={e_psi})"
    )


@pytest.mark.parametrize("e_y,e_psi", CASES)
def test_w2_fire_at_v070(e_y, e_psi):
    """(b) 発火側: v=0.70 では健常と凍結の omega_ref が一致しない。

    v=0.70 の実際の分母は 0.70+0.15=0.85 であり D0=0.60 と異なるので、
    e_y != 0 の限り lateral_term が変わり omega_ref も変わるはず。"""
    v = 0.70
    if e_y == 0.0:
        pytest.skip("e_y=0 では lateral_term=atan2(0, ·)=0 となり、分母を変えても差が出ない"
                    "（判別力の検査にならない自明ケース）")
    healthy = _omega_ref_healthy(e_y, e_psi, v)
    frozen = _omega_ref(e_y, e_psi, v, frozen_d0=D0_FROZEN)
    assert healthy != pytest.approx(frozen, abs=1e-9), (
        f"発火するはずの v=0.70 で健常と凍結が一致してしまった（判別力なし） "
        f"(healthy={healthy!r}, frozen={frozen!r}, e_y={e_y}, e_psi={e_psi})"
    )


def test_w2_spoiled_d0_would_fail_the_fire_side():
    """自己検査: (b) が本当に落ち得ることの確認（空振り防止）。

    D0 をわざと v=0.70 の実際の分母 0.85 と同じ値にすり替えると、
    (b) の非一致条件が破れて test_w2_fire_at_v070 相当のアサーションは
    失敗するはずである。これを直接再現し、(b) に判別力があることを示す。"""
    e_y, e_psi, v = 0.010, 0.05, 0.70
    healthy = _omega_ref_healthy(e_y, e_psi, v)
    spoiled_frozen = _omega_ref(e_y, e_psi, v, frozen_d0=(v + 0.15))  # 実際の分母と同じ値に細工
    assert healthy == pytest.approx(spoiled_frozen, abs=1e-9), (
        "D0 を v=0.70 の実分母に細工したのに一致しなかった。"
        "_omega_ref のセットアップ自体が壊れている可能性がある"
    )


def test_probe_is_transparent_on_the_healthy_policy():
    """計装（Probed）が健常方策の omega_ref をありのまま横取りしていることの確認。
    記録された値が有限で、飽和リミット [-turn_omega_limit, +turn_omega_limit] の
    範囲内にあることだけを検査する（式そのものの正しさは上の (a)(b) が検査する）。"""
    pol = _base_policy_cls()(None, np.array([1.0]))
    limit = pol.turn_omega_limit
    omega = _omega_ref_healthy(0.01, 0.05, 0.60)
    assert math.isfinite(omega)
    assert -limit - 1e-9 <= omega <= limit + 1e-9


# --------------------------------------------------------------------------
# 新 W2（是正3件目・PREREG §4 2026-08-18 差し替え）:
# 分母を凍結しない設定（d0=None）なら、健常と軌跡・電圧が bit 一致すること。
# --------------------------------------------------------------------------

DESIGN_BAND_DIR = ROOT / "competition" / "mazes" / "design_v4"
V_DIAG_PROBE = 0.60   # W1 と無関係な任意の1水準。飽和が起きうる高速側で確かめる


def _design_band_first_n(n: int):
    files = sorted(DESIGN_BAND_DIR.glob("maze_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    return files[:n]


def _segment(maze_path, params):
    v, h, start, goals = load_maze(str(maze_path))
    conn = connects_true(v, h)
    a, b = load_time_model()
    field = value_field([tuple(g) for g in goals], 16, 16, conn,
                        DiagonalGridModel(a, b, r=1.0))
    p = descend(field, DiagonalGridModel(a, b, r=1.0),
                cell_center_node(tuple(start)), "N", 16, 16, conn)
    s0, e0 = longest_diagonal_run(p["dirs"])
    i, j = cut_segment(p["nodes"], p["dirs"], s0, e0)
    xml = maze_path.with_suffix(".xml")
    if not xml.exists():
        build_maze_robot_xml(v, h, str(xml), model_name=f"m_{maze_path.stem}_w2", params=params)
    return xml, p["nodes"][i:j + 1], p["dirs"][i:j], v, h


def _trajectory(policy_cls, params, xml, nodes, dirs, v_walls, h_walls, max_s=40.0):
    """1 迷路を走らせ、毎ティックの (x, y, yaw, 電圧 L, 電圧 R) を返す
    （`tests/test_h1d_probe_harmless.py` の `_trajectory` と同型）。"""
    builder = run_016g_ladder.make_builder(L_C_CLOTHOID_M)
    path, kinds, _ = builder(nodes, dirs, params.cell_size, R_ARC_M)
    sim = MouseSim(str(xml), params=params)
    sim.full_reset(cell=(nodes[0][0] // 2, nodes[0][1] // 2),
                   heading_deg=math.degrees(math.atan2(DELTA8[dirs[0]][1],
                                                       DELTA8[dirs[0]][0])))
    pol = policy_cls(path, np.where(kinds == "straight", 1e9, V_DIAG_PROBE))
    pol.bind_sim(sim)
    pol.bind_maze(v_walls, h_walls)
    pol.on_maze_start(dict(width=16, height=16))
    traj = []
    for _ in range(int(max_s / params.control_dt)):
        vl, vr = pol.act(sim.observation())
        x, y, yaw = sim.privileged_pose()
        traj.append((x, y, yaw, float(vl), float(vr)))
        out = sim.step_control(vl, vr)
        if out.get("collision") or pol.finished:
            break
    return np.asarray(traj, dtype=float)


@pytest.mark.parametrize("maze_idx", [0, 1])
def test_w2_unfrozen_matches_healthy_bit_exact(maze_idx):
    """新 W2: 凍結クラスを d0=None（分母を凍結しない）で走らせると、
    健常方策と軌跡・電圧が **bit 一致** する（PREREG §4 W2・是正3件目）。

    設計帯の先頭2面（seed 昇順）で確認する。許容差は置かない。"""
    params = RobotParams()
    maze = _design_band_first_n(2)[maze_idx]
    xml, nodes, dirs, v_walls, h_walls = _segment(maze, params)
    healthy_cls = _base_policy_cls()
    unfrozen_cls = make_frozen(_base_policy_cls(), d0=None)

    healthy = _trajectory(healthy_cls, params, xml, nodes, dirs, v_walls, h_walls)
    unfrozen = _trajectory(unfrozen_cls, params, xml, nodes, dirs, v_walls, h_walls)

    assert healthy.shape == unfrozen.shape, (
        f"ティック数が違う {healthy.shape} 対 {unfrozen.shape}"
    )
    assert len(healthy) > 100, f"標本が少なすぎて検査にならない（{len(healthy)} ティック）"
    assert np.array_equal(healthy, unfrozen), (
        "d0=None（凍結しない）なのに健常と走行が食い違った。"
        "凍結クラスの写しが分母以外も変えている疑いがある（最大差 "
        f"{np.max(np.abs(healthy - unfrozen)):.3e}）"
    )


def test_w2_frozen_with_real_d0_would_fail_the_match():
    """自己検査: 新 W2 が空振りしないことの確認。

    d0=D0_FROZEN_DEFAULT（実際に分母を凍結する設定）で同じ比較をすると、
    軌跡は bit 一致しないはずである。これを実測し、新 W2 に判別力があることを示す。"""
    params = RobotParams()
    maze = _design_band_first_n(1)[0]
    xml, nodes, dirs, v_walls, h_walls = _segment(maze, params)
    healthy_cls = _base_policy_cls()
    frozen_cls = make_frozen(_base_policy_cls(), d0=D0_FROZEN)

    healthy = _trajectory(healthy_cls, params, xml, nodes, dirs, v_walls, h_walls)
    frozen = _trajectory(frozen_cls, params, xml, nodes, dirs, v_walls, h_walls)

    same_shape = healthy.shape == frozen.shape
    assert not (same_shape and np.array_equal(healthy, frozen)), (
        "D0 を実際に凍結した（d0=D0_FROZEN_DEFAULT）のに健常と bit 一致してしまった。"
        "新 W2 の比較そのものに判別力がない疑いがある"
    )
