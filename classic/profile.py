"""classic/profile.py — マウス本体が持つ速度計画器（理想時間・速度プロファイル計算）

「物理限界で走ったらどれだけ速いか」を計算する部品である。もとは判定側
（`competition/`）の物差しとして企図されたが、ユーザ裁定により**マウス本体が持つ
速度計画器**（追従制御が使う目標 v(s) の生成元）として `classic/` 側に置くことに
変わった。判定側は同じモジュールを、真の地図を入力にして呼ぶ側になる。

設計の祖先: `research_notes/scripts/check_physical_limits_and_ideal_lap.py`
（式・定数・スイープの作法をそのまま踏襲する）。

## 値の出所

- `docs/MODEL_VERIFICATION_PLAN.md` §4（目標仕様パラメータ表・導出される性能）
- `mouse/params.py`（`RobotParams`。r, N, Rw, Kt, Vmax, tau_c, tread は
  ここから 1:1 で取得し、二重管理を避ける）
- `M`, `M_eff`, `mu`, `mu_c`, `eta`, `g`, `c_eff` は `RobotParams` にフィールドが
  無い（もしくは丸め済みの導出値である）ため、祖先スクリプトの値をそのまま
  モジュール定数として複写する（§4.2/§4.3 の数値と同一）。

## I_ZZ / I_EFF / R_CASTER について（🔴 実行時に MuJoCo も迷路 XML も読まない）

`classic/` パッケージは「MuJoCo や `mouse/` のシミュレータを import してはならない層」
（`classic/__init__.py`）であり、かつ**マウス本体のモジュールが迷路 XML（壁の真値を
含む）を読む経路を作ってはならない**（引き継ぎメモの規約）。そのため、その場旋回の
慣性まわりの値はモデルから一度だけ合成した結果を**モジュール定数として固定**する。

- `I_ZZ` … `competition/mazes/design_turn_v1/maze_41000.xml` を MuJoCo で読み、
  `mouse` ボディ以下の全サブツリー（mouse, motorbox1/2, left/right_wheel,
  visual_LF/LS/RF/RS の 9 ボディ）について、各ボディの `body_inertia`（主慣性）を
  `ximat`（世界座標への回転）で world へ回し、系全体の重心まわりへ平行軸の定理で
  合成した zz 成分（車輪の自転は含まない剛体近似）。
- `I_EFF` … `I_ZZ` に、左右車輪それぞれの world 慣性テンソルの yy 成分の和
  （車輪単体の自転軸まわりの慣性）へ `((tread/2)/r)**2` を掛けたものを加算した値
  （ヨーレート運動に対する車輪自転の見かけの寄与を含めた実効慣性）。
- `R_CASTER` … 前キャスタ geom（`caster_front`）の機体中心からの x 距離
  （MJCF の `geom_pos`。前後キャスタは点対称配置なのでどちらでも同じ大きさ）。

**この 3 定数の導出の正しさは `tests/test_profile.py::test_recorded_inertia_matches_the_model`
が毎回 MuJoCo モデルから再計算して検査する。**定数が実モデルからずれたらそのテストが
落ちる（テストは XML を読んでよい。読んではいけないのは本体側のこのモジュール）。
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from mouse.params import RobotParams

__all__ = [
    "VehicleLimits",
    "vehicle_limits",
    "SpinTurn",
    "spin_turn_time",
    "Segment",
    "SegmentResult",
    "IdealTime",
    "min_time",
    "lap_time",
    "turn_segments",
    "arc_length",
    "tangent_consumption",
    "I_ZZ",
    "I_EFF",
    "R_CASTER",
]


# ============================================================================
# 0. 祖先由来の設計定数（RobotParams に対応フィールドが無い、もしくは丸め済みの
#    導出値であるため、祖先スクリプトの値をそのまま複写する。
#    docs/MODEL_VERIFICATION_PLAN.md §4.2 / §4.3）
# ============================================================================
_M = 0.106       # kg     総質量（§4.2 質量配分表の合計）
_M_EFF = 0.1272  # kg     有効並進質量 M_eff = M + 2(I_w+armature)/r^2（§4.3 公称値）
_MU = 1.0        # —      車輪-床の滑り摩擦係数（§4.2。やや楽観側と正本が明記）
_MU_C = 0.08     # —      キャスタ-床の滑り摩擦係数（§4.2）
_ETA = 0.6       # —      駆動輪の荷重比率（設計目標。§4.2/§4.3）
_G = 9.81        # m/s²
_C_EFF = 1.027   # N·s/m  有効粘性（逆起電力＋軸受粘性を並進換算。§4.3 公称値）

# その場旋回の慣性まわり（モジュール docstring 参照。MuJoCo モデルから一度だけ
# 合成した固定値。tests/test_profile.py::test_recorded_inertia_matches_the_model
# が実モデルとの一致を毎回検査する）。
I_ZZ = 9.622436e-05    # kg·m²   系全体の重心まわりのヨー慣性（車輪自転を含まない）
I_EFF = 9.975473e-05   # kg·m²   I_ZZ + 車輪自転慣性の等価寄与
R_CASTER = 0.045        # m       前キャスタの機体中心からの x 距離


# ============================================================================
# 1. VehicleLimits
# ============================================================================
@dataclass(frozen=True)
class VehicleLimits:
    """車両の物理限界一式（祖先スクリプトの式で計算した値 + その場旋回の限界）。"""

    M: float
    M_eff: float
    r: float
    N: float
    Rw: float
    Kt: float
    Vmax: float
    tau_c: float
    mu: float
    mu_c: float
    eta: float
    g: float
    c_eff: float
    tread: float

    F_stall: float
    F_fric: float
    V_TOP: float
    A_TR: float
    A_LAT: float

    I_zz: float
    I_eff: float
    r_caster: float
    tau_yaw_max: float
    tau_caster: float
    tau_yaw_net: float
    alpha_yaw_max: float


def vehicle_limits(params: Optional[RobotParams] = None) -> VehicleLimits:
    """車両の物理限界を計算する。

    `params` を省略すると `mouse.params.RobotParams()`（正本の既定値）を使う。
    r, N, Rw, Kt, Vmax, tau_c, tread は `RobotParams` から 1:1 で取得する
    （二重管理を避けるため）。M, M_eff, mu, mu_c, eta, g, c_eff は
    `RobotParams` に対応フィールドが無いため、モジュール定数（§0）を使う。
    """
    p = params if params is not None else RobotParams()

    r = p.wheel_radius
    N = p.gear_ratio
    Rw = p.motor_R
    Kt = p.motor_Kt
    Vmax = p.voltage_limit
    tau_c = p.wheel_frictionloss
    tread = p.tread

    M = _M
    M_eff = _M_EFF
    mu = _MU
    mu_c = _MU_C
    eta = _ETA
    g = _G
    c_eff = _C_EFF

    # ---- 祖先スクリプトの式そのまま ----
    F_stall = 2 * N * Kt * Vmax / (Rw * r)
    F_fric = 2 * tau_c / r + mu_c * (1 - eta) * M * g
    V_TOP = (F_stall - F_fric) / c_eff
    A_TR = mu * eta * g - mu_c * (1 - eta) * g
    A_LAT = (mu * eta + mu_c * (1 - eta)) * g

    # ---- その場旋回（新規。教授セッションが実モデルから確定済み） ----
    I_zz = I_ZZ
    I_eff = I_EFF
    r_caster = R_CASTER

    tau_yaw_max = min(mu * eta * M * g * (tread / 2), F_stall * (tread / 2))
    tau_caster = mu_c * (1 - eta) * M * g * r_caster
    tau_yaw_net = tau_yaw_max - tau_caster
    alpha_yaw_max = tau_yaw_net / I_eff

    return VehicleLimits(
        M=M, M_eff=M_eff, r=r, N=N, Rw=Rw, Kt=Kt, Vmax=Vmax, tau_c=tau_c,
        mu=mu, mu_c=mu_c, eta=eta, g=g, c_eff=c_eff, tread=tread,
        F_stall=F_stall, F_fric=F_fric, V_TOP=V_TOP, A_TR=A_TR, A_LAT=A_LAT,
        I_zz=I_zz, I_eff=I_eff, r_caster=r_caster,
        tau_yaw_max=tau_yaw_max, tau_caster=tau_caster, tau_yaw_net=tau_yaw_net,
        alpha_yaw_max=alpha_yaw_max,
    )


# ============================================================================
# 2. その場旋回（バンバン制御。モータの角速度限界を検査し三角形/台形を切り替える）
# ============================================================================
@dataclass(frozen=True)
class SpinTurn:
    """静止 → 静止のその場旋回の最小時間。"""

    theta: float        # rad（絶対値）
    time: float         # s
    regime: str          # "triangular"（三角形。角速度が頭打ちにならない）
                          # "trapezoidal"（台形。車輪角速度が V_TOP/r で頭打ち）
    alpha: float          # rad/s²  実際に使った角加速度（tau_yaw_net/I_eff）
    omega_peak: float     # rad/s   実際に到達するピーク角速度


def spin_turn_time(theta_rad: float, limits: VehicleLimits,
                    tau_caster: Optional[float] = None) -> SpinTurn:
    """静止 → 静止のその場旋回の最小時間（バンバン制御）。

    `tau_caster` を省略すると `limits.tau_caster`（キャスタ引きずり込み）を使う。
    `tau_caster=0.0` を渡すとキャスタ引きずりを含めない場合の比較ができる
    （教授セッションの手計算との照合に使う）。

    車輪角速度 omega_w = omega_z*(tread/2)/r が V_TOP/r を超えないか検査し、
    超える場合は台形（頭打ち）で計算し直す。
    """
    theta = abs(theta_rad)
    tc = limits.tau_caster if tau_caster is None else tau_caster
    tau_net = limits.tau_yaw_max - tc
    alpha = tau_net / limits.I_eff

    # 車輪角速度が V_TOP/r を超えないヨー角速度の上限（omega_z*(tread/2)/r <= V_TOP/r）
    omega_cap = 2.0 * limits.V_TOP / limits.tread

    if theta <= 0.0:
        return SpinTurn(theta=0.0, time=0.0, regime="triangular", alpha=alpha, omega_peak=0.0)

    omega_tri = math.sqrt(alpha * theta)
    if omega_tri <= omega_cap:
        t = 2.0 * math.sqrt(theta / alpha)
        return SpinTurn(theta=theta, time=t, regime="triangular", alpha=alpha, omega_peak=omega_tri)

    # 台形: 0→omega_cap（加速）、omega_cap 一定（巡航）、omega_cap→0（減速）
    t = omega_cap / alpha + theta / omega_cap
    return SpinTurn(theta=theta, time=t, regime="trapezoidal", alpha=alpha, omega_peak=omega_cap)


# ============================================================================
# 3. Segment / IdealTime / min_time
# ============================================================================
@dataclass(frozen=True)
class Segment:
    """経路の 1 区間（直線 or 円弧）。"""

    length: float       # 経路長 [m]
    curvature: float    # 曲率 1/R [1/m]。直線は 0.0
    kind: str = "straight"  # "straight" / "arc" など。集計用の札


@dataclass(frozen=True)
class SegmentResult:
    """`Segment` 1 個ぶんの最適速度プロファイルの内訳。"""

    kind: str
    length: float
    curvature: float
    t: float          # この区間の所要時間 [s]
    v_in: float        # 区間始点の速度 [m/s]
    v_out: float       # 区間終点の速度 [m/s]
    v_peak: float       # 区間内の最高速度 [m/s]
    accel_len: float    # 加速区間の距離 [m]
    cruise_len: float   # 定常速度区間の距離 [m]（0 なら「三角形」= 定常速度に未到達）
    decel_len: float    # 減速区間の距離 [m]
    accel_t: float       # 加速区間の時間 [s]
    cruise_t: float      # 定常速度区間の時間 [s]
    decel_t: float        # 減速区間の時間 [s]


@dataclass(frozen=True)
class IdealTime:
    """最小時間の速度プロファイルの計算結果。"""

    total: float                       # 総所要時間 [s]
    by_kind: Dict[str, float]           # 札ごとの秒
    path_length: float                  # 経路長 [m]
    v_max: float                        # 最高速度 [m/s]
    v_min: float                        # 最低速度 [m/s]
    by_mode: Dict[str, Tuple[float, float]]  # {"accel": (秒, m), "cruise": (...), "decel": (...)}
    segments: List[SegmentResult]        # 入力 Segment 1 個ごとの内訳
    reached_v_top: bool                  # 経路上のどこかで速度上限に達したか
    n_triangular: int                    # 直線区間のうち三角形（cruise_len==0）だった本数
    n_trapezoidal: int                   # 直線区間のうち台形（cruise_len>0）だった本数
    s_grid: List[float]                  # ds 格子の境界点の弧長座標 [m]（長さ n+1）
    v_grid: List[float]                  # 各境界点の速度 [m/s]（長さ n+1）
    kappa_grid: List[float]              # 各セルの曲率 [1/m]（長さ n）

    def v_at(self, s: float) -> float:
        """弧長 s における速度 [m/s]（線形補間。範囲外は端の値でクランプ）。"""
        grid = self.s_grid
        if not grid:
            return 0.0
        if s <= grid[0]:
            return self.v_grid[0]
        if s >= grid[-1]:
            return self.v_grid[-1]
        i = bisect.bisect_right(grid, s) - 1
        i = min(max(i, 0), len(grid) - 2)
        s0, s1 = grid[i], grid[i + 1]
        v0, v1 = self.v_grid[i], self.v_grid[i + 1]
        if s1 <= s0:
            return v0
        frac = (s - s0) / (s1 - s0)
        return v0 + frac * (v1 - v0)

    def kappa_at(self, s: float) -> float:
        """弧長 s における曲率 [1/m]（区分定数。範囲外は端のセルの値）。"""
        grid = self.s_grid
        if not self.kappa_grid:
            return 0.0
        if s <= grid[0]:
            return self.kappa_grid[0]
        if s >= grid[-1]:
            return self.kappa_grid[-1]
        i = bisect.bisect_right(grid, s) - 1
        i = min(max(i, 0), len(self.kappa_grid) - 1)
        return self.kappa_grid[i]


# eps: v[i+1]-v[i] の符号で加速/定常/減速を分類するしきい値 [m/s]。
# sqrt を伴う複数回の浮動小数点演算により定常区間でも 1e-12〜1e-15 程度の丸め差が
# 出うるが、1 セル(ds=0.0005m)で生じる最小の実質的な速度変化は
# 概ね sqrt(2*A_TR*ds) ~ sqrt(2*5.6*0.0005) ≈ 0.075 m/s のオーダーであり、
# 両者の間に十分な余裕がある 1e-9 を採用する。
_MODE_EPS = 1e-9


def _accel_cap(v_i: float, kap_i: float, limits: VehicleLimits) -> float:
    ay = v_i ** 2 * kap_i
    ax_cap = limits.A_TR * math.sqrt(max(1.0 - (ay / limits.A_LAT) ** 2, 0.0))
    ax_mot = (limits.F_stall - limits.c_eff * v_i - limits.F_fric) / limits.M_eff
    return max(min(ax_cap, ax_mot), 0.0)


def _decel_cap(v_i: float, kap_i: float, limits: VehicleLimits) -> float:
    ay = v_i ** 2 * kap_i
    return limits.A_TR * math.sqrt(max(1.0 - (ay / limits.A_LAT) ** 2, 0.0))


def _cap_speed(kap_i: float, v_limit: float, A_LAT: float) -> float:
    if kap_i == 0.0:
        return v_limit
    return min(v_limit, math.sqrt(A_LAT / abs(kap_i)))


def _build_grid(segments: Sequence[Segment], ds: float):
    s: List[float] = []
    kap: List[float] = []
    kind: List[str] = []
    seg_idx: List[int] = []
    for idx, seg in enumerate(segments):
        n_cells = max(int(seg.length / ds), 1)
        cell_len = seg.length / n_cells
        s += [cell_len] * n_cells
        kap += [seg.curvature] * n_cells
        kind += [seg.kind] * n_cells
        seg_idx += [idx] * n_cells
    return s, kap, kind, seg_idx


def _summarize_segment(start: int, end: int, s, kap, kind, v, dt) -> SegmentResult:
    length = sum(s[start:end])
    t = sum(dt[start:end])
    v_in = v[start]
    v_out = v[end]
    v_peak = max(v[start:end + 1])
    accel_len = cruise_len = decel_len = 0.0
    accel_t = cruise_t = decel_t = 0.0
    for i in range(start, end):
        if v[i + 1] > v[i] + _MODE_EPS:
            accel_len += s[i]
            accel_t += dt[i]
        elif v[i + 1] < v[i] - _MODE_EPS:
            decel_len += s[i]
            decel_t += dt[i]
        else:
            cruise_len += s[i]
            cruise_t += dt[i]
    return SegmentResult(
        kind=kind[start], length=length, curvature=kap[start], t=t,
        v_in=v_in, v_out=v_out, v_peak=v_peak,
        accel_len=accel_len, cruise_len=cruise_len, decel_len=decel_len,
        accel_t=accel_t, cruise_t=cruise_t, decel_t=decel_t,
    )


def _finalize(s, kap, kind, seg_idx, v, limits: VehicleLimits, v_limit: float) -> IdealTime:
    n = len(s)
    dt = [0.0] * n
    T = 0.0
    by_kind: Dict[str, float] = {}
    by_mode_acc: Dict[str, List[float]] = {"accel": [0.0, 0.0], "cruise": [0.0, 0.0], "decel": [0.0, 0.0]}

    for i in range(n):
        vbar = 0.5 * (v[i] + v[i + 1])
        dt_i = s[i] / max(vbar, 1e-9)
        dt[i] = dt_i
        T += dt_i
        by_kind[kind[i]] = by_kind.get(kind[i], 0.0) + dt_i
        if v[i + 1] > v[i] + _MODE_EPS:
            mode = "accel"
        elif v[i + 1] < v[i] - _MODE_EPS:
            mode = "decel"
        else:
            mode = "cruise"
        by_mode_acc[mode][0] += dt_i
        by_mode_acc[mode][1] += s[i]

    by_mode: Dict[str, Tuple[float, float]] = {k: (t, l) for k, (t, l) in by_mode_acc.items()}

    segments_out: List[SegmentResult] = []
    if n:
        start = 0
        cur_idx = seg_idx[0]
        for i in range(1, n + 1):
            if i == n or seg_idx[i] != cur_idx:
                segments_out.append(_summarize_segment(start, i, s, kap, kind, v, dt))
                if i < n:
                    cur_idx = seg_idx[i]
                    start = i

    n_triangular = sum(1 for sr in segments_out if sr.kind == "straight" and sr.cruise_len <= 0.0)
    n_trapezoidal = sum(1 for sr in segments_out if sr.kind == "straight" and sr.cruise_len > 0.0)

    v_max = max(v)
    v_min = min(v)
    reached_v_top = v_max >= v_limit - _MODE_EPS

    s_grid = [0.0] * (n + 1)
    for i in range(n):
        s_grid[i + 1] = s_grid[i] + s[i]

    return IdealTime(
        total=T, by_kind=by_kind, path_length=s_grid[-1], v_max=v_max, v_min=v_min,
        by_mode=by_mode, segments=segments_out, reached_v_top=reached_v_top,
        n_triangular=n_triangular, n_trapezoidal=n_trapezoidal,
        s_grid=s_grid, v_grid=list(v), kappa_grid=list(kap),
    )


def min_time(segments: Sequence[Segment], limits: VehicleLimits,
             v_start: float = 0.0, v_end: float = 0.0,
             v_cap: Optional[float] = None, ds: float = 0.0005,
             max_sweeps: int = 200) -> IdealTime:
    """開いた経路（周回ではない）の最小時間を、前進パス／後退パスのスイープで求める。

    祖先スクリプト `lap()` と同じ物理制約・同じスイープ構造だが、周回の巻き戻し
    （`% n`）をせず、両端に境界条件 `v[0]=v_start`, `v[-1]=v_end` を課す。

    `v_cap` が与えられたら、全点の速度上限に `min(V_TOP, v_cap)` を使う。

    🔴 スイープは収束するまで回す（前後パスを速度列が変化しなくなるまで繰り返し、
    `max_sweeps` に達したら例外を投げる。黙って打ち切らない）。
    """
    if not segments:
        raise ValueError("segments は空にできない")

    s, kap, kind, seg_idx = _build_grid(segments, ds)
    n = len(s)
    v_limit = limits.V_TOP if v_cap is None else min(limits.V_TOP, v_cap)

    kap_ext = kap + [kap[-1]]  # 境界点は n+1 個。最後の境界点は最後のセルの曲率を引き継ぐ
    v = [_cap_speed(k, v_limit, limits.A_LAT) for k in kap_ext]
    v[0] = v_start
    v[n] = v_end

    converged = False
    for _ in range(max_sweeps):
        prev = v.copy()
        for i in range(n):  # 前進パス（加速の制約）。v[0]/v[n] は書き換えない
            ax = _accel_cap(v[i], kap_ext[i], limits)
            cand = math.sqrt(max(v[i] ** 2 + 2 * ax * s[i], 0.0))
            j = i + 1
            if j != n:
                v[j] = min(v[j], cand)
        for i in range(n, 0, -1):  # 後退パス（減速の制約）
            axd = _decel_cap(v[i], kap_ext[i], limits)
            cand = math.sqrt(max(v[i] ** 2 + 2 * axd * s[i - 1], 0.0))
            j = i - 1
            if j != 0:
                v[j] = min(v[j], cand)
        if v == prev:
            converged = True
            break
    if not converged:
        raise RuntimeError(
            f"min_time: {max_sweeps} 回のスイープでも速度列が収束しなかった"
            f"（セル数={n}）。max_sweeps を増やすか、境界条件・経路を見直すこと。"
        )

    return _finalize(s, kap, kind, seg_idx, v, limits, v_limit)


def lap_time(segments: Sequence[Segment], limits: VehicleLimits,
             v_cap: Optional[float] = None, ds: float = 0.0005,
             max_sweeps: int = 200) -> IdealTime:
    """閉じた周回経路（始点と終点が滑らかに繋がる）の最小時間。

    祖先スクリプト `lap()` の周回スイープ（`% n` による巻き戻し）を、任意の
    `Segment` 列に一般化したもの。`min_time`（開いた経路。両端の境界速度を
    個別に指定できる）とは対をなす。`test_reproduces_the_ancestor_circuit_lap`
    の比較対象。
    """
    if not segments:
        raise ValueError("segments は空にできない")

    s, kap, kind, seg_idx = _build_grid(segments, ds)
    n = len(s)
    v_limit = limits.V_TOP if v_cap is None else min(limits.V_TOP, v_cap)
    v = [_cap_speed(k, v_limit, limits.A_LAT) for k in kap]

    converged = False
    for _ in range(max_sweeps):
        prev = v.copy()
        for i in range(n):
            j = (i + 1) % n
            ax = _accel_cap(v[i], kap[i], limits)
            cand = math.sqrt(max(v[i] ** 2 + 2 * ax * s[i], 0.0))
            v[j] = min(v[j], cand)
        for i in range(n - 1, -1, -1):
            j = (i - 1) % n
            axd = _decel_cap(v[i], kap[i], limits)
            cand = math.sqrt(max(v[i] ** 2 + 2 * axd * s[j], 0.0))
            v[j] = min(v[j], cand)
        if v == prev:
            converged = True
            break
    if not converged:
        raise RuntimeError(
            f"lap_time: {max_sweeps} 回のスイープでも速度列が収束しなかった（セル数={n}）。"
        )

    v_ext = v + [v[0]]
    return _finalize(s, kap, kind, seg_idx, v_ext, limits, v_limit)


# ============================================================================
# 4. 円弧の補助関数
# ============================================================================
def arc_length(delta_theta_rad: float, radius: float) -> float:
    """半径 `radius`、旋回角 `delta_theta_rad` の円弧の弧長 [m]。"""
    return radius * abs(delta_theta_rad)


def turn_segments(delta_theta_rad: float, radius: float) -> List[Segment]:
    """半径 `radius`、旋回角 `delta_theta_rad` の円弧 1 個を `Segment` にする。"""
    length = arc_length(delta_theta_rad, radius)
    curvature = 1.0 / radius if radius != 0.0 else 0.0
    return [Segment(length=length, curvature=curvature, kind="arc")]


def tangent_consumption(delta_theta_rad: float, radius: float) -> float:
    """円弧が入出の直線から食う長さ `radius*tan(|delta_theta|/2)` [m]。"""
    return radius * math.tan(abs(delta_theta_rad) / 2.0)
