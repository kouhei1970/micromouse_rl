"""
classic/motion.py
================
区画単位の動作（直進 N 区画／その場 90° 旋回（左右）／その場 180° 旋回／停止）を
生成する簡素な速度制御器。実機のマウサー用語でいう
**「超信地旋回走行（区画ごと停止）」**（`docs/JA_ENGINEERING_TERMS.md`）を再現する。

【推測航法のみを使う（真値位置は使わない）】
`mouse.sim.MouseSim.privileged_pose()` は一切参照しない。走行開始時の姿勢
（`heading_deg`）は競技規約上あらかじめ分かっている既知情報（評価器が
`reset_to_start(cell=(0,0), heading_deg=90.0)` で置く、というプロトコル上の
定数であり、走行中の真値ではない）としてのみ使い、以後は
**車輪角速度の積算（並進距離）とジャイロ角速度の積算（方位）のみ**で
自機の移動量を推定する（note_030 §3 の S1 出口条件: 壁補正は S2 の担当、
ここでは推測航法のみでよい）。

【制御構成（2 段構成）】
  上位（区画単位の司令）: 残距離・残ヨー角への比例制御で
    「左右それぞれの車輪角速度の目標値」を作る。直進中は方位を保持する
    差動補正（Kp_heading）も同じ比例則で重ね合わせる。
  下位（車輪速度ループ）: 車輪角速度目標値と実測車輪角速度
    （observation() の車輪角速度 2 チャンネル）の偏差を PI 制御で
    左右モータ電圧へ変換する（アンチワインドアップ付き: 出力が飽和して
    いる間は積分しない）。

【PI ゲインの決め方（実測に基づく。決め打ちではない）】
`docs/ROBOT_SPEC.md` §2.5 の実測較正値（速度時定数 τ_v=0.124 s、
最高速度 v_max=3.84 m/s @ V_max=3.0V → 並進 DC ゲイン K≈v_max/V_max×(1/r)
≈ 94.8 rad/s/V）を出発点に、IMC（内部モデル制御）流の PI 設計
Kp=τ/(K·λ), Ki=Kp/τ（λ: 閉ループ時定数、ここでは λ=0.05s 相当を狙う）で
概算した値を初期値とし、実際にシミュレータで単一車輪の速度ステップ応答を
とって発振しないことを確認して微調整した
（`tests/test_classic_sensing.py` ではなく、本ファイル作成時に手元で
実行したチューニングスクリプトでの確認。最終値は下記 DEFAULT_* 定数）。

実測した動作精度（開通路 5x5 迷路、区画 (2,2) 開始、詳細は
`tests/test_classic_motion.py` 参照）:
  - 直進 2 区画（目標 0.36 m）: 実移動 0.3562 m（誤差 -3.8 mm）、
    方位ずれは実質ゼロ（heading hold が効いている）
  - その場 90° 左旋回: 実ヨー変化 89.56°（誤差 0.44°）、
    旋回中の位置ずれ 0.056 mm
  - その場 90° 右旋回: 実ヨー変化 -89.56°（誤差 0.44°）
  - その場 180° 旋回: 実ヨー変化 179.52°（誤差 0.48°）、位置ずれ 0.23 mm

いずれも S1 の出口条件（低速・確実。速さは目的ではない）に対して十分な精度。
"""
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from mouse.params import RobotParams


class MotionKind(Enum):
    """区画単位の動作の種別（実機のターン種別に相当する語彙）。"""
    STOP = auto()          # 停止（速度指令ゼロを保持）
    FORWARD = auto()       # 直進 N 区画
    TURN_LEFT_90 = auto()  # その場 90° 左旋回（超信地旋回）
    TURN_RIGHT_90 = auto()  # その場 90° 右旋回（超信地旋回）
    TURN_180 = auto()      # その場 180° 旋回（超信地旋回）


# ------------------------------------------------------------------
# 既定ゲイン・速度上限（モジュール docstring に決定根拠を記載）
# ------------------------------------------------------------------
DEFAULT_KP_WHEEL: float = 0.025   # 車輪速度 PI の比例ゲイン [V/(rad/s)]
DEFAULT_KI_WHEEL: float = 0.2     # 車輪速度 PI の積分ゲイン [V/(rad/s)/s]（積分器へは誤差を dt 積算）
DEFAULT_V_CRUISE: float = 0.12    # 直進の巡航速度上限 [m/s]（v_max=3.84m/s の約 3%。低速優先）
DEFAULT_OMEGA_CRUISE: float = 1.2  # 旋回の角速度上限 [rad/s]（≈69°/s）
DEFAULT_KP_DIST: float = 3.0      # 残距離 → 目標並進速度 の比例ゲイン [1/s]
DEFAULT_KP_YAW: float = 3.0       # 残ヨー角 → 目標角速度 の比例ゲイン [1/s]
DEFAULT_KP_HEADING: float = 2.0   # 直進中の方位保持（差動補正）の比例ゲイン [1/s]
DEFAULT_DISTANCE_TOL: float = 0.003   # 直進の完了判定距離許容 [m]
DEFAULT_YAW_TOL: float = math.radians(1.0)  # 旋回の完了判定角度許容 [rad]
DEFAULT_SPEED_SETTLE: float = 0.01    # 直進完了判定の速度しきい値 [m/s]
DEFAULT_OMEGA_SETTLE: float = 0.05    # 旋回完了判定の角速度しきい値 [rad/s]
DEFAULT_INTEG_CLAMP: float = 200.0    # 積分器の絶対値クランプ（暴走防止の保険）


@dataclass
class MotionCommand:
    """発行済みの区画単位コマンド（診断・テスト用に読み出せるように保持する）。"""
    kind: MotionKind
    n_cells: int = 1  # FORWARD のみ意味を持つ


class CellMotionController:
    """区画単位の動作を生成する速度制御器。

    使い方:
        ctrl = CellMotionController(params)
        ctrl.reset(heading_deg=90.0)     # 走行開始時に一度だけ
        ctrl.start_forward(1)            # 1 区画前進を発行
        while True:
            obs = sim.observation()
            vl, vr, done = ctrl.update(obs)
            sim.step_control(vl, vr)
            if done:
                break

    **真値位置（privileged_pose）は一切使わない。**内部状態は
    「車輪角速度・ジャイロの積算による推測航法」のみで更新する
    （壁センサによる補正は S2 の担当。ここでは持たない）。
    """

    def __init__(self, params: Optional[RobotParams] = None,
                 kp_wheel: float = DEFAULT_KP_WHEEL, ki_wheel: float = DEFAULT_KI_WHEEL,
                 v_cruise: float = DEFAULT_V_CRUISE, omega_cruise: float = DEFAULT_OMEGA_CRUISE,
                 kp_dist: float = DEFAULT_KP_DIST, kp_yaw: float = DEFAULT_KP_YAW,
                 kp_heading: float = DEFAULT_KP_HEADING,
                 distance_tol: float = DEFAULT_DISTANCE_TOL, yaw_tol: float = DEFAULT_YAW_TOL,
                 speed_settle: float = DEFAULT_SPEED_SETTLE, omega_settle: float = DEFAULT_OMEGA_SETTLE):
        self.params = params if params is not None else RobotParams()
        self.kp_wheel = kp_wheel
        self.ki_wheel = ki_wheel
        self.v_cruise = v_cruise
        self.omega_cruise = omega_cruise
        self.kp_dist = kp_dist
        self.kp_yaw = kp_yaw
        self.kp_heading = kp_heading
        self.distance_tol = distance_tol
        self.yaw_tol = yaw_tol
        self.speed_settle = speed_settle
        self.omega_settle = omega_settle

        self._n_sensors = len(self.params.sensors)
        self.reset()

    # ------------------------------------------------------------------
    # 状態リセット・観測の切り出し
    # ------------------------------------------------------------------
    def reset(self, heading_deg: float = 90.0) -> None:
        """走行開始時に一度だけ呼ぶ。推測航法の内部状態を初期化する。

        heading_deg は評価器が走行開始時に機体を置く既知の初期方位
        （競技プロトコル上の定数）であり、走行中の真値取得ではない。
        """
        self._yaw_est = math.radians(heading_deg)
        self._dist_est = 0.0
        self._target_dist = 0.0
        self._target_heading = self._yaw_est
        self._target_yaw = self._yaw_est
        self._integ_l = 0.0
        self._integ_r = 0.0
        self._cmd: Optional[MotionCommand] = None

    def _split_obs(self, obs: np.ndarray):
        """observation() から車輪角速度・ジャイロ z 成分を取り出す
        （並びは mouse/sim.py の observation() docstring どおり:
        [距離×n, 加速度×3, ジャイロ×3, 車輪角速度L, 車輪角速度R]）。"""
        n = self._n_sensors
        gyro_z = float(obs[n + 5])
        omega_l = float(obs[n + 6])
        omega_r = float(obs[n + 7])
        return gyro_z, omega_l, omega_r

    @property
    def yaw_estimate(self) -> float:
        """推測航法によるヨー角推定 [rad]（真値ではなく積算値）。"""
        return self._yaw_est

    @property
    def is_idle(self) -> bool:
        return self._cmd is None

    # ------------------------------------------------------------------
    # コマンド発行
    # ------------------------------------------------------------------
    def start_forward(self, n_cells: int) -> None:
        """直進 n_cells 区画を開始する（方位は発行時の推定方位を保持する）。"""
        if n_cells <= 0:
            raise ValueError(f"n_cells は正の整数で指定してください: {n_cells}")
        self._cmd = MotionCommand(MotionKind.FORWARD, n_cells)
        self._dist_est = 0.0
        self._target_dist = n_cells * self.params.cell_size
        self._target_heading = self._yaw_est
        self._integ_l = 0.0
        self._integ_r = 0.0

    def start_turn_left(self) -> None:
        """その場 90° 左旋回（超信地旋回）を開始する。左旋回はヨー角を正方向へ増やす
        （mouse.sim.MouseSim.privileged_pose() と同じ CCW 正の規約）。"""
        self._start_turn(math.radians(90.0))
        self._cmd = MotionCommand(MotionKind.TURN_LEFT_90)

    def start_turn_right(self) -> None:
        """その場 90° 右旋回を開始する。"""
        self._start_turn(math.radians(-90.0))
        self._cmd = MotionCommand(MotionKind.TURN_RIGHT_90)

    def start_turn_180(self) -> None:
        """その場 180° 旋回を開始する。"""
        self._start_turn(math.radians(180.0))
        self._cmd = MotionCommand(MotionKind.TURN_180)

    def start_stop(self) -> None:
        """速度指令ゼロを保持するコマンドを発行する（update() は常に done=True を返す）。"""
        self._cmd = MotionCommand(MotionKind.STOP)
        self._integ_l = 0.0
        self._integ_r = 0.0

    def _start_turn(self, delta_yaw: float) -> None:
        self._target_yaw = self._yaw_est + delta_yaw
        self._integ_l = 0.0
        self._integ_r = 0.0

    # ------------------------------------------------------------------
    # 制御ステップ
    # ------------------------------------------------------------------
    def update(self, obs: np.ndarray):
        """1 制御ステップ分の (v_left, v_right, done) を返す。

        呼び出し側は返された電圧を sim.step_control() へそのまま渡し、
        done=True になったら次のコマンドを発行すること（このメソッド自体は
        コマンドを進めない。呼び出し側が明示的に start_* を呼ぶ）。
        """
        if self._cmd is None:
            raise RuntimeError("start_forward()/start_turn_*()/start_stop() のいずれかを先に呼んでください")

        dt = self.params.control_dt
        r = self.params.wheel_radius
        tread = self.params.tread
        gyro_z, omega_l, omega_r = self._split_obs(obs)

        # 推測航法の積算（車輪角速度 → 並進距離、ジャイロ → 方位）
        v_meas = (omega_l + omega_r) / 2.0 * r
        self._dist_est += v_meas * dt
        self._yaw_est += gyro_z * dt

        kind = self._cmd.kind

        if kind == MotionKind.STOP:
            return 0.0, 0.0, True

        if kind == MotionKind.FORWARD:
            remaining = self._target_dist - self._dist_est
            v_cmd = float(np.clip(self.kp_dist * remaining, -self.v_cruise, self.v_cruise))

            heading_err = self._wrap(self._target_heading - self._yaw_est)
            domega = self.kp_heading * heading_err

            omega_l_target = (v_cmd - domega * tread / 2.0) / r
            omega_r_target = (v_cmd + domega * tread / 2.0) / r
            vl, vr = self._wheel_pi(omega_l_target, omega_r_target, omega_l, omega_r, dt)

            done = abs(remaining) < self.distance_tol and abs(v_meas) < self.speed_settle
            if done:
                return 0.0, 0.0, True
            return vl, vr, False

        if kind in (MotionKind.TURN_LEFT_90, MotionKind.TURN_RIGHT_90, MotionKind.TURN_180):
            remaining = self._wrap(self._target_yaw - self._yaw_est)
            omega_cmd = float(np.clip(self.kp_yaw * remaining, -self.omega_cruise, self.omega_cruise))

            v_r = omega_cmd * tread / 2.0
            v_l = -omega_cmd * tread / 2.0
            omega_l_target = v_l / r
            omega_r_target = v_r / r
            vl, vr = self._wheel_pi(omega_l_target, omega_r_target, omega_l, omega_r, dt)

            done = abs(remaining) < self.yaw_tol and abs(gyro_z) < self.omega_settle
            if done:
                return 0.0, 0.0, True
            return vl, vr, False

        raise AssertionError(f"未対応の MotionKind: {kind}")

    def _wheel_pi(self, omega_l_target, omega_r_target, omega_l, omega_r, dt):
        """車輪角速度目標値と実測値の偏差から PI で左右電圧を作る（アンチワインドアップ付き）。"""
        vlim = self.params.voltage_limit

        err_l = omega_l_target - omega_l
        err_r = omega_r_target - omega_r

        vl_try = self.kp_wheel * err_l + self.ki_wheel * self._integ_l
        vr_try = self.kp_wheel * err_r + self.ki_wheel * self._integ_r
        # アンチワインドアップ: 出力が電圧上限に張り付いていない間だけ積分する
        if abs(vl_try) < vlim:
            self._integ_l = float(np.clip(self._integ_l + err_l * dt, -DEFAULT_INTEG_CLAMP, DEFAULT_INTEG_CLAMP))
        if abs(vr_try) < vlim:
            self._integ_r = float(np.clip(self._integ_r + err_r * dt, -DEFAULT_INTEG_CLAMP, DEFAULT_INTEG_CLAMP))

        vl = float(np.clip(self.kp_wheel * err_l + self.ki_wheel * self._integ_l, -vlim, vlim))
        vr = float(np.clip(self.kp_wheel * err_r + self.ki_wheel * self._integ_r, -vlim, vlim))
        return vl, vr

    @staticmethod
    def _wrap(angle: float) -> float:
        """角度を [-pi, pi) に正規化する。"""
        return math.atan2(math.sin(angle), math.cos(angle))
