"""
classic/pose.py
================
連続姿勢 (x, y, theta) の拡張カルマンフィルタ（**予測だけ**を持つ推定器）。

`research_notes/note_037_probabilistic_localization.md`（設計）・
`experiments/exp_031_metric_pose/PREREG.md`（判定条文。段階 E1）の実装。
観測による更新（側壁距離・前壁距離・壁切れ・静止更新）は E2 以降で足す。
1 実験 1 変更の規約（`docs/RESEARCH_PLAN.md` §9-9）により、ジャイロバイアスも
今回は状態に入れない（E2 で追加）。

【真値を絶対に読まない】
本ファイルは `mouse.sim.MouseSim.privileged_pose()` を一切参照しない（import
すらしない）。壊れていないことは `tests/test_classic_pose.py` が
ソース走査と否定対照の両方で検査する。

【状態と運動モデル（note_037 §5）】
状態 x = (x, y, theta)（単位 m, m, rad）、共分散 P は 3×3。制御周期ごとに

    ds     = r * (omega_L + omega_R) / 2 * dt
    dtheta = gyro_z * dt
    x     += ds * cos(theta + dtheta/2)
    y     += ds * sin(theta + dtheta/2)
    theta  = wrap(theta + dtheta)

**回転はジャイロだけを使う（車輪の差分からは取らない）。**車輪の左右差は
滑りに弱く、確率ロボティクスの標準的な運動モデルでも実機の慣行でも
ジャイロを使うのが定石（note_037 §5）。

【座標系】
`docs/COORDINATE_SYSTEM.md` に従う。既定の初期姿勢は競技の発進状態
（区画 (0,0) の中心 = (0.09, 0.09) m、北向き = ヨー角 +90° = pi/2 rad、
同文書 §6・§7）。

【マイコン実装の予算（note_037 §13-4。三角関数の呼び出し回数と RAM の実測は
tests/test_classic_pose.py::test_predict_raw_calls_cos_and_sin_exactly_once_each・
::test_state_and_covariance_ram_footprint_is_48_bytes が裏取りする。
乗算回数と負荷率は下記の手計算であり、実行時計測ではない）】

1 制御周期（`predict_raw` 1 回）あたり:
  - 三角関数呼び出し: **2 回**（cos・sin。`half = theta + dtheta/2` の 1 点でだけ
    評価し、他では使い回す）。角度の正規化 `wrap()` は三角関数を使わず
    剰余演算（`math.fmod`）だけで書いてあるので、ここでは増えない。
  - 乗算回数: 状態更新 5 回（ds・dtheta・half の半角・x 更新・y 更新）
    + ヤコビアン充填 4 回（ds*cos, ds*sin を F と V で使い回し、V の半分だけ
    追加で 2 回）+ 過程雑音 4 回（var_ds = alpha1*ds^2 + alpha2*dtheta^2 の
    4 項）+ 共分散伝播（3×3・3×2 の**単純な密行列積**として見積もる。
    F/V の疎性は使わない— note_037 §13-2 の見積もり方針に合わせた保守的な数え方）
    F@P: 27、(FP)@F^T: 27、V@Q: 12、(VQ)@V^T: 18 の合計 84
    → 合計 **約 97 回の乗算**（`var_dtheta = sigma_gyro^2*dt^2` は
    実行時の値に依存しない定数なので `reset()` 時に 1 回だけ計算し、
    ここには数えていない）。
  - 必要 RAM: 状態 (x, y, theta) 3 個 + 共分散 P の 9 個 = 12 個の単精度浮動小数点数
    = **48 バイト**（note_037 §13-2 の「状態と共分散で 100 バイト弱」という
    4 状態版の見積もりより少ない。3 状態のぶん小さい）。
    ヤコビアン F(3×3)・V(3×2)・過程雑音 Q(2×2) と一時行列積の作業領域は、
    実機実装ではスタック上の一時変数（呼び出しが終われば解放される）で
    足りるため、常駐 RAM には数えない。
  - 168 MHz・単精度演算器ありの構成での見込み負荷率: 乗算・加算合わせて
    数百サイクル、三角関数 2 回（ハードウェア三角関数命令を持たない
    Cortex-M4F 級では多項式近似で 1 回あたり概ね 100〜150 サイクル）を
    加えても概算 500〜600 サイクル/周期。1 kHz 制御では 1 周期に
    168,000 サイクル使えるので、**負荷率は 1% を大きく下回る**
    （0.3〜0.4% 程度）。→ **余裕で載る**。
"""
import math
from typing import Optional, Tuple

import numpy as np

from mouse.params import RobotParams

# ==========================================================================
# 過程雑音の既定値（暫定。実測で校正するのは次の作業 — E2 以降で観測との
# 突き合わせができるようになってから校正する。ここに書く根拠は
# 「桁を外していない」ことの説明であって、実測較正の代わりではない）
# ==========================================================================

# alpha1: 並進の過程雑音係数（無次元。var_ds = alpha1 * ds**2 の比例係数、
# つまり「1 制御周期の並進量に対する標準偏差の比率」の 2 乗）。
# 根拠: `classic/localization.py` 冒頭に実測記録がある「maze_41038: 10 区画
# (=1.8m) の連続直進で横方向ドリフトが 26〜28mm 蓄積」という既存の実測値を、
# 桁を合わせるための当たりとして使う。27mm/1800mm ≈ 1.5% を「1 制御周期あたりの
# 並進の標準偏差が並進量の 1.5%」という比率に読み替えると
# alpha1 = 0.015**2 ≈ 2.25e-4。ここでは 2.25e-4 をそのまま既定値にする。
# 🔴 この読み替えは横方向ドリフト（主にヨー推定誤差の積分で生じる）を
# 並進の雑音（alpha1 の担当）に単純に流用したものであり、桁の見当をつける
# ためだけの近似である。E2 で観測更新が入り、推定誤差を直接測れるように
# なってから実測較正すること。
ALPHA1_DEFAULT: float = 2.25e-4

# alpha2: 旋回に伴って生じる並進の過程雑音係数（単位 m^2/rad^2。
# var_ds = alpha2 * dtheta**2 の比例係数）。超信地旋回は車輪が左右逆回転する
# ため、旋回中心が車体中心からずれる誤差（車輪径のばらつき・トレッドの
# 較正誤差）が並進側ににじみ出る。根拠となる実測値が無いため、
# 機体自身の寸法尺度（トレッド `tread=0.072m`・車輪半径
# `wheel_radius=0.0135m`）と同程度の 1cm を「1 rad 旋回あたりに生じる
# 並進の標準偏差」の当たりとして採用する: sqrt(alpha2) ≈ 0.01m
# → alpha2 = 0.01**2 = 1.0e-4。
# 🔴 これは実測に基づかない工学的な当てずっぽうであり、暫定値である。
ALPHA2_DEFAULT: float = 1.0e-4

# sigma_gyro: ジャイロの雑音密度相当の標準偏差 [rad/s]。
# var_dtheta = sigma_gyro**2 * dt**2 は「観測が無い間に方位の不確かさが
# どれだけ育つか」を決める。本シミュレータのジャイロは既定で無雑音
# （`mouse/sim.py` の `noise_std=None`）であり、note_037 §1 決定2により
# 当面センサ劣化を入れない方針のため、この値は「もし実機の安価な MEMS
# ジャイロ相当のノイズがあったら」という保守的な想定にすぎない。
# 民生 MEMS ジャイロ（マイクロマウスで広く使われる ICM-20602 等）の
# ノイズ密度は概ね 0.007〜0.01 [°/s/√Hz] 級であり、これを 100Hz 制御帯域で
# ざっくり見積もると 0.5°/s ≈ 0.00873 rad/s 程度になる。
# 🔴 実測（シミュレータのジャイロに雑音を入れる、または実機データ）で
# 校正するまでの暫定値である。
SIGMA_GYRO_DEFAULT: float = math.radians(0.5)  # ≈ 0.00873 rad/s


class PoseEstimator:
    """姿勢 (x, y, theta) の信念を推測航法で予測するだけの推定器（E1 段階）。

    観測による更新は持たない（E2 以降で追加）。ジャイロバイアスも状態に
    含めない（1 実験 1 変更。E2 で追加）。

    使い方:
        est = PoseEstimator(params)
        est.reset(x=0.09, y=0.09, theta=math.pi / 2)   # 既定値と同じ
        while True:
            obs = sim.observation()
            est.predict(obs)
            x, y, theta = est.pose
    """

    def __init__(self, params: Optional[RobotParams] = None, *,
                 alpha1: float = ALPHA1_DEFAULT,
                 alpha2: float = ALPHA2_DEFAULT,
                 sigma_gyro: float = SIGMA_GYRO_DEFAULT,
                 x: float = 0.09, y: float = 0.09, theta: float = math.pi / 2):
        self.params = params if params is not None else RobotParams()
        # 単精度で持つ（マイコン実装を前提にした設計。§13-1「数値は単精度で
        # 書く」）。alpha1/alpha2/sigma_gyro はコンストラクタでのみ与える
        # 前提とし（後から書き換える setter は用意しない）、変わらない前提で
        # var_dtheta を reset() 時に定数として 1 回だけ計算する。
        self.alpha1 = np.float32(alpha1)
        self.alpha2 = np.float32(alpha2)
        self.sigma_gyro = np.float32(sigma_gyro)

        # 距離センサの本数はハードコードせず params から取る
        # （classic/motion.py の _split_obs と同じやり方）。
        self._n_sensors = len(self.params.sensors)

        # ------------------------------------------------------------
        # 動的確保をループ内で行わないための使い回しバッファ（すべて単精度）。
        # F・V の「変わらない」成分（0 と 1）はここで一度だけ書き込み、
        # predict_raw() では変化する成分だけを上書きする。
        # ------------------------------------------------------------
        self._F = np.eye(3, dtype=np.float32)          # 状態ヤコビアン（対角=1 は固定）
        self._V = np.zeros((3, 2), dtype=np.float32)   # 入力ヤコビアン
        self._V[2, 1] = np.float32(1.0)                 # dtheta -> theta は常に 1（固定）
        self._Q = np.zeros((2, 2), dtype=np.float32)   # 過程雑音（入力空間）
        # 行列積の作業領域（毎回上書きするので初期値は不問）
        self._FP = np.zeros((3, 3), dtype=np.float32)
        self._VQ = np.zeros((3, 2), dtype=np.float32)
        self._VQVT = np.zeros((3, 3), dtype=np.float32)

        self.reset(x, y, theta)

    # ------------------------------------------------------------------
    # 初期化
    # ------------------------------------------------------------------
    def reset(self, x: float = 0.09, y: float = 0.09, theta: float = math.pi / 2) -> None:
        """姿勢と共分散を初期化する。

        既定値は競技の発進状態（区画 (0,0) の中心 = (0.09, 0.09)、北向き
        = +90°。根拠は `docs/COORDINATE_SYSTEM.md` §6・§7）。

        初期姿勢は「発進時にあらかじめ分かっているプロトコル上の定数」
        として与えるので、初期共分散は 0 にする（`classic/motion.py` の
        `reset(heading_deg=90.0)` が初期ヨー角を真値取得ではなく既知定数
        として扱うのと同じ考え方）。
        """
        self._x = np.float32(x)
        self._y = np.float32(y)
        self._theta = np.float32(self._wrap(float(theta)))
        self._P = np.zeros((3, 3), dtype=np.float32)

        # var_dtheta = sigma_gyro**2 * dt**2 は実行時の値（ds, dtheta, ...）に
        # 一切依存しない定数なので、ここで 1 回だけ計算して Q[1,1] に置く
        # （predict_raw() の乗算回数から除外できる。マイコン予算 §13-4 参照）。
        dt = np.float32(self.params.control_dt)
        self._Q[1, 1] = self.sigma_gyro * self.sigma_gyro * dt * dt

    # ------------------------------------------------------------------
    # 読み出し
    # ------------------------------------------------------------------
    @property
    def pose(self) -> Tuple[float, float, float]:
        """(x, y, theta) を返す（単位 m, m, rad）。"""
        return float(self._x), float(self._y), float(self._theta)

    @property
    def covariance(self) -> np.ndarray:
        """共分散 P（3×3, np.float32）のコピーを返す。
        呼び出し側が書き換えても内部状態を汚さないよう、常にコピーを渡す。"""
        return self._P.copy()

    @property
    def cell(self) -> Tuple[int, int]:
        """推定姿勢から決まる区画 (int, int)。

        `floor(x / cell_size), floor(y / cell_size)`。区画境界は下側を含む
        （x=cell_size ちょうどは次の区画に属する）。**区画へ落とすのはここだけ**
        にし、内部の予測計算は常に連続の姿勢で行う（note_037 §14-2 制約1。
        斜め探索を締め出さないため）。
        """
        cell_size = self.params.cell_size
        return (int(math.floor(float(self._x) / cell_size)),
                int(math.floor(float(self._y) / cell_size)))

    # ------------------------------------------------------------------
    # 予測
    # ------------------------------------------------------------------
    def predict(self, obs: np.ndarray) -> None:
        """observation() から車輪角速度・ジャイロ z 成分を取り出して 1 ステップ予測する。

        並び（`mouse/sim.py:246-268`。`classic/motion.py` の `_split_obs` と
        同じ切り出し方）: [距離×n, accel×3, gyro×3, omega_L, omega_R]。
        距離センサの本数 n はハードコードせず `params.sensors` から取る。
        """
        n = self._n_sensors
        gyro_z = float(obs[n + 5])
        omega_l = float(obs[n + 6])
        omega_r = float(obs[n + 7])
        self.predict_raw(omega_l, omega_r, gyro_z)

    def predict_raw(self, omega_l: float, omega_r: float, gyro_z: float) -> None:
        """素の車輪角速度・ジャイロ値を直接渡す予測（検査・否定対照用）。

        真値 (`MouseSim.privileged_pose()`) は一切参照しない。引数として
        受け取るのは車輪角速度とジャイロだけである。
        """
        dt = np.float32(self.params.control_dt)
        r = np.float32(self.params.wheel_radius)

        ds = r * (np.float32(omega_l) + np.float32(omega_r)) * np.float32(0.5) * dt
        dtheta = np.float32(gyro_z) * dt

        theta = self._theta
        half = theta + dtheta * np.float32(0.5)
        # 三角関数はここで cos・sin を 1 回ずつ、計 2 回だけ呼ぶ
        # （1 制御周期あたりの呼び出し回数の上限。モジュール docstring参照）。
        c = np.float32(math.cos(float(half)))
        s = np.float32(math.sin(float(half)))

        # ---- 状態更新 ----
        self._x = self._x + ds * c
        self._y = self._y + ds * s
        self._theta = np.float32(self._wrap(float(theta) + float(dtheta)))

        # ---- ヤコビアン F（状態に関する）・V（入力 (ds, dtheta) に関する）----
        # dsc, dss は F と V の両方で使い回す（乗算回数を減らす）。
        dsc = ds * c
        dss = ds * s
        self._F[0, 2] = -dss
        self._F[1, 2] = dsc
        # F[0,0]=F[1,1]=F[2,2]=1、他の非対角=0 は __init__ で設定済みで変わらない。

        self._V[0, 0] = c
        self._V[1, 0] = s
        self._V[0, 1] = -dss * np.float32(0.5)
        self._V[1, 1] = dsc * np.float32(0.5)
        # V[2,0]=0, V[2,1]=1 は __init__ で設定済みで変わらない。

        # ---- 過程雑音 Q（入力空間、対角）----
        # var_dtheta は reset() で計算済みの定数なので Q[1,1] は触らない。
        self._Q[0, 0] = self.alpha1 * ds * ds + self.alpha2 * dtheta * dtheta

        # ---- 共分散伝播 P = F P F^T + V Q V^T ----
        # 動的確保をループ内で行わないため、あらかじめ確保済みの作業領域
        # (_FP, _VQ, _VQVT) へ out= で書き込む（F.T/V.T は転置ビューであり
        # コピーを作らない）。
        np.matmul(self._F, self._P, out=self._FP)
        np.matmul(self._FP, self._F.T, out=self._P)
        np.matmul(self._V, self._Q, out=self._VQ)
        np.matmul(self._VQ, self._V.T, out=self._VQVT)
        self._P += self._VQVT

    # ------------------------------------------------------------------
    # 角度の正規化
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap(angle: float) -> float:
        """角度を [-pi, pi) に正規化する（下側 -pi を含み、+pi は含まない）。

        三角関数を使わず剰余演算だけで書く（`atan2(sin, cos)` 方式は
        1 回の正規化に三角関数を 3 回消費し、1 制御周期あたり 2 回という
        マイコン実装の予算をこの関数だけで超えてしまう）。
        マイコン実装では `fmodf`（単精度版）に置き換える。

        境界の規約: `angle == +pi` はちょうど `-pi` に正規化される
        （`docs/COORDINATE_SYSTEM.md` §10-3b が記録する「角度の分枝の事故」を
        踏まえ、この関数の境界の振る舞いは
        `tests/test_classic_pose.py::test_theta_wraps_at_the_exact_boundary`
        が固定する）。
        """
        two_pi = 2.0 * math.pi
        wrapped = math.fmod(angle + math.pi, two_pi)
        if wrapped < 0.0:
            wrapped += two_pi
        return wrapped - math.pi
