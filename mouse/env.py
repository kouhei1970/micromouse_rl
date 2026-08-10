"""
mouse/env.py
================
MouseMazeEnvV2: Task4 ベンチマーク・今後の学習の土台となる最小 Gymnasium 環境。
報酬設計は M1 以降の実験カードで行うため、ここでは reward は常に 0.0。

センサ劣化フック（U3 対応。docs/MODEL_VERIFICATION_PLAN.md §5 手順3）:
    観測パイプラインに 量子化 → 一次遅れ → むだ時間FIFO → 加法性ガウスノイズ の順で
    劣化を適用するフックを備える（SensorDegradation で設定）。
    全パラメータ既定（None）のときはフック無し経路と bit-exact に一致する
    （これが最重要要件。tests/test_env_hooks.py で検証）。
"""
from dataclasses import dataclass
from typing import Optional, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from mouse.sim import MouseSim
from mouse.params import RobotParams

# センサ劣化フック専用 RNG のシード分離用ソルト（sim.rng や gym の np_random と
# 乱数列が一致しないようにするための固定タグ。値そのものに物理的意味はない）。
_DEGRADATION_RNG_SALT = 0x5E4507

# スカラー（全チャネル共通値としてブロードキャスト）または観測ベクトルと同じ長さの
# 配列（チャネルごと個別指定）のどちらでも受け付ける、という意味の型エイリアス。
ArrayLike = Union[float, int, np.ndarray, list, tuple]


@dataclass
class SensorDegradation:
    """観測パイプラインのセンサ劣化フック設定（U3 対応、計画書 §5 手順3）。

    適用順序は固定: 量子化 → 一次遅れ → むだ時間FIFO(dead-time) → 加法性ガウスノイズ。
    各フィールドは None のとき、そのステージ全体を素通し（計算自体を行わない）にする。
    **全フィールドが既定の None のとき、パイプライン全体が恒等写像になり、
    フック無し時の観測と bit-exact に一致する**（最重要要件）。

    各パラメータはスカラー（全チャネル共通値としてブロードキャスト）または
    観測ベクトルと同じ長さの配列（チャネルごと個別指定）のどちらでも渡せる。
    「チャネル群ごと」に一部だけ効かせたい場合は、対象外チャネルの要素に
    0（量子化・むだ時間・ノイズは無効化、一次遅れは τ=0 で実質素通し）を
    入れた配列を渡せばよい。観測の並びは mouse/sim.py の
    MouseSim.observation() のレイアウトに一致する:
    [distance × N_sensors(既定6本), accel × 3, gyro × 3, wheel_omega × 2]。

    Attributes:
        quantize_step: 量子化ステップ幅（観測と同じ物理単位）。
            round(x/step)*step で丸める。要素が 0 のチャネルは量子化しない。
        lag_tau: 一次遅れ時定数 τ [s]。制御周期 dt(= RobotParams.control_dt) で
            y[k] = α·x[k] + (1−α)·y[k−1]、α = dt/(τ+dt) と離散化する。
            要素が 0 のチャネルは α=1 となり実質素通し。
        delay_steps: むだ時間 [制御ステップ数]（非負整数）。要素が 0 の
            チャネルはむだ時間なし。制御ステップ単位の FIFO で実装する。
        noise_std: 加法性ガウスノイズの標準偏差（観測と同じ物理単位）。
            要素が 0 のチャネルはノイズを加えない。
    """
    quantize_step: Optional[ArrayLike] = None
    lag_tau: Optional[ArrayLike] = None
    delay_steps: Optional[ArrayLike] = None
    noise_std: Optional[ArrayLike] = None

    def is_noop(self) -> bool:
        """全ステージが無効（恒等写像）かどうか。"""
        return (self.quantize_step is None and self.lag_tau is None
                and self.delay_steps is None and self.noise_std is None)


class MouseMazeEnvV2(gym.Env):
    """マイクロマウス v2 の最小 Gymnasium 環境。

    XML はファイルパスで受け取る（SubprocVecEnv で pickle 可能にするため。
    MuJoCo モデル/データそのものは pickle 不可なので、コンストラクタ引数は
    文字列パスに限定し、__init__ 内で MouseSim を都度構築する）。

    sensor_degradation（SensorDegradation, 既定 None）でセンサ劣化フックを
    有効化できる。既定 None（またはフィールド全て None の SensorDegradation()）
    では観測パイプラインは恒等写像であり、フック無し時の観測と bit-exact に
    一致する。
    """
    metadata = {"render_modes": []}

    def __init__(self, xml_path, max_steps: int = 30000, noise_std=None, seed=None,
                 sensor_degradation: Optional[SensorDegradation] = None):
        super().__init__()
        self.xml_path = str(xml_path)
        self.max_steps = max_steps
        self.params = RobotParams()

        self.sim = MouseSim(self.xml_path, params=self.params, noise_std=noise_std, seed=seed)

        # action: Box(-1,1,(2,)) -> ×3.0 で電圧
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # observation: 次元は MouseSim の観測（距離センサ本数 n + accel3 + gyro3 + 車輪2）
        # に一致させる。r6 でセンサが 6→4 本になり 14→12 次元へ変わったため、
        # 固定値ではなく実際の観測長から導出する（構成変更で API 契約が壊れるのを防ぐ）
        obs_dim = int(self.sim.observation().shape[0])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                             shape=(obs_dim,), dtype=np.float32)

        self._step_count = 0

        # --- センサ劣化フック（U3 対応。既定 None = 全ステージ無効） ---
        self.sensor_degradation = sensor_degradation
        self._degradation_rng = self._make_degradation_rng(seed)
        # フィルタ/FIFO の内部状態。観測次元は実データから取得するため（ハードコード
        # 禁止）reset() で最初の observation() を得てから遅延初期化する。
        self._degradation_state = None

    @staticmethod
    def _make_degradation_rng(seed):
        """センサ劣化フック専用の乱数生成器を作る。

        self.sim.rng（既存の観測ノイズフック）や gym の self.np_random とは
        独立の Generator インスタンスとする（計画書 §5 手順3 要件5）。
        同じ seed から再現可能だが、_DEGRADATION_RNG_SALT で系列をずらして
        あるため他の乱数列とは相関しない。
        """
        if seed is None:
            return np.random.default_rng()
        return np.random.default_rng(np.random.SeedSequence([int(seed), _DEGRADATION_RNG_SALT]))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.sim.rng = np.random.default_rng(seed)
            self._degradation_rng = self._make_degradation_rng(seed)
        self.sim.full_reset()
        self._step_count = 0
        # 劣化フックの内部状態（一次遅れ・むだ時間FIFO）はエピソード毎に初期化する
        # （前エピソードの軌道に由来する状態を持ち越さない）。
        self._degradation_state = None
        raw_obs = self.sim.observation()
        obs = self._apply_sensor_degradation(raw_obs).astype(np.float32)
        return obs, {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        v_left = float(action[0]) * self.params.voltage_limit
        v_right = float(action[1]) * self.params.voltage_limit

        result = self.sim.step_control(v_left, v_right)
        self._step_count += 1

        raw_obs = self.sim.observation()
        obs = self._apply_sensor_degradation(raw_obs).astype(np.float32)
        # 報酬設計は M1 以降の実験カードで行う。ここでは常に 0.0。
        reward = 0.0
        terminated = bool(result['tipped'] or result['collision'])
        truncated = bool(self._step_count >= self.max_steps)

        info = dict(result)
        return obs, reward, terminated, truncated, info

    def render(self):
        # render は未実装（NotImplementedError にはしない）
        return None

    def close(self):
        pass

    # ------------------------------------------------------------------
    # センサ劣化フック（U3 対応。計画書 §5 手順3）
    # ------------------------------------------------------------------
    def _apply_sensor_degradation(self, raw_obs: np.ndarray) -> np.ndarray:
        """観測パイプラインにセンサ劣化フックを適用する。

        適用順: 量子化 → 一次遅れ → むだ時間FIFO → 加法性ガウスノイズ。
        self.sensor_degradation が None、またはその全フィールドが None の
        ときは raw_obs をそのまま返す（フック無し経路と bit-exact に一致。
        追加の配列コピーすら行わない）。
        """
        deg = self.sensor_degradation
        if deg is None or deg.is_noop():
            return raw_obs

        obs_dim = raw_obs.shape[0]
        state = self._degradation_state
        if state is None:
            state = self._init_degradation_state(deg, obs_dim, raw_obs)
            self._degradation_state = state

        out = raw_obs

        # 1. 量子化: round(x/step)*step。step==0 のチャネルは素通し
        #    （0 除算を避けるため、除数だけ一時的に 1.0 に差し替えてから selectする）。
        if state['quantize_step'] is not None:
            step = state['quantize_step']
            safe_step = np.where(step != 0, step, 1.0)
            quantized = np.round(out / safe_step) * step
            out = np.where(step != 0, quantized, out)

        # 2. 一次遅れ（制御周期 100Hz 離散化）: y[k] = α x[k] + (1-α) y[k-1]
        #    τ=0 のチャネルは α=1 となり実質素通し。
        if state['lag_alpha'] is not None:
            alpha = state['lag_alpha']
            y_new = alpha * out + (1.0 - alpha) * state['lag_state']
            state['lag_state'] = y_new
            out = y_new

        # 3. むだ時間 FIFO（制御ステップ単位の循環バッファ）
        if state['delay_steps'] is not None:
            buf = state['delay_buffer']
            widx = state['delay_write_idx']
            buf[widx] = out
            cap = buf.shape[0]
            read_idx = (widx - state['delay_steps']) % cap
            out = buf[read_idx, np.arange(obs_dim)]
            state['delay_write_idx'] = (widx + 1) % cap

        # 4. 加法性ガウスノイズ（専用 RNG。std==0 のチャネルは決定論的に 0 加算）
        if state['noise_std'] is not None:
            out = out + self._degradation_rng.normal(0.0, state['noise_std'], size=obs_dim)

        return out

    def _init_degradation_state(self, deg: SensorDegradation, obs_dim: int,
                                 raw_obs: np.ndarray) -> dict:
        """SensorDegradation の各フィールドを obs_dim 長の配列に正規化し、
        フィルタ/FIFO の内部状態を初期化する。

        obs_dim は実際の観測配列（raw_obs.shape[0]）から取得しており、
        センサ本数や観測次元をコードにハードコードしない
        （cutoff・車輪半径等の定数と同様の方針。計画書 §5 手順3 要件3）。
        """
        dt = self.params.control_dt  # 制御周期 [s]（ハードコードせず params から動的取得）

        def _broadcast(value, name, dtype=np.float64, nonneg=False):
            if value is None:
                return None
            arr = np.asarray(value, dtype=dtype)
            if arr.ndim == 0:
                arr = np.full(obs_dim, arr.item(), dtype=dtype)
            if arr.shape != (obs_dim,):
                raise ValueError(
                    f"SensorDegradation.{name} の形状 {arr.shape} が観測次元 "
                    f"({obs_dim},) と一致しません。")
            if nonneg and np.any(arr < 0):
                raise ValueError(f"SensorDegradation.{name} に負の値は指定できません: {arr}")
            return arr

        quantize_step = _broadcast(deg.quantize_step, 'quantize_step', nonneg=True)
        lag_tau = _broadcast(deg.lag_tau, 'lag_tau', nonneg=True)
        delay_steps = _broadcast(deg.delay_steps, 'delay_steps', dtype=np.int64, nonneg=True)
        noise_std = _broadcast(deg.noise_std, 'noise_std', nonneg=True)

        lag_alpha = None
        lag_state = None
        if lag_tau is not None:
            lag_alpha = dt / (lag_tau + dt)  # α = dt/(τ+dt)
            # 初期状態 y[-1] はリセット時観測（定常仮定）。エピソード開始時点の
            # 人為的な過渡を作らないための選択（テスト時は理論再帰式で追跡可能）。
            lag_state = raw_obs.astype(np.float64).copy()

        delay_buffer = None
        delay_write_idx = 0
        if delay_steps is not None:
            capacity = int(delay_steps.max()) + 1
            # コールドスタート: リセット時観測で FIFO を満たしておく（バッファが
            # まだ埋まっていない最初の delay_steps ステップはこの値が返る）。
            delay_buffer = np.tile(raw_obs.astype(np.float64), (capacity, 1))

        return {
            'quantize_step': quantize_step,
            'lag_alpha': lag_alpha,
            'lag_state': lag_state,
            'delay_steps': delay_steps,
            'delay_buffer': delay_buffer,
            'delay_write_idx': delay_write_idx,
            'noise_std': noise_std,
        }
