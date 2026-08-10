"""
mouse/env.py
================
MouseMazeEnvV2: Task4 ベンチマーク・今後の学習の土台となる最小 Gymnasium 環境。
報酬設計は M1 以降の実験カードで行うため、ここでは reward は常に 0.0。
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from mouse.sim import MouseSim
from mouse.params import RobotParams


class MouseMazeEnvV2(gym.Env):
    """マイクロマウス v2 の最小 Gymnasium 環境。

    XML はファイルパスで受け取る（SubprocVecEnv で pickle 可能にするため。
    MuJoCo モデル/データそのものは pickle 不可なので、コンストラクタ引数は
    文字列パスに限定し、__init__ 内で MouseSim を都度構築する）。
    """
    metadata = {"render_modes": []}

    def __init__(self, xml_path, max_steps: int = 30000, noise_std=None, seed=None):
        super().__init__()
        self.xml_path = str(xml_path)
        self.max_steps = max_steps
        self.params = RobotParams()

        self.sim = MouseSim(self.xml_path, params=self.params, noise_std=noise_std, seed=seed)

        # action: Box(-1,1,(2,)) -> ×3.0 で電圧
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # observation: Box(-inf,inf,(14,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

        self._step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.sim.rng = np.random.default_rng(seed)
        self.sim.full_reset()
        self._step_count = 0
        obs = self.sim.observation().astype(np.float32)
        return obs, {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        v_left = float(action[0]) * self.params.voltage_limit
        v_right = float(action[1]) * self.params.voltage_limit

        result = self.sim.step_control(v_left, v_right)
        self._step_count += 1

        obs = self.sim.observation().astype(np.float32)
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
