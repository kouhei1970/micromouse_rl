"""観測履歴の連結（exp_021・M2 第 2 弾 A）。

Observation-history concatenation wrapper (exp_021).

方策に「短期の文脈」を与えるための環境ラッパ。現在の観測に、指定した遅れ
（歩数）だけ過去の観測をそのまま連結して返す。

    obs_out(t) = [ o(t), o(t-lag_1), o(t-lag_2), ..., o(t-lag_k) ]

条文は `experiments/exp_021_observation_history/card.md` §3・§6-1 が正。要点:

- **遅れは 2 の冪**（既定 1,2,4,8,16,32,64,128）。制御周期 10 ms なので窓は 1.28 秒。
  等間隔で同じ窓を覆うには 128 時点（2,176 次元）要るところを 9 時点（153 次元）で覆う
- **エピソード開始時は最初の観測を全ての遅れに複製する**（0 埋めにすると
  「大きな変化があった」という偽の信号が最初の 1.28 秒に入る）
- **乱数を一切消費しない**（`np_random` に触れない）。exp_021 カード §2-3 の
  「不活性の bit 一致」は、この破れを実際に検出する設計になっている
- **`info` はそのまま素通しする**（`cell_entries`・`respawned`・`dist_to_goal` を落とさない）
- **gymnasium 1.x はラッパの属性転送を行わない**ので、外側から触られる属性は
  明示的に再公開する（`sim`・`set_d0_max`）

Notes:
    The wrapper is inert when not applied at all: `train.py` does not construct it
    unless `--obs-history` is given, so the code path is identical to exp_019.
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np

#: 既定の遅れ［歩］。exp_021 カード §3-1 で投入前に固定した組。
DEFAULT_LAGS = (1, 2, 4, 8, 16, 32, 64, 128)


def parse_lags(spec) -> tuple:
    """`"1,2,4,8"` 形式の文字列を遅れのタプルへ。空・None なら空タプル。

    Parse a comma-separated lag specification into a validated tuple.
    """
    if spec is None:
        return ()
    if isinstance(spec, (tuple, list)):
        vals = [int(v) for v in spec]
    else:
        s = str(spec).strip()
        if not s:
            return ()
        vals = [int(v) for v in s.split(",") if v.strip()]
    if not vals:
        return ()
    if any(v <= 0 for v in vals):
        raise ValueError(f"遅れは 1 以上の整数でなければならない: {vals}")
    if len(set(vals)) != len(vals):
        raise ValueError(f"遅れに重複がある: {vals}")
    return tuple(sorted(vals))


class ObsHistoryWrapper(gym.Wrapper):
    """観測に過去の観測を連結する（**乱数を消費しない**）。

    Concatenate lagged past observations to the current observation.

    Args:
        env: 元の環境（`mouse.maze6_env.Maze6Env` を想定）。
        lags: 遅れ［歩］の並び。**昇順・重複なし・すべて 1 以上**。
            空タプルは**使わない**（呼び出し側でラップしない）。
    """

    def __init__(self, env, lags=DEFAULT_LAGS, sham=False):
        super().__init__(env)
        # exp_022: にせ履歴。遅れの位置に**現在の観測を複製**する（次元もパラメータ数も
        # 同じまま、履歴の情報だけがゼロになる）。**既定 False ＝ exp_021 と完全に同一**。
        # sham history: replicate the current observation into every lag slot.
        self.sham = bool(sham)
        lags = parse_lags(lags)
        if not lags:
            raise ValueError("lags が空である。履歴を使わないならラップしないこと"
                             "（既定 off の経路を現行と完全に同一に保つため）")
        self.lags = lags
        self._depth = max(lags) + 1          # 現在（index 0）＋ 最大の遅れまで

        base = env.observation_space
        if len(base.shape) != 1:
            raise ValueError(f"1 次元の観測空間のみ対応: shape={base.shape}")
        self._n = int(base.shape[0])
        n_out = self._n * (1 + len(lags))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_out,), dtype=np.float32)

        self._buf: deque = deque(maxlen=self._depth)

    # ------------------------------------------------------------------
    # gymnasium 1.x はラッパの属性転送を行わない。外側から触られるものだけ
    # **明示的に**再公開する（暗黙の転送を作らない ＝ 誤りを隠さない）。
    # Explicit re-export: gymnasium 1.x wrappers do not forward attributes.
    # ------------------------------------------------------------------
    @property
    def sim(self):
        """`mouse/maze6_eval.py:75-76` が `env.sim` へ直接触る。"""
        return self.env.sim

    def set_d0_max(self, d0_max) -> None:
        """距離カリキュラム（exp_020）の切り替え。SB3 の `env_method` から呼ばれる。"""
        return self.env.set_d0_max(d0_max)

    # ------------------------------------------------------------------
    def _stack(self) -> np.ndarray:
        if self.sham:
            # にせ履歴（exp_022）: 遅れの位置も現在の観測にする。
            # **履歴の入れ物（_buf）の更新はやめない** — やめると reset/step の
            # 計算量が変わり、乱数以外の差が 1 つ増えるため（カード §5-1）。
            parts = [self._buf[0]] * (1 + len(self.lags))
        else:
            parts = [self._buf[0]] + [self._buf[lag] for lag in self.lags]
        return np.concatenate(parts).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.asarray(obs, dtype=np.float32)
        # 最初の観測を全ての遅れに複製する（条文・§3-1）
        self._buf.clear()
        for _ in range(self._depth):
            self._buf.append(obs)
        return self._stack(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        self._buf.appendleft(obs)            # index 0 = 現在・index k = k 歩前
        return self._stack(), reward, terminated, truncated, info
