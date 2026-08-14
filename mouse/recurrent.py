"""再帰型方策の道具（exp_023・M2 第 2 弾 B）。

Tools for the recurrent policy experiment (exp_023).

本ファイルが提供するもの:

1. **`RespawnResetRecurrentPPO`** — **リスポーン（衝突による立て直し）の歩でも
   LSTM の隠れ状態をリセットする** `RecurrentPPO` の部分クラス（**群 2 用**）
2. **`RespawnFlagCallback`** — 毎歩 `info["respawned"]` を拾って上のクラスへ渡すコールバック
3. **`RecurrentPolicyFn`** — 評価・測定で使う**隠れ状態を持ち回る方策の呼び出し口**

## なぜ部分クラスが要るか（`sb3-contrib` 2.9.0 のコードで確認済み）

| 該当箇所 | 内容 |
|---|---|
| `sb3_contrib/common/recurrent/policies.py:197-204` | リセットは **`(1.0 - episode_start) * lstm_states`** ＝ **`episode_start` が 1 の歩でのみ 0 にする** |
| `sb3_contrib/ppo_recurrent/ppo_recurrent.py:298` | **`self._last_episode_starts = dones`** |
| `mouse/maze6_env.py:1215` | **`terminated = goal_reached or (physical_fail and not respawned)`** |

**リスポーンした歩は終端にならないので `dones` は偽。既定では隠れ状態はリセットされない
＝ 汚染される側である。**

**`callback.on_step()` は `self._last_episode_starts = dones` より前に呼ばれる**ので、
**コールバックから直接は書き換えられない。**そこで **`_last_episode_starts` を property にして、
setter でコールバックが記録したリスポーン旗を論理和で混ぜる**。
**`collect_rollouts` の全複製を避けるための最小の介入**である。

**⚠️ 検査**: **リスポーンの歩で `episode_starts` が実際に 1 になっていること**（`tests/test_recurrent.py` の
T-R2）と、**群 1 の設定では素の `RecurrentPPO` と bit 一致すること**（同 T-R6）を投入前に確かめる。
"""
from __future__ import annotations

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback

#: exp_023 で使う LSTM の隠れ状態の大きさ（カード §0-2 の裁定）。
#: **`sb3-contrib` の既定 256 は使わない** — 方策のパラメータが対照の 12.92 倍になり、
#: exp_022 が実証した「容量だけで浅い裾が 9.3 倍」の再演になってしまうため。
DEFAULT_LSTM_HIDDEN = 32


class RespawnResetRecurrentPPO(RecurrentPPO):
    """リスポーンの歩でも隠れ状態をリセットする `RecurrentPPO`（exp_023 群 2）。

    Reset the LSTM hidden state on respawn steps, not only on episode boundaries.

    **`RespawnFlagCallback` と対で使う。**コールバックが `_pending_respawn` へ
    毎歩の旗を置き、`_last_episode_starts` の setter がそれを論理和で混ぜる。
    **旗が置かれていなければ（コールバックを付けなければ）素の `RecurrentPPO` と同じ挙動**である。
    """

    #: コールバックが毎歩置く旗（形は (n_envs,)）。setter が使ったら None へ戻す。
    _pending_respawn = None
    #: property の実体（`_last_episode_starts` の中身）
    _starts_store = None

    @property
    def _last_episode_starts(self):
        return self._starts_store

    @_last_episode_starts.setter
    def _last_episode_starts(self, value):
        pending = self._pending_respawn
        if pending is not None and value is not None:
            value = np.logical_or(np.asarray(value, dtype=bool),
                                  np.asarray(pending, dtype=bool))
            self._pending_respawn = None
        self._starts_store = value


class RespawnFlagCallback(BaseCallback):
    """毎歩 `info["respawned"]` を拾って方策側へ渡す（exp_023 群 2）。

    `collect_rollouts` の中で `self._last_episode_starts = dones` が実行される**直前**に
    `callback.on_step()` が呼ばれる（`ppo_recurrent.py:257` と `:298`）ので、
    ここで置いた旗は同じ歩の setter が拾う。
    """

    def _on_training_start(self) -> None:
        # 🔴 素の `RecurrentPPO` に付けても、`_pending_respawn` がただのインスタンス属性に
        # なるだけで**例外も出ず黙って無害化される** ＝ **群 1 と群 2 の取り違えに気づけない**。
        # 投入前検証で見つかった型なので、ここで落とす（2026-08-15）。
        if not isinstance(self.model, RespawnResetRecurrentPPO):
            raise TypeError(
                "RespawnFlagCallback は RespawnResetRecurrentPPO とだけ使える"
                f"（渡されたのは {type(self.model).__name__}）。"
                "素の RecurrentPPO に付けても旗は混ざらず、群 1 を群 2 と取り違える")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is not None:
            self.model._pending_respawn = np.array(
                [bool(i.get("respawned", False)) for i in infos], dtype=bool)
        return True


class RecurrentPolicyFn:
    """隠れ状態を持ち回る方策の呼び出し口（評価・測定用）。

    Stateful policy callable for evaluation of a recurrent policy.

    🔴 **これを忘れると LSTM は毎歩「エピソードの最初の 1 歩」として振る舞う。**
    **例外も出ず次元も合い、値だけが誤る**うえ、**「記憶を使えていない」という結果になるので
    帰無と区別がつかない**（准教授が投入前監査の最重の疑いとして登録した型）。

    **試行の切り替わりで必ず `reset()` を呼ぶこと**（`evaluate_maze6` の `policy_reset_fn`）。
    """

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = bool(deterministic)
        self._state = None
        self._episode_start = True

    def reset(self) -> None:
        """新しい試行の開始。隠れ状態を捨てる。"""
        self._state = None
        self._episode_start = True

    def __call__(self, obs):
        action, self._state = self.model.predict(
            obs, state=self._state,
            episode_start=np.array([self._episode_start]),
            deterministic=self.deterministic)
        self._episode_start = False
        return action
