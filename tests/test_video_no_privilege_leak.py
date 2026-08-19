"""tests/test_video_no_privilege_leak.py — 描画ラッパーが特権情報を漏らさないこと

**なぜこの検査が要るか**

発表用の動画は「このマウスはセンサだけで走っている」と主張する。
その主張は、描画のために評価器へ差し込むラッパー（`RecordingPolicyWrapper`）が
**方策へ特権情報を転送していないこと**に依存する。

2026-08-19 の是正前、ラッパーは `inner.requires_privileged` を見ずに
`bind_sim` / `bind_maze` を無条件で転送していた。実害は無かった
（`MousePolicy` の受け口が空実装なので、要求していない方策は何も受け取らない）が、
**構成としては開いていた。**主張を構成で保証するために閉じ、その状態を本検査で固定する。

note_029 の教訓「壊して変わらなければ、その経路は使われていない」に倣い、
**転送が起きたら鳴る**形で書く（発火側）。あわせて、特権を要求する方策には
ちゃんと渡ること（空振り側）も確かめる。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research_notes" / "scripts"))

from _video_l0_common import RecordingPolicyWrapper  # noqa: E402

from competition.policy_interface import MousePolicy  # noqa: E402


class _SensorOnlyPolicy(MousePolicy):
    """新実装と同じ立場の方策。特権を要求しない。"""

    name = "sensor_only"
    requires_privileged = False

    def __init__(self) -> None:
        self.got_sim = False
        self.got_maze = False

    def bind_sim(self, sim) -> None:
        self.got_sim = True          # 渡されたら記録する（渡ってはいけない）

    def bind_maze(self, v_walls, h_walls) -> None:
        self.got_maze = True

    def act(self, obs):
        return 0.0, 0.0


class _PrivilegedPolicy(_SensorOnlyPolicy):
    """特権を要求する方策（対照）。こちらには渡らなければならない。"""

    name = "privileged"
    requires_privileged = True


class _FakeSim:
    sim_time = 0.0


def _wrap(inner):
    return RecordingPolicyWrapper(inner, frame_cb=lambda *a, **k: None,
                                  run_event_cb=lambda *a, **k: None)


def test_sensor_only_policy_receives_no_privileged_information():
    """🔴 発火側: 特権を要求していない方策へ転送されたら鳴ること。

    これが破れると、動画の「センサだけで走っている」という主張が崩れる。
    """
    inner = _SensorOnlyPolicy()
    w = _wrap(inner)
    w.bind_sim(_FakeSim())
    w.bind_maze(np.zeros((17, 16), dtype=np.int8), np.zeros((16, 17), dtype=np.int8))

    assert not inner.got_sim, "requires_privileged=False の方策に sim が渡された"
    assert not inner.got_maze, "requires_privileged=False の方策に真の壁情報が渡された"
    # ラッパー自身は描画のために sim を持つ（これは正しい。カメラであって入力ではない）
    assert w.sim is not None


def test_privileged_policy_still_receives_everything():
    """空振り側: 特権を要求する方策にはちゃんと渡ること（恒真な検査ではない）。"""
    inner = _PrivilegedPolicy()
    w = _wrap(inner)
    w.bind_sim(_FakeSim())
    w.bind_maze(np.zeros((17, 16), dtype=np.int8), np.zeros((16, 17), dtype=np.int8))

    assert inner.got_sim
    assert inner.got_maze


def test_the_new_classic_policy_declares_no_privilege():
    """新実装が実際に requires_privileged=False であること。

    宣言が True へ戻れば、上の検査は通ってしまう（転送が正当になるため）。
    宣言そのものを固定する。
    """
    from classic.policy import ClassicExplorerPolicy

    assert ClassicExplorerPolicy.requires_privileged is False
