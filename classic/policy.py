"""
classic/policy.py
================
`competition/policy_interface.py` の `MousePolicy` を、`classic/explorer.py`
の状態機械（探索→帰還→悲観判定）に接続する薄いアダプタ。

`requires_privileged = False` が本実装の要点である（note_030 §0「北極星」）:
評価器は `bind_sim`/`bind_maze` を呼ばない。この方策は `act(obs)` に渡される
観測（距離センサ・加速度・ジャイロ・車輪角速度）だけで走る。真の壁配列も
`mouse.sim.MouseSim.privileged_pose()/privileged_velocity()` も一切参照しない
（実際に参照していないことは `tests/test_classic_policy.py` の否定対照
N1/N2 で実測により確かめる）。
"""
from typing import List, Optional, Tuple

import numpy as np

from classic.explorer import ClassicExplorer
from competition.policy_interface import MousePolicy
from mouse.params import RobotParams


class ClassicExplorerPolicy(MousePolicy):
    """探索走行（ClassicExplorer）を評価器の口に合わせて駆動する方策。"""

    name = "classic_explorer"
    requires_privileged = False

    def __init__(self, params: Optional[RobotParams] = None) -> None:
        self.params = params if params is not None else RobotParams()
        self._explorer: Optional[ClassicExplorer] = None
        # ティックごとの「そのとき何をしていたか」の識別子。
        # classic.checks.plan_adherence の入力（note_029 §4-1、型 C 再発防止）。
        self._plan_ids: List[str] = []

    # ------------------------------------------------------------------
    # ライフサイクル通知フック
    # ------------------------------------------------------------------
    def on_maze_start(self, maze_info: dict) -> None:
        """迷路開始時に地図・状態機械を作り直す。

        maze_info の width/height は評価器が渡す非特権情報（迷路の外形寸法。
        `competition/policy_interface.py` の docstring参照）であり、真の壁
        配列そのものは含まれない。"""
        width = int(maze_info["width"])
        height = int(maze_info["height"])
        self._explorer = ClassicExplorer(width, height, params=self.params)
        self._plan_ids = []

    def on_run_start(self, run_index: int) -> None:
        # 本方策は評価器の「走行」境界（スタート区画を出た瞬間）ではなく、
        # 区画中心への到達を根拠に自前で状態を進める（ClassicExplorer 参照）。
        # run 境界そのものに応じて内部状態を変える必要が無いので no-op。
        pass

    def on_run_end(self, outcome: str) -> None:
        # 同上。ゴール到達／帰還は ClassicExplorer が自分の推定セル座標
        # （地図上の (0,0) や中央 2x2）で判定しており、評価器の outcome 通知
        # には依存しない。
        pass

    def on_retrieval(self) -> None:
        """係員回収の通知。スタートへ再配置された前提で位置推定を初期化する
        （地図は保持する）。"""
        if self._explorer is not None:
            self._explorer.handle_retrieval()

    # ------------------------------------------------------------------
    # 既知の壁地図の露出（描画・検分用の読み取り専用）
    # ------------------------------------------------------------------
    # 🔴 これは**方策が自分で作った地図**であり、真の壁情報ではない。
    # 描画器（research_notes/scripts/_video_l0_common.py）が「マウスが今どこまで
    # 地図を作れたか」を表示するために読む。evaluator の v_walls / h_walls と
    # 同じ添字規約だが、値は WallState（0=未知・1=壁あり・2=壁なし）である。
    # 走行がまだ始まっていない（on_maze_start 前）ときは None を返す。

    @property
    def v_walls_known(self):
        """方策が把握している縦壁。**真値ではない。**"""
        return None if self._explorer is None else self._explorer.maze.v_walls

    @property
    def h_walls_known(self):
        """方策が把握している横壁。**真値ではない。**"""
        return None if self._explorer is None else self._explorer.maze.h_walls

    # ------------------------------------------------------------------
    # 制御則本体
    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray) -> Tuple[float, float]:
        if self._explorer is None:
            raise RuntimeError("on_maze_start() が呼ばれる前に act() が呼ばれました")
        vl, vr, plan_id = self._explorer.tick(obs)
        # 電圧を確定させたのと同じ呼び出しの中で識別子を記録する
        # （型 B: 検査対象と実行経路がずれることを避ける。note_029 §4-2）。
        self._plan_ids.append(plan_id)
        return vl, vr

    # ------------------------------------------------------------------
    # 検査用アクセサ
    # ------------------------------------------------------------------
    def get_plan_ids(self) -> List[str]:
        """これまでの act() 呼び出しで記録した、ティックごとの計画識別子の列。
        `classic.checks.plan_adherence` にそのまま渡せる。"""
        return list(self._plan_ids)
