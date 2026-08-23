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

    def __init__(self, params: Optional[RobotParams] = None,
                 extend_straights: bool = True, fast_mode: str = "command",
                 friction_use: float = 1.0, clearance_margin_m: float = 0.005,
                 wall_correction: bool = False,
                 map_source: str = "threshold",
                 belief_t_wall: Optional[float] = None,
                 belief_t_open: Optional[float] = None,
                 belief_update_every_tick: bool = True) -> None:
        self.params = params if params is not None else RobotParams()
        # S3 最短走行で直線を伸ばすかどうか（exp_024 PREREG §4 の 2 条件
        # "extended"/"percell"）。ClassicExplorer へそのまま渡すだけで、
        # 本クラス自身の挙動は変えない。
        self.extend_straights = bool(extend_straights)
        # S3': 最短走行(FAST/RETURN2)をコマンド方式("command"・既定)と
        # 速度プロファイル追従("profile")のどちらで走らせるか
        # （`research_notes/note_031_profile_planner_and_eta.md`、
        # `classic/explorer.py` モジュール docstring「S3': 最短走行の
        # プロファイル追従化」参照）。ClassicExplorer へそのまま渡すだけで、
        # 本クラス自身の挙動は変えない。
        self.fast_mode = str(fast_mode)
        # 摩擦円の使用率 u（既定 1.0 = 従来と同一）と壁センサ位置補正の
        # 有効/無効（既定 False = 従来と同一）。どちらも ClassicExplorer へ
        # そのまま渡すだけで、本クラス自身の挙動は変えない
        # （`classic/explorer.py` の `friction_use`/`wall_correction` 参照）。
        self.friction_use = float(friction_use)
        # 幾何の掃引と壁・柱のあいだに残す設計上の余裕（既定 0.005m(5mm) =
        # 従来と同一）。ClassicExplorer へそのまま渡すだけで、本クラス自身の
        # 挙動は変えない（`classic/explorer.py` の `clearance_margin_m` 参照）。
        self.clearance_margin_m = float(clearance_margin_m)
        self.wall_correction = bool(wall_correction)
        # 信念で走る (exp_035)。ClassicExplorer へそのまま渡すだけで、本クラス
        # 自身の挙動は変えない（`classic/explorer.py` モジュール docstring
        # 「信念で走る（map_source。exp_035）」参照）。
        self.map_source = str(map_source)
        self.belief_t_wall = belief_t_wall
        self.belief_t_open = belief_t_open
        self.belief_update_every_tick = bool(belief_update_every_tick)
        self._explorer: Optional[ClassicExplorer] = None
        # ティックごとの「そのとき何をしていたか」の識別子。
        # classic.checks.plan_adherence の入力（note_029 §4-1、型 C 再発防止）。
        self._plan_ids: List[str] = []
        # 走行が始まった瞬間（on_run_start 通知時点）の段階の列
        # （exp_024 PREREG §4 の run_phases。判定量 T_fast の入力）。
        # 🔴 記録するだけで、走行そのもの（電圧の計算）には一切関与しない
        # ことを tests/test_classic_policy.py の実測で示す。
        self._run_phases: List[dict] = []

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
        self._explorer = ClassicExplorer(
            width, height, params=self.params,
            extend_straights=self.extend_straights,
            fast_mode=self.fast_mode,
            friction_use=self.friction_use,
            clearance_margin_m=self.clearance_margin_m,
            wall_correction=self.wall_correction,
            map_source=self.map_source,
            belief_t_wall=self.belief_t_wall,
            belief_t_open=self.belief_t_open,
            belief_update_every_tick=self.belief_update_every_tick,
        )
        self._plan_ids = []
        self._run_phases = []

    def on_run_start(self, run_index: int) -> None:
        # 本方策は評価器の「走行」境界（スタート区画を出た瞬間）ではなく、
        # 区画中心への到達を根拠に自前で状態を進める（ClassicExplorer 参照）。
        # run 境界そのものに応じて内部状態を変える必要は無いので、走行制御
        # 自体は変えない。ここでは exp_024 の一次記録用に「走行が始まった
        # 瞬間の段階」を記録するだけである（PREREG §4 の run_phases）。
        # 🔴 記録のみ・電圧計算には触れない（tests/test_classic_policy.py の
        # test_on_run_start_does_not_affect_act_output で実測により確認）。
        if self._explorer is not None:
            self._run_phases.append({
                "run_index": int(run_index),
                "phase": self._explorer.phase.name,
            })

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

    def get_run_phases(self) -> List[dict]:
        """走行が始まった瞬間（on_run_start 通知時点）の段階の列。
        `[{"run_index": 1, "phase": "EXPLORE"}, ...]`（exp_024 PREREG §4）。
        `phase` は `classic.explorer.Phase` のメンバ名（"EXPLORE"/"RETURN"/
        "FAST"/"RETURN2"）の文字列。"""
        return list(self._run_phases)
