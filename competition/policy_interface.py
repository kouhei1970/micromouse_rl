"""
competition/policy_interface.py
================
マイクロマウス競技評価器 (CompetitionEvaluator) が呼び出す「方策」の
共通インタフェース定義。

方策は observation（MouseSim.observation() の 14 次元: 距離センサ6 + 加速度3
+ 角速度3 + 車輪角速度2）を受け取り、左右モータ電圧 [V] を返すコールバック
オブジェクトである（100Hz で act() が呼ばれる）。

spec_evaluator.md §2 に定義された仕様どおりの薄い基底クラス。学習済み方策
（RL policy）や古典アルゴリズム（壁沿い法・最短経路探索など）は本クラスを
継承して実装する。
"""
from typing import Tuple

import numpy as np


class MousePolicy:
    """方策の基底クラス。observation(14次元) → 左右モータ電圧 [V] のコールバック。

    評価器から 100Hz（制御周期 control_dt=0.01s 毎）で act() が呼ばれる。
    走行境界・係員回収は on_run_start / on_run_end / on_retrieval フックで
    通知される（方策の内部状態はこれらのフックを使って更新すること）。

    Attributes:
        name: 結果 JSON・出力ディレクトリ名に使われる方策名。
        requires_privileged: True の場合のみ評価器が bind_sim / bind_maze を
            呼び、MuJoCo シミュレーション本体・真の壁情報（特権情報）を渡す。
            古典ベースライン（壁情報を使う最短経路法など）向け。学習済み
            RL 方策は通常 False のままでよい（observation のみで駆動する）。
            この値は結果 JSON の policy フィールドに記録される。
    """

    name: str = "unnamed"
    requires_privileged: bool = False

    # ------------------------------------------------------------------
    # 特権情報バインド（requires_privileged=True の場合のみ呼ばれる）
    # ------------------------------------------------------------------
    def bind_sim(self, sim) -> None:
        """評価対象の MouseSim インスタンスそのものを渡す（特権アクセス）。

        迷路ごとに MouseSim が再構築されるため、迷路開始時（on_maze_start
        より前）に毎回呼ばれる。"""
        pass

    def bind_maze(self, v_walls: np.ndarray, h_walls: np.ndarray) -> None:
        """真の壁情報（v_walls, h_walls）を渡す（特権: 古典ベースライン用）。
        非特権方策の observation には壁情報は一切含まれない。"""
        pass

    # ------------------------------------------------------------------
    # ライフサイクル通知フック
    # ------------------------------------------------------------------
    def on_maze_start(self, maze_info: dict) -> None:
        """迷路開始時に呼ばれる。内部状態（探索地図・走行計画など）の初期化は
        ここで行う。maze_info には maze_id・seed・xml_path・width・height 程度
        の非特権情報のみが入る（壁情報そのものは渡らない）。"""
        pass

    def on_run_start(self, run_index: int) -> None:
        """走行開始（ロボット中心がスタート区画を出た瞬間）の通知。
        run_index は 1 始まりの走行番号。"""
        pass

    def on_run_end(self, outcome: str) -> None:
        """走行終了の通知。outcome は
        "goal" | "collision" | "tipover" | "stuck" | "timeout" のいずれか。"""
        pass

    def on_retrieval(self) -> None:
        """係員回収（スタートへ再配置）された直後の通知。走行中の失敗による
        回収・FREE 状態中（帰路等）の回収の両方で呼ばれる。"""
        pass

    # ------------------------------------------------------------------
    # 制御則本体
    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray) -> Tuple[float, float]:
        """obs（MouseSim.observation() の 14 次元）を受け取り、
        (v_left, v_right) [V] を返す。サブクラスで実装必須。"""
        raise NotImplementedError("MousePolicy のサブクラスは act() を実装してください")
