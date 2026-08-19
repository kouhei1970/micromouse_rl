"""
classic/explorer.py
================
探索走行の状態機械。実機のマイクロマウスが実際にやっていることを写す
（`research_notes/note_030_classical_rebuild_plan.md` §1・§2 L4）。

    スタート区画から出発 → 区画の中心に来るたびに壁センサで前・左・右を読んで
    地図へ書く → 楽観の歩数マップ（未知は通れる）でゴール方向の次の1区画を選ぶ
    （足立法）→ ゴール到達 → スタートへ帰還（このときも地図を更新する）→
    帰還後、悲観の歩数マップ（未知は通れない）で「既知の壁だけでゴールへ
    到達できるか」を判定する。到達できれば探索完了。できなければもう一度
    ゴールを目指す（＝未知の区画を詰めに行く）。

🔴 真値は一切使わない。
  - 位置・方位は `classic/motion.py` の `CellMotionController` が持つ
    推測航法（車輪角速度＋ジャイロの積算）のみで管理する。ここで言う
    「区画 (x,y)」は `mouse.sim.MouseSim.privileged_pose()` を読んで求める
    ものではなく、**踏んだコマンドの列を数え上げているだけ**（直進1区画が
    `CellMotionController.update()` の done=True で完了した回数だけ、
    そのときの向きへ 1 区画分カウンタを進める。旋回も同様に「その場旋回が
    done した」ことだけを根拠に向きを更新する）。位置を実測して答え合わせを
    するのはテストの中だけである。
  - 壁は `classic/sensing.py` の `sense_walls()` の判定だけを見て書く。
    `competition/evaluator.py` が持つ真の壁配列（v_walls/h_walls）は
    一切参照しない。

🔴 AMBIGUOUS（曖昧判定）の扱い方針（`classic/sensing.py` docstring 参照）:
  側方センサ (LS/RS) のしきい値には実測で重なりが生じうる帯があり、
  `sense_walls()` はその帯を握りつぶさず AMBIGUOUS として返す。
  本モジュールは AMBIGUOUS を**そのまま未知（UNKNOWN）のまま**にし、
  地図へは書き込まない（WALL/OPEN で上書きしない）。

  ただし「未知のまま放置して終わり」ではなく、**再度その方向を向いたときに
  読み直す**。前方センサ (LF/RF) は実測で重なりがゼロ（`sensing.py` docstring
  「隙間 0.0688〜0.2142 m…重なりゼロ」）なので、**進行方向として選んだ側は
  必ずそちらへ旋回してから改めて壁を読み直し、前方センサとして確定させて
  から前進する**（`_on_stationary` は区画の中心にいる限り旋回後も毎回
  呼ばれるので、これは特別な処理ではなく通常の「区画中心での読み直し」が
  結果として旋回直後にも起きているだけである）。したがって「側方が曖昧
  だったので壁の有無を知らないまま前進して衝突する」ことは起きない設計に
  なっている。曖昧だった側の壁情報が、選ばれなかった場合はそのまま未知の
  ままになり、将来別の区画からその壁に近づいたとき（前方として、または
  曖昧でない側方として）に読み直されて確定する。
"""
from enum import Enum, auto
from typing import List, Optional, Tuple

from classic.flood import FloodMode, UNREACHABLE, compute_flood, is_passable, is_reachable_known
from classic.maze_map import ALL_DIRECTIONS, Direction, MazeMap
from classic.maze_map import WallState as MapWallState
from classic.motion import CellMotionController, MotionKind
from classic.sensing import WallSensing
from classic.sensing import WallState as SenseWallState
from classic.sensing import sense_walls
from mouse.params import RobotParams

Cell = Tuple[int, int]


class Phase(Enum):
    """探索走行の大局的な段階。"""

    EXPLORE = auto()  # ゴールを目指す（楽観歩数マップ）
    RETURN = auto()   # スタートへ帰る（楽観歩数マップ）。このときも地図を更新する
    DONE = auto()      # 悲観歩数マップでゴールへ到達可能と判定済み。探索完了


def _goal_cells(width: int, height: int) -> List[Cell]:
    """中央 2x2 ゴール領域のセル座標を返す。

    `competition/evaluator.py` の `goal_cells()` と同じ定義（中央 2x2 は
    競技規約で定められた既知情報であり、真値の読み取りではない。実機の
    マウスもルールブックでゴール位置の規約を知っている）。`classic/` は
    `competition/` へ依存させない（note_030 §2 の層構造）ため、ここに
    同じ計算をごく短く複製している。
    """
    gx0, gx1 = width // 2 - 1, width // 2
    gy0, gy1 = height // 2 - 1, height // 2
    return [(gx0, gy0), (gx0, gy1), (gx1, gy0), (gx1, gy1)]


# 同一区画に留まったまま次の方向を選び直せる回数の上限（note_030 §5 の
# 「想定外を静かに扱わない」を、想定外の**継続**についても適用したもの）。
# 1 区画の候補方位は N/E/S/W の高々4つなので、素直に「未知/曖昧を1つずつ
# 確かめて確定させる」だけなら数回で決着するはず。ここまで大きな値にして
# あるのは、地図が正しく更新されていく通常の探索では数回で収束するのを
# 妨げないようにしつつ、それでも収束しない場合（下記コメント参照）を
# 打ち切るための安全弁として十分な余裕を持たせるため。
MAX_REPLANS_PER_CELL = 8


class ClassicExplorer:
    """探索走行の状態機械本体。1 ティックごとに `tick(obs)` を呼ぶ。

    呼び出し側（`classic/policy.py` の `ClassicExplorerPolicy`）は
    `on_maze_start` 相当のタイミングで本クラスを構築し、以後は 100Hz で
    `tick(obs)` を呼んで得た (v_left, v_right) をそのまま `act()` の返り値
    にする。係員回収時は `handle_retrieval()` を呼ぶこと。
    """

    def __init__(self, width: int, height: int, params: Optional[RobotParams] = None) -> None:
        self.params = params if params is not None else RobotParams()
        self.maze = MazeMap(width, height)

        # スタート区画 (0,0)・初期方位 90°(北) は競技プロトコル上の既知定数
        # （評価器が reset_to_start(cell=(0,0), heading_deg=90.0) で置く、という
        # 規約であり真値の読み取りではない。classic/motion.py の docstring と
        # 同じ根拠）。
        self.start_cell: Cell = (0, 0)
        self._goal_cell_list: List[Cell] = _goal_cells(width, height)
        self._goal_cell_set = set(self._goal_cell_list)

        self.cell: Cell = self.start_cell
        self.heading: Direction = Direction.N
        self.phase: Phase = Phase.EXPLORE

        self.motion = CellMotionController(self.params)

        # 現在実行中（または実行直後）のコマンド種別・識別子。
        self._active_kind: MotionKind = MotionKind.STOP
        self._active_plan_id: str = "idle"
        self._pending_heading: Optional[Direction] = None

        # True の間は次の tick() で「区画中心に着いた」処理（センシング＋
        # 次コマンド決定）を行う。構築直後・係員回収直後に True にする。
        self._need_replan: bool = True

        # 楽観歩数マップ上でも経路が見えなくなった（推測航法の誤差の蓄積で
        # 壁を誤判定した疑い）場合の診断用メッセージ。None なら未発生。
        # `_on_stationary` 参照。係員回収のたびにクリアし、再挑戦の機会を
        # フェアに与える。
        self._blocked_reason: Optional[str] = None

        # 同一区画に留まったまま旋回だけを繰り返している回数
        # （`_on_stationary` の「同一区画での再計画」カウンタ）。
        # 区画を移動する(直進が完了する)たびに 0 へ戻す。
        self._replans_at_cell: int = 0

    # ------------------------------------------------------------------
    # 係員回収（外部から呼ばれる）
    # ------------------------------------------------------------------
    def handle_retrieval(self) -> None:
        """係員回収された直後に呼ぶ。

        評価器のプロトコル上、回収後は必ずセル(0,0)・方位90°(北)へ再配置
        される（`competition/evaluator.py` の `reset_to_start(cell=(0,0),
        heading_deg=90.0)`。これは規約上の既知定数であり、走行中の真値
        取得ではない — `classic/motion.py` の docstring と同じ理屈）。

        地図は保持したまま（実機も回収されて記憶を失うわけではない）、
        位置・方位の推定と進行中コマンドだけを初期化する。
        """
        self.cell = self.start_cell
        self.heading = Direction.N
        self.motion.reset(heading_deg=90.0)
        self._active_kind = MotionKind.STOP
        self._active_plan_id = "idle"
        self._pending_heading = None
        self._need_replan = True
        self._blocked_reason = None
        self._replans_at_cell = 0
        # phase はそのまま保持する。地図は生きているので、探索/帰還の続きから
        # 再開する（探索が既に完了 = DONE であればそのまま何もしない）。

    # ------------------------------------------------------------------
    # 1 制御ステップ
    # ------------------------------------------------------------------
    def tick(self, obs) -> Tuple[float, float, str]:
        """1 制御ステップ分の (v_left, v_right, plan_id) を返す。

        plan_id は「このティックでどの計画を実行していたか」の識別子
        （`classic.checks.plan_adherence` の入力。note_029 §4-1 の型 C
        再発防止）。電圧の計算と同じ呼び出しの中で確定させるので、
        両者が食い違う（型 B）余地が無い。
        """
        if self._need_replan:
            self._on_stationary(obs)
            self._need_replan = False

        vl, vr, done = self.motion.update(obs)

        if done and self._active_kind is not MotionKind.STOP:
            # 直進 1 区画／旋回のどちらかが完了した瞬間。位置・方位を確定させ、
            # 同じ区画中心で（あるいは旋回直後の同じ区画で）直ちに次の判断へ
            # 進む。判断そのものは計算のみでシミュレーション時間を消費しない
            # （実機の「止まって読んで決めてまた走る」を、判断部分は瞬時と
            # 単純化して再現している）。
            self._advance_state()
            self._on_stationary(obs)
            vl, vr, _done2 = self.motion.update(obs)

        return vl, vr, self._active_plan_id

    # ------------------------------------------------------------------
    # 区画中心（または旋回直後の同一区画）での処理
    # ------------------------------------------------------------------
    def _on_stationary(self, obs) -> None:
        sensing = sense_walls(obs, self.params)
        self._update_map_from_sensing(sensing)

        if self.phase is Phase.EXPLORE and self.cell in self._goal_cell_set:
            # ゴール到達。次は帰還（このときも地図を更新し続ける）。
            self.phase = Phase.RETURN
            self._replans_at_cell = 0  # 目標が変わったので仕切り直す
        elif self.phase is Phase.RETURN and self.cell == self.start_cell:
            # 帰還完了。悲観歩数マップ（未知=通れない）で、既知の壁だけで
            # ゴールへ確実に到達できるかを判定する。
            if is_reachable_known(self.maze, self.start_cell, self._goal_cell_list):
                self.phase = Phase.DONE
            else:
                # まだ地図が足りない。未知の区画を詰めに行く
                # （＝もう一度、楽観歩数マップでゴールを目指す。新しく分かった
                # 壁を踏まえて歩数マップは毎回引き直すので、前回と同じ経路を
                # なぞるだけにはならない）。
                self.phase = Phase.EXPLORE
            self._replans_at_cell = 0  # 目標が変わったので仕切り直す

        if self.phase is Phase.DONE:
            self._issue_stop()
            return

        target = self._pick_next_direction()
        if target is not None and target != self.heading and self._replans_at_cell >= MAX_REPLANS_PER_CELL:
            # 🔴 同一区画に留まったまま MAX_REPLANS_PER_CELL 回を超えて旋回のみを
            # 繰り返している。実際に観測した事例（2026-08-19、design_v4
            # maze_41049 の探索中）: 側方センサ(LS/RS、光軸75.96°)は
            # `classic/sensing.py` の docstring が明記するとおり、前方センサ
            # より閾値の分離が狭く、較正時の重なり帯 [0.077,0.090) の**外側
            # ぎりぎり**（実測 0.0927 等）でも位置ずれ次第で誤判定しうる。
            # その結果、ある壁が「前方として確認した向き」では WALL、
            # 「側方として確認した別の向き」では（誤って）CLEAR と書き換えら
            # れ、両者が食い違うたびに地図が上書きされて次善方位の判定が
            # 反転し続ける、という**同一区画内での振動**が実際に発生した
            # （E<->S を交互に選び続け、直進が一度も完了しない）。
            # これは AMBIGUOUS を握りつぶしているのではない
            # （`_update_map_from_sensing` は AMBIGUOUS を書かない設計のまま）。
            # 確信を持って書いた 2 つの判定そのものが矛盾しているケースであり、
            # 較正の前提（位置ずれ±15mm・方位ずれ±4°）を推測航法の誤差が
            # 超えたときに起こりうる、S1（壁補正なし）の既知の限界である。
            # 無限に旋回し続けて持ち時間を消費する代わりに、ここで**停止**
            # して評価器のスタック判定に委ねる（実機に忠実な振る舞い。
            # 上の「到達不能」ケースと同じ思想）。
            self._blocked_reason = (
                f"現在地 {self.cell} で同一区画内の再計画が {self._replans_at_cell} 回を超えました "
                f"(phase={self.phase}, heading={self.heading}, target={target})"
            )
            self.motion.start_stop()
            self._active_kind = MotionKind.STOP
            self._active_plan_id = f"{self._phase_prefix()}:blocked"
            return

        if target is None:
            # 楽観歩数マップ上でも現在地からゴール/スタートへ到達できない。
            #
            # 🔴 これは実際に起こりうる（S1 の既知の限界）。本モジュールは
            # note_030 §2 の指示どおり推測航法のみで位置・方位を持ち、壁センサ
            # による位置補正（S2）は行わない。しきい値（classic/sensing.py）は
            # 「区画中心付近・位置ずれ±15mm・方位ずれ±4°」を前提に較正されて
            # いるが、この前提は**壁補正なしで多くの区画を移動し続けた場合には
            # 保証されない**。並進・旋回のたびに実測数 mm〜1°未満の誤差が乗り
            # （`classic/motion.py` docstring 実測値）、区画数が積み重なると
            # 較正の前提を超えることがある。前提を超えた位置で読んだ壁は、
            # 曖昧判定(AMBIGUOUS)には落ちずに**確信を持って誤判定**されうる
            # （前方センサでさえも、である。前方の閾値の無重なりは較正時の
            # 位置ずれレンジ内でのみ保証されている）。結果として地図の一部が
            # 実際より狭く閉じ、楽観歩数マップでも経路が見えなくなることがある
            # （2026-08-19 実測: design_v4 迷路の探索中に発生を確認）。
            #
            # これは実機でも起こりうる失敗モード（推測航法だけで長距離走ると
            # 位置を見失う）であり、握りつぶして先へ進む代わりに**停止する**
            # のが実機に忠実な振る舞いである。停止し続ければ評価器のスタック
            # 判定（20秒間ほぼ無変位）が働き、係員回収されてスタートへ戻る
            # （地図は保持されるので、コース次第では別経路が見つかり探索が
            # 続行できる）。ここで例外を投げて評価そのものを止めない
            # （note_029 の教訓は「想定外を静かに握りつぶすな」であって
            # 「実機ならしないクラッシュを起こせ」ではない）。
            self._blocked_reason = (
                f"現在地 {self.cell} から目標へ楽観歩数マップ上でも到達できません "
                f"(phase={self.phase}, goals={self._current_goals()})"
            )
            self.motion.start_stop()
            self._active_kind = MotionKind.STOP
            self._active_plan_id = f"{self._phase_prefix()}:blocked"
            return

        if target == self.heading:
            self._issue_forward()
        else:
            self._replans_at_cell += 1
            self._issue_turn_towards(target)

    def _update_map_from_sensing(self, sensing: WallSensing) -> None:
        """前方・左方・右方の判定を絶対方位へ変換して地図へ書き込む。

        AMBIGUOUS はモジュール docstring の方針どおり書き込まない
        （未知のまま。前方センサは重なりゼロなので、進行方向として選ばれた
        側は旋回後の再読み取りで必ず確定する）。
        """
        cx, cy = self.cell
        front_dir = self.heading
        left_dir = Direction((int(self.heading) - 1) % 4)
        right_dir = Direction((int(self.heading) + 1) % 4)

        for direction, state in (
            (front_dir, sensing.front),
            (left_dir, sensing.left),
            (right_dir, sensing.right),
        ):
            if state is SenseWallState.WALL:
                self.maze.set_wall(cx, cy, direction, MapWallState.WALL)
            elif state is SenseWallState.CLEAR:
                self.maze.set_wall(cx, cy, direction, MapWallState.OPEN)
            # AMBIGUOUS: 未知のまま(書き込まない)。

    def _current_goals(self) -> List[Cell]:
        if self.phase is Phase.EXPLORE:
            return self._goal_cell_list
        return [self.start_cell]

    def _pick_next_direction(self) -> Optional[Direction]:
        """楽観歩数マップ（未知=通れる）で、現在地からゴール/スタート方向へ
        1 区画進むべき方位を返す。到達不能なら None。

        歩数マップは呼び出しのたびに現在の地図から引き直す（キャッシュしない）
        ので、新しく分かった壁が常に反映される。"""
        goals = self._current_goals()
        dist = compute_flood(self.maze, goals, FloodMode.OPTIMISTIC)
        cx, cy = self.cell
        cur = dist[cx, cy]
        if cur == UNREACHABLE:
            return None
        for direction in ALL_DIRECTIONS:
            nb = self.maze.neighbor(cx, cy, direction)
            if nb is None:
                continue
            if not is_passable(self.maze, cx, cy, direction, FloodMode.OPTIMISTIC):
                continue
            nx, ny = nb
            if dist[nx, ny] == cur - 1:
                return direction
        return None

    # ------------------------------------------------------------------
    # コマンド発行
    # ------------------------------------------------------------------
    def _phase_prefix(self) -> str:
        return "explore" if self.phase is Phase.EXPLORE else "return"

    def _issue_stop(self) -> None:
        self.motion.start_stop()
        self._active_kind = MotionKind.STOP
        self._active_plan_id = "idle"

    def _issue_forward(self) -> None:
        self.motion.start_forward(1)
        self._active_kind = MotionKind.FORWARD
        self._active_plan_id = f"{self._phase_prefix()}:straight"

    def _issue_turn_towards(self, target: Direction) -> None:
        # Direction は N=0,E=1,S=2,W=3 の時計回り順（classic/maze_map.py）。
        # classic/route.py の _turn_type と同じ規約: +1=右90, +2=180, +3(-1)=左90。
        rel = (int(target) - int(self.heading)) % 4
        if rel == 1:
            self.motion.start_turn_right()
            self._active_kind = MotionKind.TURN_RIGHT_90
            label = "turn_right"
        elif rel == 2:
            self.motion.start_turn_180()
            self._active_kind = MotionKind.TURN_180
            label = "turn_180"
        elif rel == 3:
            self.motion.start_turn_left()
            self._active_kind = MotionKind.TURN_LEFT_90
            label = "turn_left"
        else:
            raise AssertionError("同一方向への旋回が要求されました（直進として扱われるべきです）")
        self._pending_heading = target
        self._active_plan_id = f"{self._phase_prefix()}:{label}"

    def _advance_state(self) -> None:
        """完了したコマンドの種別に応じて、内部の位置・方位を進める。

        位置・方位のどちらも「コマンドが done を返した」という、車輪角速度・
        ジャイロの積算に基づく `CellMotionController` 自身の完了判定だけを
        根拠にしている（真値位置は使わない）。"""
        if self._active_kind is MotionKind.FORWARD:
            nb = self.maze.neighbor(self.cell[0], self.cell[1], self.heading)
            if nb is None:
                raise RuntimeError("直進コマンドが迷路外への移動を指示しました（内部矛盾）")
            self.cell = nb
            self._replans_at_cell = 0  # 区画を進めたので同一区画カウンタを戻す
        elif self._active_kind in (
            MotionKind.TURN_LEFT_90, MotionKind.TURN_RIGHT_90, MotionKind.TURN_180,
        ):
            assert self._pending_heading is not None
            self.heading = self._pending_heading
        # MotionKind.STOP: 何もしない（呼び出し元で active_kind is STOP のときは
        # そもそもこのメソッドを呼ばない設計だが、防御的に no-op にしてある）。
