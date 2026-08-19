"""
classic/explorer.py
================
探索走行〜最短走行の状態機械。実機のマイクロマウスが実際にやっていることを写す
（`research_notes/note_030_classical_rebuild_plan.md` §1・§2 L4・§3 S3）。

    スタート区画から出発 → 区画の中心に来るたびに壁センサで前・左・右を読んで
    地図へ書く → 楽観の歩数マップ（未知は通れる）でゴール方向の次の1区画を選ぶ
    （足立法）→ ゴール到達 → スタートへ帰還（このときも地図を更新する）→
    帰還後、悲観の歩数マップ（未知は通れない）で「既知の壁だけでゴールへ
    到達できるか」を判定する。到達できれば **FAST（最短走行）** へ移り、
    `classic/route.py` の `plan_route` が返す最短経路のコマンド列を実行する。
    ゴール停止で 0.2 秒静止したのち、スタートへ戻って再び最短走行する
    （note_030 §3 S3・L4「最短走行 ×N」）。到達できなければもう一度ゴールを
    目指す（＝未知の区画を詰めに行く）。

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
  - 最短走行（FAST）中は `sense_walls()` を呼ぶことはあっても、**地図は
    一切書き換えない**（下記「S3: 最短走行」節参照）。

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

【S2: 壁センサによる区画ごとの位置補正（note_030 §3 S2）】
`classic/localization.py` の `Localizer` を持ち、区画の中心（`_on_stationary`）
で計算済みの `WallSensing` を使って横位置補正（次の直進コマンドへの目標方位
バイアス）を適用する。無効化は `ClassicExplorer(..., localization_enabled=False)`
で行う（無効化時は `Localizer.enabled=False` となり、`classic/motion.py` の
コード経路は S2 導入前と完全に同一になる。詳細は `classic/localization.py`）。

【S3: 最短走行（note_030 §3 S3、任務指示）】
探索走行は 1 区画ずつ止まって進む（超信地旋回走行・区画ごと停止）。最短走行
（Phase.FAST）では地図から `classic/route.py` の `plan_route` で求めた
ターン種別の列（`Command` の列）を先頭から順に実行する。**直線は伸ばす**
（STRAIGHT n 区画を `motion.start_forward(n)` の 1 コマンドで連続実行し、
停止・整定の空費を n-1 回分消す）。速度上限（`DEFAULT_V_CRUISE` 等）は
変えない（速度の引き上げは S4 の担当）。

  - `extend_straights` コンストラクタ引数（既定 True）で対照が作れる。
    False のときは STRAIGHT n を「1 区画の直進を n 回」として実行する
    （＝経路は同じで、直線を伸ばす効果だけを外した対照。停止・整定の
    空費が毎回入るぶん遅くなるはず）。
  - 多区画直進中の「区画ごとの位置補正」（🔴 S2 の前提を壊さないための
    要点）: `classic/localization.py` の横位置補正は「1 区画ごとに読み直して
    次の 1 区画へバイアスをかける」設計であり、n 区画を 1 コマンドにすると
    出発点で 1 回測った横ずれが n 区画ぶん一定のバイアスとして効き続けて
    しまう。これを避けるため、`CellMotionController.cells_completed` が
    k→k+1 と増えた瞬間（＝k+1 番目の区画中心を通過した瞬間）ごとに
    壁センサを読み直し、`Localizer.lateral_bias_for_forward` で横ずれを
    推定し、`CellMotionController.reanchor_heading` で目標方位を掛け直す
    （n 区画で、コマンド発行時 1 回＋途中 n-1 回＝合計 n 回、補正の入力に
    センサを使う）。
  - 🔴 **最短走行中は地図を書き換えない**（教授裁定）。地図は探索で確定済み
    であり、走行中の読みで上書きすると確定した地図を壊しうる。センサの
    読みは**位置補正にだけ**使う（`_update_map_from_sensing` を呼ばない）。
    探索走行・帰還走行（Phase.EXPLORE/RETURN/RETURN2）での地図更新は
    従来どおり続ける。
  - plan_route は `start_heading=Direction.N` 固定で呼ぶ（教授裁定）。
    実際の帰還後の向きはスタート区画の唯一の出入口の向きで決まり、北とは
    限らない（例: 北の隣接区画から入れば南向きで区画中心に着く）ので、
    実行の先頭で必要なら「現在の向き→北」への旋回を 1 回差し込んでから
    （`_issue_turn_towards` を再利用。既存の "fast:turn_*" と同じ label が
    付く）、生成されたコマンド列をそのまま実行する。
  - ゴール停止（`CommandType.GOAL_STOP`）に達したら、その場で 0.2 秒
    （`params.control_dt` から求めたティック数。既定 100Hz で 20 ティック）
    静止してからスタートへ戻る段階（Phase.RETURN2）へ移る。🔴 評価器は
    ゴール前端通過のあと「機体全体がゴール区画に完全に入り、かつ前進速度
    < 0.02 m/s」を最大 5 秒間確認して初めて走行を成立させる
    （`competition/evaluator.py` の `GOAL_STOP_TIMEOUT_S` / `body_fully_inside`）。
    すぐ動き出すと `goal_not_contained` へ書き換えられ走行が失われる。
  - スタートへ戻る段階（Phase.RETURN2）は Phase.RETURN と**同じ機構**
    （楽観の歩数マップでスタートを目指す。地図更新も従来どおり続ける）を
    使う。plan_id の接頭辞だけが "return2:" になる（探索中の "return:" と
    区別できるようにする）。スタート区画に着いたら**再び FAST を実行する**
    （note_030 §2 L4「最短走行 ×N」）。悲観歩数マップでの再判定はしない
    （RETURN 完了時に既に「既知の壁だけで到達可能」と確認済みであるため）。
    評価器が持ち時間・最大走行回数で自然に打ち切る（競技規約）。
  - 係員回収（`handle_retrieval`）を受けたときは、地図と段階を保ったまま
    位置推定だけ初期化する（現行と同じ）。段階が FAST/RETURN2 だった場合は
    物理的にスタート区画へ戻されているので、スタート区画から最短走行を
    やり直す。
  - 最短走行の経路が引けない・実行できないなど想定外の事態が起きた場合は
    例外を投げず、その場で停止して "fast:blocked" を返す（現行の
    "explore:blocked"/"return:blocked" と同じ思想。note_029 の教訓）。

【S3 の真値の禁止（変えてはならない前提）】
`classic/policy.py` の `requires_privileged = False` を維持する。
`mouse.sim.MouseSim.privileged_pose()`/`privileged_velocity()` を参照しない。
`maze_info["xml_path"]` を読まない（このパスの MJCF には壁の真値が含まれる。
評価器はこの経路を塞いでいないので、規約として自分で守る）。
"""
from enum import Enum, auto
from typing import List, Optional, Tuple

from classic.flood import FloodMode, UNREACHABLE, compute_flood, is_passable, is_reachable_known
from classic.localization import Localizer
from classic.maze_map import ALL_DIRECTIONS, Direction, MazeMap
from classic.maze_map import WallState as MapWallState
from classic.motion import CellMotionController, MotionKind
from classic.route import Command, CommandType, NoRouteError, plan_route
from classic.sensing import WallSensing
from classic.sensing import WallState as SenseWallState
from classic.sensing import sense_walls
from mouse.params import RobotParams

Cell = Tuple[int, int]


class Phase(Enum):
    """探索走行〜最短走行の大局的な段階。"""

    EXPLORE = auto()   # ゴールを目指す（楽観歩数マップ）
    RETURN = auto()    # 探索完了前の帰還。スタートへ戻る（楽観歩数マップ）。地図も更新する
    FAST = auto()       # 最短走行を実行中（note_030 §3 S3）
    RETURN2 = auto()    # 最短走行後、次の最短走行のためスタートへ戻る（RETURN と同じ機構）


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

# ゴール停止の静止ホールド時間（note_030 §3 S3 任務指示 🔴）。評価器の
# GOAL_STOP_TIMEOUT_S / body_fully_inside 対策（モジュール docstring参照）。
GOAL_STOP_HOLD_S = 0.2

# CommandType のターン種別 → (target - heading) mod 4。
# classic/route.py の _turn_type と同じ規約（Direction は N=0,E=1,S=2,W=3 の
# 時計回り順）: +1=右90, +2=180, +3(=-1)=左90。
_FAST_TURN_REL = {
    CommandType.TURN_RIGHT90: 1,
    CommandType.TURN_180: 2,
    CommandType.TURN_LEFT90: 3,
}


class ClassicExplorer:
    """探索走行〜最短走行の状態機械本体。1 ティックごとに `tick(obs)` を呼ぶ。

    呼び出し側（`classic/policy.py` の `ClassicExplorerPolicy`）は
    `on_maze_start` 相当のタイミングで本クラスを構築し、以後は 100Hz で
    `tick(obs)` を呼んで得た (v_left, v_right) をそのまま `act()` の返り値
    にする。係員回収時は `handle_retrieval()` を呼ぶこと。
    """

    def __init__(self, width: int, height: int, params: Optional[RobotParams] = None,
                 localization_enabled: bool = True, extend_straights: bool = True) -> None:
        self.params = params if params is not None else RobotParams()
        self.maze = MazeMap(width, height)

        # S2: 壁センサによる区画ごとの位置補正（note_030 §3 S2、任務指示）。
        # localization_enabled=False で完全に無効化できる（否定対照・
        # 「補正あり/なし」比較用。詳細は classic/localization.py docstring）。
        self.localizer = Localizer(self.params, enabled=localization_enabled)

        # S3: 最短走行で直線を伸ばすかどうか（note_030 §3 S3 任務指示 4）。
        # False は対照（STRAIGHT n を 1 区画の直進 n 回として実行する）。
        self.extend_straights = extend_straights

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

        self.motion = CellMotionController(self.params, localizer=self.localizer)

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

        # --- S3: 最短走行 (Phase.FAST) の実行状態 ---
        # plan_route が返した Command の列と、次に実行するインデックス。
        self._fast_commands: List[Command] = []
        self._fast_cmd_index: int = 0
        # extend_straights=False のとき、実行中の STRAIGHT コマンドを
        # 1 区画ずつに分割した「残り区画数」（0 のときは分割中でない）。
        self._fast_straight_cells_left: int = 0
        # 現在実行中の FORWARD コマンドで、区画ごとの掛け直し（reanchor）を
        # 何区画ぶんまで処理したか（motion.cells_completed の変化検出用。
        # 新しい FORWARD を発行するたびに 0 へ戻す）。
        self._fast_cells_reanchored: int = 0
        # 現在実行中の FORWARD コマンドの総区画数（`motion.start_forward(n)`
        # の n）。tick() の区画ごと掛け直しループが「最後の1区画ぶんの
        # 前進」を誤って自分で消費してしまわないよう、最大 n-1 回までしか
        # 進めないための上限として使う（🔴 実際に起きた不具合の再発防止。
        # `tick()` のコメント参照: 実機同様の慣性を持つ物理シミュレータでは
        # 目標距離をわずかに行き過ぎてから速度がゼロに収束するまでの間、
        # `done=True` になるより先に `cells_completed` が区画数ぴったり
        # (n) を指すティックが実在しうる。上限が無いと、そのティックを
        # 「n-1個目の境界通過」と誤認して n 回目の前進をここで消費してしまい、
        # 直後に `done=True` になった際の完了処理でさらに 1 回進めて
        # 合計 n+1 回進む＝迷路外へ出る内部矛盾を起こしていた
        # 2026-08-19 design_v4 maze_42134 の実走で実際に発生を確認した）。
        self._fast_straight_total_cells: int = 0
        # ゴール停止の静止ホールドの残りティック数。
        self._goal_stop_hold_ticks: int = max(1, int(round(GOAL_STOP_HOLD_S / self.params.control_dt)))
        self._goal_stop_ticks_left: int = 0

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
        # 再開する（探索が既に完了していれば FAST/RETURN2 のまま）。
        #
        # ただし FAST/RETURN2 中の回収は「物理的にスタート区画へ戻された」
        # ことと同義なので、最短走行をスタート区画からやり直す
        # （note_030 §3 任務指示: 「段階が FAST ならスタート区画から最短走行を
        # やり直せること」）。Phase.RETURN2 にしておけば、次の tick() が
        # 「スタート区画に着いた」処理（= `_begin_fast_run`）へそのまま
        # 合流する（下記 `_on_stationary` の共通処理と同じ経路）。
        if self.phase in (Phase.FAST, Phase.RETURN2):
            self.phase = Phase.RETURN2
            self._fast_straight_cells_left = 0
            self._fast_cells_reanchored = 0
            self._goal_stop_ticks_left = 0

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

        # S3 (b): FAST の多区画直進の途中、区画中心を通過するたびに測り直して
        # 掛け直す（note_030 §3 S3 任務指示。モジュール docstring 参照）。
        # 地図は書き換えない（センサは位置補正にのみ使う）。
        if (self.phase is Phase.FAST and self.extend_straights
                and self._active_kind is MotionKind.FORWARD and not done):
            completed = self.motion.cells_completed
            # 🔴 最後の1区画ぶんは _advance_state() の完了処理に必ず残す
            # （`_fast_straight_total_cells` docstring 参照。物理的な行き
            # 過ぎで cells_completed が done より先に n に達することがある）。
            max_mid_run_advances = max(self._fast_straight_total_cells - 1, 0)
            while (completed > self._fast_cells_reanchored
                   and self._fast_cells_reanchored < max_mid_run_advances):
                self._fast_cells_reanchored += 1
                sensing = sense_walls(obs, self.params)
                bias = self.localizer.lateral_bias_for_forward(
                    sensing, cell=self.cell, heading=int(self.heading))
                if bias != 0.0:
                    self.motion.reanchor_heading(bias)
                nb = self.maze.neighbor(self.cell[0], self.cell[1], self.heading)
                if nb is None:
                    # 🔴 想定外の事態（`_enter_fast_blocked` docstring参照。
                    # 地図の誤りが原因で、直進の途中に実際にはあるはずの壁へ
                    # 衝突しつつ車輪だけ空転し、推測航法の距離推定が迷路の
                    # 外まで進んでしまうことがある）。例外を投げず停止する。
                    self._enter_fast_blocked(
                        f"FAST の直進中、区画 {self.cell} から向き {self.heading} への"
                        f"移動が迷路外を指しました（地図の誤り、または未知の壁への"
                        f"衝突による車輪角速度の暴走が疑われる）"
                    )
                    vl, vr = 0.0, 0.0
                    break
                self.cell = nb

        if self._active_plan_id == "fast:goal_stop":
            # ゴール停止後の静止ホールド（note_030 §3 S3 任務指示 🔴。
            # モジュール docstring の GOAL_STOP_TIMEOUT_S 対策を参照）。
            if self._goal_stop_ticks_left > 0:
                self._goal_stop_ticks_left -= 1
            else:
                self.phase = Phase.RETURN2
                self._replans_at_cell = 0
                self._on_stationary(obs)
                vl, vr, done = self.motion.update(obs)
        elif done and self._active_kind is not MotionKind.STOP:
            # 直進 1 区画／旋回のどちらかが完了した瞬間。位置・方位を確定させ、
            # 同じ区画中心で（あるいは旋回直後の同じ区画で）直ちに次の判断へ
            # 進む。判断そのものは計算のみでシミュレーション時間を消費しない
            # （実機の「止まって読んで決めてまた走る」を、判断部分は瞬時と
            # 単純化して再現している）。
            self._advance_state()
            if self._active_plan_id == "fast:blocked":
                # _advance_state() が FAST の異常事態を検出し、既に停止
                # コマンドへ切り替え済み（下記参照）。_on_stationary を
                # 呼ぶと即座に次のコマンドを発行してしまい停止が消えるので
                # 呼ばない。
                vl, vr = 0.0, 0.0
            else:
                self._on_stationary(obs)
                vl, vr, _done2 = self.motion.update(obs)

        return vl, vr, self._active_plan_id

    # ------------------------------------------------------------------
    # 区画中心（または旋回直後の同一区画、コマンド完了直後）での処理
    # ------------------------------------------------------------------
    def _on_stationary(self, obs) -> None:
        if self.phase is Phase.FAST:
            self._on_stationary_fast(obs)
            return

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
                self._begin_fast_run(obs)
                return
            # まだ地図が足りない。未知の区画を詰めに行く
            # （＝もう一度、楽観歩数マップでゴールを目指す。新しく分かった
            # 壁を踏まえて歩数マップは毎回引き直すので、前回と同じ経路を
            # なぞるだけにはならない）。
            self.phase = Phase.EXPLORE
            self._replans_at_cell = 0  # 目標が変わったので仕切り直す
        elif self.phase is Phase.RETURN2 and self.cell == self.start_cell:
            # 最短走行後の帰還完了。地図は探索完了時点で「既知の壁だけで
            # 到達可能」と確認済みなので、悲観歩数マップでの再判定はせず、
            # そのまま次の最短走行へ入る（note_030 §2 L4「最短走行 ×N」）。
            self._begin_fast_run(obs)
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
            self._issue_forward(sensing)
        else:
            self._replans_at_cell += 1
            self._issue_turn_towards(target)

    def _update_map_from_sensing(self, sensing: WallSensing) -> None:
        """前方・左方・右方の判定を絶対方位へ変換して地図へ書き込む。

        AMBIGUOUS はモジュール docstring の方針どおり書き込まない
        （未知のまま。前方センサは重なりゼロなので、進行方向として選ばれた
        側は旋回後の再読み取りで必ず確定する）。

        🔴 Phase.FAST 中は呼ばれない（`_on_stationary_fast` を参照。最短走行
        中は地図を書き換えない）。"""
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
        return [self.start_cell]  # RETURN・RETURN2 はどちらもスタートを目指す

    def _pick_next_direction(self) -> Optional[Direction]:
        """楽観歩数マップ（未知=通れる）で、現在地からゴール/スタート方向へ
        1 区画進むべき方位を返す。到達不能なら None。

        歩数マップは呼び出しのたびに現在の地図から引き直す（キャッシュしない）
        ので、新しく分かった壁が常に反映される。Phase.EXPLORE/RETURN/RETURN2
        でのみ呼ばれる（Phase.FAST は plan_route の Command 列を使うので
        呼ばない）。"""
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
    # コマンド発行（Phase.EXPLORE/RETURN/RETURN2 共通）
    # ------------------------------------------------------------------
    def _phase_prefix(self) -> str:
        if self.phase is Phase.EXPLORE:
            return "explore"
        if self.phase is Phase.RETURN:
            return "return"
        if self.phase is Phase.FAST:
            return "fast"
        return "return2"  # Phase.RETURN2

    def _issue_forward(self, sensing: WallSensing) -> None:
        self.motion.start_forward(1)
        # S2 (a) 横位置補正: この区画で読んだ側方センサから横ずれを推定し、
        # 分かれば次の 1 区画の目標方位へバイアスとして足し込む
        # （start_forward の直後に呼ぶ必要がある。classic/motion.py の
        # bias_target_heading docstring 参照）。Localizer.enabled=False の
        # ときは常に 0.0 が返るので start_forward 直後の状態から変わらない。
        bias = self.localizer.lateral_bias_for_forward(sensing, cell=self.cell, heading=int(self.heading))
        if bias != 0.0:
            self.motion.bias_target_heading(bias)
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
        根拠にしている（真値位置は使わない）。

        Phase.FAST の多区画直進では、この呼び出しの前に `tick()` が
        `cells_completed` の変化を見て途中の n-1 区画ぶんを既に
        `self.cell` へ進めてある。ここでの 1 回の進行で、n 区画ぶんの最後の
        1 区画が進み、合計 n 回進んだことになる（note_030 §3 S3 任務指示）。
        """
        if self._active_kind is MotionKind.FORWARD:
            nb = self.maze.neighbor(self.cell[0], self.cell[1], self.heading)
            if nb is None:
                if self.phase is Phase.FAST:
                    # 🔴 想定外の事態（`_enter_fast_blocked` docstring・
                    # `tick()` のコメント参照）。例外を投げず停止する。
                    # EXPLORE/RETURN/RETURN2 では `_pick_next_direction` が
                    # `neighbor() is None` の方位を最初から候補から除外して
                    # いるためこの分岐に到達しない（現行どおり例外のまま）。
                    self._enter_fast_blocked(
                        f"FAST の直進コマンド完了時、区画 {self.cell} から向き "
                        f"{self.heading} への移動が迷路外を指しました（地図の誤りが疑われる）"
                    )
                    return
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

    # ------------------------------------------------------------------
    # S3: 最短走行 (Phase.FAST) の実行
    # ------------------------------------------------------------------
    def _enter_fast_blocked(self, reason: str) -> None:
        """FAST 実行中に想定外の事態（経路が引けない・実行できない）が
        起きたとき、例外を投げずにその場で停止する（note_030 §3 S3 任務
        指示 5・現行の "explore:blocked"/"return:blocked" と同じ思想。
        note_029 の教訓）。

        🔴 実際に起きた事例（2026-08-19、design_v4 の実走で発見）: FAST の
        多区画直進の途中で、地図の誤り（S1 の既知の限界。
        `tests/test_classic_policy.py` docstring参照）により実際には壁が
        あるのに「開いている」と誤認識した経路を計画してしまうと、その壁へ
        衝突したまま車輪だけが空転を続け、車輪角速度センサ由来の推測航法
        （距離推定）だけは進み続けてしまうことがある。この結果
        `cells_completed`（`classic/motion.py`）が実際の到達区画を超えて
        増え続け、`self.cell` を迷路の範囲外まで進めようとする内部矛盾が
        生じうる。これを検出したら、停止して評価器のスタック判定・係員
        回収に委ねる（探索走行の "blocked" 安全弁と同じ設計思想）。"""
        self._blocked_reason = reason
        self.motion.start_stop()
        self._active_kind = MotionKind.STOP
        self._active_plan_id = "fast:blocked"

    def _begin_fast_run(self, obs) -> None:
        """悲観歩数マップで「既知の壁だけでゴールへ到達できる」と判定された
        瞬間（RETURN 完了時）、または最短走行後にスタートへ戻り着いた瞬間
        （RETURN2 完了時）に呼ぶ。`plan_route` で最短経路のコマンド列を求め、
        Phase.FAST へ入って先頭のコマンドを発行する（note_030 §3 S3 ①）。"""
        try:
            _path, commands = plan_route(
                self.maze, start=self.start_cell, goals=self._goal_cell_list,
                mode=FloodMode.PESSIMISTIC, start_heading=Direction.N,
            )
        except NoRouteError as exc:
            # 🔴 通常は起こらないはず（悲観歩数マップで到達可能と確認した
            # 直後、または既に確認済みの地図で呼んでいる）。それでも起きたら
            # 例外で評価全体を落とさず、想定外を静かに握りつぶさない方針
            # （モジュール docstring・note_030 §5）どおり停止する。
            self.phase = Phase.FAST
            self._enter_fast_blocked(f"最短経路の計画に失敗しました: {exc}")
            return

        self.phase = Phase.FAST
        self._fast_commands = commands
        self._fast_cmd_index = 0
        self._fast_straight_cells_left = 0
        self._fast_cells_reanchored = 0

        if self.heading != Direction.N:
            # plan_route は start_heading=Direction.N 固定で呼ぶ設計
            # （教授裁定、note_030 §3 任務指示）。実際の帰還後の向きは
            # スタート区画の唯一の出入口の向きで決まり、北とは限らない。
            # 既存の _issue_turn_towards をそのまま使って北へ向き直してから
            # （このコマンド自体は plan_route の出力には含まれない、実行系列
            # だけへの補正である）、コマンド列を先頭から実行する。
            self._issue_turn_towards(Direction.N)
            return
        self._issue_next_fast_command(obs)

    def _on_stationary_fast(self, obs) -> None:
        """Phase.FAST 実行中、コマンドが完了した瞬間（または開始直後）の処理。

        🔴 地図は書き換えない（`_update_map_from_sensing` を呼ばない。
        note_030 §3 S3 任務指示。センサは位置補正にのみ使う）。"""
        if self._fast_straight_cells_left > 0:
            self._continue_fast_straight_leg(obs)
            return
        self._issue_next_fast_command(obs)

    def _issue_next_fast_command(self, obs) -> None:
        """`self._fast_commands[self._fast_cmd_index]` を発行し、
        インデックスを進める。"""
        if self._fast_cmd_index >= len(self._fast_commands):
            # 🔴 想定外（GOAL_STOP は必ず最後に来るはずで、通常ここには
            # 来ない）。例外を投げず、その場で停止する
            # （note_030 §3 S3 任務指示 5・モジュール docstring）。
            self._enter_fast_blocked("FAST のコマンド列が尽きた後にも実行要求が来ました（内部矛盾）")
            return

        cmd = self._fast_commands[self._fast_cmd_index]
        self._fast_cmd_index += 1

        if cmd.type == CommandType.GOAL_STOP:
            self.motion.start_stop()
            self._active_kind = MotionKind.STOP
            self._active_plan_id = "fast:goal_stop"
            self._goal_stop_ticks_left = self._goal_stop_hold_ticks
            return

        if cmd.type == CommandType.STRAIGHT:
            sensing = sense_walls(obs, self.params)
            self._issue_fast_straight(cmd.cells, sensing)
            return

        rel = _FAST_TURN_REL.get(cmd.type)
        if rel is None:
            raise AssertionError(f"未対応の CommandType: {cmd.type}")
        target = Direction((int(self.heading) + rel) % 4)
        self._issue_turn_towards(target)

    def _issue_fast_straight(self, n_cells: int, sensing: WallSensing) -> None:
        """FAST の STRAIGHT n を（新しいコマンドとして）発行する
        （note_030 §3 S3 ②③）。

        `extend_straights=True`（既定）: n 区画を `motion.start_forward(n)`
        の 1 コマンドで連続実行する（直線を伸ばす。停止・整定の空費が
        n-1 回分消える）。

        `extend_straights=False`（対照）: 1 区画ぶんの `start_forward(1)`
        だけを発行し、残り n-1 区画は `_continue_fast_straight_leg` が
        コマンドの完了のたびに 1 区画ずつ発行する（経路は同じで、直線を
        伸ばす効果だけを外す。note_030 §3 S3 任務指示 4）。"""
        self._fast_cells_reanchored = 0
        if self.extend_straights:
            self._fast_straight_cells_left = 0
            self._fast_straight_total_cells = n_cells
            self.motion.start_forward(n_cells)
        else:
            self._fast_straight_cells_left = n_cells - 1
            self._fast_straight_total_cells = 1
            self.motion.start_forward(1)
        self._apply_fast_straight_bias(sensing)
        self._active_kind = MotionKind.FORWARD
        self._active_plan_id = "fast:straight"

    def _continue_fast_straight_leg(self, obs) -> None:
        """`extend_straights=False` のときの、同一 STRAIGHT コマンド内の
        続きの 1 区画を発行する（対照。note_030 §3 S3 任務指示 4）。
        コマンド列のインデックスは進めない（同じ STRAIGHT の続きなので）。"""
        self._fast_straight_cells_left -= 1
        self._fast_cells_reanchored = 0
        self._fast_straight_total_cells = 1
        sensing = sense_walls(obs, self.params)
        self.motion.start_forward(1)
        self._apply_fast_straight_bias(sensing)
        self._active_kind = MotionKind.FORWARD
        self._active_plan_id = "fast:straight"

    def _apply_fast_straight_bias(self, sensing: WallSensing) -> None:
        """FAST の直進コマンド発行直後の横位置補正（S2 (a) と同じ入り口）。
        n 区画コマンドの「出発点で 1 回」ぶんに相当する（途中の n-1 回は
        `tick()` の区画ごと掛け直しが受け持つ。note_030 §3 S3 任務指示）。"""
        bias = self.localizer.lateral_bias_for_forward(sensing, cell=self.cell, heading=int(self.heading))
        if bias != 0.0:
            self.motion.bias_target_heading(bias)
