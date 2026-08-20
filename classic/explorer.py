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
    ゴール停止で 2.0 秒静止したのち、**RETURN2（最短走行の帰路）** で
    スタートへ戻る。RETURN2 も FAST と同じ「plan_route のコマンド列を
    実行する」機構を使う（地図は探索完了時点で既に確定しているので、
    Phase.RETURN のように 1 区画ずつ足立法で確かめながら戻る理由が無い
    ——是正6・下記「S3: 最短走行」節参照）。スタートに着いたら再び最短走行
    する（note_030 §3 S3・L4「最短走行 ×N」）。到達できなければもう一度
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
  - ゴール停止（`CommandType.GOAL_STOP`）に達したら、その場で 2.0 秒
    （`params.control_dt` から求めたティック数。既定 100Hz で 200 ティック。
    `GOAL_STOP_HOLD_S` の docstring 参照 — 是正1）静止してから、スタートへ
    戻る段階（Phase.RETURN2）へ移る。🔴 評価器は
    ゴール前端通過のあと「機体全体がゴール区画に完全に入り、かつ前進速度
    < 0.02 m/s」を最大 5 秒間確認して初めて走行を成立させる
    （`competition/evaluator.py` の `GOAL_STOP_TIMEOUT_S` / `body_fully_inside`）。
    すぐ動き出すと `goal_not_contained` へ書き換えられ走行が失われる。
  - 🔴 是正6（2026-08-19 検収）: スタートへ戻る段階（Phase.RETURN2）は、
    もはや Phase.RETURN と同じ機構（足立法で 1 区画ずつ止まりながら進む）
    ではない。**Phase.FAST と同じ「plan_route のコマンド列を実行する」
    機構**を使う。地図は探索完了時点で既に確定済みであり（悲観の歩数マップ
    で到達可能と確認済み）、帰路だけ 1 区画ずつ探る理由が無いため
    （実測: maze_41001・最短歩数52 で、持ち時間420秒のうち旧方式の帰路だけ
    で80秒以上を消費し、3本目の走行が始まらなかった）。目標をスタート区画
    `(0,0)`、`start_heading` を**現在の向き** `self.heading` にして
    `plan_route` を呼ぶ（FAST と異なり北固定ではない — スタート区画へ入る
    向きは経路そのもので決まるので、実行前の向き合わせ旋回は不要）。直線を
    伸ばす（`extend_straights` の設定に従う）・多区画直進中の区画ごとの
    位置補正の掛け直しも FAST と全く同じ仕組みを共用する（`_on_stationary_route`
    以下、内部の実行エンジンは FAST/RETURN2 で共通化してある。plan_id の
    接頭辞だけが `_phase_prefix()` により "return2:" になる）。地図は
    書き換えない（FAST と同じ理由。地図は確定済み）。
    ただし終端の扱いだけ FAST と異なる: `plan_route` が返すコマンド列の
    最後は（目的地がどこであれ）必ず `CommandType.GOAL_STOP` だが、
    RETURN2 ではこれを「ゴール停止のホールド」（"return2:goal_stop" という
    plan_id）としては扱わない。スタート区画に着いた時点で静止する意味が
    無い（評価器の判定はゴール区画にしか関係しない）ため、この GOAL_STOP
    を受け取ったら**ホールドせずそのまま** `_begin_fast_run` を呼んで次の
    最短走行へ移る。悲観歩数マップでの再判定はしない（RETURN 完了時に既に
    「既知の壁だけで到達可能」と確認済みであるため）。評価器が持ち時間・
    最大走行回数で自然に打ち切る（競技規約）。
  - 係員回収（`handle_retrieval`）を受けたときは、地図と段階を保ったまま
    位置推定だけ初期化する（現行と同じ）。段階が FAST/RETURN2 だった場合は
    物理的にスタート区画へ戻されているので、スタート区画から最短走行を
    やり直す。
  - 最短走行（FAST）の経路が引けない・実行できないなど想定外の事態が
    起きた場合は例外を投げず、その場で停止して "fast:blocked" を返す
    （現行の "explore:blocked"/"return:blocked" と同じ思想。note_029 の
    教訓）。RETURN2 で同じ事態が起きた場合は "return2:blocked" を返す
    （是正6。plan_id の接頭辞は常に `_phase_prefix()` から決まる）。

【S3 の真値の禁止（変えてはならない前提）】
`classic/policy.py` の `requires_privileged = False` を維持する。
`mouse.sim.MouseSim.privileged_pose()`/`privileged_velocity()` を参照しない。
`maze_info["xml_path"]` を読まない（このパスの MJCF には壁の真値が含まれる。
評価器はこの経路を塞いでいないので、規約として自分で守る）。

【S3': 最短走行のプロファイル追従化（`fast_mode`。note_031・任務指示）】
`research_notes/note_031_profile_planner_and_eta.md` の判定条文（`η = T_ideal /
T_measured`）を受け、Phase.FAST・Phase.RETURN2 の実行方式を選べるようにする:

  - `fast_mode="command"`（既定）: 上の【S3】節どおりのコマンド方式。
    🔴 **このときのコード経路は本節導入前と完全に同一である**（`_profile_active`
    が常に False になるだけで、他の分岐・状態遷移は一切変わらない）。
  - `fast_mode="profile"`: `classic/fast_planner.py` の `plan_fast_run()` が
    マウス自身の学習した地図（未知は壁として悲観に扱う）から速度プロファイル
    計画（`FastPlan`）を作り、`classic/tracker.py` の `ProfileTracker` で追従する。
    区間の切れ目（経路追従⇄その場旋回）で速度・角速度の収束を待たない
    （`ProfileTracker` の設計どおり）。`plan_fast_run()` が `None` を返したら
    （既知の壁だけでは到達できない・安全弁）、その走行に限り現行のコマンド方式へ
    落ちる。落ちたことは `plan_id` の接頭辞（`"fast_fallback"`/`"return2_fallback"`）
    に残る（`_phase_prefix()` 参照）。
  - **本段では S2（壁センサによる位置補正）を profile 経路に配線しない**
    （推測航法のみで走らせ、どこまで持つかを測ってから次段で補正を足す。
    任務指示の範囲）。`ProfileTracker.apply_lateral_correction()` は
    `kp_lat=0`（既定）のまま呼ばないので、経路追従は純粋な推測航法である。
  - `plan_id` は経路追従中 `"fast:profile"`/`"return2:profile"`、その場旋回中
    `"fast:profile_spin"`/`"return2:profile_spin"` になる（一次記録からどちらの
    区間を走っていたかが分かるようにする。任務指示）。
"""
import math
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

from classic.fast_planner import FastPlan, PathBlock, SpinSegment, plan_fast_run
from classic.flood import FloodMode, UNREACHABLE, compute_flood, is_passable, is_reachable_known
from classic.localization import Localizer
from classic.maze_map import ALL_DIRECTIONS, Direction, MazeMap, direction_between
from classic.maze_map import WallState as MapWallState
from classic.motion import CellMotionController, MotionKind
from classic.route import Command, CommandType, NoRouteError, plan_route
from classic.sensing import WallSensing
from classic.sensing import WallState as SenseWallState
from classic.sensing import sense_walls
from classic.tracker import ProfileTracker
from mouse.params import RobotParams

Cell = Tuple[int, int]


class Phase(Enum):
    """探索走行〜最短走行の大局的な段階。"""

    EXPLORE = auto()   # ゴールを目指す（楽観歩数マップ）
    RETURN = auto()    # 探索完了前の帰還。スタートへ戻る（楽観歩数マップ）。地図も更新する
    FAST = auto()       # 最短走行を実行中（note_030 §3 S3）
    RETURN2 = auto()    # 最短走行後、次の最短走行のためスタートへ戻る（FAST と同じ「コマンド列実行」機構。是正6）


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
#
# 🔴 是正1（2026-08-19 検収）: 0.2 → 2.0 秒。評価器は機体前端がゴール区画に
# 入ったあと、「機体全体がゴール区画に完全に入り、かつ前進速度 < 0.02 m/s」を
# 最大 5.0 秒（`competition/evaluator.py` の GOAL_STOP_TIMEOUT_S）確認して
# 初めて走行を成立させる。実測（maze_41001）では前端通過から確認完了まで
# 1.33 秒かかっており、0.2 秒のホールドでは余裕が 0.2 秒しかなかった。
# 停止位置が少しずれると走行が goal_not_contained に書き換えられて失われる。
# 2.0 秒なら 5.0 秒の上限に対して十分な余裕がある。走行タイムは前端通過の
# 時点で確定するので、**タイムには影響しない**（持ち時間だけを 2 秒使う）。
#
# 🔴 ホールドは実際には `_goal_stop_hold_ticks` より 1 ティック長く続く。
# `fast:goal_stop` を発行したティックでは `tick()` の先頭の判定が古い識別子
# （前のコマンドの plan_id）を見るため、初回の減算が行われないからである。
# 常に意図より長い方向のずれなので安全側であり、是正しない。
GOAL_STOP_HOLD_S = 2.0

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
                 localization_enabled: bool = True, extend_straights: bool = True,
                 fast_mode: str = "command") -> None:
        if fast_mode not in ("command", "profile"):
            raise ValueError(f"未知の fast_mode: {fast_mode!r}（'command' か 'profile' のいずれか）")
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

        # --- S3': 最短走行のプロファイル追従化 (fast_mode) の実行状態 ---
        # モジュール docstring「S3': 最短走行のプロファイル追従化」参照。
        self.fast_mode = fast_mode
        # ProfileTracker は fast_mode="profile" のときだけ作る
        # （fast_mode="command" のときは一切参照しないので、作らないことで
        # 「コード経路が変更前と完全に同一」の主張を明確にする）。
        self._tracker: Optional[ProfileTracker] = ProfileTracker(self.params) if fast_mode == "profile" else None
        self._fast_plan: Optional[FastPlan] = None
        self._fast_plan_step_idx: int = 0
        # 直近の FAST/RETURN2 が profile 計画を作れず、現行のコマンド方式へ
        # 落ちた（安全弁）かどうか。`_phase_prefix()` が plan_id へ反映する。
        self._fast_command_fallback: bool = False

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
            # コマンド列も空にしておく（是正6）。`_on_stationary_route` は
            # 「`_fast_commands` が空 かつ スタート区画にいる」ことを
            # 「係員回収直後で、まだ RETURN2 の経路を組んでいない」の合図に
            # 使う。ここでクリアしないと、直前に実行していた FAST/RETURN2 の
            # 使用済みコマンド列が残ったまま `_issue_next_fast_command` に
            # 渡り、内部矛盾（コマンド列が尽きた後の実行要求）を誤検出する。
            self._fast_commands = []
            self._fast_cmd_index = 0
            self._fast_straight_cells_left = 0
            self._fast_cells_reanchored = 0
            self._goal_stop_ticks_left = 0
            # profile 計画も同様にクリアする（物理的にスタートへ戻された以上、
            # 進行中だった FastPlan は無効。`_begin_fast_run` が新しく作り直す。
            # `_profile_active` はこれで False になり、次の tick() は
            # `_on_stationary_route` の既存の再合流経路（コマンド方式・profile方式
            # 共通）を通る）。
            self._fast_plan = None
            self._fast_plan_step_idx = 0
            self._fast_command_fallback = False

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

        # S3': fast_mode="profile" で FAST/RETURN2 の profile 計画を実行中なら、
        # 以下のコマンド方式の分岐は一切通らずここで確定させる（モジュール
        # docstring「S3': 最短走行のプロファイル追従化」参照）。
        # fast_mode="command"（既定）のときは `_profile_active` が常に False
        # になるので、この分岐そのものが実行されない（コード経路は不変）。
        if self._profile_active:
            return self._tick_profile(obs)

        vl, vr, done = self.motion.update(obs)

        # S3 (b): FAST・RETURN2（是正6）の多区画直進の途中、区画中心を
        # 通過するたびに測り直して掛け直す（note_030 §3 S3 任務指示。
        # モジュール docstring 参照）。地図は書き換えない
        # （センサは位置補正にのみ使う）。
        if (self.phase in (Phase.FAST, Phase.RETURN2) and self.extend_straights
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
                    # 🔴 想定外の事態（`_enter_route_blocked` docstring参照。
                    # 地図の誤りが原因で、直進の途中に実際にはあるはずの壁へ
                    # 衝突しつつ車輪だけ空転し、推測航法の距離推定が迷路の
                    # 外まで進んでしまうことがある）。例外を投げず停止する。
                    self._enter_route_blocked(
                        f"{self.phase.name} の直進中、区画 {self.cell} から向き {self.heading} への"
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
                # 是正6: 最短走行のあとの帰路（RETURN2）も、足立法ではなく
                # plan_route のコマンド列を実行する経路計画で始める。
                self._begin_return2_run(obs)
                vl, vr, done = self.motion.update(obs)
        elif done and self._active_kind is not MotionKind.STOP:
            # 直進 1 区画／旋回のどちらかが完了した瞬間。位置・方位を確定させ、
            # 同じ区画中心で（あるいは旋回直後の同じ区画で）直ちに次の判断へ
            # 進む。判断そのものは計算のみでシミュレーション時間を消費しない
            # （実機の「止まって読んで決めてまた走る」を、判断部分は瞬時と
            # 単純化して再現している）。
            self._advance_state()
            if self._active_kind is MotionKind.STOP:
                # _advance_state() が FAST/RETURN2（是正6）の異常事態を検出し、
                # 既に "fast:blocked"/"return2:blocked" へ切り替え済み
                # （`_enter_route_blocked` 参照）。_on_stationary を呼ぶと
                # 即座に次のコマンドを発行してしまい停止が消えるので呼ばない。
                vl, vr = 0.0, 0.0
            else:
                self._on_stationary(obs)
                vl, vr, _done2 = self.motion.update(obs)

        return vl, vr, self._active_plan_id

    # ------------------------------------------------------------------
    # 区画中心（または旋回直後の同一区画、コマンド完了直後）での処理
    # ------------------------------------------------------------------
    def _on_stationary(self, obs) -> None:
        if self.phase in (Phase.FAST, Phase.RETURN2):
            # 是正6: Phase.RETURN2 も Phase.FAST と同じ「plan_route の
            # コマンド列を実行する」機構を使うため、ここで合流する
            # （`_on_stationary_route` 参照）。
            self._on_stationary_route(obs)
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

        🔴 Phase.FAST・Phase.RETURN2（是正6）中は呼ばれない
        （`_on_stationary_route` を参照。最短走行・その帰路のどちらも
        地図を書き換えない）。"""
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
        base = "fast" if self.phase is Phase.FAST else "return2"  # Phase.RETURN2
        if self.fast_mode == "profile" and self._fast_command_fallback:
            # fast_mode="profile" で plan_fast_run() が計画できず（安全弁）、
            # 現行のコマンド方式へ落ちたことを一次記録(plan_id)へ残す
            # （モジュール docstring「S3': 最短走行のプロファイル追従化」参照）。
            # fast_mode="command"（既定）のときは self.fast_mode=="profile" が
            # 常に False なのでこの分岐へ入らず、返り値は変更前と同一のまま。
            return f"{base}_fallback"
        return base

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
                if self.phase in (Phase.FAST, Phase.RETURN2):
                    # 🔴 想定外の事態（`_enter_route_blocked` docstring・
                    # `tick()` のコメント参照）。例外を投げず停止する。
                    # EXPLORE/RETURN では `_pick_next_direction` が
                    # `neighbor() is None` の方位を最初から候補から除外して
                    # いるためこの分岐に到達しない（現行どおり例外のまま）。
                    self._enter_route_blocked(
                        f"{self.phase.name} の直進コマンド完了時、区画 {self.cell} から向き "
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
    # S3: 最短走行 (Phase.FAST) とその帰路 (Phase.RETURN2) の実行
    # ------------------------------------------------------------------
    # 🔴 是正6（2026-08-19 検収）: 以下のコマンド列実行の仕組みは
    # Phase.FAST と Phase.RETURN2 の両方から使う共通実装である。
    # plan_id の接頭辞は `_phase_prefix()` により実行時の self.phase から
    # 自動的に決まる（"fast:*" / "return2:*"）。内部の状態変数名は
    # （元が FAST 専用実装だった名残で）`_fast_*` のままだが、意味は
    # 「現在実行中のコマンド列」であり RETURN2 実行中も同じ変数を使う
    # （同時に FAST と RETURN2 が実行中になることは無いので、専用の変数を
    # 増やしてもコードが重複するだけで意味が無い）。
    def _enter_route_blocked(self, reason: str) -> None:
        """FAST・RETURN2 実行中に想定外の事態（経路が引けない・実行できない）
        が起きたとき、例外を投げずにその場で停止する（note_030 §3 S3 任務
        指示 5・現行の "explore:blocked"/"return:blocked" と同じ思想。
        note_029 の教訓）。plan_id は呼び出し時点の self.phase から
        `_phase_prefix()` で決まる（"fast:blocked" / "return2:blocked"）。

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
        self._active_plan_id = f"{self._phase_prefix()}:blocked"

    def _plan_route_or_block(self, start: Cell, goals: List[Cell], start_heading: Direction,
                              reason_prefix: str) -> Optional[List[Command]]:
        """`plan_route` を呼び、失敗（`NoRouteError`）したら例外を投げず
        `_enter_route_blocked` で停止する（FAST・RETURN2 共通。是正6で
        `_begin_fast_run`・`_begin_return2_run` の重複を無くすために切り出した）。

        呼び出し側は**先に `self.phase` を確定させてから**呼ぶこと
        （`_enter_route_blocked` の plan_id が `self.phase` に依存するため）。
        成功時はコマンド列を返し、失敗時は None を返す（呼び出し側は
        None なら何もせず return するだけでよい。blocked への遷移は
        ここで完了している）。"""
        try:
            _path, commands = plan_route(
                self.maze, start=start, goals=goals,
                mode=FloodMode.PESSIMISTIC, start_heading=start_heading,
            )
        except NoRouteError as exc:
            self._enter_route_blocked(f"{reason_prefix}: {exc}")
            return None
        return commands

    def _begin_fast_run(self, obs) -> None:
        """悲観歩数マップで「既知の壁だけでゴールへ到達できる」と判定された
        瞬間（RETURN 完了時）、または最短走行の帰路がスタートへ戻り着いた
        瞬間（RETURN2 完了時）に呼ぶ。`plan_route` で最短経路のコマンド列を
        求め、Phase.FAST へ入って先頭のコマンドを発行する（note_030 §3 S3 ①）。

        fast_mode="profile" のときは `plan_fast_run()` の速度プロファイル計画を
        代わりに使う（モジュール docstring「S3': 最短走行のプロファイル
        追従化」参照）。計画できなければ（`_fast_command_fallback=True`）、
        以下の既存のコマンド方式へそのままフォールスルーする（安全弁）。
        fast_mode="command"（既定）のときは以下の処理が変更前と完全に同一。"""
        self.phase = Phase.FAST
        if self.fast_mode == "profile":
            # 🔴 profile 計画は現在の実際の向き(self.heading)から作る
            # （command方式のような「北固定で計画してから北へ向き直す」補正は
            # 不要 — plan_fast_run/ideal.py は任意の start_heading をそのまま
            # 扱えるため、実行前の余計な旋回を挟まない）。
            self._begin_profile_run(
                obs, start=self.start_cell, goals=self._goal_cell_list, start_heading=self.heading,
            )
            if not self._fast_command_fallback:
                return
        commands = self._plan_route_or_block(
            start=self.start_cell, goals=self._goal_cell_list, start_heading=Direction.N,
            # 🔴 通常は起こらないはず（悲観歩数マップで到達可能と確認した
            # 直後、または既に確認済みの地図で呼んでいる）。それでも起きたら
            # 例外で評価全体を落とさず、想定外を静かに握りつぶさない方針
            # （モジュール docstring・note_030 §5）どおり停止する。
            reason_prefix="最短経路の計画に失敗しました",
        )
        if commands is None:
            return

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

    def _begin_return2_run(self, obs) -> None:
        """最短走行のゴール停止ホールドが終わった瞬間に呼ぶ（是正6・
        note_030 §3 S3 改訂）。地図は探索完了時点で既に確定しているので、
        `Phase.RETURN`（探索完了前の帰還）のように足立法で 1 区画ずつ確かめる
        必要が無い。`_begin_fast_run` と同じ「plan_route のコマンド列を
        求めて実行する」機構を、目標をスタート区画にして使う。

        🔴 FAST と異なり `start_heading` は `Direction.N` 固定ではなく
        **現在の実際の向き** `self.heading` をそのまま渡す。ゴール区画から
        スタートへ戻る向きは経路そのもの（ゴール区画のどの出口から出るか）
        で決まり、FAST のように「plan_route が想定する向き」と「実際の向き」
        がずれる余地が無い（FAST は北固定で計画するため、帰還後の実際の
        向きとのズレを実行時の旋回で吸収する必要があった）。したがって
        `_begin_fast_run` にある「先頭への向き合わせ旋回」は不要。

        fast_mode="profile" のときの扱いは `_begin_fast_run` と同じ
        （モジュール docstring参照。計画できなければ既存のコマンド方式へ
        フォールスルーする）。"""
        self.phase = Phase.RETURN2
        if self.fast_mode == "profile":
            self._begin_profile_run(
                obs, start=self.cell, goals=[self.start_cell], start_heading=self.heading,
            )
            if not self._fast_command_fallback:
                return
        commands = self._plan_route_or_block(
            start=self.cell, goals=[self.start_cell], start_heading=self.heading,
            # 🔴 通常は起こらないはず（悲観歩数マップで既に「既知の壁だけで
            # 到達可能」と確認済みの地図であり、しかも今回は FAST 自身が
            # 実際に通った経路の終点から出発するため）。それでも起きたら
            # 想定外を静かに握りつぶさない方針どおり停止する。
            reason_prefix="スタートへの最短経路の計画に失敗しました",
        )
        if commands is None:
            return

        self._fast_commands = commands
        self._fast_cmd_index = 0
        self._fast_straight_cells_left = 0
        self._fast_cells_reanchored = 0
        self._issue_next_fast_command(obs)

    # ------------------------------------------------------------------
    # S3': 最短走行のプロファイル追従化 (fast_mode="profile")
    # ------------------------------------------------------------------
    # 🔴 このブロックは fast_mode="profile" のときにしか呼ばれない
    # （`_begin_fast_run`/`_begin_return2_run` の `if self.fast_mode ==
    # "profile":` 分岐、および `tick()` の `_profile_active` チェック経由のみ）。
    # fast_mode="command"（既定）のコード経路はここを一切通らない。
    @property
    def _profile_active(self) -> bool:
        """このティック、`tick()` を `_tick_profile()` へ委ねるべきか。

        profile 計画（`self._fast_plan`）が読み込まれている間だけ True になる。
        計画の全ステップが完了する（`_finish_profile_plan()`）と `self._fast_plan`
        が None に戻るので、以後は既存のコマンド方式の `tick()` 本体（ゴール
        停止ホールドや `_begin_return2_run`/`_begin_fast_run` の呼び出しを
        含む）が自然に引き継ぐ。"""
        return (self.fast_mode == "profile" and self.phase in (Phase.FAST, Phase.RETURN2)
                and self._fast_plan is not None)

    def _profile_plan_id(self, step: Union[PathBlock, SpinSegment]) -> str:
        """profile 実行中の1ステップの plan_id。経路追従は
        `"fast:profile"`/`"return2:profile"`、その場旋回は
        `"fast:profile_spin"`/`"return2:profile_spin"`（任務指示）。"""
        prefix = "fast" if self.phase is Phase.FAST else "return2"
        suffix = "profile_spin" if isinstance(step, SpinSegment) else "profile"
        return f"{prefix}:{suffix}"

    def _begin_profile_run(self, obs, start: Cell, goals: List[Cell], start_heading: Direction) -> None:
        """profile モードで最短走行(FAST/RETURN2共通)の計画を作り、実行を
        始める。`classic/fast_planner.py` の `plan_fast_run()` が `None` を
        返した（既知の壁だけでは到達できない）場合は `self._fast_command_fallback
        = True` にするだけで、実際のフォールバック実行は呼び出し元
        （`_begin_fast_run`/`_begin_return2_run`）の既存コマンド方式コードに
        委ねる（本メソッドはここで停止も例外も出さない）。"""
        assert self._tracker is not None  # fast_mode="profile" のときだけ呼ばれる
        plan = plan_fast_run(self.maze, start=start, goals=goals, start_heading=start_heading,
                              params=self.params)
        if plan is None:
            self._fast_command_fallback = True
            self._fast_plan = None
            return

        self._fast_command_fallback = False
        self._fast_plan = plan
        self._fast_plan_step_idx = 0

        if not plan.steps:
            # 既にゴール/スタートにいる(trivial)。追従するステップが無いので
            # 即座に完了扱いにする（`_complete_profile_plan` が次の遷移
            # — ゴール停止ホールド、またはRETURN2ならそのまま次のFASTへ — を行う）。
            self._complete_profile_plan(obs)
            return

        self._tracker.reset(heading_deg=math.degrees(plan.steps[0].psi_start))
        self._load_current_profile_step()

    def _load_current_profile_step(self) -> None:
        """`self._fast_plan.steps[self._fast_plan_step_idx]` をトラッカーへ
        ロードする（経路追従なら `load_plan`、その場旋回なら `load_spin_plan`）。
        呼び出し前に添字が範囲内であることは呼び出し元が保証する。"""
        assert self._fast_plan is not None and self._tracker is not None
        step = self._fast_plan.steps[self._fast_plan_step_idx]
        if isinstance(step, PathBlock):
            self._tracker.load_plan(step.s_grid, step.v_ref, step.kappa_ref, psi_start=step.psi_start)
            # `ProfileTracker.load_plan()` は弧長 s をクリアしない（複数の計画を
            # 順に読み込む前提の設計。`tests/test_tracker.py` の引き継ぎ検査と
            # 同じ作法で、新しい区間として明示的に 0 へ戻す）。
            self._tracker._s = 0.0
        else:
            assert isinstance(step, SpinSegment)
            self._tracker.load_spin_plan(step.delta_theta, psi_start=step.psi_start)
        self._active_plan_id = self._profile_plan_id(step)

    def _finish_profile_plan(self) -> None:
        """profile 計画（`self._fast_plan`）の全ステップが完了したときに呼ぶ。
        自前の位置カウンタ（`self.cell`/`self.heading`。command方式の
        `_advance_state()` に相当）を計画の終点へ合わせ、計画状態をクリアする
        （クリアすることで `_profile_active` が False に戻り、以後は既存の
        `tick()`/`_on_stationary` の経路が引き継ぐ）。"""
        assert self._fast_plan is not None
        cells = self._fast_plan.cells
        if len(cells) >= 2:
            self.heading = direction_between(cells[-2], cells[-1])
        self.cell = cells[-1]
        self._fast_plan = None
        self._fast_plan_step_idx = 0

    def _complete_profile_plan(self, obs) -> Tuple[float, float]:
        """profile 計画の全ステップが完了した（または最初から1つも無かった）
        ときの遷移。Phase.FAST ならゴール停止ホールドへ（既存のコマンド方式と
        同じ `"fast:goal_stop"` の仕組みをそのまま再利用する）、Phase.RETURN2
        ならホールドせずそのまま次の最短走行へ（是正6と同じ思想）。
        このティックの (v_left, v_right) を返す。"""
        self._finish_profile_plan()
        if self.phase is Phase.FAST:
            self.motion.start_stop()
            self._active_kind = MotionKind.STOP
            self._active_plan_id = "fast:goal_stop"
            self._goal_stop_ticks_left = self._goal_stop_hold_ticks
            return 0.0, 0.0
        # Phase.RETURN2: 是正6と同じくホールドせず次の最短走行へ。
        self._begin_fast_run(obs)
        return self._drive_active(obs)

    def _drive_active(self, obs) -> Tuple[float, float]:
        """現在アクティブな実行系（profileのtracker、またはコマンド方式の
        motion）からこのティックの電圧だけを取り出す（done判定の結果は
        使わない。`_begin_fast_run`/`_begin_return2_run` が profile
        計画を積んだか、コマンド方式へ落ちたかのどちらでも正しく動く）。"""
        if self._profile_active:
            vl, vr, _done = self._tracker.update(obs)
            return vl, vr
        vl, vr, _done = self.motion.update(obs)
        return vl, vr

    def _tick_profile(self, obs) -> Tuple[float, float, str]:
        """Phase.FAST・Phase.RETURN2 の profile モード実行本体
        （`_profile_active` が True のときだけ `tick()` から呼ばれる）。

        `ProfileTracker.update()` を毎ティック回し、計画の1ステップ（経路追従
        またはその場旋回）が完了したら、区間の切れ目で速度・角速度が
        しきい値未満に収束するのを待たず、即座に次のステップへ進む
        （`classic/tracker.py` の設計そのもの。既存のコマンド方式の
        `_advance_state()` 直後に `_on_stationary()`→`motion.update()` を
        同じティックでもう一度呼ぶのと同じ作法を踏襲する）。

        🔴 S2 の壁センサ位置補正はここでは行わない（推測航法のみ。本任務の
        範囲外 — モジュール docstring参照）。"""
        assert self._fast_plan is not None and self._tracker is not None
        vl, vr, done = self._tracker.update(obs)
        if not done:
            return vl, vr, self._active_plan_id

        self._fast_plan_step_idx += 1
        if self._fast_plan_step_idx < len(self._fast_plan.steps):
            self._load_current_profile_step()
            vl, vr, _done2 = self._tracker.update(obs)
            return vl, vr, self._active_plan_id

        # 計画の全ステップが完了 = ゴール(FAST)またはスタート(RETURN2)に到達した。
        vl, vr = self._complete_profile_plan(obs)
        return vl, vr, self._active_plan_id

    def _on_stationary_route(self, obs) -> None:
        """Phase.FAST・Phase.RETURN2 共通: コマンド列（plan_route の出力）の
        実行中、コマンドが完了した瞬間（または開始直後）の処理。

        🔴 地図は書き換えない（`_update_map_from_sensing` を呼ばない。
        note_030 §3 S3 任務指示・是正6。センサは位置補正にのみ使う）。"""
        if (self.phase is Phase.RETURN2 and self.cell == self.start_cell
                and not self._fast_commands):
            # 係員回収（`handle_retrieval`）直後、物理的にスタート区画へ
            # 戻された状態でここへ最初に来たケース。RETURN2 のコマンド列を
            # まだ何も組んでいない（`_fast_commands` が空。
            # `handle_retrieval` がそのためにクリアしてある）ので、改めて
            # スタートへの経路を計画し直す代わりに、既にスタートにいる以上
            # そのまま次の最短走行へ入る（note_030 §3 任務指示: 「段階が
            # FAST/RETURN2 ならスタート区画から最短走行をやり直せること」）。
            self._begin_fast_run(obs)
            return
        if self._fast_straight_cells_left > 0:
            self._continue_fast_straight_leg(obs)
            return
        self._issue_next_fast_command(obs)

    def _issue_next_fast_command(self, obs) -> None:
        """`self._fast_commands[self._fast_cmd_index]` を発行し、
        インデックスを進める（Phase.FAST・Phase.RETURN2 共通）。"""
        if self._fast_cmd_index >= len(self._fast_commands):
            # 🔴 想定外（GOAL_STOP は必ず最後に来るはずで、通常ここには
            # 来ない）。例外を投げず、その場で停止する
            # （note_030 §3 S3 任務指示 5・モジュール docstring）。
            self._enter_route_blocked(
                f"{self.phase.name} のコマンド列が尽きた後にも実行要求が来ました（内部矛盾）")
            return

        cmd = self._fast_commands[self._fast_cmd_index]
        self._fast_cmd_index += 1

        if cmd.type == CommandType.GOAL_STOP:
            if self.phase is Phase.RETURN2:
                # 🔴 是正6: RETURN2 の終端（スタート区画に着いた）は
                # "return2:goal_stop" というホールドを持たない。スタート
                # 区画での静止は評価器の判定に関係しないため、ホールドせず
                # そのまま次の最短走行へ移る（モジュール docstring
                # 「S3: 最短走行」節参照）。
                self._begin_fast_run(obs)
                return
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
        """FAST・RETURN2（是正6）共通: STRAIGHT n を（新しいコマンドとして）
        発行する（note_030 §3 S3 ②③）。plan_id は `_phase_prefix()` により
        "fast:straight" / "return2:straight" のどちらかになる。

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
        self._active_plan_id = f"{self._phase_prefix()}:straight"

    def _continue_fast_straight_leg(self, obs) -> None:
        """`extend_straights=False` のときの、同一 STRAIGHT コマンド内の
        続きの 1 区画を発行する（対照。note_030 §3 S3 任務指示 4。
        FAST・RETURN2 共通・是正6）。コマンド列のインデックスは進めない
        （同じ STRAIGHT の続きなので）。"""
        self._fast_straight_cells_left -= 1
        self._fast_cells_reanchored = 0
        self._fast_straight_total_cells = 1
        sensing = sense_walls(obs, self.params)
        self.motion.start_forward(1)
        self._apply_fast_straight_bias(sensing)
        self._active_kind = MotionKind.FORWARD
        self._active_plan_id = f"{self._phase_prefix()}:straight"

    def _apply_fast_straight_bias(self, sensing: WallSensing) -> None:
        """FAST・RETURN2（是正6）共通: 直進コマンド発行直後の横位置補正
        （S2 (a) と同じ入り口）。n 区画コマンドの「出発点で 1 回」ぶんに
        相当する（途中の n-1 回は `tick()` の区画ごと掛け直しが受け持つ。
        note_030 §3 S3 任務指示）。"""
        bias = self.localizer.lateral_bias_for_forward(sensing, cell=self.cell, heading=int(self.heading))
        if bias != 0.0:
            self.motion.bias_target_heading(bias)
