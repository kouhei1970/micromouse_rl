"""
competition/baseline_classical.py
================
古典ベースライン方策 AdachiPolicy（M0-T3）。

自律度ラダー L0（docs/RESEARCH_PLAN.md §7）:
  運動制御 = 古典（プログラム動作 + PD 制御）
  経路計画・探索 = 古典（足立法 flood-fill）
  地図・記憶 = 古典（明示的な壁地図、方策の内部状態として走行間保持）

「学習しない基準」。特権情報（真の自己位置・真の壁情報）を使用する
（requires_privileged=True。competition/policy_interface.py の bind_sim/bind_maze
経由で評価器から渡される）。

--------------------------------------------------------------------------
足立法（flood-fill）
--------------------------------------------------------------------------
方策は壁地図 v_walls_known (W+1,H) / h_walls_known (W,H+1) を内部状態として持つ
（値は {0: 壁なし, 1: 壁あり, -1: 未知}）。初期値は外周=1・スタート区画規定
（competition/maze_gen.py の規約: 北開放 h[0,1]=0・東壁あり v[1,0]=1）・
それ以外は -1（未知）。

セル中心に到達し停止するたび（SENSE）、bind_maze で受け取った真の壁配列から
現在セルの 4 方向を読み取り、既知地図に書き込む。

flood-fill（PLAN）は目標セル集合（探索中はゴール 4 セル、帰路はスタート
(0,0)）から multi-source BFS で距離場を計算する。**未知壁 (-1) は「壁なし」
とみなす（楽観的）**。ループ・複数最短経路の存在を前提に BFS で自然に
対応する。次セルは距離最小の隣接セル（同距離が複数あれば決定的タイブレーク:
直進優先 → 前回進行方向から時計回り優先）。

現在セルが目標集合に属する（距離 0）ときは、目標集合をゴール⇔スタートで
反転させてから次セルを選ぶ。これにより「ゴール到達→自走帰還→再出発」が
状態機械の変更なしに自然に実現される（評価器側はロボットの物理位置のみで
走行境界を判定するため、方策は単に走り続ければよい）。

--------------------------------------------------------------------------
stop-and-go 走行（状態機械）
--------------------------------------------------------------------------
act() は 100Hz で呼ばれ、内部状態機械 IDLE → (SENSE → PLAN → TURN* → FORWARD)*
を駆動する。SENSE/PLAN は瞬時（同一 act() 呼び出し内で完結）、TURN/FORWARD は
複数ティックにまたがって PD 制御でセル中心・目標方位に収束させる。

- TURN（超信地旋回）: 目標方位に対する yaw 誤差 + ジャイロ角速度の PD で
  左右逆方向の目標車輪角速度を作り、|yaw誤差|<2° かつ |gyro_z|<0.2rad/s で完了。
- FORWARD（1セル直進）: 台形速度プロファイル（v_max, a_forward）+ 固定セル軸
  方位の維持・経路からの横偏差の PD 補正。速度指令の「大きさ」は目標点までの
  符号なしユークリッド残り距離から（横方向のみの誤差が残っていても0に
  張り付かない）、「向き」（前進/後退）は経路方向への符号付き進捗から決める
  （試行錯誤の経緯は baseline_classical.py の _do_forward docstring 参照）。
  セル中心まで <1cm かつ |v|<0.05m/s で完全停止を待って完了。
- 電圧生成: 目標車輪角速度 ω_des から V = Ke_eff・ω_des + Kp_wheel・(ω_des−ω_act)
  （Ke_eff = motor_Ke × gear_ratio は bind_sim で受け取った sim.params から導出。
  ハードコード禁止）。出力は必ず ±voltage_limit にクリップ。

運動学（差動二輪、点旋回）は標準的な符号規約に従う（実測較正で確認済み:
右車輪 正方向・左車輪 負方向で機体 yaw が増加＝反時計回り。単純な
前後2点だけを比較する較正は角度のラップ（±180°をまたぐ周回）で見かけの
符号を誤読しやすく、実際に軌跡を時系列でサンプルして確認した）:
  ωL_des = v_cmd/r − ω_robot_cmd・tread/(2r)
  ωR_des = v_cmd/r + ω_robot_cmd・tread/(2r)
（r=wheel_radius, tread=車輪間隔, ω_robot_cmd: 正=反時計回り）

on_retrieval() 受信時は状態機械を IDLE に戻し自己位置推定を (0,0) 北向きに
初期化する（地図は保持。評価器が sim を物理的にも reset_to_start 済み）。
"""
import math
from collections import deque

import numpy as np

from competition.evaluator import goal_cells, pos_to_cell
from competition.explore_e1 import explore_targets, is_shortest_confirmed, passable
from competition.policy_interface import MousePolicy

# 4 方位。北=+y, 東=+x, 南=-y, 西=-x（evaluator.py の v_walls/h_walls 座標規約と
# heading_deg=90(北) の初期姿勢に整合）。
DIRS = ("N", "E", "S", "W")
_DELTA = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
_HEADING_RAD = {"N": math.pi / 2, "E": 0.0, "S": -math.pi / 2, "W": math.pi}
# 時計回りの次方位（タイブレーク規則: 直進優先→前回進行方向から時計回り優先 で使用）
_CW_NEXT = {"N": "E", "E": "S", "S": "W", "W": "N"}


def _wrap_pi(angle: float) -> float:
    """角度を [-pi, pi) にラップする。"""
    return math.atan2(math.sin(angle), math.cos(angle))


class AdachiPolicy(MousePolicy):
    """足立法（flood-fill）+ stop-and-go（超信地旋回）による古典ベースライン。

    自律度ラダー L0 実装（docs/RESEARCH_PLAN.md §7 表）。特権情報
    （bind_sim/bind_maze）を使用するため requires_privileged=True。

    ゲイン等はコンストラクタ引数で調整可能（既定値は spec §2.2 の初期値、
    または tests/test_baseline.py での整定結果）。
    """

    name = "adachi_classical"
    requires_privileged = True

    def __init__(self,
                 v_max: float = 0.3, a_forward: float = 2.0,
                 kp_turn: float = 8.0, kd_turn: float = 0.6, turn_omega_limit: float = 10.0,
                 kp_heading: float = 3.0, kp_lateral: float = 8.0, kd_heading: float = 0.35,
                 kp_wheel: float = 0.05,
                 turn_done_deg: float = 2.0, turn_done_gyro: float = 0.2,
                 forward_done_dist: float = 0.01, forward_done_speed: float = 0.05):
        # --- 走行プロファイル（spec §2.2 既定値） ---
        self.v_max = v_max
        self.a_forward = a_forward
        # --- TURN PD ゲイン ---
        self.kp_turn = kp_turn
        self.kd_turn = kd_turn
        self.turn_omega_limit = turn_omega_limit  # 目標ロボット角速度の上限 [rad/s]
        # --- FORWARD PD ゲイン（固定セル軸方位の維持 + 経路横偏差の補正） ---
        self.kp_heading = kp_heading
        self.kp_lateral = kp_lateral
        self.kd_heading = kd_heading
        # --- 車輪速度フィードバックゲイン（電圧生成） ---
        self.kp_wheel = kp_wheel
        # --- 完了判定しきい値 ---
        self.turn_done_rad = math.radians(turn_done_deg)
        self.turn_done_gyro = turn_done_gyro
        self.forward_done_dist = forward_done_dist
        self.forward_done_speed = forward_done_speed

        # --- bind_sim/bind_maze で埋まる特権情報 ---
        self._sim = None
        self._true_v = None
        self._true_h = None

        # --- params から導出する物理定数（bind_sim で上書きされるまでの既定値） ---
        self.cell_size = 0.18
        self.wheel_radius = 0.0135
        self.tread = 0.072
        self.Ke_eff = 9.9e-3
        self.voltage_limit = 3.0
        self.friction_ff_v = 0.1
        self.control_dt = 0.01

        self.width = 16
        self.height = 16
        self.v_walls_known = None
        self.h_walls_known = None

        self._reset_run_state()

    # ------------------------------------------------------------------
    # 内部状態リセット
    # ------------------------------------------------------------------
    def _reset_run_state(self):
        """状態機械・現在の走行脚（TURN/FORWARD）の作業変数をリセットする。
        壁地図（v_walls_known/h_walls_known）と target_mode は保持しない
        呼び出し元（on_maze_start）と、保持する呼び出し元（on_retrieval）の
        両方から呼ばれるため、target_mode のリセットは on_retrieval 側の
        要件（自己位置を(0,0)に戻す＝再びゴールを目指す）に合わせて
        ここで "to_goal" にする。on_maze_start は迷路開始時なのでこれで正しい。
        """
        self._state = "IDLE"
        self.target_mode = "to_goal"
        self._heading_dir = "N"
        self._planned_dir = None
        self._planned_next_cell = None
        self._turn_target_yaw = None
        self._fwd_start = None
        self._fwd_target = None
        self._fwd_total_dist = None
        self._fwd_v_setpoint = 0.0  # 台形速度プロファイルのレート制限済み速度設定値 [m/s]（常に>=0）

    # ------------------------------------------------------------------
    # 特権情報バインド（policy_interface.py の契約）
    # ------------------------------------------------------------------
    def bind_sim(self, sim) -> None:
        self._sim = sim
        p = sim.params
        # 観測ベクトルの並びは [距離 ×n, accel(3), gyro(3), 車輪角速度(2)] であり、
        # 距離センサ本数 n は構成で変わる（研究計画書 r6 でセンサ 6→4 本に変更）。
        # 固定インデックスを持つと構成変更で静かに壊れるため、本数から導出する。
        n_dist = len(p.sensors)
        self._i_gyro_z = n_dist + 5   # gyro z（accel3 のあと gyro x,y,z の 3 番目）
        self._i_wheel = n_dist + 6    # 車輪角速度 左（次が右）
        # Ke_eff = motor_Ke * gear_ratio 等、制御ゲインの元になる物理定数は
        # すべて params から導出する（ハードコード禁止）。
        self.cell_size = p.cell_size
        self.wheel_radius = p.wheel_radius
        self.tread = p.tread
        self.Ke_eff = p.motor_Ke * p.gear_ratio
        self.voltage_limit = p.voltage_limit
        self.control_dt = p.control_dt
        # 乾性摩擦（frictionloss）のフィードフォワード補償電圧。
        # 車輪ジョイントの静止摩擦トルクを乗り越えるのに必要な最小電圧
        # V_break = wheel_frictionloss / gainprm0（gainprm0 = N*Kt/R [N・m/V]）。
        # 目標値近傍で P 制御の指令が V_break を下回ると車輪が乾性摩擦で
        # 全く動かず、収束が極端に遅くなる（テストで実測確認済み）ため、
        # 進行方向に応じた符号付きの一定電圧として上乗せする（安全率 1.3 倍）。
        self.friction_ff_v = 1.3 * p.wheel_frictionloss / p.gainprm0

    def bind_maze(self, v_walls, h_walls) -> None:
        self._true_v = v_walls
        self._true_h = h_walls

    # ------------------------------------------------------------------
    # ライフサイクル通知フック
    # ------------------------------------------------------------------
    def on_maze_start(self, maze_info: dict) -> None:
        self.width = int(maze_info["width"])
        self.height = int(maze_info["height"])

        self.v_walls_known = np.full((self.width + 1, self.height), -1, dtype=int)
        self.h_walls_known = np.full((self.width, self.height + 1), -1, dtype=int)
        # 外周は既知（壁あり）
        self.v_walls_known[0, :] = 1
        self.v_walls_known[self.width, :] = 1
        self.h_walls_known[:, 0] = 1
        self.h_walls_known[:, self.height] = 1
        # スタート区画規定（competition/maze_gen.py 手順4・spec §2.1）:
        # 北開放・東壁あり
        if self.height > 1:
            self.h_walls_known[0, 1] = 0
        if self.width > 1:
            self.v_walls_known[1, 0] = 1

        self._reset_run_state()

    def on_run_start(self, run_index: int) -> None:
        pass

    def on_run_end(self, outcome: str) -> None:
        # outcome=="goal" 以外は評価器が直後に on_retrieval() を呼ぶため、
        # 状態リセットはそちらに一任する。outcome=="goal" は状態機械側で
        # 既に「ゴール到達→target_mode反転→帰路継続」を処理済み。
        pass

    def on_retrieval(self) -> None:
        # 係員回収: 状態機械を IDLE に戻し、自己位置推定を (0,0)・北向きに
        # 初期化する（地図=v_walls_known/h_walls_known は保持）。
        # sim 自体は評価器が reset_to_start((0,0), 90deg) 済み。
        self._reset_run_state()

    # ------------------------------------------------------------------
    # 足立法（flood-fill）
    # ------------------------------------------------------------------
    def _target_cells(self):
        if self.target_mode == "to_goal":
            return goal_cells(self.width, self.height)
        if self.target_mode == "verify":
            # 追加探索（E1）: 「開いていたら最短経路に使われうる未知壁」に隣接する
            # 区画を目標にする。そこへ行って壁を観測すれば未知壁が確定する
            t = explore_targets(self.v_walls_known, self.h_walls_known,
                                 self.width, self.height, (0, 0),
                                 goal_cells(self.width, self.height))
            if t:
                return sorted(t)
            # 確定済み → 帰路へ切り替え
            self.target_mode = "to_start"
        return [(0, 0)]

    def _shortest_confirmed(self) -> bool:
        """現在の楽観最短経路が真の最短経路として確定しているか（E1）。"""
        return is_shortest_confirmed(self.v_walls_known, self.h_walls_known,
                                      self.width, self.height, (0, 0),
                                      goal_cells(self.width, self.height))

    def _connects_known(self, x: int, y: int, nx: int, ny: int) -> bool:
        """既知壁配列で (x,y)-(nx,ny) が通行可能か。

        E1（研究計画書 §7）の**楽観・悲観の非対称性**:
        往路探索（to_goal）と追加探索（verify）は**楽観的**（未知壁 = 通行可）に進む。
        **帰路（to_start）だけは悲観的**（未知壁 = 壁）に既知の経路だけで戻る
        — 未確認の壁を信じて帰路に突っ込むと行き止まりに追い込まれるため。
        """
        if self.target_mode == "to_start":
            return passable(self.v_walls_known, self.h_walls_known, x, y, nx, ny,
                             pessimistic=True)
        if (nx, ny) == (x + 1, y):
            v = self.v_walls_known[x + 1, y]
        elif (nx, ny) == (x - 1, y):
            v = self.v_walls_known[x, y]
        elif (nx, ny) == (x, y + 1):
            v = self.h_walls_known[x, y + 1]
        elif (nx, ny) == (x, y - 1):
            v = self.h_walls_known[x, y]
        else:
            raise ValueError(f"({x},{y}) と ({nx},{ny}) は隣接していません")
        return v != 1

    def _flood_fill(self, targets) -> dict:
        """targets からの multi-source BFS 距離場（未知壁=壁なし扱い）。
        ループ・複数最短経路のある壁配置でも BFS なので自然に正しい距離になる。"""
        dist = {c: 0 for c in targets}
        q = deque(targets)
        while q:
            x, y = q.popleft()
            for d in DIRS:
                dx, dy = _DELTA[d]
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if (nx, ny) in dist:
                    continue
                if self._connects_known(x, y, nx, ny):
                    dist[(nx, ny)] = dist[(x, y)] + 1
                    q.append((nx, ny))
        return dist

    @staticmethod
    def _tiebreak_order(prev_dir: str):
        """決定的タイブレーク順: 直進優先 → 前回進行方向から時計回り優先。"""
        order = [prev_dir]
        d = prev_dir
        for _ in range(3):
            d = _CW_NEXT[d]
            order.append(d)
        return order

    # ------------------------------------------------------------------
    # SENSE
    # ------------------------------------------------------------------
    def _do_sense(self, cx: int, cy: int) -> None:
        """現在セル (cx,cy) の4方向の真の壁を読み、既知地図に書き込む。"""
        self.v_walls_known[cx + 1, cy] = self._true_v[cx + 1, cy]
        self.v_walls_known[cx, cy] = self._true_v[cx, cy]
        self.h_walls_known[cx, cy + 1] = self._true_h[cx, cy + 1]
        self.h_walls_known[cx, cy] = self._true_h[cx, cy]

    # ------------------------------------------------------------------
    # PLAN
    # ------------------------------------------------------------------
    def _do_plan(self, x: float, y: float) -> None:
        cur_cell = pos_to_cell(x, y, self.width, self.height, self.cell_size)
        targets = self._target_cells()
        dist_field = self._flood_fill(targets)

        if dist_field.get(cur_cell) == 0:
            # 現在セルが目標集合に到達済み: ゴール⇔スタートで目標を反転
            # （ゴール到達後の自走帰還・帰還後の次走行再出発を同じ機構で扱う）。
            # E1（研究計画書 §7）: ゴール到達時、最短経路がまだ確定していなければ
            # **追加探索フェーズ**へ入る。確定していれば帰路（悲観的）へ。
            # 帰路の終端（スタート到達）では次走行のためゴールへ向け直す。
            if self.target_mode == "to_goal":
                self.target_mode = "to_start" if self._shortest_confirmed() else "verify"
            elif self.target_mode == "verify":
                self.target_mode = "to_start" if self._shortest_confirmed() else "verify"
            else:
                self.target_mode = "to_goal"
            targets = self._target_cells()
            dist_field = self._flood_fill(targets)

        candidates = []
        for d in DIRS:
            dx, dy = _DELTA[d]
            nx, ny = cur_cell[0] + dx, cur_cell[1] + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                continue
            if not self._connects_known(cur_cell[0], cur_cell[1], nx, ny):
                continue
            nd = dist_field.get((nx, ny))
            if nd is None:
                continue
            candidates.append((d, nd))

        if not candidates:
            # 安全弁（想定外: 既知壁だけで到達可能な隣接セルが無い）。
            # IDLE に留まり 0V を返す→評価器のスタック検出に委ねる。
            self._state = "IDLE"
            return

        min_dist = min(nd for _, nd in candidates)
        best_set = {d for d, nd in candidates if nd == min_dist}
        chosen = next(d for d in self._tiebreak_order(self._heading_dir) if d in best_set)

        self._planned_dir = chosen
        self._planned_next_cell = (cur_cell[0] + _DELTA[chosen][0], cur_cell[1] + _DELTA[chosen][1])

        if chosen == self._heading_dir:
            self._enter_forward(x, y)
        else:
            self._enter_turn(chosen)

    # ------------------------------------------------------------------
    # TURN（超信地旋回）
    # ------------------------------------------------------------------
    def _enter_turn(self, direction: str) -> None:
        self._turn_target_yaw = _HEADING_RAD[direction]
        self._state = "TURN"

    def _do_turn(self, obs: np.ndarray, yaw: float):
        gyro_z = float(obs[self._i_gyro_z])
        yaw_err = _wrap_pi(self._turn_target_yaw - yaw)

        if (abs(yaw_err) < self.turn_done_rad) and (abs(gyro_z) < self.turn_done_gyro):
            self._heading_dir = self._planned_dir
            x, y, _yaw = self._sim.privileged_pose()
            self._enter_forward(x, y)
            return 0.0, 0.0

        omega_cmd = self.kp_turn * yaw_err - self.kd_turn * gyro_z
        omega_cmd = max(-self.turn_omega_limit, min(self.turn_omega_limit, omega_cmd))
        return self._robot_cmd_to_voltage(0.0, omega_cmd, obs)

    # ------------------------------------------------------------------
    # FORWARD（1セル直進）
    # ------------------------------------------------------------------
    def _enter_forward(self, x: float, y: float) -> None:
        nx, ny = self._planned_next_cell
        tx = nx * self.cell_size + self.cell_size / 2
        ty = ny * self.cell_size + self.cell_size / 2
        self._fwd_start = (x, y)
        self._fwd_target = (tx, ty)
        self._fwd_total_dist = math.hypot(tx - x, ty - y)
        self._fwd_v_setpoint = 0.0  # 新しい脚は静止状態から台形プロファイルを開始
        self._state = "FORWARD"

    def _do_forward(self, obs: np.ndarray, x: float, y: float, yaw: float):
        tx, ty = self._fwd_target
        dx, dy = tx - x, ty - y
        dist_remain = math.hypot(dx, dy)  # 目標点までの真のユークリッド距離（符号なし）

        v_forward, _omega_z = self._sim.privileged_velocity()
        if (dist_remain < self.forward_done_dist) and (abs(v_forward) < self.forward_done_speed):
            cx, cy = pos_to_cell(x, y, self.width, self.height, self.cell_size)
            self._do_sense(cx, cy)
            self._state = "PLAN"
            return 0.0, 0.0

        # 操舵: 目標方位（現在の進行方向のセル軸、固定）を維持しつつ、開始点→
        # 目標点を結ぶ経路からの横偏差を PD で補正する。
        #
        # 試行錯誤の経緯（実測で確認済みの不具合と対策。詳細は各版の履歴参照）:
        #   (a) 速度プロファイルの減速区間を「経路方向への符号付き進捗」から
        #       計算し0未満をクリップする方式は、横方向の誤差だけが残った
        #       状態で前進指令が0へ張り付き、回頭だけでは（差動二輪は横に
        #       並進できないため）解消できず永久停止した。
        #   (b) 目標点への実ベアリングを追いかける pure pursuit 型は、目標
        #       近傍でベアリング角が暴れ急旋回で振動した。
        #   (c) 「その場の残り距離から毎ティック sqrt(2a・d) を計算し直す」
        #       速度指令は車輪速度追従の遅れで目標近傍でオーバーシュートする。
        #       オーバーシュートのたびに素直に符号反転で後退させる設計は、
        #       進行方向の符号付き誤差が0近傍で反転を繰り返し、横偏差PDの
        #       符号も一緒に反転して減衰しないリミットサイクルに陥った。
        #   (d) 後退を一切禁止し、横方向の誤差が残る間だけ前進の下限速度
        #       （creep floor）を与える設計は、横偏差がほぼ0で経路方向にだけ
        #       小さく行き過ぎたケースを救えず（前進しても遠ざかるだけなので
        #       creep を発動させない）、その場に永久停止した。
        # 最終的に3つの領域に分けた: ①経路方向にまだ十分手前
        #  （dist_along_signed > 到達しきい値）は通常の加減速プロファイル、
        #  ②経路方向に明確に行き過ぎた（dist_along_signed < -到達しきい値）
        #  場合のみ一定の低速で後退（毎ティック符号反転しないよう帯域を
        #  跨いだときだけ切り替える）、③その中間帯（ほぼ経路方向には
        #  行き着いている）は横偏差が残っていれば一定の低速で前進、
        #  残っていなければ速度0で待機する。
        heading_target = _HEADING_RAD[self._heading_dir]
        heading_err = _wrap_pi(heading_target - yaw)

        sx, sy = self._fwd_start
        path_dx, path_dy = tx - sx, ty - sy
        path_len = math.hypot(path_dx, path_dy)
        if path_len > 1e-9:
            ux, uy = path_dx / path_len, path_dy / path_len
        else:
            ux, uy = math.cos(heading_target), math.sin(heading_target)
        rel_x, rel_y = x - sx, y - sy
        lateral_err = rel_x * (-uy) + rel_y * ux
        progress = rel_x * ux + rel_y * uy  # 経路方向への符号付き進捗
        dist_along_signed = self._fwd_total_dist - progress  # 正=まだ手前, 負=行き過ぎ（符号保持）

        dt = self.control_dt
        a = self.a_forward
        creep_speed = 0.3 * self.forward_done_speed
        band = self.forward_done_dist  # 「ほぼ経路方向に行き着いた」とみなす帯域幅

        if dist_along_signed > band:
            # ① まだ十分手前: 通常の台形加減速プロファイル（レート制限、常に前進）
            v_set = self._fwd_v_setpoint
            stop_dist = (v_set * v_set) / (2.0 * a) if a > 0.0 else 0.0
            accel = a if dist_along_signed > stop_dist + 1e-9 else -a
            v_set = max(0.0, min(self.v_max, v_set + accel * dt))
        elif dist_along_signed < -band:
            # ② 経路方向に明確に行き過ぎた: 一定の低速で後退して戻る
            v_set = -creep_speed
        else:
            # ③ 経路方向にはほぼ行き着いている: 目標点までの残り距離
            #    dist_remain（到達判定と同じ量）がまだしきい値を超えていれば
            #    低速前進、超えていなければ静止（位置判定・速度判定に委ねる）。
            #
            # v1.0 の不具合（stuck 頻発の根本原因）: ここを abs(lateral_err) と
            # forward_done_dist の比較にしていたため、横偏差が僅かに
            # しきい値未満（例: 0.98cm）でも縦偏差との合成距離 dist_remain は
            # しきい値超過（例: 1.03cm）になりうるケースで、creep 不成立
            # （v_set=0固定）かつ到達判定も不成立（dist_remain>=しきい値）の
            # 板挟みに陥り永久停止していた（実測で maze_1004/1008/1017/1019
            # 等の stuck を再現・特定）。到達判定と同じ dist_remain を使うことで
            # 両判定のしきい値が完全に一致し、この境界の死角が原理的に無くなる。
            v_set = creep_speed if dist_remain > self.forward_done_dist else 0.0

        self._fwd_v_setpoint = v_set
        v_profile = v_set

        gyro_z = float(obs[self._i_gyro_z])
        omega_cmd = (self.kp_heading * heading_err
                     - self.kp_lateral * lateral_err
                     - self.kd_heading * gyro_z)
        omega_cmd = max(-self.turn_omega_limit, min(self.turn_omega_limit, omega_cmd))

        return self._robot_cmd_to_voltage(v_profile, omega_cmd, obs)

    # ------------------------------------------------------------------
    # 運動学（差動二輪・点旋回） → 電圧変換
    # ------------------------------------------------------------------
    def _robot_cmd_to_voltage(self, v_cmd: float, omega_cmd: float, obs: np.ndarray):
        """(前進速度指令 v_cmd [m/s], ロボット角速度指令 omega_cmd [rad/s, 正=反時計回り])
        から目標車輪角速度を求め、V = Ke_eff・ω_des + Kp_wheel・(ω_des−ω_act) で
        左右電圧を生成する（±voltage_limit にクリップ）。

        符号規約は実機モデルの実測較正による（本ファイル冒頭 docstring 参照）:
        右車輪 正回転・左車輪 逆回転 で機体 yaw が増加（反時計回り）。
        """
        r = self.wheel_radius
        tread = self.tread
        omega_l_des = v_cmd / r - omega_cmd * tread / (2.0 * r)
        omega_r_des = v_cmd / r + omega_cmd * tread / (2.0 * r)

        omega_l_act = float(obs[self._i_wheel])
        omega_r_act = float(obs[self._i_wheel + 1])

        vl = self.Ke_eff * omega_l_des + self.kp_wheel * (omega_l_des - omega_l_act)
        vr = self.Ke_eff * omega_r_des + self.kp_wheel * (omega_r_des - omega_r_act)

        # 乾性摩擦フィードフォワード補償（bind_sim で導出した friction_ff_v）。
        # 目標車輪角速度が非ゼロ（動かそうとしている）方向にのみ加える。
        if omega_l_des > 1e-6:
            vl += self.friction_ff_v
        elif omega_l_des < -1e-6:
            vl -= self.friction_ff_v
        if omega_r_des > 1e-6:
            vr += self.friction_ff_v
        elif omega_r_des < -1e-6:
            vr -= self.friction_ff_v

        vl = max(-self.voltage_limit, min(self.voltage_limit, vl))
        vr = max(-self.voltage_limit, min(self.voltage_limit, vr))
        return vl, vr

    # ------------------------------------------------------------------
    # 制御則本体
    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray):
        if self._sim is None:
            return 0.0, 0.0

        x, y, yaw = self._sim.privileged_pose()

        if self._state == "IDLE":
            cx, cy = pos_to_cell(x, y, self.width, self.height, self.cell_size)
            self._do_sense(cx, cy)
            self._state = "PLAN"

        if self._state == "PLAN":
            self._do_plan(x, y)
            # _do_plan は同一ティック内で TURN/FORWARD/IDLE(安全弁) のいずれかに遷移する

        if self._state == "TURN":
            return self._do_turn(obs, yaw)
        if self._state == "FORWARD":
            return self._do_forward(obs, x, y, yaw)

        return 0.0, 0.0
