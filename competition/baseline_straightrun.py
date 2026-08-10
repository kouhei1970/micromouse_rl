"""
competition/baseline_straightrun.py
================
古典ベースライン方策 StraightRunPolicy（L0-b: 超信地旋回走行・直進連続）。

docs/L0PLUS_DESIGN.md §1・§7 の設計。`competition/baseline_classical.py`
（L0-a = AdachiPolicy, 超信地旋回走行・区画ごと停止）を丸ごとコピーし、
**走行部のみ**を変更したもの。足立法（flood-fill）の経路決定・地図保持・
壁観測・帰還ロジックは L0-a と完全に同一（本ファイルでも同じ実装をそのまま
複製している。差分は「直進区間で区画ごとに停止するか否か」の 1 点のみ）。

--------------------------------------------------------------------------
L0-a との差分（§7）
--------------------------------------------------------------------------
- L0-a: 1 区画進むたびに停止し、直進でも一旦速度 0 まで減速してから次の
  区画へ台形速度プロファイルで加速し直す。
- L0-b: 現在の経路上で「次に曲がるまでに連続する直進区画数 n」を先読みし、
  距離 n×cell_size に対する 1 本の台形速度プロファイル（加速度上限
  a_forward、上限速度 v_max、終端速度 0）を組む。**区画境界では減速せず
  走り抜け**、曲がる手前でのみ停止して超信地旋回する（TURN の PD 制御・
  完了判定は L0-a と同一）。

先読み（_lookahead_straight_cells）は 2 つの独立した規則を組み合わせる:
  (a) 経路決定そのもの（どの方向に進むか）は _select_direction が
      _connects_known の楽観的規則（未知壁=壁なし）で判定する。これは
      L0-a の _do_plan の候補選定ロジックと完全に同一（重複を避けるため
      両者から共通に呼ぶ）。
  (b) 「区画境界を減速せずに通過してよいか」は _known_open で判定し、
      未知壁(-1)は保守的に「壁あり」として扱う（拡張を打ち切る）。
      未知壁のある区画は、まだ物理的に確認されていない以上、高速で
      走り抜ける根拠が無いため。
  この 2 つを分けることで、「経路探索は楽観的（未知は壁なし）」という
  L0-a の性質を保ったまま、「高速巡航してよい区間だけは既知壁に限る」
  という §7 の要件を満たす。

--------------------------------------------------------------------------
速度上限の物理的根拠（教授裁定 2026-08-10・追加指示）
--------------------------------------------------------------------------
直進区間の速度上限 v_max は、単に「L0-a より速くしてよい」という設計自由度
だけでなく、**壁を検出してから停止できる速度**を上回ってはならない
（実機の距離センサに基づく現実的な上限）:

    v <= sqrt(2 * a_forward * d_detect)

- a_forward: 減速度上限 [m/s^2]（§7 既定 3.4）
- d_detect: 壁を検出してから停止までに使える距離 [m]

d_detect の見積り（bind_sim() で sim.params から導出。ハードコード禁止の
方針に従う）:
  - 距離センサの検出範囲 sensor_cutoff（既定 0.3m。mouse/params.py）
  - 区画境界の壁までの幾何距離 = cell_size（既定 0.18m）。センサは前方
    正面だけでなく側方寄り（LS/RS は zaxis に大きな横成分を持つ）も
    含むため、正面の壁を検出できる実効距離は sensor_cutoff より短くなり
    うる。安全側に立ち、**幾何距離と検出範囲の小さい方**を生の d_detect
    とする: d_detect_raw = min(sensor_cutoff, cell_size)
  - その上でさらに安全率 detect_margin（既定 0.6。docs/L0PLUS_DESIGN.md
    §2.2 の全体安全率 0.6 と同じ考え方で統一）を掛ける:
    d_detect = detect_margin * d_detect_raw

既定値 (sensor_cutoff=0.3, cell_size=0.18, detect_margin=0.6) では:
  d_detect_raw = min(0.3, 0.18) = 0.18 m
  d_detect     = 0.6 * 0.18 = 0.108 m
  v_cap        = sqrt(2 * 3.4 * 0.108) = sqrt(0.7344) ≈ 0.857 m/s

コンストラクタ引数 v_max は「要求値」であり、bind_sim() 時にこの物理上限
v_cap でクリップされる（要求値がすでに上限未満ならそのまま）。

**既定値 v_max=0.6 m/s は、この物理上限 0.857 m/s よりさらに保守的な整定値**
（validation seed 4000-4002 での実測）。

> 【2026-08-10 是正】以下の「stuck が頻発する」記述は **kp_heading=3.0 を
> 流用していた当時のもの**である。原因は下記のとおり「L0-a の低速向けゲインを
> 速度2倍のまま使っていたこと」であり、kp_heading を速度に比例させて 6.0 に
> 上げたことで、検証帯20面・100走行で**衝突0・スタック0**になった
> （是正前は同条件でスタック7件）。v_max を物理上限まで詰められるかは未検証で、
> 既定値は 0.6 のままにしてある（速度上限の再設定は別途の課題）。

L0-a から流用した横偏差・方位PDの
ゲイン（kp_heading/kp_lateral/kd_heading）は L0-a の低速（v_max=0.3）向けに
調整されたものであり、v_max を物理上限近く（0.7〜0.8 m/s）まで上げると、
1区画の急な台形加減速の終端で「方位偏差を戻すトルク」と「横偏差を戻す
トルク」がほぼ相殺して機体が非ゼロの向きのまま静止する準停留点に陥り、
低速クリープ（regime③、本ファイル _do_forward docstring 末尾参照）が
収束しない stuck が頻発することを実測で確認した。v_max=0.6 まで下げる
ことでこの頻度は大きく下がるが完全には無くならないため、併せて
forward_done_dist（到達判定の位置許容誤差）も L0-a の既定 0.01m から
0.02m へ緩めた（§7 で高速化した分、台形減速の終端に残る位置残差が
L0-aの低速時より大きくなるための整定。区画180mm・機体トレッド72mmに
対し20mmの許容は壁余裕の観点でも妥当）。PD ゲイン自体の再設計（例:
Stanley法本来の速度依存ゲインスケジューリング）はスラローム走行(L0-c)
以降の課題とし、本ファイルでは L0-a のPD構造をそのまま流用しつつ
上記2点のみを整定する方針をとった。

--------------------------------------------------------------------------
その他（L0-a と同一。詳細は competition/baseline_classical.py docstring 参照）
--------------------------------------------------------------------------
- 足立法（flood-fill）: 壁地図・未知壁の楽観的扱い・タイブレーク規則は
  L0-a と完全に同一。
- TURN（超信地旋回）: PD 制御・完了判定 |yaw誤差|<2° かつ |gyro_z|<0.2rad/s
  は L0-a と完全に同一。
- 直進中の姿勢維持（横偏差・方位の PD 補正）: L0-a の _do_forward をそのまま
  流用（総距離が 1 区画から n 区画に変わるだけで、ロジックは無変更）。
- 電圧生成・運動学: L0-a と完全に同一。
- 位置推定は特権情報（privileged_pose）を使用（requires_privileged=True）。
"""
import math
from collections import deque

import numpy as np

from competition.evaluator import goal_cells, pos_to_cell
from competition.policy_interface import MousePolicy

# 4 方位。北=+y, 東=+x, 南=-y, 西=-x（evaluator.py の v_walls/h_walls 座標規約と
# heading_deg=90(北) の初期姿勢に整合）。L0-a（baseline_classical.py）と同一。
DIRS = ("N", "E", "S", "W")
_DELTA = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
_HEADING_RAD = {"N": math.pi / 2, "E": 0.0, "S": -math.pi / 2, "W": math.pi}
# 時計回りの次方位（タイブレーク規則: 直進優先→前回進行方向から時計回り優先 で使用）
_CW_NEXT = {"N": "E", "E": "S", "S": "W", "W": "N"}


def _wrap_pi(angle: float) -> float:
    """角度を [-pi, pi) にラップする。"""
    return math.atan2(math.sin(angle), math.cos(angle))


class StraightRunPolicy(MousePolicy):
    """足立法（flood-fill）+ 超信地旋回走行・直進連続による古典ベースライン（L0-b）。

    docs/L0PLUS_DESIGN.md §7。足立法の経路決定・地図保持・壁観測・帰還ロジックは
    L0-a（competition/baseline_classical.py の AdachiPolicy）と完全に同一。
    走行部のみ、直進区間の先読みにより区画境界で停止せず走り抜けるよう変更した。

    特権情報（bind_sim/bind_maze）を使用するため requires_privileged=True。
    """

    name = "l0b_straightrun"
    requires_privileged = True

    def __init__(self,
                 v_max: float = 0.6, a_forward: float = 3.4, detect_margin: float = 0.6,
                 kp_turn: float = 8.0, kd_turn: float = 0.6, turn_omega_limit: float = 10.0,
                 # 【2026-08-10 是正】kp_heading 3.0 -> 6.0。
                 # 3.0 は L0-a の v_max=0.3 m/s 向けに調整された値をそのまま流用した
                 # ものだったが、L0-b は v_max=0.6 m/s（2倍）で走る。方位偏差 e_psi が
                 # 横位置に効く速さは v*e_psi なので、**同じ空間分解能で誤差を潰すには
                 # 速度に比例した方位ゲインが要る**。速度が2倍なら kp_heading も2倍。
                 # 検証帯 seed 4000-4019（20面・100走行）での実測:
                 #   kp_heading= 3.0: 衝突0 / **スタック7** / 失敗率 7.0% / 探索中央 27.22s
                 #   kp_heading= 6.0: 衝突0 /   スタック0   / 失敗率 0.0% / 探索中央 23.18s
                 # kp_lateral・kd_heading はほぼ効かない（5面で 2.0〜8.0 / 0.35〜0.70 の
                 # どれでも失敗率0%・タイム差 0.1s 未満）。感度はこの1つのゲインに集中する。
                 # 「方式ごとに最適値が違う」のではなく「動作点（速度）が違えば必要な
                 # ゲインが違う」。詳細は research_notes/note_007（層(2)）。
                 kp_heading: float = 6.0, kp_lateral: float = 8.0, kd_heading: float = 0.35,
                 kp_wheel: float = 0.05,
                 turn_done_deg: float = 2.0, turn_done_gyro: float = 0.2,
                 forward_done_dist: float = 0.02, forward_done_speed: float = 0.05,
                 regress_margin: float = 3.0):
        # --- 走行プロファイル（§7: a_forward 既定 3.4 m/s^2。v_max は要求値で
        #     あり、bind_sim() で物理上限（壁検出→停止の制約）にクリップされる。
        #     既定 0.6 m/s・forward_done_dist 既定 0.02m の整定根拠は本ファイル
        #     冒頭の docstring「速度上限の物理的根拠」節末尾を参照） ---
        self.v_max = v_max
        self.a_forward = a_forward
        self.detect_margin = detect_margin
        # --- TURN PD ゲイン（L0-a と同一既定値） ---
        self.kp_turn = kp_turn
        self.kd_turn = kd_turn
        self.turn_omega_limit = turn_omega_limit  # 目標ロボット角速度の上限 [rad/s]
        # --- FORWARD PD ゲイン（固定セル軸方位の維持 + 経路横偏差の補正。L0-aと同一既定値） ---
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
        # 後退クリープ（regime②）に入るヒステリシス倍率。L0-a の _do_forward を
        # そのまま流用した後、validation 迷路での実機検証（本ファイル §7
        # 実装時、seed 4000）で発見したチャタリング（リミットサイクル）の
        # 整定用パラメータ。詳細は _do_forward の docstring 末尾を参照。
        self.regress_margin = regress_margin

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
        # 速度上限の物理的根拠（壁検出→停止）。bind_sim() で sim.params から導出。
        self._d_detect = None
        self._v_max_physical_cap = None

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
        self._fwd_n_cells = None    # 直進で走り抜ける区画数（§7 先読み。診断・テスト用）

    # ------------------------------------------------------------------
    # 特権情報バインド（policy_interface.py の契約）
    # ------------------------------------------------------------------
    def bind_sim(self, sim) -> None:
        self._sim = sim
        p = sim.params
        # 観測ベクトルの並びは [距離 ×n, accel(3), gyro(3), 車輪角速度(2)] であり、
        # 距離センサ本数 n は構成で変わる。固定インデックスを持つと構成変更で
        # 静かに壊れるため、本数から導出する（L0-a と同一方式）。
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
        # 乾性摩擦（frictionloss）のフィードフォワード補償電圧（L0-a と同一導出）。
        self.friction_ff_v = 1.3 * p.wheel_frictionloss / p.gainprm0

        # --- 直進区間の速度上限の物理的根拠（教授裁定 2026-08-10）。
        # 「壁を検出してから停止できる速度」v <= sqrt(2*a_forward*d_detect) を
        # 上回らないよう、コンストラクタ要求値 v_max をここでクリップする。
        # d_detect は sensor_cutoff（検出範囲）と cell_size（区画境界の壁までの
        # 幾何距離）の小さい方に安全率 detect_margin を掛けたもの（本ファイル
        # 冒頭 docstring 参照。数値はハードコードせず sim.params から導出）。
        d_detect_raw = min(p.sensor_cutoff, p.cell_size)
        self._d_detect = self.detect_margin * d_detect_raw
        self._v_max_physical_cap = math.sqrt(2.0 * self.a_forward * self._d_detect)
        self.v_max = min(self.v_max, self._v_max_physical_cap)

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
    # 足立法（flood-fill）— L0-a と完全に同一
    # ------------------------------------------------------------------
    def _target_cells(self):
        if self.target_mode == "to_goal":
            return goal_cells(self.width, self.height)
        return [(0, 0)]

    def _wall_value(self, x: int, y: int, nx: int, ny: int) -> int:
        """既知壁配列で (x,y)-(nx,ny) 間の壁値を返す（0=壁なし, 1=壁あり, -1=未知）。"""
        if (nx, ny) == (x + 1, y):
            return int(self.v_walls_known[x + 1, y])
        if (nx, ny) == (x - 1, y):
            return int(self.v_walls_known[x, y])
        if (nx, ny) == (x, y + 1):
            return int(self.h_walls_known[x, y + 1])
        if (nx, ny) == (x, y - 1):
            return int(self.h_walls_known[x, y])
        raise ValueError(f"({x},{y}) と ({nx},{ny}) は隣接していません")

    def _connects_known(self, x: int, y: int, nx: int, ny: int) -> bool:
        """既知壁配列で (x,y)-(nx,ny) が通行可能か。未知(-1)は楽観的に通行可能扱い
        （足立法の経路決定。L0-a と同一ロジック）。"""
        return self._wall_value(x, y, nx, ny) != 1

    def _known_open(self, x: int, y: int, nx: int, ny: int) -> bool:
        """既知壁配列で (x,y)-(nx,ny) が「確認済みで開放」か。未知(-1)は保守的に
        不可扱い（§7: 直進連続の先読み専用。経路決定そのものの楽観的規則
        _connects_known は変更しない）。"""
        return self._wall_value(x, y, nx, ny) == 0

    def _flood_fill(self, targets) -> dict:
        """targets からの multi-source BFS 距離場（未知壁=壁なし扱い）。L0-a と同一。"""
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
        """決定的タイブレーク順: 直進優先 → 前回進行方向から時計回り優先。L0-a と同一。"""
        order = [prev_dir]
        d = prev_dir
        for _ in range(3):
            d = _CW_NEXT[d]
            order.append(d)
        return order

    def _select_direction(self, cell, prev_dir: str, dist_field: dict):
        """cell における flood-fill 候補選定（L0-a の _do_plan と同一ロジックを
        切り出したもの。_do_plan と §7 の先読み _lookahead_straight_cells の
        両方から呼ぶ）。候補が無ければ None。"""
        cx, cy = cell
        candidates = []
        for d in DIRS:
            dx, dy = _DELTA[d]
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                continue
            if not self._connects_known(cx, cy, nx, ny):
                continue
            nd = dist_field.get((nx, ny))
            if nd is None:
                continue
            candidates.append((d, nd))
        if not candidates:
            return None
        min_dist = min(nd for _, nd in candidates)
        best_set = {d for d, nd in candidates if nd == min_dist}
        return next(d for d in self._tiebreak_order(prev_dir) if d in best_set)

    # ------------------------------------------------------------------
    # SENSE — L0-a と完全に同一
    # ------------------------------------------------------------------
    def _do_sense(self, cx: int, cy: int) -> None:
        """現在セル (cx,cy) の4方向の真の壁を読み、既知地図に書き込む。"""
        self.v_walls_known[cx + 1, cy] = self._true_v[cx + 1, cy]
        self.v_walls_known[cx, cy] = self._true_v[cx, cy]
        self.h_walls_known[cx, cy + 1] = self._true_h[cx, cy + 1]
        self.h_walls_known[cx, cy] = self._true_h[cx, cy]

    # ------------------------------------------------------------------
    # PLAN — 経路決定は L0-a と同一。直進確定時に呼ぶ先が
    # _enter_forward(1区画) から _start_straight_run(n区画先読み) に変わる。
    # ------------------------------------------------------------------
    def _do_plan(self, x: float, y: float) -> None:
        cur_cell = pos_to_cell(x, y, self.width, self.height, self.cell_size)
        targets = self._target_cells()
        dist_field = self._flood_fill(targets)

        if dist_field.get(cur_cell) == 0:
            # 現在セルが目標集合に到達済み: ゴール⇔スタートで目標を反転
            # （ゴール到達後の自走帰還・帰還後の次走行再出発を同じ機構で扱う）。
            self.target_mode = "to_start" if self.target_mode == "to_goal" else "to_goal"
            targets = self._target_cells()
            dist_field = self._flood_fill(targets)

        chosen = self._select_direction(cur_cell, self._heading_dir, dist_field)

        if chosen is None:
            # 安全弁（想定外: 既知壁だけで到達可能な隣接セルが無い）。
            # IDLE に留まり 0V を返す→評価器のスタック検出に委ねる。
            self._state = "IDLE"
            return

        self._planned_dir = chosen
        self._planned_next_cell = (cur_cell[0] + _DELTA[chosen][0], cur_cell[1] + _DELTA[chosen][1])

        if chosen == self._heading_dir:
            self._start_straight_run(x, y)
        else:
            self._enter_turn(chosen)

    # ------------------------------------------------------------------
    # TURN（超信地旋回）— L0-a と完全に同一。完了後の遷移先のみ
    # _enter_forward → _start_straight_run に変更（§7）。
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
            self._start_straight_run(x, y)
            return 0.0, 0.0

        omega_cmd = self.kp_turn * yaw_err - self.kd_turn * gyro_z
        omega_cmd = max(-self.turn_omega_limit, min(self.turn_omega_limit, omega_cmd))
        return self._robot_cmd_to_voltage(0.0, omega_cmd, obs)

    # ------------------------------------------------------------------
    # §7 直進区間の先読み（L0-a との唯一の実質差分）
    # ------------------------------------------------------------------
    def _lookahead_straight_cells(self, next_cell, heading_dir: str, dist_field: dict) -> int:
        """next_cell（flood-fill で直進が確定した最初の区画）から heading_dir
        方向へ、次に曲がるまで（または目標セル集合に到達するまで）連続する
        直進区画数 n（next_cell を含む、>=1）を返す。

        2つの独立した判定を毎ステップ組み合わせる:
          - この先も flood-fill が直進を選び続けるか（_select_direction。
            未知壁=壁なしの楽観的規則。経路決定そのものは変更しない）
          - 区画境界を減速せず通過してよいか（_known_open。未知壁=壁ありの
            保守的規則。§7「先読みは既知範囲に限る」の要件はここで効く）
        どちらかが不成立になった時点で打ち切る。
        """
        n = 1
        cx, cy = next_cell
        d = heading_dir
        while True:
            if dist_field.get((cx, cy)) == 0:
                break  # 目標セル集合に到達: ここで打ち切り（目標反転はしない。
                       # 状態機械は通常どおり PLAN で処理する）
            nxt_dir = self._select_direction((cx, cy), d, dist_field)
            if nxt_dir != d:
                break  # この先で曲がる（または経路が確定できない）

            dx, dy = _DELTA[d]
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                break
            if not self._known_open(cx, cy, nx, ny):
                break  # 未知/壁ありは保守的に拡張しない
            n += 1
            cx, cy = nx, ny
        return n

    def _start_straight_run(self, x: float, y: float) -> None:
        """flood-fill で直進が確定した直後（_do_plan の直進分岐、または
        _do_turn 完了直後）に呼ぶ。次に曲がるまでの連続直進区画数 n を
        先読みし、区画境界で止まらず n 区画先まで一括で走る（§7）。
        L0-a（_enter_forward を1区画先に対して直接呼ぶだけ）との唯一の差分。
        """
        dist_field = self._flood_fill(self._target_cells())
        n = self._lookahead_straight_cells(self._planned_next_cell, self._heading_dir, dist_field)
        dx, dy = _DELTA[self._heading_dir]
        nx0, ny0 = self._planned_next_cell
        final_cell = (nx0 + dx * (n - 1), ny0 + dy * (n - 1))
        self._fwd_n_cells = n
        self._enter_forward(x, y, final_cell)

    # ------------------------------------------------------------------
    # FORWARD（直進区間: 1〜n セル）
    # ------------------------------------------------------------------
    def _enter_forward(self, x: float, y: float, target_cell) -> None:
        """target_cell（直進区間の最終区画。§7の先読みで決まる。n=1ならL0-aと
        同じ「次の1区画」）の中心を目標点として直進区間を開始する。"""
        tx = target_cell[0] * self.cell_size + self.cell_size / 2
        ty = target_cell[1] * self.cell_size + self.cell_size / 2
        self._fwd_start = (x, y)
        self._fwd_target = (tx, ty)
        self._fwd_total_dist = math.hypot(tx - x, ty - y)
        self._fwd_v_setpoint = 0.0  # 新しい脚は静止状態から台形プロファイルを開始
        self._state = "FORWARD"

    def _do_forward(self, obs: np.ndarray, x: float, y: float, yaw: float):
        """直進区間（1〜n セル、§7の先読みで決まる distance = n×cell_size）の
        台形速度プロファイル + 姿勢維持。ロジックは L0-a の _do_forward と
        完全に同一（総距離が 1 区画固定から n 区画可変になっただけで、
        _fwd_start/_fwd_target/_fwd_total_dist を使う計算はそのまま通用する）。

        操舵: 目標方位（現在の進行方向のセル軸、固定）を維持しつつ、開始点→
        目標点を結ぶ経路からの横偏差を PD で補正する。

        試行錯誤の経緯（L0-a で実測確認済みの不具合と対策。詳細は
        competition/baseline_classical.py の同名メソッド docstring 参照）:
          (a) 速度プロファイルの減速区間を「経路方向への符号付き進捗」から
              計算し0未満をクリップする方式は、横方向の誤差だけが残った
              状態で前進指令が0へ張り付き、回頭だけでは（差動二輪は横に
              並進できないため）解消できず永久停止した。
          (b) 目標点への実ベアリングを追いかける pure pursuit 型は、目標
              近傍でベアリング角が暴れ急旋回で振動した。
          (c) 「その場の残り距離から毎ティック sqrt(2a・d) を計算し直す」
              速度指令は車輪速度追従の遅れで目標近傍でオーバーシュートする。
              オーバーシュートのたびに素直に符号反転で後退させる設計は、
              進行方向の符号付き誤差が0近傍で反転を繰り返し、横偏差PDの
              符号も一緒に反転して減衰しないリミットサイクルに陥った。
          (d) 後退を一切禁止し、横方向の誤差が残る間だけ前進の下限速度
              （creep floor）を与える設計は、横偏差がほぼ0で経路方向にだけ
              小さく行き過ぎたケースを救えず（前進しても遠ざかるだけなので
              creep を発動させない）、その場に永久停止した。
        最終的に3つの領域に分けた: ①経路方向にまだ十分手前
         （dist_along_signed > 到達しきい値）は通常の加減速プロファイル、
         ②経路方向に明確に行き過ぎた（dist_along_signed < -後退しきい値）
         場合のみ一定の低速で後退、③その中間帯（ほぼ経路方向には
         行き着いている）は横偏差が残っていれば一定の低速で前進、
         残っていなければ速度0で待機する。

        追加の整定（L0-b固有。§7実装時にvalidation迷路 seed 4000 で発見）:
        ②の後退しきい値に、①③の帯域幅 band とは別のヒステリシス
        regress_band = regress_margin * band（既定 regress_margin=3.0）を
        使う。band と同じ値だと、横偏差 PD の効きで機体がわずかに
        横へ流れているだけの状況でも dist_along_signed がちょうど
        -band 付近に張り付き、②(後退)⇔③(前進)を数ティックごとに
        往復する減衰しないリミットサイクルに陥ることを実測で確認した
        （回転慣性により、後退指令に切り替えても速度反転が数ティック
        遅れて追随するため、②に入った直後もわずかに前進し続けて
        band を再び超えてしまい、往復が終わらない）。② の入口だけ
        band の3倍まで許容範囲を広げることで、この程度の横偏差起因の
        小さな行き過ぎは③(前進クリープ。dist_remainのみで判定する
        単調な収束)に留め、②は明確な大きい行き過ぎのときだけ発動する
        ようにした。L0-a の3領域構成・各領域内の計算式そのものは
        変更していない（後退の発動しきい値のみの整定）。
        """
        tx, ty = self._fwd_target
        dx, dy = tx - x, ty - y
        dist_remain = math.hypot(dx, dy)  # 目標点までの真のユークリッド距離（符号なし）

        v_forward, _omega_z = self._sim.privileged_velocity()
        if (dist_remain < self.forward_done_dist) and (abs(v_forward) < self.forward_done_speed):
            cx, cy = pos_to_cell(x, y, self.width, self.height, self.cell_size)
            self._do_sense(cx, cy)
            self._state = "PLAN"
            return 0.0, 0.0

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
        regress_band = self.regress_margin * band  # ②(後退)発動のヒステリシス（上のdocstring参照）

        if dist_along_signed > band:
            # ① まだ十分手前: 通常の台形加減速プロファイル（レート制限、常に前進）
            v_set = self._fwd_v_setpoint
            stop_dist = (v_set * v_set) / (2.0 * a) if a > 0.0 else 0.0
            accel = a if dist_along_signed > stop_dist + 1e-9 else -a
            v_set = max(0.0, min(self.v_max, v_set + accel * dt))
        elif dist_along_signed < -regress_band:
            # ② 経路方向に明確に行き過ぎた: 一定の低速で後退して戻る
            v_set = -creep_speed
        else:
            # ③ 経路方向にはほぼ行き着いている: 目標点までの残り距離
            #    dist_remain（到達判定と同じ量）がまだしきい値を超えていれば
            #    低速前進、超えていなければ静止（位置判定・速度判定に委ねる）。
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
    # 運動学（差動二輪・点旋回） → 電圧変換 — L0-a と完全に同一
    # ------------------------------------------------------------------
    def _robot_cmd_to_voltage(self, v_cmd: float, omega_cmd: float, obs: np.ndarray):
        """(前進速度指令 v_cmd [m/s], ロボット角速度指令 omega_cmd [rad/s, 正=反時計回り])
        から目標車輪角速度を求め、V = Ke_eff・ω_des + Kp_wheel・(ω_des−ω_act) で
        左右電圧を生成する（±voltage_limit にクリップ）。

        符号規約は実機モデルの実測較正による（L0-a と同一。
        competition/baseline_classical.py 冒頭 docstring 参照）:
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
    # 制御則本体 — L0-a と完全に同一
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
