"""
mouse/sim.py
================
MouseSim: 評価器・古典ベースライン・Gymnasium 環境が共通して使う
シミュレーションコア。MuJoCo モデル/データの薄いラッパー。
"""
from pathlib import Path

import numpy as np
import mujoco

from classic.geometry import Rect
from mouse.ir_sensor import (
    IrSensorSpec,
    SurfaceSpec,
    fast_response_or_direct,
    load_table,
    response,
)
from mouse.params import RobotParams


class MouseSim:
    """マイクロマウス v2 の MuJoCo シミュレーションコア。

    - step_control(): 100Hz の 1 制御ステップ（= n_substeps 物理サブステップ）を進める。
      電圧は ZOH（Zero-Order Hold）で n_substeps 保持し data.ctrl に直接書き込む。
      電気項（gainprm/biasprm）+ 機械損失項（joint の damping/frictionloss）は
      MuJoCo の <general> アクチュエータが物理サブステップ毎に自動再計算する
      （docs/MODEL_VERIFICATION_PLAN.md §4.1。Python 側での stale トルク再計算は行わない）。
    - observation(): 学習・評価に使う 14 次元センサ観測。
    - privileged_pose()/privileged_velocity(): 評価器・古典ベースライン専用の特権情報
      （学習エージェントの観測には含めないこと）。
    """

    def __init__(self, xml_path, params=None, noise_std=None, seed=None,
                 ir_sensor_mode="rangefinder", ir_use_table=False, ir_surf=None):
        """
        ir_sensor_mode: "rangefinder"（既定）は MuJoCo 純正 rangefinder の値
            （observation() の ranges 部分と同じ・毎ティック計算しても軽い・後方互換）。
            "physical" は `mouse.ir_sensor` の物理モデル（note_034）で `ir_ranges()`
            が呼ばれた時点で計算する（**戻り値の単位・意味が違う**: rangefinder は
            幾何距離 [m]、physical は AD 変換器が見るであろう生の値＝任意単位の
            光強度。距離に直すかは呼び出し側の判断 — note_034「設計方針」）。
        ir_use_table: "physical" のときだけ効く。False（既定）は `response()`
            （直接積分・厳密）、True は `fast_response_or_direct()`（表を使う高速版。
            実迷路500姿勢×3乱数種で壁の有無の判定の食い違い0/500を確認済みだが
            近似。表の精度到達点は note_034 追記3・4 を参照）。
        ir_surf: "physical" のときの反射面の性質（`SurfaceSpec`）。None なら既定値。
        """
        self.params = params if params is not None else RobotParams()
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # 観測ノイズ用: 専用の乱数生成器を持つ（None なら常にゼロ、フックのみ用意）
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        # --- IR センサの物理モデル（note_034）関連。既定は "rangefinder" で
        # 現行の挙動を一切変えない（研究ノート追記4: 古典探索は必要なときだけ
        # 呼べば直接積分のままで足りると判明したため、表はまだ採用しない）。
        self.ir_sensor_mode = ir_sensor_mode
        self.ir_use_table = ir_use_table
        self.ir_surf = ir_surf if ir_surf is not None else SurfaceSpec()
        self._step_count = 0
        self._ir_ranges_cache = None
        self._ir_ranges_cache_step = -1
        self._ir_sensor_specs = None   # 遅延構築（_ir_sensor_specs_cached() 参照）
        self._wall_rects_cache = None  # 遅延構築（_wall_rects() 参照）
        self._ir_tables_cache = None   # 遅延読み込み（_ir_tables() 参照。表を使う時だけ）

        # --- 名前解決（body/joint id をキャッシュ） ---
        self._mouse_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'mouse')
        if self._mouse_body_id < 0:
            raise ValueError("body 'mouse' が見つかりません。XML を確認してください。")
        self._root_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'root')
        self._left_wheel_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, 'left_wheel_joint')
        self._right_wheel_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, 'right_wheel_joint')

        self._left_wheel_qvel_adr = self.model.jnt_dofadr[self._left_wheel_joint_id]
        # 距離センサ（rangefinder）の本数をモデルから数える。sensordata の並びは
        # 登録順（距離 ×n → accelerometer(3) → gyro(3)）であり、r6 でセンサ本数が
        # 6→4 に変わったため、切り出し位置を固定値で持たない（observation() 参照）。
        self._n_rangefinders = int(np.sum(
            self.model.sensor_type == mujoco.mjtSensor.mjSENS_RANGEFINDER))
        self._right_wheel_qvel_adr = self.model.jnt_dofadr[self._right_wheel_joint_id]

        # ロボット geom 集合（body "mouse" のサブツリー）を起動時に前計算
        self._robot_geom_ids = self._collect_subtree_geoms(self._mouse_body_id)
        # 迷路 geom 集合（壁・柱）を起動時に前計算
        self._maze_geom_ids = self._collect_maze_geoms()

        # 物理サブステップ数 = 制御周期 / 物理タイムステップ（= 10）
        self._n_substeps = int(round(self.params.control_dt / self.params.physics_dt))

        self.reset_to_start()

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------
    def _collect_subtree_geoms(self, root_body_id):
        """root_body_id を根とするボディサブツリーに属する geom の id 集合を返す。"""
        ids = set()
        for g in range(self.model.ngeom):
            bb = self.model.geom_bodyid[g]
            while True:
                if bb == root_body_id:
                    ids.add(g)
                    break
                if bb == 0:  # worldbody に到達（見つからなかった）
                    break
                bb = self.model.body_parentid[bb]
        return ids

    def _collect_maze_geoms(self):
        """迷路壁・柱（v_wall*/h_wall*/post_*）の geom id 集合を返す（衝突判定用。
        赤い壁上端 geom（*_top_*）も含む＝「接触したかどうか」だけを見るので
        二重に数えても問題にならない）。"""
        ids = set()
        prefixes = ('v_wall', 'h_wall', 'post_')
        for g in range(self.model.ngeom):
            name = self.model.geom(g).name
            if name and name.startswith(prefixes):
                ids.add(g)
        return ids

    def _wall_rects(self):
        """真の壁・柱を `Rect`（`classic.geometry.Rect` と同じ形）のリストにする
        （`mouse.ir_sensor.response()`/`fast_response_or_direct()` にそのまま渡せる）。

        赤い壁上端 geom（`*_top_*`）は同じ xy 足跡・高さだけが違う飾りなので除外する
        （`classic.geometry.wall_obstacles` の docstring・`mouse/ir_sensor.py` の
        docstringと同じ扱い。含めると同じ壁の反射を二重に数えてしまう）。
        1区画ぶんに分かれた `v_wall`/`h_wall`/`post_` の geom をそのまま使う
        （継ぎ目での結合は `mouse.ir_sensor` 側の同一平面統合が担うので、ここでは
        1 geom = 1 `Rect` でよい）。

        MJCF は迷路 geom を worldbody 直下に置いている（body 変換を挟まない）ので、
        `geom_pos`/`geom_size` がそのまま world 座標・半長になる（`mouse/mjcf.py`
        参照）。迷路は起動後に変わらないので初回だけ計算してキャッシュする。
        """
        if self._wall_rects_cache is not None:
            return self._wall_rects_cache
        rects = []
        for g in sorted(self._maze_geom_ids):
            name = self.model.geom(g).name
            if '_top_' in name:
                continue
            pos = self.model.geom_pos[g]
            size = self.model.geom_size[g]
            rects.append(Rect(cx=float(pos[0]), cy=float(pos[1]),
                               hx=float(size[0]), hy=float(size[1])))
        self._wall_rects_cache = rects
        return rects

    def _ir_sensor_specs_cached(self):
        """`self.params.sensors`（辞書の列。pos/zaxis は空白区切りの文字列）を
        `IrSensorSpec` の列にする（初回だけ変換してキャッシュする）。"""
        if self._ir_sensor_specs is not None:
            return self._ir_sensor_specs
        specs = []
        for s in self.params.sensors:
            pos = tuple(float(v) for v in s['pos'].split())
            axis = tuple(float(v) for v in s['zaxis'].split())
            specs.append(IrSensorSpec(name=s['name'], pos=pos, axis=axis))
        self._ir_sensor_specs = specs
        return specs

    def _ir_tables(self):
        """`ir_use_table=True` のときだけ使う、壁表・柱表・床基準値を遅延読み込みする
        （`mouse/build_ir_response_table.py` の生成物。使わない構成では
        ファイル I/O をしない）。"""
        if self._ir_tables_cache is not None:
            return self._ir_tables_cache
        data_dir = Path(__file__).resolve().parent / "data"
        wall_table = load_table(data_dir / "ir_response_table.npz")
        post_table = load_table(data_dir / "ir_post_table.npz")
        floor_value = float(wall_table.meta["floor_baseline"])
        self._ir_tables_cache = (wall_table, post_table, floor_value)
        return self._ir_tables_cache

    # ------------------------------------------------------------------
    # 制御ステップ
    # ------------------------------------------------------------------
    def step_control(self, v_left: float, v_right: float) -> dict:
        """100Hz の 1 制御ステップ（n_substeps 物理サブステップ）を実行する。

        v_left, v_right: 電圧指令 [V]。±voltage_limit にクランプして data.ctrl に
        直接書き込み、n_substeps 回の物理サブステップの間 ZOH で保持する
        （トルクへの変換は MuJoCo の <general> アクチュエータがサブステップ毎に行う）。

        Returns:
            dict(collision, max_contact_force, tipped, sim_time)
        """
        p = self.params
        self.data.ctrl[0] = float(np.clip(v_left, -p.voltage_limit, p.voltage_limit))
        self.data.ctrl[1] = float(np.clip(v_right, -p.voltage_limit, p.voltage_limit))

        # 制御ステップ番号（ir_ranges() のキャッシュのキー。tick=1回の step_control()
        # 呼び出しの間は姿勢が変わらないという前提でキャッシュするので、ここでだけ進める
        # （物理サブステップごとには進めない。classic/explorer.py の tick() は obs を
        # 1回だけ取得し、複数箇所の sense_walls() に使い回すことをコードで確認済み）。
        self._step_count += 1

        collision = False
        max_contact_force = 0.0
        force6 = np.zeros(6, dtype=np.float64)

        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

            # 接触判定: ロボット geom と迷路 geom (v_wall*/h_wall*/post_*) の接触法線力
            for i in range(self.data.ncon):
                con = self.data.contact[i]
                g1, g2 = con.geom1, con.geom2
                is_robot_maze_pair = (
                    (g1 in self._robot_geom_ids and g2 in self._maze_geom_ids) or
                    (g2 in self._robot_geom_ids and g1 in self._maze_geom_ids)
                )
                if not is_robot_maze_pair:
                    continue
                mujoco.mj_contactForce(self.model, self.data, i, force6)
                normal_force = abs(float(force6[0]))
                if normal_force > max_contact_force:
                    max_contact_force = normal_force
                if normal_force > self.params.collision_force_threshold:
                    collision = True

        # 転倒判定: 車体 z 軸の世界 z 成分（body の姿勢行列の (2,2) 成分）
        xmat = self.data.xmat[self._mouse_body_id]
        body_z_axis_world_z = float(xmat[8])
        tipped = body_z_axis_world_z < self.params.tipover_zaxis_threshold

        return {
            'collision': bool(collision),
            'max_contact_force': float(max_contact_force),
            'tipped': bool(tipped),
            'sim_time': float(self.data.time),
        }

    # ------------------------------------------------------------------
    # 観測
    # ------------------------------------------------------------------
    def observation(self) -> np.ndarray:
        """観測を返す（次元は距離センサ本数 n に依存し n + 8 次元）:
        [距離 ×n (m), accel x,y,z, gyro x,y,z, omega_wheel_L, omega_wheel_R]

        r6（センサ 6→4 本）以降、距離センサの本数は構成で変わりうるため、
        sensordata の切り出し位置は**モデルから導出**する（ハードコード禁止）。
        """
        sd = self.data.sensordata
        cutoff = self.params.sensor_cutoff
        n = self._n_rangefinders

        ranges = np.array(sd[0:n], dtype=np.float64)
        # 生値 < 0（ヒットなし）は cutoff に置き換え、cutoff でクリップ
        ranges = np.where(ranges < 0, cutoff, ranges)
        ranges = np.clip(ranges, 0.0, cutoff)

        accel = np.array(sd[n:n + 3], dtype=np.float64)
        gyro = np.array(sd[n + 3:n + 6], dtype=np.float64)

        omega_l = float(self.data.qvel[self._left_wheel_qvel_adr])
        omega_r = float(self.data.qvel[self._right_wheel_qvel_adr])

        obs = np.concatenate([ranges, accel, gyro, [omega_l, omega_r]])

        if self.noise_std is not None:
            obs = obs + self.rng.normal(0.0, self.noise_std, size=obs.shape)

        return obs

    def ir_ranges(self) -> np.ndarray:
        """距離センサの読み（n本ぶん）を返す。`ir_sensor_mode`（コンストラクタ引数）で
        中身が変わる:

          - "rangefinder"（既定）: MuJoCo 純正 rangefinder の値（`observation()` の
            ranges 部分と同じ、単位 [m]）。呼ぶたびに計算し直す（キャッシュ不要なほど
            軽い。MuJoCo が物理ステップ毎に自動計算済みの値を読むだけ）。
          - "physical": `mouse.ir_sensor`（note_034）の物理モデルで、**呼ばれた時点で**
            計算する（単位は AD 変換器が見るであろう生の値＝任意単位の光強度で、
            "rangefinder" の距離 [m] とは別物。距離に直すかどうかはこの関数では
            決めない — note_034「設計方針」）。`ir_use_table` で `response()`
            （直接積分・厳密。既定）と `fast_response_or_direct()`（表・高速）を
            切り替える。

        "physical" は同一の制御ステップ（`step_control()` 1回ぶん）の中で複数回
        呼ばれてもキャッシュを使い回す（`_step_count` をキーにする。姿勢は
        `step_control()` の中でしか変わらないので、これで足りる。
        `classic/explorer.py::tick()` が obs を1回だけ取得し、複数箇所の
        `sense_walls()` に使い回していることをソースを読んで確認した上での判断
        — research_notes/note_034 追記5参照）。
        """
        if self.ir_sensor_mode == "rangefinder":
            n = self._n_rangefinders
            sd = self.data.sensordata
            cutoff = self.params.sensor_cutoff
            ranges = np.array(sd[0:n], dtype=np.float64)
            ranges = np.where(ranges < 0, cutoff, ranges)
            return np.clip(ranges, 0.0, cutoff)

        if self._ir_ranges_cache is not None and self._ir_ranges_cache_step == self._step_count:
            return self._ir_ranges_cache

        pose = self.privileged_pose()  # (x, y, yaw) — シミュレータ内部でのみ使う
        surfaces = self._wall_rects()
        specs = self._ir_sensor_specs_cached()
        values = np.empty(len(specs), dtype=np.float64)
        if self.ir_use_table:
            wall_table, post_table, floor_value = self._ir_tables()
            for i, spec in enumerate(specs):
                values[i] = fast_response_or_direct(
                    spec, pose, surfaces, wall_table, floor_value,
                    surf=self.ir_surf, post_table=post_table,
                )
        else:
            for i, spec in enumerate(specs):
                values[i] = response(spec, pose, surfaces, self.ir_surf)

        self._ir_ranges_cache = values
        self._ir_ranges_cache_step = self._step_count
        return values

    # ------------------------------------------------------------------
    # リセット
    # ------------------------------------------------------------------
    def reset_to_start(self, cell=(0, 0), heading_deg: float = 90):
        """フリージョイント qpos をセル中心・指定向きにセットする。
        全 qvel=0、車輪角=0。data.time は保持（走行間の持ち時間連続のため）。"""
        cx, cy = cell
        cell_size = self.params.cell_size
        x = cx * cell_size + cell_size / 2
        y = cy * cell_size + cell_size / 2
        z = 0.002

        heading_rad = np.radians(heading_deg)

        qpos_adr = self.model.jnt_qposadr[self._root_joint_id]
        qvel_adr = self.model.jnt_dofadr[self._root_joint_id]

        self.data.qpos[qpos_adr:qpos_adr + 3] = [x, y, z]
        # z 軸回りの回転のみのクォータニオン [w, x, y, z]
        self.data.qpos[qpos_adr + 3] = np.cos(heading_rad / 2.0)
        self.data.qpos[qpos_adr + 4] = 0.0
        self.data.qpos[qpos_adr + 5] = 0.0
        self.data.qpos[qpos_adr + 6] = np.sin(heading_rad / 2.0)
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

        for jid in (self._left_wheel_joint_id, self._right_wheel_joint_id):
            qp = self.model.jnt_qposadr[jid]
            qv = self.model.jnt_dofadr[jid]
            self.data.qpos[qp] = 0.0
            self.data.qvel[qv] = 0.0

        self.data.ctrl[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        # 姿勢を直接書き換えたので、ir_ranges() のキャッシュ（制御ステップ番号を
        # キーにしている）を無効化する。ここで _step_count 自体は進めない
        # （step_control() を呼んだ回数と紐付ける意味を保つため、キャッシュの
        # ステップ番号だけを「絶対に一致しない」値に倒す）。
        self._ir_ranges_cache = None
        self._ir_ranges_cache_step = -1

    def full_reset(self, cell=(0, 0), heading_deg: float = 90):
        """完全初期化: mj_resetData 後に reset_to_start、time=0。"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.time = 0.0
        self.reset_to_start(cell=cell, heading_deg=heading_deg)

    # ------------------------------------------------------------------
    # 特権情報（評価器・古典ベースライン専用。学習エージェントの観測には使わないこと）
    # ------------------------------------------------------------------
    def privileged_pose(self):
        """(x, y, yaw) を返す。yaw は世界座標系での機体正面方向 [rad]。"""
        qpos_adr = self.model.jnt_qposadr[self._root_joint_id]
        x = float(self.data.qpos[qpos_adr])
        y = float(self.data.qpos[qpos_adr + 1])
        w, qx, qy, qz = self.data.qpos[qpos_adr + 3:qpos_adr + 7]
        yaw = float(np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
        return x, y, yaw

    def privileged_velocity(self):
        """(v_forward, omega_z) を返す。
        v_forward: 機体正面方向（ローカル +x 軸）への世界座標系速度の射影 [m/s]
        omega_z: 世界座標系での角速度 z 成分 [rad/s]
        """
        res = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self._mouse_body_id, res, 0
        )
        omega_world = res[0:3]
        v_world = res[3:6]

        xmat = self.data.xmat[self._mouse_body_id].reshape(3, 3)
        forward_axis_world = xmat[:, 0]  # ボディローカル x 軸を世界座標系で表したもの

        v_forward = float(np.dot(v_world, forward_axis_world))
        omega_z = float(omega_world[2])
        return v_forward, omega_z

    @property
    def sim_time(self) -> float:
        return float(self.data.time)
