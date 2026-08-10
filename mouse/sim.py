"""
mouse/sim.py
================
MouseSim: 評価器・古典ベースライン・Gymnasium 環境が共通して使う
シミュレーションコア。MuJoCo モデル/データの薄いラッパー。
"""
import numpy as np
import mujoco

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

    def __init__(self, xml_path, params=None, noise_std=None, seed=None):
        self.params = params if params is not None else RobotParams()
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # 観測ノイズ用: 専用の乱数生成器を持つ（None なら常にゼロ、フックのみ用意）
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

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
        """迷路壁・柱（v_wall*/h_wall*/post_*）の geom id 集合を返す。"""
        ids = set()
        prefixes = ('v_wall', 'h_wall', 'post_')
        for g in range(self.model.ngeom):
            name = self.model.geom(g).name
            if name and name.startswith(prefixes):
                ids.add(g)
        return ids

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
        """14 次元観測を返す:
        [LF, LS, RF, RS, FL, FR (m), accel x,y,z, gyro x,y,z, omega_wheel_L, omega_wheel_R]
        """
        sd = self.data.sensordata
        cutoff = self.params.sensor_cutoff

        ranges = np.array(sd[0:6], dtype=np.float64)
        # 生値 < 0（ヒットなし）は cutoff に置き換え、cutoff でクリップ
        ranges = np.where(ranges < 0, cutoff, ranges)
        ranges = np.clip(ranges, 0.0, cutoff)

        accel = np.array(sd[6:9], dtype=np.float64)
        gyro = np.array(sd[9:12], dtype=np.float64)

        omega_l = float(self.data.qvel[self._left_wheel_qvel_adr])
        omega_r = float(self.data.qvel[self._right_wheel_qvel_adr])

        obs = np.concatenate([ranges, accel, gyro, [omega_l, omega_r]])

        if self.noise_std is not None:
            obs = obs + self.rng.normal(0.0, self.noise_std, size=obs.shape)

        return obs

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
