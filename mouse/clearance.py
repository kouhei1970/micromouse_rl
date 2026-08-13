"""
mouse/clearance.py
==================
**最小壁余裕**（機体外形と最も近い壁との距離）と**高周波エネルギー比**の計算。
廊下（`CorridorEnv`）と 6x6 迷路（`Maze6Env`）の両方で使う共通実装。

## なぜ横偏差ではなく最小壁余裕か（2026-08-11 教授指摘）

横偏差（経路中心線からの距離）の**最大値は、ほぼ確実に「壁が無い場所」で記録される**。
交差点・分岐・開口部では側壁が無いので、中心線から大きく外れても何にも当たらない。
つまり横偏差の最大は「**最も安全な場所での逸脱量**」を拾っており、安全性の指標に
なっていない（実際、壁接触なしで完走しながら横偏差 54.6 mm に達する走行が実在した）。

**最小壁余裕は安全余裕そのもの**で、壁が無い場所は自動的に候補から外れる。
「接触水準 54 mm」という定数も不要になる（0 に近づくほど危ない、で読める）。
なおその 54 mm はシャーシ半幅から計算された誤りである。
**機体の最外部はコーナー片（mein_body2..5）で半幅 40.0 mm** なので
**真の接触水準は 44.0 mm**（通路半幅 84 mm − 40.0 mm）。
🔴 2026-08-13 是正: 従来ここに「最外部は車輪の外面 39.5 mm ／ 接触水準 44.5 mm」と
書いていたのは誤り（裁定 R25・R26、ROBOT_SPEC §2.1）。

## なぜ `mj_geomDistance` を使わないか

**mujoco 3.11 の `mj_geomDistance` は本モデルで誤った値を返す**（2026-08-11 実測）。
毎ステップ 0.0 mm と 38 mm を交互に返し、0.0 が返るときの相手の壁は実距離 125 mm
離れていた。メッシュ geom だけでなく円柱でも起きる。代わりに幾何計算で求める:

1. 機体の各プリミティブ geom の表面を**代表点**で標本化する
2. 毎ステップ、代表点を `geom_xpos` / `geom_xmat` で世界座標へ移す
3. 各壁（箱）について、点を壁のローカル座標へ移して
   `dist = ‖max(|p_local| − halfsize, 0)‖` を厳密に計算する

**メッシュ geom は除外する。**外殻の薄板は機体座標で ±35.5 × ±28.7 mm にあり、
シャーシ箱（半サイズ 50 × 30 mm）の内側なので xy の外形に寄与しない。

## 高周波エネルギー比

符号反転数は**振幅を見ていない**（平均 0 の対称方策では反転確率がちょうど 1/2 で
σ に依存せず、理論値 50 回/s）。害の実体は $I^2R$ とギヤの打撃で、**振幅の 2 乗**に
比例する。そこで車輪が追従できない帯域の指令成分のエネルギー比を測る:

    ā_t = α·ā_(t−1) + (1 − α)·a_t,   α = exp(−Δt/τ_wheel) = 0.616
    HF比 = sqrt( E‖a_t − ā_t‖² / 2 ) / 1.0        （無次元。全振幅比）

理論最悪値（全振幅で Nyquist 交番する方形波）は 2α/(1+α) = 0.762。
"""
import math

import mujoco
import numpy as np

from mouse.params import RobotParams

PREFILTER_M = 0.30      # この距離より遠い壁は最小値を与えない
WALL_PREFIXES = ("h_wall", "v_wall", "post")


def wheel_tau(p: RobotParams = None) -> float:
    """車輪ジョイントの時定数 [s]（docs/MODEL_VERIFICATION_PLAN.md §4.2 の量から）。"""
    p = p or RobotParams()
    armature = p.gear_ratio ** 2 * p.rotor_inertia
    I_w = 0.5 * p.mass_wheel * p.wheel_radius ** 2
    b_elec = p.gear_ratio ** 2 * p.motor_Kt * p.motor_Ke / p.motor_R
    return (I_w + armature) / (b_elec + p.wheel_damping)


def alpha_from_physics(p: RobotParams = None) -> float:
    """HF比に使う α = exp(−Δt/τ_wheel)。案 3 の罰に使う α=0.5 とは別物。"""
    p = p or RobotParams()
    return math.exp(-p.control_dt / wheel_tau(p))


HF_ALPHA = alpha_from_physics()
HF_WORST = 2 * HF_ALPHA / (1 + HF_ALPHA)   # 全振幅 Nyquist 方形波での理論最悪値


def hf_energy_ratio(actions, alpha: float = None) -> float:
    """高周波エネルギー比。actions は (T, 2)、値域 [-1, 1]。"""
    alpha = HF_ALPHA if alpha is None else alpha
    bar, acc = np.zeros(2), 0.0
    for a in np.asarray(actions, dtype=np.float64):
        bar = alpha * bar + (1.0 - alpha) * a
        d = a - bar
        acc += float(np.dot(d, d))
    return math.sqrt(acc / max(len(actions), 1) / 2.0)


def classify_geoms(model):
    """機体側 geom と 壁・柱側 geom の id 配列を返す（メッシュは除外）。"""
    wall_ids, robot_ids = [], []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith(WALL_PREFIXES):
            wall_ids.append(gid)
        elif (model.geom_bodyid[gid] != 0
              and model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH):
            robot_ids.append(gid)
    return np.array(robot_ids, dtype=int), np.array(wall_ids, dtype=int)


def build_surface_samples(model, robot_ids, n_ring: int = 8):
    """機体 geom の表面を代表点で標本化する。戻り値 [(gid, local_pts, r_offset)]。"""
    out = []
    for gid in robot_ids:
        t, s = model.geom_type[gid], model.geom_size[gid]
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            sx, sy, sz = s
            pts = np.array([[ex * sx, ey * sy, ez * sz]
                            for ex in (-1, 1) for ey in (-1, 1) for ez in (-1, 1)])
            out.append((int(gid), pts, 0.0))
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, h = s[0], s[1]
            th = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
            ring = np.stack([r * np.cos(th), r * np.sin(th), np.zeros_like(th)], axis=1)
            out.append((int(gid), np.concatenate([ring + [0, 0, h], ring - [0, 0, h]]), 0.0))
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            out.append((int(gid), np.zeros((1, 3)), float(s[0])))
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            out.append((int(gid), np.array([[0, 0, s[1]], [0, 0, -s[1]]]), float(s[0])))
    return out


class ClearanceMeter:
    """1 エピソードを通して最小壁余裕を追う。`reset()` 直後に作ること。"""

    def __init__(self, sim):
        self.model, self.data = sim.model, sim.data
        robot_ids, self.wall_ids = classify_geoms(self.model)
        self.samples = build_surface_samples(self.model, robot_ids)
        # 壁は world 固定なのでエピソード中は不変
        self.wall_c = self.data.geom_xpos[self.wall_ids].copy()
        self.wall_R = self.data.geom_xmat[self.wall_ids].reshape(-1, 3, 3).copy()
        self.wall_s = self.model.geom_size[self.wall_ids].copy()
        self.worst = float("inf")

    def update(self) -> float:
        """現在の姿勢での壁余裕 [m] を返し、最小値を更新する。"""
        if not self.samples or len(self.wall_ids) == 0:
            return float("inf")
        com = self.data.xpos[self.model.geom_bodyid[self.samples[0][0]]]
        sel = np.linalg.norm(self.wall_c - com, axis=1) < PREFILTER_M
        if not sel.any():
            return float("inf")
        c, R, hs = self.wall_c[sel], self.wall_R[sel], self.wall_s[sel]
        best = np.inf
        for gid, local, r_off in self.samples:
            xm = self.data.geom_xmat[gid].reshape(3, 3)
            pw = self.data.geom_xpos[gid] + local @ xm.T
            rel = pw[:, None, :] - c[None, :, :]
            loc = np.einsum("wij,kwi->kwj", R, rel)
            gap = np.maximum(np.abs(loc) - hs[None, :, :], 0.0)
            best = min(best, float((np.linalg.norm(gap, axis=2) - r_off).min()))
        self.worst = min(self.worst, best)
        return best
