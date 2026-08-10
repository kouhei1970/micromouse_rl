"""
competition/model_verification.py
================
物理モデル検証スイート（docs/MODEL_VERIFICATION_PLAN.md §6/§7 の実装、r2 改訂反映）。

実行方法（リポジトリルートで）:
    .venv/bin/python -m pytest competition/model_verification.py -v

正本は docs/MODEL_VERIFICATION_PLAN.md。矛盾があればそちらが勝つ。
期待値は mouse.params.RobotParams とロード済み mujoco.MjModel の実測値
（body_mass, actuator_gainprm, actuator_biasprm, dof_damping, dof_frictionloss,
dof_armature, sensor_cutoff, opt.timestep, d.contact.friction 等）から実行時に
導出し、数値をハードコードしない。棄却済み所見 R1/R2（計画書 §2.2）の数値・
r1（改訂前）の公称値は使用しない。

キャスターは r2 改訂により μ_c=0.08 の滑りキャスターへ変更済み（mouse/params.py
caster_pair_friction, assets/mouse_v2.xml 反映済み・本ファイルは再生成しない）。
μ_c は本ファイルでもハードコードせず、d.contact の実効摩擦（<pair> 実測値）を
第一情報源とし、取得できない場合のみ params.caster_pair_friction を解析する。

qacc 監視は計画書 §6 共通規約(v)により2段構成（教授承認済みの正式仕様）:
  (a) 全テスト共通: NaN/Inf 即不合格 + max|qacc| < QACC_LOOSE (=1e6 rad/s^2)
  (b) 接触過渡を含まない滑らか力学のテスト(Test0/1/2/3/5/7a、Test6静止区間)は
      追加で max|qacc| < QACC_STRICT (=1e4 rad/s^2) を厳格適用
  Test4・Test7b/7c（Test4トラジェクトリの延長/再利用）・Test8動的・接触健全性は
  (a) のみ（全開スリップ・急制動の正当な接触過渡を誤検出しないため）。
"""
import atexit
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import mujoco
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import xml.etree.ElementTree as ET

from mouse.params import RobotParams
from mouse.sim import MouseSim
from mouse.mjcf import build_open_floor_xml, build_maze_robot_xml, _new_mujoco_root, _write_xml
from mouse.robot_v2 import add_micromouse_v2
from mouse import motor as motor_ref
from common.output_manager import OutputManager

# ======================================================================
# 定数（計画書 §6 共通規約(v)。qacc 監視の2閾値のみ、ここに集約）
# ======================================================================
QACC_LOOSE = 1.0e6    # rad/s^2: 全テスト共通の数値発散検出（真の発散 6.6e7 級を検出）


def qacc_strict_limit(params=None):
    """滑らか力学テスト用の厳格 qacc 閾値 [rad/s^2]（計画書 r4、コミット 360ab5b）。

    r4 で固定値 1e4 を廃止し「解析起動過渡の 2 倍」の実行時導出へ変更:
        limit = 2 * (gainprm0 * V_max - tau_c) / (I_w + armature)
    理由: コアレス化（r3）で armature=1.475e-6 となり、3V ステップの正当な起動角加速度
    が解析値 1.39e4 rad/s^2 に達し、固定値 1e4 では接触ゼロの空中試験（Test0）でも
    誤検出するため。ハードコード禁止原則の帰結でもある。
    """
    p = params if params is not None else RobotParams()
    wheel_inertia = 0.5 * p.mass_wheel * p.wheel_radius ** 2
    alpha0 = (p.gainprm0 * p.voltage_limit - p.wheel_frictionloss) / (wheel_inertia + p.armature)
    return 2.0 * alpha0


QACC_STRICT = qacc_strict_limit()   # rad/s^2（r4: 実行時導出。r3 パラメータで ≈2.78e4）

# データシート整合アンカー（計画書 r3。FAULHABER 1717T003SR 公式カタログ値）。
# これは「モデルから導出する期待値」ではなく外部実機カタログとの独立照合であり、
# Test0/1.3〜0.5V 等の試験電圧値と同様に手順定義の一部として扱う（数値ハードコード
# 禁止の対象はモデルから導出すべき期待/合格値であり、外部データシート引用値ではない）。
# 出典: https://eshop.faulhaber.com/en/1717T003SR/1717T003SR (EN_1717_SR_DFF.pdf)
MOTOR_CATALOG_NO_LOAD_RPM = 14100.0

# ======================================================================
# 結果収集用グローバル（pytest 実行でも JSON を出すため atexit で保存する。
# conftest.py は追加禁止のため、本ファイル内の atexit 登録で完結させる）
# ======================================================================
RESULTS = {}       # test_key -> dict(checks=[...], passed=bool, calibration={...})
CALIB = {}         # テスト間の段階較正値（大きい配列を含みうる。JSON化しない）


def _jsonable(o):
    """numpy 型・配列を JSON 化可能な素の Python 型へ変換する。"""
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (float, int, str, bool)) or o is None:
        return o
    return str(o)


def _save_results():
    """pytest セッション終了時（プロセス終了時）に結果 JSON を保存する。"""
    if not RESULTS:
        return
    try:
        om = OutputManager("model_verification")
        n_pass = sum(1 for v in RESULTS.values() if v.get('passed'))
        n_total = len(RESULTS)
        om.save_metrics(
            {"summary": {"n_pass": n_pass, "n_total": n_total}},
            phase_specific={"tests": _jsonable(RESULTS)},
        )
        om.finalize(summary=f"model_verification: {n_pass}/{n_total} テスト PASS")
        print(f"\n[model_verification] 結果 JSON 保存先: {om.archive_dir / 'metrics.json'}")
    except Exception as e:  # noqa: BLE001 - 保存失敗でテスト結果自体は握りつぶさない
        print(f"[model_verification] JSON 保存に失敗: {e!r}")


atexit.register(_save_results)


def _finish(test_key, checks, calib=None):
    """checks: list of dict(name, expected, actual, tol, ok[, note])。
    RESULTS に集約し、(all_ok, message) を返す。"""
    all_ok = all(bool(c['ok']) for c in checks)
    RESULTS[test_key] = dict(checks=_jsonable(checks), passed=all_ok,
                              calibration=_jsonable(calib or {}))
    lines = [f"\n=== {test_key}: {'PASS' if all_ok else 'FAIL'} "
             f"({sum(1 for c in checks if c['ok'])}/{len(checks)} 項目) ==="]
    for c in checks:
        status = "PASS" if c['ok'] else "FAIL"
        lines.append(f"  [{status}] {c['name']}: expected={c['expected']} actual={c['actual']} "
                      f"tol={c.get('tol','-')} {c.get('note','')}")
    msg = "\n".join(lines)
    print(msg)
    return all_ok, msg


# ======================================================================
# qacc 発散監視（2段構成。計画書 §6 共通規約(v)）
# ======================================================================
class QaccTracker:
    def __init__(self):
        self.max_abs = 0.0
        self.bad = False  # NaN/Inf 検出

    def update(self, data):
        qacc = data.qacc
        if qacc.size == 0:
            return
        if (not np.all(np.isfinite(qacc)) or not np.all(np.isfinite(data.qvel))
                or not np.all(np.isfinite(data.qpos))):
            self.bad = True
            return
        m = float(np.max(np.abs(qacc)))
        if m > self.max_abs:
            self.max_abs = m

    def check(self, strict_limit=None):
        """strict_limit を指定すると (b) の厳格判定も追加適用する。
        常に (a) 共通判定（NaN/Inf・QACC_LOOSE）は適用する。"""
        if self.bad:
            return False, "NaN/Inf が qacc/qvel/qpos に検出された"
        if self.max_abs >= QACC_LOOSE:
            return False, f"max|qacc|={self.max_abs:.3e} >= QACC_LOOSE({QACC_LOOSE:.0e})"
        if strict_limit is not None and self.max_abs >= strict_limit:
            return False, f"max|qacc|={self.max_abs:.3e} >= strict({strict_limit:.0e})"
        return True, f"max|qacc|={self.max_abs:.3e} rad/s^2"


# ======================================================================
# モデルハンドル解決（body/joint/geom/actuator id。モデルごとに毎回解決する）
# ======================================================================
def get_handles(model):
    def bid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    def jid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def gid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def aid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def geom_of_body(bidx):
        cands = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bidx]
        return cands[0] if cands else -1

    mouse_body_id = bid('mouse')
    root_joint_id = jid('root')
    left_wj = jid('left_wheel_joint')
    right_wj = jid('right_wheel_joint')
    left_wheel_body_id = bid('left_wheel')
    right_wheel_body_id = bid('right_wheel')

    return SimpleNamespace(
        mouse_body_id=mouse_body_id,
        root_joint_id=root_joint_id,
        root_qpos_adr=int(model.jnt_qposadr[root_joint_id]),
        root_dof_adr=int(model.jnt_dofadr[root_joint_id]),
        left_wheel_joint_id=left_wj,
        right_wheel_joint_id=right_wj,
        left_dof=int(model.jnt_dofadr[left_wj]),
        right_dof=int(model.jnt_dofadr[right_wj]),
        left_wheel_body_id=left_wheel_body_id,
        right_wheel_body_id=right_wheel_body_id,
        left_wheel_geom_id=geom_of_body(left_wheel_body_id),
        right_wheel_geom_id=geom_of_body(right_wheel_body_id),
        caster_front_geom_id=gid('caster_front'),
        caster_back_geom_id=gid('caster_back'),
        floor_geom_id=gid('floor'),
        act_left_id=aid('act_left'),
        act_right_id=aid('act_right'),
    )


# ======================================================================
# 静定・接触計測ヘルパー（計画書 §6 共通規約(iv)）
# ======================================================================
def settle_contact(model, data, handles, tracker=None, max_time=2.0, tol=0.005, hold_time=0.1):
    """ctrl=0 のまま重力静定させ、接触法線力合計 = Mg ± tol(相対) が hold_time
    連続で成立したら終了する。戻り値: (settled: bool, elapsed_time: float)。"""
    Mg = float(model.body_subtreemass[handles.mouse_body_id] * abs(model.opt.gravity[2]))
    dt = float(model.opt.timestep)
    hold_steps = max(1, int(round(hold_time / dt)))
    max_steps = int(round(max_time / dt))
    consec = 0
    force6 = np.zeros(6, dtype=np.float64)
    for step in range(max_steps):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        if tracker is not None:
            tracker.update(data)
        total_normal = 0.0
        for i in range(data.ncon):
            mujoco.mj_contactForce(model, data, i, force6)
            total_normal += abs(float(force6[0]))
        if Mg > 0 and abs(total_normal - Mg) / Mg < tol:
            consec += 1
            if consec >= hold_steps:
                return True, step * dt
        else:
            consec = 0
    return False, max_steps * dt


def measure_contacts(model, data, handles):
    """現在の data.contact から、車輪-床/キャスター-床の法線力合計・摩擦係数を実測する。"""
    force6 = np.zeros(6, dtype=np.float64)
    N_wheel = 0.0
    N_caster = 0.0
    N_front = 0.0
    N_back = 0.0
    mu_wheel = None
    mu_caster = None
    contacts = []
    wheel_geoms = {handles.left_wheel_geom_id, handles.right_wheel_geom_id}
    for i in range(data.ncon):
        c = data.contact[i]
        mujoco.mj_contactForce(model, data, i, force6)
        nf = abs(float(force6[0]))
        pair = (c.geom1, c.geom2)
        if handles.floor_geom_id in pair:
            other = c.geom1 if c.geom2 == handles.floor_geom_id else c.geom2
            if other in wheel_geoms:
                N_wheel += nf
                if mu_wheel is None:
                    mu_wheel = float(c.friction[0])
            elif other == handles.caster_front_geom_id:
                N_caster += nf
                N_front += nf
                if mu_caster is None:
                    mu_caster = float(c.friction[0])
            elif other == handles.caster_back_geom_id:
                N_caster += nf
                N_back += nf
                if mu_caster is None:
                    mu_caster = float(c.friction[0])
        contacts.append(dict(geom1=model.geom(c.geom1).name, geom2=model.geom(c.geom2).name,
                              friction0=float(c.friction[0]), normal=nf))
    return SimpleNamespace(N_wheel=N_wheel, N_caster=N_caster, N_front=N_front, N_back=N_back,
                            mu_wheel=mu_wheel, mu_caster=mu_caster, contacts=contacts)


def measure_contacts_avg(model, data, handles, ctrl=(0.0, 0.0), n_samples=200):
    """指定 ctrl を維持したまま n_samples 物理ステップ進め、車輪/キャスター法線力を
    時間平均する。

    【発見事項・実装上の理由】本検証中に、車輪2+キャスター2の計4点接地が
    静力学的に不静定（3点で足りる支持を4点で行っている）であるため、瞬時の
    d.contact 読み取り1点だけでは荷重配分（N_wheel/N_caster の分配）がタイム
    ステップ単位で大きくチャタリングし得ることを確認した（例: 停止状態でも
    数百ms継続すると N_caster が静止直後値の約20倍まで増加し得る。走行中は
    さらに顕著）。単一瞬間値は測定として頑健でないため、以後の全ての
    N_wheel/N_caster 測定はこの時間平均版を用いる（数値を追う後付けではなく、
    測定手法として本来あるべき平均化 — 詳細は報告書「計画書との矛盾・懸念」参照）。
    """
    accum_w = accum_c = accum_f = accum_b = 0.0
    mu_wheel = None
    mu_caster = None
    for _ in range(n_samples):
        data.ctrl[0] = float(ctrl[0])
        data.ctrl[1] = float(ctrl[1])
        mujoco.mj_step(model, data)
        c = measure_contacts(model, data, handles)
        accum_w += c.N_wheel
        accum_c += c.N_caster
        accum_f += c.N_front
        accum_b += c.N_back
        if mu_wheel is None and c.mu_wheel is not None:
            mu_wheel = c.mu_wheel
        if mu_caster is None and c.mu_caster is not None:
            mu_caster = c.mu_caster
    n = float(max(1, n_samples))
    return SimpleNamespace(N_wheel=accum_w / n, N_caster=accum_c / n, N_front=accum_f / n,
                            N_back=accum_b / n, mu_wheel=mu_wheel, mu_caster=mu_caster)


def parse_mu_caster(params):
    """<pair> のキャスター摩擦文字列先頭値（滑り摩擦係数 mu_c）を解析する
    （接触が取得できない場合のフォールバックのみに使用。第一情報源は d.contact 実測）。"""
    return float(params.caster_pair_friction.split()[0])


def parse_mu_roll_floor(params):
    """床 geom 摩擦文字列の第3成分（転がり摩擦係数 mu_roll [m]）を解析する。"""
    return float(params.floor_friction.split()[2])


# ======================================================================
# 低レベルロギング付きステッピング（mj_step 直接ループ。1kHz=2サブステップ毎ログ）
# ======================================================================
def run_logged(model, data, duration, ctrl_fn, tracker=None, log_every=2, t0=None,
               voltage_limit=3.0):
    """duration 秒だけ mj_step を直接ループし、log_every サブステップごとに状態をログする。
    ctrl_fn(t_rel) -> (v_left, v_right)。t0 は励振開始時刻の絶対 data.time
    （None なら現在の data.time を t=0 とする。計画書 §6 共通規約(iii)）。"""
    if t0 is None:
        t0 = data.time
    dt = float(model.opt.timestep)
    n_steps = int(round(duration / dt))
    h = get_handles(model)
    t_list, qposl, qposr, qvell, qvelr = [], [], [], [], []
    vfwd, vlat, xs, ys, yaws, pitches = [], [], [], [], [], []
    sensord, actf = [], []
    for step in range(n_steps):
        t_rel = data.time - t0
        vl, vr = ctrl_fn(t_rel)
        data.ctrl[0] = float(np.clip(vl, -voltage_limit, voltage_limit))
        data.ctrl[1] = float(np.clip(vr, -voltage_limit, voltage_limit))
        mujoco.mj_step(model, data)
        if tracker is not None:
            tracker.update(data)
        if (step + 1) % log_every == 0:
            xmat = data.xmat[h.mouse_body_id].reshape(3, 3)
            v_world = data.qvel[h.root_dof_adr:h.root_dof_adr + 3]
            v_fwd = float(np.dot(v_world, xmat[:, 0]))
            v_lat = float(np.dot(v_world, xmat[:, 1]))  # 機体ローカル y 軸方向速度（横滑り検出用。r5確定）
            qadr = h.root_qpos_adr
            t_list.append(data.time - t0)
            qposl.append(float(data.qpos[model.jnt_qposadr[h.left_wheel_joint_id]]))
            qposr.append(float(data.qpos[model.jnt_qposadr[h.right_wheel_joint_id]]))
            qvell.append(float(data.qvel[h.left_dof]))
            qvelr.append(float(data.qvel[h.right_dof]))
            vfwd.append(v_fwd)
            vlat.append(v_lat)
            xs.append(float(data.qpos[qadr]))
            ys.append(float(data.qpos[qadr + 1]))
            yaws.append(float(np.arctan2(xmat[1, 0], xmat[0, 0])))
            pitches.append(float(np.arcsin(np.clip(-xmat[2, 0], -1.0, 1.0))))
            sensord.append(data.sensordata.copy())
            actf.append(data.actuator_force.copy())
    return dict(
        t=np.array(t_list), qpos_l=np.array(qposl), qpos_r=np.array(qposr),
        qvel_l=np.array(qvell), qvel_r=np.array(qvelr), v_fwd=np.array(vfwd), v_lat=np.array(vlat),
        x=np.array(xs), y=np.array(ys), yaw=np.array(yaws), pitch=np.array(pitches),
        sensordata=np.array(sensord) if sensord else np.zeros((0, 12)),
        actuator_force=np.array(actf) if actf else np.zeros((0, 2)),
    )


# ======================================================================
# フィッティング・回帰ヘルパー（scipy 不使用。numpy 最小二乗のみ）
# ======================================================================
def linreg(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(r2)


def fit_first_order_rise(t, y, y0=0.0):
    """y(t) = y0 + (y_ss-y0)*(1-exp(-t/tau)) 型応答の y_ss（末尾5%平均）・
    tau（63.2%到達時刻）・当てはめ R^2 を推定する（scipy 不使用、63.2%法）。"""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    tail = max(1, len(y) // 20)
    y_ss = float(np.mean(y[-tail:]))
    if y_ss == y0:
        return y_ss, float('nan'), float('nan')
    target = y0 + 0.632 * (y_ss - y0)
    crossed = (y >= target) if y_ss > y0 else (y <= target)
    if not np.any(crossed):
        tau = float('nan')
    else:
        idx = int(np.argmax(crossed))
        tau = float(t[idx]) if idx > 0 else float(t[0])
    if not np.isfinite(tau) or tau <= 0:
        return y_ss, float('nan'), float('-inf')
    y_fit = y_ss - (y_ss - y0) * np.exp(-t / tau)
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return y_ss, tau, r2


def fit_affine_decay(t, v, M_eff):
    """coast-down: M_eff*dv/dt = -c_eff*v - Fc の affine 関係を最小二乗で同定する。
    有限差分 dv/dt を v に対して回帰し、傾き=-c_eff/M_eff、切片=-Fc/M_eff を得る。
    停止直前の低速域（ノイズ支配）は除外する。"""
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    dvdt = np.gradient(v, t)
    v0 = v[0]
    mask = v > 0.05 * v0
    if np.sum(mask) < 5:
        mask = np.ones_like(v, dtype=bool)
    slope, intercept, r2 = linreg(v[mask], dvdt[mask])
    c_eff_meas = -slope * M_eff
    Fc_meas = -intercept * M_eff
    tau_meas = M_eff / c_eff_meas if c_eff_meas > 0 else float('nan')
    return dict(c_eff_meas=c_eff_meas, Fc_meas=Fc_meas, tau_meas=tau_meas, r2=r2)


# ======================================================================
# 派生定数（params とロード済み MjModel から実行時に導出。計画書 §4.2/§4.3 の式）
# ======================================================================
def derive_constants(params, xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    h = get_handles(model)

    N = params.gear_ratio
    Kt = params.motor_Kt
    Ke = params.motor_Ke
    R = params.motor_R
    b = params.wheel_damping
    tau_c = params.wheel_frictionloss
    r = params.wheel_radius
    L = params.tread
    armature = params.armature

    gainprm0 = float(model.actuator_gainprm[h.act_left_id, 0])
    biasprm2 = float(model.actuator_biasprm[h.act_left_id, 2])
    assert abs(gainprm0 - params.gainprm0) < 1e-9 * max(1.0, abs(params.gainprm0)), (
        "モデルの gainprm[0] が params.gainprm0 と不一致（XML 生成コードの退行の疑い）")
    assert abs(biasprm2 - params.biasprm2) < 1e-9 * max(1.0, abs(params.biasprm2)), (
        "モデルの biasprm[2] が params.biasprm2 と不一致（XML 生成コードの退行の疑い）")

    M = float(model.body_subtreemass[h.mouse_body_id])
    m_wheel = float(model.body_mass[h.left_wheel_body_id])
    I_w = 0.5 * m_wheel * r ** 2  # 計画書 §4.3: I_w = (1/2) m_w r^2（円柱の対称軸まわり）

    M_eff = M + 2 * (I_w + armature) / r ** 2
    c_eff = 2 * (N ** 2 * Kt * Ke / R + b) / r ** 2
    tau_v = M_eff / c_eff

    g = float(abs(model.opt.gravity[2]))
    Mg = M * g

    L_over_2r = L / (2 * r)
    b_rot = 2 * (N ** 2 * Kt * Ke / R + b) * (L_over_2r) ** 2

    caster_arm = abs(float(model.geom_pos[h.caster_front_geom_id][0]))

    return SimpleNamespace(
        N=N, Kt=Kt, Ke=Ke, R=R, b=b, tau_c=tau_c, r=r, L=L, armature=armature,
        gainprm0=gainprm0, biasprm2=biasprm2, M=M, m_wheel=m_wheel, I_w=I_w,
        M_eff=M_eff, c_eff=c_eff, tau_v=tau_v, g=g, Mg=Mg, b_rot=b_rot,
        L_over_2r=L_over_2r, caster_arm=caster_arm, voltage_limit=params.voltage_limit,
    )


# ======================================================================
# pytest フィクスチャ（conftest.py 不使用。本ファイル内で完結）
# ======================================================================
@pytest.fixture(scope="session")
def params():
    return RobotParams()


@pytest.fixture(scope="session")
def xml_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("mv_jigs")


@pytest.fixture(scope="session")
def open_xml(params, xml_dir):
    path = xml_dir / "open_floor.xml"
    build_open_floor_xml(str(path), params=params)
    return str(path)


@pytest.fixture(scope="session")
def D(params, open_xml):
    return derive_constants(params, open_xml)


# ======================================================================
# Test 0: 空中無負荷モータ試験（計画書 §6 Test0）
# ======================================================================
def test_00_air_noload(params, open_xml):
    model = mujoco.MjModel.from_xml_path(open_xml)
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    h = get_handles(model)

    def place_airborne():
        mujoco.mj_resetData(model, data)
        data.qpos[h.root_qpos_adr:h.root_qpos_adr + 3] = [0.0, 0.0, 5.0]
        data.qpos[h.root_qpos_adr + 3:h.root_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    place_airborne()
    assert data.ncon == 0, "Test0 前提: 空中配置なのに接触が発生している"

    I_w = 0.5 * params.mass_wheel * params.wheel_radius ** 2
    theory_tau0 = (I_w + params.armature) / (
        params.gear_ratio ** 2 * params.motor_Kt * params.motor_Ke / params.motor_R + params.wheel_damping
    )

    voltages = [0.3, 1.5, 3.0]
    tracker = QaccTracker()
    meas = {}
    checks = []

    for side in ('left', 'right'):
        for V in voltages:
            place_airborne()

            def ctrl_fn(t, side=side, V=V):
                return (V, 0.0) if side == 'left' else (0.0, V)

            log = run_logged(model, data, duration=1.5, ctrl_fn=ctrl_fn, tracker=tracker,
                              log_every=2, voltage_limit=params.voltage_limit)
            omega_driven = log['qvel_l'] if side == 'left' else log['qvel_r']
            omega_other = log['qvel_r'] if side == 'left' else log['qvel_l']

            tail = max(1, len(omega_driven) // 20)
            omega_ss_meas = float(np.mean(omega_driven[-tail:]))
            max_other = float(np.max(np.abs(omega_other)))

            theory_omega = motor_ref.omega_ss(V, params)
            target = 0.632 * omega_ss_meas
            crossed = (omega_driven >= target) if omega_ss_meas > 0 else (omega_driven <= target)
            if np.any(crossed):
                idx = int(np.argmax(crossed))
                tau0_meas = float(log['t'][idx]) if idx > 0 else float(log['t'][0])
            else:
                tau0_meas = float('nan')

            meas[(side, V)] = dict(omega_ss_meas=omega_ss_meas, tau0_meas=tau0_meas)

            ok_omega = abs(omega_ss_meas - theory_omega) / abs(theory_omega) <= 0.02 if theory_omega != 0 else False
            checks.append(dict(name=f"omega_ss_{side}_{V}V", expected=theory_omega, actual=omega_ss_meas,
                                tol="2%", ok=ok_omega))
            ok_cross = max_other < 0.1
            checks.append(dict(name=f"crosstalk_{side}_{V}V", expected="<0.1 rad/s", actual=max_other,
                                tol="0.1 rad/s", ok=ok_cross))
            ok_tau0 = (np.isfinite(tau0_meas) and abs(tau0_meas - theory_tau0) / theory_tau0 <= 0.10)
            checks.append(dict(name=f"tau0_{side}_{V}V", expected=theory_tau0, actual=tau0_meas,
                                tol="10%", ok=ok_tau0))

    for V in voltages:
        oL = meas[('left', V)]['omega_ss_meas']
        oR = meas[('right', V)]['omega_ss_meas']
        diff = abs(oL - oR) / max(abs(oL), 1e-9)
        checks.append(dict(name=f"left_right_symmetry_{V}V", expected=0.0, actual=diff, tol="0.5%",
                            ok=diff <= 0.005))

    # データシート整合アンカー（r3、モータ軸換算無負荷回転数）
    omega_ss_3V = meas[('left', 3.0)]['omega_ss_meas']
    model_rpm = params.gear_ratio * omega_ss_3V * 60.0 / (2.0 * np.pi)
    ratio = (model_rpm - MOTOR_CATALOG_NO_LOAD_RPM) / MOTOR_CATALOG_NO_LOAD_RPM
    ok_catalog = (-0.03 <= ratio <= 0.005)
    checks.append(dict(name="catalog_no_load_rpm_anchor", expected=f"~{MOTOR_CATALOG_NO_LOAD_RPM:.0f}rpm (-3%以内)",
                        actual=model_rpm, tol="-3%〜0%", ok=ok_catalog,
                        note=f"ratio={ratio*100:+.2f}%"))

    qacc_ok, qacc_msg = tracker.check(strict_limit=QACC_STRICT)
    checks.append(dict(name="qacc_monitor_strict", expected="OK", actual=qacc_msg, tol="strict", ok=qacc_ok))

    CALIB['test0_tau0_theory'] = theory_tau0
    ok, msg = _finish("Test0_air_noload", checks, calib=dict(meas={f"{s}_{v}": m for (s, v), m in meas.items()},
                                                               catalog_ratio=ratio))
    assert ok, msg


# ======================================================================
# Test 1: 電圧ステップ応答（直進）— 一次遅れ整合（計画書 §6 Test1）
# ======================================================================
def test_01_voltage_step(params, open_xml, D):
    sim = MouseSim(open_xml, params=params)
    h = get_handles(sim.model)
    tracker = QaccTracker()

    sim.full_reset(cell=(0, 0), heading_deg=0)
    settled0, t_settle0 = settle_contact(sim.model, sim.data, h, tracker=tracker)
    contacts0 = measure_contacts_avg(sim.model, sim.data, h, ctrl=(0.0, 0.0), n_samples=100)
    N_c_static = contacts0.N_caster
    N_w_static = contacts0.N_wheel
    mu_caster = contacts0.mu_caster if contacts0.mu_caster is not None else parse_mu_caster(params)

    checks = [dict(name="settled_before_excitation", expected=True, actual=settled0, tol="-", ok=settled0,
                    note=f"t={t_settle0:.3f}s")]

    voltages = [0.3, 0.5]
    v_ss_meas = {}
    tau_meas = {}
    N_c_dynamic = {}
    last_log = None
    last_state = None

    for V0 in voltages:
        sim.full_reset(cell=(0, 0), heading_deg=0)
        settle_contact(sim.model, sim.data, h, tracker=tracker)
        t0 = sim.data.time

        def ctrl_fn(t, V0=V0):
            return (V0, V0)

        log = run_logged(sim.model, sim.data, duration=2.5, ctrl_fn=ctrl_fn, tracker=tracker,
                          log_every=2, t0=t0, voltage_limit=params.voltage_limit)

        v_ss, tau, r2 = fit_first_order_rise(log['t'], log['v_fwd'], y0=0.0)
        v_ss_meas[V0] = v_ss
        tau_meas[V0] = tau

        # 定常巡航中のキャスター荷重を、当該電圧を維持したまま追加で時間平均測定する
        # （静止直後の静的値とは大きく異なりうる。4点接地の不静定性に起因する
        # チャタリングの発見に基づく実装。下記報告の「計画書との矛盾・懸念」参照）
        contacts_dyn = measure_contacts_avg(sim.model, sim.data, h, ctrl=(V0, V0), n_samples=200)
        N_c = contacts_dyn.N_caster
        N_c_dynamic[V0] = N_c

        v_ss_theory = (2 * (D.gainprm0 / D.r) * V0 - 2 * (D.tau_c / D.r) - mu_caster * N_c) / D.c_eff

        ok_tau = np.isfinite(tau) and abs(tau - D.tau_v) / D.tau_v <= 0.20
        checks.append(dict(name=f"tau_{V0}V", expected=D.tau_v, actual=tau, tol="20%", ok=ok_tau))
        ok_vss = abs(v_ss - v_ss_theory) / v_ss_theory <= 0.10 if v_ss_theory != 0 else False
        checks.append(dict(name=f"v_ss_{V0}V", expected=v_ss_theory, actual=v_ss, tol="10%", ok=ok_vss))
        ok_r2 = r2 >= 0.98
        checks.append(dict(name=f"r2_{V0}V", expected=">=0.98", actual=r2, tol="-", ok=ok_r2))

        max_v = float(np.max(log['v_fwd']))
        overshoot = (max_v - v_ss) / v_ss if v_ss > 0 else 0.0
        ok_over = overshoot <= 0.02
        checks.append(dict(name=f"overshoot_{V0}V", expected="<=2%", actual=overshoot, tol="-", ok=ok_over))

        yaw_drift_deg = float(np.degrees(abs(log['yaw'][-1] - log['yaw'][0])))
        ok_yaw = yaw_drift_deg <= 1.0
        checks.append(dict(name=f"yaw_drift_deg_{V0}V", expected="<=1deg", actual=yaw_drift_deg, tol="-", ok=ok_yaw))

        omega_avg_end = float(np.mean((log['qvel_l'][-10:] + log['qvel_r'][-10:]) / 2.0))
        v_end = float(np.mean(log['v_fwd'][-10:]))
        wheel_surface_v = D.r * omega_avg_end
        slip_pct = abs(wheel_surface_v - v_end) / v_end if v_end > 1e-6 else 0.0
        ok_slip = slip_pct < 0.02
        checks.append(dict(name=f"nonslip_check_{V0}V", expected="<2%", actual=slip_pct, tol="-", ok=ok_slip))

        last_log = log
        last_state = dict(qpos=sim.data.qpos.copy(), qvel=sim.data.qvel.copy(), v_fwd=v_end)

    grad_meas = (v_ss_meas[0.5] - v_ss_meas[0.3]) / (0.5 - 0.3)
    grad_theory = 2 * D.gainprm0 / (D.r * D.c_eff)  # = 2*(N*Kt/R)/(r*c_eff)
    ok_grad = abs(grad_meas - grad_theory) / grad_theory <= 0.10
    checks.append(dict(name="v_ss_gradient", expected=grad_theory, actual=grad_meas, tol="10%", ok=ok_grad))

    qacc_ok, qacc_msg = tracker.check(strict_limit=QACC_STRICT)
    checks.append(dict(name="qacc_monitor_strict", expected="OK", actual=qacc_msg, tol="strict", ok=qacc_ok))

    # Test2/Test6 で使う較正値・スナップショットを保存
    CALIB['N_caster_static'] = N_c_static
    CALIB['N_caster_dynamic_05V'] = N_c_dynamic[0.5]
    CALIB['N_wheel_static'] = N_w_static
    CALIB['mu_caster_meas'] = mu_caster
    CALIB['test1_tau_meas_05V'] = tau_meas[0.5]
    CALIB['test1_05V_final_state'] = last_state
    CALIB['test1_v_ss_theory'] = {V0: (2 * (D.gainprm0 / D.r) * V0 - 2 * (D.tau_c / D.r) - mu_caster * N_c_dynamic[V0])
                                   / D.c_eff for V0 in voltages}
    CALIB['test1_accelx_log'] = dict(t=last_log['t'].copy(), accel_x=last_log['sensordata'][:, 6].copy(),
                                      v_ss_theory=CALIB['test1_v_ss_theory'][0.5], tau_v=D.tau_v)

    ok, msg = _finish("Test1_voltage_step", checks, calib=dict(v_ss_meas=v_ss_meas, tau_meas=tau_meas,
                                                                 grad_meas=grad_meas, N_c_static=N_c_static,
                                                                 N_c_dynamic=N_c_dynamic))
    assert ok, msg


# ======================================================================
# Test 2: 惰行試験（coast-down）— 摩擦パラメータ同定（計画書 §6 Test2）
# ======================================================================
def test_02_coast_down(params, open_xml, D):
    checks = []
    tracker = QaccTracker()

    snap = CALIB.get('test1_05V_final_state')
    tau1_meas = CALIB.get('test1_tau_meas_05V', D.tau_v)
    # coast-down は Test1 0.5V の巡航（動的定常）状態から開始するため、キャスター荷重は
    # 静止直後の静的値ではなく Test1 が実測した動的（巡航中）値を使う
    # （4点接地の不静定性に起因するチャタリングの発見に基づく。報告書参照）
    N_c = CALIB.get('N_caster_dynamic_05V', CALIB.get('N_caster_static', 0.0))
    N_w = CALIB.get('N_wheel_static', 0.0)
    mu_caster = CALIB.get('mu_caster_meas', parse_mu_caster(params))
    mu_roll = parse_mu_roll_floor(params)

    sim = MouseSim(open_xml, params=params)
    h = get_handles(sim.model)

    if snap is not None:
        sim.data.qpos[:] = snap['qpos']
        sim.data.qvel[:] = snap['qvel']
        mujoco.mj_forward(sim.model, sim.data)
        v0 = snap['v_fwd']
        note_src = "Test1 0.5V 定常状態から継続"
    else:
        sim.full_reset(cell=(0, 0), heading_deg=0)
        settle_contact(sim.model, sim.data, h, tracker=tracker)
        t0 = sim.data.time
        run_logged(sim.model, sim.data, duration=2.0, ctrl_fn=lambda t: (0.5, 0.5), tracker=tracker,
                   log_every=2, t0=t0, voltage_limit=params.voltage_limit)
        v0 = sim.privileged_velocity()[0]
        note_src = "Test1 較正値取得失敗のためフォールバック実行"

    t0 = sim.data.time
    log = run_logged(sim.model, sim.data, duration=2.0, ctrl_fn=lambda t: (0.0, 0.0), tracker=tracker,
                      log_every=2, t0=t0, voltage_limit=params.voltage_limit)

    fit = fit_affine_decay(log['t'], log['v_fwd'], D.M_eff)
    Fc_theory = 2 * D.tau_c / D.r + mu_roll * N_w / D.r + mu_caster * N_c
    t_stop_theory = D.tau_v * np.log(1 + v0 * D.c_eff / Fc_theory) if Fc_theory > 0 else float('nan')

    hold = max(1, int(round(0.02 / (log['t'][1] - log['t'][0])))) if len(log['t']) > 1 else 1
    below = np.abs(log['v_fwd']) < 0.001
    t_stop_meas = float('nan')
    for i in range(len(below) - hold):
        if np.all(below[i:i + hold]):
            t_stop_meas = float(log['t'][i])
            break

    ok_tau_cross = (np.isfinite(fit['tau_meas']) and np.isfinite(tau1_meas)
                     and abs(fit['tau_meas'] - tau1_meas) / tau1_meas <= 0.15)
    checks.append(dict(name="tau_coast_vs_test1", expected=tau1_meas, actual=fit['tau_meas'], tol="15%",
                        ok=ok_tau_cross))

    ok_tstop = np.isfinite(t_stop_meas) and abs(t_stop_meas - t_stop_theory) / t_stop_theory <= 0.25
    checks.append(dict(name="t_stop", expected=t_stop_theory, actual=t_stop_meas, tol="25%", ok=ok_tstop))

    ok_fc_order = (Fc_theory / 3.0) <= fit['Fc_meas'] <= (Fc_theory * 3.0)
    checks.append(dict(name="Fc_meas_order_of_magnitude", expected=Fc_theory, actual=fit['Fc_meas'],
                        tol="1/3〜3倍", ok=ok_fc_order))

    dv = np.diff(log['v_fwd'])
    n_up = int(np.sum(dv > 1e-4))
    ok_monotonic = n_up <= max(3, int(0.01 * len(dv)))
    checks.append(dict(name="monotonic_decay", expected="単調減衰", actual=f"上昇サンプル数={n_up}/{len(dv)}",
                        tol="-", ok=ok_monotonic))

    ok_no_remotion = True
    if np.isfinite(t_stop_meas) and t_stop_meas + 0.5 <= log['t'][-1]:
        mask_post = (log['t'] > t_stop_meas + 0.05) & (log['t'] <= t_stop_meas + 0.5)
        if np.any(mask_post):
            ok_no_remotion = bool(np.max(np.abs(log['v_fwd'][mask_post])) < 0.005)
    checks.append(dict(name="no_remotion_after_stop", expected=True, actual=ok_no_remotion, tol="-",
                        ok=ok_no_remotion))

    qacc_ok, qacc_msg = tracker.check(strict_limit=QACC_STRICT)
    checks.append(dict(name="qacc_monitor_strict", expected="OK", actual=qacc_msg, tol="strict", ok=qacc_ok))

    CALIB['test2_Fc_meas'] = fit['Fc_meas']
    ok, msg = _finish("Test2_coast_down", checks, calib=dict(Fc_meas=fit['Fc_meas'], Fc_theory=Fc_theory,
                                                               tau_meas=fit['tau_meas'], t_stop_meas=t_stop_meas,
                                                               t_stop_theory=t_stop_theory, v0=v0, note=note_src))
    assert ok, msg


# ======================================================================
# Test 3: 超信地旋回 — トレッドとヨー慣性の整合（計画書 §6 Test3）
# ======================================================================
def test_03_pivot_turn(params, open_xml, D):
    checks = []
    tracker = QaccTracker()
    sim = MouseSim(open_xml, params=params)
    h = get_handles(sim.model)

    # (a) 較正: トルクパルスによる Izz 同定
    sim.full_reset(cell=(0, 0), heading_deg=0)
    settle_contact(sim.model, sim.data, h, tracker=tracker)
    contacts_pulse = measure_contacts_avg(sim.model, sim.data, h, ctrl=(0.0, 0.0), n_samples=50)
    N_c_pulse = contacts_pulse.N_caster
    mu_caster_pulse = contacts_pulse.mu_caster if contacts_pulse.mu_caster is not None else parse_mu_caster(params)

    # r5 確定（2026-08-10 教授承認）: スクラブ項（車輪幅スリップ抵抗）の較正値。
    # note_004_r4_diagnostics.py の切り分け実験（キャスターμ≈0化 → +0.37rad/sのみ理論と厳密一致、
    # さらに frictionloss=0 → 欠損トルク ≈3.2mN·mが残存）により、円柱車輪の線接触が MuJoCo の
    # 2点接触に離散化され、超信地旋回中の接触点間（車輪幅方向）速度差スリップが実効捩れ抵抗
    # τ_scrub=μ_floor_slide*η_meas*M*g*w_wheel/2 を生む機序と特定済み（現行条件で理論−実測 -4%、
    # frictionloss=0条件で -0.3% の定量一致）。η_meas は本テストの静定接地実測（S1 と同一手法）、
    # μ_floor_slide は d.contact 実測、w_wheel は geom_size からの実行時導出でハードコードしない。
    N_w_pulse = contacts_pulse.N_wheel
    mu_floor_slide = (contacts_pulse.mu_wheel if contacts_pulse.mu_wheel is not None
                       else float(params.floor_friction.split()[0]))
    eta_meas = N_w_pulse / D.Mg if D.Mg > 0 else float('nan')
    model_for_wheel = mujoco.MjModel.from_xml_path(open_xml)
    w_wheel = 2.0 * float(model_for_wheel.geom_size[h.left_wheel_geom_id][1])
    pulse_V = 0.3
    pulse_T = 0.05
    t0 = sim.data.time
    log_pulse = run_logged(sim.model, sim.data, duration=pulse_T, ctrl_fn=lambda t: (pulse_V, -pulse_V),
                            tracker=tracker, log_every=1, t0=t0, voltage_limit=params.voltage_limit)

    # 車輪軸トルク τ_actuator（電気項のみ）から、関節の粘性減衰・乾性摩擦損失を差し引いた
    # 正味トルク τ_w = τ_actuator - b*ω - τ_c*sign(ω) を使う（当初 τ_actuator をそのまま
    # 使っており τ_c=9.1e-4 N・m の損失を見落として I_eff を過大評価するバグがあった。
    # b_rot/I_eff の定義は電気項と関節損失を分離しているため、パルス法でも同じ分離を
    # 一貫して適用する必要がある。テスト側の明白な実装バグとして修正）
    omega_L = log_pulse['qvel_l']
    omega_R = log_pulse['qvel_r']
    tau_L = log_pulse['actuator_force'][:, 0] - D.b * omega_L - D.tau_c * np.sign(omega_L)
    tau_R = log_pulse['actuator_force'][:, 1] - D.b * omega_R - D.tau_c * np.sign(omega_R)
    F_L = tau_L / D.r
    F_R = tau_R / D.r
    tau_net_wheels = (F_L - F_R) * (D.L / 2.0)
    delta_omega = float(log_pulse['sensordata'][-1, 11])
    sign_rot = 1.0 if delta_omega >= 0 else -1.0
    tau_net = tau_net_wheels - sign_rot * mu_caster_pulse * N_c_pulse * D.caster_arm
    integral_tau = float(np.trapezoid(tau_net, log_pulse['t']))
    I_eff_meas = abs(integral_tau / delta_omega) if delta_omega != 0 else float('nan')
    wheel_term = 2 * (D.I_w + D.armature) * (D.L / (2 * D.r)) ** 2
    I_zz_meas = I_eff_meas - wheel_term if np.isfinite(I_eff_meas) else float('nan')

    # (b) 定常旋回 ±0.3V
    sim.full_reset(cell=(0, 0), heading_deg=0)
    settle_contact(sim.model, sim.data, h, tracker=tracker)

    V0 = 0.3
    t0b = sim.data.time
    logb = run_logged(sim.model, sim.data, duration=2.0, ctrl_fn=lambda t: (V0, -V0), tracker=tracker,
                       log_every=2, t0=t0b, voltage_limit=params.voltage_limit)
    omega_z = logb['sensordata'][:, 11]

    # 旋回中（動的）のキャスター荷重を、同じ差動電圧を維持したまま追加で時間平均測定する
    # （静止時の静的値とは大きく異なりうる。4点接地の不静定性由来のチャタリングの発見に
    # 基づく実装。報告書「計画書との矛盾・懸念」参照）
    contacts_dyn = measure_contacts_avg(sim.model, sim.data, h, ctrl=(V0, -V0), n_samples=200)
    N_c = contacts_dyn.N_caster
    mu_caster = contacts_dyn.mu_caster if contacts_dyn.mu_caster is not None else parse_mu_caster(params)

    Omega_ss_meas, tau3_meas, r2_3 = fit_first_order_rise(logb['t'], omega_z, y0=0.0)
    if Omega_ss_meas < 0:
        Omega_ss_meas, tau3_meas, r2_3 = fit_first_order_rise(logb['t'], -omega_z, y0=0.0)
        omega_z = -omega_z

    term_drive = 2 * (D.gainprm0 / D.r) * V0 * (D.L / 2.0)
    term_coulomb = 2 * (D.tau_c / D.r) * (D.L / 2.0)
    term_caster = mu_caster * N_c * D.caster_arm
    # r5 確定（2026-08-10 教授承認）: スクラブ項（車輪幅の2点接触離散化による速度差スリップ抵抗）を追加。
    # w_wheel/eta_meas/mu_floor_slide は上の (a) 較正ブロックで実行時導出済み（ハードコード禁止）。
    term_scrub = mu_floor_slide * eta_meas * D.M * D.g * w_wheel / 2.0
    Omega_ss_theory = (term_drive - term_coulomb - term_caster - term_scrub) / D.b_rot

    # r5 確定（2026-08-10 教授承認）: τ3 の期待値は静的合成 Izz（パラレルアクシス、compute_izz_breakdown）を
    # 常に使用する。パルス法同定 I_zz_meas は削除せず算出・記録は続けるが「参考記録」に格下げし
    # 合否判定からは外す（スクラブのクーロン抵抗がパルス応答を鈍らせ、I を過大評価する系統誤差が
    # あるため。note_004_r4_diagnostics.py の切り分けで確認済み）。
    model_tmp = mujoco.MjModel.from_xml_path(open_xml)
    I_zz_static = compute_izz_breakdown(model_tmp)['total_izz']
    I_eff_theory = I_zz_static + wheel_term
    tau3_theory = I_eff_theory / D.b_rot
    calib_note = "τ3期待値は静的合成Izzを使用（r5確定）。I_zz_meas(パルス法)は参考記録のみ"

    tail = max(1, len(omega_z) // 20)
    omega_wheel_ss = float(np.mean((np.abs(logb['qvel_l'][-tail:]) + np.abs(logb['qvel_r'][-tail:])) / 2.0))
    Omega_ss_for_L = float(np.mean(np.abs(omega_z[-tail:])))
    L_inf = 2 * D.r * omega_wheel_ss / Omega_ss_for_L if Omega_ss_for_L > 0 else float('nan')

    dx = float(logb['x'][-1] - logb['x'][0])
    dy = float(logb['y'][-1] - logb['y'][0])
    com_disp = float(np.hypot(dx, dy))

    # r5 確定（2026-08-10 教授承認）: CoM 変位判定を「純ピボット（旋回中心=車軸上）なら変位ゼロ」から
    # 「旋回中心=車軸上、CG は車軸前方 x_cg だけオフセットしている設計（r4 の3点接地化・
    # CG前方配置）のため、CG 自体は半径 |x_cg| で公転する」という設計帰結に合わせて改訂。
    # x_cg はモデルから実行時導出（mj_forward の運動学解決のみ、compute_tipover_margin と同じ手法）。
    model_cg = mujoco.MjModel.from_xml_path(open_xml)
    data_cg = mujoco.MjData(model_cg)
    h_cg = get_handles(model_cg)
    data_cg.qpos[h_cg.root_qpos_adr:h_cg.root_qpos_adr + 3] = [0.0, 0.0, 0.05]
    data_cg.qpos[h_cg.root_qpos_adr + 3:h_cg.root_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model_cg, data_cg)
    x_cg = float(data_cg.subtree_com[h_cg.mouse_body_id][0] - data_cg.qpos[h_cg.root_qpos_adr])
    com_disp_limit = 2.0 * abs(x_cg) + 0.002

    ok_L = np.isfinite(L_inf) and abs(L_inf - params.tread) / params.tread <= 0.05
    checks.append(dict(name="tread_L_inf", expected=params.tread, actual=L_inf, tol="5%", ok=ok_L))

    ok_tau3 = np.isfinite(tau3_meas) and abs(tau3_meas - tau3_theory) / tau3_theory <= 0.20
    checks.append(dict(name="tau3", expected=tau3_theory, actual=tau3_meas, tol="20%", ok=ok_tau3))

    ok_omega3 = abs(Omega_ss_meas - Omega_ss_theory) / abs(Omega_ss_theory) <= 0.15 if Omega_ss_theory != 0 else False
    checks.append(dict(name="Omega_ss", expected=Omega_ss_theory, actual=Omega_ss_meas, tol="15%", ok=ok_omega3))

    ok_pivot = com_disp <= com_disp_limit
    checks.append(dict(name="com_displacement_pure_pivot", expected=f"<=2*|x_cg|+2mm={com_disp_limit*1000:.3f}mm",
                        actual=com_disp, tol="-", ok=ok_pivot, note=f"x_cg={x_cg*1000:.3f}mm(r5確定)"))

    qacc_ok, qacc_msg = tracker.check(strict_limit=QACC_STRICT)
    checks.append(dict(name="qacc_monitor_strict", expected="OK", actual=qacc_msg, tol="strict", ok=qacc_ok))

    CALIB['test3_Izz_meas'] = I_zz_meas
    CALIB['test3_I_eff_meas'] = I_eff_meas
    CALIB['test3_Izz_static'] = I_zz_static
    CALIB['test3_I_eff_theory'] = I_eff_theory
    CALIB['test3_eta_meas'] = eta_meas
    ok, msg = _finish("Test3_pivot_turn", checks, calib=dict(
        I_zz_meas_pulse_reference=I_zz_meas, I_eff_meas_pulse_reference=I_eff_meas,
        I_zz_static=I_zz_static, I_eff_theory=I_eff_theory, wheel_term=wheel_term, L_inf=L_inf,
        Omega_ss_meas=Omega_ss_meas, tau3_meas=tau3_meas, eta_meas=eta_meas,
        mu_floor_slide=mu_floor_slide, w_wheel=w_wheel, term_scrub=term_scrub,
        x_cg=x_cg, com_disp_limit=com_disp_limit, note=calib_note))
    assert ok, msg


# ======================================================================
# 静的解析ヘルパー（Izz パラレルアクシス分解・転倒余裕。mj_forward 1回のみ、
# 時間積分なしの意味で「シミュレーション実行なし」の静的判定として扱う）
# ======================================================================
def collect_subtree(model, root_id):
    ids = []
    for b in range(model.nbody):
        bp = b
        while True:
            if bp == root_id:
                ids.append(b)
                break
            if bp == 0:
                break
            bp = model.body_parentid[bp]
    return ids


def compute_izz_breakdown(model):
    """パラレルアクシス合成により、mouse サブツリー全体の CoM まわりの Izz と、
    車輪+モータボディの寄与率を計算する（計画書 §7 S1、C5 の検証）。"""
    data = mujoco.MjData(model)
    h = get_handles(model)
    data.qpos[h.root_qpos_adr:h.root_qpos_adr + 3] = [0.0, 0.0, 0.05]
    data.qpos[h.root_qpos_adr + 3:h.root_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[model.jnt_qposadr[h.left_wheel_joint_id]] = 0.0
    data.qpos[model.jnt_qposadr[h.right_wheel_joint_id]] = 0.0
    mujoco.mj_forward(model, data)

    subtree_ids = collect_subtree(model, h.mouse_body_id)
    com = data.subtree_com[h.mouse_body_id].copy()

    motorbox1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'motorbox1')
    motorbox2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'motorbox2')
    wheel_motor_bodies = {h.left_wheel_body_id, h.right_wheel_body_id, motorbox1_id, motorbox2_id}

    total_izz = 0.0
    wheel_motor_izz = 0.0
    for b in subtree_ids:
        m = float(model.body_mass[b])
        if m <= 0:
            continue
        xi = data.xipos[b]
        ximat = data.ximat[b].reshape(3, 3)
        Idiag = model.body_inertia[b]
        I_world = ximat @ np.diag(Idiag) @ ximat.T
        izz_own = float(I_world[2, 2])
        dx = float(xi[0] - com[0])
        dy = float(xi[1] - com[1])
        izz_body = izz_own + m * (dx ** 2 + dy ** 2)
        total_izz += izz_body
        if b in wheel_motor_bodies:
            wheel_motor_izz += izz_body

    frac = wheel_motor_izz / total_izz if total_izz > 0 else float('nan')
    return dict(total_izz=total_izz, wheel_motor_izz=wheel_motor_izz, fraction=frac, com=com.tolist())


def compute_tipover_margin(params, open_xml, D):
    """静的転倒余裕（計画書 §6 Test8 静的部・§7 S4）。x_cg, z_cg は mj_forward の
    運動学解決のみで求め、a_avail の摩擦項のみ短い静定シミュレーションで実測する。"""
    model = mujoco.MjModel.from_xml_path(open_xml)
    data = mujoco.MjData(model)
    h = get_handles(model)
    data.qpos[h.root_qpos_adr:h.root_qpos_adr + 3] = [0.0, 0.0, 0.05]
    data.qpos[h.root_qpos_adr + 3:h.root_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    com = data.subtree_com[h.mouse_body_id]
    root_pos = data.qpos[h.root_qpos_adr:h.root_qpos_adr + 3]
    x_cg = float(com[0] - root_pos[0])
    z_cg = float(com[2] - root_pos[2])
    x_f = float(model.geom_pos[h.caster_front_geom_id][0])
    x_r = float(model.geom_pos[h.caster_back_geom_id][0])
    g = float(abs(model.opt.gravity[2]))
    a_crit = g * (x_f - x_cg) / z_cg

    sim2 = MouseSim(open_xml, params=params)
    h2 = get_handles(sim2.model)
    sim2.full_reset(cell=(0, 0), heading_deg=0)
    settle_contact(sim2.model, sim2.data, h2)
    contacts = measure_contacts_avg(sim2.model, sim2.data, h2, ctrl=(0.0, 0.0), n_samples=100)
    mu_wheel = contacts.mu_wheel if contacts.mu_wheel is not None else 1.0
    mu_caster = contacts.mu_caster if contacts.mu_caster is not None else parse_mu_caster(params)

    a_avail_motor = 2 * D.N * D.Kt * params.voltage_limit / (D.R * D.r * D.M_eff)
    a_avail_traction = (mu_wheel * contacts.N_wheel + mu_caster * contacts.N_caster) / D.M
    a_avail = min(a_avail_motor, a_avail_traction)
    margin = a_crit / a_avail if a_avail > 0 else float('nan')

    return dict(x_cg=x_cg, z_cg=z_cg, x_f=x_f, x_r=x_r, a_crit=a_crit, a_avail=a_avail,
                a_avail_motor=a_avail_motor, a_avail_traction=a_avail_traction, margin=margin,
                N_wheel=contacts.N_wheel, N_caster=contacts.N_caster, N_front=contacts.N_front,
                N_back=contacts.N_back, mu_wheel=mu_wheel, mu_caster=mu_caster)


# ======================================================================
# 治具 XML ビルダー（Test5/Test6/S2/接触健全性 用。/tmp 配下に生成し
# mouse/, assets/ は一切変更しない）
# ======================================================================
def _make_corridor(length):
    """1 セル幅・length セル長の直線廊下（全周壁のみ、内部壁なし）を作る。"""
    v_walls = np.zeros((length + 1, 1), dtype=int)
    h_walls = np.zeros((length, 2), dtype=int)
    v_walls[0, 0] = 1
    v_walls[length, 0] = 1
    h_walls[:, 0] = 1
    h_walls[:, 1] = 1
    return v_walls, h_walls


def build_sensor_wall_jig(params, out_path, wall_pos=None, wall_normal=None):
    """床 + ロボット + （任意）1 枚の無限平面壁からなる最小治具。
    壁は D ごとに再生成する方式（供補足仕様書の推奨どおり: 再生成が単純で確実）。"""
    mujoco_el = _new_mujoco_root('sensor_jig', params)
    worldbody = ET.SubElement(mujoco_el, 'worldbody')
    ET.SubElement(worldbody, 'geom', {
        'name': 'floor', 'type': 'plane', 'material': 'mat_floor',
        'pos': '0 0 0', 'size': '9 9 0.1', 'friction': params.floor_friction,
    })
    ET.SubElement(worldbody, 'light', {'diffuse': '.5 .5 .5', 'pos': '0 0 3', 'dir': '0 0 -1'})
    if wall_pos is not None:
        ET.SubElement(worldbody, 'geom', {
            'name': 'sensor_wall', 'type': 'plane', 'size': '2 2 0.1',
            'pos': f'{wall_pos[0]:.8f} {wall_pos[1]:.8f} {wall_pos[2]:.8f}',
            'zaxis': f'{wall_normal[0]:.8f} {wall_normal[1]:.8f} {wall_normal[2]:.8f}',
            'material': 'mat_wall', 'friction': params.wall_body_friction,
        })
    actuator = ET.SubElement(mujoco_el, 'actuator')
    sensor = ET.SubElement(mujoco_el, 'sensor')
    contact = ET.SubElement(mujoco_el, 'contact')
    add_micromouse_v2(worldbody, actuator, sensor, contact, pos='0 0 0.002', euler='0 0 0', params=params)
    return _write_xml(mujoco_el, out_path)


def measure_wall_contact_friction(params, xml_dir):
    """機体外殻(mein_body1)-壁の実効接触摩擦係数を、1 セル廊下治具に機体を
    わずかに食い込ませて直接検査する（計画書 §7 S2、U4 是正確認）。"""
    v_walls, h_walls = _make_corridor(1)
    xml_path = str(Path(xml_dir) / 's2_wall_jig.xml')
    build_maze_robot_xml(v_walls, h_walls, xml_path, model_name='s2_wall_jig',
                          center_goal=False, params=params)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    h = get_handles(model)

    wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'v_wall_1_0')
    wall_face_x = float(model.geom_pos[wall_id][0]) - float(model.geom_size[wall_id][0])
    mein1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'mein_body1')
    front_x_local = float(model.geom_pos[mein1_id][0]) + float(model.geom_size[mein1_id][0])

    y_center = params.cell_size / 2.0
    x_robot = wall_face_x - front_x_local + 0.001  # 前面を壁面より1mm奥へ食い込ませ確実に接触させる
    data.qpos[h.root_qpos_adr:h.root_qpos_adr + 3] = [x_robot, y_center, 0.002]
    data.qpos[h.root_qpos_adr + 3:h.root_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    for i in range(data.ncon):
        c = data.contact[i]
        if wall_id in (c.geom1, c.geom2):
            return float(c.friction[0])
    return None


def build_impact_jig(params, out_path, mass, wall_x, wall_half_y=0.1, wall_half_z=0.05):
    """接触健全性試験用の最小衝突治具: 固定壁(box) + 1自由度(slide)の質点。
    solref/timestep はロボット本体と同一値（params 由来）を使う。"""
    mujoco_el = ET.Element('mujoco', {'model': 'impact_jig'})
    ET.SubElement(mujoco_el, 'compiler', {'angle': 'degree', 'coordinate': 'local'})
    ET.SubElement(mujoco_el, 'option', {'timestep': str(params.physics_dt), 'gravity': '0 0 0'})
    default = ET.SubElement(mujoco_el, 'default')
    ET.SubElement(default, 'geom', {'solref': params.solref})
    worldbody = ET.SubElement(mujoco_el, 'worldbody')
    ET.SubElement(worldbody, 'geom', {
        'name': 'wall', 'type': 'box',
        'size': f'0.006 {wall_half_y} {wall_half_z}', 'pos': f'{wall_x} 0 0',
        'friction': params.wall_body_friction,
    })
    body = ET.SubElement(worldbody, 'body', {'name': 'impactor', 'pos': '0 0 0'})
    ET.SubElement(body, 'joint', {'name': 'slide_x', 'type': 'slide', 'axis': '1 0 0', 'damping': '0'})
    ET.SubElement(body, 'geom', {
        'name': 'impactor_geom', 'type': 'box', 'size': '0.02 0.02 0.02',
        'friction': params.wall_body_friction, 'mass': str(mass),
    })
    return _write_xml(mujoco_el, out_path)


# ======================================================================
# Test 4: 全力直進加速 — 二相モデル（計画書 §6 Test4。qacc は (a)のみ=loose）
# ======================================================================
def test_04_full_throttle(params, open_xml, D):
    checks = []
    tracker = QaccTracker()
    sim = MouseSim(open_xml, params=params)
    h = get_handles(sim.model)

    sim.full_reset(cell=(0, 0), heading_deg=0)
    settled, _ = settle_contact(sim.model, sim.data, h)
    contacts = measure_contacts_avg(sim.model, sim.data, h, ctrl=(0.0, 0.0), n_samples=100)
    N_w = contacts.N_wheel
    N_c = contacts.N_caster
    mu_wheel = contacts.mu_wheel if contacts.mu_wheel is not None else 1.0
    mu_caster = contacts.mu_caster if contacts.mu_caster is not None else parse_mu_caster(params)

    checks.append(dict(name="settled_before_excitation", expected=True, actual=settled, tol="-", ok=settled))

    Vmax = params.voltage_limit
    t0 = sim.data.time
    log = run_logged(sim.model, sim.data, duration=2.5, ctrl_fn=lambda t: (Vmax, Vmax), tracker=tracker,
                      log_every=2, t0=t0, voltage_limit=Vmax)

    # 全開巡航（電圧律速相）中のキャスター荷重を、同じ電圧を維持したまま追加で時間平均
    # 測定する（相1のスリップ律速はほぼ静止直後の静的値、相2の巡航は動的値を使う。
    # 4点接地の不静定性由来のチャタリングの発見に基づく実装。報告書参照）
    contacts_cruise = measure_contacts_avg(sim.model, sim.data, h, ctrl=(Vmax, Vmax), n_samples=200)
    N_c_cruise = contacts_cruise.N_caster

    t = log['t']
    v = log['v_fwd']
    omega_avg = (log['qvel_l'] + log['qvel_r']) / 2.0
    slip_signal = D.r * omega_avg - v

    a1_theory = (mu_wheel * N_w - mu_caster * N_c) / D.M
    v_ss_theory = (2 * (D.gainprm0 / D.r) * Vmax - 2 * (D.tau_c / D.r) - mu_caster * N_c_cruise) / D.c_eff
    v1_theory = (2 * (D.gainprm0 / D.r) * Vmax - 2 * (D.tau_c / D.r) - mu_caster * N_c_cruise - mu_wheel * N_w) / D.c_eff
    t1_theory = v1_theory / a1_theory if a1_theory > 0 else float('nan')

    mask1 = (t >= 0.02) & (t <= 0.15)
    if np.sum(mask1) >= 3:
        a1_meas, _, r2_a1 = linreg(t[mask1], v[mask1])
    else:
        a1_meas, r2_a1 = float('nan'), float('nan')

    # スリップ信号は t=0 で omega=v=0 のため恒等的に 0（=閾値以下）から始まる。
    # 「閾値を初めて下回る点」を単純探索すると t=0 を誤検出するバグがあったため、
    # スリップ信号のピーク（相1の最中）以降で初めて閾値を下回る点を相転移とする
    # （テスト側の明白な実装バグとして修正）。
    thresh = 0.05 * v1_theory
    peak_idx = int(np.argmax(slip_signal))
    tail_region = slip_signal[peak_idx:]
    below = tail_region <= thresh
    if np.any(below):
        idx1 = peak_idx + int(np.argmax(below))
        v1_meas = float(v[idx1])
        t1_meas = float(t[idx1])
    else:
        v1_meas, t1_meas = float('nan'), float('nan')

    tail = max(1, len(v) // 20)
    v_ss_meas = float(np.mean(v[-tail:]))

    idx_1s = int(np.argmax(t >= 1.0)) if np.any(t >= 1.0) else len(t) - 1
    x_meas_1s = float(log['x'][idx_1s] - log['x'][0])

    tau_v = D.tau_v
    if np.isfinite(t1_theory) and t1_theory < 1.0:
        x1_theory = 0.5 * a1_theory * t1_theory ** 2
        dt2 = 1.0 - t1_theory
        x_theory_1s = x1_theory + v_ss_theory * dt2 + (v_ss_theory - v1_theory) * tau_v * (np.exp(-dt2 / tau_v) - 1)
    else:
        x_theory_1s = 0.5 * a1_theory * 1.0 ** 2

    ok_a1 = np.isfinite(a1_meas) and abs(a1_meas - a1_theory) / a1_theory <= 0.10
    checks.append(dict(name="a1_slip_phase", expected=a1_theory, actual=a1_meas, tol="10%", ok=ok_a1))

    ok_v1 = np.isfinite(v1_meas) and abs(v1_meas - v1_theory) / v1_theory <= 0.20
    checks.append(dict(name="v1_transition", expected=v1_theory, actual=v1_meas, tol="20%", ok=ok_v1))

    ok_vss = abs(v_ss_meas - v_ss_theory) / v_ss_theory <= 0.10
    checks.append(dict(name="v_ss", expected=v_ss_theory, actual=v_ss_meas, tol="10%", ok=ok_vss))

    ok_x1s = abs(x_meas_1s - x_theory_1s) / x_theory_1s <= 0.15 if x_theory_1s != 0 else False
    checks.append(dict(name="x_at_1s", expected=x_theory_1s, actual=x_meas_1s, tol="15%", ok=ok_x1s))

    ok_slip1 = bool(np.isfinite(t1_meas) and np.any(slip_signal[t < t1_meas] > 0.05 * v1_theory)) if np.isfinite(t1_meas) else True
    checks.append(dict(name="slip_phase1_present", expected=True, actual=ok_slip1, tol="-", ok=ok_slip1))

    mask_phase2 = t > 1.5 * t1_theory if np.isfinite(t1_theory) else np.zeros_like(t, dtype=bool)
    if np.any(mask_phase2):
        slip_phase2 = float(np.mean(np.abs(slip_signal[mask_phase2])))
        ok_slip2 = slip_phase2 < 0.02 * v_ss_theory
    else:
        slip_phase2, ok_slip2 = float('nan'), False
    checks.append(dict(name="slip_phase2_negligible", expected=f"<{0.02*v_ss_theory:.4f}", actual=slip_phase2,
                        tol="-", ok=ok_slip2))

    qacc_ok, qacc_msg = tracker.check(strict_limit=None)
    checks.append(dict(name="qacc_monitor_loose", expected="OK", actual=qacc_msg, tol="loose", ok=qacc_ok))

    # 下流（Test5/Test7）は steady cruise 条件を扱うため、N_caster は巡航中の動的平均値を渡す。
    # N_wheel はスリップ律速相（相1）を支配する静的値のまま渡す（相1は運動開始直後で
    # チャタリングが顕在化する前のため静的値が妥当。報告書「計画書との矛盾・懸念」参照）
    CALIB['N_wheel_meas'] = N_w
    CALIB['N_wheel_static'] = N_w
    CALIB['N_caster_meas'] = N_c_cruise
    CALIB['N_caster_static'] = CALIB.get('N_caster_static', N_c)
    CALIB['mu_wheel_meas'] = mu_wheel
    CALIB['mu_caster_meas_t4'] = mu_caster
    CALIB['test4_t1_meas'] = t1_meas
    CALIB['test4_v_ss_theory'] = v_ss_theory
    CALIB['test4_log'] = log
    CALIB['test4_a1_theory'] = a1_theory

    ok, msg = _finish("Test4_full_throttle", checks, calib=dict(
        N_w=N_w, N_c=N_c, mu_wheel=mu_wheel, mu_caster=mu_caster, a1_theory=a1_theory,
        v1_theory=v1_theory, t1_theory=t1_theory, v_ss_theory=v_ss_theory, x_theory_1s=x_theory_1s))
    assert ok, msg


# ======================================================================
# Test 5: 定常円旋回 — スキッド限界（計画書 §6 Test5）
# ======================================================================
def test_05_steady_turn(params, open_xml, D):
    checks = []
    tracker = QaccTracker()
    sim = MouseSim(open_xml, params=params)
    h = get_handles(sim.model)

    N_w = CALIB.get('N_wheel_meas')
    N_c = CALIB.get('N_caster_meas')
    mu_wheel = CALIB.get('mu_wheel_meas', 1.0)
    mu_caster = CALIB.get('mu_caster_meas_t4', parse_mu_caster(params))

    sim.full_reset(cell=(0, 0), heading_deg=0)
    settled, _ = settle_contact(sim.model, sim.data, h, tracker=tracker)
    if N_w is None or N_c is None:
        contacts = measure_contacts_avg(sim.model, sim.data, h, ctrl=(0.0, 0.0), n_samples=100)
        N_w, N_c = contacts.N_wheel, contacts.N_caster

    v_ref = 0.8       # 試験手順の設定値（計画書 §6 Test5、期待値ではない）
    Kp_v = 1.0         # 速度 P ゲイン（古典制御・学習なし。試験治具パラメータ）
    Vff = (v_ref * D.c_eff + 2 * D.tau_c / D.r + mu_caster * N_c) * D.r / (2 * D.gainprm0)

    # r5 確定（2026-08-10 教授承認）: ヨーレート制御を、フィードフォワード主体（r4実装）から
    # 教授指示の古典 PI 閉ループへ変更する（ΔV = Kp*(Ω_ref-Ω) + Ki*∫、Ω_ref = v_ref*κ）。
    # プラント（差動電圧 ΔV → ヨーレート Ω）を一次遅れとしてモデル化する:
    #   Ω(s)/ΔV(s) = K_plant/(τ_plant s + 1)、K_plant = K_drive/b_rot、τ_plant = I_eff_yaw/b_rot
    # （K_drive: 差動電圧1Vあたりのヨートルク[= Test3のterm_drive/V0と同型]。I_eff_yaw は
    # Test3(r5)と同じ静的合成 Izz ベースで実行時導出。r4 実装の純 P フィードバックは b_rot が
    # 極小[r3コアレス化で低慣性・低粘性]のためプラントゲイン K_plant が巨大になり、素朴な
    # ゲインで容易に不安定化することを実測確認済み[Kp_yaw=0.05でqacc発散]。本設計はプラント
    # 定数 b_rot, I_eff で正規化した極配置PIのため、この問題を構造的に回避する）。
    # 極配置: 閉ループを 2 次遅れ ζ=1.0（臨界制動、オーバーシュートなし）・
    # ωn = k_bandwidth/τ_plant（k_bandwidth=5、開ループプラントの5倍速応答）に配置し、
    #   Ki = ωn^2 * τ_plant / K_plant、Kp = (2*ζ*ωn*τ_plant - 1) / K_plant
    # で導出する（ζ, k_bandwidth は試験治具の設計選定値。数値はここに明記しCALIBにも記録）。
    K_drive = 2 * (D.gainprm0 / D.r) * (D.L / 2.0)
    izz_model = mujoco.MjModel.from_xml_path(open_xml)
    I_eff_yaw = (compute_izz_breakdown(izz_model)['total_izz']
                 + 2 * (D.I_w + D.armature) * (D.L / (2 * D.r)) ** 2)
    tau_plant = I_eff_yaw / D.b_rot
    K_plant = K_drive / D.b_rot
    zeta_design = 1.0     # 臨界制動（試験治具設計値）
    k_bandwidth = 5.0     # 閉ループ帯域 = 開ループプラントの5倍（試験治具設計値）
    omega_n = k_bandwidth / tau_plant
    Ki_yaw = omega_n ** 2 * tau_plant / K_plant
    Kp_yaw = (2.0 * zeta_design * omega_n * tau_plant - 1.0) / K_plant
    ramp_rate = 6.0  # kappa/s（曲率指令の急変回避。試験治具パラメータ、物理期待値ではない）
    dt_phys = float(sim.model.opt.timestep)

    # r5 確定（2026-08-10 教授承認）: 各κ段の整定時間 ≥ 4*τ_v（D.tau_v で実行時導出 ≈0.124s → 0.5s
    # 以上）。マージンを見て 4.5*τ_v を採用する。
    hold_dur = 4.5 * D.tau_v

    kappa_state = {'target': 0.0, 'ramp': 0.0}
    yaw_state = {'integral': 0.0}

    def make_warmup_ctrl(ramp_dur=0.6):
        def ctrl_fn(t):
            v_world = sim.data.qvel[h.root_dof_adr:h.root_dof_adr + 3]
            xmat = sim.data.xmat[h.mouse_body_id].reshape(3, 3)
            v_meas = float(np.dot(v_world, xmat[:, 0]))
            frac = min(1.0, t / ramp_dur)
            Vbase = frac * Vff + Kp_v * (frac * v_ref - v_meas)
            return Vbase, Vbase
        return ctrl_fn

    def make_ctrl():
        def ctrl_fn(t):
            v_world = sim.data.qvel[h.root_dof_adr:h.root_dof_adr + 3]
            xmat = sim.data.xmat[h.mouse_body_id].reshape(3, 3)
            v_meas = float(np.dot(v_world, xmat[:, 0]))
            omega_meas = float(sim.data.sensordata[11])
            step = ramp_rate * dt_phys
            if kappa_state['ramp'] < kappa_state['target']:
                kappa_state['ramp'] = min(kappa_state['target'], kappa_state['ramp'] + step)
            elif kappa_state['ramp'] > kappa_state['target']:
                kappa_state['ramp'] = max(kappa_state['target'], kappa_state['ramp'] - step)
            omega_ref = v_ref * kappa_state['ramp']
            Vbase = Vff + Kp_v * (v_ref - v_meas)
            err_yaw = omega_ref - omega_meas
            yaw_state['integral'] += err_yaw * dt_phys
            # アンチワインドアップ（積分クランプ）: 積分項単独で ±voltage_limit を超えて
            # 指令できないよう積分値を制限する。試験治具の標準的な古典PI実装作法であり、
            # 数値ハードコード禁止の対象である「物理期待値」ではなく制御則の健全性条件。
            integral_bound = params.voltage_limit / Ki_yaw if Ki_yaw > 0 else 0.0
            yaw_state['integral'] = float(np.clip(yaw_state['integral'], -integral_bound, integral_bound))
            Vdiff = Kp_yaw * err_yaw + Ki_yaw * yaw_state['integral']
            # 符号規約（実測確認済み・テスト側の実装バグ修正）: 本モデルでは
            # 右輪電圧>左輪電圧 が正の omega_z（gyro/世界系）を生む（左右逆ではない）。
            # r4実装は Vbase+Vdiff を左輪に加えており符号が逆だった（Kff_yawが正の
            # フィードフォワードとして常に逆方向に効いていたため、r4のTest5不合格の
            # 一因だったと推定される）。Vdiff は右輪に加える。
            return Vbase - Vdiff, Vbase + Vdiff
        return ctrl_fn

    # 助走: 速度を v_ref まで滑らかに立ち上げる
    t0 = sim.data.time
    run_logged(sim.model, sim.data, duration=1.2, ctrl_fn=make_warmup_ctrl(), tracker=tracker, log_every=2,
               t0=t0, voltage_limit=params.voltage_limit)

    kappas = np.linspace(0.0, 20.0, 15)
    omega_ref_level = v_ref * kappas
    a_lat_theory_linear = v_ref ** 2 * kappas
    a_lat_max_theory = (mu_wheel * N_w + mu_caster * N_c) / D.M
    # 直線域マスクは kappa（既知の等差数列）のみに依存し、スイープ実行前に確定する。
    # kappa は昇順のため lin_mask は必ず先頭側の連続区間（プレフィックス）になる。
    lin_mask = a_lat_theory_linear <= 0.5 * a_lat_max_theory
    n_lin = int(np.sum(lin_mask))

    # r5 確定（2026-08-10 教授承認）: 破綻検出に第3条件を追加する。
    # (i) ヨーレートが Ω_ref から 50% 以上乖離、(ii) 横滑り角 β がテール窓で発散
    # （線形回帰スロープ |dβ/dt| が閾値超）、に加えて
    # (iii) a_lat が直線域フィット（lin_mask 区間の直線）のトレンドから 30% 以上急落する
    # 状態が 2 段連続、のいずれかで機械判定する。(iii) は「Ω は正しく追従し続けるが
    # a_lat が実質ゼロへ崩壊する」差動駆動特有のスピンアウトモード（先行レビューで診断済み:
    # Ω偏差<1%を維持したまま a_lat が 8.6→0.07 m/s^2 に崩壊し、qacc 7〜9e4 の接触過渡を
    # 伴う。(i)(ii) だけでは検出が1〜4段遅れていた）を捕捉するために追加した。
    # 破綻を検出した時点でκランプを終了し、以降の高κ段はシミュレートしない。
    BETA_DIVERGE_THRESH = 0.5   # rad/s（テール窓終端でβがこの速さ以上で成長=未整定とみなす治具閾値）
    A_LAT_DROP_THRESH = 0.30    # 直線トレンドからの急落率閾値（教授承認値）

    a_lat_meas = []
    omega_meas_level = []
    beta_slope_level = []
    qacc_pre_step_snapshots = []  # [i] = tracker.max_abs（κ段 i 実行直前のスナップショット）
    slope_lin, intercept_lin, r2_lin = float('nan'), float('nan'), float('nan')
    prev_drop = False
    breakdown_idx = None
    breakdown_reason = None

    for i, kappa in enumerate(kappas):
        qacc_pre_step_snapshots.append(tracker.max_abs)
        kappa_state['target'] = float(kappa)
        # 各κ段の開始時に積分状態をリセットする（各段を独立な新しい定常動作点への
        # ステップ応答として評価する設計。段をまたいだ積分ワインドアップの蓄積を防ぐ）。
        yaw_state['integral'] = 0.0
        t0k = sim.data.time
        logk = run_logged(sim.model, sim.data, duration=hold_dur, ctrl_fn=make_ctrl(), tracker=tracker,
                           log_every=2, t0=t0k, voltage_limit=params.voltage_limit)
        # テール窓 = ホールド末尾40%（hold_dur を 4*tau_v 基準に変更したため絶対時間でなく
        # 割合窓とし、hold_dur の値によらず妥当なマージンを確保する）
        n_tail = max(1, int(round(0.4 * len(logk['t']))))
        # 中央値を使用（4点接地の不静定性由来の散発的チャタリングスパイクに対して
        # 平均よりロバスト。報告書「計画書との矛盾・懸念」参照）
        a_lat = float(np.median(logk['sensordata'][-n_tail:, 7]))
        a_lat_meas.append(a_lat)
        omega_tail = float(np.median(logk['sensordata'][-n_tail:, 11]))
        omega_meas_level.append(omega_tail)
        beta_tail = np.arctan2(logk['v_lat'][-n_tail:], logk['v_fwd'][-n_tail:])
        if n_tail >= 3:
            beta_slope, _, _ = linreg(logk['t'][-n_tail:], beta_tail)
        else:
            beta_slope = 0.0
        beta_slope_level.append(float(beta_slope))

        if n_lin >= 3 and (i + 1) == n_lin:
            # 直線域点が出揃った直後にトレンド直線を1回だけフィットする
            slope_lin, intercept_lin, r2_lin = linreg(
                kappas[:n_lin], np.abs(np.array(a_lat_meas[:n_lin])))

        cond1 = cond2 = False
        if i >= 1:
            denom = abs(omega_ref_level[i])
            dev = abs(omega_meas_level[i] - omega_ref_level[i]) / denom if denom > 0 else 0.0
            cond1 = dev >= 0.5
            cond2 = abs(beta_slope_level[i]) > BETA_DIVERGE_THRESH

        cur_drop = False
        if i >= n_lin and np.isfinite(slope_lin):
            trend = slope_lin * kappa + intercept_lin
            if trend > 0:
                cur_drop = (trend - abs(a_lat_meas[i])) / trend >= A_LAT_DROP_THRESH
        cond3 = cur_drop and prev_drop
        prev_drop = cur_drop

        if cond1 or cond2 or cond3:
            breakdown_idx = i
            breakdown_reason = ('omega_deviation' if cond1 else
                                 'beta_divergence' if cond2 else
                                 'a_lat_trend_drop_2consecutive')
            break

    a_lat_meas = np.array(a_lat_meas)
    omega_meas_level = np.array(omega_meas_level)
    beta_slope_level = np.array(beta_slope_level)
    n_run = len(a_lat_meas)

    if breakdown_idx is not None:
        back = 2 if breakdown_reason == 'a_lat_trend_drop_2consecutive' else 1
        plateau_src_idx = max(0, breakdown_idx - back)
        plateau_meas = float(abs(a_lat_meas[plateau_src_idx]))
        qacc_strict_reference = qacc_pre_step_snapshots[plateau_src_idx + 1]
        plateau_note = (f"破綻検出[{breakdown_reason}]: kappa[{breakdown_idx}]={kappas[breakdown_idx]:.2f}rad/m。"
                         f"直前(idx={plateau_src_idx})の点を『破綻直前の最大持続a_lat』に採用。"
                         f"以降のκ段はシミュレート打ち切り")
    else:
        plateau_meas = float(abs(a_lat_meas[-1]))
        qacc_strict_reference = tracker.max_abs
        plateau_note = "sweep内(kappa<=20)で破綻未検出。最終kappa点を『最大持続a_lat』として採用"

    ratio_plateau = plateau_meas / a_lat_max_theory if a_lat_max_theory > 0 else float('nan')
    ok_plateau = np.isfinite(ratio_plateau) and (0.60 <= ratio_plateau <= 1.10)
    checks.append(dict(name="plateau_a_lat_max", expected=f"60%-110% of {a_lat_max_theory:.3f}(上限参考値)",
                        actual=plateau_meas, tol="60%-110%", ok=ok_plateau, note=plateau_note))

    ok_lin_r2 = np.isfinite(r2_lin) and r2_lin >= 0.95
    checks.append(dict(name="linear_region_r2", expected=">=0.95", actual=r2_lin, tol="-", ok=ok_lin_r2))

    idxs_lin = np.where(lin_mask)[0]
    idxs_lin = idxs_lin[(idxs_lin > 0) & (idxs_lin < n_run)][:3]
    pt_ok = True
    for i in idxs_lin:
        expect_pt = a_lat_theory_linear[i]
        err = abs(abs(a_lat_meas[i]) - expect_pt) / expect_pt if expect_pt > 0 else 0.0
        pt_ok = pt_ok and (err <= 0.10)
    checks.append(dict(name="initial_points_linear_fit", expected="±10%", actual=f"n={len(idxs_lin)}", tol="-",
                        ok=pt_ok))

    g = D.g
    plateau_G = plateau_meas / g
    ok_range = 0.5 <= plateau_G <= 1.5
    checks.append(dict(name="plateau_within_benchmark_range_G", expected="[0.5,1.5]G", actual=plateau_G, tol="-",
                        ok=ok_range))

    # r5 確定（2026-08-10 教授承認）: strict qacc 判定は破綻前区間（qacc_strict_reference。
    # plateau 採用点までの累積最大値）にのみ適用する。NaN/Inf・QACC_LOOSE の共通監視
    # （計画書§6共通規約(v)(a)）は破綻段を含む全区間に適用する（全テスト共通の安全網であり、
    # Test5 固有のスコープ限定の対象外）。
    if tracker.bad:
        qacc_ok, qacc_msg = False, "NaN/Inf が qacc/qvel/qpos に検出された"
    elif tracker.max_abs >= QACC_LOOSE:
        qacc_ok, qacc_msg = False, f"max|qacc|={tracker.max_abs:.3e} >= QACC_LOOSE({QACC_LOOSE:.0e})"
    else:
        qacc_ok = bool(qacc_strict_reference < QACC_STRICT)
        qacc_msg = f"max|qacc|(破綻前区間)={qacc_strict_reference:.3e} rad/s^2 (全区間max={tracker.max_abs:.3e})"
    checks.append(dict(name="qacc_monitor_strict", expected="OK", actual=qacc_msg, tol="strict", ok=qacc_ok))

    ok, msg = _finish("Test5_steady_turn", checks, calib=dict(
        a_lat_max_theory=a_lat_max_theory, plateau_meas=plateau_meas, plateau_G=plateau_G,
        breakdown_idx=breakdown_idx, breakdown_reason=breakdown_reason, n_run=n_run, n_lin=n_lin,
        slope_lin=slope_lin, intercept_lin=intercept_lin, r2_lin=r2_lin,
        qacc_strict_reference=qacc_strict_reference, qacc_full_max=tracker.max_abs,
        kappas_run=kappas[:n_run].tolist(), a_lat_meas=a_lat_meas.tolist(),
        omega_meas_level=omega_meas_level.tolist(), beta_slope_level=beta_slope_level.tolist(),
        Kp_yaw=Kp_yaw, Ki_yaw=Ki_yaw, tau_plant=tau_plant, K_plant=K_plant, hold_dur=hold_dur))
    assert ok, msg


# ======================================================================
# Test 6: 全8センサ較正 — 直線性・飽和・座標変換（計画書 §6 Test6）
# ======================================================================
def test_06_sensors(params, open_xml, xml_dir, D):
    checks = []
    calib = {}

    template_path = build_sensor_wall_jig(params, str(Path(xml_dir) / 'sensor_template.xml'))
    model_t = mujoco.MjModel.from_xml_path(template_path)
    data_t = mujoco.MjData(model_t)
    mujoco.mj_forward(model_t, data_t)

    frames = {}
    for s in params.sensors:
        site_id = mujoco.mj_name2id(model_t, mujoco.mjtObj.mjOBJ_SITE, f"site_{s['name']}")
        pos = data_t.site_xpos[site_id].copy()
        xmat = data_t.site_xmat[site_id].reshape(3, 3)
        direction = xmat[:, 2].copy()
        direction = direction / np.linalg.norm(direction)
        frames[s['name']] = (pos, direction)

    D_LIST = [0.01, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.29, 0.30, 0.32, 0.50]
    cutoff = params.sensor_cutoff
    name_to_idx = {s['name']: i for i, s in enumerate(params.sensors)}

    regressions = {}
    for name, (pos, direction) in frames.items():
        idx = name_to_idx[name]
        Ds, meas = [], []
        for Dval in D_LIST:
            wall_pos = pos + Dval * direction
            jig_path = str(Path(xml_dir) / f'sensor_{name}_{Dval:.3f}.xml')
            # MuJoCo の plane 幾何は法線(zaxis)と反対側から来る光線しか検出しないため、
            # 壁面の法線はボアサイト方向の逆向き（センサに向く向き）にする
            # （実測で確認したテスト側の明白な実装バグとして修正）。
            build_sensor_wall_jig(params, jig_path, wall_pos=wall_pos, wall_normal=-direction)
            m = mujoco.MjModel.from_xml_path(jig_path)
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            Ds.append(Dval)
            meas.append(float(d.sensordata[idx]))
        Ds = np.array(Ds)
        meas = np.array(meas)
        below = Ds < cutoff - 1e-9
        above = ~below
        slope, intercept, r2 = linreg(Ds[below], meas[below])
        ok_slope = abs(slope - 1.0) <= 0.01
        ok_intercept = abs(intercept) < 0.0005
        ok_r2 = r2 >= 0.999
        # 【発見事項】計画書は「D>=cutoff で生値が厳密に -cutoff」としているが、実測では
        # 「幾何的に交差する物体が存在する場合は距離に関わらず +cutoff にクリップされる」
        # （D=0.32, 0.50, さらに 5.0 まで試しても sensordata=+0.3。無限平面は光線が必ず
        # どこかで交差するため）。生値が -cutoff になるのは「光線経路上に何も存在しない」
        # 場合のみ（別途 no_hit ジグで直接確認する）。よって本チェックは +cutoff への
        # クリップを期待値とする（テスト側の実装バグ＝計画書想定との相違点として修正。
        # 詳細は報告書「計画書との矛盾・懸念」参照）。
        ok_cutoff = bool(np.all(np.isclose(meas[above], cutoff, atol=1e-6)))
        checks.append(dict(name=f"range_{name}_slope", expected=1.0, actual=slope, tol="±0.01", ok=ok_slope))
        checks.append(dict(name=f"range_{name}_intercept", expected=0.0, actual=intercept, tol="<0.5mm",
                            ok=ok_intercept))
        checks.append(dict(name=f"range_{name}_r2", expected=">=0.999", actual=r2, tol="-", ok=ok_r2))
        checks.append(dict(name=f"range_{name}_cutoff_clip_when_hit_beyond", expected=cutoff,
                            actual=meas[above].tolist(), tol="exact", ok=ok_cutoff))

        if name == 'LF':
            # ジオメトリが何も存在しない場合の真の「ヒットなし」生値を直接確認する
            # （計画書 §6 Test6 の -cutoff 期待値はこちらのケースに対応する）
            no_hit_path = str(Path(xml_dir) / 'sensor_no_hit.xml')
            build_sensor_wall_jig(params, no_hit_path, wall_pos=None, wall_normal=None)
            m_nohit = mujoco.MjModel.from_xml_path(no_hit_path)
            d_nohit = mujoco.MjData(m_nohit)
            mujoco.mj_forward(m_nohit, d_nohit)
            val_nohit = float(d_nohit.sensordata[idx])
            ok_nohit = abs(val_nohit - (-cutoff)) < 1e-6
            checks.append(dict(name=f"range_{name}_no_geometry_raw", expected=-cutoff, actual=val_nohit,
                                tol="exact", ok=ok_nohit))
        regressions[name] = dict(slope=slope, intercept=intercept, r2=r2)

    # --- Accel 静止5s平均 ---
    tracker_static = QaccTracker()
    sim = MouseSim(open_xml, params=params)
    h = get_handles(sim.model)
    sim.full_reset(cell=(0, 0), heading_deg=0)
    settle_contact(sim.model, sim.data, h, tracker=tracker_static)
    t0 = sim.data.time
    log = run_logged(sim.model, sim.data, duration=5.0, ctrl_fn=lambda t: (0.0, 0.0), tracker=tracker_static,
                      log_every=2, t0=t0, voltage_limit=params.voltage_limit)
    accel_static = np.mean(log['sensordata'][-1000:, 6:9], axis=0)
    g = D.g
    ok_ax = abs(accel_static[0] - 0.0) <= 0.05
    ok_ay = abs(accel_static[1] - 0.0) <= 0.05
    ok_az = abs(accel_static[2] - g) <= 0.05
    checks += [
        dict(name="accel_static_x", expected=0.0, actual=float(accel_static[0]), tol="0.05", ok=ok_ax),
        dict(name="accel_static_y", expected=0.0, actual=float(accel_static[1]), tol="0.05", ok=ok_ay),
        dict(name="accel_static_z", expected=g, actual=float(accel_static[2]), tol="0.05", ok=ok_az),
    ]

    # r5 確定（2026-08-10 教授承認）: Gyro 独立クロスチェックを「平面仮定 d(yaw)/dt vs gyro_z」相当の
    # z 成分単独比較から、隣接ログの回転行列差分から機体系角速度ベクトル ω_body を算出する
    # 3 軸照合（ω×=R^T・(dR/dt) の反対称成分を x,y,z 全成分について取り出す）へ変更する。
    # 物理サブステップ毎（間引きなし）にログし、中心差分の打ち切り誤差を最小化する。
    sim3 = MouseSim(open_xml, params=params)
    h3 = get_handles(sim3.model)
    sim3.full_reset(cell=(0, 0), heading_deg=0)
    settle_contact(sim3.model, sim3.data, h3)
    dt_phys = float(sim3.model.opt.timestep)
    n_steps = int(round(0.3 / dt_phys))
    xmats, gyros, ts = [], [], []
    for step in range(n_steps):
        sim3.data.ctrl[0] = 1.0
        sim3.data.ctrl[1] = -1.0
        mujoco.mj_step(sim3.model, sim3.data)
        xmats.append(sim3.data.xmat[h3.mouse_body_id].reshape(3, 3).copy())
        gyros.append(sim3.data.sensordata[9:12].copy())
        ts.append(float(sim3.data.time))
    # 中心差分（xmat[i-1], xmat[i+1] から時刻 ts[i] 上の角速度を推定）を用いる。
    # 前進差分 (xmat[i],xmat[i+1])/dt は時刻 ts[i]+dt/2 の値を表すため gyro[i]
    # （時刻 ts[i] の瞬時値）と比べると半ステップ分ずれ、急な角加速度期間で有意な
    # 誤差が出るバグがあった（テスト側の明白な実装バグとして修正）。
    def _omega_body_from_xmats(xmats_seq, ts_seq):
        """隣接ログの回転行列差分から機体系角速度ベクトル ω_body=[wx,wy,wz] を
        中心差分で算出する（ω×=R^T・(dR/dt) の反対称成分。r5確定）。"""
        out = []
        for i in range(1, len(xmats_seq) - 1):
            R1, R2 = xmats_seq[i - 1], xmats_seq[i + 1]
            dR = R1.T @ R2
            dt_i = ts_seq[i + 1] - ts_seq[i - 1]
            skew = (dR - dR.T) / 2.0
            out.append(np.array([skew[2, 1], skew[0, 2], skew[1, 0]]) / dt_i)
        return np.array(out) if out else np.zeros((0, 3))

    # r5 確定（2026-08-10 教授承認）: ジャイロ3軸照合は、比較前に両信号（Gyroセンサ値と
    # xmat差分角速度）へ同一のローパスフィルタを適用する。カットオフ 100Hz = 制御周期
    # （control_dt=0.01s の逆数）。センサ較正の検証は制御帯域で行うのが目的整合であり、
    # 接触ソルバ由来の物理サブステップ周期(1〜2kHz域)の数値チャタリングは制御ループには
    # 現れないため照合対象から除外するのが妥当、という設計判断。scipy 未導入のため
    # 単純な1次IIR（指数移動平均）をフォワード・バックワード両方向に適用してゼロ位相化する
    # （filtfilt 相当。numpy のみで実装）。
    def _lowpass_filtfilt_1axis(x, dt, fcut):
        x = np.asarray(x, dtype=np.float64)
        if len(x) == 0:
            return x
        rc = 1.0 / (2.0 * np.pi * fcut)
        alpha = dt / (rc + dt)

        def _forward(sig):
            y = np.empty_like(sig)
            y[0] = sig[0]
            for k in range(1, len(sig)):
                y[k] = y[k - 1] + alpha * (sig[k] - y[k - 1])
            return y

        fwd = _forward(x)
        return _forward(fwd[::-1])[::-1]

    def _lowpass_filtfilt(arr, dt, fcut):
        """arr: (N,3) 配列。列（軸）ごとに独立にゼロ位相ローパスを適用する。"""
        if arr.size == 0:
            return arr
        return np.stack([_lowpass_filtfilt_1axis(arr[:, k], dt, fcut) for k in range(arr.shape[1])], axis=1)

    GYRO_LPF_CUTOFF_HZ = 1.0 / params.control_dt  # = 100Hz（制御周期の逆数）

    # 離散恒等式に基づく比較対象の補正（テスト側の明白な実装バグ修正）:
    # MuJoCo の半陰的積分では xmat の中心差分（区間 [i-1, i+1]、ステップ i と i+1 の
    # 回転を跨ぐ）は厳密に (ω[i]+ω[i+1])/2 に対応する（ω[k] は step k 後の qvel =
    # gyro の読み値）。旧実装は瞬時値 ω[i] と比較しており、差 = 隣接サブステップ間の
    # 速度ジャンプの半分が残差として現れていた（接触チャタリング時に顕在化。
    # LPF 後残差 0.083/0.0137 rad/s の主因）。比較対象を 2 サンプル平均に修正する。
    def _gyro_matched_to_fd(gyros_seq):
        """FD（中心 i、ステップ i と i+1 を跨ぐ）に整合する (ω[i]+ω[i+1])/2 の系列。"""
        arr = np.array(gyros_seq)
        if len(arr) < 3:
            return np.zeros((0, 3))
        return 0.5 * (arr[1:-1] + arr[2:])

    omega_fd = _omega_body_from_xmats(xmats, ts)
    gyro_center = _gyro_matched_to_fd(gyros)
    omega_fd_f = _lowpass_filtfilt(omega_fd, dt_phys, GYRO_LPF_CUTOFF_HZ)
    gyro_center_f = _lowpass_filtfilt(gyro_center, dt_phys, GYRO_LPF_CUTOFF_HZ)
    skip = min(10, max(0, len(omega_fd_f) - 1))
    err = np.abs(omega_fd_f[skip:] - gyro_center_f[skip:])
    ok_gyro = bool(err.size > 0 and np.all(err <= 0.05))
    checks.append(dict(name="gyro_vs_xmat_3axis_dynamic", expected="<=0.05 rad/s (100Hz LPF後)",
                        actual=float(np.max(err)) if err.size else float('nan'), tol="0.05", ok=ok_gyro))

    # r5 確定（2026-08-10 教授承認）: Test1 の直進定常区間（回転がほぼゼロの状況）でも
    # 同じ 3 軸クロスチェック（同一 100Hz LPF 適用後に比較）を課す。動的区間より厳しい
    # 許容 0.01 rad/s を課す。
    t1_snap = CALIB.get('test1_05V_final_state')
    if t1_snap is not None:
        sim4 = MouseSim(open_xml, params=params)
        h4 = get_handles(sim4.model)
        sim4.data.qpos[:] = t1_snap['qpos']
        sim4.data.qvel[:] = t1_snap['qvel']
        mujoco.mj_forward(sim4.model, sim4.data)
        n_steps4 = int(round(0.2 / dt_phys))
        xmats4, gyros4, ts4 = [], [], []
        for step in range(n_steps4):
            sim4.data.ctrl[0] = 0.5
            sim4.data.ctrl[1] = 0.5
            mujoco.mj_step(sim4.model, sim4.data)
            xmats4.append(sim4.data.xmat[h4.mouse_body_id].reshape(3, 3).copy())
            gyros4.append(sim4.data.sensordata[9:12].copy())
            ts4.append(float(sim4.data.time))
        omega_fd4 = _omega_body_from_xmats(xmats4, ts4)
        gyro_center4 = _gyro_matched_to_fd(gyros4)  # 離散恒等式に整合する 2 サンプル平均（上記参照）
        omega_fd4_f = _lowpass_filtfilt(omega_fd4, dt_phys, GYRO_LPF_CUTOFF_HZ)
        gyro_center4_f = _lowpass_filtfilt(gyro_center4, dt_phys, GYRO_LPF_CUTOFF_HZ)
        skip4 = min(10, max(0, len(omega_fd4_f) - 1))
        err4 = np.abs(omega_fd4_f[skip4:] - gyro_center4_f[skip4:])
        ok_gyro_static = bool(err4.size > 0 and np.all(err4 <= 0.01))
        checks.append(dict(name="gyro_vs_xmat_3axis_test1_straight", expected="<=0.01 rad/s (100Hz LPF後)",
                            actual=float(np.max(err4)) if err4.size else float('nan'), tol="0.01",
                            ok=ok_gyro_static))
    else:
        checks.append(dict(name="gyro_vs_xmat_3axis_test1_straight", expected="Test1 スナップショット利用可能",
                            actual="欠落", tol="-", ok=False, note="CALIB['test1_05V_final_state'] が無い"))

    # --- 動的 Accel-X: Test1 の a0=v_ss/tau_v 系列と一致 ---
    # r5 確定（2026-08-10 教授承認）: 判定を相対誤差(分母フロア0.02)から絶対値+相対の混合判定
    # |err| <= max(10%*|a_theory|, 0.05 m/s^2) へ変更する。指数減衰する a_theory がゼロへ
    # 近づく領域（t>0.35s）で相対誤差の分母が縮小し、背景チャタリングのノイズフロア
    # （実測0.01〜0.03 m/s^2程度）に対して相対誤差が病的に発散していたことの是正
    # （数値そのものは変更せず、判定式を目的に整合させる）。
    t1log = CALIB.get('test1_accelx_log')
    if t1log is not None:
        tt = t1log['t']
        ax = t1log['accel_x']
        v_ss_th = t1log['v_ss_theory']
        tau_v_th = t1log['tau_v']
        # 生の Accel-X 単一サンプルは 4点接地の不静定性由来の高周波チャタリング
        # ノイズが乗るため、点ごとの比較ではなく ±25ms の窓平均で比較する
        # （報告書「計画書との矛盾・懸念」の発見事項に対するロバスト化。実装バグ修正）。
        sample_ts = np.arange(0.05, min(1.0, tt[-1]), 0.1)
        abs_errs, bounds = [], []
        for st in sample_ts:
            window = (tt >= st - 0.025) & (tt <= st + 0.025)
            if not np.any(window):
                continue
            a_theory = (v_ss_th / tau_v_th) * np.exp(-st / tau_v_th)
            a_meas = float(np.mean(ax[window]))
            abs_errs.append(abs(a_meas - a_theory))
            bounds.append(max(0.10 * abs(a_theory), 0.05))
        abs_errs = np.array(abs_errs)
        bounds = np.array(bounds)
        ok_accel_dyn = bool(len(abs_errs) > 0 and np.all(abs_errs <= bounds))
        margin = float(np.max(abs_errs - bounds)) if len(abs_errs) else float('nan')
        checks.append(dict(name="accel_x_dynamic_vs_test1", expected="|err|<=max(10%|a_theory|,0.05m/s^2)",
                            actual=f"max(|err|-bound)={margin:.4f}", tol="mixed", ok=ok_accel_dyn))
    else:
        checks.append(dict(name="accel_x_dynamic_vs_test1", expected="Test1 ログ利用可能", actual="欠落",
                            tol="-", ok=False, note="CALIB['test1_accelx_log'] が無い"))

    qacc_ok1, qacc_msg1 = tracker_static.check(strict_limit=QACC_STRICT)
    checks.append(dict(name="qacc_monitor_static_strict", expected="OK", actual=qacc_msg1, tol="strict", ok=qacc_ok1))

    calib = dict(regressions=regressions, accel_static=accel_static.tolist())
    ok, msg = _finish("Test6_sensors", checks, calib=calib)
    assert ok, msg


# ======================================================================
# Test 7: エネルギー整合性（計画書 §6 Test7）
# ======================================================================
def test_07_energy(params, open_xml, D):
    checks = []

    # --- 7a: 静止5s、自発運動なし ---
    tracker_a = QaccTracker()
    sim = MouseSim(open_xml, params=params)
    h = get_handles(sim.model)
    sim.full_reset(cell=(0, 0), heading_deg=0)
    settle_contact(sim.model, sim.data, h, tracker=tracker_a)
    x0, y0, yaw0 = sim.privileged_pose()
    t0 = sim.data.time
    log_a = run_logged(sim.model, sim.data, duration=5.0, ctrl_fn=lambda t: (0.0, 0.0), tracker=tracker_a,
                        log_every=2, t0=t0, voltage_limit=params.voltage_limit)
    max_v = float(np.max(np.abs(log_a['v_fwd'])))
    dpos = float(np.hypot(log_a['x'][-1] - x0, log_a['y'][-1] - y0))
    dyaw_deg = float(np.degrees(abs(log_a['yaw'][-1] - yaw0)))
    ok_v = max_v < 0.001
    ok_pos = dpos < 0.0005
    ok_yaw = dyaw_deg < 0.1
    qacc_ok_a, qacc_msg_a = tracker_a.check(strict_limit=QACC_STRICT)
    checks += [
        dict(name="7a_max_v", expected="<1mm/s", actual=max_v, tol="-", ok=ok_v),
        dict(name="7a_dpos", expected="<0.5mm", actual=dpos, tol="-", ok=ok_pos),
        dict(name="7a_dyaw_deg", expected="<0.1deg", actual=dyaw_deg, tol="-", ok=ok_yaw),
        dict(name="7a_qacc_strict", expected="OK", actual=qacc_msg_a, tol="strict", ok=qacc_ok_a),
    ]

    # --- 7b/7c: Test4 相当を3sまで延長し、仕事収支を評価 ---
    tracker_b = QaccTracker()
    sim2 = MouseSim(open_xml, params=params)
    h2 = get_handles(sim2.model)
    sim2.full_reset(cell=(0, 0), heading_deg=0)
    settle_contact(sim2.model, sim2.data, h2)
    contacts_b = measure_contacts_avg(sim2.model, sim2.data, h2, ctrl=(0.0, 0.0), n_samples=100)
    N_w = contacts_b.N_wheel
    mu_wheel = contacts_b.mu_wheel if contacts_b.mu_wheel is not None else 1.0
    mu_caster = contacts_b.mu_caster if contacts_b.mu_caster is not None else parse_mu_caster(params)

    t0b = sim2.data.time
    log_b = run_logged(sim2.model, sim2.data, duration=3.0, ctrl_fn=lambda t: (params.voltage_limit, params.voltage_limit),
                        tracker=tracker_b, log_every=2, t0=t0b, voltage_limit=params.voltage_limit)

    # 巡航（電圧律速相、t=3s時点）のキャスター荷重を時間平均測定し、v_ss 理論式に用いる
    # （静止直後の静的値とは大きく異なりうる。4点接地の不静定性由来のチャタリングの
    # 発見に基づく実装。報告書「計画書との矛盾・懸念」参照）
    contacts_cruise = measure_contacts_avg(sim2.model, sim2.data, h2,
                                            ctrl=(params.voltage_limit, params.voltage_limit), n_samples=200)
    N_c = contacts_cruise.N_caster
    v_ss_theory = (2 * (D.gainprm0 / D.r) * params.voltage_limit - 2 * (D.tau_c / D.r) - mu_caster * N_c) / D.c_eff

    v_end = float(log_b['v_fwd'][-1])
    # 2点差分は 4点接地の不静定性由来の速度チャタリング（本報告書の他の発見事項と
    # 同一現象）に敏感すぎるバグがあったため、末尾100msの線形回帰でロバストに推定する
    # （テスト側の実装バグ修正）。
    tail_mask_dvdt = log_b['t'] >= (log_b['t'][-1] - 0.1)
    if np.sum(tail_mask_dvdt) >= 5:
        dvdt_end, _, _ = linreg(log_b['t'][tail_mask_dvdt], log_b['v_fwd'][tail_mask_dvdt])
        dvdt_end = float(dvdt_end)
    else:
        dvdt_end = float((log_b['v_fwd'][-1] - log_b['v_fwd'][-6]) / (log_b['t'][-1] - log_b['t'][-6]))
    ok_vend = abs(v_end - v_ss_theory) / v_ss_theory <= 0.05
    ok_dvdt = abs(dvdt_end) < 0.1
    qacc_ok_b, qacc_msg_b = tracker_b.check(strict_limit=None)
    checks += [
        dict(name="7b_v_at_3s", expected=v_ss_theory, actual=v_end, tol="5%", ok=ok_vend),
        dict(name="7b_dvdt_at_3s", expected="<0.1 m/s^2", actual=dvdt_end, tol="-", ok=ok_dvdt),
        dict(name="7b_qacc_loose", expected="OK", actual=qacc_msg_b, tol="loose", ok=qacc_ok_b),
    ]

    t = log_b['t']
    v = log_b['v_fwd']
    x = log_b['x'] - log_b['x'][0]
    omega_avg = (log_b['qvel_l'] + log_b['qvel_r']) / 2.0
    power = log_b['actuator_force'][:, 0] * log_b['qvel_l'] + log_b['actuator_force'][:, 1] * log_b['qvel_r']
    slip_signal = D.r * omega_avg - v
    v_ss_th_for_slip = v_ss_theory
    # Test4 と同型のバグ（t=0 で slip_signal が恒等的に0のため閾値を初めて下回る点の
    # 単純探索が t=0 を誤検出する）を同じ方法で修正: ピーク以降で初めて閾値を下回る点。
    thresh_slip = 0.05 * v_ss_th_for_slip
    peak_idx_7c = int(np.argmax(slip_signal))
    below_7c = slip_signal[peak_idx_7c:] <= thresh_slip
    idx_t1 = peak_idx_7c + int(np.argmax(below_7c)) if np.any(below_7c) else 0

    Fc_theory = 2 * D.tau_c / D.r + parse_mu_roll_floor(params) * N_w / D.r + mu_caster * N_c

    # 【発見・修正】W を data.actuator_force（= gainprm0*V + biasprm2*omega、逆起電力を
    # 既に差し引いた正味トルク）から積分する場合、散逸項に c_eff*v^2 をそのまま使うと
    # 逆起電力由来の項 2*(N^2 Kt Ke/R)*v^2/r^2 を二重計上してしまう（W が既にその損失分
    # 減った正味仕事になっているため）。第一原理（車輪トルク収支 τ_act = (I_w+arm)dω/dt
    # + b*ω + τ_c*sign(ω) + F_ground*r を両輪について ω 倍し積分、r*ω=v の転がり拘束で
    # 整理）から再導出すると、W と整合する速度比例損失係数は c_eff ではなく
    # b_trans = 2*b/r^2（関節粘性 b の寄与のみ、逆起電力寄与を含まない）である。
    # 数値をいじる代わりに整合する物理式へ差し替える（テスト側の実装バグ修正）。
    b_trans = 2 * D.b / D.r ** 2

    W_2 = float(np.trapezoid(power[idx_t1:], t[idx_t1:]))
    dKE_2 = 0.5 * D.M_eff * v[-1] ** 2 - 0.5 * D.M_eff * v[idx_t1] ** 2
    E_diss_2 = float(np.trapezoid(b_trans * v[idx_t1:] ** 2, t[idx_t1:])) + Fc_theory * (x[-1] - x[idx_t1])
    resid_2 = abs(W_2 - (dKE_2 + E_diss_2)) / abs(W_2) if W_2 != 0 else float('nan')
    ok_2term = np.isfinite(resid_2) and resid_2 <= 0.05
    checks.append(dict(name="7c_2term_balance", expected=0.0, actual=resid_2, tol="<=5%", ok=ok_2term))

    W_3 = float(np.trapezoid(power, t))
    dKE_3 = 0.5 * D.M_eff * v[-1] ** 2 - 0.5 * D.M_eff * v[0] ** 2
    E_diss_3 = float(np.trapezoid(b_trans * v ** 2, t)) + Fc_theory * (x[-1] - x[0])
    slip_heat_integrand = np.where(slip_signal > 0, mu_wheel * N_w * slip_signal, 0.0)
    E_slip = float(np.trapezoid(slip_heat_integrand, t))
    resid_3 = abs(W_3 - (dKE_3 + E_diss_3 + E_slip)) / abs(W_3) if W_3 != 0 else float('nan')
    ok_3term = np.isfinite(resid_3) and resid_3 <= 0.10
    checks.append(dict(name="7c_3term_balance", expected=0.0, actual=resid_3, tol="<=10%", ok=ok_3term))

    calib = dict(N_w=N_w, N_c=N_c, mu_wheel=mu_wheel, mu_caster=mu_caster, v_ss_theory=v_ss_theory,
                 resid_2term=resid_2, resid_3term=resid_3, W_2=W_2, W_3=W_3)
    ok, msg = _finish("Test7_energy", checks, calib)
    assert ok, msg


# ======================================================================
# Test 8: 急制動ピッチ安定（計画書 §6 Test8。動的部は qacc (a)のみ=loose）
# ======================================================================
def test_08_braking_pitch(params, open_xml, D):
    # r5 確定（2026-08-10 教授承認）: Test8 を二部構成に改訂する。
    #  (a) 実用制動試験: v≈2.5m/s まで加速→ctrl=[0,0]（短絡ブレーキ。プラギングなし）で
    #      停止までログし、ピッチ・荷重移動・復帰を検証する（RL方策が通常発行する制動）。
    #  (b) 乱用耐性試験（教授指示。検収待ちマーク不要）: v≈2.5m/sから ctrl=[-3,-3] を1.0s
    #      印加（プラギング=逆転制動。RL方策が指令しうる最悪入力）し、転倒しないことのみを
    #      確認する。ピッチ最大値は記録のみで合否対象外
    #      （機序: note_004_r4_diagnostics.py 実験1でプラギング反動トルク ≈0.09N·m ≫
    #      重力復元 ≈0.006N·m と確定済み。後傾ホッピングは軽量・高トルク車の実在物理であり
    #      「転倒しないか」のみを問う）。
    # 静的判定（転倒余裕）・qacc監視方針（loose）は従来どおり。
    checks = []
    margin_info = compute_tipover_margin(params, open_xml, D)
    ok_margin = margin_info['margin'] >= 1.5
    checks.append(dict(name="static_tipover_margin", expected=">=1.5", actual=margin_info['margin'], tol="-",
                        ok=ok_margin))

    V_TARGET = 2.5  # m/s（計画書 r5 Test8(a)/(b) 共通の助走目標速度。手順の設定値であり期待値ではない）
    x_f = margin_info['x_f']
    z_cg = margin_info['z_cg']

    def _accelerate_to(sim, tracker, v_target, max_time=3.0):
        """ctrl=[Vmax,Vmax] で v_target [m/s] に到達するまで加速する（到達可否を返す）。
        固定時間でなく速度到達を停止条件とする（note_004_r4_diagnostics.py 実験1と同じ手順）。"""
        max_steps = int(round(max_time / params.physics_dt))
        steps = 0
        while sim.privileged_velocity()[0] < v_target and steps < max_steps:
            sim.data.ctrl[0] = params.voltage_limit
            sim.data.ctrl[1] = params.voltage_limit
            mujoco.mj_step(sim.model, sim.data)
            tracker.update(sim.data)
            steps += 1
        return sim.privileged_velocity()[0] >= v_target

    # ------------------------------------------------------------------
    # (a) 実用制動試験（r5確定）
    # ------------------------------------------------------------------
    tracker_a = QaccTracker()
    sim_a = MouseSim(open_xml, params=params)
    h_a = get_handles(sim_a.model)
    sim_a.full_reset(cell=(0, 0), heading_deg=0)
    settle_contact(sim_a.model, sim_a.data, h_a)
    contacts_static = measure_contacts_avg(sim_a.model, sim_a.data, h_a, ctrl=(0.0, 0.0), n_samples=100)
    N_front_static = contacts_static.N_front
    xmat0 = sim_a.data.xmat[h_a.mouse_body_id].reshape(3, 3)
    pitch0 = float(np.arcsin(np.clip(-xmat0[2, 0], -1.0, 1.0)))

    reached_a = _accelerate_to(sim_a, tracker_a, V_TARGET)
    checks.append(dict(name="8a_reached_v_target", expected=f">={V_TARGET} m/s",
                        actual=sim_a.privileged_velocity()[0], tol="-", ok=reached_a))

    floor_id = h_a.floor_geom_id
    front_id = h_a.caster_front_geom_id
    back_id = h_a.caster_back_geom_id
    res6 = np.zeros(6)
    dt_a = float(sim_a.model.opt.timestep)
    t0brake = sim_a.data.time
    a_t, a_pitch, a_Nf, a_Nb, a_v = [], [], [], [], []
    t_stop_a = None
    max_brake_steps = int(round(2.5 / dt_a))
    LOG_EVERY_A = 2
    # 【発見・実装上の理由】物理サブステップ単位の単発 d.contact 読み取りは、他の全テストで
    # 既知の「4点/3点接地の不静定性由来のチャタリング」（measure_contacts_avg のdocstring
    # 参照）と同一機序で、走行中の前キャスター接触が log_every 窓内で 0↔実測値の間を毎ステップ
    # 交互に往復することを実測で確認した（減速の物理量そのもの[荷重移動 ΔN_f]は健全で
    # 8a_load_transfer_dNf_3point は合格する）。他テストの measure_contacts_avg と同じ
    # 考え方で、log_every 窓内の全サブステップを時間平均してから記録する（テスト側の
    # 明白な実装バグ修正。合否閾値[静止時1%]の数値自体は変更しない）。
    #
    # r5 確定（2026-08-10 教授承認）: 接地品質判定（8a_front_caster_contact_quality）専用に、
    # 制御周期 control_dt（=100Hz、Test6のLPFカットオフと同一の「センサ較正の検証は制御帯域で
    # 行う」設計思想）幅で別途時間平均した a_Nf_ctrl 系列を追加で記録する。実測診断の結果、
    # 上記 LOG_EVERY_A=2(1ms)窓の平均でも接地喪失イベントが1〜8ms周期で多発し（各イベント
    # 自体は50ms未満で全て解消するが）時間比率が71%まで低下する一方、10ms(=control_dt)窓
    # 平均では96%まで回復することを実測確認した（20ms窓では100%）。ピッチ・速度・荷重移動
    # 等の他量は従来どおり LOG_EVERY_A=2 のまま変更しない。
    CONTROL_WINDOW_STEPS = max(1, int(round(params.control_dt / dt_a)))
    Nf_accum = Nb_accum = 0.0
    Nf_ctrl_accum = 0.0
    a_t_ctrl, a_Nf_ctrl = [], []
    for step in range(max_brake_steps):
        sim_a.data.ctrl[0] = 0.0
        sim_a.data.ctrl[1] = 0.0
        mujoco.mj_step(sim_a.model, sim_a.data)
        tracker_a.update(sim_a.data)
        Nf_step = Nb_step = 0.0
        for i in range(sim_a.data.ncon):
            c = sim_a.data.contact[i]
            pair = (c.geom1, c.geom2)
            if floor_id in pair:
                other = c.geom1 if c.geom2 == floor_id else c.geom2
                mujoco.mj_contactForce(sim_a.model, sim_a.data, i, res6)
                nf = abs(float(res6[0]))
                if other == front_id:
                    Nf_step += nf
                elif other == back_id:
                    Nb_step += nf
        Nf_accum += Nf_step
        Nb_accum += Nb_step
        Nf_ctrl_accum += Nf_step
        if (step + 1) % CONTROL_WINDOW_STEPS == 0:
            a_t_ctrl.append(sim_a.data.time - t0brake)
            a_Nf_ctrl.append(Nf_ctrl_accum / CONTROL_WINDOW_STEPS)
            Nf_ctrl_accum = 0.0
        if (step + 1) % LOG_EVERY_A == 0:
            xmat = sim_a.data.xmat[h_a.mouse_body_id].reshape(3, 3)
            pitch = float(np.arcsin(np.clip(-xmat[2, 0], -1.0, 1.0)))
            Nf = Nf_accum / LOG_EVERY_A
            Nb = Nb_accum / LOG_EVERY_A
            Nf_accum = Nb_accum = 0.0
            v_fwd = sim_a.privileged_velocity()[0]
            t_rel = sim_a.data.time - t0brake
            a_t.append(t_rel)
            a_pitch.append(pitch)
            a_Nf.append(Nf)
            a_Nb.append(Nb)
            a_v.append(v_fwd)
            if t_stop_a is None and v_fwd <= 0.0:
                t_stop_a = t_rel
            if t_stop_a is not None and t_rel >= t_stop_a + 1.0:
                break
    a_t = np.array(a_t)
    a_pitch = np.array(a_pitch)
    a_Nf = np.array(a_Nf)
    a_v = np.array(a_v)

    ok_reached_stop = t_stop_a is not None
    checks.append(dict(name="8a_reached_full_stop", expected=True, actual=ok_reached_stop, tol="-",
                        ok=ok_reached_stop))

    mask_braking = (a_t <= t_stop_a) if ok_reached_stop else np.ones_like(a_t, dtype=bool)
    max_pitch_deg_a = float(np.degrees(np.max(np.abs(a_pitch[mask_braking])))) if np.any(mask_braking) else float('nan')
    min_Nf_a = float(np.min(a_Nf[mask_braking])) if np.any(mask_braking) else float('nan')

    ok_pitch_a = np.isfinite(max_pitch_deg_a) and max_pitch_deg_a <= 3.0
    checks.append(dict(name="8a_max_pitch_deg", expected="<=3deg", actual=max_pitch_deg_a, tol="-", ok=ok_pitch_a))

    # r5 確定（2026-08-10 教授承認）: 前キャスター接地判定を「常時 min>1%静止時」の単一点
    # 判定から、「(A) 制動サンプルの95%以上で法線力>静止時1% かつ (B) 接地喪失
    # （<1%静止時）の連続区間はすべて50ms以内に回復」の2条件へ変更する。実測診断の結果、
    # 全開巡航→短絡ブレーキ切替の瞬間に前キャスターを含む全接点が数ms間完全に浮く実在の
    # 跳ね（qacc最大1.6e5、荷重移動は理論と整合）が生じることを確認しており、瞬間的な
    # 接地喪失そのものは前転の予兆ではない。前転（転倒）検出の本体はピッチ・転倒・復帰判定
    # （8a_max_pitch_deg・8a_pitch_recovery_1s_after_stop_deg 等）で別途担保しているため、
    # 本判定は「短時間の跳ねは許容しつつ、持続的な接地喪失・多発する接地喪失は不合格」と
    # する準静的な接地品質判定に位置づける。
    threshold_Nf = 0.01 * N_front_static
    a_t_ctrl = np.array(a_t_ctrl)
    a_Nf_ctrl = np.array(a_Nf_ctrl)
    mask_ctrl = (a_t_ctrl <= t_stop_a) if ok_reached_stop else np.ones_like(a_t_ctrl, dtype=bool)
    a_t_b = a_t_ctrl[mask_ctrl]
    a_Nf_b = a_Nf_ctrl[mask_ctrl]
    n_brake_samples = int(len(a_Nf_b))
    if n_brake_samples > 0:
        lost = a_Nf_b <= threshold_Nf
        frac_contact_ok = 1.0 - float(np.mean(lost))
        ok_frac = frac_contact_ok >= 0.95

        max_loss_dur = 0.0
        run_start_t = None
        run_end_t = None
        sample_dt_a = float(a_t_b[1] - a_t_b[0]) if n_brake_samples > 1 else dt_a * CONTROL_WINDOW_STEPS
        for idx in range(n_brake_samples):
            if lost[idx]:
                if run_start_t is None:
                    run_start_t = a_t_b[idx]
                run_end_t = a_t_b[idx]
            else:
                if run_start_t is not None:
                    max_loss_dur = max(max_loss_dur, run_end_t - run_start_t + sample_dt_a)
                    run_start_t = None
        if run_start_t is not None:
            max_loss_dur = max(max_loss_dur, run_end_t - run_start_t + sample_dt_a)
        ok_recovery = max_loss_dur <= 0.05
    else:
        frac_contact_ok, max_loss_dur = float('nan'), float('nan')
        ok_frac = ok_recovery = False

    ok_contact_a = bool(ok_frac and ok_recovery)
    checks.append(dict(
        name="8a_front_caster_contact_quality",
        expected=">=95%接地 かつ 接地喪失連続区間<=50ms",
        actual=f"接地率={frac_contact_ok*100:.1f}% 最大喪失連続時間={max_loss_dur*1000:.1f}ms",
        tol="-", ok=ok_contact_a))

    # 荷重移動検証（3点接地の準静的予測。制動開始直後の準定常区間で評価。r4式を踏襲）
    if ok_reached_stop:
        mask_q = (a_t >= 0.02) & (a_t <= min(0.2, t_stop_a))
    else:
        mask_q = np.zeros_like(a_t, dtype=bool)
    if np.any(mask_q):
        dNf_a = float(np.mean(a_Nf[mask_q]) - N_front_static)
        a_meas_a = float(np.mean(np.gradient(a_v, a_t)[mask_q]))
    else:
        dNf_a, a_meas_a = float('nan'), float('nan')
    x_w = float(sim_a.model.body('left_wheel').pos[0])
    dN_theory_a = D.M * abs(a_meas_a) * z_cg / (x_f - x_w) if np.isfinite(a_meas_a) else float('nan')
    ok_loadtransfer_a = bool(np.isfinite(dN_theory_a) and dN_theory_a != 0
                              and abs(dNf_a - dN_theory_a) / dN_theory_a <= 0.30)
    checks.append(dict(name="8a_load_transfer_dNf_3point", expected=dN_theory_a, actual=dNf_a, tol="±30%",
                        ok=ok_loadtransfer_a))

    if ok_reached_stop:
        mask_recover = a_t >= t_stop_a + 1.0
        pitch_recover_deg_a = (float(np.degrees(abs(a_pitch[mask_recover][0] - pitch0)))
                                if np.any(mask_recover) else float('nan'))
    else:
        pitch_recover_deg_a = float('nan')
    ok_recover_a = np.isfinite(pitch_recover_deg_a) and pitch_recover_deg_a <= 0.5
    checks.append(dict(name="8a_pitch_recovery_1s_after_stop_deg", expected="<=0.5deg",
                        actual=pitch_recover_deg_a, tol="-", ok=ok_recover_a))

    qacc_ok_a, qacc_msg_a = tracker_a.check(strict_limit=None)
    checks.append(dict(name="8a_qacc_monitor_loose", expected="OK", actual=qacc_msg_a, tol="loose", ok=qacc_ok_a))

    # ------------------------------------------------------------------
    # (b) 乱用耐性試験（教授指示。検収待ちマーク不要）
    # ------------------------------------------------------------------
    tracker_b = QaccTracker()
    sim_b = MouseSim(open_xml, params=params)
    h_b = get_handles(sim_b.model)
    sim_b.full_reset(cell=(0, 0), heading_deg=0)
    settle_contact(sim_b.model, sim_b.data, h_b)
    reached_b = _accelerate_to(sim_b, tracker_b, V_TARGET)
    checks.append(dict(name="8b_reached_v_target", expected=f">={V_TARGET} m/s",
                        actual=sim_b.privileged_velocity()[0], tol="-", ok=reached_b))

    dt_b = float(sim_b.model.opt.timestep)
    n_plug = int(round(1.0 / dt_b))
    min_zaxis = 1.0
    max_pitch_plug_deg = 0.0
    for _ in range(n_plug):
        sim_b.data.ctrl[0] = -params.voltage_limit
        sim_b.data.ctrl[1] = -params.voltage_limit
        mujoco.mj_step(sim_b.model, sim_b.data)
        tracker_b.update(sim_b.data)
        xmat = sim_b.data.xmat[h_b.mouse_body_id].reshape(3, 3)
        min_zaxis = min(min_zaxis, float(xmat[2, 2]))
        pitch_deg = float(np.degrees(np.arcsin(np.clip(-xmat[2, 0], -1.0, 1.0))))
        max_pitch_plug_deg = max(max_pitch_plug_deg, abs(pitch_deg))

    rec_t, rec_pitch = [], []
    n_rec = int(round(1.2 / dt_b))
    t0rec = sim_b.data.time
    for step in range(n_rec):
        sim_b.data.ctrl[0] = 0.0
        sim_b.data.ctrl[1] = 0.0
        mujoco.mj_step(sim_b.model, sim_b.data)
        tracker_b.update(sim_b.data)
        xmat = sim_b.data.xmat[h_b.mouse_body_id].reshape(3, 3)
        min_zaxis = min(min_zaxis, float(xmat[2, 2]))
        if (step + 1) % 2 == 0:
            rec_t.append(sim_b.data.time - t0rec)
            rec_pitch.append(float(np.degrees(np.arcsin(np.clip(-xmat[2, 0], -1.0, 1.0)))))
    rec_t = np.array(rec_t)
    rec_pitch = np.array(rec_pitch)
    idx_1s = int(np.argmin(np.abs(rec_t - 1.0))) if len(rec_t) else None
    pitch_1s_after_deg = float(abs(rec_pitch[idx_1s])) if idx_1s is not None else float('nan')

    ok_no_tip = min_zaxis > params.tipover_zaxis_threshold
    checks.append(dict(name="8b_no_tipover", expected=f">{params.tipover_zaxis_threshold}", actual=min_zaxis,
                        tol="-", ok=ok_no_tip,
                        note=f"最大ピッチ(記録のみ・合否対象外)={max_pitch_plug_deg:.2f}deg"))

    ok_pitch_recover_b = np.isfinite(pitch_1s_after_deg) and pitch_1s_after_deg <= 1.0
    checks.append(dict(name="8b_pitch_recovery_1s_deg", expected="<=1deg", actual=pitch_1s_after_deg, tol="-",
                        ok=ok_pitch_recover_b))

    qacc_ok_b, qacc_msg_b = tracker_b.check(strict_limit=None)
    checks.append(dict(name="8b_qacc_monitor_loose", expected="OK", actual=qacc_msg_b, tol="loose", ok=qacc_ok_b))

    calib = dict(margin=margin_info, a_part=dict(a_meas=a_meas_a, dNf=dNf_a, dN_theory=dN_theory_a,
                                                  t_stop=t_stop_a, max_pitch_deg=max_pitch_deg_a,
                                                  min_Nf=min_Nf_a, frac_contact_ok=frac_contact_ok,
                                                  max_loss_dur_ms=max_loss_dur * 1000.0),
                 b_part=dict(max_pitch_plug_deg=max_pitch_plug_deg, min_zaxis=min_zaxis,
                             pitch_1s_after_deg=pitch_1s_after_deg))
    ok, msg = _finish("Test8_braking_pitch", checks, calib)
    assert ok, msg


# ======================================================================
# Test 9: 直進方向安定性（ヨー外乱減衰）— 教授指示 2026-08-10 新設（検収待ちマーク不要）
# 前キャスター支持+CG前方配置（r4 の3点接地化）の直進安定性を、ヨー外乱インパルス後の
# 減衰で実測担保する。v ∈ {0.5, 2.0, v_max付近} の3水準で、定電圧整定→ヨー外乱→2sログ
# →減衰判定を行う。qacc は loose 分類（外乱インパルス+高速水準のスリップ可能性のため）。
# ======================================================================
def test_09_yaw_stability(params, open_xml, D):
    checks = []
    calib_levels = {}

    # 静的接地・v_max目安の実行時導出（Test1の定常式 v_ss(V) を V について逆算する）
    sim0 = MouseSim(open_xml, params=params)
    h0 = get_handles(sim0.model)
    sim0.full_reset(cell=(0, 0), heading_deg=0)
    settle_contact(sim0.model, sim0.data, h0)
    contacts0 = measure_contacts_avg(sim0.model, sim0.data, h0, ctrl=(0.0, 0.0), n_samples=100)
    N_c_static = contacts0.N_caster
    mu_caster = contacts0.mu_caster if contacts0.mu_caster is not None else parse_mu_caster(params)
    mu_wheel = contacts0.mu_wheel if contacts0.mu_wheel is not None else 1.0

    def v_ss_of_V(V0):
        return (2 * (D.gainprm0 / D.r) * V0 - 2 * (D.tau_c / D.r) - mu_caster * N_c_static) / D.c_eff

    def V_of_v_target(v_target):
        return (v_target * D.c_eff + 2 * D.tau_c / D.r + mu_caster * N_c_static) * D.r / (2 * D.gainprm0)

    v_max_est = v_ss_of_V(params.voltage_limit)
    levels = [
        ("low_0.5", 0.5),
        ("mid_2.0", 2.0),
        ("high_0.95vmax", 0.95 * v_max_est),
    ]

    # ヨー有効慣性 I_eff（Test3/Test5 r5 と同じ静的合成 Izz ベースで実行時導出）
    izz_model = mujoco.MjModel.from_xml_path(open_xml)
    h_izz = get_handles(izz_model)
    wheel_term = 2 * (D.I_w + D.armature) * (D.L / (2 * D.r)) ** 2
    I_eff = compute_izz_breakdown(izz_model)['total_izz'] + wheel_term
    x_f = abs(float(izz_model.geom_pos[h_izz.caster_front_geom_id][0]))

    PULSE_DUR = 0.01     # s（教授指示の外乱印加時間）
    OMEGA_TARGET = 1.0   # rad/s（教授指示の目安擾乱量）
    tau_pulse = I_eff * OMEGA_TARGET / PULSE_DUR   # 印加トルク[N・m]（I_eff 実行時導出により算出）
    HEADING_BOUND = 0.1  # rad（後半1sの方位変化がこれ以下なら「増幅していない」とみなす治具閾値）

    for label, v_target in levels:
        sim = MouseSim(open_xml, params=params)
        h = get_handles(sim.model)
        tracker = QaccTracker()
        sim.full_reset(cell=(0, 0), heading_deg=0)
        settle_contact(sim.model, sim.data, h, tracker=tracker)

        V0 = float(np.clip(V_of_v_target(v_target), -params.voltage_limit, params.voltage_limit))
        settle_dur = 6.0 * D.tau_v  # >=6*tau_v でほぼ完全整定（残留 <exp(-6)=0.25%）
        t0 = sim.data.time
        log_settle = run_logged(sim.model, sim.data, duration=settle_dur, ctrl_fn=lambda t: (V0, V0),
                                 tracker=tracker, log_every=2, t0=t0, voltage_limit=params.voltage_limit)
        v_settled = float(np.mean(log_settle['v_fwd'][-20:]))

        # ヨー外乱インパルスを data.qfrc_applied（フリージョイントのヨー回転DOF）に0.01s印加する。
        # DOF 割付けは root_dof_adr+5（世界座標系 z 軸回りトルク。worldbody直結の free joint では
        # 角速度DOFが世界座標系で表現されることをサンドボックス実験で確認済み）。
        yaw_dof = h.root_dof_adr + 5
        dt_phys = float(sim.model.opt.timestep)
        n_pulse = max(1, int(round(PULSE_DUR / dt_phys)))
        x0, y0, yaw0 = sim.privileged_pose()
        for _ in range(n_pulse):
            sim.data.ctrl[0] = V0
            sim.data.ctrl[1] = V0
            sim.data.qfrc_applied[yaw_dof] = tau_pulse
            mujoco.mj_step(sim.model, sim.data)
            tracker.update(sim.data)
        sim.data.qfrc_applied[yaw_dof] = 0.0

        t0log = sim.data.time
        log = run_logged(sim.model, sim.data, duration=2.0, ctrl_fn=lambda t: (V0, V0), tracker=tracker,
                          log_every=2, t0=t0log, voltage_limit=params.voltage_limit)

        omega = log['sensordata'][:, 11]
        t = log['t']
        omega0 = float(omega[0])

        # 判定1: 外乱後1sでヨーレートが初期擾乱の10%以下
        idx_1s = int(np.argmax(t >= 1.0)) if np.any(t >= 1.0) else len(t) - 1
        window_1s = (t >= max(0.0, t[idx_1s] - 0.02)) & (t <= t[idx_1s] + 0.02)
        omega_1s = float(np.mean(np.abs(omega[window_1s]))) if np.any(window_1s) else float(abs(omega[idx_1s]))
        ok_decay = abs(omega0) > 1e-9 and omega_1s <= 0.10 * abs(omega0)
        checks.append(dict(name=f"decay_10pct_{label}", expected=f"<=10%*|omega0|={0.10*abs(omega0):.4f}rad/s",
                            actual=omega_1s, tol="10%", ok=ok_decay))

        # 判定2: ヘディングが発散しない（新しい定常方位への整定は可 — 非ホロノミック系では
        # 方位は積分器であり、「擾乱が増幅しない」ことが判定基準。ω が判定1で減衰済みなら、
        # 後半1s [1s,2s] での方位変化がごく小さいことが「増幅していない」ことの直接証拠になる。
        # 新しい定常方位そのものの値は合否対象外。教材コメントとして本文に明記する。
        mask_1s = (t >= max(0.0, t[idx_1s] - 0.05)) & (t <= t[idx_1s] + 0.05)
        yaw_at_1s = float(np.mean(log['yaw'][mask_1s])) if np.any(mask_1s) else float(log['yaw'][idx_1s])
        yaw_at_2s = float(log['yaw'][-1])
        heading_drift_2nd_half = abs(yaw_at_2s - yaw_at_1s)
        ok_heading = heading_drift_2nd_half <= HEADING_BOUND
        checks.append(dict(name=f"heading_bounded_{label}", expected=f"<={HEADING_BOUND}rad(1s->2s方位変化)",
                            actual=heading_drift_2nd_half, tol="-", ok=ok_heading))

        # 判定3: 横変位が有界（<0.5m）。外乱直前の進行方向を基準とした横方向変位で評価する
        dx = log['x'] - x0
        dy = log['y'] - y0
        lateral = -np.sin(yaw0) * dx + np.cos(yaw0) * dy
        max_lateral = float(np.max(np.abs(lateral)))
        ok_lateral = max_lateral < 0.5
        checks.append(dict(name=f"lateral_bounded_{label}", expected="<0.5m", actual=max_lateral, tol="-",
                            ok=ok_lateral))

        # 参考記録（合否対象外）: 前スキッドのヨー減衰解析式 M_damp=-(mu_c*N_c*x_f^2/v)*omega
        # による減衰時定数のオーダー照合と、車輪差動逆起電力減衰 b_rot を含む合成参考値
        # tau_yaw≈I_eff/(b_rot+mu_c*N_c*x_f^2/v)。すべて実行時導出。
        v_for_damp = v_settled if abs(v_settled) > 1e-6 else v_target
        c_yaw_caster = mu_caster * N_c_static * x_f ** 2 / abs(v_for_damp)
        c_yaw_total = D.b_rot + c_yaw_caster
        tau_yaw_theory = I_eff / c_yaw_total if c_yaw_total > 0 else float('nan')
        _, tau_yaw_meas, r2_decay = fit_first_order_rise(t, omega, y0=omega0)

        slip_saturation_max = None
        if label.startswith("high"):
            # 高速水準のみ: 車輪接触の横力/(mu*法線力) 飽和率の最大値を併記する（合否対象外）
            slip_ratios = []
            force6 = np.zeros(6)
            wheel_geoms = {h.left_wheel_geom_id, h.right_wheel_geom_id}
            for _ in range(int(round(0.5 / dt_phys))):
                sim.data.ctrl[0] = V0
                sim.data.ctrl[1] = V0
                mujoco.mj_step(sim.model, sim.data)
                tracker.update(sim.data)
                for i in range(sim.data.ncon):
                    c = sim.data.contact[i]
                    pair = (c.geom1, c.geom2)
                    if h.floor_geom_id in pair:
                        other = c.geom1 if c.geom2 == h.floor_geom_id else c.geom2
                        if other in wheel_geoms:
                            mujoco.mj_contactForce(sim.model, sim.data, i, force6)
                            normal = abs(float(force6[0]))
                            tangential = float(np.hypot(force6[1], force6[2]))
                            if normal > 1e-9:
                                slip_ratios.append(tangential / (mu_wheel * normal))
            slip_saturation_max = float(np.max(slip_ratios)) if slip_ratios else float('nan')

        qacc_ok, qacc_msg = tracker.check(strict_limit=None)
        checks.append(dict(name=f"qacc_monitor_loose_{label}", expected="OK", actual=qacc_msg, tol="loose",
                            ok=qacc_ok))

        calib_levels[label] = dict(
            v_target=v_target, V0=V0, v_settled=v_settled, omega0=omega0, omega_1s=omega_1s,
            tau_yaw_theory=tau_yaw_theory, tau_yaw_meas=tau_yaw_meas, decay_fit_r2=r2_decay,
            c_yaw_caster=c_yaw_caster, c_yaw_total=c_yaw_total,
            heading_drift_2nd_half=heading_drift_2nd_half, max_lateral=max_lateral,
            slip_saturation_max=slip_saturation_max,
        )

    CALIB['test9_levels'] = calib_levels
    ok, msg = _finish("Test9_yaw_stability", checks, calib=dict(
        v_max_est=v_max_est, I_eff=I_eff, tau_pulse=tau_pulse, x_f=x_f, levels=calib_levels))
    assert ok, msg


# ======================================================================
# S1: 質量・慣性（計画書 §7 S1、モデル読込+短い静定のみ）
# ======================================================================
def test_10_s1_mass_inertia(params, open_xml, D):
    checks = []
    model = mujoco.MjModel.from_xml_path(open_xml)
    h = get_handles(model)
    M = float(model.body_subtreemass[h.mouse_body_id])
    ok_mass = 0.090 <= M <= 0.120
    checks.append(dict(name="total_mass_kg", expected="[0.090,0.120]", actual=M, tol="-", ok=ok_mass))

    m_wheel = float(model.body_mass[h.left_wheel_body_id])
    gsize = model.geom_size[h.left_wheel_geom_id]
    volume_m3 = float(np.pi * (gsize[0] ** 2) * (2 * gsize[1]))
    density_g_cm3 = (m_wheel * 1000.0) / (volume_m3 * 1e6)
    ok_density = 0.8 <= density_g_cm3 <= 2.0
    checks.append(dict(name="wheel_density_g_cm3", expected="[0.8,2.0]", actual=density_g_cm3, tol="-",
                        ok=ok_density))

    motorbox1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'motorbox1')
    motorbox2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'motorbox2')
    wheel_motor_mass = (float(model.body_mass[h.left_wheel_body_id]) + float(model.body_mass[h.right_wheel_body_id])
                          + float(model.body_mass[motorbox1_id]) + float(model.body_mass[motorbox2_id]))
    ratio = wheel_motor_mass / M
    ok_ratio = ratio <= 0.45
    checks.append(dict(name="wheel_motor_mass_ratio", expected="<=0.45", actual=ratio, tol="-", ok=ok_ratio))

    izz_info = compute_izz_breakdown(model)
    ok_izz = izz_info['fraction'] < 0.40
    checks.append(dict(name="izz_wheel_motor_fraction", expected="<0.40", actual=izz_info['fraction'], tol="-",
                        ok=ok_izz))

    sim = MouseSim(open_xml, params=params)
    hh = get_handles(sim.model)
    sim.full_reset(cell=(0, 0), heading_deg=0)
    settle_contact(sim.model, sim.data, hh)
    contacts = measure_contacts_avg(sim.model, sim.data, hh, ctrl=(0.0, 0.0), n_samples=100)
    eta = contacts.N_wheel / (M * abs(sim.model.opt.gravity[2]))
    ok_eta = eta >= 0.55
    checks.append(dict(name="driven_wheel_load_ratio_eta", expected=">=0.55", actual=eta, tol="-", ok=ok_eta))

    CALIB['s1_eta'] = eta
    ok, msg = _finish("S1_mass_inertia", checks, calib=dict(izz=izz_info, eta=eta,
                                                              wheel_motor_mass_ratio=ratio, total_mass=M))
    assert ok, msg


# ======================================================================
# S2: 接触（計画書 §7 S2）
# ======================================================================
def test_11_s2_contact(params, open_xml, xml_dir):
    checks = []
    sim = MouseSim(open_xml, params=params)
    h = get_handles(sim.model)
    sim.full_reset(cell=(0, 0), heading_deg=0)
    settle_contact(sim.model, sim.data, h)
    contacts = measure_contacts(sim.model, sim.data, h)

    mu_c_expect = parse_mu_caster(params)
    ok_caster = contacts.mu_caster is not None and abs(contacts.mu_caster - mu_c_expect) < 1e-6
    checks.append(dict(name="caster_floor_friction0", expected=mu_c_expect, actual=contacts.mu_caster,
                        tol="1e-6", ok=ok_caster))

    ok_wheel = contacts.mu_wheel is not None and abs(contacts.mu_wheel - 1.0) < 1e-6
    checks.append(dict(name="wheel_floor_friction0", expected=1.0, actual=contacts.mu_wheel, tol="1e-6",
                        ok=ok_wheel))

    wall_mu = measure_wall_contact_friction(params, xml_dir)
    mu_wall_expect = float(params.wall_body_friction.split()[0])
    ok_wall = wall_mu is not None and abs(wall_mu - mu_wall_expect) < 1e-6
    checks.append(dict(name="body_wall_friction0", expected=mu_wall_expect, actual=wall_mu, tol="1e-6",
                        ok=ok_wall))

    model = mujoco.MjModel.from_xml_path(open_xml)
    floor_id = get_handles(model).floor_geom_id
    timeconst = float(model.geom_solref[floor_id][0])
    ok_tc = abs(timeconst - 0.002) < 1e-9
    checks.append(dict(name="solref_timeconst", expected=0.002, actual=timeconst, tol="-", ok=ok_tc))

    ts = float(model.opt.timestep)
    ratio_ts = timeconst / ts
    ok_ratio_ts = abs(ratio_ts - 4.0) < 1e-6
    checks.append(dict(name="timeconst_over_timestep_ratio", expected=4.0, actual=ratio_ts, tol="-",
                        ok=ok_ratio_ts))

    ok, msg = _finish("S2_contact", checks)
    assert ok, msg


# ======================================================================
# S3: センサ（計画書 §7 S3）
# ======================================================================
def test_12_s3_sensors(params, open_xml):
    checks = []
    model = mujoco.MjModel.from_xml_path(open_xml)
    data = mujoco.MjData(model)
    h = get_handles(model)
    # open_xml のロボット初期姿勢は euler="0 0 90"（mouse/mjcf.py build_open_floor_xml）で
    # 焼き込まれているため、機体ローカル座標系そのものと比較するにはルート姿勢を
    # 恒等姿勢（yaw=0）へ上書きしてから mj_forward する（S3 は姿勢非依存の幾何検査のため）。
    data.qpos[h.root_qpos_adr:h.root_qpos_adr + 3] = [0.0, 0.0, 0.05]
    data.qpos[h.root_qpos_adr + 3:h.root_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    names = [s['name'] for s in params.sensors]
    rf_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, n) for n in names]
    ok_count = all(i >= 0 for i in rf_ids) and len(rf_ids) == 6
    checks.append(dict(name="rangefinder_count_and_names", expected=6, actual=len(rf_ids), tol="-", ok=ok_count))

    for n, i in zip(names, rf_ids):
        cutoff = float(model.sensor_cutoff[i]) if i >= 0 else float('nan')
        ok = i >= 0 and abs(cutoff - params.sensor_cutoff) < 1e-9
        checks.append(dict(name=f"cutoff_{n}", expected=params.sensor_cutoff, actual=cutoff, tol="1e-9", ok=ok))

    accel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'Accel')
    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'Gyro')
    checks.append(dict(name="accel_exists", expected=True, actual=accel_id >= 0, tol="-", ok=accel_id >= 0))
    checks.append(dict(name="gyro_exists", expected=True, actual=gyro_id >= 0, tol="-", ok=gyro_id >= 0))

    for s in params.sensors:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"visual_{s['name']}")
        pos_expect = np.array([float(v) for v in s['pos'].split()])
        pos_actual = np.array(model.body_pos[body_id])
        ok_pos = bool(np.allclose(pos_actual, pos_expect, atol=1e-6))
        checks.append(dict(name=f"pos_{s['name']}", expected=pos_expect.tolist(), actual=pos_actual.tolist(),
                            tol="1e-6", ok=ok_pos))

        dir_expect = np.array([float(v) for v in s['zaxis'].split()])
        dir_expect = dir_expect / np.linalg.norm(dir_expect)
        xmat = data.xmat[body_id].reshape(3, 3)
        dir_actual = xmat[:, 2]
        ok_dir = bool(np.allclose(dir_actual, dir_expect, atol=1e-6))
        checks.append(dict(name=f"zaxis_{s['name']}", expected=dir_expect.tolist(), actual=dir_actual.tolist(),
                            tol="1e-6", ok=ok_dir))

    ok, msg = _finish("S3_sensors", checks)
    assert ok, msg


# ======================================================================
# S4: 転倒余裕（計画書 §7 S4）
# ======================================================================
def test_13_s4_tipover(params, open_xml, D):
    margin_info = compute_tipover_margin(params, open_xml, D)
    ok = margin_info['margin'] >= 1.5
    checks = [dict(name="tipover_margin", expected=">=1.5", actual=margin_info['margin'], tol="-", ok=ok)]
    ok2, msg = _finish("S4_tipover_margin", checks, calib=margin_info)
    assert ok2, msg


# ======================================================================
# 接触健全性（計画書 §7、C2 是正の最終確認）
# ======================================================================
def test_14_contact_health(params, xml_dir, D):
    checks = []
    M = D.M
    dt = params.physics_dt
    wall_x = 0.5
    wall_near_face = wall_x - 0.006
    box_half = 0.02

    def build_and_run(v0, phase):
        base_gap = v0 * 0.01
        offset = phase * (v0 * dt) / 21.0
        x0 = wall_near_face - box_half - base_gap + offset
        jig_path = str(Path(xml_dir) / f"impact_{v0:.1f}_{phase}.xml")
        build_impact_jig(params, jig_path, mass=M, wall_x=wall_x)
        model = mujoco.MjModel.from_xml_path(jig_path)
        data = mujoco.MjData(model)
        jadr = model.jnt_qposadr[0]
        dadr = model.jnt_dofadr[0]
        data.qpos[jadr] = x0
        data.qvel[dadr] = v0
        mujoco.mj_forward(model, data)
        min_dist = 0.0
        n = int(round(0.1 / dt))
        for _ in range(n):
            mujoco.mj_step(model, data)
            for c in range(data.ncon):
                dd = float(data.contact[c].dist)
                if dd < min_dist:
                    min_dist = dd
        x_final = float(data.qpos[jadr])
        tunneled = x_final > wall_x
        return -min_dist, tunneled

    max_pen_3 = 0.0
    for phase in range(21):
        pen, _ = build_and_run(3.0, phase)
        max_pen_3 = max(max_pen_3, pen)
    ok_pen = max_pen_3 < 0.006
    checks.append(dict(name="max_penetration_v3", expected="<6mm", actual=max_pen_3, tol="-", ok=ok_pen))

    tunnel_any = False
    max_pen_5 = 0.0
    for phase in range(21):
        pen5, tun = build_and_run(5.0, phase)
        max_pen_5 = max(max_pen_5, pen5)
        tunnel_any = tunnel_any or tun
    ok_tunnel = not tunnel_any
    checks.append(dict(name="tunneling_v5", expected=False, actual=tunnel_any, tol="-", ok=ok_tunnel))

    ok, msg = _finish("ContactHealth_wall_penetration", checks,
                       calib=dict(max_pen_3=max_pen_3, max_pen_5=max_pen_5))
    assert ok, msg
