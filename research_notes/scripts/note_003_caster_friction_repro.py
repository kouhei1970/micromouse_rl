# note_003 再現スクリプト: キャスター摩擦欠陥の検証数値を再生成する
# Reproduction script for note_003: regenerates the caster-friction-flaw numbers.
#
# 使い方 / Usage:
#   .venv/bin/python research_notes/scripts/note_003_caster_friction_repro.py
#
# 前提: assets/mouse_v2.xml は修正済み（キャスターに priority="1" condim="1"）。
# 本スクリプトは修正版 XML から priority/condim を剥がした「バグ版」を一時生成し、
# 修正前後の挙動を並べて再現する。
# Note: the numbers depend on mouse/params.py values at the time of note_003
# (FA-130RA-class placeholder constants, total mass 106 g).
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_REPO)
sys.path.insert(0, _REPO)

import mujoco  # noqa: E402

from mouse.params import RobotParams  # noqa: E402
from mouse.sim import MouseSim  # noqa: E402

P = RobotParams()
# 無負荷終端角速度の分母: 逆起電力による実効粘性 + ジョイント粘性減衰
_DENOM = (P.gear_ratio**2 * P.motor_Kt * P.motor_Ke / P.motor_R) + P.wheel_damping


def omega_noload(v):
    """無負荷理論終端車輪角速度 [rad/s]"""
    return (P.gear_ratio * P.motor_Kt * v / P.motor_R) / _DENOM


def make_bugged_xml(tmp_path):
    """修正版 XML からキャスターの priority/condim を剥がしてバグ版を再現する。"""
    xml = open("assets/mouse_v2.xml").read()
    bugged = xml.replace(' priority="1" condim="1"', "")
    assert bugged != xml, "修正属性が見つからない (assets/mouse_v2.xml は修正版か?)"
    open(tmp_path, "w").write(bugged)
    return tmp_path


def geom_name(model, gid):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    if n:
        return n
    b = model.geom_bodyid[gid]
    return f"geom@{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)}"


def contact_table(sim):
    """現在の接触を (ペア名, 実効μ, 法線力計, 接線力計) で集計する。"""
    agg = {}
    for ci in range(sim.data.ncon):
        c = sim.data.contact[ci]
        F = np.zeros(6)
        mujoco.mj_contactForce(sim.model, sim.data, ci, F)
        pair = (geom_name(sim.model, c.geom1), geom_name(sim.model, c.geom2))
        a = agg.setdefault(pair, [float(c.friction[0]), 0.0, 0.0])
        a[1] += float(F[0])
        a[2] += float(np.hypot(F[1], F[2]))
    return agg


def const_voltage_run(xml_path, v, seconds=3.0):
    sim = MouseSim(xml_path)
    sim.full_reset()
    n = int(seconds / P.control_dt)
    ncon_zero = 0
    max_qacc = 0.0
    w_hist = []
    for _ in range(n):
        sim.step_control(v, v)
        if sim.data.ncon == 0:
            ncon_zero += 1
        qa = float(np.max(np.abs(sim.data.qacc)))
        if np.isfinite(qa):
            max_qacc = max(max_qacc, qa)
        obs = sim.observation()
        w_hist.append(0.5 * (obs[12] + obs[13]))
    w_term = float(np.mean(w_hist[-20:]))
    vx, _ = sim.privileged_velocity()
    return dict(w=w_term, vx=vx, ncon0=ncon_zero / n, max_qacc=max_qacc)


def airborne_test(xml_path):
    """空中無負荷試験: 接触を排除しモータ+ジョイントモデルのみを検証する。"""
    sim = MouseSim(xml_path)
    sim.full_reset()
    sim.model.opt.gravity[:] = 0
    sim.data.qpos[2] = 0.5
    mujoco.mj_forward(sim.model, sim.data)
    out = []
    for v in (0.3, 1.5, 3.0):
        sim.data.qvel[:] = 0
        for _ in range(300):
            sim.step_control(v, v)
        obs = sim.observation()
        out.append((v, 0.5 * (obs[12] + obs[13]), omega_noload(v)))
    return out


def main():
    bugged = make_bugged_xml("/tmp/note003_mouse_v2_bugged.xml")
    fixed = "assets/mouse_v2.xml"

    print("=== 1. 静的輪荷重（修正前モデル・0V 2s 静置） ===")
    sim = MouseSim(bugged)
    sim.full_reset()
    for _ in range(200):
        sim.step_control(0.0, 0.0)
    m_total = float(sum(sim.model.body_mass[1:]))
    print(f"総質量 {m_total*1000:.1f} g / 重量 {m_total*9.81:.3f} N")
    fw = 0.0
    for pair, (mu, fn, ft) in contact_table(sim).items():
        print(f"  {pair[0]} <-> {pair[1]}: 実効μ={mu:.3f}, 法線力={fn:.3f} N")
        if "wheel" in pair[0] + pair[1]:
            fw += fn
    print(f"車輪法線力合計 {fw:.3f} N (= 重量の {fw/(m_total*9.81)*100:.0f}%)")

    print("\n=== 2. 0.3V 駆動 2s 後の接触力（修正前: キャスターが接線力を発生） ===")
    for _ in range(200):
        sim.step_control(0.3, 0.3)
    for pair, (mu, fn, ft) in contact_table(sim).items():
        print(f"  {pair[0]} <-> {pair[1]}: 実効μ={mu:.3f}, 法線力={fn:.3f} N, 接線力={ft:.3f} N")
    obs = sim.observation()
    print(f"車輪ω={(obs[12]+obs[13])/2:.2f} rad/s (無負荷理論 {omega_noload(0.3):.2f})")

    print("\n=== 3. 空中無負荷モータ試験（修正前モデルでも一致 = モータモデルは正しい） ===")
    for v, w, w_th in airborne_test(bugged):
        print(f"  V={v}V: ω={w:.2f} rad/s, 理論={w_th:.2f} rad/s, 差={(w/w_th-1)*100:+.2f}%")

    print("\n=== 4. 電圧スイープ 修正前 vs 修正後（3s 定電圧） ===")
    print(f"{'V':>5} | {'修正前ω':>9} {'ncon=0%':>8} | {'修正後ω':>9} {'ncon=0%':>8} {'max|qacc|':>10} | {'無負荷理論':>9}")
    for v in (0.3, 1.5, 3.0):
        rb = const_voltage_run(bugged, v)
        rf = const_voltage_run(fixed, v)
        print(f"{v:>5} | {rb['w']:>9.1f} {rb['ncon0']*100:>7.0f}% | "
              f"{rf['w']:>9.1f} {rf['ncon0']*100:>7.0f}% {rf['max_qacc']:>10.1f} | {omega_noload(v):>9.1f}")

    print("\n=== 5. 中間修正 (priority のみ・condim=3 のまま) の数値不安定の再現 ===")
    xml = open(fixed).read().replace(' priority="1" condim="1"', ' priority="1"')
    open("/tmp/note003_mouse_v2_prio_only.xml", "w").write(xml)
    r = const_voltage_run("/tmp/note003_mouse_v2_prio_only.xml", 0.0, seconds=5.0)
    print(f"0V 静置 5s: max|qacc|={r['max_qacc']:.1f}  (修正完全版では ~700 程度、"
          f"condim=3 のままだと摩擦円錐退化で 1e6 超の発散が発生しうる)")


if __name__ == "__main__":
    main()
