# note_004 素材: r4 反映後の検証スイート残不合格 4 件の原因診断スクリプト
# Diagnostics for the 4 remaining failures after plan revision r4.
#
# 実行: .venv/bin/python research_notes/scripts/note_004_r4_diagnostics.py
#
# 実験1 (Test8): 急制動ピッチ 13〜15° の発生区間の特定
#   仮説「停止後の後進全開発進で発生」→ 棄却（最大ピッチは制動中 v>0 で発生）
#   確定機序: 逆転制動（プラギング）の反動トルク ≈0.089 N·m ≫ 重力復元 ≈0.006 N·m
#   が車体を後傾させ、バンプストップ上でホッピングする（軽量・高トルク車の実在物理）
#
# 実験2 (Test3): Ω_ss 理論 7.11 rad/s vs 実測 4.93 rad/s (−29%) の欠損トルク源の切り分け
#   (a) キャスター μ≈0 化 → +0.37 rad/s のみ（理論のキャスター項 0.37 と厳密一致 = 理論項は正確）
#   (b) さらに frictionloss=0 → 8.71 rad/s（理論 11.12、欠損 ≈3.2 mN·m が残存）
#   確定機序: 車輪シリンダ接触は線接触が 2 点接触に離散化され、超信地旋回中に接触点間
#   （車輪幅 ≈7 mm）の速度差スリップが実効捩れ抵抗を生む（タイヤ幅スクラブ）。
#   スクラブトルク理論 τ_scrub = μ·η·M·g·w/2 ≈ 3.16 mN·m を組み込むと:
#     現行条件:  Ω 予測 4.74 vs 実測 4.93（−4%）
#     fl=0条件: Ω 予測 8.74 vs 実測 8.71（−0.3%）
#   → 両条件で定量一致（欠落項の同定完了）
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_REPO)
sys.path.insert(0, _REPO)

import mujoco  # noqa: E402

from mouse.params import RobotParams  # noqa: E402
from mouse.sim import MouseSim  # noqa: E402


def experiment_1_test8_pitch():
    """急制動ピッチの発生区間（制動中 vs 停止後の後進加速）を時系列で分離する。"""
    print("=== 実験1: Test8 急制動ピッチの発生区間 ===")
    sim = MouseSim("assets/mouse_v2.xml")
    sim.full_reset()
    for _ in range(100):
        sim.step_control(0.0, 0.0)
    while sim.privileged_velocity()[0] < 2.5:
        sim.step_control(3.0, 3.0)
    mid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
    t0 = sim.sim_time
    v_zero_t = None
    max_pitch_brake = max_pitch_reverse = 0.0
    for _ in range(50):
        sim.step_control(-3.0, -3.0)
        v = sim.privileged_velocity()[0]
        xmat = sim.data.xmat[mid].reshape(3, 3)
        pitch = abs(np.degrees(np.arcsin(np.clip(-xmat[2, 0], -1, 1))))
        if v_zero_t is None and v <= 0.0:
            v_zero_t = sim.sim_time - t0
        if v_zero_t is None:
            max_pitch_brake = max(max_pitch_brake, pitch)
        else:
            max_pitch_reverse = max(max_pitch_reverse, pitch)
    print(f"v=0 到達 t={v_zero_t:.2f}s / 制動区間の最大|ピッチ|={max_pitch_brake:.2f}° / "
          f"停止後の最大|ピッチ|={max_pitch_reverse:.2f}°")
    p = RobotParams()
    tau_plug = 2 * (p.gainprm0 * 3.0 + (-p.biasprm2) * 180.0)  # 逆転制動時の反動トルク目安
    print(f"プラギング反動トルク目安 ≈{tau_plug*1000:.1f} mN·m vs 重力復元 ≈6 mN·m")


def _spin(xml_path, label, vd=0.3, secs=4.0):
    sim = MouseSim(xml_path)
    sim.full_reset()
    for _ in range(100):
        sim.step_control(0.0, 0.0)
    gz = []
    for _ in range(int(secs / 0.01)):
        sim.step_control(vd, -vd)
        gz.append(sim.observation()[11])
    w = abs(float(np.mean(gz[-50:])))
    print(f"  [{label}] Ω_ss = {w:.2f} rad/s")
    return w


def experiment_2_test3_yaw_drag():
    """Ω_ss 欠損トルクの切り分け（キャスター / frictionloss / 車輪幅スクラブ）。"""
    print("=== 実験2: Test3 ヨー抗力の切り分け ===")
    _spin("assets/mouse_v2.xml", "現行 (μ_c=0.08)")
    xml = open("assets/mouse_v2.xml").read()
    xml2 = xml.replace('friction="0.08 0.08 1e-4 1e-4 1e-4"',
                       'friction="1e-4 1e-4 1e-4 1e-4 1e-4"')
    open("/tmp/note004_nocaster.xml", "w").write(xml2)
    _spin("/tmp/note004_nocaster.xml", "キャスターμ≈0")
    import re
    xml3 = re.sub(r'frictionloss="[0-9.e-]+"', 'frictionloss="0"', xml2)
    open("/tmp/note004_nofl.xml", "w").write(xml3)
    _spin("/tmp/note004_nofl.xml", "キャスターμ≈0 + frictionloss=0")

    p = RobotParams()
    lever = (0.072 / 2) / p.wheel_radius
    b_rot = 2 * ((-p.biasprm2) + p.wheel_damping) * lever ** 2
    drive = 2 * (p.gainprm0 * 0.3 / p.wheel_radius) * (0.072 / 2)
    fl = 2 * (p.wheel_frictionloss / p.wheel_radius) * (0.072 / 2)
    # 車輪幅スクラブ（2点接触の速度差スリップ）: w = 車輪幅 = 2×half-width(0.0035)
    w_wheel = 0.007
    eta = 0.87  # 静的輪荷重比率（S1 実測）
    scrub = 1.0 * eta * 0.106 * 9.81 * w_wheel / 2
    print(f"  理論分解: drive={drive*1e3:.2f} / frictionloss項={fl*1e3:.2f} / "
          f"スクラブ={scrub*1e3:.2f} [mN·m], b_rot={b_rot*1e3:.3f} mN·m·s")
    print(f"  Ω予測(全項) = {(drive-fl-4.86e-4-scrub)/b_rot:.2f} rad/s（実測 ≈4.93）")
    print(f"  Ω予測(fl=0,キャスター無,スクラブのみ) = {(drive-scrub)/b_rot:.2f} rad/s（実測 ≈8.71）")


if __name__ == "__main__":
    experiment_1_test8_pitch()
    experiment_2_test3_yaw_drag()
