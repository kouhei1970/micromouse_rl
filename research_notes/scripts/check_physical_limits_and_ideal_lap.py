#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""物理限界と、サーキット（16×16 外周 1 周）の理想ラップタイムを計算する。

ユーザ裁定 2026-08-15（`research_notes/note_025` 第 0 章）:
**現行制御の真値走行タイムを基準に使わず、物理限界から計算した理想タイムを基準にする。**

パラメータはすべて `docs/MODEL_VERIFICATION_PLAN.md` §4 から引く（値と出典は下の定数の注釈）。
走行は一切しない — 純粋な計算だけである。

    .venv/bin/python research_notes/scripts/check_physical_limits_and_ideal_lap.py
"""
import math

# ===== 正本（docs/MODEL_VERIFICATION_PLAN.md §4）から引いた値 =====
M = 0.106        # kg     総質量
M_eff = 0.1272   # kg     有効並進質量（車輪とロータの慣性を並進換算して加えたもの）
r = 0.0135       # m      車輪半径
N = 5.0          # —      減速比
Rw = 1.07        # Ω      巻線抵抗（FAULHABER 1717T003SR）
Kt = Ke = 1.98e-3  # N·m/A モータのトルク定数（= 逆起電力定数）
Vmax = 3.0       # V      最大印加電圧
tau_c = 9.1e-4   # N·m    車輪軸の乾性摩擦
mu = 1.0         # —      車輪-床の滑り摩擦係数（**やや楽観側と正本が明記**）
mu_c = 0.08      # —      キャスタ-床の滑り摩擦係数
eta = 0.6        # —      駆動輪の荷重比率（設計目標 ≥0.6）
g = 9.81         # m/s²
c_eff = 1.027    # N·s/m  有効粘性（逆起電力＋軸受粘性を並進換算）

# ===== 迷路の幾何 =====
CELL = 0.18      # m      区画の一辺
WALL = 0.012     # m      壁厚
HALF = 0.040     # m      機体の最外側半幅（docs/ROBOT_SPEC.md §2.1）
NCELL = 16

F_stall = 2 * N * Kt * Vmax / (Rw * r)          # 停止時の駆動力 [N]
F_fric = 2 * tau_c / r + mu_c * (1 - eta) * M * g   # 走行抵抗（速度によらない分）[N]
V_TOP = (F_stall - F_fric) / c_eff              # 最高速度 [m/s]
A_TR = mu * eta * g - mu_c * (1 - eta) * g      # 前後の最大加減速（トラクション）[m/s²]
A_LAT = (mu * eta + mu_c * (1 - eta)) * g       # 横加速度の上限 [m/s²]

W_C = (CELL - WALL) / 2 - HALF                  # 機体中心が動ける横方向の余地（片側）[m]
SIDE = (NCELL - 1) * CELL                       # 外周区画の中心線がなす正方形の一辺 [m]
# 通路幅いっぱいに取れる 90° の最大半径。
# 導出: 直線を外側（中心線から −W_C）に寄せ、円弧を内側の角に接するまで大きくする。
# 内側の角 (−W_C, +W_C) と円弧の中心 (W_C−R, R−W_C) の距離 |R−2W_C|·√2 が R 以下、から
# R ≤ (4+2√2)·W_C。等号のとき円弧は内側の角にちょうど接する（**余裕ゼロ**）。
R_MAX = (4 + 2 * math.sqrt(2)) * W_C


def lap(R, offset, ds=0.0005, n_sweep=3):
    """最小時間の速度プロファイルで 1 周の所要時間を出す。

    R      … コーナー半径 [m]
    offset … 直線を外側へずらす量 [m]（0 なら通路の中心線どおり）
    戻り値 … (所要時間, 直線ぶん, コーナーぶん, 経路長, 最高速, 最低速)

    摩擦円: 横に a_y を使っている間、前後に使える分は
            a_x ≤ A_TR·√(1 − (a_y/A_LAT)²)（前後と横で上限が違うので楕円にする）。
    加速側はさらにモータの出せる力（速度とともに落ちる）でも頭打ちにする。
    周回なので前進パスと後退パスを何周かまわして収束させる。
    """
    straight = SIDE - 2 * (R - offset)
    seg = [("straight", straight, 0.0), ("corner", math.pi * R / 2, 1.0 / R)] * 4
    s, kap, kind = [], [], []
    for k, ln, c in seg:
        n = max(int(ln / ds), 1)
        s += [ln / n] * n
        kap += [c] * n
        kind += [k] * n
    n = len(s)
    v = [min(V_TOP, math.sqrt(A_LAT / abs(c)) if c else V_TOP) for c in kap]
    for _ in range(n_sweep):
        for i in range(n):                        # 前進パス（加速の制約）
            j = (i + 1) % n
            ay = v[i] ** 2 * kap[i]
            ax_cap = A_TR * math.sqrt(max(1 - (ay / A_LAT) ** 2, 0.0))
            ax_mot = (F_stall - c_eff * v[i] - F_fric) / M_eff
            ax = max(min(ax_cap, ax_mot), 0.0)
            v[j] = min(v[j], math.sqrt(max(v[i] ** 2 + 2 * ax * s[i], 0.0)))
        for i in range(n - 1, -1, -1):            # 後退パス（減速の制約）
            j = (i - 1) % n
            ay = v[i] ** 2 * kap[i]
            ax_cap = A_TR * math.sqrt(max(1 - (ay / A_LAT) ** 2, 0.0))
            v[j] = min(v[j], math.sqrt(max(v[i] ** 2 + 2 * ax_cap * s[j], 0.0)))
    T = Ts = Tc = 0.0
    for i in range(n):
        dt = s[i] / max(0.5 * (v[i] + v[(i + 1) % n]), 1e-9)
        T += dt
        if kind[i] == "straight":
            Ts += dt
        else:
            Tc += dt
    return T, Ts, Tc, sum(s), max(v), min(v)


def main():
    print("=== 1. 車両の性能限界（正本の式を再計算して照合）===")
    print(f"  停止時の駆動力 F_stall = {F_stall:.3f} N   走行抵抗 F_fric = {F_fric:.4f} N")
    print(f"  最高速度       = {V_TOP:.3f} m/s   （正本の記載 3.84）")
    print(f"  最大加減速     = {A_TR:.3f} m/s² = {A_TR/g:.3f} G （正本 ≈5.6）")
    print(f"  横加速度上限   = {A_LAT:.3f} m/s² = {A_LAT/g:.3f} G （正本 ≈6.2）")
    print(f"  モータ由来の加速（v=0）= {F_stall/M_eff:.1f} m/s²"
          f"  ⇒ トラクションの方が小さいので、そちらが効く")
    for RR in (0.045, 0.060, 0.090, R_MAX):
        print(f"    コーナリング速度 v(R={RR*1000:5.1f} mm) = √(a_lat·R) = {math.sqrt(A_LAT*RR):.3f} m/s")

    print("\n=== 2. サーキットの定義（16×16 外周 1 周・提案）===")
    print(f"  外周区画の中心線がなす正方形の一辺 = {SIDE:.3f} m（周長 {4*SIDE:.3f} m）")
    print(f"  通路の自由幅 {CELL-WALL:.3f} m ／ 機体中心の横方向の余地 ±{W_C*1000:.1f} mm")
    print(f"  通路幅いっぱいに取れる 90° の最大半径 R_max = {R_MAX*1000:.1f} mm（**余裕ゼロ**）")

    print("\n=== 3. 理想ラップタイム ===")
    rows = [("(a)  現行計画器の幾何 R=60 mm", 0.060, 0.0),
            ("(a') 同 R=90 mm（候補の最大）", 0.090, 0.0),
            ("(b)  経路も理論限界 R=300.5 mm", R_MAX, W_C)]
    out = {}
    for label, R, off in rows:
        T, Ts, Tc, ln, vmx, vmn = lap(R, off)
        out[label] = T
        print(f"  {label}: **{T:.3f} s**   経路長 {ln:.3f} m")
        print(f"      直線 {Ts:.3f} s ({Ts/T*100:.1f}%) ／ コーナー {Tc:.3f} s ({Tc/T*100:.1f}%)"
              f"   最高 {vmx:.3f} / 最低 {vmn:.3f} m/s")
    ta, tb = out[rows[0][0]], out[rows[2][0]]
    print(f"\n  経路計画の伸びしろ = (a) − (b) = {ta-tb:.3f} s（{(ta-tb)/ta*100:.1f} %）")


if __name__ == "__main__":
    main()
