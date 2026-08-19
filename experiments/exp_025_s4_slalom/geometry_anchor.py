#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_025 錨 A3: 四分円スラロームの幾何余裕と所要時間を、**走行データを一切使わずに**計算する。

事前登録 `PREREG.md` §9 A3。走行前に実行し、余裕が負なら投入しない（§12 の投入前確認）。

    .venv/bin/python experiments/exp_025_s4_slalom/geometry_anchor.py

値の出所（このファイルでは一切独自に決めない）:
  - 区画 180mm・壁厚 12mm・柱一辺 12mm … `mouse/mjcf.py`（WALL_THICKNESS / POST_SIZE）と
    `mouse/params.py`（cell_size）
  - 機体の最外側半幅 40mm … `docs/ROBOT_SPEC.md` §2.1
    （`mouse/maze6_env.py:155` の _GEO_CLEARANCE = WALL_THICKNESS/2 + 0.040 と一致する）
  - シャーシ外形長 100mm … `mouse/params.py` chassis_length
  - 巡航速度 0.12 m/s … `classic/motion.py` DEFAULT_V_CRUISE
  - 摩擦の上限 … `research_notes/scripts/check_physical_limits_and_ideal_lap.py` と同じ式・同じ定数

【幾何の置き方】
入ってくる通路の中心線と、出ていく通路の中心線に接する四分円を引く。
半径を `R = cell_size/2` に取ると、円弧の中心は**内側の柱の中心**に一致し、
円弧は旋回区画の入口境界から出口境界までにちょうど収まる（区画中心線から外れない）。

  内側の余裕 = (R - 機体半幅) - 柱の半対角
  外側の余裕 = (弧中心から外壁内面までの距離) - (機体の外側前角の掃引半径)
"""
import math
import sys

# ---- 正本から引いた値 ----
CELL = 0.180        # mouse/params.py cell_size
WALL = 0.012        # mouse/mjcf.py WALL_THICKNESS
POST = 0.012        # mouse/mjcf.py POST_SIZE（一辺）
HALF_W = 0.040      # docs/ROBOT_SPEC.md §2.1 機体の最外側半幅
HALF_L = 0.050      # mouse/params.py chassis_length=0.100 の半分
V_CRUISE = 0.12     # classic/motion.py DEFAULT_V_CRUISE
OMEGA_CRUISE = 1.2  # classic/motion.py DEFAULT_OMEGA_CRUISE（その場旋回専用）
MU, MU_C, ETA, G = 1.0, 0.08, 0.6, 9.81  # check_physical_limits_and_ideal_lap.py と同じ

R_ARC = CELL / 2.0

# ---- 弧そのもの ----
arc_len = R_ARC * math.pi / 2.0
t_arc = arc_len / V_CRUISE
omega_needed = V_CRUISE / R_ARC
a_lat = V_CRUISE ** 2 / R_ARC
A_LAT = (MU * ETA + MU_C * (1 - ETA)) * G      # 横加速度の上限

# ---- 余裕 ----
# 内側: 弧の中心 = 内側の柱の中心。機体の最も内側の点は半径 R-HALF_W。
inner_clear = (R_ARC - HALF_W) - POST / 2.0 * math.sqrt(2.0)
# 外側: 弧中心から外壁の内面までは CELL - WALL/2（= 0.174 m）。
#       機体の外側前角の掃引半径は hypot(R+HALF_W, HALF_L)。
d_outer_wall = CELL - WALL / 2.0
r_sweep_outer = math.hypot(R_ARC + HALF_W, HALF_L)
outer_clear = d_outer_wall - r_sweep_outer

rows = [
    ("弧の半径 R", f"{R_ARC * 1000:.1f} mm", "= cell_size/2"),
    ("弧長", f"{arc_len * 1000:.1f} mm", "R·π/2"),
    ("弧 1 回の所要時間", f"{t_arc:.3f} s", f"弧長 / {V_CRUISE} m/s ← 錨 A1 の期待値"),
    ("必要な角速度", f"{omega_needed:.3f} rad/s", f"その場旋回の上限 {OMEGA_CRUISE} を超える → 弧専用の上限が要る"),
    ("横加速度", f"{a_lat:.3f} m/s²", f"摩擦上限 {A_LAT:.2f} m/s² の {a_lat / A_LAT * 100:.2f}%"),
    ("内側（柱）の余裕", f"{inner_clear * 1000:.1f} mm", f"(R−{HALF_W * 1000:.0f}mm) − 柱の半対角"),
    ("外側（壁）の余裕", f"{outer_clear * 1000:.1f} mm", f"{d_outer_wall * 1000:.0f}mm − 外側前角の掃引 {r_sweep_outer * 1000:.1f}mm"),
]
w = max(len(a) for a, _, _ in rows)
for a, b, c in rows:
    print(f"{a:<{w}} : {b:>12}   {c}")

ok = inner_clear > 0 and outer_clear > 0 and a_lat < A_LAT
print()
print(f"判定: {'投入可（余裕はいずれも正）' if ok else '🔴 投入不可'}")
sys.exit(0 if ok else 1)
