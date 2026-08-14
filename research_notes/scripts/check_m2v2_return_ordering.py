"""
research_notes/scripts/check_m2v2_return_ordering.py
=====================================================
**M2 起案 §2 の凍結報酬形について、割引後総収益の順序を机上で検算する**
（2026-08-14。教授の待機任務 1。**実装はしない。計算のみ**）。

研究計画書 §9-11 の規律「**報酬をスケールの違う課題へ転用するときは総収益を
再計算する**」の適用。前回（`check_m2_return_ordering.py`・2026-08-11）は
M2-0 の実条件（k=8.7e-3・実際のエピソード長）で計算した。今回はその上に
**起案の 2 変更**を乗せる:

  (1) **k = 0**（行動の高周波成分への罰を M2 では掛けない）
  (2) **ゴールの支払いを規約終端へ**（機体中心が区画に入った時点ではなく、
      **機体全体がゴール区画に内包された時点**で +1.0 を払う）

(2) は**ゴール到達が ΔT 歩ぶん遅くなる**ことを意味する。その ΔT は
「中心が境界を越えてから、機体全体が内側に入るまでに進む距離」÷ 1 歩の距離 であり、
**距離は評価ハーネスと同じ外形（`competition/evaluator.body_footprint`）から導く**
（寸法のハードコードをしない。R34 の作法）。

## 検算する順序（研究計画書 §9-8）

    ゴール > 探索（6000 歩の時間切れ）> 滞留 > 衝突

**検証帯 20 面の実際の $D_0$** で面ごとに計算し、**崩れる面があればそのまま報告する**
（起案の是正材料にする。順序が崩れないことを示すのが目的ではない）。

使い方:
    .venv/bin/python research_notes/scripts/check_m2v2_return_ordering.py
"""
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mouse.maze6_gen import generate_maze, shortest_distances  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

GAMMA = 0.995
DT = RobotParams().control_dt          # 0.01 s
CELL = RobotParams().cell_size         # 0.18 m
TIME_PENALTY = 0.001
GOAL_BONUS = 1.0
COLLISION_PENALTY = -1.0
VISIT_BONUS = 0.02
N_CELLS_TOTAL = 36
TIME_LIMIT_STEPS = 6000
VALID_SEEDS = list(range(7000, 7020))

# 参考: exp_010/011 で使った罰の実測（k=8.7e-3・α=0.5 の方策で E‖a−ā‖² = 0.483）
K_OLD, MEAN_HP2 = 8.7e-3, 0.483


def S(T: float) -> float:
    """時間罰の等比和 Σ_{t<T} γ^t。"""
    return (1.0 - GAMMA ** T) / (1.0 - GAMMA)


def visit_discounted(n_cells: int, steps_per_cell: float) -> float:
    """訪問報酬の割引後総和（i 区画目は i·steps_per_cell 歩目に入ると仮定）。"""
    return VISIT_BONUS * sum(GAMMA ** (i * steps_per_cell) for i in range(n_cells))


def containment_depth_m() -> float:
    """中心が境界を越えてから**機体全体が内包される**までに要る進入深さ [m]。

    **外形は評価ハーネスの実装から導く**（`competition/evaluator.body_footprint`）。
    最悪は機体が境界に対して斜めを向いている場合なので、機体座標の 4 隅を回して
    「進行方向に測った最大の張り出し」を全方位で取った値（＝外接円の半径）を使う。
    """
    import tempfile

    import mujoco

    from competition.evaluator import body_footprint
    from mouse.mjcf import build_maze_robot_xml

    m = generate_maze(VALID_SEEDS[0], mode="loop")
    fd, tmp = tempfile.mkstemp(suffix=".xml")
    import os
    os.close(fd)
    try:
        build_maze_robot_xml(m["v_walls"], m["h_walls"], tmp)
        model = mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
    fx0, fx1, fy0, fy1 = body_footprint(model, data, bid)
    # 4 隅の最大半径（向きに依存しない上界。斜め進入でもこの深さがあれば内包される）
    r = max(math.hypot(lx, ly) for lx in (fx0, fx1) for ly in (fy0, fy1))
    return r, (fx0, fx1, fy0, fy1)


def returns_for_face(d0_cells: int, steps_per_cell: float, k: float,
                     hp2: float, dT_contain: float) -> dict:
    """1 面についての割引後総収益（4 つの挙動）。"""
    c = k * hp2                       # 1 歩あたりの滑らかさ罰
    per_step = TIME_PENALTY + c
    D0 = d0_cells * CELL
    Tg = d0_cells * steps_per_cell    # 中心がゴール区画へ入るまでの歩数
    Tg_pay = Tg + dT_contain          # **規約終端で支払う**ぶん遅れる

    goal = (GAMMA ** Tg_pay * D0 - per_step * S(Tg_pay)
            + GAMMA ** (Tg_pay - 1) * GOAL_BONUS
            + visit_discounted(d0_cells, steps_per_cell))
    # 探索して時間切れ: 6000 歩・全 36 区画を訪問・終端 Φ は 0 とみなす（保守側）
    explore = (-per_step * S(TIME_LIMIT_STEPS)
               + visit_discounted(N_CELLS_TOTAL, steps_per_cell))
    # 滞留: 動かないので訪問も進捗も無い。**罰は動いたときだけ掛かる**ので c は乗らない
    #（2026-08-11 の誤りの再発防止: 滞留に走行時の滑らかさ罰を適用しない）
    stay = -TIME_PENALTY * S(TIME_LIMIT_STEPS)
    # 衝突: 最短の半分まで進んで当たる
    Tc = Tg / 2
    collide = (GAMMA ** Tc * (D0 / 2) - per_step * S(Tc)
               + GAMMA ** (Tc - 1) * COLLISION_PENALTY
               + visit_discounted(max(d0_cells // 2, 1), steps_per_cell))
    return dict(goal=goal, explore=explore, stay=stay, collide=collide,
                Tg=Tg, Tg_pay=Tg_pay)


def ordering_ok(r: dict) -> bool:
    return r["goal"] > r["explore"] > r["stay"] > r["collide"]


def failing_pairs(r: dict):
    """順序のどの不等式が破れているかを返す（**どれが崩れたかを名指しする**）。"""
    pairs = (("ゴール>探索", r["goal"] - r["explore"]),
             ("探索>滞留", r["explore"] - r["stay"]),
             ("滞留>衝突", r["stay"] - r["collide"]))
    return [name for name, gap in pairs if gap <= 0], pairs


def main() -> int:
    r_circ, fp = containment_depth_m()
    print("=" * 92)
    print("M2 起案 §2 の凍結報酬形の机上検算（実装なし・計算のみ）")
    print("=" * 92)
    print(f"  γ = {GAMMA}（実効地平 {1/(1-GAMMA):.0f} 歩 = {1/(1-GAMMA)*DT:.0f} s）／"
          f"時間罰 {TIME_PENALTY}／ゴール +{GOAL_BONUS}／衝突 {COLLISION_PENALTY}／"
          f"訪問 +{VISIT_BONUS}")
    print(f"  機体外形（評価ハーネスの body_footprint から導出）: "
          f"x [{fp[0]:.4f}, {fp[1]:.4f}]・y [{fp[2]:.4f}, {fp[3]:.4f}] m")
    print(f"  → **内包に要る進入深さ（向き非依存の上界 = 外接円半径）= {r_circ:.4f} m**")

    print("\n" + "=" * 92)
    print("§1 規約終端化で増える歩数 ΔT（速度依存）")
    print("=" * 92)
    print(f"{'速度[m/s]':>10}{'1 歩[mm]':>10}{'ΔT[歩]':>9}{'ΔT[s]':>8}"
          f"{'γ^ΔT':>9}   意味")
    speeds = (0.25, 0.5, 0.96, 1.5)
    for v in speeds:
        step_m = v * DT
        dT = r_circ / step_m
        print(f"{v:>10.2f}{step_m*1000:>10.1f}{dT:>9.1f}{dT*DT:>8.2f}"
              f"{GAMMA ** dT:>9.4f}   ゴール報酬が {1-GAMMA**dT:.1%} 目減り")

    print("\n" + "=" * 92)
    print("§2 検証帯 20 面での順序（**起案の形: k = 0・規約終端で支払い**）")
    print("=" * 92)
    v = 0.96
    spc = CELL / (v * DT)
    dT = r_circ / (v * DT)
    print(f"  速度 {v} m/s（M1 実測）→ {spc:.2f} 歩/区画・ΔT = {dT:.1f} 歩")
    print(f"{'面':>6}{'D₀':>4}{'T_goal':>8}{'T_pay':>7}"
          f"{'ゴール':>9}{'探索':>9}{'滞留':>9}{'衝突':>9}   順序")
    n_ok = 0
    worst_margin = (1e9, None)
    for seed in VALID_SEEDS:
        m = generate_maze(seed, mode="loop")
        d0 = int(shortest_distances(m["v_walls"], m["h_walls"])[tuple(m["start"])])
        r = returns_for_face(d0, spc, k=0.0, hp2=MEAN_HP2, dT_contain=dT)
        ok = ordering_ok(r)
        n_ok += ok
        margin = min(r["goal"] - r["explore"], r["explore"] - r["stay"],
                     r["stay"] - r["collide"])
        if margin < worst_margin[0]:
            worst_margin = (margin, seed)
        bad, _ = failing_pairs(r)
        print(f"{seed:>6}{d0:>4}{r['Tg']:>8.0f}{r['Tg_pay']:>7.0f}"
              f"{r['goal']:>9.3f}{r['explore']:>9.3f}{r['stay']:>9.3f}{r['collide']:>9.3f}"
              f"   {'✅ 成立' if ok else '🔴 ' + '・'.join(bad) + ' が崩れる'}")
    print(f"\n  **順序が成立した面: {n_ok} / {len(VALID_SEEDS)}**"
          f"／最小の余裕 {worst_margin[0]:+.3f}（面 {worst_margin[1]}）")
    # どの不等式が何面で崩れたかを集計する（「順序が崩れた」で終わらせない）
    tally = {}
    for seed in VALID_SEEDS:
        m = generate_maze(seed, mode="loop")
        d0 = int(shortest_distances(m["v_walls"], m["h_walls"])[tuple(m["start"])])
        bad, _ = failing_pairs(returns_for_face(d0, spc, 0.0, MEAN_HP2, dT))
        for b in bad:
            tally[b] = tally.get(b, 0) + 1
    print("  🔴 崩れた不等式の内訳: "
          + ("／".join(f"{k}: {v} 面" for k, v in sorted(tally.items())) if tally else "なし"))
    print("  ✅ 崩れなかった不等式: "
          + "／".join(k for k in ("ゴール>探索", "探索>滞留", "滞留>衝突") if k not in tally))

    print("\n" + "=" * 92)
    print("§3 変更の寄与を分解する（面 D₀ = 4・9・15 の代表 3 面）")
    print("=" * 92)
    print(f"{'条件':>34}{'ゴール':>9}{'探索':>9}{'滞留':>9}{'衝突':>9}   順序")
    for d0 in (4, 9, 15):
        print(f"  --- D₀ = {d0} 区画 ---")
        cases = [
            ("① 現行 M2-0（k=8.7e-3・中心で支払い）", K_OLD, 0.0),
            ("② k = 0 のみ（中心で支払い）", 0.0, 0.0),
            ("③ 規約終端のみ（k=8.7e-3）", K_OLD, dT),
            ("④ **起案の形**（k = 0・規約終端）", 0.0, dT),
        ]
        for label, k, dt_c in cases:
            r = returns_for_face(d0, spc, k=k, hp2=MEAN_HP2, dT_contain=dt_c)
            print(f"{label:>34}{r['goal']:>9.3f}{r['explore']:>9.3f}"
                  f"{r['stay']:>9.3f}{r['collide']:>9.3f}"
                  f"   {'✅' if ordering_ok(r) else '🔴'}")

    print("\n" + "=" * 92)
    print("§4 速度が遅いとどうなるか（ΔT が伸びる。**起案の形**で）")
    print("=" * 92)
    print(f"{'速度[m/s]':>10}{'歩/区画':>9}{'ΔT[歩]':>8}"
          f"{'D₀=4 の順序':>14}{'D₀=9':>10}{'D₀=15':>10}   最小余裕")
    for vv in speeds:
        spc_v = CELL / (vv * DT)
        dT_v = r_circ / (vv * DT)
        marks, margins = [], []
        for d0 in (4, 9, 15):
            r = returns_for_face(d0, spc_v, k=0.0, hp2=MEAN_HP2, dT_contain=dT_v)
            marks.append("✅" if ordering_ok(r) else "🔴")
            margins.append(min(r["goal"] - r["explore"], r["explore"] - r["stay"],
                               r["stay"] - r["collide"]))
        print(f"{vv:>10.2f}{spc_v:>9.1f}{dT_v:>8.1f}"
              f"{marks[0]:>14}{marks[1]:>10}{marks[2]:>10}   {min(margins):+.3f}")

    print("\n" + "=" * 92)
    print("§5 ΔT はどこまで許されるか（起案の形で、順序が崩れる ΔT の下限）")
    print("=" * 92)
    print(f"{'D₀':>5}{'臨界 ΔT[歩]':>13}{'臨界 ΔT[s]':>12}"
          f"{'進入深さ換算[m]':>17}   束縛")
    for d0 in (4, 6, 9, 12, 15):
        lo, hi = 0.0, 4000.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if ordering_ok(returns_for_face(d0, spc, 0.0, MEAN_HP2, mid)):
                lo = mid
            else:
                hi = mid
        r = returns_for_face(d0, spc, 0.0, MEAN_HP2, lo)
        bad, pairs = failing_pairs(returns_for_face(d0, spc, 0.0, MEAN_HP2, lo + 1e-6))
        bind = "・".join(bad) if bad else min(pairs, key=lambda p: p[1])[0]
        note = "（ΔT=0 でも崩れている）" if lo <= 0.0 else ""
        print(f"{d0:>5}{lo:>13.0f}{lo*DT:>12.2f}{lo*v*DT:>17.3f}   {bind}{note}")
    print("\n  ※ 「進入深さ換算」は速度 0.96 m/s で ΔT 歩ぶん走る距離。"
          "\n     実際に要る深さは §1 の 1 値（外接円半径）なので、余裕の大きさが読める。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
