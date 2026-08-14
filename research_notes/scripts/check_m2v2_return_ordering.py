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


#: 衝突シナリオの「進む割合」α（最短経路のどこまで進んで当たるか）。
#: **模型の中の任意の定数**であり、結論はここに強く依存する（准教授 REVIEW_005）。
#: 既定 0.5 の根拠は exp_011（k=0 の実走）の実測との一致（§7 で再現）。
CRASH_PROGRESS_ALPHA = 0.5


def returns_for_face(d0_cells: int, steps_per_cell: float, k: float,
                     hp2: float, dT_contain: float, forfeit: bool = False,
                     alpha: float = CRASH_PROGRESS_ALPHA,
                     limit_steps: int = TIME_LIMIT_STEPS) -> dict:
    """1 面についての割引後総収益（4 つの挙動）。

    `forfeit=True` は**対処案 (a')**: 衝突罰を「稼いだ整形の没収」型
    **−(1.0 + Φ(s_T))** に置く（教授の追検算指示 2026-08-14）。
    このとき衝突の総収益は

        γ^Tc·Φ_T + γ^(Tc−1)·(−1.0 − Φ_T) − 時間罰
        = γ^(Tc−1)·( −1.0 − (1−γ)·Φ_T ) − 時間罰 + 訪問

    となり、**稼いだ整形 γ^Tc·Φ_T が構成上打ち消される**（残るのは
    −(1−γ)Φ_T の小さな負項なので、**進んでから当たるほど僅かに損**になる）。
    """
    c = k * hp2                       # 1 歩あたりの滑らかさ罰
    per_step = TIME_PENALTY + c
    D0 = d0_cells * CELL
    Tg = d0_cells * steps_per_cell    # 中心がゴール区画へ入るまでの歩数
    Tg_pay = Tg + dT_contain          # **規約終端で支払う**ぶん遅れる

    goal = (GAMMA ** Tg_pay * D0 - per_step * S(Tg_pay)
            + GAMMA ** (Tg_pay - 1) * GOAL_BONUS
            + visit_discounted(d0_cells, steps_per_cell))
    # 探索して時間切れ: 上限まで走り・全 36 区画を訪問・終端 Φ は 0 とみなす（保守側）。
    # ⚠️ 上限 2000 歩でも全 36 区画は幾何的に可能（36 × 18.75 = 675 歩 < 2000）。
    # 変わるのは時間罰の等比和 S(limit) だけである。
    explore = (-per_step * S(limit_steps)
               + visit_discounted(N_CELLS_TOTAL, steps_per_cell))
    # 滞留: 動かないので訪問も進捗も無い。**罰は動いたときだけ掛かる**ので c は乗らない
    #（2026-08-11 の誤りの再発防止: 滞留に走行時の滑らかさ罰を適用しない）
    stay = -TIME_PENALTY * S(limit_steps)
    # 衝突: 最短経路の α 倍まで進んで当たる（終端の Φ は稼いだ進捗 α·D0）
    Tc = Tg * alpha
    phi_T = D0 * alpha
    coll_term = (COLLISION_PENALTY - phi_T) if forfeit else COLLISION_PENALTY
    collide = (GAMMA ** Tc * phi_T - per_step * S(Tc)
               + GAMMA ** (Tc - 1) * coll_term
               + visit_discounted(max(int(d0_cells * alpha), 1), steps_per_cell))
    # 参考: **即時衝突**（1 歩目で当たる。進捗も訪問も無い）。没収型が
    # 「進んでから当たる」を消したときに、**即時衝突より悪くなっていないか**を見る
    coll_now = COLLISION_PENALTY      # γ^0·(−1.0 − Φ=0) は没収型でも同じ
    return dict(goal=goal, explore=explore, stay=stay, collide=collide,
                collide_now=coll_now, Tg=Tg, Tg_pay=Tg_pay)


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

    print("\n" + "=" * 92)
    print("§5-bis 🔴 機構の訂正と、模型の任意定数への感度（准教授 REVIEW_005）")
    print("=" * 92)
    print("  【訂正】前報の「長い面ほど γ^(T−1) で衝突罰が減衰する一方 γ^T·Φ_T は残る」は**誤り**。")
    print("  **両項とも同じ γ^Tc を持つので割引は完全に相殺する。**自分で展開して確認した:")
    print("      R_衝突 − R_滞留 = γ^Tc·(Φ_T + 0.2 − 1/γ) + 訪問 = γ^Tc·(Φ_T − 0.8050) + 訪問")
    print(f"      （0.2 = 時間罰 × S(6000)、1/γ = {1/GAMMA:.6f}）")
    print("  ⇒ **符号を決めるのは Φ_T − 0.805 だけ。速度にも γ にも依存しない。**")
    print(f"  ⇒ 閾値は **Φ_T = 0.805 m = {0.805/CELL:.2f} 区画**（訪問ボーナスがさらに押し下げる）")
    print("  ⇒ 崩れは「長い面で偶然起きる」のではなく"
          "**「一定距離進んで当たれば必ず起きる」構造**である")
    print("\n  【感度】衝突シナリオの「進む割合」α は**模型の中の任意の定数**である:")
    print(f"{'α':>8}{'崩れる面数':>12}{'理論の閾値 D₀':>15}   備考")
    for a in (0.25, 1.0/3.0, 0.5, 2.0/3.0, 1.0):
        n_bad = 0
        for seed in VALID_SEEDS:
            m = generate_maze(seed, mode="loop")
            d0 = int(shortest_distances(m["v_walls"], m["h_walls"])[tuple(m["start"])])
            r = returns_for_face(d0, spc, 0.0, MEAN_HP2, dT, alpha=a)
            if not ordering_ok(r):
                n_bad += 1
        note = "← 本報告が採った値" if abs(a - 0.5) < 1e-9 else ""
        print(f"{a:>8.3f}{n_bad:>12}{0.805/CELL/a:>15.1f}   {note}")
    print("\n  **「14/20」は模型の中の任意の定数が作った数値であり、実測ではない。**")
    print("  **ただし「崩れる面が存在する」という定性的な結論は α ≥ 1/3 で成立する。**")
    print("  α = 0.5 を採った根拠は、次節の実走との一致である。")

    print("\n" + "=" * 92)
    print("§6 対処案 (a') — 衝突罰を「稼いだ整形の没収」型 −(1.0 + Φ(s_T)) に置く")
    print("=" * 92)
    print("  形: 衝突の総収益 = γ^(Tc−1)·( −1.0 − (1−γ)·Φ_T ) − 時間罰 + 訪問")
    print("      → **稼いだ整形 γ^Tc·Φ_T が構成上打ち消される**")
    print(f"{'面':>6}{'D₀':>4}{'ゴール':>9}{'探索':>9}{'滞留':>9}"
          f"{'衝突(現行)':>12}{'衝突(没収型)':>13}{'即時衝突':>10}   順序(没収型)")
    n_ok2, tally2 = 0, {}
    for seed in VALID_SEEDS:
        m = generate_maze(seed, mode="loop")
        d0 = int(shortest_distances(m["v_walls"], m["h_walls"])[tuple(m["start"])])
        r_old = returns_for_face(d0, spc, 0.0, MEAN_HP2, dT, forfeit=False)
        r_new = returns_for_face(d0, spc, 0.0, MEAN_HP2, dT, forfeit=True)
        ok = ordering_ok(r_new)
        n_ok2 += ok
        bad, _ = failing_pairs(r_new)
        for b in bad:
            tally2[b] = tally2.get(b, 0) + 1
        print(f"{seed:>6}{d0:>4}{r_new['goal']:>9.3f}{r_new['explore']:>9.3f}"
              f"{r_new['stay']:>9.3f}{r_old['collide']:>12.3f}{r_new['collide']:>13.3f}"
              f"{r_new['collide_now']:>10.3f}"
              f"   {'✅ 成立' if ok else '🔴 ' + '・'.join(bad)}")
    print(f"\n  **順序が成立した面: {n_ok2} / {len(VALID_SEEDS)}**"
          f"（現行の衝突罰では 6 / 20 だった）")
    print("  崩れた不等式の内訳: "
          + ("／".join(f"{k}: {v} 面" for k, v in sorted(tally2.items())) if tally2 else "**なし**"))

    print("\n  --- 確認点 ---")
    print("  (i)  滞留 > 衝突: "
          + ("**全面で回復**" if "滞留>衝突" not in tally2
             else f"🔴 {tally2['滞留>衝突']} 面で未回復"))
    print("  (ii) 既成立部分（ゴール>探索・探索>滞留）: "
          + ("**壊れていない**" if not ({"ゴール>探索", "探索>滞留"} & set(tally2))
             else "🔴 壊れた"))
    print("  (iii) 方策不変性（Ng ら 1999）: **崩れない**。"
          "\n       不変性が主張するのは**整形項 F = γΦ(s\') − Φ(s) を足しても最適方策が"
          "変わらない**ことであり、\n       本案が変えるのは**元の課題報酬（衝突罰）の側**である。"
          "整形項には手を触れていないので、\n       「新しい課題報酬 ＋ 同じ整形項」の MDP は"
          "「新しい課題報酬だけ」の MDP と同じ最適方策を持つ。\n"
          "       ⚠️ ただし**課題そのものが変わる**ので、"
          "**新しい最適方策が望むものかは別途の判断**が要る\n"
          "       （不変性は『整形が悪さをしない』ことしか保証しない）。")

    respawn_section(spc, dT)

    v2_final_section()

    print("\n" + "=" * 92)
    print("§7 🔴 模型の根拠を実走で再現する（准教授の未確認 ①への回答）")
    print("=" * 92)
    print("  引き継ぎメモの「k=0 では衝突の総収益が 20 面中 15 面で正」は、"
          "3 代目の分析の記述であって\n  本検算では未再現だった。**実走から計算し直す**"
          "（模型ではなく env が返した報酬列から Σ γ^t r_t を求める）。")
    empirical_check()
    return 0


def respawn_section(spc: float, dT: float) -> None:
    """§8 対処案 (c): **衝突リスポーン形**の検算（教授の発注 2026-08-14）。

    仕様: **衝突で終了せず、−1.0 を払って開始点へ戻り、エピソードは継続**
    （上限 6000 歩・ゴールか時間切れでのみ終了）。

    🔴 **鍵は「報酬契約に触れない没収」**である。ポテンシャル整形の恒等式

        Σ_t γ^t (γΦ(s_{t+1}) − Φ(s_t)) = γ^T Φ(s_T) − Φ(s_0)

    は**経路によらず終端の Φ だけで決まる**。開始点へ戻る遷移では
    γΦ(start) − Φ(s_衝突) = **−Φ_c** が 1 度だけ入り、**稼いだ整形をポテンシャル自身が
    取り返す**。しかも**走行中の 1 歩ごとの信号（γΦ(s') − Φ(s)）は一切変わらない**。
    """
    print("\n" + "=" * 92)
    print("§5-ter エピソード上限の裁定（2000 歩）と、その感度")
    print("=" * 92)
    print("  裁定（2026-08-14）: **v2 のエピソード上限 = 2000 歩**"
          "（(c) 導入でゴール以外は必ず上限まで走るため、多様性が上限に直結する）")
    print(f"{'上限[歩]':>10}{'S(上限)':>10}{'探索':>9}{'滞留':>9}"
          f"{'1 本の迷路数':>14}{'順序が崩れる面':>16}")
    for lim in (1000, 2000, 6000):
        r4 = returns_for_face(4, spc, 0.0, MEAN_HP2, dT, limit_steps=lim)
        n_bad = 0
        for seed in VALID_SEEDS:
            m = generate_maze(seed, mode="loop")
            d0 = int(shortest_distances(m["v_walls"], m["h_walls"])[tuple(m["start"])])
            if not ordering_ok(returns_for_face(d0, spc, 0.0, MEAN_HP2, dT,
                                                limit_steps=lim)):
                n_bad += 1
        print(f"{lim:>10}{S(lim):>10.2f}{r4['explore']:>9.3f}{r4['stay']:>9.3f}"
              f"{2_000_000 // lim:>14}{n_bad:>16}")
    print("\n  ⇒ **時間罰の等比和 S は 1000〜6000 歩でほとんど変わらない**"
          "（γ の実効地平 200 歩を超えると飽和するため）。")
    print("  ⇒ **総収益も順序も上限にほぼ不感**であり、**2000 は崖の縁ではない**。")
    print("  ⇒ **上限が効くのは総収益ではなく「1 本で触れる迷路の数」**である"
          "（2000 歩なら 1,000 面/本）。")

    print("\n" + "=" * 92)
    print("§8 対処案 (c): 衝突リスポーン形（−1.0 を払って開始点へ戻り、継続）")
    print("=" * 92)
    print("  【機構】ポテンシャル整形の恒等式 Σ γ^t(γΦ(s')−Φ(s)) = γ^T Φ_T − Φ_0 は"
          "**終端の Φ だけで決まる**。")
    print("  開始点へ戻る遷移で **γΦ(start) − Φ_c = −Φ_c** が 1 度入り、"
          "**稼いだ整形をポテンシャル自身が取り返す**。")
    print("  ⇒ **報酬契約には一切触れない没収**である（(a') は課題報酬側を変えていた）。")
    print("  ⇒ **走行中の 1 歩ごとの信号は変わらない**（(a') が壊した進捗信号は保たれる）。")

    print("\n  (i) 4 終点の順序（k=0・規約終端で支払い・速度 0.96 m/s）")
    print(f"{'α':>6}{'N(衝突回数)':>12}{'面 D₀':>7}"
          f"{'ゴール':>9}{'探索':>9}{'滞留':>9}{'衝突×N→時間切れ':>18}   順序")
    for alpha in (0.25, 0.5, 1.0):
        for n_crash in (1, 3, 10):
            for d0 in (4, 9, 15):
                D0 = d0 * CELL
                Tg = d0 * spc
                phi_c = alpha * D0
                Tc = alpha * Tg                      # 1 回の衝突までの歩数
                # 衝突を N 回繰り返し、その後は 6000 歩まで走って時間切れ。
                # 🔴 是正（2026-08-14・AUDIT_025 と同型の欠陥をここでも発見）:
                # 旧式は各衝突で **−1.0（罰）と −Φ_c（巻き戻し）の両方**を課していたが、
                # **区間で稼いだ整形 +γ^(iTc)·Φ_c を足していなかった** ＝ 二重没収。
                # リスポーンは Φ=0 の開始点へ戻すので**1 サイクルの整形は telescoping して 0**。
                # 課されるのは**衝突罰 −1.0 だけ**である（下の print の機構説明と一致する形）。
                crash_cost = sum((GAMMA ** (i * Tc)) * 1.0
                                 for i in range(1, n_crash + 1)
                                 if i * Tc < TIME_LIMIT_STEPS)
                # 訪問: 1 周目に通った区画のみ（**再訪問は無報酬**が現行仕様）
                n_vis = max(int(d0 * alpha), 1)
                crashes = (-TIME_PENALTY * S(TIME_LIMIT_STEPS) - crash_cost
                           + visit_discounted(n_vis, spc))
                r = returns_for_face(d0, spc, 0.0, MEAN_HP2, dT, alpha=alpha)
                ok = (r["goal"] > r["explore"] > r["stay"] > crashes)
                print(f"{alpha:>6.2f}{n_crash:>12}{d0:>7}"
                      f"{r['goal']:>9.3f}{r['explore']:>9.3f}{r['stay']:>9.3f}"
                      f"{crashes:>18.3f}   {'✅' if ok else '🔴'}")
    print("\n  ※ 「衝突×N→時間切れ」は **開始点へ戻るので終端 Φ = 0**（整形の取り分は残らない）。")

    # 🔴 模型の中の α ではなく、**実走で観測された regime** で比べる（§7 の測定値を使う）。
    print("\n  🔴 (i-bis) **実走で観測された regime での比較**（§7 の実測値を入力にする）")
    print("       実測: 衝突までの歩数 T ≈ 608 歩・進む割合 α ≈ 0.133・訪問の取り分 ≈ +0.055")
    T_obs, alpha_obs, visit_obs = 608.0, 0.133, 0.055
    print(f"{'面 D₀':>7}{'Φ_c[m]':>9}{'現行(終端)':>12}{'(a\') 没収':>12}"
          f"{'(c) 1 回で凍結':>16}{'(c) 600 歩ごと×9':>18}{'滞留':>9}")
    for d0 in (4, 9, 15):
        phi_c = alpha_obs * d0 * CELL
        time_full = -TIME_PENALTY * S(TIME_LIMIT_STEPS)
        # 現行（終端で衝突）: 実走と同じ形
        cur = (GAMMA ** T_obs * phi_c - TIME_PENALTY * S(T_obs)
               + GAMMA ** (T_obs - 1) * COLLISION_PENALTY + visit_obs)
        # (a') 没収型: 追加分は終端 1 歩の −Φ_c
        forf = cur - GAMMA ** (T_obs - 1) * phi_c
        # (c) リスポーン: 時間罰は 6000 歩ぶん全部・衝突罰は回数ぶん
        # 🔴 是正（2026-08-14・AUDIT_025 と同型）: 旧式は −(1.0 + Φ_c) を課していたが、
        # **稼いだ整形 +γ^(iT)·Φ_c を足していなかった** ＝ 二重没収。
        # リスポーンは Φ=0 の開始点へ戻すので**1 サイクルの整形は telescoping して 0**で、
        # 課されるのは**衝突罰 −1.0 だけ**である。
        one = time_full - GAMMA ** (T_obs - 1) * 1.0 + visit_obs
        many = time_full + visit_obs - sum(
            GAMMA ** (i * T_obs - 1) * 1.0
            for i in range(1, 10) if i * T_obs < TIME_LIMIT_STEPS)
        print(f"{d0:>7}{phi_c:>9.3f}{cur:>12.3f}{forf:>12.3f}"
              f"{one:>16.3f}{many:>18.3f}{time_full:>9.3f}")
    print("       ⇒ **現行**は滞留を上回る（崩れ）。**(a\') はほとんど改善しない**（§7 の反実仮想と同じ）。")
    # 🔴 是正（2026-08-14・AUDIT_025）: 式の二重没収を直した結果、**結論が反転した**。
    # 旧記述「(c) は滞留を下回る（＝順序が回復する）」は**式の誤りに依存していた**。
    print("       ⇒ 🔴 **是正後: (c) は滞留を上回る**（＝ この regime では**順序が崩れる**）。\n"
          "          滞留 −0.2000 に対し **(c) 1 回 −0.1927（差 +0.0073）／(c) 9 回 −0.1951"
          "（差 +0.0049）**。\n"
          "          **Φ_c が式から消えるので D₀ にも依存しない**（旧式では D₀ ごとに違う値が出ていた）。\n"
          "          ⚠️ **旧記述「(c) は滞留を下回る」は式の二重没収に依存した誤りだった**"
          "（准教授 AUDIT_025）。")
    print("       ⚠️ **この薄さの正体は訪問ボーナス（+0.055）**である。"
          "衝突罰は γ^607 ≈ 0.05 まで割り引かれているので、\n"
          "          **この regime では「衝突罰の設計」ではなく「訪問ボーナスと時間罰の比」が"
          "順序を決めている**。")

    print("\n  🔴 (i-ter) **反実仮想の厳密さの違い（(a\') と (c) で物差しが同じでない）**")
    print("       (a\') は**終端の 1 歩に項を足すだけ**なので、実走の記録から"
          "**R − γ^(T−1)·Φ_T で厳密に**計算できた（§7）。")
    print("       (c) は**衝突後も走り続ける**ので軌道そのものが変わる。"
          "**厳密な反実仮想は原理的に作れない。**")
    print("       上の (i-bis) は「**実測の T・α・訪問の取り分を入力にした半実証の模型**」であり、"
          "**(a\') の反実仮想より弱い**。\n"
          "       この非対称は隠さず記録する（同じ物差しで測れていない）。")
    print("\n  🔴 (i-quater) **「衝突を繰り返し続ける方策」は探索・滞留に勝てるか**")
    print(f"       探索（6000 歩・全 36 区画）= +0.015／滞留 = −0.200 に対し:")
    # 🔴 是正（2026-08-14・AUDIT_025）: 下の数値と結論は式の二重没収を直したもの。
    print("       模型（α 掃引・N=10）: **−0.355〜−6.363** ＝ **どちらにも負ける** ✅（是正後）")
    print("       実走 regime（T≈608 歩ごとに 9 回）: **−0.1951**（**Φ_c が消えるので D₀ 非依存**）\n"
          "         ＝ **探索（+0.015）には負ける** ✅ ／ "
          "🔴 **滞留（−0.2000）には勝ってしまう**（差 +0.0049）")
    print("       ⇒ 🔴 **是正後の結論**: **この regime では「繰り返し衝突」が滞留より得になる。**\n"
          "          **順序の回復を (c) だけに依存させることはできない**"
          "（旧記述は「薄氷」と書いていたが、**式を直すと薄氷ではなく崩れている**）。\n"
          "       ⚠️ ただし **T≈608 歩は v1 の実走から取った値**である。"
          "**v2 の実走 T で引き直すこと**\n"
          "          （准教授の暫定値は中央値 186.5 歩）。"
          "**「崩れる／回復する」は T に依存するので確定は v2 の実測 T で行う**（AUDIT_030）。")

    print("\n  (ii) 進捗への信号（REVIEW_006 と同じ定義: 同時刻で進んだ/進まないの差）")
    print("       **1 歩ごとの整形 γΦ(s') − Φ(s) は (c) では一切変わらない**"
          "（Φ の定義も配線も同じ）。")
    print("       没収は**衝突の 1 歩だけ**に集中して入る（−Φ_c）。")
    print("       ⇒ **(a') が進捗信号を 1/8〜1/10 にした問題は (c) では起きない**"
          "（(a') は終端報酬の側を変えたので\n"
          "          走行中の信号設計に影響したが、(c) は Φ の遷移だけで没収する）。")

    print("\n  (iii) 訪問ボーナスの扱い（**仕様の確認事項**）")
    print("       現行実装は `self._visited` を**エピソード単位**で持つので、"
          "**リスポーン後の再訪問は無報酬**になる。")
    print("       - 利点: 「衝突→リスポーン→同じ区画を通り直して稼ぐ」ポンプが**構成上作れない**")
    print("       - 注意: 1 エピソードで探索できる訪問報酬の総量は"
          "**最大 35 区画ぶん（+0.70）で変わらない**")
    print("       - **この仕様でよいかは裁定事項**（リスポーンごとに `_visited` を"
          "リセットする設計も可能だが、\n         その場合は報酬ポンプの検査が要る）")
    return None


def v2_final_section() -> None:
    """§9 環境 v2 の**確定仕様**での順序（⑥ の再実行。実装後の実測値を入力にする）。

    v2 の確定仕様:
      - $k$ = 0（行動の高周波成分への罰なし）
      - **規約終端**（機体全体の内包）。**ΔT は実装で実測した 7 歩**を使う
        （模型の 6.8 歩ではなく実測値。深さの実測 56.9 mm は上界 65.3 mm の内側）
      - **(c) 衝突リスポーン**（衝突で終わらず開始点へ戻る）
      - **エピソード上限 2000 歩**（裁定 2026-08-14）
    """
    print("\n" + "=" * 92)
    print("§9 🔴 環境 v2 の確定仕様での順序（⑥ の再実行・実装の実測値を入力にする）")
    print("=" * 92)
    LIMIT = 2000
    dT_meas = 7.0            # 実装で実測（3 面とも 7 歩・深さ 56.9 mm）
    spc = CELL / (0.96 * DT)
    print(f"  仕様: k=0／規約終端（**ΔT = {dT_meas:.0f} 歩・実装で実測**）／"
          f"(c) リスポーン／上限 {LIMIT} 歩")
    print(f"{'面':>6}{'D₀':>4}{'ゴール':>9}{'探索':>9}{'滞留':>9}"
          f"{'衝突×N→上限':>14}{'N':>4}   順序")
    n_ok = 0
    for seed in VALID_SEEDS:
        m = generate_maze(seed, mode="loop")
        d0 = int(shortest_distances(m["v_walls"], m["h_walls"])[tuple(m["start"])])
        r = returns_for_face(d0, spc, 0.0, MEAN_HP2, dT_meas, limit_steps=LIMIT)
        # (c): 実走 regime（衝突まで約 608 歩）で上限 2000 なら N ≈ 3 回
        T_obs, alpha_obs, visit_obs = 608.0, 0.133, 0.055
        phi_c = alpha_obs * d0 * CELL
        n_crash = int(LIMIT // T_obs)
        # 🔴 是正（2026-08-14・准教授 AUDIT_025）: 旧式は
        #   crash_cost = Σ γ^(iT−1)·(1.0 + Φ_c)
        # と書いており、**区間で稼いだ整形を足さずに没収 −Φ_c を課す二重没収**だった。
        #
        # 正しい会計: **リスポーンは Φ = 0 の開始点へ戻す**ので、1 衝突サイクル
        # （開始点 → 衝突 → 開始点）の整形項は**厳密に telescoping して 0 になる**:
        #   Σ_{t=a}^{b} γ^t (γΦ_{t+1} − Φ_t) = γ^(b+1)·Φ(start) − γ^a·Φ(start) = 0
        # （AUDIT_024 が実測で恒等式を確認済み・残差 ≤ 1.235e-15）。
        # したがって課されるのは**衝突罰 −1.0 だけ**で、**Φ_c は式から消える**
        # ＝ 衝突の総収益は **D₀ に依存しない**。
        #
        # ⚠️ 没収型（(a')・報酬で没収する案）と遷移型（(c)・リスポーンで戻す案）を
        # 同じ式で書いたのが誤りの原因である（同スクリプトの `forfeit=True` は正しい）。
        crash_cost = sum(GAMMA ** (i * T_obs - 1) * 1.0
                         for i in range(1, n_crash + 1))
        crashes = -TIME_PENALTY * S(LIMIT) - crash_cost + visit_obs
        ok = (r["goal"] > r["explore"] > r["stay"] > crashes)
        n_ok += ok
        print(f"{seed:>6}{d0:>4}{r['goal']:>9.3f}{r['explore']:>9.3f}{r['stay']:>9.3f}"
              f"{crashes:>14.3f}{n_crash:>4}   {'✅' if ok else '🔴'}")
    print(f"\n  **順序が成立した面: {n_ok} / {len(VALID_SEEDS)}**")
    print("  （比較: 現行仕様〈終端衝突・上限 6000〉では 6/20 だった）")
    print("  ⚠️ 衝突シナリオは**実走 regime の実測値**（T≈608 歩・α≈0.133・訪問 +0.055）を"
          "入力にした半実証の模型である\n"
          "     — (a') のときと違い**厳密な反実仮想は作れない**（軌道が変わるため）。")


def empirical_check() -> None:
    """exp_011（k=0 の実走）の検証帯 20 面で、**実際の割引後総収益**を計算する。"""
    from stable_baselines3 import PPO

    from mouse.maze6_env import Maze6Env
    from mouse.maze6_eval import VALIDATION_MAZE_DIR, _trial_seed

    import statistics
    stay_return = -TIME_PENALTY * S(TIME_LIMIT_STEPS)
    all_alphas = []
    for name in ("exp_011_m2_0_k0_seed1", "exp_011_m2_0_k0_seed2"):
        path = REPO_ROOT / f"models/{name}.zip"
        if not path.exists():
            print(f"  [skip] {name}: モデルが無い")
            continue
        model = PPO.load(str(path), device="cpu")
        n_pos = n_beat_stay = n_coll = 0
        rows, alphas, steps_list, forfeit_beats = [], [], [], []
        for ms in VALID_SEEDS:
            env = Maze6Env(maze_dir=str(REPO_ROOT / VALIDATION_MAZE_DIR),
                           maze_seeds=[ms], max_cache=2, mode="fixed", maze_mode="loop",
                           gamma=GAMMA, action_highpass_penalty=0.0)
            obs, _ = env.reset(seed=_trial_seed(0, ms, 0))
            ret, disc, n_steps = 0.0, 1.0, 0
            while True:
                n_steps += 1
                a, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                ret += disc * float(r)
                disc *= GAMMA
                if term or trunc:
                    break
            outcome = ("goal" if info["goal"] else
                       "collision" if info["collision"] else "timeout")
            env.close()
            # 実測の「進む割合 α」= (稼いだ区画数)/(最短区画数)。**模型の任意定数を
            # 実走で決め直すための量**（准教授 REVIEW_005 の感度の指摘への回答）。
            d_end = int(info["dist_to_goal"])
            d_start = int(info["d_start"])
            alpha_obs = (d_start - d_end) / d_start if d_start else float("nan")
            # 🔴 反実仮想: **同じ走行に没収型の衝突罰 −(1.0 + Φ_T) を当てたら**総収益は
            # いくらだったか。追加分は終端の 1 歩だけなので R − γ^(T−1)·Φ_T で厳密に出る。
            phi_T = (d_start - d_end) * CELL
            ret_forfeit = ret - (GAMMA ** (n_steps - 1)) * phi_T
            rows.append((ms, outcome, ret, alpha_obs, n_steps, ret_forfeit))
            if outcome == "collision":
                n_coll += 1
                n_pos += (ret > 0)
                n_beat_stay += (ret > stay_return)
                alphas.append(alpha_obs)
                steps_list.append(n_steps)
                forfeit_beats.append(ret_forfeit > stay_return)
        a_med = statistics.median(alphas) if alphas else float("nan")
        print(f"  **{name}**: 衝突で終わった走行 {n_coll}/20 面 → "
              f"**総収益が正 {n_pos} 面**・**滞留({stay_return:.3f}) を上回る {n_beat_stay} 面**")
        print(f"    実測の進む割合 α: 中央値 **{a_med:.3f}**"
              f"（最小 {min(alphas):.3f}・最大 {max(alphas):.3f}）")
        print(f"    衝突時のエピソード長 T: 中央値 **{statistics.median(steps_list):.0f} 歩**"
              f"（γ^(T−1) = {GAMMA ** (statistics.median(steps_list)-1):.3f} ＝ "
              f"**衝突罰 −1.0 がここまで割り引かれる**）")
        print(f"    🔴 **没収型 −(1.0+Φ_T) を同じ走行に当てた反実仮想**: "
              f"滞留を上回る面が {n_beat_stay} → **{sum(forfeit_beats)} 面**")
        print("    " + " ".join(f"{ms}:{o[:4]}{ret:+.2f}" for ms, o, ret, _, _, _ in rows[:10]))
        print("    " + " ".join(f"{ms}:{o[:4]}{ret:+.2f}" for ms, o, ret, _, _, _ in rows[10:]))
        all_alphas.extend(alphas)
    if all_alphas:
        a_med = statistics.median(all_alphas)
        print(f"\n  🔴 **実走の α の中央値 = {a_med:.3f}**（2 モデル・{len(all_alphas)} 走行）。"
              f"\n  模型が採った 0.5 より**かなり小さい**。**この α を模型へ戻す**と:")
        spc_ = CELL / (0.96 * DT)
        r_circ, _ = containment_depth_m()
        dT_ = r_circ / (0.96 * DT)
        n_bad = 0
        for seed in VALID_SEEDS:
            m = generate_maze(seed, mode="loop")
            d0 = int(shortest_distances(m["v_walls"], m["h_walls"])[tuple(m["start"])])
            if not ordering_ok(returns_for_face(d0, spc_, 0.0, MEAN_HP2, dT_, alpha=a_med)):
                n_bad += 1
        print(f"    **模型（α = 実測値 {a_med:.3f}）で崩れる面: {n_bad} / 20**")
        print(f"    **実走で滞留を上回った面: 10 / 20（両 seed とも）**")
        print("  → **数は一致しないが、どちらも『半分前後の面で 滞留 > 衝突 が崩れる』"
              "という定性的な結論は同じ**。")
        print("\n  🔴 **ただし機構が違う（模型の想定と実走で別）**:")
        print("     模型: **稼いだ整形 Φ_T が大きいから**崩れる（α = 0.5 が要る）")
        print("     実走: **エピソードが長い（中央値 600 歩超）ので衝突罰 −1.0 が"
              "γ^(T−1) ≈ 0.05 まで割り引かれる**から崩れる（Φ_T は小さいまま）")
        print("     ⇒ **対処案 (a') は前者の経路しか塞がない。**"
              "反実仮想（上の各モデルの行）が示すとおり、\n"
              "        実走の regime では没収型を当てても滞留を上回る面はほとんど減らない。")


if __name__ == "__main__":
    raise SystemExit(main())
