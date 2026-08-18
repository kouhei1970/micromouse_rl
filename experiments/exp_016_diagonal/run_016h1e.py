#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""016-H1e — 第三容疑（是正距離の速度依存）の否定対照と、減衰長 L の計装。

事前登録 `PREREG_016h1e.md`・カード `card_016h1e.md` §2〜§3。
**本スクリプトは制御コードを 1 行も変更しない。**否定対照は既存方策の
サブクラス化で作る（`run_016h1_diag.py` の `make_probed` と同型）。
既存の診断ハーネス `run_016h1_diag.py` を import して再利用する
（裁定 R23。定義を写さず import する）。**`run_016h1_diag.py` は改変しない。**

    .venv/bin/python experiments/exp_016_diagonal/run_016h1e.py --group frozen --v-diag 0.45

否定対照（凍結）の実行経路の注記（**重要**）:
    測定条件（カード §3）は F0 + F0-b（`ref_interp=True`）+ 安全率 0.75。
    この構成では `_do_drive_control` の実体は
    `competition/reference_interp.py` の `ReferenceInterpMixin._do_drive_control`
    （`TwoDofControlMixin` は tau_la=k_r=0 のとき super() へ委譲してスキップされる。
    `competition/control_2dof.py:71`）。**したがって否定対照はこのメソッドの写しに
    分母だけ差し替えて作る**（`baseline_slalom.py:1405` の生の実装はこの構成では
    呼ばれない）。
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for p in (REPO_ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.seed_bands import assert_seeds_allowed, describe_seeds        # noqa: E402
from competition.baseline_slalom import _wrap_pi                          # noqa: E402
from competition.baseline_slalom_diag_cal import L_C_CLOTHOID_M           # noqa: E402
from competition.baseline_slalom_e1_tr import load_time_model             # noqa: E402
from competition.route_planner import value_field                        # noqa: E402
from mouse.mjcf import build_maze_robot_xml                               # noqa: E402
from mouse.params import RobotParams                                      # noqa: E402

import run_016f0_ladder                                                   # noqa: E402
import run_016g_ladder                                                    # noqa: E402
import run_016h1_diag                                                     # noqa: E402  ← ハーネス本体はここから借りる
from diagonal_model import (DELTA8, DiagonalGridModel, cell_center_node,  # noqa: E402
                            descend)
from geometry import git_rev                                              # noqa: E402
from route_model import connects_true, load_maze                          # noqa: E402
from run_016b import cut_segment, longest_diagonal_run                    # noqa: E402
from run_016c import R_ARC_M                                              # noqa: E402

D0_FROZEN_DEFAULT = 0.60          # v=0.45 水準の公称値（0.45+0.15）。カード §2
MIN_ENTRY_STRAIGHT_M = 0.10       # 限界1（PREREG §5-1・カード §5-1）


def make_frozen(policy_cls, d0: float | None = D0_FROZEN_DEFAULT):
    """lateral_term の分母を定数 d0 へ凍結した派生クラス（否定対照）。

    **写し元は `competition/reference_interp.py:113-143`
    （`ReferenceInterpMixin._do_drive_control`。ref_interp=True の枝）。**
    変えたのは元 138〜139 行の分母 1 箇所だけである。
    親（`competition/reference_interp.py` / `competition/control_2dof.py`）が
    変わったら、この写しも追随が要る。

    **`d0=None`（是正3件目・PREREG §4 新 W2）**: 分母を凍結しない経路。
    健常と同じ `v_for_gain = max(abs(v_fwd), 1e-6); denom = v_for_gain + v_eps`
    を使う。この写しは分母の 1 箇所以外は健常の本体と同一なので、
    `d0=None` なら健常と**軌跡・電圧が bit 一致するはず**（W2 の必然性の検査）。
    """

    class FrozenDenomPolicy(policy_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            # この否定対照は「測定条件（カード §3）= ref_interp=True・tau_la=k_r=0」
            # 専用に作った写しである。前提が外れると健常と同一の保証が崩れるので固定する。
            if not getattr(self, "ref_interp", False):
                raise AssertionError("凍結対照は ref_interp=True の構成専用（写し元が変わる）")
            if getattr(self, "tau_la", 0.0) != 0.0 or getattr(self, "k_r", 0.0) != 0.0:
                raise AssertionError("凍結対照は tau_la=k_r=0 の構成専用（写し元が変わる）")

        # --- competition/reference_interp.py:113-143 の写し。分母のみ差し替え ---
        def _do_drive_control(self, obs, x: float, y: float, yaw: float):
            path = self._path
            idx = self._cursor
            px, py = float(path.x[idx]), float(path.y[idx])
            phead = float(path.heading[idx])
            pcurv = float(path.curvature[idx])
            v_target = self._reference_speed(idx, x, y)

            if self._v_setpoint < v_target:
                self._v_setpoint = min(v_target, self._v_setpoint + self.a_max * self.control_dt)
            else:
                self._v_setpoint = v_target
            v_ref = self._v_setpoint

            ux, uy = math.cos(phead), math.sin(phead)
            rel_x, rel_y = x - px, y - py
            e_y = rel_x * (-uy) + rel_y * ux
            e_psi = _wrap_pi(phead - yaw)

            v_fwd, _omega_z = self._sim.privileged_velocity()
            # 🔴 否定対照: 分母 (v_for_gain + v_eps) を定数 d0 に凍結する。
            #     健常（写し元 138-139 行）: v_for_gain = max(abs(v_fwd), 1e-6);
            #                                lateral_term = atan2(k_y*e_y, v_for_gain+v_eps)
            #     d0=None なら凍結せず、健常と同じ式に戻す（新 W2・PREREG §4）。
            denom = d0 if d0 is not None else (max(abs(v_fwd), 1e-6) + self.v_eps)
            lateral_term = math.atan2(self.k_y * e_y, denom)

            omega_ref = pcurv * v_ref + self.k_psi * e_psi - lateral_term
            omega_ref = max(-self.turn_omega_limit, min(self.turn_omega_limit, omega_ref))
            return self._wheel_targets_to_voltage(v_ref, omega_ref, obs)

    return FrozenDenomPolicy


def entry_straight_segments(r, min_len_m: float = MIN_ENTRY_STRAIGHT_M, kappa_eps: float = 1e-6):
    """進入直線（PREREG §3 の測定条件）を検出し、区間ごとの計装値を返す。

    定義: `path.curvature` の |κ| < kappa_eps が続く区間のうち、
    **直後に非ゼロ区間が来るもの**（円弧・クロソイドの直前の直線）。
    記録は drive_and_record が `sim.step_control()` の前に取った値をそのまま使う
    （run_016h1_diag.py:102-104 の教訓）。
    Λ_seg < min_len_m の区間は `excluded_short=True` を立てるだけで、
    母集団からの実際の除外は集計側（main）で行う（限界1）。

    L_seg の符号（是正2件目・PREREG §4 W1）:
    条文の式 L_seg = Λ_seg / ln(|e_y|_start / |e_y|_end) は**符号を問わない**。
    |e_y| が区間内で増えた（ey_start <= ey_end）場合は ln が 0 以下になり
    L_seg は 0 以下（負・または比が 1 で undefined）になるが、これも棄却せず計算する。
    比が取れない（start か end が 0、または ln の分母が 0=比が 1）場合のみ
    `ratio_undefined=True` を立てて nan にする。
    """
    kappa, s, v_act = r["kappa"], r["s"], r["v_act"]
    ey = np.abs(r["e_y"])
    n = len(kappa)
    straight = np.abs(kappa) < kappa_eps
    segs = []
    i = 0
    while i < n:
        if not straight[i]:
            i += 1
            continue
        j = i
        while j < n and straight[j]:
            j += 1
        if j < n:  # 直後に非ゼロ区間が来る（末尾で切れた直線は対象外）
            lam = float(s[j - 1] - s[i])
            ey_start, ey_end = float(ey[i]), float(ey[j - 1])
            excluded = lam < min_len_m
            l_seg = float("nan")
            ratio_undefined = False
            if not excluded:
                # 符号を問わず PREREG §4 W1 の式どおりに計算する（是正2件目）。
                # 比が取れる（両端とも 0 でなく、比が 1 でない）区間だけ計算し、
                # 増加区間（ey_start <= ey_end）は負の l_seg を与える。
                if ey_start <= 0.0 or ey_end <= 0.0:
                    ratio_undefined = True
                else:
                    log_ratio = math.log(ey_start / ey_end)
                    if log_ratio == 0.0:
                        ratio_undefined = True
                    else:
                        l_seg = lam / log_ratio
            segs.append(dict(lambda_seg_m=lam, ey_start_m=ey_start, ey_end_m=ey_end,
                             v_act_med=float(np.median(v_act[i:j])),
                             n_ticks=int(j - i), l_seg_m=l_seg,
                             excluded_short=bool(excluded),
                             ratio_undefined=bool(ratio_undefined)))
        i = j
    return segs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=["healthy", "frozen"], default="healthy")
    ap.add_argument("--safety", type=float, default=0.75)
    ap.add_argument("--v-diag", type=float, required=True)
    ap.add_argument("--d0", type=float, default=D0_FROZEN_DEFAULT)
    ap.add_argument("--maze-dir", default="competition/mazes/design_v4")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-mazes", type=int, default=None,
                    help="煙試験専用。全水準・全面の本走行では指定しないこと")
    args = ap.parse_args()

    maze_files = sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"),
                        key=lambda p: int(p.stem.split("_")[1]))
    if args.max_mazes is not None:
        maze_files = maze_files[:args.max_mazes]
    seeds = [int(q.stem.split("_")[1]) for q in maze_files]
    print(describe_seeds(seeds, "competition"))
    assert_seeds_allowed(seeds, namespace="competition", purpose="validate")

    params = RobotParams()
    dt = params.control_dt
    a, b = load_time_model()
    policy_cls = run_016f0_ladder.make_policy_class(k_acc_ff=1.0, ref_interp=True,
                                                    safety=args.safety)
    if args.group == "frozen":
        policy_cls = make_frozen(policy_cls, d0=args.d0)
    builder = run_016g_ladder.make_builder(L_C_CLOTHOID_M)
    print(f"群={args.group}／採用構成: F0 + F0-b + 安全率 {args.safety:g}"
          + (f" + 分母凍結 D0={args.d0:g}" if args.group == "frozen" else "")
          + f"／速度水準 {args.v_diag:g} m/s／制御周期 {dt*1000:.1f} ms\n")

    rows = []
    n_seg_total, n_seg_excluded = 0, 0
    n_seg_ratio_undefined = 0   # 比が取れない区間（start/end が 0、または比が 1）
    n_seg_increasing = 0        # |e_y| が区間内で増えた（l_seg < 0）区間（是正2件目）
    for f in maze_files:
        v, h, start, goals = load_maze(str(f))
        conn = connects_true(v, h)
        model = DiagonalGridModel(a, b, r=1.0)
        field = value_field([tuple(g) for g in goals], 16, 16, conn, model)
        p = descend(field, model, cell_center_node(tuple(start)), "N", 16, 16, conn)
        s0, e0 = longest_diagonal_run(p["dirs"])
        i, j = cut_segment(p["nodes"], p["dirs"], s0, e0)
        xml = f.with_suffix(".xml")
        if not xml.exists():
            build_maze_robot_xml(v, h, str(xml), model_name=f"m_{f.stem}", params=params)
        r, collided, pol_probe = run_016h1_diag.drive_and_record(
            xml, params, p["nodes"][i:j + 1], p["dirs"][i:j], args.v_diag, v, h,
            policy_cls, builder)
        if not r or len(r["t"]) == 0:
            print(f"{f.stem}: 記録なし")
            continue

        ee = run_016h1_diag.entry_exit(r, float(getattr(pol_probe, "voltage_limit", 3.0)))
        segs = entry_straight_segments(r)
        n_seg_total += len(segs)
        n_seg_excluded += sum(1 for sgm in segs if sgm["excluded_short"])
        n_seg_ratio_undefined += sum(1 for sgm in segs if sgm["ratio_undefined"])
        n_seg_increasing += sum(1 for sgm in segs if sgm["l_seg_m"] < 0.0)

        rows.append(dict(
            maze=f.stem, collided=bool(collided), n_ticks=int(len(r["t"])),
            n_arcs=len(ee),
            entry_exit=[[float(x0), float(x1), float(x2)] for x0, x1, x2 in ee],
            entry_straights=segs,
        ))
        print(f"{f.stem}: 円弧 {len(ee)} 本／進入直線 {len(segs)} 本"
              f"（うち Λ<{MIN_ENTRY_STRAIGHT_M:g}m 除外 "
              f"{sum(1 for sgm in segs if sgm['excluded_short'])}）", flush=True)

    ex_rate = (n_seg_excluded / n_seg_total) if n_seg_total else float("nan")
    print(f"\n【進入直線の母集団】総数 {n_seg_total}／除外 {n_seg_excluded}"
          f"（率 {ex_rate*100:.1f}%、限界1: Λ<{MIN_ENTRY_STRAIGHT_M:g}m）")
    # L_seg の計算に使えた区間数（短い区間・比が取れない区間を除いた分母）
    n_seg_usable = n_seg_total - n_seg_excluded - n_seg_ratio_undefined
    inc_rate = (n_seg_increasing / n_seg_usable) if n_seg_usable else float("nan")
    print(f"【L_seg の符号（是正2件目）】比が取れない区間（除外短区間を除く）"
          f" {n_seg_ratio_undefined}／算出できた区間 {n_seg_usable}"
          f"（うち増加区間 l_seg<0: {n_seg_increasing} 件、率 {inc_rate*100:.1f}%）")

    ein = [abs(x0) for row in rows for x0, x1, x2 in row["entry_exit"]]
    if ein:
        print(f"【E_in】円弧 {len(ein)} 本・{len(rows)} 面連結・中央値 "
              f"{np.median(ein)*1000:.3f} mm")

    out = Path(args.out or (REPO_ROOT / "outputs" / "exp_016_diagonal" / "016h1e"
                            / f"{args.group}_sf{args.safety:g}_v{args.v_diag:g}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps(dict(
        git_rev=git_rev(), maze_dir=args.maze_dir, group=args.group, safety=args.safety,
        v_diag=args.v_diag, control_dt=dt, L_c=L_C_CLOTHOID_M,
        d0_frozen=(args.d0 if args.group == "frozen" else None),
        min_entry_straight_m=MIN_ENTRY_STRAIGHT_M,
        entry_straight_n_total=n_seg_total, entry_straight_n_excluded=n_seg_excluded,
        entry_straight_excluded_rate=ex_rate,
        entry_straight_n_ratio_undefined=n_seg_ratio_undefined,
        entry_straight_n_usable=n_seg_usable,
        entry_straight_n_increasing=n_seg_increasing,
        entry_straight_increasing_rate=inc_rate,
        rows=rows), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
