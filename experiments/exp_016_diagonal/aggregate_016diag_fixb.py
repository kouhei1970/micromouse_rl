#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**是正 (B) の判定**（`card_016diag_fixB.md` §2 の予測 B1〜B4）。

**読むだけで走らせない。**入力は `run_016diag_switch.py` が書いた
`runs_detail.json` と `traj/*.npz`。

| # | 予測 | 量 | 閾値 |
|---|---|---|---|
| **B1** | 衝突と係員回収が消える | 1 迷路あたりの**事故件数**と**回収回数** | **20 迷路すべて 0 件** |
| **B2** | 🔴 **斜め経路が計時走行で実際に使われる**（**主判定**） | 最速走行で**斜めを含む経路に乗っていたティックの割合** | **20 迷路の中央値 20 % 以上** |
| **B3** | 探索時の速度上限に張り付かなくなる | 最速走行の**最大速度** | **20 迷路すべて 0.60 m/s 超** |
| **B4** | 経路効率は変わらない | **(e′)** の同じ迷路どうしの差 | **20 迷路すべて ±0.0000** |

**斜めの判定**（カード §2 の B2 の定義そのまま）:
**参照の曲率が 0 かつ参照の方位が 45° の奇数倍**の標本。
**⚠️ 曲率で絞らないと、円弧の掃引角が 45° を通過するのを数えてしまう。**

**⚠️ 走行タイムは判定に使わない**（カード §2-1。本カードは修理であって速さの実験ではない）。

使い方:
    .venv/bin/python experiments/exp_016_diagonal/aggregate_016diag_fixb.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "experiments" / "exp_016_diagonal"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from aggregate_016diag_switch import d_best_time, e_prime, load_arm  # noqa: E402

ROOT = REPO_ROOT / "outputs" / "exp_016_diagonal" / "016diag_switch"
KAPPA_EPS = 1e-9         # これ以下を「曲率 0（直線）」とみなす
ANGLE_TOL_DEG = 1.0      # 45° の奇数倍からのずれの許容
B2_THRESHOLD = 0.20      # カード §2 の B2 の閾値
B3_THRESHOLD = 0.60      # カード §2 の B3 の閾値


def fast_run_window(arm_dir: Path, maze: str, runs):
    """その迷路の**最速走行**（= (d) を作った走行）の時間窓。"""
    goals = [r for r in runs if r["outcome"] == "goal" and r.get("run_time") is not None]
    if not goals:
        return None, None
    best = min(goals, key=lambda r: r["run_time"])
    z = np.load(arm_dir / "traj" / f"{maze}.npz")
    idx = {int(i): (float(a), float(b))
           for i, a, b in zip(z["run_index"], z["run_t_start"], z["run_t_end"])}
    return z, idx.get(best["run"])


def diag_fraction_and_vmax(z, win):
    """最速走行での (斜め標本の割合, 最大速度)。"""
    t = np.asarray(z["t"])
    m = (t >= win[0] - 1e-9) & (t <= win[1] + 1e-9)
    kap = np.abs(np.asarray(z["ref_curvature"])[m])
    hd = np.degrees(np.asarray(z["ref_heading"])[m]) % 90.0
    v = np.abs(np.asarray(z["v"])[m])
    fin = np.isfinite(kap) & np.isfinite(hd)
    # **曲率 0 かつ方位が 45° の奇数倍**（曲率で絞らないと円弧の掃引を数えてしまう）
    is_diag = fin & (kap <= KAPPA_EPS) & (np.abs(hd - 45.0) < ANGLE_TOL_DEG)
    return float(is_diag.mean()), float(v.max()), int(m.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--before", default="diag", help="是正前の載せ替え版")
    ap.add_argument("--after", default="diag_fixb", help="是正 (B) 後")
    ap.add_argument("--control", default="control", help="参考（斜めなしの新既定）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out or (root / "fixb_aggregate.json"))
    (mb, B) = load_arm(root / args.before)
    (ma, A) = load_arm(root / args.after)
    (mc, C) = load_arm(root / args.control)
    common = sorted(set(A) & set(B) & set(C))
    print(f"是正前 = {mb['policy']}\n是正後 = {ma['policy']}\n"
          f"参考   = {mc['policy']}\n迷路 {len(common)} 件／{ma['maze_dir']}\n")

    print("【迷路ごと】")
    print(f"{'迷路':<12}{'事故':>5}{'回収':>5}{'一致':>5}{'委ね':>5}"
          f"{'斜め割合 前':>12}{'斜め割合 後':>12}{'最大速度 前':>12}{'最大速度 後':>12}"
          f"{'(e′) 前':>9}{'(e′) 後':>9}")
    rows = []
    for mz in common:
        # ⚠️ **事故・回収などの迷路ごとの量は、最後の走行の記録に付いている**
        # （`run_016diag_switch.py` が `detail[-1].update(...)` で付けるため）。
        # **先頭の走行を見ると None になる** — 2026-08-15 に空のデータで
        # B1 を「的中」と判定しかけた欠陥の是正
        det = next((r for r in reversed(A[mz]) if r.get("n_incidents") is not None),
                   A[mz][-1])
        zb, wb = fast_run_window(root / args.before, mz, B[mz])
        za, wa = fast_run_window(root / args.after, mz, A[mz])
        fb, vb, _ = diag_fraction_and_vmax(zb, wb)
        fa, va, na = diag_fraction_and_vmax(za, wa)
        dtrue = int(det["d_true"])
        eb, ea = e_prime(B[mz], dtrue), e_prime(A[mz], dtrue)
        rows.append(dict(maze=mz, n_incidents=det.get("n_incidents"),
                         n_retrieval=det.get("n_retrieval"),
                         n_align_ok=det.get("n_align_ok"),
                         n_align_defer=det.get("n_align_defer"),
                         diag_frac_before=fb, diag_frac_after=fa,
                         v_max_before=vb, v_max_after=va,
                         e_prime_before=eb, e_prime_after=ea,
                         d_before=d_best_time(B[mz]), d_after=d_best_time(A[mz]),
                         d_control=d_best_time(C[mz])))
        q = rows[-1]
        print(f"{mz:<12}{q['n_incidents']!s:>5}{q['n_retrieval']!s:>5}"
              f"{q['n_align_ok']!s:>5}{q['n_align_defer']!s:>5}"
              f"{fb*100:>11.2f}%{fa*100:>11.2f}%{vb:>12.3f}{va:>12.3f}"
              f"{_f(eb, 3):>9}{_f(ea, 3):>9}")

    # ---------------- B1 ------------------------------------------------
    ninc = [q["n_incidents"] for q in rows if q["n_incidents"] is not None]
    nret = [q["n_retrieval"] for q in rows if q["n_retrieval"] is not None]
    # 🔴 **データが揃っていなければ判定しない**（空の合計が 0 になって
    # 「的中」に見える事故を防ぐ。2026-08-15 に実際にやりかけた）
    if len(ninc) != len(rows) or len(nret) != len(rows):
        b1 = None
        print(f"\n【B1】🔴 **判定不能** — 記録が {len(ninc)}/{len(rows)} 迷路ぶんしかない")
    else:
        b1 = (sum(ninc) == 0 and sum(nret) == 0)
        print(f"\n【B1】事故と係員回収（予測: 20 迷路すべて 0 件）")
        print(f"  事故 合計 {sum(ninc)} 件（0 でない迷路 {sum(1 for x in ninc if x)} 件）")
        print(f"  回収 合計 {sum(nret)} 回（0 でない迷路 {sum(1 for x in nret if x)} 件"
              + (f": {[q['maze'] for q in rows if q['n_retrieval']]}）"
                 if any(nret) else "）"))
        print(f"  → **{'的中' if b1 else '外れ'}**")
        if any(nret) and not any(ninc):
            print("  ⚠️ **衝突は 0 件だが回収が起きている迷路がある。**")
            print("     評価器は `goal_not_contained`（機体全体が区画に入りきらなかった）でも")
            print("     回収するが、**その場合は `incidents` に記録されない**。")
            print("     → 走行の結末を確かめること")

    # ---------------- B2（主判定）---------------------------------------
    fa = [q["diag_frac_after"] for q in rows]
    fb = [q["diag_frac_before"] for q in rows]
    med = float(np.median(fa))
    b2 = med >= B2_THRESHOLD
    print(f"\n【B2】🔴 **主判定** 斜め経路に乗っていたティックの割合"
          f"（予測: 20 迷路の中央値 {B2_THRESHOLD*100:.0f} % 以上）")
    print(f"  是正前 中央値 {np.median(fb)*100:.2f} %（{min(fb)*100:.2f}〜{max(fb)*100:.2f}）")
    print(f"  **是正後 中央値 {med*100:.2f} %**（{min(fa)*100:.2f}〜{max(fa)*100:.2f}）")
    print(f"  20 % 以上だった迷路 {sum(1 for x in fa if x >= B2_THRESHOLD)}/{len(fa)}")
    print(f"  → **{'的中' if b2 else '外れ'}**")

    # ---------------- B3 ------------------------------------------------
    va_l = [q["v_max_after"] for q in rows]
    b3 = all(x > B3_THRESHOLD for x in va_l)
    print(f"\n【B3】最速走行の最大速度（予測: 20 迷路すべて {B3_THRESHOLD} m/s 超）")
    print(f"  是正前 {min(q['v_max_before'] for q in rows):.3f}〜"
          f"{max(q['v_max_before'] for q in rows):.3f} m/s")
    print(f"  **是正後 {min(va_l):.3f}〜{max(va_l):.3f} m/s**"
          f"（{B3_THRESHOLD} m/s 以下の迷路 {sum(1 for x in va_l if x <= B3_THRESHOLD)} 件）")
    print(f"  → **{'的中' if b3 else '外れ'}**")

    # ---------------- B4 ------------------------------------------------
    pairs = [(q["maze"], q["e_prime_before"], q["e_prime_after"]) for q in rows
             if q["e_prime_before"] is not None and q["e_prime_after"] is not None]
    diffs = [c - b for _m, b, c in pairs]
    b4 = bool(pairs) and all(d == 0.0 for d in diffs)
    print(f"\n【B4】経路効率 (e′)（予測: 20 迷路すべて ±0.0000）")
    if pairs:
        print(f"  n={len(pairs)}／差の中央値 {np.median(diffs):+.4f}"
              f"（{min(diffs):+.4f}〜{max(diffs):+.4f}）"
              f"／1 ビットも動かなかった迷路 {sum(1 for d in diffs if d == 0.0)}/{len(diffs)}")
        for m, b, c in pairs:
            if c != b:
                print(f"    {m}: {b:.4f} → {c:.4f}")
    print(f"  → **{'的中' if b4 else '外れ'}**")

    # ---------------- 参考（判定に使わない）------------------------------
    print("\n【参考】走行タイム（**判定には使わない** — カード §2-1）")
    rel = [(q["d_after"] - q["d_control"]) / q["d_control"] for q in rows
           if q["d_after"] and q["d_control"]]
    relb = [(q["d_before"] - q["d_control"]) / q["d_control"] for q in rows
            if q["d_before"] and q["d_control"]]
    if rel:
        print(f"  対照との対応差の中央値: 是正前 {np.median(relb)*100:+.1f} %"
              f" → **是正後 {np.median(rel)*100:+.1f} %**")

    json.dump(dict(before=mb["policy"], after=ma["policy"], maze_dir=ma["maze_dir"],
                   n_mazes=len(common), B1=b1, B2=b2, B3=b3, B4=b4,
                   b2_median=med, b3_min=min(va_l),
                   rel_median_after=float(np.median(rel)) if rel else None,
                   rel_median_before=float(np.median(relb)) if relb else None,
                   per_maze=rows),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"\n書き出し: {out}")
    return 0


def _f(x, prec=2):
    return "  —  " if x is None else f"{x:.{prec}f}"


if __name__ == "__main__":
    sys.exit(main())
