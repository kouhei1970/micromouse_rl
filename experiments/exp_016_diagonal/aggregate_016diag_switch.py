#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**載せ替えカードの集計と判定**（`card_016diag_switch.md` §4 の予測 P1〜P5・§5 の報告項目）。

**読むだけで走らせない。**入力は `run_016diag_switch.py` が書いた
`runs_detail.json` と `traj/*.npz`。

--------------------------------------------------------------------------
出すもの
--------------------------------------------------------------------------
| # | 予測 | 量 |
|---|---|---|
| **P1** | 差は縮むが、まだ負けている | **(d) の同じ迷路どうしの対応差の中央値 ÷ 対照** |
| **P2** | 完走率は落ちない | **(a) が成立した迷路数** |
| **P3** | 斜め区間が最も長く時間を使っている | 最速走行の**区間ごとの所要時間の割合** |
| **P4** | 円弧区間は速くなっている | **円弧区間の平均速度** |
| **P5** | (e′) は 016-D と変わらない | **経路効率の同じ迷路どうしの差**（対照 = 016-D の斜めあり） |

**区間の復元は走行後に行う**（`run_016diag_switch.py` の冒頭の規則）:

- **円弧** = 参照の曲率が 0 でない
- **斜め** = 曲率が 0 かつ**参照の方位が 45° の奇数倍**
- **直進** = 曲率が 0 かつ**参照の方位が 90° の倍数**

**(e′) の定義は裁定 R14**（分子 = 節点数 = 移動回数 + 1）。

使い方:
    .venv/bin/python experiments/exp_016_diagonal/aggregate_016diag_switch.py
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition.evaluator import maze_kpi  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

ROOT = REPO_ROOT / "outputs" / "exp_016_diagonal" / "016diag_switch"
KAPPA_EPS = 1.0          # [1/m] これを超えたら円弧とみなす（円弧は 1/0.06 = 16.7）
ANGLE_TOL_DEG = 20.0     # 方位の判定の許容差


# ==========================================================================
def load_arm(arm_dir: Path):
    """1 条件の記録を読む。"""
    d = json.load(open(arm_dir / "runs_detail.json", encoding="utf-8"))
    by_maze = {}
    for r in d["runs"]:
        by_maze.setdefault(r["maze"], []).append(r)
    return d, {k: sorted(v, key=lambda q: q["run"]) for k, v in sorted(by_maze.items())}


def d_best_time(runs):
    """**(d) 最速タイム = 完走走行の最速値**（研究計画書 §2）。"""
    ts = [float(r["run_time"]) for r in runs
          if r["outcome"] == "goal" and r.get("run_time") is not None]
    return min(ts) if ts else None


def e_prime(runs, d_true):
    """**(e′) 経路効率**（裁定 R14: 分子 = 移動回数 + 1）。初回の最短走行で測る。"""
    goals = [r for r in runs if r["outcome"] == "goal" and r.get("run_time") is not None]
    if not goals:
        return None
    first_goal = min(goals, key=lambda r: r["run"])
    later = [r for r in goals if r["run"] > first_goal["run"]]
    if not later:
        return None
    ff = min(later, key=lambda r: r["run"])
    return (ff["n_cells"] + 1) / d_true if d_true else None


def segment_split(npz, t0, t1, dt):
    """走行 [t0, t1] を区間（直進／斜め／円弧）に分け、時間・距離・平均速度を返す。"""
    t = np.asarray(npz["t"])
    m = (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
    if m.sum() < 2:
        return None
    kap = np.abs(np.asarray(npz["ref_curvature"])[m])
    hd = np.degrees(np.asarray(npz["ref_heading"])[m]) % 90.0
    v = np.abs(np.asarray(npz["v"])[m])

    is_arc = kap > KAPPA_EPS
    near45 = np.abs(hd - 45.0) < ANGLE_TOL_DEG
    near0 = np.minimum(hd, 90.0 - hd) < ANGLE_TOL_DEG
    kind = np.where(is_arc, "arc", np.where(near45, "diagonal",
                                            np.where(near0, "straight", "other")))
    out = {}
    for k in ("straight", "diagonal", "arc", "other"):
        sel = kind == k
        n = int(sel.sum())
        if n == 0:
            out[k] = dict(t_s=0.0, dist_m=0.0, v_mean=None, frac=0.0)
            continue
        tt = n * dt
        dd = float(np.sum(v[sel]) * dt)
        out[k] = dict(t_s=tt, dist_m=dd, v_mean=(dd / tt if tt > 0 else None),
                      frac=n / len(kind))
    out["_total_t_s"] = len(kind) * dt
    return out


def fast_run_split(arm_dir: Path, maze: str, runs, dt):
    """その迷路の**最速走行**（= (d) を作った走行）の区間ごとの内訳。"""
    goals = [r for r in runs if r["outcome"] == "goal" and r.get("run_time") is not None]
    if not goals:
        return None
    best = min(goals, key=lambda r: r["run_time"])
    z = np.load(arm_dir / "traj" / f"{maze}.npz")
    idx = {int(i): (float(a), float(b))
           for i, a, b in zip(z["run_index"], z["run_t_start"], z["run_t_end"])}
    if best["run"] not in idx:
        return None
    t0, t1 = idx[best["run"]]
    sp = segment_split(z, t0, t1, dt)
    if sp is not None:
        sp["_run"] = best["run"]
        sp["_run_time"] = float(best["run_time"])
    return sp


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--control", default="control")
    ap.add_argument("--treat", default="diag")
    ap.add_argument("--d016", default=str(REPO_ROOT / "outputs" / "exp_016_design_check"
                                          / "l0c_diag"),
                    help="016-D 版の同じ迷路の記録（P5 の対照）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    dt = RobotParams().control_dt
    out = Path(args.out or (root / "aggregate.json"))

    (mc, A), (mt, B) = load_arm(root / args.control), load_arm(root / args.treat)
    common = sorted(set(A) & set(B))
    print(f"対照 = {mc['policy']}\n処理 = {mt['policy']}\n迷路 {len(common)} 件"
          f"／{mt['maze_dir']}／制御周期 {dt*1000:.0f} ms\n")

    # ---------------- P1・P2・P5 -------------------------------------
    print("【迷路ごとの (d) 最速タイムと経路効率】")
    print(f"{'迷路':<12}{'D':>4}{'対照 (d)':>10}{'処理 (d)':>10}{'差':>9}{'差 %':>8}"
          f"{'(a)':>6}{'(e′) 対照':>10}{'(e′) 処理':>10}")
    per, rel, n_a_c, n_a_t = [], [], 0, 0
    for mz in common:
        ra, rb = A[mz], B[mz]
        da, db = d_best_time(ra), d_best_time(rb)
        dtrue = int(ra[0]["d_true"])
        ka, kb = maze_kpi(_with_times(ra)), maze_kpi(_with_times(rb))
        n_a_c += bool(ka["goal_reached"])
        n_a_t += bool(kb["goal_reached"])
        r = None if (da in (None, 0) or db is None) else (db - da) / da
        if r is not None:
            rel.append(r)
        ea, eb = e_prime(ra, dtrue), e_prime(rb, dtrue)
        print(f"{mz:<12}{dtrue:>4}"
              f"{_f(da):>10}{_f(db):>10}"
              f"{_f(None if (da is None or db is None) else db - da):>9}"
              f"{('%+.1f%%' % (r*100)) if r is not None else '   —  ':>8}"
              f"{('○' if kb['goal_reached'] else '**×**'):>6}"
              f"{_f(ea, 3):>10}{_f(eb, 3):>10}")
        per.append(dict(maze=mz, d_true=dtrue, d_control=da, d_treat=db,
                        rel_diff=r, e_prime_control=ea, e_prime_treat=eb,
                        a_control=bool(ka["goal_reached"]), a_treat=bool(kb["goal_reached"])))

    # ---- 🔴 判定の前に全数の存在検査（教授指示・B1 の教訓）------------
    # **欠測が黙って「良い側の値」に化ける型**を防ぐ。
    # 例: (d) が None の迷路が落ちると、n が減ったまま中央値が出てしまう
    print("\n【全数の存在検査】**判定の前に、記録が全迷路ぶん揃っているかを確かめる**")
    holes = []
    for key, lab in (("d_control", "対照の (d)"), ("d_treat", "処理の (d)"),
                     ("rel_diff", "対応差"), ("e_prime_treat", "処理の (e′)")):
        n_miss = sum(1 for q in per if q[key] is None)
        print(f"  {lab:<14} 欠測 {n_miss} / {len(per)} 迷路")
        if n_miss and key != "e_prime_treat":
            holes.append(lab)
    if holes:
        print(f"  → 🔴 **判定不能**（{'・'.join(holes)} に欠測がある）")
        return 1
    print(f"  → ✅ **(d) と対応差は 20/20 迷路そろっている**（判定に進んでよい）")

    print(f"\n【P1】(d) の同じ迷路どうしの対応差")
    if rel:
        med = float(np.median(rel))
        n_worse = sum(1 for x in rel if x > 0)
        print(f"  **中央値 {med*100:+.1f} %**（範囲 {min(rel)*100:+.1f}〜{max(rel)*100:+.1f} %）")
        print(f"  悪化 {n_worse} 迷路 / 改善 {len(rel)-n_worse} 迷路 / n={len(rel)}")
        print(f"  **予測の範囲 +15 %〜+33 %** → "
              f"{'**的中**' if 0.15 <= med <= 0.33 else '**外れ**'}")
        print(f"  参考: 016-D の調整用迷路での値 **+34.3 %**"
              f"（差の縮み {(0.343-med)*100:+.1f} ポイント）")
    print(f"\n【P2】(a) ゴール到達: 対照 {n_a_c}/{len(common)}／**処理 {n_a_t}/{len(common)}**"
          f" → {'**的中**' if n_a_t == len(common) else '**外れ（安全側が削れた。P1 より先に原因を突き止める）**'}")

    # ---------------- P3・P4 -----------------------------------------
    print("\n【P3・P4】最速走行の区間ごとの内訳（**処理**）")
    print(f"{'迷路':<12}{'直進 %':>8}{'斜め %':>8}{'円弧 %':>8}{'他 %':>7}"
          f"{'円弧 v [m/s]':>13}{'斜め v [m/s]':>13}")
    splits = []
    for mz in common:
        sp = fast_run_split(root / args.treat, mz, B[mz], dt)
        if sp is None:
            continue
        splits.append((mz, sp))
        print(f"{mz:<12}{sp['straight']['frac']*100:>8.1f}{sp['diagonal']['frac']*100:>8.1f}"
              f"{sp['arc']['frac']*100:>8.1f}{sp['other']['frac']*100:>7.1f}"
              f"{_f(sp['arc']['v_mean'], 3):>13}{_f(sp['diagonal']['v_mean'], 3):>13}")
    if splits:
        fd = float(np.median([s['diagonal']['frac'] for _m, s in splits])) * 100
        va = [s['arc']['v_mean'] for _m, s in splits if s['arc']['v_mean']]
        vam = float(np.median(va)) if va else float("nan")
        print(f"\n  **斜め区間の割合 中央値 {fd:.1f} %**（予測 ≥ 30 %）→ "
              f"{'**的中**' if fd >= 30.0 else '**外れ（ボトルネックが移った）**'}")
        print(f"  **円弧区間の平均速度 中央値 {vam:.3f} m/s**（予測 ≥ 0.47・016-D は 0.451）→ "
              f"{'**的中**' if vam >= 0.47 else '**外れ**'}")

    # ---------------- P5（016-D との (e′) の比較）----------------------
    print("\n【P5】(e′) 経路効率 — **016-D 版（斜めあり）との比較**")
    d016 = Path(args.d016)
    p5 = None
    if (d016 / "runs_detail.json").exists():
        _m, D = load_arm(d016)
        pairs = []
        for mz in common:
            if mz not in D:
                continue
            ed = e_prime(D[mz], int(D[mz][0]["d_true"]))
            eb = e_prime(B[mz], int(B[mz][0]["d_true"]))
            if ed is not None and eb is not None:
                pairs.append((mz, ed, eb, eb - ed))
        if pairs:
            diffs = [q[3] for q in pairs]
            p5 = float(np.median(diffs))
            n_same = sum(1 for x in diffs if x == 0.0)
            print(f"  n={len(pairs)}／**差の中央値 {p5:+.4f}**"
                  f"（範囲 {min(diffs):+.4f}〜{max(diffs):+.4f}）")
            print(f"  **1 ビットも動かなかった迷路 {n_same}/{len(pairs)}** → "
                  f"{'**的中**' if n_same == len(pairs) else '**外れ（制御の変更が経路の選択に影響した）**'}")
            for mz, ed, eb, dd in pairs:
                if dd != 0.0:
                    print(f"    {mz}: {ed:.4f} → {eb:.4f}（{dd:+.4f}）")
    else:
        print(f"  ⚠️ 016-D の記録が無い（{d016}）。**判定不能**")

    json.dump(dict(control=mc["policy"], treat=mt["policy"], maze_dir=mt["maze_dir"],
                   git_rev_control=mc.get("git_rev"), git_rev_treat=mt.get("git_rev"),
                   n_mazes=len(common), per_maze=per,
                   p1_median_rel=float(np.median(rel)) if rel else None,
                   p2_a_control=n_a_c, p2_a_treat=n_a_t,
                   p3_p4=[dict(maze=m, **{k: v for k, v in s.items() if not k.startswith("_")},
                               run=s.get("_run"), run_time=s.get("_run_time"))
                          for m, s in splits],
                   p5_median_diff=p5),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"\n書き出し: {out}")
    return 0


def _with_times(runs):
    """`maze_kpi` は t_start / t_end を要る。走行番号から単調な代理値を作る。

    ⚠️ **(a)(b)(c) の判定に必要なのは「初回ゴールより後に開始したか」の順序だけ**で、
    実時刻の値そのものは使わない。走行は番号順に実行されるので順序は保たれる。
    """
    return [dict(r, t_start=float(r["run"]) * 1e6,
                 t_end=float(r["run"]) * 1e6 + 1.0) for r in runs]


def _f(x, prec=2):
    return "  —  " if x is None else f"{x:.{prec}f}"


if __name__ == "__main__":
    sys.exit(main())
