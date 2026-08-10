# research_notes/scripts/check_goal_recognizability.py
# ゴール認識の机上検証（2026-08-10、学生B。教授指示により M2 の実装前に必須）。
#
# 問い: **距離センサ 4 本の瞬時値だけで「いまゴール区画にいる」と分かるか。**
# 分かるなら M2 のゴール認識は記憶なしで成立する（設計案 §1 の案 A）。
# 分からないなら自己位置推定（案 B）が必須になる。
#
# 方法: 6x6 迷路 20 本について、全区画 × 4 方位（36×4 = 144 姿勢／迷路）で
# 機体を置き、MuJoCo の距離センサを実際に読む（幾何を自前で書くと学習時と
# 値がずれるため、実物のセンサモデルをそのまま使う）。得られた 4 次元ベクトルに
# 「ゴール区画か否か」のラベルを付け、次を数える:
#   (1) 原理的に識別不能な姿勢: センサ値が（許容差内で）一致するのにラベルが違う組
#   (2) 最近傍法の誤り率（leave-one-out）
#   (3) 単純な閾値則がどこまで効くか（実装しやすさの目安）
#
# 学習 1 本 30 分に対し本検証は数分。失敗を先に潰すための投資。
#
# 実行: .venv/bin/python research_notes/scripts/check_goal_recognizability.py
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mouse.maze6_gen import (  # noqa: E402
    GOAL_CELLS, SIZE, generate_maze, initial_heading_deg,
)
from mouse.mjcf import build_maze_robot_xml  # noqa: E402
from mouse.params import RobotParams  # noqa: E402
from mouse.sim import MouseSim  # noqa: E402

SEEDS = list(range(6000, 6020))
HEADINGS = (0.0, 90.0, 180.0, 270.0)
TOL_MM = 5.0     # 「同じセンサ値」とみなす許容差 [mm]


def collect(mode):
    """全迷路 × 全区画 × 4 方位のセンサ値とラベルを集める。"""
    params = RobotParams()
    n_dist = len(params.sensors)
    names = [s["name"] for s in params.sensors]
    X, y, meta = [], [], []

    for seed in SEEDS:
        m = generate_maze(seed, mode=mode)
        fd, tmp = tempfile.mkstemp(suffix=".xml", prefix=f"maze6_{seed}_")
        os.close(fd)
        try:
            cs = params.cell_size
            sx, sy = m["start"]
            build_maze_robot_xml(
                m["v_walls"], m["h_walls"], tmp, model_name=f"maze6_{seed}",
                mouse_pos=f"{sx * cs + cs / 2} {sy * cs + cs / 2} 0.002",
                mouse_euler=f"0 0 {initial_heading_deg(m['v_walls'], m['h_walls'], m['start'])}",
                center_goal=False, params=params)
            sim = MouseSim(tmp, params=params)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

        for cx in range(SIZE):
            for cy in range(SIZE):
                for hd in HEADINGS:
                    sim.full_reset(cell=(cx, cy), heading_deg=hd)
                    X.append(np.asarray(sim.observation()[0:n_dist], dtype=np.float64))
                    y.append(1 if (cx, cy) in GOAL_CELLS else 0)
                    meta.append((seed, cx, cy, hd))
    return np.array(X), np.array(y), meta, names


def analyse(X, y, meta, names, mode):
    n, n_goal = len(y), int(y.sum())
    print(f"\n{'='*74}\nmode={mode}: 姿勢 {n} 件（うちゴール区画 {n_goal} 件 = {n_goal/n:.1%}）\n{'='*74}")

    Xmm = X * 1000.0
    print(f"センサ値 [mm] の平均（{'/'.join(names)}）")
    print(f"  ゴール区画  : {np.round(Xmm[y == 1].mean(axis=0), 1)}")
    print(f"  それ以外    : {np.round(Xmm[y == 0].mean(axis=0), 1)}")

    # (1) 原理的に識別不能な姿勢: 許容差内で一致するのにラベルが違う組
    key = lambda v: tuple(np.round(v / TOL_MM).astype(int))  # noqa: E731
    buckets = {}
    for i in range(n):
        buckets.setdefault(key(Xmm[i]), []).append(i)
    # ゴールは全体の 11% しかないので、「混在バケットの全件」を数えると
    # 大きな非ゴールの塊に 1 件混じっただけで巨大な数字になり、過大評価になる。
    # 知りたいのは「**ゴール姿勢のうち、非ゴールと区別できないもの**」の割合。
    amb_goal = 0
    amb_examples = []
    for k, idxs in buckets.items():
        labs = [y[i] for i in idxs]
        if 1 in labs and 0 in labs:
            amb_goal += sum(labs)          # そのバケット内のゴール姿勢の件数
            if len(amb_examples) < 3:
                amb_examples.append([meta[i] + (int(y[i]),) for i in idxs[:4]])
    print(f"\n(1) ゴール姿勢のうち、±{TOL_MM:.0f} mm 以内で非ゴール姿勢と一致してしまうもの: "
          f"{amb_goal} / {n_goal} 件 = {amb_goal/max(n_goal,1):.1%}")
    print(f"    （＝この割合のゴール姿勢は、センサ 4 本の瞬時値では原理的にゴールと確定できない）")
    for ex in amb_examples:
        print(f"    例: {ex}")

    # (2) 最近傍法の誤り率（leave-one-out）
    d2 = ((Xmm[:, None, :] - Xmm[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = d2.argmin(axis=1)
    err = (y[nn] != y).mean()
    goal_recall = (y[nn][y == 1] == 1).mean()
    goal_prec = (y[y[nn] == 1] == 1).mean() if (y[nn] == 1).any() else 0.0
    baseline = y.mean()   # 「全部が非ゴール」と答えたときの誤り率
    print(f"\n(2) 最近傍法（leave-one-out）: 誤り率 {err:.1%} "
          f"（全部を非ゴールと答えるだけで {baseline:.1%} なので、この値だけでは判断できない）")
    print(f"    **ゴールの再現率 {goal_recall:.1%} / 適合率 {goal_prec:.1%}** ← 実質的な指標")

    # (3) 単純な閾値則: 「4 本すべてが閾値より遠い」で判定できるか
    print("\n(3) 単純な閾値則『4 本すべて > t mm ならゴール』の性能")
    best = None
    for t in range(50, 300, 10):
        pred = (Xmm > t).all(axis=1).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if best is None or f1 > best[1]:
            best = (t, f1, tp, fp, fn)
    t, f1, tp, fp, fn = best
    print(f"    最良 t={t} mm: F1={f1:.3f}（正しく検出 {tp} / 誤検出 {fp} / 見逃し {fn}）")
    return dict(ambiguous_goal_rate=amb_goal / max(n_goal, 1), nn_error=err,
                goal_recall=goal_recall, goal_precision=goal_prec, best_threshold_f1=f1)


def main():
    results = {}
    for mode in ("loop", "full"):
        X, y, meta, names = collect(mode)
        results[mode] = analyse(X, y, meta, names, mode)

    print(f"\n{'='*74}\n判定\n{'='*74}")
    for mode, r in results.items():
        # 判定は「ゴールを見つけられるか」で行う。再現率・適合率がともに 95% 以上なら
        # 幾何だけでゴールを確定できる。M2 の gate は完走率 90% なので、ゴール認識で
        # 20% 取りこぼす方策では届かない。
        ok = r["goal_recall"] >= 0.95 and r["goal_precision"] >= 0.95
        verdict = ("案 A（幾何でゴール認識）は成立しうる" if ok
                   else "**案 A は不成立** → 自己位置推定（案 B）が必要")
        print(f"  {mode}: ゴール再現率 {r['goal_recall']:.1%} / 適合率 {r['goal_precision']:.1%} "
              f"/ 原理的に確定できないゴール姿勢 {r['ambiguous_goal_rate']:.1%} → {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
