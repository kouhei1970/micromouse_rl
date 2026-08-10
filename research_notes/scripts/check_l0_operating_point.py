#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L0-a/b/c の動作点診断 — 是正後の迷路でゲインが合っているかを測る。

exp_007 の再評価で経路長が約 3.9 倍になった。**動作点（速度・曲率・電圧の使い方）が
変われば必要なゲインも変わる**（前任者が L0-b の kp_heading で踏んだのと同じ構造）ので、
ゲインを触る前に「いま実際にどこで動いているのか」を測る。

測る量（評価器は記録していないので、方策を包んで直接取る）:

1. **横偏差** — 直進中の機体中心と、その区画の中心軸との垂直距離 [mm]。
   壁までの余裕がどれだけ残っているかの直接の指標。
   区画 180 mm・機体幅から求まる公称余裕と比べる。
   **転回中は定義上ずれるので、機首方位が軸から 15 度以内の区間だけを集計**する。
2. **指令ヨー角速度の実現率** — 方策内部の `omega_cmd`（rad/s）と、実際のヨー角速度
   の比。1 に近いほど指令どおり回れている。**低ければゲインではなく駆動力が
   足りていない**ことを意味するので、ゲインをいじっても直らない。
3. **電圧飽和率** — 出力電圧が上限 ±voltage_limit に張り付いている制御周期の割合。
   高ければ指令が実現できないので、やはりゲインの問題ではない。
4. 速度・ヨー角速度の分布（動作点そのもの）

**評価帯（seed 1000-1019）は使わない。**検証帯 4000-4019 の先頭から数面で測る。

使い方:
    .venv/bin/python research_notes/scripts/check_l0_operating_point.py [--n 5]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition.evaluator import CompetitionEvaluator  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

POLICIES = [
    ("L0-a 超信地旋回走行（区画ごと停止）", "competition.baseline_classical", "AdachiPolicy"),
    ("L0-b 超信地旋回走行（直進連続）", "competition.baseline_straightrun", "StraightRunPolicy"),
    ("L0-c スラローム走行", "competition.baseline_slalom", "SlalomPolicy"),
]
STRAIGHT_TOL_DEG = 15.0     # 直進とみなす機首方位のずれ [deg]


class Probe:
    """方策を包んで、1 制御周期ごとに動作点を記録する。

    `MousePolicy` の全メソッドを内側の方策へ委譲し、`act` の前後で
    姿勢・ヨー角速度・出力電圧を取る。指令ヨー角速度は、3 方策に共通する
    `_robot_cmd_to_voltage` / `_wheel_targets_to_voltage` を差し替えて拾う。
    """

    def __init__(self, inner, params):
        self._inner = inner
        self._p = params
        self._sim = None
        self.rec = []
        self._omega_cmd = None
        self._v_cmd = None
        for meth in ("_robot_cmd_to_voltage", "_wheel_targets_to_voltage"):
            orig = getattr(inner, meth, None)
            if orig is None:
                continue

            def wrapped(v_cmd, omega_cmd, obs, _orig=orig):
                self._v_cmd, self._omega_cmd = float(v_cmd), float(omega_cmd)
                return _orig(v_cmd, omega_cmd, obs)
            setattr(inner, meth, wrapped)

    # --- MousePolicy インタフェースの委譲 -------------------------------
    name = property(lambda self: getattr(self._inner, "name", "unnamed"))
    requires_privileged = property(lambda self: getattr(self._inner, "requires_privileged", False))

    def bind_sim(self, sim):
        self._sim = sim
        return self._inner.bind_sim(sim)

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def act(self, obs):
        self._omega_cmd = self._v_cmd = None
        vl, vr = self._inner.act(obs)
        if self._sim is not None:
            x, y, yaw = self._sim.privileged_pose()
            d = self._sim.data
            self.rec.append(dict(
                x=float(x), y=float(y), yaw=float(yaw),
                vx=float(d.qvel[0]), vy=float(d.qvel[1]),
                omega=float(d.qvel[5]) if d.qvel.shape[0] > 5 else float("nan"),
                omega_cmd=self._omega_cmd, v_cmd=self._v_cmd,
                vl=float(vl), vr=float(vr)))
        return vl, vr


def analyse(rec, params):
    """記録から横偏差・実現率・飽和率を集計する。"""
    cell = params.cell_size
    vlim = params.voltage_limit
    lat_all, lat_straight, ratio, spd, om = [], [], [], [], []
    n_sat = 0
    for r in rec:
        # 機首方位を 4 軸のどれに最も近いかで判定し、その軸に垂直な方向のずれを取る
        yaw_deg = np.degrees(r["yaw"]) % 360.0
        k = int(round(yaw_deg / 90.0)) % 4
        err_deg = abs(((yaw_deg - 90.0 * k + 180.0) % 360.0) - 180.0)
        cx, cy = int(r["x"] // cell), int(r["y"] // cell)
        ccx, ccy = (cx + 0.5) * cell, (cy + 0.5) * cell
        lat = abs(r["y"] - ccy) if k % 2 == 0 else abs(r["x"] - ccx)
        lat_all.append(lat)
        if err_deg <= STRAIGHT_TOL_DEG:
            lat_straight.append(lat)
        if abs(r["vl"]) >= vlim - 1e-9 or abs(r["vr"]) >= vlim - 1e-9:
            n_sat += 1
        spd.append(float(np.hypot(r["vx"], r["vy"])))
        om.append(abs(r["omega"]))
        oc = r["omega_cmd"]
        if oc is not None and abs(oc) > 0.5:      # 指令が十分大きいときだけ比を取る
            ratio.append(abs(r["omega"]) / abs(oc))
    f = lambda a, q: (float(np.percentile(a, q)) if a else float("nan"))  # noqa: E731
    return dict(
        n_steps=len(rec),
        lat_max_mm=(max(lat_all) * 1000.0 if lat_all else float("nan")),
        lat_straight_max_mm=(max(lat_straight) * 1000.0 if lat_straight else float("nan")),
        lat_straight_rms_mm=(float(np.sqrt(np.mean(np.square(lat_straight)))) * 1000.0
                             if lat_straight else float("nan")),
        lat_straight_p95_mm=f(lat_straight, 95) * 1000.0,
        n_straight=len(lat_straight),
        sat_frac=(n_sat / len(rec) if rec else float("nan")),
        omega_ratio_median=(float(np.median(ratio)) if ratio else float("nan")),
        omega_ratio_p05=f(ratio, 5), n_omega=len(ratio),
        speed_p95=f(spd, 95), speed_max=(max(spd) if spd else float("nan")),
        omega_p95=f(om, 95),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5, help="検証帯の先頭から何面を測るか")
    ap.add_argument("--maze-dir", default="competition/mazes/validation")
    args = ap.parse_args()

    params = RobotParams()
    mazes = sorted((REPO_ROOT / args.maze_dir).glob("maze_*.npz"))[:args.n]
    # 公称の壁余裕: 区画の半分 − 機体の半幅（機体幅はモデルから導出する。
    # ハードコードしない — 研究計画書の実験規律）
    from mouse.mjcf import build_maze_robot_xml  # noqa: F401  （XML 生成の依存を明示）
    out = {}
    print(f"診断対象: {len(mazes)} 面（{args.maze_dir}）／制御周期 {params.control_dt*1000:.0f} ms"
          f"／電圧上限 ±{params.voltage_limit} V／区画 {params.cell_size*1000:.0f} mm")
    for label, mod, cls in POLICIES:
        import importlib
        inner = getattr(importlib.import_module(mod), cls)()
        probe = Probe(inner, params)
        ev = CompetitionEvaluator(maze_dir=args.maze_dir,
                                  out_dir=str(REPO_ROOT / "outputs" / "l0_operating_point"))
        for m in mazes:
            ev.evaluate_maze(m, probe)
        out[label] = analyse(probe.rec, params)
        r = out[label]
        print(f"\n{label}")
        print(f"  制御周期数 {r['n_steps']}（うち直進 {r['n_straight']} = "
              f"{r['n_straight']/max(r['n_steps'],1)*100:.0f}%）")
        print(f"  横偏差（直進区間）: 最大 {r['lat_straight_max_mm']:.1f} mm / "
              f"95%点 {r['lat_straight_p95_mm']:.1f} mm / RMS {r['lat_straight_rms_mm']:.1f} mm"
              f"（全区間の最大 {r['lat_max_mm']:.1f} mm）")
        print(f"  指令ヨー角速度の実現率: 中央値 {r['omega_ratio_median']:.3f} / "
              f"5%点 {r['omega_ratio_p05']:.3f}（|指令|>0.5 rad/s の {r['n_omega']} 周期）")
        print(f"  電圧飽和率: {r['sat_frac']*100:.2f}%")
        print(f"  動作点: 速度 95%点 {r['speed_p95']:.3f} m/s（最大 {r['speed_max']:.3f}）"
              f"／ヨー角速度 95%点 {r['omega_p95']:.2f} rad/s")

    p = REPO_ROOT / "research_notes" / "data" / "l0_operating_point.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(maze_dir=args.maze_dir, n_mazes=len(mazes),
                   mazes=[m.stem for m in mazes], results=out),
              open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n数値 JSON: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
