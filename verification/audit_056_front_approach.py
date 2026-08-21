"""
verification/audit_056_front_approach.py
================
AUDIT_056 の**副次**の測定（事前登録 追記3）: 行き止まり（状況 B）で前方の壁へ近づく姿勢を足す。

本体の格子は前後を `dx ∈ {−40, 0, +40}mm` で刻んでいたが、機体が壁と重なる姿勢を除外する
規則により状況 B の `dx=+40` が全て除外され、**前方センサから壁までの距離が 53.0mm の
1 点しか標本に無かった**。機体が触れずに前進できるのは +34mm までで、応答の山（42.5mm）を
通り抜ける範囲がまるごと抜けている。そこを `dx ∈ {+10, +20, +30}mm` で埋める。

🔴 **`M5` の主判定は本体の格子（1000 点）だけで行う。本スクリプトの結果は `M5_front` として
別に報告する**（後から標本を足して判定量を動かさない。事前登録 追記3）。

使い方（10 分以内に分割して走らせる）:
  .venv/bin/python verification/audit_056_front_approach.py --stage grid
  .venv/bin/python verification/audit_056_front_approach.py --stage response_s
  .venv/bin/python verification/audit_056_front_approach.py --stage raycast --bounces 1
  .venv/bin/python verification/audit_056_front_approach.py --stage raycast --bounces 4 --start 0 --end 250
  .venv/bin/python verification/audit_056_front_approach.py --stage summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mouse.ir_sensor import response                                  # noqa: E402
from verification.audit_050_raycast import raycast_response, Sensor   # noqa: E402
from verification.audit_056_interreflection import (                  # noqa: E402
    CHASSIS_HALF_LENGTH_M, CHASSIS_HALF_WIDTH_M, DTHETA_DEG, DY_MM_LIST, I_FULL, MAZE_PATH,
    OUT_DIR, SURF, SURF08, _chassis_clearance, _pct95, _situation_cells, _situation_pose,
    build_ir_specs, load_geometry,
)

DX_MM_FRONT = (10, 20, 30)     # 事前登録 追記3
SITUATION = "B"
N_RAYS = 15_000                # AUDIT_050 追記3 の決定（同一乱数種の差なので雑音は打ち消える）
SEED = 777001
PREFIX = "front_"


def _path(name: str) -> Path:
    return OUT_DIR / f"{PREFIX}{name}.json"


def _save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False))


def build_grid() -> Dict:
    p, rects, W, H, cell = load_geometry()
    d = np.load(MAZE_PATH)
    situations = _situation_cells(d["v_walls"], d["h_walls"], W, H)
    cx, cy = situations[SITUATION]
    specs = build_ir_specs(p)
    names = [s.name for s in specs]

    points: List[Dict] = []
    n_total = n_excl = idx = 0
    for dx_mm in DX_MM_FRONT:
        for dy_mm in DY_MM_LIST:
            for dtheta_deg in DTHETA_DEG:
                x, y, theta = _situation_pose(cx, cy, cell, dx_mm, dy_mm, dtheta_deg)
                n_total += 1
                if _chassis_clearance(x, y, theta, rects) < 0.0:
                    n_excl += 1
                    continue
                for si, name in enumerate(names):
                    points.append({
                        "idx": idx, "situation": SITUATION, "cell_cx": cx, "cell_cy": cy,
                        "dx_mm": dx_mm, "dy_mm": dy_mm, "dtheta_deg": dtheta_deg,
                        "sensor_idx": si, "sensor_name": name, "x": x, "y": y, "theta": theta,
                    })
                    idx += 1
    meta = {
        "dx_mm": list(DX_MM_FRONT), "dy_mm_list": list(DY_MM_LIST), "dtheta_deg": list(DTHETA_DEG),
        "situation": SITUATION, "cell": [cx, cy],
        "n_pose_total": n_total, "n_pose_excluded": n_excl, "n_pose_included": n_total - n_excl,
        "n_points": len(points),
        "chassis_half_width_m": CHASSIS_HALF_WIDTH_M, "chassis_half_length_m": CHASSIS_HALF_LENGTH_M,
        "maze": str(MAZE_PATH.relative_to(REPO_ROOT)),
        "note": "AUDIT_056 事前登録 追記3 の副次の格子。M5 の主判定には混ぜない",
    }
    out = {"meta": meta, "points": points}
    _save(_path("grid"), out)
    print(f"[front grid] 姿勢 {n_total-n_excl}/{n_total}（除外 {n_excl}）・点数 {len(points)}")
    return out


def _grid() -> Dict:
    return json.loads(_path("grid").read_text())


def run_response_s(led_half_angle_deg=None) -> None:
    """モデル S（面積分・更新後の既定値）。否定対照は LED 半値角 7°。"""
    p, rects, _, _, _ = load_geometry()
    specs = build_ir_specs(p)
    name = "response_s" if led_half_angle_deg is None else "response_s_negctrl"
    rec: Dict[str, Dict] = {}
    for pt in _grid()["points"]:
        spec = specs[pt["sensor_idx"]]
        if led_half_angle_deg is not None:
            spec = type(spec)(**{**spec.__dict__, "led_half_angle_deg": led_half_angle_deg})
        v = response(spec, (pt["x"], pt["y"], pt["theta"]), rects, SURF,
                     occlusion=True, include_floor=True)
        rec[str(pt["idx"])] = {"value": v, "i_full_ratio": v / I_FULL}
    _save(_path(name), {"meta": {"led_half_angle_deg": led_half_angle_deg}, "records": rec})
    print(f"[front {name}] {len(rec)} 件")


def run_raycast(bounces: int, start: int, end: int) -> None:
    p, rects, _, _, _ = load_geometry()
    specs = build_ir_specs(p)
    name = f"ray_b{bounces}"
    path = _path(name)
    prev = json.loads(path.read_text())["records"] if path.exists() else {}
    pts = _grid()["points"]
    for pt in pts:
        if not (start <= pt["idx"] < end) or str(pt["idx"]) in prev:
            continue
        spec = specs[pt["sensor_idx"]]
        s = Sensor(name=spec.name, pos=tuple(spec.pos), axis=tuple(spec.axis))
        v = raycast_response(
            s, (pt["x"], pt["y"], pt["theta"]), rects, n_rays=N_RAYS, seed=SEED,
            max_bounces=bounces, led_half_angle_deg=spec.led_half_angle_deg,
            pt_half_angle_deg=spec.pt_half_angle_deg, separation_m=spec.separation_m)
        prev[str(pt["idx"])] = {"value": v, "i_full_ratio": v / I_FULL}
        _save(path, {"meta": {"max_bounces": bounces, "n_rays": N_RAYS, "seed": SEED}, "records": prev})
    print(f"[front {name}] {len(prev)}/{len(pts)} 件")


def run_summary() -> Dict:
    grid = _grid()
    by_idx = {p["idx"]: p for p in grid["points"]}
    b1 = json.loads(_path("ray_b1").read_text())["records"]
    b4 = json.loads(_path("ray_b4").read_text())["records"]
    s = json.loads(_path("response_s").read_text())["records"]
    common = sorted(set(b1) & set(b4), key=int)
    rows = []
    for k in common:
        pt = by_idx[int(k)]
        delta = b4[k]["i_full_ratio"] - b1[k]["i_full_ratio"]
        sv = s[k]["i_full_ratio"] if k in s else None
        rows.append({**pt, "delta": delta, "s": sv,
                     "rel": (delta / sv) if (sv and abs(sv) > 1e-12) else None})
    m5f = _pct95([abs(r["delta"]) for r in rows])
    band = "帯1(<=0.01)" if m5f <= 0.01 else ("帯2(0.01<..<=0.05)" if m5f <= 0.05 else "帯3(>0.05)")
    print(f"[front summary] n={len(rows)}/{len(grid['points'])}  M5_front = {m5f:.5f}  {band}")
    for dx in DX_MM_FRONT:
        sub = [r for r in rows if r["dx_mm"] == dx]
        if sub:
            print(f"  dx=+{dx}mm: n={len(sub)} |Δ|中央値={np.median([abs(r['delta']) for r in sub]):.5f} "
                  f"95%点={_pct95([abs(r['delta']) for r in sub]):.5f} "
                  f"最大={max(abs(r['delta']) for r in sub):.5f}")
    out = {"M5_front": m5f, "band": band, "n": len(rows), "rows": rows}
    _save(_path("summary"), out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["grid", "response_s", "response_s_negctrl", "raycast", "summary"],
                    required=True)
    ap.add_argument("--bounces", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    a = ap.parse_args()
    if a.stage == "grid":
        build_grid()
    elif a.stage == "response_s":
        run_response_s()
    elif a.stage == "response_s_negctrl":
        run_response_s(led_half_angle_deg=7.0)
    elif a.stage == "raycast":
        run_raycast(a.bounces, a.start, a.end)
    else:
        run_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
