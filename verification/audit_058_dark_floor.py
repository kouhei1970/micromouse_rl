"""
verification/audit_058_dark_floor.py
================
床を「艶消し黒のベニヤ」として扱い直したときに、相互反射の寄与がどう変わるかを測る。

きっかけ（ユーザ指摘・2026-08-22）:
  「床はベニヤ板のざらざらの艶消し黒であり、これは会場の迷路によって表面がだいぶ違う。
    これは光の反射に大きな影響を与えると思いますがどうでしょう」

🔴 **`AUDIT_056` は床を壁と同じ拡散反射率 0.8・鏡面 0.10 で計算していた。**白い壁と同じ
扱いであり、実物と食い違う。しかも `AUDIT_050` が突き止めた間接光の主経路は
**壁→床→壁→PT の反射 3 回**なので、この経路は床の反射率をそのまま 1 個含む。

本スクリプトは `verification/audit_050_raycast.py` に追加した `floor_diffuse`（床だけ別の
反射率。既定 None は従来と厳密に同じ挙動）を使い、`AUDIT_056` と同じ姿勢の標本で
床の反射率を振って増分を測り直す。

使い方:
  .venv/bin/python verification/audit_058_dark_floor.py --set grid  --floor 0.10
  .venv/bin/python verification/audit_058_dark_floor.py --set front --floor 0.10
  .venv/bin/python verification/audit_058_dark_floor.py --summary
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verification.audit_050_raycast import raycast_response, Sensor          # noqa: E402
from verification.audit_056_interreflection import (                          # noqa: E402
    I_FULL, OUT_DIR, build_ir_specs, load_geometry, _pct95, stage2_rho_subsample_indices,
)

N_RAYS, SEED = 15_000, 777001          # AUDIT_050 追記3 の決定と同じ
WALL_RHO = 0.8                          # 壁は据え置き（もっともらしい上端）


def load_points(which):
    if which == "grid":
        g = json.loads((OUT_DIR / "stage2_grid.json").read_text())["points"]
        idxs, step = stage2_rho_subsample_indices()
        keep = set(idxs)
        return [p for p in g if p["idx"] in keep], f"本体の格子から {step} 点おきに抜いた {len(keep)} 点"
    g = json.loads((OUT_DIR / "front_grid.json").read_text())["points"]
    return g, f"行き止まりの壁へ接近する {len(g)} 点"


def run(which, floor):
    pts, desc = load_points(which)
    p, rects, _, _, _ = load_geometry()
    specs = build_ir_specs(p)
    path = OUT_DIR / f"audit058_{which}_floor{floor:.2f}.json"
    rec = json.loads(path.read_text())["records"] if path.exists() else {}
    for i, pt in enumerate(pts):
        k = str(pt["idx"])
        if k in rec:
            continue
        sp = specs[pt["sensor_idx"]]
        s = Sensor(name=sp.name, pos=tuple(sp.pos), axis=tuple(sp.axis))
        kw = dict(n_rays=N_RAYS, seed=SEED, led_half_angle_deg=sp.led_half_angle_deg,
                  pt_half_angle_deg=sp.pt_half_angle_deg, separation_m=sp.separation_m,
                  diffuse=WALL_RHO, floor_diffuse=floor)
        pose = (pt["x"], pt["y"], pt["theta"])
        b1 = raycast_response(s, pose, rects, max_bounces=1, **kw) / I_FULL
        b4 = raycast_response(s, pose, rects, max_bounces=4, **kw) / I_FULL
        rec[k] = {"b1": b1, "delta": b4 - b1}
        if i % 10 == 0:
            path.write_text(json.dumps({"meta": {"set": which, "floor": floor, "desc": desc},
                                        "records": rec}))
    path.write_text(json.dumps({"meta": {"set": which, "floor": floor, "desc": desc},
                                "records": rec}))
    print(f"[{which} 床ρ={floor}] {len(rec)}/{len(pts)} 点")


def summary():
    print(f"{'標本':<10}{'床ρ':>7}{'n':>6}{'|Δ| 95%点':>12}{'中央値':>10}{'最大':>10}{'帯':>8}")
    for f in sorted(OUT_DIR.glob("audit058_*.json")):
        o = json.loads(f.read_text()); r = o["records"]
        d = [abs(v["delta"]) for v in r.values()]
        m = _pct95(d)
        band = "帯1" if m <= 0.01 else ("帯2" if m <= 0.05 else "帯3")
        print(f"{o['meta']['set']:<10}{o['meta']['floor']:7.2f}{len(d):6d}{m:12.5f}"
              f"{np.median(d):10.5f}{max(d):10.5f}{band:>8}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["grid", "front"])
    ap.add_argument("--floor", type=float)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary()
    else:
        run(a.set, a.floor)
