#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""斜め走行の幾何を**モデルから導出**する（exp_016 段階 016-A・カード §1 の再現）。

**寸法をハードコードしない。**機体はロボットモデル（`assets/mouse_v2.xml`）から、
区画・壁・柱は `mouse/mjcf.py` と `mouse/params.py` から取る。

--------------------------------------------------------------------------
なぜ `body_footprint` を使わないのか（裁定 R33）
--------------------------------------------------------------------------
`competition/evaluator.py` の `body_footprint` は各 geom の **AABB の 8 隅**を
機体座標へ写して外接箱を取る。**回転した geom では AABB の隅が実体の外側にある**
ので、幅が実際より大きく出る（82.81 mm 対 真値 80.00 mm）。

ゴール成立判定では**安全側に大きく見積もってよい**のでこれで正しいが、
**斜め走路の余裕はまさにその 1.4 mm が効く**ので、ここでは厳密に測る:

- **メッシュ**: 頂点をすべて機体座標へ写す（`mesh_vert`）
- **円柱・カプセル**: 支持関数（半径 $r\\sqrt{d_x^2+d_y^2}$ ＋ 半長 $h|d_z|$）
- **球**: 半径
- **箱**: 8 隅（回転しても厳密）

--------------------------------------------------------------------------
斜め走路の幾何
--------------------------------------------------------------------------
柱は格子点（区画の角）に一辺 `POST_SIZE` の正方形として立つ。斜め走路の中心線は
隣り合う柱の中点を結ぶ 45° 線で、**中心線から柱中心までの距離は
$(\\text{区画}/2)/\\sqrt2$**、柱の角までを差し引いた片側自由幅は

$$\\text{free} = \\frac{c/2}{\\sqrt2} - \\frac{p}{2}\\sqrt2$$

（$c$ = 区画寸法、$p$ = 柱の一辺）。

**機体が方位 $\\psi$ だけ走路からずれると、走路に直交する張り出しが増える**:

$$w(\\psi) = \\frac{L}{2}|\\sin\\psi| + \\frac{W}{2}|\\cos\\psi|,\\qquad
  e_{y,\\max}(\\psi) = \\text{free} - w(\\psi)$$

**これは機体を $L \\times W$ の矩形とみなした保守的な上界である**（真の凸包は
矩形に内包される）。准教授の $h(\\theta)$ 表が届いたら 016-B で置き換える。

使い方:
    .venv/bin/python experiments/exp_016_diagonal/geometry.py [--out <json>]
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco  # noqa: E402

from mouse.mjcf import POST_SIZE, WALL_THICKNESS  # noqa: E402
from mouse.params import RobotParams  # noqa: E402

MODEL_XML = REPO_ROOT / "assets" / "mouse_v2.xml"
MOUSE_BODY = "mouse"

# mjtGeom の番号（読みやすさのため名前で持つ）
G_SPHERE, G_CAPSULE, G_CYLINDER, G_BOX, G_MESH = 2, 3, 5, 6, 7


def _descendant_bodies(model, root_bid):
    out = {root_bid}
    for b in range(model.nbody):
        p = b
        while p != 0:
            p = model.body_parentid[p]
            if p == root_bid:
                out.add(b)
                break
    return out


def _geom_points_body_frame(model, data, g, bpos, bmat):
    """geom g の外形を代表する点列を**機体座標**で返す。

    メッシュは頂点そのもの、箱は 8 隅（どちらも回転しても厳密）。
    円柱・カプセル・球は点列にできないので None を返し、呼び出し側が
    支持関数で扱う。
    """
    t = int(model.geom_type[g])
    c = data.geom_xpos[g]
    R = data.geom_xmat[g].reshape(3, 3)
    if t == G_MESH:
        mid = model.geom_dataid[g]
        a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        V = model.mesh_vert[a:a + n].reshape(-1, 3)
        W = (R @ V.T).T + c
        return (bmat.T @ (W - bpos).T).T
    if t == G_BOX:
        hs = model.geom_size[g]
        pts = [c + R @ np.array([sx * hs[0], sy * hs[1], sz * hs[2]])
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        return np.array([bmat.T @ (p - bpos) for p in pts])
    return None


def _support(model, data, g, direction_body, bpos, bmat):
    """方向 `direction_body`（機体座標の単位ベクトル）への支持量と中心座標。"""
    t = int(model.geom_type[g])
    c = data.geom_xpos[g]
    R = data.geom_xmat[g].reshape(3, 3)
    s = model.geom_size[g]
    d_world = bmat @ direction_body
    dg = R.T @ d_world                    # geom 座標での方向
    if t == G_SPHERE:
        sup = float(s[0])
    elif t == G_CYLINDER:
        sup = float(s[0] * math.hypot(dg[0], dg[1]) + s[1] * abs(dg[2]))
    elif t == G_CAPSULE:
        sup = float(s[0] + s[1] * abs(dg[2]))
    else:
        raise ValueError(f"支持関数を持たない geom 種別: {t}")
    center = bmat.T @ (c - bpos)
    return sup, np.asarray(center, dtype=float)


def body_extent_exact(xml_path=MODEL_XML, body_name=MOUSE_BODY):
    """機体の**厳密な**外形（機体座標）を返す。

    Returns:
        dict(half_width_m, length_m, x_min_m, x_max_m, per_geom=[...])
    """
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise SystemExit(f"body '{body_name}' が見つからない")
    bodies = _descendant_bodies(model, bid)
    bpos, bmat = data.xpos[bid], data.xmat[bid].reshape(3, 3)

    ey = np.array([0.0, 1.0, 0.0])
    ex = np.array([1.0, 0.0, 0.0])
    half_w, x_lo, x_hi, rows = 0.0, float("inf"), float("-inf"), []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in bodies:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"(無名 g{g})"
        pts = _geom_points_body_frame(model, data, g, bpos, bmat)
        if pts is not None:
            y = float(np.abs(pts[:, 1]).max())
            xl, xh = float(pts[:, 0].min()), float(pts[:, 0].max())
        else:
            sy, ctr = _support(model, data, g, ey, bpos, bmat)
            sx, _ = _support(model, data, g, ex, bpos, bmat)
            y = float(abs(ctr[1]) + sy)
            xl, xh = float(ctr[0] - sx), float(ctr[0] + sx)
        rows.append(dict(geom=name, type=int(model.geom_type[g]),
                         abs_y_max_m=y, x_min_m=xl, x_max_m=xh))
        half_w = max(half_w, y)
        x_lo, x_hi = min(x_lo, xl), max(x_hi, xh)
    rows.sort(key=lambda r: -r["abs_y_max_m"])
    return dict(half_width_m=half_w, width_m=2 * half_w,
                length_m=x_hi - x_lo, x_min_m=x_lo, x_max_m=x_hi, per_geom=rows)


def rotation_center(xml_path=MODEL_XML, body_name=MOUSE_BODY):
    """**掃引の計算に要る回転中心**と、各基準点の関係を機体座標で返す。

    差動二輪の回転中心は**左右の車輪ヒンジの中点**である。これが
    `privileged_pose()` の基準点（機体原点）や外接矩形の中心とずれていると、
    **掃引の計算に座標変換が要る**（准教授の申し送り・教授割当 2026-08-13）。

    Returns:
        dict(axle_mid_m, hinges, com_m, note)
    """
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    bpos, bmat = data.xpos[bid], data.xmat[bid].reshape(3, 3)
    bodies = _descendant_bodies(model, bid)

    hinges = []
    for j in range(model.njnt):
        if model.jnt_bodyid[j] not in bodies:
            continue
        if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        loc = bmat.T @ (np.asarray(data.xanchor[j]) - bpos)
        hinges.append(dict(joint=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
                           pos_m=[float(v) for v in loc]))
    axle = (np.mean([h["pos_m"] for h in hinges], axis=0) if hinges
            else np.zeros(3))
    com = bmat.T @ (data.subtree_com[bid] - bpos)
    return dict(hinges=hinges, axle_mid_m=[float(v) for v in axle],
                com_m=[float(v) for v in com],
                note=("回転中心 = 左右車輪ヒンジの中点。機体原点 = privileged_pose の基準点。"
                      "外接矩形の中心は x,y とも 0（外形が原点対称のため）"))


def diagonal_clearance(cell_size_m, post_size_m):
    """斜め走路の片側自由幅 [m]（中心線から柱の角まで）。"""
    d_center = (cell_size_m / 2.0) / math.sqrt(2.0)
    return d_center - (post_size_m / 2.0) * math.sqrt(2.0), d_center


def lateral_budget(free_m, length_m, width_m, psi_deg):
    """方位偏差 psi [deg] のときに許される横偏差 [m]（矩形とみなした保守的上界）。"""
    p = math.radians(psi_deg)
    w = (length_m / 2.0) * abs(math.sin(p)) + (width_m / 2.0) * abs(math.cos(p))
    return free_m - w, w


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                        text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "exp_016_diagonal"
                                          / "016a" / "geometry.json"))
    args = ap.parse_args()

    p = RobotParams()
    body = body_extent_exact()
    free, d_center = diagonal_clearance(p.cell_size, POST_SIZE)

    mm = 1000.0
    print("【機体（モデルから厳密に）】")
    print(f"  幅 {body['width_m']*mm:.3f} mm ／ 長 {body['length_m']*mm:.2f} mm"
          f"（x [{body['x_min_m']*mm:.2f}, {body['x_max_m']*mm:.2f}]）")
    print("  |y|max の大きい geom:")
    for r in body["per_geom"][:5]:
        print(f"    {r['geom']:<22} type={r['type']}  |y|max {r['abs_y_max_m']*mm:7.3f} mm")

    rot = rotation_center()
    print("\n【基準点の一致（掃引の座標変換が要るか）】")
    for h in rot["hinges"]:
        print(f"  ヒンジ {h['joint']:<20} 機体座標 "
              f"({h['pos_m'][0]*mm:7.2f}, {h['pos_m'][1]*mm:7.2f}, {h['pos_m'][2]*mm:7.2f}) mm")
    ax = rot["axle_mid_m"]
    print(f"  **回転中心（車軸中点） ({ax[0]*mm:.3f}, {ax[1]*mm:.3f}) mm**"
          f" ／ 機体原点 = privileged_pose の基準点 (0.000, 0.000) mm")
    print(f"  外接矩形の中心 ({(body['x_min_m']+body['x_max_m'])/2*mm:.3f}, 0.000) mm")
    print(f"  重心 ({rot['com_m'][0]*mm:.3f}, {rot['com_m'][1]*mm:.3f}) mm"
          f"（**掃引には使わない** — 要るのは回転中心）")
    d_axle = math.hypot(ax[0], ax[1])
    print(f"  → 回転中心と基準点のずれ **{d_axle*mm:.6f} mm**"
          f"（{'一致。座標変換は不要' if d_axle < 1e-9 else '⚠️ ずれあり。掃引に座標変換が要る'}）")

    print("\n【斜め走路（区画・柱から導出）】")
    print(f"  区画 {p.cell_size*mm:.1f} mm ／ 壁厚 {WALL_THICKNESS*mm:.1f} mm"
          f" ／ 柱 {POST_SIZE*mm:.1f} mm")
    print(f"  中心線から柱中心まで {d_center*mm:.3f} mm")
    print(f"  **片側自由幅 {free*mm:.3f} mm**")
    print(f"  斜めが成立する最大機体幅 {2*free*mm:.3f} mm（現機体 {body['width_m']*mm:.2f} mm）")

    print("\n【許容横偏差（機体を矩形とみなした保守的上界）】")
    print(f"  {'ψ [deg]':>8}{'w(ψ) [mm]':>12}{'e_y,max [mm]':>14}")
    table = []
    for psi in (0.0, 1.0, 2.0, 5.0, 10.0, 15.0):
        e, w = lateral_budget(free, body["length_m"], body["width_m"], psi)
        table.append(dict(psi_deg=psi, w_m=w, e_y_max_m=e))
        print(f"  {psi:>8.1f}{w*mm:>12.3f}{e*mm:>14.3f}")
    e0, _ = lateral_budget(free, body["length_m"], body["width_m"], 0.0)
    e1, _ = lateral_budget(free, body["length_m"], body["width_m"], 1.0)
    print(f"  → 感度: 方位 1° につき横偏差の許容が {(e0-e1)*mm:.3f} mm 減る")

    payload = dict(
        git_rev=git_rev(), model_xml=str(MODEL_XML.relative_to(REPO_ROOT)),
        cell_size_m=p.cell_size, wall_thickness_m=WALL_THICKNESS, post_size_m=POST_SIZE,
        body=body, rotation_center=rot,
        diagonal_free_half_width_m=free, center_to_post_m=d_center,
        max_body_width_m=2 * free, lateral_budget=table,
        note=("機体は 100.0 x 80.0 mm の矩形とみなした保守的上界。"
              "真の凸包は矩形に内包されるので、h(theta) 表が届いたら 016-B で置き換える"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n書き出し: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
