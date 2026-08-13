"""機体の方位別実効半幅 h(θ) の数値表（exp_016 用。教授発注）。

**定義（操作的に書く。裁定 R44 の指示・§9-17）**:
  機体を**機体座標系**で見る（前方 = +x、左 = +y、原点 = 車体の基準点）。
  方向 $\\varphi$ の**支持関数**を
      $h(\\varphi) = \\max_{p \\in \\text{機体}} (p \\cdot (\\cos\\varphi, \\sin\\varphi))$
  と定義する（＝ その向きへの張り出しの最大値）。

  **主用途の列**「**進行方向 θ で走るときの横方向半幅**」は、進行方向 θ に**直交する**
  向きへの張り出しなので **$h(\\theta+90°)$** である。取り違えを消すため両方を併記する。

**測り方**: MuJoCo モデルを建て、`mj_forward` の後に**世界座標の geom 位置・姿勢**を読み、
機体の全 geom（迷路の壁・柱・床を除く）について形状ごとに厳密な支持を計算して最大を取る。
  - **メッシュ**: 頂点を姿勢で回して最大射影（頂点は float32 で保持されている）
  - **箱**: $c\\cdot u + \\sum_i |s_i (R^\\top u)_i|$
  - **円柱**: $c\\cdot u + r\\sqrt{u_x'^2+u_y'^2} + h|u_z'|$（$u'=R^\\top u$）
  - **球**: $c\\cdot u + r$
  - **カプセル**: 円柱と同じ軸に沿う線分 ＋ 半径

**検算**: `COMMIT_001` の 10 方向（+x 0.050000000 / +y 0.040000000 / +x+y 0.056568542 ほか）と
突き合わせる。**一致しなければ本表を出さない。**
"""
import json
import math
import sys

REPO_ROOT = "/Users/kouhei/tmp/github/micromouse_rl"
sys.path.insert(0, REPO_ROOT)

import mujoco
import numpy as np

from mouse.maze6_env import Maze6Env

MAZE_PREFIXES = ("v_wall", "h_wall", "post_", "floor")
# COMMIT_001 で開示済みの 10 方向（検算用）
REF = {(1, 0): 0.050000000, (-1, 0): 0.050000000, (0, 1): 0.040000000, (0, -1): 0.040000000,
       (1, 1): 0.056568542, (1, -1): 0.056568542, (-1, 1): 0.056568542, (-1, -1): 0.056568542}


def robot_geoms(model):
    """機体側の geom id（迷路・床を除く）。"""
    out = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not name.startswith(MAZE_PREFIXES):
            out.append(g)
    return out


def geom_support(model, data, g, u, origin):
    """geom g の、方向 u（単位ベクトル・世界系）への支持（origin を基準）。"""
    c = data.geom_xpos[g] - origin
    R = data.geom_xmat[g].reshape(3, 3)
    s = model.geom_size[g]
    t = model.geom_type[g]
    base = float(c @ u)
    up = R.T @ u                                  # geom 局所系での方向
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        adr = model.geom_dataid[g]
        v0 = model.mesh_vertadr[adr]
        nv = model.mesh_vertnum[adr]
        V = model.mesh_vert[v0:v0 + nv].reshape(-1, 3).astype(np.float64)
        return base + float(np.max(V @ up))
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return base + float(np.sum(np.abs(s[:3] * up)))
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return base + float(s[0])
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return base + float(s[0] * math.hypot(up[0], up[1]) + s[1] * abs(up[2]))
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return base + float(s[0] + s[1] * abs(up[2]))
    if t == mujoco.mjtGeom.mjGEOM_PLANE:
        return -math.inf
    raise AssertionError(f"未対応の geom 種別: {t}（黙って無視しない）")


def support(model, data, gids, phi_deg, origin, yaw):
    """機体座標系の方位 phi_deg（度）への支持関数 [m]。"""
    a = math.radians(phi_deg) + yaw            # 機体の向きぶん回す
    u = np.array([math.cos(a), math.sin(a), 0.0])
    return max(geom_support(model, data, g, u, origin) for g in gids)


def main():
    env = Maze6Env(maze_dir=REPO_ROOT, mode="generate", maze_mode="loop")
    env.reset(seed=0)
    model, data = env.sim.model, env.sim.data
    mujoco.mj_forward(model, data)
    gids = robot_geoms(model)
    x, y, yaw = env.sim.privileged_pose()
    origin = np.array([x, y, 0.0])
    print(f"機体の geom 数: {len(gids)}  基準点 ({x:.4f}, {y:.4f})  方位 {math.degrees(yaw):.2f}°")

    # --- 検算: COMMIT_001 の 10 方向 ---
    print()
    print("=" * 78)
    print("検算: COMMIT_001 の開示値との突き合わせ")
    print("=" * 78)
    ok = True
    for (dx, dy), ref in REF.items():
        phi = math.degrees(math.atan2(dy, dx))
        v = support(model, data, gids, phi, origin, yaw)
        d = abs(v - ref)
        ok &= d < 1e-8
        print(f"  {('+x' if dx>0 else '-x' if dx<0 else '')}{('+y' if dy>0 else '-y' if dy<0 else '')}"
              f"（{phi:>6.1f}°）: {v:.9f} 対 {ref:.9f}  差 {d:.1e}"
              + ("" if d < 1e-8 else "  🔴"))
    print(f"  → {'一致（本表を出してよい）' if ok else '🔴 不一致。表は出さない'}")
    if not ok:
        env.close()
        return

    # --- 本表 ---
    print()
    print("=" * 78)
    print("h(θ) の数値表（θ = 0〜90°・刻み 5°／要所は 1°）")
    print("=" * 78)
    print(f"{'θ[°]':>6} {'h(θ)[m]':>12} {'横方向半幅 h(θ+90°)[m]':>24}")
    rows = []
    grid = sorted(set(list(range(0, 91, 5)) + list(range(40, 51)) + [45]))
    for th in grid:
        h1 = support(model, data, gids, th, origin, yaw)
        h2 = support(model, data, gids, th + 90.0, origin, yaw)
        rows.append({"theta_deg": th, "h_theta_m": h1, "lateral_half_width_m": h2})
        print(f"{th:>6} {h1:>12.6f} {h2:>24.6f}")

    lat = [r["lateral_half_width_m"] for r in rows]
    k = int(np.argmax(lat))
    print()
    print(f"  横方向半幅の最大: {max(lat):.6f} m（θ = {rows[k]['theta_deg']}°）")
    print(f"  横方向半幅の最小: {min(lat):.6f} m（θ = {rows[int(np.argmin(lat))]['theta_deg']}°）")
    out = {"definition": "h(phi) = max_{p in body} p·(cos phi, sin phi)、機体座標系",
           "main_column": "lateral_half_width_m = h(theta+90deg)（進行方向 theta で走るときの横方向半幅）",
           "n_geoms": len(gids), "rows": rows}
    with open(f"{REPO_ROOT}/verification/out/support_function.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"\n書き出し: {REPO_ROOT}/verification/out/support_function.json")
    env.close()


if __name__ == "__main__":
    main()
