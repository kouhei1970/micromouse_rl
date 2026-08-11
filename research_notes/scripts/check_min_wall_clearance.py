"""
research_notes/scripts/check_min_wall_clearance.py
==================================================
**最小壁余裕**（機体外形と最も近い壁との距離の最小値）を測る。

## なぜ横偏差では駄目か（2026-08-11 教授指摘）

横偏差（経路中心線からの距離）の**最大値は、ほぼ確実に「壁が無い場所」で記録される**。
交差点・分岐・開口部では側壁が存在しないので、中心線から大きく外れても何にも当たらない。
つまり横偏差の最大は「**最も安全な場所での逸脱量**」を拾っており、安全性の指標に
なっていない。実際、**壁接触なしで完走しながら横偏差 54.6 mm に達する走行が実在した**
（k=1e-4 seed2。接触水準とされる 54 mm を超えている）。

**最小壁余裕は安全余裕そのもの**であり、壁が無い場所は自動的に候補から外れる。
「接触水準 54 mm」という出所不明の定数も不要になる（0 に近づくほど危ない、で読める）。

## 測り方

**MuJoCo の `mj_geomDistance` は使えない**（mujoco 3.11 で実測、2026-08-11）。
本モデルでは毎ステップ 0.0 mm と 38 mm を交互に返す。0.0 が返るとき、
その相手の壁は実距離 125 mm 離れており、明らかに誤りである
（物理的に、壁余裕が 100 Hz で 0 と 38 mm を往復することはありえない）。
メッシュ geom だけでなく円柱でも起きるため、この API には依存しない。

代わりに**幾何計算**で求める:

1. 機体の各プリミティブ geom の表面を**代表点**で標本化する
   （箱 → 8 頂点、円柱 → 上下の縁を 8 分割、球 → 中心＋半径オフセット）
2. 毎ステップ、代表点を `geom_xpos` / `geom_xmat` で世界座標へ移す
3. 各壁（箱）について、点を壁のローカル座標へ移して
   `dist = ‖max(|p_local| − halfsize, 0)‖` を厳密に計算する
4. 全代表点 × 近傍の壁 の最小値から、球の半径オフセットを引く

壁は機体に比べて十分長いので（1 枚 180 mm 対 機体 100 mm）、機体側の頂点が
壁の面に射影されて入る。したがって頂点の標本化で最小距離を取り落とさない。

計算量を抑えるため、機体重心から半径 `PREFILTER_M` 内にある壁だけを候補にする。

**完走した走行のみ**を集計する（失敗走行は定義上壁に当たっており、余裕 0 になるため）。

使い方:
    .venv/bin/python research_notes/scripts/check_min_wall_clearance.py
    .venv/bin/python research_notes/scripts/check_min_wall_clearance.py --n-trials 3
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO  # noqa: E402

from mouse.corridor_env import CorridorEnv  # noqa: E402
from mouse.corridor_eval import DEFAULT_COURSE_DIR, _trial_seed  # noqa: E402

MODELS = [
    ("k=0",       0, "models/exp_006_control_k0.zip"),
    ("k=0",       1, "models/exp_006c_seed1.zip"),
    ("k=0",       2, "models/exp_006c_seed2.zip"),
    ("k=1e-4",    1, "models/exp_006c_k1e-4_seed1.zip"),
    ("k=1e-4",    2, "models/exp_006c_k1e-4_seed2.zip"),
    ("k=1e-3",    1, "models/exp_006c_k1e-3_seed1.zip"),
    ("案3 k=2e-3", 1, "models/exp_006d_hp_k2e-3_seed1.zip"),
    ("案3 k=2e-3", 2, "models/exp_006d_hp_k2e-3_seed2.zip"),
    ("案3 k=2e-3", 3, "models/exp_006d_hp_k2e-3_seed3.zip"),
    ("案3 k=5e-3", 1, "models/exp_006d_hp_k5e-3_seed1.zip"),
    ("案3 k=5e-3", 2, "models/exp_006d_hp_k5e-3_seed2.zip"),
]

PREFILTER_M = 0.30    # この距離より遠い壁は最小値を与えない（機体の外接半径 << 0.3 m）
DISTMAX = 0.30        # mj_geomDistance の打ち切り


def classify_geoms(model):
    """機体側 geom と 壁・柱側 geom の id 配列を返す。

    壁・柱は生成 XML で `h_wall_*` / `v_wall_*` / `post_*` と名前がついている
    （mouse/corridor_gen.py が付与）。機体側はそれ以外の可動ボディ配下の geom。

    **メッシュ geom は除外する**（2026-08-11 実測）。mujoco 3.11 の
    `mj_geomDistance` は本モデルのメッシュ geom（外殻の薄板 mein_body2〜5）に対し
    **実距離 125 mm の壁へ 0.0 を返す**（誤り）。除外の妥当性は幾何で確認済み:
    薄板は機体座標で ±35.5 × ±28.7 mm の位置にあり、シャーシ箱 mein_body1
    （半サイズ 50 × 30 mm）の内側に収まるので、xy の外形には寄与しない。
    残るプリミティブ（シャーシ箱・車輪・センサポッド・キャスタ）が外形を覆う。

    検算: 起点でシャーシ箱の壁余裕が 33.2 mm。通路半幅 84 mm（区画 180 − 壁厚 12 → 168、
    その半分）からシャーシ半幅 30 mm を引くと 54 mm、初期の横擾乱 ±20 mm を差し引くと
    約 34 mm となり一致する。
    """
    wall_ids, robot_ids = [], []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        bodyid = model.geom_bodyid[gid]
        if name.startswith(("h_wall", "v_wall", "post")):
            wall_ids.append(gid)
        elif bodyid != 0 and model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            robot_ids.append(gid)
    return np.array(robot_ids, dtype=int), np.array(wall_ids, dtype=int)


def build_surface_samples(model, robot_ids, n_ring: int = 8):
    """機体 geom の表面を代表点で標本化する（geom ローカル座標）。

    Returns:
        list of (gid, local_pts (K,3), radius_offset)
    """
    out = []
    for gid in robot_ids:
        t, s = model.geom_type[gid], model.geom_size[gid]
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            sx, sy, sz = s
            pts = np.array([[ex * sx, ey * sy, ez * sz]
                            for ex in (-1, 1) for ey in (-1, 1) for ez in (-1, 1)])
            out.append((int(gid), pts, 0.0))
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, h = s[0], s[1]
            th = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
            ring = np.stack([r * np.cos(th), r * np.sin(th), np.zeros_like(th)], axis=1)
            pts = np.concatenate([ring + [0, 0, h], ring - [0, 0, h]], axis=0)
            out.append((int(gid), pts, 0.0))
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            out.append((int(gid), np.zeros((1, 3)), float(s[0])))
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            r, h = s[0], s[1]
            out.append((int(gid), np.array([[0, 0, h], [0, 0, -h]]), float(r)))
        # メッシュは除外（上記 classify_geoms の説明を参照）
    return out


def min_clearance_now(model, data, samples, wall_ids, wall_c, wall_R, wall_s):
    """現在の姿勢での「機体外形と最も近い壁との距離」[m]（幾何計算）。"""
    com = data.xpos[model.geom_bodyid[samples[0][0]]]
    sel = np.linalg.norm(wall_c - com, axis=1) < PREFILTER_M
    if not sel.any():
        return float("inf")
    c, R, hs = wall_c[sel], wall_R[sel], wall_s[sel]      # (W,3), (W,3,3), (W,3)

    best = np.inf
    for gid, local, r_off in samples:
        # 代表点を世界座標へ
        xm = data.geom_xmat[gid].reshape(3, 3)
        pw = data.geom_xpos[gid] + local @ xm.T           # (K,3)
        # 各壁のローカル座標へ: p_local = R^T (p − c)
        rel = pw[:, None, :] - c[None, :, :]              # (K,W,3)
        loc = np.einsum("wij,kwi->kwj", R, rel)           # (K,W,3)
        gap = np.maximum(np.abs(loc) - hs[None, :, :], 0.0)
        d = np.linalg.norm(gap, axis=2) - r_off           # (K,W)
        best = min(best, float(d.min()))
    return best


def rollout_min_clearance(model_ppo, env):
    """1 走行の最小壁余裕と完走可否を返す。"""
    m, d = env.sim.model, env.sim.data
    robot_ids, wall_ids = classify_geoms(m)
    samples = build_surface_samples(m, robot_ids)
    # 壁は world 固定なので姿勢はエピソード中で不変
    wall_c = d.geom_xpos[wall_ids].copy()
    wall_R = d.geom_xmat[wall_ids].reshape(-1, 3, 3).copy()
    wall_s = m.geom_size[wall_ids].copy()

    obs = env._make_observation()
    worst = float("inf")
    info = {}
    for _ in range(env._max_steps + 1):
        a, _ = model_ppo.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(a)
        worst = min(worst, min_clearance_now(m, d, samples, wall_ids,
                                             wall_c, wall_R, wall_s))
        if term or trunc:
            break
    ok = bool(info.get("goal", False) and not info.get("collision", False))
    return worst, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=2)
    ap.add_argument("--out", type=str, default="outputs/min_wall_clearance.json")
    args = ap.parse_args()

    course_seeds = sorted(int(np.load(p)["seed"])
                          for p in Path(DEFAULT_COURSE_DIR).glob("corridor_*.npz"))

    rows = []
    for label, seed, path in MODELS:
        mp = REPO_ROOT / path
        if not mp.exists():
            print(f"[skip] {path} が無い")
            continue
        ppo = PPO.load(str(mp), device="cpu")
        vals = []
        for cs in course_seeds:
            env = CorridorEnv(course_dir=DEFAULT_COURSE_DIR, course_seeds=[cs],
                              max_cache=2, gamma=0.995, obs_dist_diff=True)
            for t in range(args.n_trials):
                env.reset(seed=_trial_seed(0, cs, t))
                worst, ok = rollout_min_clearance(ppo, env)
                if ok:                       # **完走走行のみ**
                    vals.append(worst)
            env.close()
        if not vals:
            print(f"[skip] {label} seed={seed}: 完走走行が 0 本")
            continue
        v = np.array(vals)
        rows.append(dict(label=label, seed=seed, n=len(v),
                         min_mm=float(v.min()) * 1000,
                         p5_mm=float(np.percentile(v, 5)) * 1000,
                         mean_mm=float(v.mean()) * 1000))
        print(f"[done] {label} seed={seed}: 完走 {len(v)} 本, "
              f"最小壁余裕 最小 {v.min() * 1000:.1f} mm / 平均 {v.mean() * 1000:.1f} mm",
              flush=True)

    rows.sort(key=lambda r: r["min_mm"])
    print("\n" + "=" * 84)
    print(f"最小壁余裕（gate 帯 20 コース ×{args.n_trials} 試行、**完走走行のみ**）")
    print("  0 に近いほど危ない。機体外形と壁の距離なので、壁が無い場所は候補に入らない。")
    print("=" * 84)
    print(f"{'条件':<14}{'seed':>5}{'完走n':>7}{'最小[mm]':>12}{'下位5%[mm]':>14}{'平均[mm]':>12}")
    for r in rows:
        print(f"{r['label']:<14}{r['seed']:>5}{r['n']:>7}{r['min_mm']:>12.1f}"
              f"{r['p5_mm']:>14.1f}{r['mean_mm']:>12.1f}")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(n_trials_per_course=args.n_trials, prefilter_m=PREFILTER_M,
                       rows=rows), f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
