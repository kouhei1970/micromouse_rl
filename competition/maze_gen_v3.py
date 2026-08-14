"""規定準拠 評価迷路生成器 v3（多軸受理・層別サンプリング）
Rule-conforming evaluation maze generator v3 with multi-axis stratified acceptance.

2026-08-11、教授裁定により新設。**v2（`competition/maze_gen_v2.py`）は変更しない**
（v2 で凍結した帯を再現できなくなるため。v2 は凍結記録として残す）。

■ なぜ v3 が要るのか

v2 の是正は**経路長**（$D_0$）を大会実迷路に合わせることに成功したが、
**探索の遠回り**という軸では**是正前より実態から遠ざかった**:

    経路比 R = 初回探索の移動区画数 / D_0
      是正前 eval 1.205 → 是正後 eval **1.013** （大会実迷路 1.446）

是正後の帯は **20 面中 10 面で初回探索がそのまま真の最短経路**であり、
**探索の質を測る土俵になっていない**（`research_notes/note_010_*.md`）。

機構は分かっている。v2 の経路保護型除去は「最短距離を 1 区画も縮めない壁だけを
開ける」ので、**誘導される閉路が最短経路の区間を丸ごと含む弦を必ず棄却する**。
結果として閉路が最短経路から系統的に遠ざかり、最短経路が一本道の廊下になる。
**経路長を守る規則が、そのまま探索の曖昧さを消す規則になっていた。**

■ v3 の方針 — 生成手順は変えない。受理条件を足す

実測により、**現行の生成器は高 $R$ の迷路を作れないのではなく、稀にしか作らない**
ことが分かった（候補 400 面で $R$ 中央値 1.046・最大 3.075）。したがって
**生成手順の改造は不要**で、候補を多めに作って選ぶだけでよい。

  1. **候補の生成**: v2 と同一手順（DFS 全域木 → ゴールリング処理 → 経路保護型除去
     → 孤立柱修復 → 規定 6 項目）。**ただし $D_0$ の窓は最終 $D$ に掛ける**（下記）
  2. **層別サンプリング**: 大会実迷路の $R$ の四分位で 4 帯に切り、**各帯から同数**を
     採る。中央値だけでなく**散らばり**も大会に合わせるため

■ v2 から直した欠陥: 窓の掛け先

v2 は窓を**除去前**の $D_0$ に掛けていた。経路保護があるうちは $D$ が縮まないので
実害が出ないが、**保護を外した瞬間に「窓に入れたのに最終的に短い迷路」が出る**
（実測: 保護なし・窓 [45,110] で最終 $D$ 中央値 24）。v3 では**最終 $D$ に掛ける**。

■ 選抜は「敵対的選抜」ではない（教授裁定 2026-08-11）

$R$ を**最大化**すると「足立法が特に苦手な迷路」を選ぶことになり、古典ベースライン
L0-a/b/c を系統的に不利にする。評価帯は RL 方策と古典の**両方**を測る土俵なので、
それは公正さを損なう。**したがって最大化ではなく、大会実迷路の分布に一致させる。**

  **位置づけ: 「実物がそうなっているから合わせる」であって
  「古典が苦手なものを選ぶ」ではない。**

敵対的迷路（`docs/ADVERSARIAL_MAZE_DESIGN.md` の `adv_eval`）は**別層**であり、
目的が違う（方策の弱点を突く）。混同しないこと。

■ 再現性（研究計画書 §9-2）

選抜を入れると帯は「無作為抽出」ではなく**選ばれた部分集合**になる。したがって:

  - **候補プールを seed 範囲で定義**する（`--seed-from` / `--seed-to`）
  - 選抜規則を決定的に書く（同じ seed 範囲＋同じ規則 → 常に同じ 20 面）
  - **候補全数の $R$ と採否**を manifest に残す（なぜこの面が選ばれたかを後から追える）

使い方:
    .venv/bin/python -m competition.maze_gen_v3 --out-dir competition/mazes/eval_v3 \\
        --seed-from 1000 --seed-to 2999 --n-per-band 5
"""
import argparse
import json
import os
import random
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from competition import maze_gen_v2 as G  # noqa: E402
from competition.explore_cost import (  # noqa: E402
    detour_ratio, n_delta, true_shortest)
from competition.macro_routes import n_macro_routes  # noqa: E402

W = H = 16
GOAL_CELLS = ((7, 7), (7, 8), (8, 7), (8, 8))

# 真の最短距離の受理窓（v2 から継承。**最終 D に掛ける**のが v3 の変更点）
D_WINDOW = (45, 110)
# 追加で開ける内部壁の目標数（v2 と同じ。経路保護つき）
EXTRA_OPEN_TARGET = 15

# 経路比 R の層の境界。**大会実迷路（窓 [45,110] 内の 33 面）の四分位**から取る。
# 実測: 最小 1.058 / Q1 1.339 / 中央 1.625 / Q3 1.880 / 最大 3.267
# ハードコードを避けるため `--recompute-bands` で参照迷路から再計算できるようにし、
# 既定値は再計算結果をそのまま置いている（生成のたびに参照迷路を読まないため）。
R_BAND_EDGES = (1.058, 1.339, 1.625, 1.880, 3.267)

# 候補経路の本数 K の定義に使うパラメータ。**受理条件に使うので凍結する**
# （後から変えられる自由度を残すと、目標を満たすように調整できてしまう。教授指示）。
# 大会実迷路 窓内 33 面で中央値 2.0・82% が 2 本以上になる設定。掃引で広い範囲が
# 中央値 2 になることを確認済み（competition/macro_routes.py の docstring）。
MACRO_DELTA, MACRO_THETA = 4, 3

# K の階級ごとの目標面数（20 面）。大会実迷路 窓内 33 面の実測比率
# K=1 19% / K=2 41% / K>=3 41% を 20 面に割り当てたもの。
# **最大化ではなく分布一致**（R について教授が示した方針と同じ）。
K_TARGETS = {1: 4, 2: 8, 3: 8}
# ⚠️ **K_TARGETS は絶対数であり、合計 20 が帯の面数の上限になる**
# （大会の実測比率 19% / 41% / 41% を **20 面**へ割り当てたもの）。
# **より大きな帯が要る場合は `--k-scale` で比率を保ったまま倍にする**
# （**契約は「大会の実測比率の維持」であって絶対数ではない**。裁定 R43-2・2026-08-14）。
# **K_TARGETS そのものは変更しない** — 変えると凍結帯 eval_v4 を再生成できなくなる。

CONTEST_DIR = os.path.join(_REPO_ROOT, "competition", "reference_mazes", "contest")


# ==========================================================================
# 候補の生成（v2 と同一手順。窓の掛け先だけが違う）
# ==========================================================================
def generate_candidate(seed, extra_open_target=EXTRA_OPEN_TARGET, d_window=D_WINDOW,
                       with_k=True):
    """1 つの seed から候補を 1 面作る。受理条件を満たさなければ None。

    v2 の `generate_maze` と**同じ手順**だが、
      - seed 内での再試行をしない（1 seed = 1 候補。候補プールは seed 範囲で定義する）
      - $D$ の窓を**最終形**に掛ける
    """
    rng = random.Random(seed)
    v = np.ones((W + 1, H), dtype=int)
    h = np.ones((W, H + 1), dtype=int)

    G._spanning_tree(rng, v, h)
    for e in G.GOAL_INNER:
        G._set(v, h, e, 0)
    for e in G.RING_EDGES:
        G._set(v, h, e, 1)
    gateway = rng.choice(G.RING_EDGES)
    G._set(v, h, gateway, 0)
    for e in G.FORCED_OPEN:
        G._set(v, h, e, 0)
    v[0, 0] = 1; h[0, 0] = 1; v[1, 0] = 1; h[0, 1] = 0

    protected_open = set(G.FORCED_OPEN) | {gateway} | set(G.GOAL_INNER) | {("h", 0, 1)}
    protected_wall = (set(G.RING_EDGES) - {gateway}) | {("v", 1, 0)}

    d0 = G.shortest_distance_to_goal(v, h)
    if d0 < 0:
        return None

    # 経路保護型の内壁除去（v2 と同一）
    internal = ([("v", x, y) for x in range(1, W) for y in range(H)]
                + [("h", x, y) for x in range(W) for y in range(1, H)])
    rng.shuffle(internal)
    opened = 0
    for e in internal:
        if opened >= extra_open_target:
            break
        if e in protected_wall or G._get(v, h, e) == 0:
            continue
        G._set(v, h, e, 0)
        k, x, y = e
        posts = ((x, y), (x, y + 1)) if k == "v" else ((x, y), (x + 1, y))
        if any(p != G.CENTER_POST and not any(G._get(v, h, pe) == 1 for pe in G.post_walls(*p))
               for p in posts):
            G._set(v, h, e, 1)          # 広場禁止規定で取り消し
            continue
        if G.shortest_distance_to_goal(v, h) < d0:
            G._set(v, h, e, 1)          # 経路保護で取り消し
        else:
            opened += 1

    # 孤立格子点の修復
    for (px, py) in G.isolated_posts(v, h):
        cands = [e for e in G.post_walls(px, py) if e not in protected_open]
        if not cands:
            return None
        G._set(v, h, rng.choice(cands), 1)

    # 規定 6 項目（v2 と同一）
    if sum(1 for e in G.RING_EDGES if G._get(v, h, e) == 0) != 1:
        return None
    if not G.all_cells_reachable(v, h):
        return None
    if G.isolated_posts(v, h):
        return None
    if any(G._get(v, h, e) == 1 for e in G.post_walls(*G.CENTER_POST)):
        return None
    if G.wall_follow_reaches_goal(v, h, "left") or G.wall_follow_reaches_goal(v, h, "right"):
        return None

    # **v3 の変更点**: 窓は最終形の D に掛ける
    d_fin = true_shortest(v, h)
    if d_window is not None and not (d_window[0] <= d_fin <= d_window[1]):
        return None

    cycles, open_edges = G.independent_cycles(v, h)
    info = dict(seed=int(seed), d_true=int(d_fin), d0_tree=int(d0),
                cycles=int(cycles), open_edges=int(open_edges),
                gateway=list(gateway), extra_opened=int(opened),
                R=float(detour_ratio(v, h)),
                R_compass=float(detour_ratio(v, h, tiebreak="compass")))
    # K（候補経路の本数）は経路長ぶんの BFS が要る高価な量なので、
    # 必要なときだけ計算する（選抜側が R の層に空きを見てから求める）
    if with_k:
        info["K"] = int(n_macro_routes(v, h, delta=MACRO_DELTA, theta=MACRO_THETA))
    return v, h, info


# ==========================================================================
# 層別サンプリング
# ==========================================================================
def band_of(r, edges=R_BAND_EDGES):
    """経路比 r がどの層に入るか（0..len(edges)-2）。範囲外は None。"""
    if not np.isfinite(r):
        return None
    if r < edges[0] or r > edges[-1]:
        return None
    for i in range(len(edges) - 1):
        if r < edges[i + 1] or (i == len(edges) - 2 and r <= edges[-1]):
            return i
    return None


def contest_band_edges(contest_dir=CONTEST_DIR, d_window=D_WINDOW):
    """参照迷路（大会実迷路）から層の境界を測り直す。

    ハードコードの検算用。既定値 R_BAND_EDGES はこの関数の出力そのもの。
    """
    import glob
    rs = []
    for f in sorted(glob.glob(os.path.join(contest_dir, "contest_*.npz"))):
        z = np.load(f)
        v, h = z["v_walls"], z["h_walls"]
        start = (int(z["start_x"]), int(z["start_y"]))
        goals = tuple((int(a), int(b)) for a, b in zip(z["goals_x"], z["goals_y"]))
        d = true_shortest(v, h, start, goals)
        if not (d_window[0] <= d <= d_window[1]):
            continue
        r = detour_ratio(v, h, start, goals)
        if np.isfinite(r):
            rs.append(r)
    rs = np.array(sorted(rs))
    return tuple(float(np.percentile(rs, q)) for q in (0, 25, 50, 75, 100)), len(rs)


def k_class(k):
    """候補経路の本数 K を 3 階級に丸める（n=20 では細かく切りすぎないため）。"""
    return 1 if k <= 1 else (2 if k == 2 else 3)


def select_stratified(seed_from, seed_to, n_per_band, edges=R_BAND_EDGES,
                      extra_open_target=EXTRA_OPEN_TARGET, d_window=D_WINDOW,
                      progress_every=200, k_targets=None):
    """seed を昇順に走査し、各層が n_per_band に達するまで採る（決定的）。

    `k_targets` を与えると、**経路比 $R$ の層別に加えて、候補経路の本数 $K$ の
    階級ごとの面数も目標に合わせる**（二重の層別）。

      k_targets: {1: n1, 2: n2, 3: n3}（3 は「3 本以上」）。
        大会実迷路 窓内 33 面の実測比率 K=1 19% / K=2 41% / K>=3 41% を
        20 面に割り当てると {1: 4, 2: 8, 3: 8}。

    **$K$ を最大化しないのが要点。**最大化は分布一致にならない（$R$ について
    教授が指摘したのと同じ問題）。実物がそうなっているから合わせる。

    2 巡する:
      1 巡目 … $R$ の層と $K$ の階級の**両方**に空きがある候補だけを採る
      2 巡目 … 1 巡目で埋まらなかった $R$ の層を、$K$ の制約を外して埋める
    2 巡目が必要になった場合は、その旨を log に残す（黙って緩めない）。

    返り値: (選ばれた面のリスト, 候補全数の記録)
    **同じ seed 範囲・同じ規則なら常に同じ結果**になる（seed 昇順・先着順）。
    """
    n_bands = len(edges) - 1
    chosen = [[] for _ in range(n_bands)]
    k_count = {1: 0, 2: 0, 3: 0}
    log = []
    cache = {}

    def scan(pass_no, use_k):
        for seed in range(seed_from, seed_to + 1):
            if progress_every and (seed - seed_from) % progress_every == 0:
                done = sum(len(c) for c in chosen)
                print(f"  [maze_gen_v3] {pass_no} 巡目 seed={seed} 採用 "
                      f"{done}/{n_bands * n_per_band}（層ごと {[len(c) for c in chosen]}"
                      f"／K 階級 {dict(k_count)}）", flush=True)
            if all(len(c) >= n_per_band for c in chosen):
                return
            if seed in cache:
                out = cache[seed]
            else:
                out = generate_candidate(seed, extra_open_target, d_window,
                                         with_k=False)
                cache[seed] = out
            if out is None:
                if pass_no == 1:
                    log.append(dict(seed=int(seed), accepted=False, reason="規定/窓で不合格"))
                continue
            v, h, info = out
            b = band_of(info["R"], edges)
            if any(x[2]["seed"] == seed for c in chosen for x in c):
                continue                      # 1 巡目で採用済み
            rec = dict(seed=int(seed), accepted=False, R=info["R"],
                       d_true=info["d_true"], band=b, pass_no=pass_no, reason=None)
            if b is None:
                rec["reason"] = "R が層の範囲外"
            elif len(chosen[b]) >= n_per_band:
                rec["reason"] = f"層 {b + 1} は充足済み"
            else:
                # ここまで来た候補だけ K を計算する（高価なので遅延）
                if "K" not in info:
                    info["K"] = int(n_macro_routes(v, h, delta=MACRO_DELTA,
                                                   theta=MACRO_THETA))
                kc = k_class(info["K"])
                rec["K"] = info["K"]
                rec["k_class"] = kc
                if use_k and k_targets and k_count[kc] >= k_targets.get(kc, 0):
                    rec["reason"] = f"K 階級 {kc} は充足済み"
                    if pass_no == 1 or rec["accepted"]:
                        log.append(rec)
                    continue
                chosen[b].append((v, h, info))
                k_count[kc] += 1
                rec["accepted"] = True
                rec["reason"] = (f"層 {b + 1} / K 階級 {kc} に採用"
                                 + ("（2 巡目・K の制約を外して充足）" if pass_no == 2 else ""))
            if pass_no == 1 or rec["accepted"]:
                log.append(rec)

    scan(1, use_k=bool(k_targets))
    if k_targets and not all(len(c) >= n_per_band for c in chosen):
        print("  [maze_gen_v3] 1 巡目で層が埋まらなかったため、K の制約を外して 2 巡目",
              flush=True)
        scan(2, use_k=False)
    return chosen, log


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="規定準拠 評価迷路の生成（v3・層別選抜）")
    ap.add_argument("--out-dir", default="competition/mazes/eval_v3")
    ap.add_argument("--seed-from", type=int, default=1000)
    ap.add_argument("--seed-to", type=int, default=2999)
    ap.add_argument("--n-per-band", type=int, default=5)
    ap.add_argument("--extra-open", type=int, default=EXTRA_OPEN_TARGET)
    ap.add_argument("--recompute-bands", action="store_true",
                    help="層の境界を参照迷路から測り直して使う（既定値の検算）")
    ap.add_argument("--dry-run", action="store_true", help="npz/XML を書かずに集計だけ")
    ap.add_argument("--k-scale", type=int, default=1,
                    help="K 階級の目標面数の倍率（**比率は保つ**。既定 1 = 従来と同一。裁定 R43-2）")
    ap.add_argument("--no-k-strat", action="store_true",
                    help="候補経路の本数 K の層別を行わない（R だけで層別）")
    args = ap.parse_args()

    edges = R_BAND_EDGES
    if args.recompute_bands:
        edges, n_ref = contest_band_edges()
        print(f"[maze_gen_v3] 層の境界を参照迷路 {n_ref} 面から再計算: "
              + " / ".join(f"{e:.3f}" for e in edges))
        if max(abs(a - b) for a, b in zip(edges, R_BAND_EDGES)) > 1e-3:
            print("  ** 既定値 R_BAND_EDGES と食い違います。既定値の更新が要ります **")

    print(f"[maze_gen_v3] seed {args.seed_from}〜{args.seed_to}、"
          f"層ごと {args.n_per_band} 面、除去目標 {args.extra_open}")
    # **既定（--k-scale 1）では K_TARGETS と等しい辞書**になるので、従来と挙動は変わらない
    k_targets = (None if args.no_k_strat
                 else {k: v * args.k_scale for k, v in K_TARGETS.items()})
    if k_targets:
        print(f"  K の階級ごとの目標面数: {k_targets}"
              f"（Δ={MACRO_DELTA}, θ={MACRO_THETA}）")
    chosen, log = select_stratified(args.seed_from, args.seed_to, args.n_per_band,
                                    edges, args.extra_open, k_targets=k_targets)

    n_bands = len(edges) - 1
    print(f"\n{'層':<4}{'R の範囲':>18}{'採用':>6}{'候補':>8}{'採用率':>9}")
    for i in range(n_bands):
        cand = sum(1 for r in log if r.get("band") == i)
        print(f"{i + 1:<4}[{edges[i]:.3f}, {edges[i + 1]:.3f})".rjust(22)
              + f"{len(chosen[i]):>6}{cand:>8}{(len(chosen[i]) / cand * 100 if cand else 0):>8.1f}%")
    total = sum(len(c) for c in chosen)
    n_cand = sum(1 for r in log if r.get("R") is not None)
    print(f"\n採用 {total} 面／規定通過の候補 {n_cand} 面／消費 seed {len(log)}")
    if total < n_bands * args.n_per_band:
        print("** 面数が足りません。seed 範囲を広げてください **")
        return 1

    flat = [x for band in chosen for x in band]
    R = [x[2]["R"] for x in flat]
    D = [x[2]["d_true"] for x in flat]
    B = [x[2]["cycles"] for x in flat]
    print(f"  経路比 R: 中央値 {np.median(R):.3f}（四分位 {np.percentile(R, 25):.3f}"
          f"〜{np.percentile(R, 75):.3f}、範囲 {min(R):.3f}〜{max(R):.3f}）")
    print(f"  D_true : 中央値 {np.median(D):.0f}（範囲 {min(D)}〜{max(D)}）")
    print(f"  β      : 中央値 {np.median(B):.0f}（範囲 {min(B)}〜{max(B)}）")
    K = [x[2]["K"] for x in flat]
    print(f"  候補経路 K: 中央値 {np.median(K):.1f}（範囲 {min(K)}〜{max(K)}）"
          f"／K>=2 {np.mean(np.array(K) >= 2) * 100:.0f}%"
          f"／K>=3 {np.mean(np.array(K) >= 3) * 100:.0f}%"
          f"／階級ごと {{1: {sum(1 for k in K if k <= 1)}, 2: {sum(1 for k in K if k == 2)}, "
          f"3: {sum(1 for k in K if k >= 3)}}}")
    print("  （目標: 大会実迷路 窓内 33 面 → 中央値 2.0／K>=2 82%／K>=3 39%）")

    if args.dry_run:
        print("\n（--dry-run のため書き出しは行いません）")
        return 0

    from mouse.mjcf import build_maze_robot_xml
    from mouse.params import RobotParams
    params = RobotParams()
    os.makedirs(args.out_dir, exist_ok=True)
    infos = []
    for v, h, info in sorted(flat, key=lambda x: x[2]["seed"]):
        s = info["seed"]
        npz = os.path.join(args.out_dir, f"maze_{s}.npz")
        np.savez(npz, v_walls=v, h_walls=h, seed=s, width=W, height=H)
        build_maze_robot_xml(v, h, npz[:-4] + ".xml", model_name=f"maze_{s}", params=params)
        infos.append(info)
        print(f"[maze_gen_v3] seed={s} R={info['R']:.3f} D={info['d_true']} "
              f"β={info['cycles']} 除去{info['extra_opened']}枚")

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(dict(
            generator="competition/maze_gen_v3.py",
            policy=("経路比 R の層別サンプリング。**最大化ではなく大会実迷路の分布への"
                    "一致**。位置づけは「実物がそうなっているから合わせる」であって"
                    "「古典が苦手なものを選ぶ」ではない（教授裁定 2026-08-11）"),
            rules=["IEEE R2SAC 4.3 (gateway=1, start 3 walls)",
                   "IEEE R2SAC 4.5 (wall-follower cannot reach goal, multiple paths)",
                   "NTF 注意9 (no walls/posts inside goal)",
                   "NTF 2-4 (every post has >=1 wall except goal center; outer walls complete)"],
            seed_from=args.seed_from, seed_to=args.seed_to,
            n_per_band=args.n_per_band, r_band_edges=list(edges),
            macro_delta=MACRO_DELTA, macro_theta=MACRO_THETA,
            k_targets=k_targets,
            d_window=list(D_WINDOW), d_window_applied_to="final",
            extra_open_target=args.extra_open,
            mazes=infos, candidate_log=log), f, indent=2, ensure_ascii=False)
    print(f"\nmanifest: {os.path.join(args.out_dir, 'manifest.json')}"
          f"（候補全数 {len(log)} 件の採否を記録）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
