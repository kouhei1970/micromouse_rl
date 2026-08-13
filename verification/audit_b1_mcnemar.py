"""発注 B1 — exp_005 §4-1 の gate 表の対応二値比較（McNemar 検定）。

指標は **`no_contact_complete`（壁接触なし完走）**。§4-1 の主眼は
「exp_005 の最終モデル 0.93」対「exp_003b の 60 万ステップ選択 0.92」である。

**対応（ペア）が成立することを先に検定する**（`mouse/corridor_eval.py`）:
  試行 seed = `_trial_seed(base_seed, course_seed, trial_index)`
            = base_seed·10⁶ + course_seed·100 + trial_index
3 つの実行はいずれも **base_seed=0・同じ `course_dir`・`deterministic=True`** なので、
**(course_seed, trial_index) が同じ試行は初期条件が同一**である。
→ **方策だけが違う対応データ**であり、McNemar の前提を満たす。

scipy が無いので**二項分布の厳密計算を自前で行う**（正規近似・連続修正は使わない）。
"""
import json
import math
import sys
from itertools import combinations

REPO_ROOT = "/Users/kouhei/tmp/github/micromouse_rl"

RUNS = {
    "exp_005 最終": "exp_005_collision_penalty",
    "exp_003b 60万(選択)": "exp_003b_single_env_ckpt600k",
    "exp_003b 最終": "exp_003b_single_env",
}
METRIC = "no_contact_complete"


def load(name):
    """(course_seed, trial_index) -> bool の辞書と、実行のメタ情報を返す。"""
    d = json.load(open(f"{REPO_ROOT}/outputs/{name}/latest/metrics.json"))
    out = {}
    for pc in d["per_course"]:
        for k, t in enumerate(pc["trials"]):
            out[(pc["course_seed"], k)] = bool(t[METRIC])
    meta = {k: d[k] for k in ("deterministic", "seed", "course_dir",
                              "n_courses", "n_trials_per_course",
                              "no_contact_completion_rate", "timestamp")}
    return out, meta


def binom_cdf(k, n, p=0.5):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def mcnemar_exact(b, c):
    """McNemar の厳密検定（両側）。b, c は不一致対の数。"""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * binom_cdf(min(b, c), n))


def _log_pmf(k, m, p):
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == m else -math.inf
    return (math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1)
            + k * math.log(p) + (m - k) * math.log1p(-p))


def crit_k(m, alpha=0.05):
    """不一致対 m 個のときの棄却域の境界 k_crit（`k <= k_crit` または `k >= m-k_crit`）。

    帰無仮説 p=0.5 の下で両側 alpha を満たす最大の k を返す（無ければ -1）。
    """
    kc, cdf = -1, 0.0
    for k in range(0, m // 2 + 1):
        cdf += math.exp(_log_pmf(k, m, 0.5))
        if 2.0 * cdf <= alpha:
            kc = k
        else:
            break
    return kc


def power_exact(m, p_true, alpha=0.05):
    """真の偏りが p_true のときの、McNemar 厳密検定（両側）の検出力。O(m)。"""
    kc = crit_k(m, alpha)
    if kc < 0:
        return 0.0
    lo = sum(math.exp(_log_pmf(k, m, p_true)) for k in range(0, kc + 1))
    hi = sum(math.exp(_log_pmf(k, m, p_true)) for k in range(m - kc, m + 1))
    return lo + hi


def min_detectable(n_disc, alpha=0.05):
    """不一致対が n_disc 個のとき、両側 alpha で有意になる最小の偏り（b:c の形）。"""
    for b in range(n_disc // 2, -1, -1):
        if mcnemar_exact(b, n_disc - b) <= alpha:
            return b, n_disc - b
    return None


def main():
    data, metas = {}, {}
    for label, name in RUNS.items():
        data[label], metas[label] = load(name)

    print("=" * 78)
    print("0. 対応（ペア）の前提の検定")
    print("=" * 78)
    keys = [set(v.keys()) for v in data.values()]
    same_keys = all(k == keys[0] for k in keys)
    print(f"  試行の識別子 (course_seed, trial_index) の集合が 3 実行で同一: {same_keys}"
          f"  (n = {len(keys[0])})")
    for label, m in metas.items():
        print(f"  {label:<22} deterministic={m['deterministic']}  base_seed={m['seed']}  "
              f"course_dir={m['course_dir']}  完走率={m['no_contact_completion_rate']}")
    ok = same_keys and len({(m["seed"], m["course_dir"], m["deterministic"])
                            for m in metas.values()}) == 1
    print(f"  → 対応の前提: {'成立（方策だけが違う）' if ok else '🔴 不成立'}")

    print()
    print("=" * 78)
    print("1. McNemar 検定（両側・二項の厳密計算）")
    print("=" * 78)
    results = []
    for a, b_ in combinations(RUNS.keys(), 2):
        A, B = data[a], data[b_]
        n11 = sum(1 for k in A if A[k] and B[k])
        n10 = sum(1 for k in A if A[k] and not B[k])
        n01 = sum(1 for k in A if not A[k] and B[k])
        n00 = sum(1 for k in A if not A[k] and not B[k])
        p = mcnemar_exact(n10, n01)
        results.append((a, b_, n11, n10, n01, n00, p))
        print(f"\n  【{a}】 対 【{b_}】")
        print(f"    両方成功 {n11:>3} / A のみ成功 {n10:>3} / B のみ成功 {n01:>3} / 両方失敗 {n00:>3}")
        print(f"    率: {(n11+n10)/len(A):.2f} 対 {(n11+n01)/len(A):.2f}"
              f"   不一致対 {n10+n01} 個")
        print(f"    McNemar 厳密検定（両側）: p = {p:.4f}"
              f"  → {'有意（α=0.05）' if p <= 0.05 else '有意でない'}")

    print()
    print("=" * 78)
    print("2. 検出力 — この n で何が言えるか")
    print("=" * 78)
    for a, b_, n11, n10, n01, n00, p in results:
        nd = n10 + n01
        md = min_detectable(nd)
        if md is None:
            print(f"  {a} 対 {b_}: 不一致対 {nd} 個 → "
                  f"**どんな偏りでも α=0.05 では有意にならない**（原理的に判定不能）")
        else:
            print(f"  {a} 対 {b_}: 不一致対 {nd} 個 → "
                  f"有意になるのは {md[1]}:{md[0]} 以上に偏ったときだけ"
                  f"（実測 {max(n10,n01)}:{min(n10,n01)}）")
    # 不一致対が 0〜8 個のときに必要な偏り
    print("\n  参考: 不一致対の数ごとに、α=0.05 で有意になる最小の偏り")
    for nd in range(1, 13):
        md = min_detectable(nd)
        print(f"    不一致対 {nd:>2} 個: " +
              (f"{md[1]}:{md[0]}（p={mcnemar_exact(md[0], md[1]):.4f}）" if md
               else "**有意にできない**"))

    print()
    print("=" * 78)
    print("3. 必要な試行数 — 観測された効果量を検出するには")
    print("=" * 78)
    req = []
    for a, b_, n11, n10, n01, n00, p in results:
        nd = n10 + n01
        if nd == 0:
            continue
        p_hat = max(n10, n01) / nd                 # 不一致対のうち優位側の割合
        rate = nd / len(data[a])                   # 不一致対の出現率
        need = next((m for m in range(2, 4001) if power_exact(m, p_hat) >= 0.80), None)
        n_trials = math.ceil(need / rate) if need else None
        req.append({"a": a, "b": b_, "p_hat": p_hat, "discordance_rate": rate,
                    "need_discordant": need, "need_trials": n_trials})
        print(f"  {a} 対 {b_}:")
        print(f"    観測: 不一致対の偏り {p_hat:.3f}（{max(n10,n01)}:{min(n10,n01)}）・"
              f"不一致率 {rate:.2f}")
        print(f"    検出力 80% に要る不一致対 {need} 個 → **試行数 約 {n_trials} 対**"
              f"（実測は {len(data[a])} 対）")

    print()
    print("=" * 78)
    print("4. クラスタ構造 — 100 試行は独立ではない（20 コース × 5 試行）")
    print("=" * 78)
    cluster = []
    for a, b_, n11, n10, n01, n00, p in results:
        A, B = data[a], data[b_]
        courses = sorted({c for c, _ in A})
        disc_by_course = {c: sum(1 for k in range(5) if A[(c, k)] != B[(c, k)])
                          for c in courses}
        n_c_with = sum(1 for c in courses if disc_by_course[c] > 0)
        # コース水準の対応比較（各コースの成功数 0〜5 の差の符号検定）
        diffs = [sum(A[(c, k)] for k in range(5)) - sum(B[(c, k)] for k in range(5))
                 for c in courses]
        pos = sum(1 for d in diffs if d > 0)
        neg = sum(1 for d in diffs if d < 0)
        p_sign = mcnemar_exact(pos, neg)
        cluster.append({"a": a, "b": b_, "courses_with_discordance": n_c_with,
                        "n_courses": len(courses), "course_pos": pos,
                        "course_neg": neg, "p_sign_course": p_sign})
        print(f"\n  【{a}】 対 【{b_}】")
        print(f"    不一致対が出たコース: {n_c_with} / {len(courses)}  "
              f"（不一致対 {n10+n01} 個がこの数のコースに集中）")
        print(f"    コース水準の符号検定: A が良い {pos} コース / B が良い {neg} コース / "
              f"同点 {len(courses)-pos-neg} → p = {p_sign:.4f}")

    print("\n  → 試行は**コース内で相関する**（同じコースで繰り返し失敗する）。")
    print("     試行水準の検定は独立を仮定するので **p 値を小さい側へ外す**。")
    print("     本件はいずれも有意でないので、**クラスタを考慮すると結論はより強くなる**。")

    out = {"metric": METRIC, "paired_premise_ok": ok, "power": req,
           "cluster": cluster,
           "n_pairs": len(keys[0]),
           "comparisons": [{"a": a, "b": b_, "n11": n11, "n10": n10, "n01": n01,
                            "n00": n00, "p_exact_two_sided": p}
                           for a, b_, n11, n10, n01, n00, p in results]}
    path = f"{REPO_ROOT}/verification/out/b1_mcnemar.json"
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"\n書き出し: {path}")


if __name__ == "__main__":
    main()
