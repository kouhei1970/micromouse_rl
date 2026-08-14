#!/usr/bin/env python3
"""`judge_recurrent.py`（exp_023 の判定スクリプト）の単体テスト。

Unit tests for the exp_023 verdict script - synthetic data only.

🔴 **合成データだけで検査する**（カード §7 手順 5）。**実験の測定出力は一切読まない** —
**結果を見る前に判定コードを確定させる**ため（事前登録の趣旨。教授裁定 2026-08-15）。

| # | 検査 | 要点 |
|---|---|---|
| T-J1 | **同値（ちょうど閾値）の扱いが投入前の確定どおり** | R1 は当たり・R3 は外れ・R4 は当たり・R5 は外れ・R6 は外れ・R7 は当たり |
| T-J2 | **R3 の曖昧な帯 `[1.5625, 1.563)` で警告が出る**。**判定は字句どおり当たり** | 字句と導出で逆転する唯一の帯 |
| T-J3 | **R1 の件数が seed ごとの内訳つきで出る**。**2 seed だけで閾値に届く形を検出できる** | 研究計画書 §9-18 |
| T-J4 | **錨が事前登録と違えば安全弁が落とす** | 錨を後から差し替えられない形 |
| T-J5 | **測定のフラグの取り違え（対照に `--recurrent`／群に付け忘れ）を安全弁が落とす** | 対照は再帰型ではない |
| T-J6 | **p 値の実装が既知の値と一致する**（6 対 6 の完全分離 = 2/924） | 記述の p（判定には使わない） |
| T-J7 | **空振り防止 — 外れるべき入力で実際に外れる** | 「常に当たる判定」でないことの確認 |
| T-J8 | **集約が 2 段（20 迷路の中央値 → 6 seed の中央値）であり、120 走行のプールではない** | 条文どおりの集約 |

実行方法（リポジトリルートで）:
    .venv/bin/python tests/test_judge_recurrent.py
"""
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "exp_023_recurrent_policy"))

import judge_recurrent as J  # noqa: E402

N_MAZES = 20
D0 = 10          # 合成データの D₀（到達区画数 = D0 − min_d）


def make_meas(reach_per_seed, net_per_seed, respawn_per_seed, *,
              recurrent, lags=None, sham=False, n_ts=2000896, n_mazes=N_MAZES):
    """合成の測定出力を作る。**実験の出力は使わない**。

    reach_per_seed: 6 個の「20 走行の到達区画数」の列
    net_per_seed / respawn_per_seed: 同じ形の判定量の列
    """
    lags = J.EXPECTED_LAGS if lags is None else lags
    detail = {}
    for i, (rs, ns, ps) in enumerate(zip(reach_per_seed, net_per_seed, respawn_per_seed), 1):
        detail[f"seed{i}"] = dict(metrics=[
            dict(maze_seed=7000 + k, d0=D0, min_d=D0 - r, outcome="timeout", n_steps=2000,
                 net_progress_per_1000=n, respawn_per_1000=p)
            for k, (r, n, p) in enumerate(zip(rs, ns, ps))])
    per_seed_median = {k: dict(
        net_progress_per_1000=statistics.median([m["net_progress_per_1000"] for m in v["metrics"]]),
        respawn_per_1000=statistics.median([m["respawn_per_1000"] for m in v["metrics"]]))
        for k, v in detail.items()}
    across = {f: statistics.median([v[f] for v in per_seed_median.values()])
              for f in ("net_progress_per_1000", "respawn_per_1000")}
    return dict(label="synthetic", history_lags=list(lags), history_sham=bool(sham),
                recurrent=bool(recurrent),
                models=[dict(name=f"seed{i}", num_timesteps=n_ts) for i in range(1, 7)],
                detail=detail,
                summary=dict(across_seeds_median=across, per_seed_median=per_seed_median))


def const_meas(reach_counts, net, respawn, *, recurrent=True, n_mazes=N_MAZES):
    """seed ごとに「深さ 7 以上の件数」を指定して作る（残りは深さ 1）。

    net / respawn は 6 seed とも同じ定数（中央値がその値になる）。
    """
    reach = [[7] * c + [1] * (n_mazes - c) for c in reach_counts]
    nets = [[net] * n_mazes for _ in reach_counts]
    resp = [[respawn] * n_mazes for _ in reach_counts]
    return make_meas(reach, nets, resp, recurrent=recurrent)


results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"  {'✅' if cond else '❌'} {name}" + (f" — {detail}" if detail else ""))
    return bool(cond)


# ---------------------------------------------------------------------------
print("[T-J1] 同値（ちょうど閾値）の扱いが投入前の確定どおり")

# R1: ちょうど 24 件 = 当たり
g1 = const_meas([4, 4, 4, 4, 4, 4], 1.0, 1.0)          # 4 × 6 = 24 件
check("R1 ちょうど 24 件は当たり", J.judge_r1(g1)["hit"] is True,
      f"件数 {J.judge_r1(g1)['value']}")
g1_23 = const_meas([4, 4, 4, 4, 4, 3], 1.0, 1.0)       # 23 件
check("R1 23 件は外れ", J.judge_r1(g1_23)["hit"] is False)

# R3: ちょうど 1.563 = 外れ（「未満」なので）
r3 = J.judge_r3(const_meas([0] * 6, J.R3_THRESHOLD, 1.0))
check("R3 ちょうど 1.563 は外れ", r3["hit"] is False, f"値 {r3['value']}")

# R4: ちょうど 2.125 = 当たり（「以下」なので）
r4 = J.judge_r4(const_meas([0] * 6, 1.0, J.R4_THRESHOLD))
check("R4 ちょうど 2.125 は当たり", r4["hit"] is True, f"値 {r4['value']}")

# R5: 厳密に 0.05 = 外れ
gr = {f"seed{i}": dict(total_timesteps=2000896, goal_rate=v)
      for i, v in enumerate([0.0, 0.0, 0.05, 0.05, 0.05, 0.05], 1)}
r5 = J.judge_r5(gr)
check("R5 中央値が厳密に 0.05 は外れ", r5["hit"] is False, f"中央値 {r5['value']}")

# R6: 同数 = 外れ／R7: 同値 = 当たり
a = const_meas([2, 2, 2, 2, 2, 2], 1.0, 1.5)
b = const_meas([2, 2, 2, 2, 2, 2], 1.0, 1.5)
check("R6 同数は外れ", J.judge_r6(a, b)["hit"] is False,
      f"群1 {J.judge_r6(a, b)['reference']} / 群2 {J.judge_r6(a, b)['value']}")
check("R7 同値は当たり", J.judge_r7(a, b)["hit"] is True)

# ---------------------------------------------------------------------------
print("\n[T-J2] R3 の曖昧な帯 [1.5625, 1.563) で警告が出て、判定は字句どおり当たり")
r3a = J.judge_r3(const_meas([0] * 6, J.R3_DERIVED, 1.0))
check("R3 ちょうど 1.5625 は（字句 1.563 のもとで）当たり", r3a["hit"] is True)
check("R3 曖昧な帯で警告が出る", r3a.get("warning") == "R3_AMBIGUOUS_STRIP",
      str(r3a.get("warning")))
r3b = J.judge_r3(const_meas([0] * 6, 1.5, 1.0))
check("R3 帯の外（1.5）では警告が出ない", r3b.get("warning") is None)

# ---------------------------------------------------------------------------
print("\n[T-J3] R1 の件数が seed ごとの内訳つきで出る（2 seed だけで閾値に届く形を検出）")
skew = const_meas([0, 0, 0, 0, 12, 12], 1.0, 1.0)      # 合計 24・非ゼロは 2 seed
j = J.judge_r1(skew)
check("合計 24 で当たりになる", j["hit"] is True, f"合計 {j['value']}")
check("非ゼロの seed 数が 2 と分かる", j["n_seeds_nonzero"] == 2)
check("最大の 1 seed の寄与が分かる", j["max_single_seed"] == 12)
check("seed ごとの内訳が全 6 本ある", len(j["per_seed"]) == 6, str(list(j["per_seed"].values())))

# ---------------------------------------------------------------------------
print("\n[T-J4] 錨が事前登録と違えば安全弁が落とす")
GOAL_ANCHOR = {f"seed{i}": dict(total_timesteps=2000896, goal_rate=v)
               for i, v in enumerate([0.0, 0.0, 0.05, 0.0, 0.0, 0.05], 1)}   # 中央値 0.000
good_control = const_meas([0, 0, 0, 2, 5, 5], J.ANCHORS["net_progress_per_1000"],
                          J.ANCHORS["respawn_per_1000"], recurrent=False)
check("正しい錨なら安全弁は通る", J.check_anchors(good_control, GOAL_ANCHOR) == [],
      str(J.check_anchors(good_control, GOAL_ANCHOR)))
bad_control = const_meas([0, 0, 0, 2, 5, 6], J.ANCHORS["net_progress_per_1000"],
                         J.ANCHORS["respawn_per_1000"], recurrent=False)   # 13 件
msgs = J.check_anchors(bad_control, GOAL_ANCHOR)
check("錨の件数が違えば落ちる", len(msgs) >= 1, str(msgs[:1]))
bad_net = const_meas([0, 0, 0, 2, 5, 5], 1.5, J.ANCHORS["respawn_per_1000"], recurrent=False)
check("錨の中央値が違えば落ちる", len(J.check_anchors(bad_net, GOAL_ANCHOR)) >= 1)

# ---------------------------------------------------------------------------
print("\n[T-J5] 測定のフラグの取り違えを安全弁が落とす")
ctrl_wrong = const_meas([0, 0, 0, 2, 5, 5], 1.25, 2.125, recurrent=True)   # 対照に --recurrent
m = J.check_measurement("対照", ctrl_wrong, recurrent=False)
check("対照が再帰型として測られていたら落ちる", any("recurrent" in x for x in m), str(m[:1]))
g_wrong = const_meas([0] * 6, 1.0, 1.0, recurrent=False)                   # 群に付け忘れ
m2 = J.check_measurement("群 1", g_wrong, recurrent=True)
check("群に --recurrent が無ければ落ちる", any("recurrent" in x for x in m2))
sham = make_meas([[1] * N_MAZES] * 6, [[1.0] * N_MAZES] * 6, [[1.0] * N_MAZES] * 6,
                 recurrent=True, sham=True)
check("にせ履歴が混ざっていたら落ちる",
      any("history_sham" in x for x in J.check_measurement("群 1", sham, recurrent=True)))
short = make_meas([[1] * N_MAZES] * 6, [[1.0] * N_MAZES] * 6, [[1.0] * N_MAZES] * 6,
                  recurrent=True, lags=[1, 2, 4])
check("遅れの組が違えば落ちる",
      any("遅れ" in x for x in J.check_measurement("群 1", short, recurrent=True)))

# ---------------------------------------------------------------------------
print("\n[T-J6] p 値の実装が既知の値と一致する")
# 6 対 6 の完全分離: |U − 18| ≥ 18 になる並べ替えは 2 通り（C(12,6) = 924）
ex = J.mannwhitney_exact([10, 11, 12, 13, 14, 15], [1, 2, 3, 4, 5, 6])
check("6 対 6 の完全分離で p = 2/924", abs(ex["p_two_sided"] - 2 / 924) < 1e-12,
      f"p = {ex['p_two_sided']:.6f}・並べ替え {ex['n_permutations']} 通り")
check("並べ替えの総数が C(12,6) = 924", ex["n_permutations"] == 924)
same = J.mannwhitney_exact([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
check("同一の 2 群では p = 1", abs(same["p_two_sided"] - 1.0) < 1e-12,
      f"p = {same['p_two_sided']:.6f}")
na = J.mannwhitney_normal([5] * 120, [5] * 120)
check("全値が同順位なら正規近似は p = 1", abs(na["p_two_sided"] - 1.0) < 1e-9)
nb = J.mannwhitney_normal(list(range(120, 240)), list(range(120)))
check("完全分離の 120 対 120 では正規近似の p が極めて小さい", nb["p_two_sided"] < 1e-30,
      f"p = {nb['p_two_sided']:.3e}")

# ---------------------------------------------------------------------------
print("\n[T-J7] 空振り防止 — 外れるべき入力で実際に外れる")
worse = const_meas([0, 0, 0, 0, 0, 0], 2.0, 3.0)     # 深さ 0 件・前進が改善・衝突が増加
j1, j3, j4 = J.judge_r1(worse), J.judge_r3(worse), J.judge_r4(worse)
check("R1 が 0 件なら外れる", j1["hit"] is False)
check("R3 が改善（2.0）なら外れる", j3["hit"] is False, f"値 {j3['value']}")
check("R4 が悪化（3.0）なら外れる", j4["hit"] is False, f"値 {j4['value']}")
check("R3 の外れの向きが「予測と逆に改善した」と出る（AUDIT_049 要是正 1）",
      j3["direction"] == "miss_improved_against_prediction", j3["direction"])
check("R4 の外れの向きが逆向きと分かる", j4["direction"] == "miss_reverse", j4["direction"])
better = const_meas([0, 0, 0, 0, 0, 0], 1.0, 1.0)
check("R4 が対照より少なければ当たる", J.judge_r4(better)["hit"] is True)

# ---------------------------------------------------------------------------
print("\n[T-J8] 集約は 2 段（20 迷路の中央値 → 6 seed の中央値）であり 120 走行のプールではない")
# 5 seed は 0、1 seed だけ 20 本すべて 100 → プール中央値 0、2 段の中央値も 0
# 3 seed が 0・3 seed が 100 → 2 段の中央値は 50、プール中央値も 50 なので、
# 区別できる形を作る: 各 seed 内で偏らせる
reach = [[1] * N_MAZES for _ in range(6)]
nets = [[0.0] * 19 + [100.0] for _ in range(3)] + [[100.0] * 19 + [0.0] for _ in range(3)]
resp = [[1.0] * N_MAZES for _ in range(6)]
m = make_meas(reach, nets, resp, recurrent=True)
pooled = statistics.median([x for s in nets for x in s])
two_stage = J.median_of_seed_medians(m, "net_progress_per_1000")
check("2 段の中央値とプールの中央値が違う入力で、2 段の値を返す",
      abs(two_stage - 50.0) < 1e-9 and abs(pooled - 50.0) < 1e-9 or two_stage != pooled,
      f"2 段 {two_stage} / プール {pooled}")
# 上の入力ではプールも 50 になるので、seed 内の分布で差が出る形をもう 1 つ
nets2 = [[0.0] * N_MAZES for _ in range(3)] + [[0.0] * 10 + [100.0] * 10 for _ in range(3)]
m2 = make_meas(reach, nets2, resp, recurrent=True)
pooled2 = statistics.median([x for s in nets2 for x in s])
two_stage2 = J.median_of_seed_medians(m2, "net_progress_per_1000")
check("2 段 = 25.0・プール = 0.0 で値が分かれる",
      abs(two_stage2 - 25.0) < 1e-9 and abs(pooled2 - 0.0) < 1e-9,
      f"2 段 {two_stage2} / プール {pooled2}")
check("要約と再計算の食い違いを安全弁が検出する",
      any("食い違う" in x for x in J.check_measurement(
          "群 1", {**m2, "summary": {**m2["summary"],
                                     "across_seeds_median": {"net_progress_per_1000": 99.0,
                                                             "respawn_per_1000": 1.0}}},
          recurrent=True)))

# ---------------------------------------------------------------------------
print("\n[T-J9] 准教授 AUDIT_049 の是正 6 件の再発検出")

# 是正 1 — R3 の外れは「閾値より上」。**「閾値未達」と表示してはいけない**
for v in (1.75, 2.50):
    d = J.judge_r3(const_meas([0] * 6, v, 1.0))
    check(f"是正1 R3 の外れ（値 {v}）が「予測と逆に改善した」と出る",
          d["direction"] == "miss_improved_against_prediction", d["direction"])
d = J.judge_r3(const_meas([0] * 6, 1.0, 1.0))
check("是正1 R3 が当たるときは hit", d["direction"] == "hit")

# 是正 2 — R5 の外れは「対照より悪化」ではない（ゴール率が上がるのは機体としては改善）
for v in (0.05, 0.10, 0.20):
    gr = {f"seed{i}": dict(total_timesteps=2000896, goal_rate=v) for i in range(1, 7)}
    d = J.judge_r5(gr)
    check(f"是正2 R5 の外れ（ゴール率 {v}）が「予測と逆に改善した」と出る",
          d["direction"] == "miss_improved_against_prediction", d["direction"])
# R1・R4 は旧来どおりであること（是正が別の判定を壊していない）
check("是正1・2 の巻き添えなし: R1 の 6 件は miss_reverse",
      J.judge_r1(const_meas([1, 1, 1, 1, 1, 1], 1.0, 1.0))["direction"] == "miss_reverse")
check("是正1・2 の巻き添えなし: R1 の 18 件は miss_below_threshold",
      J.judge_r1(const_meas([3, 3, 3, 3, 3, 3], 1.0, 1.0))["direction"] == "miss_below_threshold")
check("是正1・2 の巻き添えなし: R4 の 3.0 は miss_reverse",
      J.judge_r4(const_meas([0] * 6, 1.0, 3.0))["direction"] == "miss_reverse")

# 是正 4 — 定期評価の最終点が seed 間で揃っていなければ落とす
gr_ok = {f"seed{i}": dict(total_timesteps=2000896, goal_rate=0.0) for i in range(1, 7)}
check("是正4 最終点が揃っていれば通る",
      J.check_goal_rate_timepoints("群 1", gr_ok, 2000896) == [])
gr_bad = dict(gr_ok, seed3=dict(total_timesteps=1900000, goal_rate=0.0))
msgs = J.check_goal_rate_timepoints("群 1", gr_bad, 2000896)
check("是正4 1 本だけ 190 万歩なら落ちる", any("揃っていない" in x for x in msgs), str(msgs[:1]))
check("是正4 保存済みモデルの学習量と違えば落ちる",
      any("違う" in x for x in J.check_goal_rate_timepoints("群 1", gr_ok, 1900000)))
check("是正4 seed 数が 6 でなければ落ちる",
      any("seed 数" in x for x in J.check_goal_rate_timepoints(
          "群 1", {k: v for k, v in list(gr_ok.items())[:5]}, 2000896)))

# 是正 5 — R5 の錨も再計算・照合の対象
ctrl = const_meas([0, 0, 0, 2, 5, 5], J.ANCHORS["net_progress_per_1000"],
                  J.ANCHORS["respawn_per_1000"], recurrent=False)
gr_anchor = {f"seed{i}": dict(total_timesteps=2000896, goal_rate=v)
             for i, v in enumerate([0.0, 0.0, 0.05, 0.0, 0.0, 0.05], 1)}
check("是正5 正しい R5 の錨なら通る", J.check_anchors(ctrl, gr_anchor) == [],
      str(J.check_anchors(ctrl, gr_anchor)))
gr_anchor_bad = {f"seed{i}": dict(total_timesteps=2000896, goal_rate=0.10) for i in range(1, 7)}
check("是正5 R5 の錨が違えば落ちる",
      any("goal_rate" in x for x in J.check_anchors(ctrl, gr_anchor_bad)),
      str(J.check_anchors(ctrl, gr_anchor_bad)[:1]))

# 是正 6 — 🔴 当初定義は「常に旗が立つ」。**この欠陥自体を検査する**（裁定が下りるまでの回帰防止）
import itertools  # noqa: E402
always = all(J.bracket_check({"q": dict(zip(["対照", "群 1", "群 2"], v))})["any_outside"]
             for v in itertools.permutations([10, 20, 30]))
check("是正6 当初定義は 3 群が相異なる全 6 通りで必ず旗が立つ（＝ 判別力が無い）", always is True,
      "真ん中は 1 つだけ・検査対象は 2 つなので必ず一方が外に出る")
check("是正6 定義が裁定待ちであることが出力に出る",
      J.bracket_check({"q": {"対照": 12, "群 1": 20, "群 2": 30}})["ruling_pending"] is True)
# 裁定待ちの間は、旗が立っても表を抑制しない（抑制すると表が一度も返らないため）
r1_hit = J.judge_r1(const_meas([4] * 6, 1.0, 1.0))
r6_hit = J.judge_r6(const_meas([2] * 6, 1.0, 1.0), const_meas([3] * 6, 1.0, 1.0))
flagged = J.bracket_check({"q": {"対照": 12, "群 1": 20, "群 2": 30}})
cov = J.read_coverage(r1_hit, r6_hit, flagged)
check("是正6 裁定待ちの間は旗が立っても表を返す（期待される結果でも旗が立つため）",
      cov["table_returned"] is True and bool(cov["reading"]), str(cov.get("reading")))
check("是正6 順序そのものは記述として出る",
      flagged["per_quantity"]["q"]["order_low_to_high"] == ["対照", "群 1", "群 2"],
      str(flagged["per_quantity"]["q"]["order_low_to_high"]))

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
n_ok = sum(1 for _, ok, _ in results if ok)
for name, ok, _ in results:
    if not ok:
        print(f"  ❌ FAIL  {name}")
print(f"  {n_ok} / {len(results)} PASS")
raise SystemExit(0 if n_ok == len(results) else 1)
