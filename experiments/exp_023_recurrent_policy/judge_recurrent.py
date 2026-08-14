#!/usr/bin/env python3
"""exp_023（再帰型方策実験・第 2 弾 B）の R1・R3〜R7 を機械的に判定する。

Mechanically evaluate the pre-registered predictions R1, R3-R7 for exp_023.

🔴 **判定は手で計算しない。**本 script は **`card.md` §3 の条文と 1 対 1 対応**する形で書かれており、
**閾値・対照値（錨）・同値の扱いはすべて投入前に確定した定数**である。
**錨は測定出力から再計算して照合し、食い違えば落ちる**（**後から錨を差し替えられない形にする**）。

## 条文（`card.md` §3）との対応

| 条文 | 実装 |
|---|---|
| **R1**（主）7 区画以上到達の走行数が **≥ 24 件**（対照 12/120） | `judge_r1()` |
| **R3** 正味の前進は改善しない = `net_progress_per_1000` の 6 seed 中央値 **< 1.563**（対照 1.250） | `judge_r3()` |
| **R4** 衝突は増えない = `respawn_per_1000` の 6 seed 中央値 **≤ 2.125**（対照 2.125） | `judge_r4()` |
| **R5** ゴール率は床のまま = 6 seed 中央値 **< 0.05**（対照 0.000） | `judge_r5()` |
| **R6** 群 2 の `n_reach_ge7` が**群 1 より多い** | `judge_r6()` |
| **R7** 群 2 の `respawn_per_1000` 中央値が**群 1 以下** | `judge_r7()` |
| **旧 R2** は**判定から外し記述としてのみ報告**（§3-1-bis・§8 限界 6） | `describe_reach()`（`verdict` を持たない） |
| §3-3 外れの向きの区別（閾値未達／逆向き／どちらの群よりも外） | 各 `judge_*()` の `direction` |
| §3-4 全域被覆（R1 × R6 の 4 通り ＋ 第 3 の読み） | `read_coverage()` |

## 🔴 投入前に確定した「同値（判定量がちょうど閾値に一致した場合）」の扱い

**教授裁定 2026-08-15（カード改訂は不要 — 登録済みの字句そのものから決まるため）。**

| 判定 | 条文の字句 | 同値のとき |
|---|---|---|
| R1 `≥ 24` | 「24 件以上」 | **当たり** |
| R3 `< 1.563` | 「1.563 未満」 | **外れ** |
| R4 `≤ 2.125` | 「2.125 以下」 | **当たり** |
| R5 `< 0.05` | 「0.05 未満」（カード明示） | **外れ** |
| R6 | 「群 1 **より多い**」 | **外れ**（真に多いことを要求） |
| R7 | 「群 1 **以下**」 | **当たり**（等号を含む） |

**R6 と R7 が非対称なのは、非対称なのが登録済みの字句のほうだからである。**

## 🔴 R3 の閾値の丸め（准教授 AUDIT_048_PREREG・投入前に確定）

**カードの `1.563` は導出値 1.25 × 1.250 = `1.5625` の 3 桁丸めである。**
**判定量は 0.0625 の倍数を取りうる**（20 迷路の中央値 → 0.125 刻み・その 6 seed 中央値 → 0.0625 刻み）
ので、**厳密に 1.5625 が出る経路が存在する**。

- **処置: カードの字句 `1.563` に従う**（＝ 1.5625 は当たり）。**事前登録した閾値を後から動かさない**
- **ただし黙って通さない**: **測定値が `[1.5625, 1.563)` の帯に入ったら警告を出し、
  「字句と導出で判定が逆転する」ことを出力に明記する**（`R3_AMBIGUOUS_STRIP`）

## 🔴 報告の形（准教授の事前登録・研究計画書 §9-18）

- **R1・R6 の件数は seed ごとの内訳を必ず併記する。**
  **対照 exp_021 の 12 件の内訳は `[0, 0, 0, 2, 5, 5]`** ＝ **6 本のうち 3 本が 0 件・12 件中 10 件が 2 seed 由来**。
  **閾値 24 件は 20 走行 × 2 seed で作れる**（1 seed の上限は 20 件）ので、**合計だけで「増えた」と書かない**
- **R5 は seed ごとの 6 値を必ず併記する。**対照は `0, 0, 0, 0, 0.05, 0.05` で中央値 0.000 ＝
  **あと 2 本増えると中央値が厳密に 0.05 になり「外れ」側に乗る**

## 使い方

```bash
.venv/bin/python experiments/exp_023_recurrent_policy/judge_recurrent.py \
    --control outputs/exp_021_driving_treat_final.json \
    --group1  outputs/exp_023a_driving_final.json \
    --group2  outputs/exp_023b_driving_final.json \
    --logs-g1 logs/exp_023a_seed{1,2,3,4,5,6} \
    --logs-g2 logs/exp_023b_seed{1,2,3,4,5,6} \
    --logs-control logs/exp_021_seed{1,2,3,4,5,6} \
    --out outputs/exp_023_judgment.json
```

## 准教授 AUDIT_049 の是正（**教授が全件採択・2026-08-15**）

| # | 是正 | 実装 |
|---|---|---|
| 1・2 | **外れの向きの旗を 2 つに分けた** — `hit_is_low`（判定の当たりの向き）と `improvement_is_high`（量として良い向き）。**R3・R5 はこの 2 つが分かれる**ので、旧実装は R3 の外れを「閾値未達」（実際は閾値より上）・R5 の外れを「対照より悪化」（実際はゴール率が上がった ＝ 改善）と表示していた | `_direction()` |
| 3 | **`git diff --quiet` の終了コードを 0／1／2 以上で分けた**。**git 自体の失敗を「群が別版で走った」と誤診断しない** | `check_same_version()` |
| 4 | **定期評価の記録の最終点が 6 本で同じ時点かを検査**（ある seed だけ最終評価が欠けると、別の時点の値が静かに混ざる） | `check_goal_rate_timepoints()` |
| 5 | **R5 の錨 0.000 も再計算・照合の対象にした**（`--logs-control` を必須にした）。**4 つの錨すべてで原則を守る** | `check_anchors()` |
| 6 | **§3-3 (iii)「どちらの群よりも外」を計算する**（教授裁定）。**判定量ごとに 3 群の順序を出し、群 1 か群 2 が他の 2 群の張る範囲の外なら旗を立て、`read_coverage()` が表を返さない** | `bracket_check()` |
"""
import argparse
import json
import math
import statistics
from itertools import combinations
from pathlib import Path

# ---------------------------------------------------------------------------
# 🔴 事前登録した定数（`card.md` §3・投入前に確定。**変更しない**）
# ---------------------------------------------------------------------------

#: 対照群 = exp_021（観測履歴の連結）の最終方策。**錨は測定出力から再計算して照合する**。
ANCHORS = dict(
    n_reach_ge7=12,                 # 120 走行の件数（内訳 [0, 0, 0, 2, 5, 5]）
    n_reach_ge7_per_seed=[0, 0, 0, 2, 5, 5],
    net_progress_per_1000=1.250,    # 6 seed 中央値
    respawn_per_1000=2.125,         # 6 seed 中央値
    goal_rate=0.000,                # 定期評価の最終点の 6 seed 中央値
)

REACH_DEPTH = 7                     # R1・R6 の深さ［区画］
R1_THRESHOLD = 24                   # 件数（≥ で当たり）
R3_THRESHOLD = 1.563                # 未満で当たり（**カードの字句。導出値は 1.5625**）
R3_DERIVED = 1.5625                 # 導出値（1.25 × 1.250）。**判定には使わない**
R4_THRESHOLD = 2.125                # 以下で当たり
R5_THRESHOLD = 0.05                 # 未満で当たり（**厳密に 0.05 は外れ**）

#: 錨の照合の許容差（浮動小数の丸めのみを吸収する幅。**設計上の余裕ではない**）
ANCHOR_TOL = 1e-9

#: 旧 R2（記述としてのみ報告）で並べる深さ（カード §3-1-bis）
DESCRIPTIVE_DEPTHS = (5, 6, 7, 8, 10)

#: 観測履歴の遅れ［歩］。3 群すべてで同一でなければならない。
EXPECTED_LAGS = [1, 2, 4, 8, 16, 32, 64, 128]

N_SEEDS = 6
N_MAZES = 20


# ---------------------------------------------------------------------------
# 測定出力の読み取り（**要約を信用せず、走行ごとの記録から再計算して照合する**）
# ---------------------------------------------------------------------------

def _seed_keys(meas: dict) -> list:
    """seed の呼び名を測定出力の記録順で返す（並べ替えない）。"""
    return list(meas["detail"].keys())


def reach_depths(meas: dict) -> dict:
    """seed ごとの到達区画数の列を返す。

    **到達区画数 = D₀ − エピソード中の最小 D**（`d0` − `min_d`）。
    D はゴールまでの迷路距離［区画数］。**カードの `n_reach_ge7` はこの量の閾値件数**である。
    """
    return {k: [m["d0"] - m["min_d"] for m in v["metrics"]] for k, v in meas["detail"].items()}


def count_reach_ge(meas: dict, depth: int) -> dict:
    """深さ `depth` 以上に到達した走行の件数を、合計と seed ごとの内訳で返す。

    🔴 **合計だけを返さない**（研究計画書 §9-18。合計は少数の seed に支配されうる）。
    """
    per_seed = {k: sum(1 for r in v if r >= depth) for k, v in reach_depths(meas).items()}
    counts = list(per_seed.values())
    return dict(total=sum(counts), per_seed=per_seed, per_seed_counts=counts,
                n_seeds_nonzero=sum(1 for c in counts if c > 0),
                max_single_seed=max(counts) if counts else 0)


def median_of_seed_medians(meas: dict, field: str) -> float:
    """迷路 20 本の中央値 → 6 seed の中央値（`measure_driving.py` の集約と同じ形）。

    **要約の値をそのまま使わず、走行ごとの記録から計算し直す**（要約と食い違えば安全弁で落ちる）。
    """
    per_seed = [statistics.median([m[field] for m in v["metrics"]])
                for v in meas["detail"].values()]
    return statistics.median(per_seed)


def seed_medians(meas: dict, field: str) -> dict:
    return {k: statistics.median([m[field] for m in v["metrics"]])
            for k, v in meas["detail"].items()}


def check_goal_rate_timepoints(tag: str, goal_rates: dict, expected_steps=None) -> list:
    """🔴 定期評価の記録の**最終点が 6 本で同じ学習量の時点**かを検査する。

    **成立する事故（准教授 AUDIT_049 要是正 4）**: **ある seed だけ最終評価が書かれていないと、
    その seed だけ 190 万歩の値が混ざり、6 seed 中央値が別の時点の混合になる。例外は出ない。**

    **安全弁が確かめている `num_timesteps` は保存済みモデルの学習量であって、
    定期評価の記録の最終点ではない** — 別の記録なので、こちらを独立に検査する。
    """
    msgs = []
    ts = sorted({v["total_timesteps"] for v in goal_rates.values()})
    if len(goal_rates) != N_SEEDS:
        msgs.append(f"{tag}: ゴール率の seed 数が {N_SEEDS} でない（{len(goal_rates)}）")
    if len(ts) != 1:
        msgs.append(f"{tag}: 定期評価の最終点が seed 間で揃っていない（{ts}）"
                    " ＝ 別の時点の値が混ざっている")
    elif expected_steps is not None and ts[0] != expected_steps:
        msgs.append(f"{tag}: 定期評価の最終点 {ts[0]:,} 歩が"
                    f"保存済みモデルの学習量 {expected_steps:,} 歩と違う")
    return msgs


def final_goal_rates(log_dirs) -> dict:
    """定期評価の記録の**最終点**の `goal_rate` を seed ごとに返す。

    🔴 **走行測定の `outcome == "goal"` ではない**（准教授の事前登録・カード R5）。
    exp_021 の判定 Q3 と同じ経路である。
    """
    out = {}
    for d in log_dirs:
        h = json.loads((Path(d) / "validation_history.json").read_text(encoding="utf-8"))
        if not h:
            raise SystemExit(f"{d}/validation_history.json が空である")
        last = h[-1]
        out[Path(d).name] = dict(total_timesteps=last["total_timesteps"],
                                 goal_rate=last["goal_rate"])
    return out


def group_flag_from_train_log(log_dir) -> dict:
    """`train.log` から「リスポーンで隠れ状態をリセット」の有効/無効を読む。

    🔴 **脆い経路である。**`run_summary.json` は第 2 弾で追加した設定
    （`obs_history`・`recurrent`・`lstm_hidden`・`respawn_reset`）を **1 つも記録していない**ため、
    **走行が群 1 か群 2 かの機械可読な記録が存在しない**。**現状ではこの日本語 1 行しか根拠が無い。**
    （作法 9 の設定版。**exp_021・exp_022 も同じ欠落**。是正は両群完走後 = R11 へ登録済み）
    """
    txt = (Path(log_dir) / "train.log").read_text(encoding="utf-8", errors="replace")
    reset = None
    for line in txt.splitlines():
        if "リスポーンで隠れ状態をリセット" in line:
            reset = ("有効" in line)
            break
    recurrent = "RecurrentPPO" in txt
    return dict(respawn_reset=reset, recurrent=recurrent)


# ---------------------------------------------------------------------------
# 判定（条文と 1 対 1）
# ---------------------------------------------------------------------------

#: 外れの向きの表示（§3-3）。**旗を 2 つに分ける**（准教授 AUDIT_049 要是正 1・2／教授発注）。
#:
#: 🔴 **1 つの旗に 2 つの意味を乗せてはいけない**:
#: - **`hit_is_low`** = **判定の当たりの向き**（「低いままなら当たり」＝ 帰無の予測か）
#: - **`improvement_is_high`** = **量として良い向き**（値が高いほど機体として良いか）
#:
#: **この 2 つは R3・R5 で分かれる** — **どちらも「低いままなら当たり」だが、
#: 正味の前進もゴール率も「高いほど良い」**。旧実装は 1 つの旗（`better_is_high`）に
#: 両方を乗せていたため、**R3 の外れが「閾値未達」（実際は閾値より上）・
#: R5 の外れが「対照より悪化」（実際はゴール率が上がった ＝ 機体としては改善）と表示された**。
DIRECTIONS = {
    "hit": "当たり",
    "miss_improved_against_prediction": "外れ（**予測と逆に改善した**）",
    "miss_reverse": "外れ（**対照より悪化**）",
    "miss_below_threshold": "外れ（**閾値に届かない**。対照よりは良い側）",
}


def _direction(value, threshold, control, *, hit_is_low: bool,
               improvement_is_high: bool, hit: bool) -> str:
    """§3-3 外れの向きの区別。**予測に対してどちら側へ出たか**で分ける。"""
    if hit:
        return "hit"
    # 外れがどちら側へ出たかは、当たりの向きで決まる
    #   hit_is_low  なら外れ = 閾値より「上」  → 上へ出たことが改善か？
    #   hit_is_low でないなら外れ = 閾値より「下」→ 下へ出たことが改善か？
    improved = improvement_is_high if hit_is_low else (not improvement_is_high)
    if improved:
        return "miss_improved_against_prediction"
    worse_than_control = (value < control) if improvement_is_high else (value > control)
    return "miss_reverse" if worse_than_control else "miss_below_threshold"


def judge_r1(g1: dict) -> dict:
    c = count_reach_ge(g1, REACH_DEPTH)
    hit = c["total"] >= R1_THRESHOLD          # 同値（ちょうど 24）は当たり
    return dict(
        key="R1", primary=True,
        name=f"R1 {REACH_DEPTH} 区画以上に到達した走行数",
        clause=f"120 走行の件数 ≥ {R1_THRESHOLD}（同値は当たり）",
        value=c["total"], threshold=R1_THRESHOLD, control=ANCHORS["n_reach_ge7"],
        per_seed=c["per_seed"], n_seeds_nonzero=c["n_seeds_nonzero"],
        max_single_seed=c["max_single_seed"],
        hit=bool(hit),
        direction=_direction(c["total"], R1_THRESHOLD, ANCHORS["n_reach_ge7"],
                             hit_is_low=False, improvement_is_high=True, hit=hit),
        note=(f"1 seed の上限は {N_MAZES} 件なので、閾値 {R1_THRESHOLD} 件は "
              f"{N_MAZES} 走行 × 2 seed で作れる。**合計だけで「増えた」と書かない** "
              f"（研究計画書 §9-18）。対照の内訳は {ANCHORS['n_reach_ge7_per_seed']}"))


def judge_r3(g1: dict) -> dict:
    v = median_of_seed_medians(g1, "net_progress_per_1000")
    hit = v < R3_THRESHOLD                     # 「改善しない」の予測。同値は外れ
    ambiguous = (R3_DERIVED <= v < R3_THRESHOLD)
    out = dict(
        key="R3", primary=False, name="R3 正味の前進は改善しない",
        clause=f"net_progress_per_1000 の 6 seed 中央値 < {R3_THRESHOLD}（同値は外れ）",
        value=v, threshold=R3_THRESHOLD, derived_threshold=R3_DERIVED,
        control=ANCHORS["net_progress_per_1000"],
        per_seed=seed_medians(g1, "net_progress_per_1000"),
        hit=bool(hit),
        # 🔴 「低いままなら当たり」だが「正味の前進は高いほど良い」＝ 2 つの向きが分かれる判定
        direction=_direction(v, R3_THRESHOLD, ANCHORS["net_progress_per_1000"],
                             hit_is_low=True, improvement_is_high=True, hit=hit))
    if ambiguous:
        out["warning"] = "R3_AMBIGUOUS_STRIP"
        out["warning_detail"] = (
            f"🔴 測定値 {v} は [{R3_DERIVED}, {R3_THRESHOLD}) の帯に入った。"
            f"**カードの字句 {R3_THRESHOLD} では当たり・導出値 {R3_DERIVED} では外れ**で判定が逆転する。"
            "投入前の確定どおり字句を採ったが、判定文にこの逆転を明記すること"
            "（准教授 AUDIT_048_PREREG・教授承認 2026-08-15）")
    return out


def judge_r4(g1: dict) -> dict:
    v = median_of_seed_medians(g1, "respawn_per_1000")
    hit = v <= R4_THRESHOLD                    # 同値は当たり
    return dict(
        key="R4", primary=False, name="R4 衝突は対照より増えない",
        clause=f"respawn_per_1000 の 6 seed 中央値 ≤ {R4_THRESHOLD}（同値は当たり）",
        value=v, threshold=R4_THRESHOLD, control=ANCHORS["respawn_per_1000"],
        per_seed=seed_medians(g1, "respawn_per_1000"),
        hit=bool(hit),
        # 衝突は「低いままなら当たり」かつ「低いほど良い」＝ 2 つの向きが一致する判定
        direction=_direction(v, R4_THRESHOLD, ANCHORS["respawn_per_1000"],
                             hit_is_low=True, improvement_is_high=False, hit=hit))


def judge_r5(goal_rates: dict) -> dict:
    vals = [v["goal_rate"] for v in goal_rates.values()]
    med = statistics.median(vals)
    hit = med < R5_THRESHOLD                   # 厳密に 0.05 は外れ（カード明示）
    return dict(
        key="R5", primary=False, name="R5 ゴール率は床のまま",
        clause=(f"定期評価の記録の**最終点**の goal_rate の 6 seed 中央値 < {R5_THRESHOLD}"
                "（厳密に 0.05 は外れ）"),
        value=med, threshold=R5_THRESHOLD, control=ANCHORS["goal_rate"],
        per_seed={k: v["goal_rate"] for k, v in goal_rates.items()},
        per_seed_sorted=sorted(vals),
        hit=bool(hit),
        # 🔴 「低いままなら当たり」だが「ゴール率は高いほど良い」＝ 2 つの向きが分かれる判定。
        # ゴール率が上がって外れた場合は「対照より悪化」ではなく「予測と逆に改善した」である
        direction=_direction(med, R5_THRESHOLD, ANCHORS["goal_rate"],
                             hit_is_low=True, improvement_is_high=True, hit=hit),
        note=("判定量の刻みは 1/20 = 0.05。**seed ごとの値を必ず併記する** — "
              "対照は 0, 0, 0, 0, 0.05, 0.05 で中央値 0.000 であり、"
              "あと 2 本増えると中央値が厳密に 0.05 になって外れ側に乗る"))


def judge_r6(g1: dict, g2: dict) -> dict:
    c1, c2 = count_reach_ge(g1, REACH_DEPTH), count_reach_ge(g2, REACH_DEPTH)
    hit = c2["total"] > c1["total"]            # 同数は外れ（「より多い」＝ 真に多い）
    return dict(
        key="R6", primary=True, name="R6 群 2（リセットする）は群 1 より深く入る",
        clause=f"群 2 の n_reach_ge{REACH_DEPTH} が群 1 より多い（同数は外れ）",
        value=c2["total"], reference=c1["total"],
        per_seed_g2=c2["per_seed"], per_seed_g1=c1["per_seed"],
        hit=bool(hit),
        direction=("hit" if hit else ("miss_tie" if c2["total"] == c1["total"] else "miss_reverse")),
        note=("群ごとの seed の内訳を必ず併記する。**同数は外れ**（投入前に確定）"))


def judge_r7(g1: dict, g2: dict) -> dict:
    v1 = median_of_seed_medians(g1, "respawn_per_1000")
    v2 = median_of_seed_medians(g2, "respawn_per_1000")
    hit = v2 <= v1                             # 同値は当たり（「以下」＝ 等号を含む）
    return dict(
        key="R7", primary=False, name="R7 群 2 は群 1 より衝突が少ない",
        clause="群 2 の respawn_per_1000 の 6 seed 中央値が群 1 以下（同値は当たり）",
        value=v2, reference=v1,
        per_seed_g2=seed_medians(g2, "respawn_per_1000"),
        per_seed_g1=seed_medians(g1, "respawn_per_1000"),
        hit=bool(hit),
        direction=("hit" if hit else "miss_reverse"),
        note="R6 が「より多い」（同数は外れ）なのに R7 が「以下」（同値は当たり）なのは、字句が非対称だから")


#: 🔴 判定量ごとの「**量として良い向き**」（**予測の向きとは別**。教授裁定 2026-08-15 の追加 ①）。
#:
#: **`direction` の実装で踏んだ罠と同じもの**である — **ゴール率は「高いほど良い」が
#: 「低いままだと予測した」量**なので、**この 2 つを 1 つの旗にまとめてはいけない。**
#: **本表は「良い向き」だけを持ち、予測の向き（`hit_is_low`）は各 `judge_*()` が持つ。**
QUANTITY_IMPROVEMENT_IS_HIGH = {
    "n_reach_ge7": True,              # 深くまで到達するほど良い
    "net_progress_per_1000": True,    # 正味の前進は大きいほど良い
    "respawn_per_1000": False,        # 衝突（立て直し）は少ないほど良い
    "goal_rate": True,                # ゴール率は高いほど良い
}

#: §3-4 の全域被覆表を抑止しうる判定量（教授裁定の追加 ②）。
#: **表の構成量（R1・R6 が使う `n_reach_ge7`）の旗だけが表を消す。**
#: **無関係な量の悪化で表を消さない。**
TABLE_SUPPRESSING_QUANTITY = f"n_reach_ge{REACH_DEPTH}"


def bracket_check(quantities: dict) -> dict:
    """🔴 §3-3 (iii)「どちらの群よりも外」を計算する（**代案 B**・教授裁定 2026-08-15）。

    **定義: 群 1 か群 2 が「対照より悪い側」へ出たら旗を立てる。**

    **当初裁定された定義（「群 1 か群 2 が他の 2 群の張る範囲の外」）は棄却された** —
    **実装して 3 群の全 6 通りの順序で叩いたところ、値が相異なるかぎり必ず旗が立つ**ことが
    分かったため（**相異なる 3 つの値のうち真ん中は 1 つだけで、検査の対象は 2 つあるから**）。
    **その定義では §3-4 の表が一度も返らず、期待する結果（対照 < 群 1 < 群 2）でも旗が立つ。**
    **誤りの型 5（構成上必ず成立する形）の裏返しだった。**

    **代案 B が拾うのは「§3-4 の 4 通りの表が表現できない事象」＝ 悪化**である。
    表の「外」の行は**「閾値に届かない」**を意味しており、**「対照より悪くなった」は別の事象**
    （この区別は `direction` の `miss_reverse` としても計算している）。

    quantities: {判定量の名前: {"対照": 値, "群 1": 値, "群 2": 値}}
    """
    out, any_flag = {}, False
    for name, vals in quantities.items():
        if name not in QUANTITY_IMPROVEMENT_IS_HIGH:
            raise KeyError(f"判定量 {name!r} の「良い向き」が事前登録されていない")
        high_is_good = QUANTITY_IMPROVEMENT_IS_HIGH[name]
        control = vals["対照"]
        worse = {}
        for target in ("群 1", "群 2"):
            v = vals[target]
            worse[target] = (v < control) if high_is_good else (v > control)
        any_flag = any_flag or any(worse.values())
        out[name] = dict(values=vals,
                         order_low_to_high=sorted(vals, key=lambda k: vals[k]),
                         improvement_is_high=high_is_good,
                         worse_than_control=worse,
                         flag=any(worse.values()))
    suppress = bool(out.get(TABLE_SUPPRESSING_QUANTITY, {}).get("flag"))
    return dict(any_flag=any_flag, suppress_table=suppress, per_quantity=out,
                suppressing_quantity=TABLE_SUPPRESSING_QUANTITY,
                note=("群 1 か群 2 が『対照より悪い側』へ出たら旗を立てる（代案 B・教授裁定）。"
                      "旗は判定量ごとに立てて全部報告するが、"
                      f"§3-4 の表を抑止するのは表の構成量 {TABLE_SUPPRESSING_QUANTITY} の旗だけ"
                      "（無関係な量の悪化で表を消さない）"))


def read_coverage(r1: dict, r6: dict, bracket: dict) -> dict:
    """§3-4 全域被覆（R1 × R6 の 4 通り）。**第 3 の読みが出たら表を引かない**。"""
    if bracket["suppress_table"]:
        return dict(r1_hit=r1["hit"], r6_hit=r6["hit"], table_returned=False,
                    reading=None, next_move=None,
                    bracket_flag_raised=True, suppressing_quantity=bracket["suppressing_quantity"],
                    reason=("🔴 §3-3 (iii) が成立した ＝ **表の構成量 "
                            f"{bracket['suppressing_quantity']} で群 1 か群 2 が対照より悪い側へ出た**。"
                            "カード :277 により §3-4 の表は引かず、そのまま報告する"),
                    outside=bracket["per_quantity"])
    table = {
        (True, True): ("再帰構造が効き、汚染を除くと更に効く",
                       "M2 の本命として記憶構造を進める。汚染の除去を標準に"),
        (True, False): ("再帰構造は効くが、汚染の除去は効かない",
                        "記憶構造を進める。リセットは既定のままでよい"),
        (False, True): ("汚染されたままでは効かないが、除けば効く",
                        "(C) が主因。環境側（リスポーンの仕組み）の見直しへ"),
        (False, False): ("再帰構造でも届かない",
                         "記憶の方向を打ち切る前に、§8 限界 3 の後続（容量の軸）を検討する"),
    }
    reading, next_move = table[(r1["hit"], r6["hit"])]
    # 🔴 旗が立っているのに表を返す（裁定待ち）のと、旗が立たないから表を返すのは別の状態である。
    # **同じ説明文を出すと、前者で「成立していない」という偽の文が判定文書に載る**
    # （准教授 AUDIT_049 §10）。
    if bracket["any_flag"]:
        flagged = [k for k, v in bracket["per_quantity"].items() if v["flag"]]
        caveat = ("🔴 §3-3 (iii) の旗が**別の判定量で立っている**"
                  f"（{'・'.join(flagged)}）が、**表の構成量 "
                  f"{bracket['suppressing_quantity']} では立っていない**ので表は返す"
                  "（教授裁定: 無関係な量の悪化で表を消さない）。旗の中身は bracket_check を見ること")
    else:
        caveat = "§3-3 (iii) は計算済みで、どの判定量でも成立していない"
    return dict(r1_hit=r1["hit"], r6_hit=r6["hit"], table_returned=True,
                reading=reading, next_move=next_move,
                bracket_flag_raised=bool(bracket["any_flag"]),
                caveat=caveat)


# ---------------------------------------------------------------------------
# 旧 R2 = 記述としてのみ報告（**判定の分岐には使わない**・§3-1-bis・§8 限界 6）
# ---------------------------------------------------------------------------

def _ranks(values: list) -> list:
    """同順位に平均順位を割り当てる（tie の扱い = 平均順位法）。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _u_statistic(a: list, b: list) -> float:
    """Mann-Whitney の U（群 a 側）。同順位は平均順位法で数える。"""
    pooled = list(a) + list(b)
    r = _ranks(pooled)
    r_a = sum(r[: len(a)])
    return r_a - len(a) * (len(a) + 1) / 2.0


def mannwhitney_exact(a: list, b: list) -> dict:
    """並べ替えを全数えして両側 p を出す（小標本用。6 対 6 なら C(12,6) = 924 通り）。

    **両側 p の定義**: **観測と同じかそれ以上に極端な並べ替えの割合**
    （`|U − n₁n₂/2|` が観測以上）。**片側の 2 倍ではない。**
    """
    n1, n2 = len(a), len(b)
    pooled = list(a) + list(b)
    ranks = _ranks(pooled)
    mu = n1 * n2 / 2.0
    u_obs = _u_statistic(a, b)
    dev_obs = abs(u_obs - mu)
    idx = range(n1 + n2)
    total = extreme = 0
    for comb in combinations(idx, n1):
        r_a = sum(ranks[i] for i in comb)
        u = r_a - n1 * (n1 + 1) / 2.0
        total += 1
        if abs(u - mu) >= dev_obs - 1e-12:
            extreme += 1
    return dict(method="exact_permutation", u=u_obs, n1=n1, n2=n2,
                p_two_sided=extreme / total, n_permutations=total,
                tie_handling="平均順位法", two_sided_definition="|U − n₁n₂/2| が観測以上の割合")


def mannwhitney_normal(a: list, b: list) -> dict:
    """正規近似（同順位補正つき・連続性の補正つき）。**大標本用**（120 対 120 など）。

    **作法 40 は「小さい n で同順位が多いとき近似を使うな」なので、120 対 120 は対象外。**
    """
    n1, n2 = len(a), len(b)
    pooled = list(a) + list(b)
    n = n1 + n2
    u = _u_statistic(a, b)
    mu = n1 * n2 / 2.0
    counts = {}
    for v in pooled:
        counts[v] = counts.get(v, 0) + 1
    tie_term = sum(t ** 3 - t for t in counts.values())
    var = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    if var <= 0:
        return dict(method="normal_approx", u=u, n1=n1, n2=n2, p_two_sided=1.0,
                    note="分散が 0（全値が同順位）")
    z = (abs(u - mu) - 0.5) / math.sqrt(var)      # 連続性の補正
    z = max(z, 0.0)
    p = math.erfc(z / math.sqrt(2.0))             # 両側
    return dict(method="normal_approx", u=u, n1=n1, n2=n2, z=z, p_two_sided=min(p, 1.0),
                tie_handling="平均順位法＋同順位補正", continuity_correction=True,
                two_sided_definition="片側の 2 倍（正規近似）")


def describe_reach(meas_by_name: dict) -> dict:
    """🔴 **記述**（判定の分岐には使わない・§3-1-bis）。

    - 深さ 5・6・7・8・10 のそれぞれの件数（合計と seed ごとの内訳）
    - **seed 水準（6 対 6・正確法）**と**走行水準（120 対 120・正規近似）**の両方の p

    **seed の代表値 = その seed の 20 走行の到達区画数の中央値**（准教授の 4 軸のうち軸 1。
    **他の取り方でも再計算できるよう、seed ごとの列そのものも出力する**）。
    """
    out = dict(note=("🔴 判定には使わない（§3-1-bis・§8 限界 6）。"
                     "研究計画書 §9-18 により、走行水準の p は条件比較の p として解釈できない"),
               seed_representative="その seed の 20 走行の到達区画数の中央値",
               counts={}, per_seed_depths={}, p_values={})
    for name, meas in meas_by_name.items():
        depths = reach_depths(meas)
        out["per_seed_depths"][name] = depths
        out["counts"][name] = {f"ge{d}": count_reach_ge(meas, d) for d in DESCRIPTIVE_DEPTHS}
    names = list(meas_by_name)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a_d = out["per_seed_depths"][a_name]
            b_d = out["per_seed_depths"][b_name]
            a_seed = [statistics.median(v) for v in a_d.values()]
            b_seed = [statistics.median(v) for v in b_d.values()]
            a_run = [x for v in a_d.values() for x in v]
            b_run = [x for v in b_d.values() for x in v]
            out["p_values"][f"{a_name} 対 {b_name}"] = dict(
                seed_level=mannwhitney_exact(a_seed, b_seed),
                run_level=mannwhitney_normal(a_run, b_run),
                seed_representatives={a_name: a_seed, b_name: b_seed})
    return out


# ---------------------------------------------------------------------------
# 安全弁（**判定の前に落とす**）
# ---------------------------------------------------------------------------

def check_measurement(tag: str, meas: dict, *, recurrent: bool) -> list:
    msgs = []
    n = len(meas["detail"])
    if n != N_SEEDS:
        msgs.append(f"{tag}: seed 数が {N_SEEDS} でない（{n}）")
    for k, v in meas["detail"].items():
        if len(v["metrics"]) != N_MAZES:
            msgs.append(f"{tag}/{k}: 迷路数が {N_MAZES} でない（{len(v['metrics'])}）")
    if bool(meas.get("recurrent")) != recurrent:
        msgs.append(f"{tag}: recurrent の記録が {meas.get('recurrent')} "
                    f"（期待 {recurrent}）。測定のフラグを取り違えている")
    if meas.get("history_sham"):
        msgs.append(f"{tag}: にせ履歴（history_sham）が入っている")
    if list(meas.get("history_lags") or []) != EXPECTED_LAGS:
        msgs.append(f"{tag}: 遅れが {meas.get('history_lags')}（期待 {EXPECTED_LAGS}）")
    steps = sorted({m["num_timesteps"] for m in meas["models"]})
    if len(steps) != 1:
        msgs.append(f"{tag}: 学習量が seed 間で揃っていない（{steps}）")
    # 要約と走行ごとの記録の整合（要約だけを信用しない）
    for field in ("net_progress_per_1000", "respawn_per_1000"):
        got = median_of_seed_medians(meas, field)
        want = meas["summary"]["across_seeds_median"][field]
        if abs(got - want) > 1e-9:
            msgs.append(f"{tag}: {field} が要約と再計算で食い違う（要約 {want} / 再計算 {got}）")
    return msgs


def check_anchors(control: dict, control_goal_rates: dict) -> list:
    """🔴 錨を測定出力から再計算して照合する（**後から差し替えられない形にする**）。

    **4 つの錨すべてを対象にする**（准教授 AUDIT_049 要記載 1）。
    **`goal_rate` だけを裸の定数のままにしない** — **「錨は再計算して照合する」という
    本スクリプト自身の原則が、4 つのうち 1 つで守られていない**状態を解消する。
    """
    msgs = []
    got_goal = statistics.median([v["goal_rate"] for v in control_goal_rates.values()])
    if abs(got_goal - ANCHORS["goal_rate"]) > ANCHOR_TOL:
        msgs.append(f"錨 goal_rate が事前登録と違う"
                    f"（登録 {ANCHORS['goal_rate']} / 実測 {got_goal}）")
    c = count_reach_ge(control, REACH_DEPTH)
    if c["total"] != ANCHORS["n_reach_ge7"]:
        msgs.append(f"錨 n_reach_ge{REACH_DEPTH} が事前登録と違う"
                    f"（登録 {ANCHORS['n_reach_ge7']} / 実測 {c['total']}）")
    if sorted(c["per_seed_counts"]) != sorted(ANCHORS["n_reach_ge7_per_seed"]):
        msgs.append(f"錨の seed ごとの内訳が事前登録と違う"
                    f"（登録 {ANCHORS['n_reach_ge7_per_seed']} / 実測 {sorted(c['per_seed_counts'])}）")
    for field in ("net_progress_per_1000", "respawn_per_1000"):
        got = median_of_seed_medians(control, field)
        if abs(got - ANCHORS[field]) > ANCHOR_TOL:
            msgs.append(f"錨 {field} が事前登録と違う（登録 {ANCHORS[field]} / 実測 {got}）")
    return msgs


#: 実行経路（**ここに差分があれば別版**。§8-6 のブラックリスト方式）
CODE_PATHS = ["mouse/", "common/", "assets/", "competition/",
              "experiments/exp_012_continuous_potential/train.py"]

#: 准教授が走行中に保全した投入時の argv（**群の識別の第 3 の根拠**。起動時刻つき）
ARGV_EVIDENCE = {"群 1": "verification/evidence/exp_023a_launch_argv.txt",
                 "群 2": "verification/evidence/exp_023b_launch_argv.txt"}


def check_same_version(log_dirs_all) -> tuple:
    """🔴 12 本の `run_summary.json` の `git_rev` を読み、**実行経路が同一版か**を git に確かめさせる。

    **群 2 は群 1 の完走後に同じファイルを読み直して起動する**ので、
    **走行中に実行コードが変更されると群 2 だけが違う版になる**（R6・R7 の「差は 1 変更」が壊れる）。
    **文書コミットによる `git_rev` の相違は許容し、実行経路に差分があれば落とす。**
    """
    import subprocess
    msgs, revs = [], {}
    for d in log_dirs_all:
        f = Path(d) / "run_summary.json"
        if not f.exists():
            msgs.append(f"{d}: run_summary.json が無い（完走していない）")
            continue
        revs[Path(d).name] = json.loads(f.read_text(encoding="utf-8")).get("git_rev")
    uniq = sorted({r for r in revs.values() if r})
    if len(uniq) > 1:
        base = uniq[0]
        for other in uniq[1:]:
            r = subprocess.run(["git", "diff", "--quiet", base, other, "--"] + CODE_PATHS,
                               capture_output=True)
            # 🔴 `git diff --quiet` の終了コード: 0 = 差分なし・1 = 差分あり・2 以上 = git 自体の失敗。
            # **失敗を「別版で走った」と混同しない**（准教授 AUDIT_049 要是正 3）。
            # 「群が別版で走った」は設計の前提が壊れたという重い主張なので、
            # git の一時的な失敗でそれを出すと実験そのものを疑って時間を失う。
            if r.returncode == 1:
                msgs.append(f"🔴 実行経路に差分がある: {base[:7]} 対 {other[:7]}"
                            f"（{' '.join(CODE_PATHS)}）＝ 群が別版で走っている")
            elif r.returncode > 1:
                msgs.append(f"⚠️ git diff の実行自体が失敗した（{base[:7]} 対 {other[:7]}・"
                            f"終了コード {r.returncode}）: {r.stderr.decode()[:200]}"
                            "／**版の食い違いが確認できていない**（別版だと断定してはいない）")
    return msgs, dict(git_rev_per_run=revs, unique_revs=uniq,
                      note=("git_rev の相違は文書コミットで起こりうるので許容し、"
                            "実行経路の差分だけを落とす基準にしている（§8-6 ブラックリスト方式）"))


def argv_evidence_status() -> dict:
    """准教授が保全した投入時の argv 記録の在否を出力に残す（**群の識別の第 3 の根拠**）。

    **`run_summary.json` に群の識別が無い**ため、根拠は 3 本立てになる:
    **(1) `train.log` の行（脆い）・(2) 実行経路の同一版検査・(3) 走行中に保全した argv**。
    """
    out = {}
    for tag, rel in ARGV_EVIDENCE.items():
        p = Path(rel)
        out[tag] = dict(path=rel, exists=p.exists(),
                        n_lines=(len(p.read_text(encoding="utf-8", errors="replace").splitlines())
                                 if p.exists() else 0))
    return out


def check_group_logs(tag: str, log_dirs, *, respawn_reset: bool) -> list:
    msgs = []
    if len(log_dirs) != N_SEEDS:
        msgs.append(f"{tag}: log ディレクトリが {N_SEEDS} 本でない（{len(log_dirs)}）")
    for d in log_dirs:
        f = group_flag_from_train_log(d)
        if f["respawn_reset"] is None:
            msgs.append(f"{tag}/{d}: train.log にリセットの記録行が無い")
        elif f["respawn_reset"] != respawn_reset:
            msgs.append(f"{tag}/{d}: リセットの記録が {f['respawn_reset']}（期待 {respawn_reset}）"
                        " ＝ 群を取り違えている")
        if not f["recurrent"]:
            msgs.append(f"{tag}/{d}: train.log に RecurrentPPO の記録が無い")
    return msgs


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="exp_023 の R1・R3〜R7 の機械的判定")
    p.add_argument("--control", required=True, help="対照群 exp_021 の最終方策の測定出力")
    p.add_argument("--group1", required=True, help="群 1（リセットしない）の測定出力")
    p.add_argument("--group2", required=True, help="群 2（リセットする）の測定出力")
    p.add_argument("--logs-g1", nargs="+", required=True, help="群 1 の log ディレクトリ 6 本")
    p.add_argument("--logs-g2", nargs="+", required=True, help="群 2 の log ディレクトリ 6 本")
    p.add_argument("--logs-control", nargs="+", required=True,
                   help="対照 exp_021 の log ディレクトリ 6 本（**R5 の錨の再計算に使う**）")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    control = json.loads(Path(args.control).read_text(encoding="utf-8"))
    g1 = json.loads(Path(args.group1).read_text(encoding="utf-8"))
    g2 = json.loads(Path(args.group2).read_text(encoding="utf-8"))

    bad = []
    bad += check_measurement("対照 exp_021", control, recurrent=False)
    bad += check_measurement("群 1", g1, recurrent=True)
    bad += check_measurement("群 2", g2, recurrent=True)
    steps = [sorted({m["num_timesteps"] for m in d["models"]})[0] for d in (control, g1, g2)]
    if len(set(steps)) != 1:
        bad.append(f"3 群の学習量が違う（対照/群1/群2 = {steps}）")
    goal_control = final_goal_rates(args.logs_control)
    goal_g1 = final_goal_rates(args.logs_g1)
    goal_g2 = final_goal_rates(args.logs_g2)
    bad += check_anchors(control, goal_control)
    bad += check_goal_rate_timepoints("対照 exp_021", goal_control, steps[0])
    bad += check_goal_rate_timepoints("群 1", goal_g1, steps[1])
    bad += check_goal_rate_timepoints("群 2", goal_g2, steps[2])
    bad += check_group_logs("群 1", args.logs_g1, respawn_reset=False)
    bad += check_group_logs("群 2", args.logs_g2, respawn_reset=True)
    ver_msgs, version_info = check_same_version(list(args.logs_g1) + list(args.logs_g2))
    bad += ver_msgs
    if bad:
        raise SystemExit("安全弁で停止:\n  " + "\n  ".join(bad))

    r1, r3, r4 = judge_r1(g1), judge_r3(g1), judge_r4(g1)
    r5 = judge_r5(goal_g1)
    r6, r7 = judge_r6(g1, g2), judge_r7(g1, g2)
    verdicts = [r1, r3, r4, r5, r6, r7]

    # §3-3 (iii)「どちらの群よりも外」を判定量ごとに計算する（教授裁定）
    def _med(m, f):
        return median_of_seed_medians(m, f)

    def _gr(g):
        return statistics.median([v["goal_rate"] for v in g.values()])

    bracket = bracket_check({
        f"n_reach_ge{REACH_DEPTH}": {"対照": count_reach_ge(control, REACH_DEPTH)["total"],
                                     "群 1": count_reach_ge(g1, REACH_DEPTH)["total"],
                                     "群 2": count_reach_ge(g2, REACH_DEPTH)["total"]},
        "net_progress_per_1000": {"対照": _med(control, "net_progress_per_1000"),
                                  "群 1": _med(g1, "net_progress_per_1000"),
                                  "群 2": _med(g2, "net_progress_per_1000")},
        "respawn_per_1000": {"対照": _med(control, "respawn_per_1000"),
                             "群 1": _med(g1, "respawn_per_1000"),
                             "群 2": _med(g2, "respawn_per_1000")},
        "goal_rate": {"対照": _gr(goal_control), "群 1": _gr(goal_g1), "群 2": _gr(goal_g2)},
    })

    out = dict(
        clause="experiments/exp_023_recurrent_policy/card.md §3",
        num_timesteps=steps[0], history_lags=g1["history_lags"],
        anchors=ANCHORS,
        inputs=dict(control=str(args.control), group1=str(args.group1), group2=str(args.group2),
                    logs_g1=[str(x) for x in args.logs_g1],
                    logs_g2=[str(x) for x in args.logs_g2],
                    logs_control=[str(x) for x in args.logs_control]),
        verdicts=verdicts,
        version_check=version_info,
        group_identification=dict(
            note=("run_summary.json に群の識別（respawn_reset 等）が無いため、根拠は 3 本立てである: "
                  "(1) train.log の行〔脆い〕・(2) 実行経路の同一版検査・(3) 走行中に保全した argv"),
            argv_evidence=argv_evidence_status()),
        bracket_check=bracket,
        coverage=read_coverage(r1, r6, bracket),
        descriptive_former_r2=describe_reach({"対照 exp_021": control, "群 1": g1, "群 2": g2}),
        goal_rate_detail=dict(control=goal_control, group1=goal_g1, group2=goal_g2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"学習量: 3 群とも {steps[0]:,} 歩 / 遅れ {g1['history_lags']}")
    print("=" * 78)
    for v in verdicts:
        mark = "✅ 当たり" if v["hit"] else "❌ 外れ"
        ref = v.get("threshold", v.get("reference"))
        print(f"  {mark}  {v['name']}: {v['value']}（基準 {ref} / 対照 {v.get('control', '—')}）")
        print(f"        条文: {v['clause']}")
        if "per_seed" in v:
            print(f"        seed ごと: {list(v['per_seed'].values())}")
        if "per_seed_g1" in v:
            print(f"        seed ごと 群1: {list(v['per_seed_g1'].values())} / "
                  f"群2: {list(v['per_seed_g2'].values())}")
        if v.get("warning"):
            print(f"        ⚠️ {v['warning']}: {v['warning_detail']}")
    print("=" * 78)
    if bracket["any_flag"]:
        print("🔴 §3-3 (iii)「対照より悪い側へ出た」が成立:")
        for name, d in bracket["per_quantity"].items():
            if d["flag"]:
                who = [k for k, v in d["worse_than_control"].items() if v]
                print(f"    {name}: {d['values']} → {'・'.join(who)} が対照より悪い側")
    cov = out["coverage"]
    if cov["table_returned"]:
        print(f"§3-4 全域被覆: R1 {'当' if cov['r1_hit'] else '外'} × "
              f"R6 {'当' if cov['r6_hit'] else '外'} → {cov['reading']}")
        print(f"  次の一手: {cov['next_move']}")
    else:
        print(f"§3-4 全域被覆: **表を引かない** — {cov['reason']}")
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
