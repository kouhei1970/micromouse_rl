#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
課題 J: M5 確定基準表（新 20 値・安全率 0.75）の生データからの独立再計算
Task J: independent recomputation of the M5 reference table (new 20 values, safety factor 0.75)

准教授セッション（8 代目）・2026-08-14
Associate professor session (8th), 2026-08-14

背景 / Background:
  docs/RESEARCH_PLAN.md §5 の確定基準表（現行・2026-08-14 再導出）には
  「新表の生データからの独立再計算は未実施」と明記されている。
  旧表（安全率 0.70）には verification/REPORT_003 r2 による 20/20 一致の記録がある。
  本スクリプトはその欠を埋めるための検算を行う。

  The current M5 reference table in docs/RESEARCH_PLAN.md §5 carries the note
  "independent recomputation from raw data not yet performed". The old table
  (safety factor 0.70) has such a record (REPORT_003 r2, 20/20 agreement).

検査の層 / Layers of the check:
  L1  出所の照合           — SHA-256 と版管理の状態
  L2  抽出の照合           — 生データ rows[].fast_time → §5 の表 20 値
  L3  要約統計の再計算     — 中央値・完走率などを rows から独立に計算
  L4  内部整合性           — summary の全フィールドを rows から再導出
  L5  迷路集合の照合       — 20 迷路が確保済みの評価用迷路と一致するか
  L6  計時分解能           — 全タイムが 10 ms 刻みに乗っているか
  L7  旧表との差の再計算   — 「−2.66%・20/20 改善・範囲 −2.89〜−2.03%」の検算

  L1-L7 are all performed on the aggregate file. Recomputing fast_time itself
  from trajectories (as REPORT_003 r2 did) is NOT possible from the surviving
  artifacts — see the report for this limitation.
"""

import hashlib
import json
import os
import statistics
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 生データ / raw data (measurement M-1)
M1_JSON = os.path.join(REPO, "outputs/exp_016_diagonal/016cal_switch/m1_plain_l0c_sf0.75.json")
M1_SHA256_DOCUMENTED = "6a03f34b"  # 研究計画書 §5 / card_016cal_switch.md :246 の記載（先頭 8 桁）

# 旧表の生データ / raw data behind the superseded table (safety factor 0.70)
OLD_JSON = os.path.join(REPO, "outputs/exp_013_band_v4_reeval/l0c/runs_detail.json")

EVAL_MAZE_DIR = os.path.join(REPO, "competition/mazes/eval")

# docs/RESEARCH_PLAN.md §5 の確定基準表（現行）を手で転記したもの。
# 転記そのものが誤りうるので、行番号を添えて出所を明示する。
# Transcribed by hand from docs/RESEARCH_PLAN.md:164-170 (current table).
TABLE_NEW = {
    "maze_1018": 18.58, "maze_1019": 18.78, "maze_1021": 11.54, "maze_1023": 18.18,
    "maze_1024": 10.97, "maze_1041": 15.14, "maze_1046": 9.65, "maze_1073": 19.82,
    "maze_1103": 16.06, "maze_1126": 11.33,
    "maze_1138": 12.68, "maze_1161": 11.14, "maze_1240": 15.17, "maze_14037": 17.34,
    "maze_1804": 14.20, "maze_2304": 14.24, "maze_3707": 13.45, "maze_5188": 20.72,
    "maze_7837": 21.68, "maze_8209": 11.20,
}
TABLE_NEW_MEDIAN = 14.690  # docs/RESEARCH_PLAN.md:162

# docs/RESEARCH_PLAN.md:174-180 の旧表（安全率 0.70・失効・記録として残置）
TABLE_OLD = {
    "maze_1018": 19.10, "maze_1019": 19.25, "maze_1021": 11.86, "maze_1023": 18.64,
    "maze_1024": 11.27, "maze_1041": 15.55, "maze_1046": 9.92, "maze_1073": 20.41,
    "maze_1103": 16.47, "maze_1126": 11.64,
    "maze_1138": 13.00, "maze_1161": 11.41, "maze_1240": 15.60, "maze_14037": 17.82,
    "maze_1804": 14.60, "maze_2304": 14.60, "maze_3707": 13.82, "maze_5188": 21.15,
    "maze_7837": 22.29, "maze_8209": 11.47,
}
TABLE_OLD_MEDIAN = 15.075  # docs/RESEARCH_PLAN.md:172

# §5 が主張する旧表比の変化 / claimed change vs. the old table (RESEARCH_PLAN.md:162)
CLAIM_OVERALL_PCT = -2.66
CLAIM_RANGE_PCT = (-2.89, -2.03)
CLAIM_N_IMPROVED = 20

TOL = 5e-3  # 表は小数第 2 位まで。丸め幅の半分 = 5e-3 を一致の許容差とする

results = []  # (層, 項目, 判定, 詳細)


def record(layer, item, ok, detail):
    results.append((layer, item, ok, detail))
    mark = "PASS" if ok is True else ("FAIL" if ok is False else "INFO")
    print(f"  [{mark}] {item}: {detail}")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    try:
        out = subprocess.run(["git"] + list(args), cwd=REPO,
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip(), out.returncode
    except Exception as e:  # pragma: no cover
        return f"<error: {e}>", 1


# ---------------------------------------------------------------------------
# L1 出所の照合 / provenance
# ---------------------------------------------------------------------------
print("\n=== L1 出所の照合 / provenance ===")

if not os.path.exists(M1_JSON):
    print(f"生データが見つからない: {M1_JSON}")
    sys.exit(2)

digest = sha256_of(M1_JSON)
record("L1", "M-1 生データの SHA-256",
       digest.startswith(M1_SHA256_DOCUMENTED),
       f"実測 {digest[:16]}… / 文書記載 {M1_SHA256_DOCUMENTED}…")

# 版管理下にあるか（旧表は 727d0f7 で意図的に版管理下へ入れられている）
_, rc_tracked = git("ls-files", "--error-unmatch", os.path.relpath(M1_JSON, REPO))
ignored, rc_ign = git("check-ignore", "-v", os.path.relpath(M1_JSON, REPO))
record("L1", "M-1 生データが版管理下にあるか",
       rc_tracked == 0,
       "追跡されている" if rc_tracked == 0
       else f"🔴 追跡されていない（無視規則: {ignored or 'なし'}）")

_, rc_old_tracked = git("ls-files", "--error-unmatch", os.path.relpath(OLD_JSON, REPO))
record("L1", "（対照）旧表の生データが版管理下にあるか",
       rc_old_tracked == 0,
       "追跡されている" if rc_old_tracked == 0 else "追跡されていない")

# 軌跡データの有無（REPORT_003 r2 はここから再計算した）
m1_traj = os.path.join(os.path.dirname(M1_JSON), "traj")
old_traj = os.path.join(os.path.dirname(OLD_JSON), "traj")
n_m1_traj = len(os.listdir(m1_traj)) if os.path.isdir(m1_traj) else 0
n_old_traj = len(os.listdir(old_traj)) if os.path.isdir(old_traj) else 0
record("L1", "M-1 の軌跡データ（fast_time の再計算に要る）",
       n_m1_traj > 0,
       f"M-1: {n_m1_traj} 本 / （対照）旧表: {n_old_traj} 本")

# ---------------------------------------------------------------------------
# データ読み込み / load
# ---------------------------------------------------------------------------
with open(M1_JSON) as f:
    m1 = json.load(f)
rows = {r["maze"]: r for r in m1["rows"]}
summ = m1["summary"]

print(f"\n読み込み: {len(rows)} 迷路 / 方策 {m1['policy']} / 迷路ディレクトリ {m1['maze_dir']}")
print(f"          安全率 {summ['safety_factor']}")

# ---------------------------------------------------------------------------
# L2 抽出の照合 / extraction
# ---------------------------------------------------------------------------
print("\n=== L2 抽出の照合: 生データ rows[].fast_time → §5 の表 20 値 ===")
print("  迷路          生データ      §5 の表    差        判定")

n_match = 0
for maze in sorted(TABLE_NEW, key=lambda m: int(m.split("_")[1])):
    if maze not in rows:
        record("L2", maze, False, "生データに該当迷路が無い")
        continue
    got = rows[maze]["fast_time"]
    want = TABLE_NEW[maze]
    diff = got - want
    ok = abs(diff) <= TOL
    n_match += ok
    print(f"  {maze:<13} {got:>10.4f}  {want:>9.2f}  {diff:>+9.2e}  {'一致' if ok else '🔴 不一致'}")

record("L2", "20 値の一致数", n_match == 20, f"{n_match}/20 一致")

missing = set(rows) - set(TABLE_NEW)
extra = set(TABLE_NEW) - set(rows)
record("L2", "迷路集合の一致", not missing and not extra,
       "生データと表の迷路集合は同一" if not missing and not extra
       else f"生データのみ {sorted(missing)} / 表のみ {sorted(extra)}")

# ---------------------------------------------------------------------------
# L3 要約統計の再計算 / summary statistics recomputed from rows
# ---------------------------------------------------------------------------
print("\n=== L3 要約統計の再計算（rows から独立に計算） ===")

fast = [r["fast_time"] for r in m1["rows"] if r.get("fast_run_done")]
fast_sorted = sorted(fast)
med = statistics.median(fast)

record("L3", "中央値の再計算", abs(med - TABLE_NEW_MEDIAN) <= TOL,
       f"再計算 {med:.6f} s / 文書記載 {TABLE_NEW_MEDIAN} s（差 {med - TABLE_NEW_MEDIAN:+.2e}）")

# n=20 なので中央値は 10 番目と 11 番目の平均。手計算でも示す。
lo, hi = fast_sorted[9], fast_sorted[10]
record("L3", "中央値の手順（n=20 → 10・11 番目の平均）", abs((lo + hi) / 2 - med) <= 1e-12,
       f"({lo:.2f} + {hi:.2f}) / 2 = {(lo + hi) / 2:.4f}")

record("L3", "最速・最遅", None,
       f"最速 {fast_sorted[0]:.2f} s ({[k for k in rows if rows[k]['fast_time'] == fast_sorted[0]][0]}) / "
       f"最遅 {fast_sorted[-1]:.2f} s ({[k for k in rows if rows[k]['fast_time'] == fast_sorted[-1]][0]})")

# ---------------------------------------------------------------------------
# L4 内部整合性 / summary fields re-derived from rows
# ---------------------------------------------------------------------------
print("\n=== L4 内部整合性: summary の全フィールドを rows から再導出 ===")

n_completed = sum(1 for r in m1["rows"] if r["completed"])
record("L4", "n_completed", n_completed == summ["n_completed"],
       f"再計算 {n_completed} / 記載 {summ['n_completed']}")

comp_rate = n_completed / len(m1["rows"])
record("L4", "completion_rate", abs(comp_rate - summ["completion_rate"]) <= 1e-12,
       f"再計算 {comp_rate:.6f} / 記載 {summ['completion_rate']:.6f}")

n_fast = sum(1 for r in m1["rows"] if r.get("fast_run_done"))
record("L4", "n_fast", n_fast == summ["n_fast"],
       f"再計算 {n_fast} / 記載 {summ['n_fast']}")

ey_max = [r["e_y_max_m"] for r in m1["rows"] if r.get("e_y_max_m") is not None]
record("L4", "e_y_max_median_m",
       abs(statistics.median(ey_max) - summ["e_y_max_median_m"]) <= 1e-12,
       f"再計算 {statistics.median(ey_max):.9f} / 記載 {summ['e_y_max_median_m']:.9f}")
record("L4", "e_y_max_max_m", abs(max(ey_max) - summ["e_y_max_max_m"]) <= 1e-12,
       f"再計算 {max(ey_max):.9f} / 記載 {summ['e_y_max_max_m']:.9f}")

ey_rms = [r["e_y_rms_m"] for r in m1["rows"] if r.get("e_y_rms_m") is not None]
record("L4", "e_y_rms_median_m",
       abs(statistics.median(ey_rms) - summ["e_y_rms_median_m"]) <= 1e-12,
       f"再計算 {statistics.median(ey_rms):.9f} / 記載 {summ['e_y_rms_median_m']:.9f}")

# completion_lower95: 二項比率の片側 95% 下限。
# 実装は Wilson スコア法（experiments/exp_016_diagonal/run_016cal.py:157 の wilson_lower）。
# 私は当初 Clopper-Pearson（正確法）だと思い込んで不一致を出した — 実装を読んで是正した。
# 実装は z=1.645（丸め値）を既定値にしている（run_016cal.py:76）。
# 厳密な 0.95 分位点 1.6448536… を使うと下限が 1.9e-5 ずれるので、実装に合わせる。
z = 1.645
n_tot, k = len(m1["rows"]), n_completed
wilson_center = (k + z * z / 2) / (n_tot + z * z)
wilson_half = z / (n_tot + z * z) * (k * (n_tot - k) / n_tot + z * z / 4) ** 0.5
wilson_lower = wilson_center - wilson_half
record("L4", "completion_lower95（Wilson 片側・実装と同じ式）",
       abs(wilson_lower - summ["completion_lower95"]) <= 1e-9,
       f"再計算 {wilson_lower:.10f} / 記載 {summ['completion_lower95']:.10f}")

# 参考: 正確法（Clopper-Pearson）だと同じデータで下限はいくつになるか。
# k=n（全数成功）のとき Wilson は正確法より高い値を出す = 甘い側に外れる。
cp_lower = 0.05 ** (1.0 / n_tot) if k == n_tot else None
record("L4", "（参考）正確法との差", None,
       f"Wilson {wilson_lower:.4f} 対 Clopper-Pearson {cp_lower:.4f}"
       f"（差 {wilson_lower - cp_lower:+.4f}。全数成功では Wilson が甘い側）"
       if cp_lower is not None else "全数成功ではないため省略")

n_broken = sum(r["n_broken"] for r in m1["rows"])
record("L4", "破損走行の総数", n_broken == 0, f"{n_broken} 件")
record("L4", "failed_faces / failure_kinds",
       summ["failed_faces"] == [] and summ["failure_kinds"] == [],
       f"{summ['failed_faces']} / {summ['failure_kinds']}")

# 全走行の結末ラベル / outcome labels across all runs
all_out = [o for r in m1["rows"] for o in r["outcomes"]]
from collections import Counter
record("L4", "全走行の結末ラベル", None,
       f"{len(all_out)} 走行 = {dict(Counter(all_out))}")

# ---------------------------------------------------------------------------
# L5 迷路集合の照合 / eval maze set
# ---------------------------------------------------------------------------
print("\n=== L5 迷路集合の照合: 確保済みの評価用 20 迷路と一致するか ===")

if os.path.isdir(EVAL_MAZE_DIR):
    on_disk = sorted(f[:-4] for f in os.listdir(EVAL_MAZE_DIR) if f.endswith(".npz"))
    record("L5", "評価用迷路ディレクトリの中身", len(on_disk) == 20,
           f"{EVAL_MAZE_DIR} に {len(on_disk)} 件")
    record("L5", "測定した 20 迷路 = 評価用迷路の全数",
           set(on_disk) == set(rows),
           "完全一致" if set(on_disk) == set(rows)
           else f"ディスクのみ {sorted(set(on_disk) - set(rows))} / 測定のみ {sorted(set(rows) - set(on_disk))}")
else:
    record("L5", "評価用迷路ディレクトリ", False, f"{EVAL_MAZE_DIR} が無い")

# ---------------------------------------------------------------------------
# L6 計時分解能 / timing quantisation
# ---------------------------------------------------------------------------
print("\n=== L6 計時分解能: 全タイムが 10 ms 刻みに乗っているか ===")
# 研究計画書 §5 の限界「計時分解能 ±10 ms」と整合するかの検査。
off_grid = []
for r in m1["rows"]:
    for key in ("fast_time", "explore_time"):
        v = r.get(key)
        if v is None:
            continue
        rem = abs(v * 100 - round(v * 100))
        if rem > 1e-6:
            off_grid.append((r["maze"], key, v, rem))
record("L6", "10 ms 格子への乗り", not off_grid,
       f"40 値すべて 10 ms の倍数（最大ずれ {max((o[3] for o in off_grid), default=0.0):.2e}）"
       if not off_grid else f"🔴 格子外 {len(off_grid)} 件: {off_grid[:3]}")

# ---------------------------------------------------------------------------
# L7 旧表との差の再計算 / claimed change vs. old table
# ---------------------------------------------------------------------------
print("\n=== L7 旧表との差の再計算（§5 の主張「−2.66%・20/20 改善・−2.89〜−2.03%」） ===")

# 旧表の値は生データから引き直す（表の転記を信用しない）。
# 旧ファイルは走行ごとの明細（runs = 20 迷路 × 5 走行）なので、
# §2 の定義「(d) 最速タイム = 完走走行の最速値」をそのまま適用して迷路ごとに算出する。
# ここは M-1 側と違い、集約値ではなく一次データからの真の再計算になっている。
old_from_raw, old_runs_by_maze = {}, {}
if os.path.exists(OLD_JSON):
    with open(OLD_JSON) as f:
        old_raw = json.load(f)
    for r in old_raw.get("runs", []):
        old_runs_by_maze.setdefault(r["maze"], []).append(r)
    for maze, rs in old_runs_by_maze.items():
        done = [r["run_time"] for r in rs if r["outcome"] == "goal"]
        if done:
            old_from_raw[maze] = min(done)
    record("L7", "旧生データの構造", len(old_runs_by_maze) == 20,
           f"{len(old_raw.get('runs', []))} 走行 = {len(old_runs_by_maze)} 迷路 × "
           f"{len(next(iter(old_runs_by_maze.values())))} 走行（走行ごとのタイムあり）")

    n_ok = sum(1 for m, v in TABLE_OLD.items()
               if m in old_from_raw and abs(old_from_raw[m] - v) <= TOL)
    record("L7", "旧表 20 値を走行ごとの生データから再計算", n_ok == 20,
           f"{n_ok}/20 一致（REPORT_003 r2 の記録を独立に再現）")

    om = statistics.median(old_from_raw.values())
    record("L7", "旧表の中央値の再計算", abs(om - TABLE_OLD_MEDIAN) <= TOL,
           f"再計算 {om:.4f} s / 文書記載 {TABLE_OLD_MEDIAN} s")

    # (d) の定義の確認: 「完走走行の最速値」は探索走行も母集団に含むが、
    # 実際に最速を与えるのは最短走行なので、両者を分けても値が変わらないことを見る。
    same = 0
    for maze, rs in old_runs_by_maze.items():
        fastruns = [r["run_time"] for r in rs if r["outcome"] == "goal" and r["run"] >= 2]
        if fastruns and abs(min(fastruns) - old_from_raw[maze]) <= 1e-9:
            same += 1
    record("L7", "(d) の母集団（全完走走行 対 最短走行のみ）", same == 20,
           f"{same}/20 の迷路で同値 — 探索走行が最速を与えることはない")
else:
    record("L7", "旧表の生データ", False, f"{OLD_JSON} が無い")

# M-1 側に同じ再計算ができるか（できない ＝ 本課題の核心）
has_perrun_time = any("run_time" in r or "run_times" in r for r in m1["rows"])
record("L7", "M-1 側に走行ごとのタイムがあるか", has_perrun_time,
       "ある" if has_perrun_time
       else "🔴 無い（outcomes はラベルのみ・fast_time は集約済み）— "
            "旧表と同じ形の再計算が新表には適用できない")

base = old_from_raw if len(old_from_raw) == 20 else TABLE_OLD
pct = {}
for maze in rows:
    if maze in base:
        pct[maze] = (rows[maze]["fast_time"] - base[maze]) / base[maze] * 100.0

n_improved = sum(1 for p in pct.values() if p < 0)
record("L7", "改善した迷路の数", n_improved == CLAIM_N_IMPROVED,
       f"再計算 {n_improved}/{len(pct)} / 主張 {CLAIM_N_IMPROVED}/20")

pmin, pmax = min(pct.values()), max(pct.values())
range_ok = (abs(pmin - CLAIM_RANGE_PCT[0]) <= 0.05 and abs(pmax - CLAIM_RANGE_PCT[1]) <= 0.05)
record("L7", "迷路ごとの変化率の範囲", range_ok,
       f"再計算 {pmin:+.2f} 〜 {pmax:+.2f} % / 主張 {CLAIM_RANGE_PCT[0]:+.2f} 〜 {CLAIM_RANGE_PCT[1]:+.2f} %")

# 「−2.66%」がどの統計量かは §5 に明記が無い。候補を全部出して同定する。
med_pct = statistics.median(pct.values())
mean_pct = statistics.mean(pct.values())
med_ratio = (statistics.median([rows[m]["fast_time"] for m in pct])
             / statistics.median([base[m] for m in pct]) - 1) * 100
sum_ratio = (sum(rows[m]["fast_time"] for m in pct) / sum(base[m] for m in pct) - 1) * 100
print("\n  「−2.66%」の候補（§5 は統計量を明記していない）:")
print(f"    迷路ごとの変化率の中央値 : {med_pct:+.4f} %")
print(f"    迷路ごとの変化率の平均   : {mean_pct:+.4f} %")
print(f"    中央値どうしの比         : {med_ratio:+.4f} %")
print(f"    総和どうしの比           : {sum_ratio:+.4f} %")
cands = {"変化率の中央値": med_pct, "変化率の平均": mean_pct,
         "中央値どうしの比": med_ratio, "総和どうしの比": sum_ratio}
hit = [k for k, v in cands.items() if abs(v - CLAIM_OVERALL_PCT) <= 0.005]
record("L7", "「−2.66%」の再現", bool(hit),
       f"一致する統計量: {hit}" if hit
       else f"🔴 どの候補とも 0.005 pt 以内で一致しない（最近傍 "
            f"{min(cands, key=lambda k: abs(cands[k] - CLAIM_OVERALL_PCT))}"
            f" = {min(cands.values(), key=lambda v: abs(v - CLAIM_OVERALL_PCT)):+.4f} %）")

print("\n  迷路ごとの内訳:")
print("  迷路          旧(生)     新(生)    変化率")
for maze in sorted(pct, key=lambda m: pct[m]):
    print(f"  {maze:<13} {base[maze]:>7.2f}  {rows[maze]['fast_time']:>8.2f}  {pct[maze]:>+8.2f} %")

# ---------------------------------------------------------------------------
# 総括 / verdict
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("総括 / verdict")
print("=" * 72)
n_pass = sum(1 for _, _, ok, _ in results if ok is True)
n_fail = sum(1 for _, _, ok, _ in results if ok is False)
n_info = sum(1 for _, _, ok, _ in results if ok is None)
print(f"  PASS {n_pass} / FAIL {n_fail} / INFO {n_info}")
if n_fail:
    print("\n  🔴 不合格の項目:")
    for layer, item, ok, detail in results:
        if ok is False:
            print(f"    [{layer}] {item}: {detail}")
print("""
⚠️ 本スクリプトの限界（報告に必ず併記すること）:
  fast_time そのものを軌跡から再計算してはいない。M-1 の成果物には軌跡
  （traj/*.npz）が残っていないため、REPORT_003 r2 と同水準の独立再計算は
  現存物からは実行できない。L2 は「教授の抽出が正しいか」の検査であって、
  測定値そのものの独立検証ではない。
""")
sys.exit(1 if n_fail else 0)
