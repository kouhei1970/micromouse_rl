#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
016-H1e 事前登録（`PREREG_016h1e.md` §1・§2、2026-08-18 改訂版）の錨の独立再計算。

## なぜこのスクリプトが要るか

`PREREG_016h1e.md` §1 は「判定に使う対照値は、カードや過去の報告から引き写さない。
一次記録から自分で出し直す」と定めている。手本は `verification/audit_exp023_anchors.py`
（その冒頭が「exp_023 のデータを一切読まない」と明記しているのと同じ精神）。

**本スクリプトは `competition/` 配下のコードを import しない。**
`competition/baseline_slalom.py:649` 付近の定数定義行を**正規表現で目視に代えて読み取り**、
その値で模型（tau・L・傾き a）を**このスクリプト自身の中で**独立計算する。
`check_016h1e_model.py` の import はしない（あれを呼ぶとカード §1 の a=1.110 の出所と
同じ経路をなぞるだけになり、独立再計算にならない — 是正2件目）。

出す値（PREREG §1 の表・§2 の L1〜L2 に対応）:
  1. E_in(v)  = 各水準・全円弧（20 面連結）の入口 |e_y| 中央値 [mm]（面ごとの内訳つき）
     🔴 是正1: collided 面も含めた「当該水準の全円弧」を母集団とする（条文どおり）。
     collided な面数・弧数は必ず別途報告する。
  2. E_out(v) = 同・出口 |e_y| 中央値 [mm]（面ごとの内訳つき）
  3. r = (E_in(0.70)-E_in(0.45)) / (E_out(0.70)-E_out(0.45))
     → card_016h1d.md の「上乗せの 62%」との照合。
     🔴 是正3: 条文どおり |差|<=5e-4 でのみ 🟢。丸め一致は注記に格下げする。
  4. 模型値（tau, L, 傾き a）を baseline_slalom.py:649 から読み取った定数で
     このスクリプト内で独立計算し、PREREG §4 W1 の模型値 a=1.110 s と照合。
     🔴 是正2: check_016h1e_model.py は import しない（循環を切る）。
  5. 🔴 是正4: 進入直線区間の v_act の中央値（PREREG §1 の 4 行目、2026-08-18 追記）。
     `outputs/exp_016_diagonal/016h1e/healthy_sf0.75_v{level}.json` の
     `rows[].entry_straights[].v_act_med` から出す（円弧の arc_v_med でも指令値でもない）。
     このファイルが無い水準は「未測定」と明示し、エラーにせず完走する。
  6. 使った JSON の SHA-256・git_rev（PREREG §2 L1）

走行は一切しない — 一次記録の JSON を読むだけである。

    .venv/bin/python verification/anchor_016h1e_entry_ey.py
"""

import hashlib
import json
import os
import re
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "outputs", "exp_016_diagonal", "016h1")
H1E_OUT_DIR = os.path.join(REPO, "outputs", "exp_016_diagonal", "016h1e")
EVIDENCE_DIR = os.path.join(REPO, "verification", "evidence")
BASELINE_SLALOM_PATH = os.path.join(REPO, "competition", "baseline_slalom.py")

LEVELS = ("0.45", "0.50", "0.55", "0.60", "0.65", "0.70")
# ファイル名の水準表記（0.50 は "0.5" ファイル名）に注意 — 見た目のゼロ落ちに引きずられない。
LEVEL_FILE = {"0.45": "0.45", "0.50": "0.5", "0.55": "0.55",
              "0.60": "0.6", "0.65": "0.65", "0.70": "0.7"}

CARD_R_TEXT = 0.62          # card_016h1d.md 冒頭の記述「上乗せの 62%」
CARD_A_MODEL = 1.110        # PREREG_016h1e.md §4 W1 / カード §1 の模型値 [s]
TOL = 5e-4                  # PREREG §1 の照合規則（条文どおり。丸め一致では代替しない）

# 是正2: baseline_slalom.py:649 から読み取った定数が、模型が前提にしている
# 12.0/10.0/0.15 と食い違ったら大きく警告する基準値。
EXPECTED_K_PSI, EXPECTED_K_Y, EXPECTED_V_EPS = 12.0, 10.0, 0.15


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_gain_constants_from_source(path: str):
    """`competition/baseline_slalom.py` のソースを正規表現で読み、
    k_psi・k_y・v_eps の既定値を取り出す（import しない — 是正2）。

    対象行（649 行付近）:
        k_psi: float = 12.0, k_y: float = 10.0, v_eps: float = 0.15, v_creep: float = 0.15,
    """
    with open(path, encoding="utf-8") as f:
        src = f.read()
    pat = re.compile(
        r"k_psi\s*:\s*float\s*=\s*([0-9.eE+-]+)\s*,\s*"
        r"k_y\s*:\s*float\s*=\s*([0-9.eE+-]+)\s*,\s*"
        r"v_eps\s*:\s*float\s*=\s*([0-9.eE+-]+)"
    )
    m = pat.search(src)
    if not m:
        raise RuntimeError(
            f"{path} から k_psi/k_y/v_eps の定義行を正規表現で見つけられなかった。"
            "行の書式が変わっている可能性がある — 目視で確認すること。"
        )
    k_psi, k_y, v_eps = (float(m.group(i)) for i in (1, 2, 3))
    return k_psi, k_y, v_eps


def slow_time_constant(k_psi: float, wn2: float) -> float:
    """過減衰2次系の遅い極の時定数[s]（このスクリプト内での独立実装。是正2）。

    e_y'' + k_psi*e_y' + wn2*e_y = 0 の特性方程式の遅い根から出す。
    """
    disc = k_psi * k_psi - 4.0 * wn2
    if disc <= 0.0:
        raise ValueError(f"振動域に入っている（wn2={wn2:.4f}）。模型の前提が崩れている")
    return 2.0 / (k_psi - disc ** 0.5)


def decay_length(k_psi: float, k_y: float, v_eps: float, v: float):
    """速度 v での (wn^2, tau, L=v*tau) を独立計算する（健常＝分母 v+v_eps）。"""
    wn2 = v * k_y / (v + v_eps)
    tau = slow_time_constant(k_psi, wn2)
    return wn2, tau, v * tau


def linear_fit(xs, ys):
    """最小二乗の (傾き, 切片)。numpy に依存しない独立実装。"""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    return a, my - a * mx


def load_level(level: str):
    fname = f"d1d2_sf0.75_v{LEVEL_FILE[level]}.json"
    path = os.path.join(OUT_DIR, fname)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return doc, path


def gather_ey(doc, idx: int):
    """entry_exit[][idx] を面ごと・連結の両方で集める（是正1: collided も含める）。

    idx=0: 入口, 1: 出口。
    戻り値: (per_maze: [(maze_name, [mm...], collided_bool), ...], pooled_mm: [mm...],
             collided_mazes: [maze_name...])
    """
    per_maze = []
    pooled = []
    collided_mazes = []
    for row in doc["rows"]:
        if row["collided"]:
            collided_mazes.append(row["maze"])
        vals_mm = [arc[idx] * 1000.0 for arc in row["entry_exit"]]
        per_maze.append((row["maze"], vals_mm, bool(row["collided"])))
        pooled.extend(vals_mm)
    return per_maze, pooled, collided_mazes


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def print_per_maze_table(title, per_maze):
    print(f"    --- {title}（面ごとの内訳） ---")
    for name, vals, collided in per_maze:
        tag = " [collided]" if collided else ""
        print(f"      {name:<14} n_arc={len(vals):>3}  中央値={median(vals):8.4f} mm"
              f"  (min={min(vals):.4f}, max={max(vals):.4f}){tag}")


def load_v_act_median_for_level(level: str):
    """是正4: 進入直線区間の v_act 中央値を、016h1e の一次記録から独立に出す。

    ファイルが無ければ (None, None) を返す（未測定・エラーにしない）。
    """
    fname = f"healthy_sf0.75_v{LEVEL_FILE[level]}.json"
    path = os.path.join(H1E_OUT_DIR, fname)
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    v_meds = []
    n_excluded_short = 0
    for row in doc["rows"]:
        for seg in row.get("entry_straights", []):
            v_meds.append(float(seg["v_act_med"]))
            if seg.get("excluded_short"):
                n_excluded_short += 1
    if not v_meds:
        return None, path
    return {
        "median_v_act": median(v_meds),
        "n_segments": len(v_meds),
        "n_excluded_short": n_excluded_short,
        "path": path,
    }, path


def main() -> int:
    print("=" * 84)
    print("016-H1e 錨の独立再計算（2026-08-18 改訂条文対応） — 健常群一次記録のみを読む")
    print("competition/ 配下は import しない（定数は正規表現で読み取る）")
    print("=" * 84)

    docs = {}
    paths = {}
    sha = {}
    git_revs = set()

    print("\n--- L1: 出所（JSON パス・SHA-256・git_rev） ---")
    for lv in LEVELS:
        doc, path = load_level(lv)
        docs[lv] = doc
        rel = os.path.relpath(path, REPO)
        paths[lv] = rel
        digest = sha256_of(path)
        sha[lv] = digest
        git_revs.add(doc["git_rev"])
        print(f"  v={lv}: {rel}")
        print(f"          sha256={digest}")
        print(f"          git_rev={doc['git_rev']}  maze_dir={doc['maze_dir']}"
              f"  safety={doc['safety']}  v_diag={doc['v_diag']}")

    if len(git_revs) != 1:
        print(f"  🔴 6 水準の git_rev が揃っていない: {git_revs}")
    else:
        print(f"  → 6 水準とも git_rev 一致: {next(iter(git_revs))}")

    print("\n--- 是正1: collided の扱い（条文どおり・除外しない） ---")
    print("  PREREG §3-1: 「当該水準の全円弧（20 面連結）の中央値」— 衝突面の除外は条文にない。")
    print("  したがって E_in/E_out は collided 面も含めて連結する。")
    print("  collided 件数は水準ごとに必ず報告する（判定文書で言及できるように）。")

    e_in_pooled_median = {}
    e_out_pooled_median = {}
    collided_report = {}

    for kind, idx, store in (("E_in（入口）", 0, e_in_pooled_median),
                              ("E_out（出口）", 1, e_out_pooled_median)):
        print(f"\n=== {kind} ===")
        for lv in LEVELS:
            doc = docs[lv]
            per_maze, pooled, collided_mazes = gather_ey(doc, idx)
            n_total_mazes = len(doc["rows"])
            store[lv] = median(pooled)
            if kind.startswith("E_in"):
                collided_report[lv] = {
                    "n_collided": len(collided_mazes),
                    "n_total_mazes": n_total_mazes,
                    "collided_mazes": collided_mazes,
                }
            print(f"\n  v={lv}  collided={len(collided_mazes)}/{n_total_mazes}"
                  f"（{collided_mazes if collided_mazes else 'なし'}、除外せず含める）"
                  f"  連結弧数={len(pooled)}  連結中央値={store[lv]:.4f} mm")
            print_per_maze_table(f"v={lv} {kind}", per_maze)

    print("\n" + "=" * 84)
    print("--- E_in(v) まとめ [mm]（連結中央値、丸める前の値も併記。collided 込み） ---")
    for lv in LEVELS:
        c = collided_report[lv]
        print(f"  v={lv}: {e_in_pooled_median[lv]:.6f} mm  (小数第3位まで: "
              f"{round(e_in_pooled_median[lv], 3)} mm)  collided={c['n_collided']}件")

    print("\n--- E_out(v) まとめ [mm] ---")
    for lv in LEVELS:
        print(f"  v={lv}: {e_out_pooled_median[lv]:.6f} mm  (小数第3位まで: "
              f"{round(e_out_pooled_median[lv], 3)} mm)")

    total_collided = sum(c["n_collided"] for c in collided_report.values())
    print(f"\n【collided 総括】全水準合計 {total_collided} 面が衝突（0 件でも必ず表示）")
    for lv in LEVELS:
        c = collided_report[lv]
        print(f"  v={lv}: {c['n_collided']} 件  {c['collided_mazes']}")

    # --- 「上乗せの 62%」の独立再計算 ---
    print("\n" + "=" * 84)
    print("--- 第2段『上乗せの 62%』の独立再計算 ---")
    d_in = e_in_pooled_median["0.70"] - e_in_pooled_median["0.45"]
    d_out = e_out_pooled_median["0.70"] - e_out_pooled_median["0.45"]
    r = d_in / d_out if d_out != 0 else float("nan")
    print(f"  E_in(0.70)-E_in(0.45)   = {e_in_pooled_median['0.70']:.4f} - "
          f"{e_in_pooled_median['0.45']:.4f} = {d_in:.4f} mm")
    print(f"  E_out(0.70)-E_out(0.45) = {e_out_pooled_median['0.70']:.4f} - "
          f"{e_out_pooled_median['0.45']:.4f} = {d_out:.4f} mm")
    print(f"  r = {d_in:.4f} / {d_out:.4f} = {r:.6f}  ({r * 100:.4f} %)")

    diff_exact = abs(r - CARD_R_TEXT)
    match_exact = diff_exact <= TOL
    print(f"\n  カード記載（card_016h1d.md 冒頭の文章）: 「62%」= {CARD_R_TEXT}")
    print(f"  是正3: 条文どおりの照合（|差| <= {TOL}）: 差={diff_exact:.6f}  "
          f"→ {'🟢 一致' if match_exact else '🔴 不一致'}")
    round_match = round(r * 100) == 62
    print(f"  【注記・判定には使わない】round(r*100)={round(r * 100)} % vs カード 62% → "
          f"{'丸めては一致' if round_match else '丸めても不一致'}")
    if not match_exact:
        print("  🔴 不一致の原因: card_016h1d.md はパーセントを整数（有効数字2桁）で")
        print("        しか記していない（「62%」）ため、厳密な小数値との比較では")
        print(f"        差 {diff_exact:.4f}（={diff_exact*100:.2f} 個の百分率ポイント相当）が生じる。")
        print("        これは記載の精度不足によるものであり、再計算の誤りではないと考えられる。")
        print("        条文（PREREG §1）は不一致を丸めて救うことを認めていないため、🔴 のまま表に出す。")

    # --- 模型値の独立計算（是正2: check_016h1e_model.py を import しない） ---
    print("\n" + "=" * 84)
    print("--- 是正2: 模型値の独立計算（baseline_slalom.py:649 を正規表現で読み取り、")
    print("    このスクリプト内で計算。check_016h1e_model.py は import しない） ---")
    k_psi, k_y, v_eps = read_gain_constants_from_source(BASELINE_SLALOM_PATH)
    print(f"  読み取った定数: k_psi={k_psi}, k_y={k_y}, v_eps={v_eps}"
          f"（{os.path.relpath(BASELINE_SLALOM_PATH, REPO)} より正規表現で抽出）")
    if (k_psi, k_y, v_eps) != (EXPECTED_K_PSI, EXPECTED_K_Y, EXPECTED_V_EPS):
        print(f"  🔴🔴🔴 警告: 読み取った定数がカード §1 の前提"
              f"（k_psi={EXPECTED_K_PSI}, k_y={EXPECTED_K_Y}, v_eps={EXPECTED_V_EPS}）"
              f"と食い違っている！ soft-code が変わった可能性が高い。判定を止めて確認すること。")
    else:
        print(f"  → カード §1 の前提と一致（k_psi=12.0, k_y=10.0, v_eps=0.15）")

    level_floats = [float(lv) for lv in LEVELS]
    lengths_healthy = []
    print(f"\n  {'v[m/s]':>7} {'wn^2':>9} {'tau[s]':>8} {'L=v*tau[m]':>11}")
    for v in level_floats:
        wn2, tau, length = decay_length(k_psi, k_y, v_eps, v)
        lengths_healthy.append(length)
        print(f"  {v:7.2f} {wn2:9.3f} {tau:8.3f} {length:11.4f}")
    a_model, c_model = linear_fit(level_floats, lengths_healthy)
    print(f"\n  L(v) 健常回帰（このスクリプト内で独立計算）: 傾き a = {a_model:.4f} s"
          f"  切片 c = {c_model:.4f} m")
    diff_a = abs(a_model - CARD_A_MODEL)
    print(f"  カード §1 / PREREG §4 W1 の模型値記載: a = {CARD_A_MODEL:.3f} s")
    print(f"  |差| = {diff_a:.6f}  → "
          f"{'🟢 一致（5e-4 以内）' if diff_a <= TOL else '🔴 不一致（5e-4 を超過）'}")

    # --- 是正4: 実測速度 v_act の中央値（進入直線区間） ---
    print("\n" + "=" * 84)
    print("--- 是正4: 実測速度の錨（進入直線区間の v_act 中央値、指令値でも arc_v_med でもない） ---")
    print(f"  出所: {os.path.relpath(H1E_OUT_DIR, REPO)}/healthy_sf0.75_v{{level}}.json")
    v_act_report = {}
    any_measured = False
    for lv in LEVELS:
        result, path = load_v_act_median_for_level(lv)
        if result is None:
            status = "（ファイルなし）" if path is None else "（ファイルはあるが進入直線区間が空）"
            print(f"  v={lv}: 未測定 {status}"
                  f"{'' if path is None else f' — {os.path.relpath(path, REPO)}'}")
            v_act_report[lv] = None
        else:
            any_measured = True
            print(f"  v={lv}: v_act 中央値={result['median_v_act']:.4f} m/s"
                  f"（進入直線 {result['n_segments']} 区間連結、うち Λ<0.10m 除外対象 "
                  f"{result['n_excluded_short']} 区間・出所={os.path.relpath(result['path'], REPO)}）")
            v_act_report[lv] = result
    if not any_measured:
        print("\n  → 6 水準とも未測定。outputs/exp_016_diagonal/016h1e/ 配下の走行が")
        print("    まだ無いため（教授セッションが走行後に本スクリプトを再実行すること）。")
        print("    エラーにはせず、ここまでの錨（E_in/E_out/r/模型）は通常どおり出力済み。")

    # --- 結果を JSON でも保存（機械可読）---
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    evidence_path = os.path.join(EVIDENCE_DIR, "anchor_016h1e_entry_ey.json")
    payload = {
        "sources": {lv: {"path": paths[lv], "sha256": sha[lv], "git_rev": docs[lv]["git_rev"]}
                    for lv in LEVELS},
        "collided": collided_report,
        "E_in_mm": {lv: e_in_pooled_median[lv] for lv in LEVELS},
        "E_out_mm": {lv: e_out_pooled_median[lv] for lv in LEVELS},
        "r_second_stage": r,
        "card_r_text": CARD_R_TEXT,
        "r_match_exact_5e4": match_exact,
        "r_round_match_only_note": round_match,
        "gain_constants_read": {"k_psi": k_psi, "k_y": k_y, "v_eps": v_eps},
        "gain_constants_expected": {"k_psi": EXPECTED_K_PSI, "k_y": EXPECTED_K_Y,
                                    "v_eps": EXPECTED_V_EPS},
        "model": {"a_s": a_model, "c_m": c_model, "card_a_s": CARD_A_MODEL,
                 "a_match_5e4": diff_a <= TOL},
        "v_act_entry_straight": {
            lv: (None if v_act_report[lv] is None else {
                "median_v_act": v_act_report[lv]["median_v_act"],
                "n_segments": v_act_report[lv]["n_segments"],
                "n_excluded_short": v_act_report[lv]["n_excluded_short"],
                "path": os.path.relpath(v_act_report[lv]["path"], REPO),
            })
            for lv in LEVELS
        },
    }
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n機械可読な結果を書いた: {os.path.relpath(evidence_path, REPO)}")

    print("\n" + "=" * 84)
    print("この値を PREREG_016h1e.md §1 の錨として使う。")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
