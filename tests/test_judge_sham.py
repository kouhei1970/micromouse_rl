#!/usr/bin/env python3
"""experiments/exp_022_sham_history/judge_sham.py の単体テスト。

judge_sham.py は exp_022（にせ履歴群）の P1〜P4（`card.md` §3 の条文）を
機械的に判定するスクリプト。**このテストは配管（条文→コードへの落とし込み）だけを
検査する。実験の当たり外れそのものは検査対象ではない。**

🔴 最重要方針: **合成データ（自分で作った模擬の測定出力）だけを使う。**
`outputs/exp_022_driving_sham_final.json` などの本物の測定出力や
`logs/exp_022_seed*/` の本物のログは一切読まない。本物を入れると実験の判定の帰結
（(A/C)説／(B)説のどちらが当たったか）が出てしまい、「配管の検査」という本テストの
存在理由（教授裁定 2026-08-14）が崩れる。tests/test_judge.py と同じ方針・書式に倣う。

| # | 検査 | judge_sham.py 対応 |
|---|---|---|
| T1 | r = ln(sham/control)/ln(treat/control) の定義（r=0・r=1・r=0.5 の3点、P1 の錨） | _relative_position() |
| T2 | 向きの自動処理（P4 は treat<control でも式が向きを扱う） | _relative_position() |
| T3 | 境界: r がちょうど 0.5 のとき (B) 側（P1 の境界 1.031） | judge_one() |
| T4 | 3 つ目の読み: r<0（対照群より外側）と r>1（参照群より外側）で verdict="outside" | judge_one() |
| T5 | P3 の別扱い: ゴール件数 0→(A/C)寄り・1件以上→(B)寄り、r は None のまま | judge_p3() |
| T6 | 錨の照合: _check_anchors が事前登録との食い違いを検出し、一致時は空を返す | _check_anchors() |
| T7 | main() の安全弁 6 条件が非ゼロ終了コードで落ちること | main() の前提検査 |
| T8 | 空振り防止の自己検査（合成データを壊すと P1 の判定が反転すること） | judge_one() |

pytest は使わない plain Python スクリプト（tests/test_judge.py と同じ流儀）。
実行方法（リポジトリルートで）:
    .venv/bin/python tests/test_judge_sham.py
"""
import copy
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

JUDGE_SHAM_PY = REPO_ROOT / "experiments/exp_022_sham_history/judge_sham.py"
PYTHON = str(REPO_ROOT / ".venv/bin/python")


def load_judge_sham_module():
    """judge_sham.py は experiments/ 配下で __init__.py が無い（パッケージでない）ため、
    tests/test_judge.py と同じ流儀でファイルパスから直接読み込む。"""
    spec = importlib.util.spec_from_file_location("exp022_judge_sham", JUDGE_SHAM_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_judge = load_judge_sham_module()
_relative_position = _judge._relative_position
judge_one = _judge.judge_one
judge_p3 = _judge.judge_p3
_check_anchors = _judge._check_anchors
ANCHORS = _judge.ANCHORS
P3 = _judge.P3
R_BOUNDARY = _judge.R_BOUNDARY

P1, P2, P4 = ANCHORS["P1"], ANCHORS["P2"], ANCHORS["P4"]


# ======================================================================
# 合成データの組み立てヘルパ（measure_driving.py の出力の形を最小限で真似る）
# ======================================================================
def median_summary(**fields) -> dict:
    """judge_one が読む最小構成（summary.across_seeds_median だけを埋める）。"""
    return {"across_seeds_median": dict(fields)}


LAGS = [1, 2, 4, 8, 16, 32, 64, 128]  # にせ履歴群・参照群の遅れ（card.md の既定）


def anchor_summary(role: str) -> dict:
    """role='control' または 'treat'。ANCHORS・P3 の定数から、_check_anchors が
    「一致」と判定する summary（本物の measure_driving.py 出力と同じ形）を作る。"""
    return {
        "across_seeds_median": {
            P1["field"]: P1[role],
            P4["field"]: P4[role],
        },
        P2["field"]: P2[role],
        P3["field"]: P3[role],
    }


def full_summary(respawn, net_progress, n_reach_ge5, n_goal_rollout, n_seeds=6) -> dict:
    """main() の安全弁テスト用の完全な summary（per_seed_median の seed 数チェックも通す）。"""
    return {
        "per_seed_median": {f"seed{i}": {} for i in range(n_seeds)},
        "across_seeds_median": {P1["field"]: respawn, P4["field"]: net_progress},
        P2["field"]: n_reach_ge5,
        P3["field"]: n_goal_rollout,
        "n_runs": n_seeds * 20,
    }


def full_group(num_timesteps, lags, history_sham, summary, n_seeds=6) -> dict:
    return {
        "models": [{"name": f"m{i}", "path": "dummy", "num_timesteps": num_timesteps}
                   for i in range(n_seeds)],
        "history_lags": list(lags),
        "history_sham": history_sham,
        "summary": summary,
    }


def baseline_control() -> dict:
    return full_group(1_000_000, [], False,
                       full_summary(P1["control"], P4["control"], P2["control"], P3["control"]))


def baseline_treat() -> dict:
    return full_group(1_000_000, LAGS, False,
                       full_summary(P1["treat"], P4["treat"], P2["treat"], P3["treat"]))


def baseline_sham() -> dict:
    # にせ履歴群自体は錨照合の対象外なので値は任意（型だけ本物に合わせる）。
    return full_group(1_000_000, LAGS, True, full_summary(1.0, 1.4, 10, 2))


def run_judge_sham(tmp_root: Path, control: dict, treat: dict, sham: dict,
                    tag: str) -> subprocess.CompletedProcess:
    """judge_sham.py を実際に subprocess で起動する（前提検査は main() の中にあるため、
    関数を直接呼ぶだけでは検査できない）。"""
    c_path, t_path, s_path, out_path = (
        tmp_root / f"{tag}_control.json", tmp_root / f"{tag}_treat.json",
        tmp_root / f"{tag}_sham.json", tmp_root / f"{tag}_out.json")
    c_path.write_text(json.dumps(control), encoding="utf-8")
    t_path.write_text(json.dumps(treat), encoding="utf-8")
    s_path.write_text(json.dumps(sham), encoding="utf-8")
    cmd = [PYTHON, str(JUDGE_SHAM_PY), "--control", str(c_path), "--treat", str(t_path),
           "--sham", str(s_path), "--out", str(out_path)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


# ======================================================================
# T1: r = ln(sham/control)/ln(treat/control) の定義（P1 の錨で 3 点確認）
# ======================================================================
def t1_r_definition() -> bool:
    print("\n[T1] r の定義: sham=control→r=0 / sham=treat→r=1 / sham=幾何中央→r=0.5（P1 の錨）")
    control, treat = P1["control"], P1["treat"]   # 0.500, 2.125
    # 手計算: sham=control → ln(control/control)=ln(1)=0 → r=0
    r_control = _relative_position(control, control, treat)
    # 手計算: sham=treat → ln(treat/control)/ln(treat/control)=1 → r=1
    r_treat = _relative_position(treat, control, treat)
    # 手計算: sham=sqrt(control*treat)=sqrt(0.500*2.125)=sqrt(1.0625)≈1.030776…
    #         ln(sham/control) = ln(sqrt(treat/control)) = 0.5*ln(treat/control) → r=0.5
    geo_mid = math.sqrt(control * treat)
    r_mid = _relative_position(geo_mid, control, treat)
    print(f"  sham=control({control}): r={r_control}（期待 0.0）")
    print(f"  sham=treat({treat}): r={r_treat}（期待 1.0）")
    print(f"  sham=幾何中央({geo_mid:.6f}): r={r_mid}（期待 0.5）")
    return r_control == 0.0 and r_treat == 1.0 and r_mid == 0.5


# ======================================================================
# T2: 向きの自動処理（P4 は treat < control）
# ======================================================================
def t2_direction_auto() -> bool:
    print("\n[T2] 向きの自動処理: P4 は treat(1.250) < control(1.500) でも式が向きを扱う")
    control, treat = P4["control"], P4["treat"]   # 1.500, 1.250 （treat<control）
    assert treat < control, "P4 は treat<control の前提が崩れている"
    # 手計算: sham=control(1.500) → ln(1.500/1.500)=0 → r=0（分母 ln(1.250/1.500) は負だが 0/負=0）
    r_control = _relative_position(control, control, treat)
    # 手計算: sham=treat(1.250) → ln(1.250/1.500)/ln(1.250/1.500)=1 → r=1
    r_treat = _relative_position(treat, control, treat)
    print(f"  sham=control({control}): r={r_control}（期待 0.0）")
    print(f"  sham=treat({treat}): r={r_treat}（期待 1.0）")
    return r_control == 0.0 and r_treat == 1.0


# ======================================================================
# T3: 境界（r がちょうど 0.5 のとき (B) 側）
# ======================================================================
def t3_boundary() -> bool:
    print("\n[T3] 境界: r がちょうど 0.5 のとき (B) 側（P1 の境界 1.031 ちょうどの値）")
    control, treat = P1["control"], P1["treat"]
    # 手計算（T1 と同じ幾何中央だが、judge_one() 経由で verdict を見る）
    boundary_value = math.sqrt(control * treat)   # ≈ 1.030776 （docstring の「境界 1.031」）
    sham_summary = median_summary(**{P1["field"]: boundary_value})
    result = judge_one("P1", P1, sham_summary)
    print(f"  sham={boundary_value:.6f}（境界ちょうど）: r={result['r']}（期待 0.5）"
          f" verdict={result['verdict']!r}（期待 'B'。条文は r>=0.5→(B)）")
    return result["r"] == 0.5 and result["verdict"] == "B"


# ======================================================================
# T4: 3 つ目の読み（対照群より外側／参照群より外側）
# ======================================================================
def t4_outside_reading() -> bool:
    print("\n[T4] 3 つ目の読み: r<0（対照群より外側）・r>1（参照群より外側）で verdict='outside'")
    control, treat = P1["control"], P1["treat"]   # 0.500, 2.125
    # 対照より外（対照の半分。control より小さいので r<0 になるはず）
    below = control * 0.5   # = 0.25
    r_below = _relative_position(below, control, treat)
    result_below = judge_one("P1", P1, median_summary(**{P1["field"]: below}))
    # 参照より外（参照の1.5倍。treat より大きいので r>1 になるはず）
    above = treat * 1.5   # = 3.1875
    r_above = _relative_position(above, control, treat)
    result_above = judge_one("P1", P1, median_summary(**{P1["field"]: above}))
    print(f"  sham={below}（対照より小さい）: r={r_below:.4f}（期待 <0） "
          f"verdict={result_below['verdict']!r} reading={result_below['reading']!r}")
    print(f"  sham={above}（参照より大きい）: r={r_above:.4f}（期待 >1） "
          f"verdict={result_above['verdict']!r} reading={result_above['reading']!r}")
    ok_below = (r_below < 0.0 and result_below["verdict"] == "outside"
                and "対照群より外側" in result_below["reading"])
    ok_above = (r_above > 1.0 and result_above["verdict"] == "outside"
                and "参照群より外側" in result_above["reading"])
    return ok_below and ok_above


# ======================================================================
# T5: P3 の別扱い（ゴール件数 0→(A/C)寄り・1件以上→(B)寄り、r は None）
# ======================================================================
def t5_p3_special_case() -> bool:
    print("\n[T5] P3 の別扱い: ゴール件数 0→(A/C)寄り・1件以上→(B)寄り、r は計算しない")
    r_zero = judge_p3({P3["field"]: 0})
    r_one = judge_p3({P3["field"]: 1})
    print(f"  ゴール 0 件: verdict={r_zero['verdict']!r}（期待 'A_or_C'） r={r_zero['r']!r}（期待 None）")
    print(f"  ゴール 1 件: verdict={r_one['verdict']!r}（期待 'B'） r={r_one['r']!r}（期待 None）")
    return (r_zero["verdict"] == "A_or_C" and r_zero["r"] is None
            and r_one["verdict"] == "B" and r_one["r"] is None)


# ======================================================================
# T6: 錨の照合（_check_anchors）
# ======================================================================
def t6_check_anchors() -> bool:
    print("\n[T6] 錨の照合: 一致時は空、事前登録と食い違うときは指摘を返す")
    # (a) 事前登録どおり → 空
    bad_match = _check_anchors(anchor_summary("control"), anchor_summary("treat"))
    print(f"  一致するとき: bad={bad_match}（期待 []）")

    # (b) 対照群の P1 がずれている → P1 の指摘を含む
    c_bad = anchor_summary("control")
    c_bad["across_seeds_median"][P1["field"]] = P1["control"] + 0.01
    bad_p1 = _check_anchors(c_bad, anchor_summary("treat"))
    print(f"  対照の P1 がずれる: bad={bad_p1}")

    # (c) 参照群の P4 がずれている → P4 の指摘を含む
    t_bad = anchor_summary("treat")
    t_bad["across_seeds_median"][P4["field"]] = P4["treat"] - 0.01
    bad_p4 = _check_anchors(anchor_summary("control"), t_bad)
    print(f"  参照の P4 がずれる: bad={bad_p4}")

    # (d) 対照群の P3 がずれている → P3 の指摘を含む
    c_bad_p3 = anchor_summary("control")
    c_bad_p3[P3["field"]] = P3["control"] + 1
    bad_p3 = _check_anchors(c_bad_p3, anchor_summary("treat"))
    print(f"  対照の P3 がずれる: bad={bad_p3}")

    ok_match = bad_match == []
    ok_p1 = len(bad_p1) == 1 and "P1" in bad_p1[0] and "事前登録" in bad_p1[0]
    ok_p4 = len(bad_p4) == 1 and "P4" in bad_p4[0] and "事前登録" in bad_p4[0]
    ok_p3 = len(bad_p3) == 1 and "P3" in bad_p3[0] and "事前登録" in bad_p3[0]
    return ok_match and ok_p1 and ok_p4 and ok_p3


# ======================================================================
# T7: main() の安全弁 6 条件（subprocess で実際に起動して確認する）
# ======================================================================
def t7_safety_valves() -> bool:
    print("\n[T7] main() の安全弁 6 条件が非ゼロ終了コードで落ちること")
    results = []
    with tempfile.TemporaryDirectory(prefix="judge_sham_safety_") as root:
        root = Path(root)

        # ① seed 数が 6 でない（対照の per_seed_median を 5 seed にする）
        c1 = baseline_control()
        c1["summary"]["per_seed_median"] = {f"seed{i}": {} for i in range(5)}
        p1 = run_judge_sham(root, c1, baseline_treat(), baseline_sham(), "case1")
        msg1 = p1.stdout + p1.stderr
        want1 = "の seed 数が 6 でない"
        results.append(("①seed 数が 6 でない", p1.returncode != 0 and want1 in msg1, p1.returncode))

        # ② 3 群の学習量が違う（にせ履歴群だけ学習量をずらす）
        s2 = baseline_sham()
        s2["models"] = [{"name": f"m{i}", "path": "dummy", "num_timesteps": 1_100_000}
                         for i in range(6)]
        p2 = run_judge_sham(root, baseline_control(), baseline_treat(), s2, "case2")
        msg2 = p2.stdout + p2.stderr
        want2 = "3 群の学習量が違う"
        results.append(("②3 群の学習量が違う", p2.returncode != 0 and want2 in msg2, p2.returncode))

        # ③ にせ履歴群に history_sham が無い
        s3 = baseline_sham()
        s3["history_sham"] = False
        p3 = run_judge_sham(root, baseline_control(), baseline_treat(), s3, "case3")
        msg3 = p3.stdout + p3.stderr
        want3 = "にせ履歴群の測定に history_sham が記録されていない"
        results.append(("③にせ履歴群に history_sham が無い", p3.returncode != 0 and want3 in msg3, p3.returncode))

        # ④ 対照群に history_sham が入っている
        c4 = baseline_control()
        c4["history_sham"] = True
        p4 = run_judge_sham(root, c4, baseline_treat(), baseline_sham(), "case4")
        msg4 = p4.stdout + p4.stderr
        want4 = "対照群または参照群の測定に history_sham が入っている"
        results.append(("④対照群に history_sham が入っている", p4.returncode != 0 and want4 in msg4, p4.returncode))

        # ⑤ にせ履歴群と参照群の遅れが違う
        s5 = baseline_sham()
        s5["history_lags"] = LAGS[:-1]   # 128 を欠かせて参照群の遅れと違えさせる
        p5 = run_judge_sham(root, baseline_control(), baseline_treat(), s5, "case5")
        msg5 = p5.stdout + p5.stderr
        want5 = "にせ履歴群と参照群の遅れが違う"
        results.append(("⑤にせ履歴群と参照群の遅れが違う", p5.returncode != 0 and want5 in msg5, p5.returncode))

        # ⑥ 錨が事前登録と食い違う（対照群の respawn_per_1000 をずらす）
        c6 = baseline_control()
        c6["summary"]["across_seeds_median"][P1["field"]] = P1["control"] + 0.1
        p6 = run_judge_sham(root, c6, baseline_treat(), baseline_sham(), "case6")
        msg6 = p6.stdout + p6.stderr
        want6 = "錨が事前登録と食い違う"
        results.append(("⑥錨が事前登録と食い違う", p6.returncode != 0 and want6 in msg6, p6.returncode))

        # 参考: 何も壊していないベースラインは通ること（このテスト自体の合成データが
        # 壊れていないことの確認。集計には含めない）
        p_ok = run_judge_sham(root, baseline_control(), baseline_treat(), baseline_sham(), "baseline")
        print(f"  （参考）ベースライン: returncode={p_ok.returncode}（期待 0）")
        if p_ok.returncode != 0:
            print(f"    stdout/stderr: {(p_ok.stdout + p_ok.stderr)[-500:]}")

    for name, ok, rc in results:
        print(f"  {name}: returncode={rc}（非ゼロが期待） 判定={ok}")
    return all(ok for _, ok, _ in results)


# ======================================================================
# T8: 空振り防止の自己検査（合成データを壊すと P1 の判定が反転すること）
# ======================================================================
def t8_self_check_breaks() -> bool:
    print("\n[T8] 空振り防止の自己検査: P1 の sham を対照寄り→参照寄りに書き換えると判定が反転すること")
    control, treat = P1["control"], P1["treat"]   # 0.500, 2.125
    # 対照寄り: control の 1.05 倍 = 0.525
    # 手計算: r = ln(0.525/0.500)/ln(2.125/0.500) = ln(1.05)/ln(4.25) ≈ 0.04879/1.44692 ≈ 0.0337 (<0.5)
    sham_ac = control * 1.05
    result_ac = judge_one("P1", P1, median_summary(**{P1["field"]: sham_ac}))
    # 参照寄り（壊した値）: treat の 0.95 倍 = 2.01875
    # 手計算: r = ln(2.01875/0.500)/ln(4.25) = ln(4.0375)/1.44692 ≈ 1.39555/1.44692 ≈ 0.9645 (>=0.5)
    sham_b = treat * 0.95
    result_b = judge_one("P1", P1, median_summary(**{P1["field"]: sham_b}))
    print(f"  元データ（sham={sham_ac}、対照寄り）: verdict={result_ac['verdict']!r}（期待 'A_or_C'）"
          f" r={result_ac['r']:.4f}")
    print(f"  壊したデータ（sham={sham_b}、参照寄り）: verdict={result_b['verdict']!r}（期待 'B'）"
          f" r={result_b['r']:.4f}")
    return result_ac["verdict"] == "A_or_C" and result_b["verdict"] == "B"


# ======================================================================
def main() -> int:
    print("=" * 78)
    print("experiments/exp_022_sham_history/judge_sham.py の単体テスト")
    print("=" * 78)
    tests = [
        ("T1 r の定義（3 点）", t1_r_definition),
        ("T2 向きの自動処理（P4）", t2_direction_auto),
        ("T3 境界（r=0.5→(B)）", t3_boundary),
        ("T4 3 つ目の読み（outside）", t4_outside_reading),
        ("T5 P3 の別扱い", t5_p3_special_case),
        ("T6 錨の照合（_check_anchors）", t6_check_anchors),
        ("T7 main() の安全弁 6 条件", t7_safety_valves),
        ("T8 空振り防止の自己検査", t8_self_check_breaks),
    ]
    results = []
    for name, fn in tests:
        try:
            ok = fn()
        except Exception as exc:  # noqa: BLE001 — 1 項目の失敗で全体を止めない
            import traceback
            print(f"  🔴 例外: {exc!r}")
            print("  " + "\n  ".join(traceback.format_exc().splitlines()[-6:]))
            ok = False
        results.append((name, ok))

    print("\n" + "=" * 78)
    n_ok = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✅ PASS' if ok else '🔴 FAIL'}  {name}")
    print(f"\n  {n_ok} / {len(results)} PASS")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
