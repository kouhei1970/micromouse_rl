#!/usr/bin/env python3
"""experiments/exp_021_observation_history/judge.py の単体テスト。

judge.py は exp_021 の Q1〜Q5（`card.md` §5 の条文）を機械的に判定するスクリプト。
**このテストは配管（条文→コードへの落とし込み）だけを検査する。実験の当たり外れ
そのものは検査対象ではない。**

🔴 最重要方針: **合成データ（自分で作った模擬の測定出力）だけを使う。**
`outputs/exp_021_driving_*.json`（本物の測定出力）や `logs/exp_021_seed*/`（本物の
ログ）は一切読まない。本物を入れると実験の判定の帰結（当たり／外れ）が出てしまい、
「配管の検査」という本テストの存在理由（教授裁定 2026-08-14）が崩れる。

| # | 検査 | judge.py 対応 |
|---|---|---|
| T1 | Q1: 介入群中央値 >= 対照群中央値 × 1.25 の境界（ちょうど／わずかに下回る） | judge_q1() |
| T2 | Q2: リスポーン 0.80 倍以下 かつ 前進 0.90 倍以上 の複合条件（3 通り） | judge_q2() |
| T3 | Q3: goal_rate の 6 seed 中央値が厳密に 0.05 なら外れ（境界） | judge_q3() |
| T4 | Q4: 立て直し成立割合の境界・分母 0 の除外・全 seed 分母 0 で判定不能 | judge_q4() |
| T5 | Q5: 打ち切り条文（10 点全て<0.05）の成立・不成立・欠測時の扱い | judge_q5() |
| T6 | main() の安全弁 4 条件が非ゼロ終了コードで落ちること | main() の前提検査 |
| T7 | 集約がプール集計でなく中央値であること | _medians() / judge_q1() |
| T8 | 空振り防止の自己検査（データを壊すと判定が反転すること） | judge_q1() |

pytest は使わない plain Python スクリプト（tests/test_curriculum.py・tests/test_obs_history.py
と同じ流儀）。実行方法（リポジトリルートで）:
    .venv/bin/python tests/test_judge.py
"""
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

JUDGE_PY = REPO_ROOT / "experiments/exp_021_observation_history/judge.py"
PYTHON = str(REPO_ROOT / ".venv/bin/python")


def load_judge_module():
    """judge.py は experiments/ 配下で __init__.py が無い（パッケージでない）ため、
    tests/test_exp012_goal_snapshot.py と同じ流儀でファイルパスから直接読み込む。"""
    spec = importlib.util.spec_from_file_location("exp021_judge", JUDGE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_judge = load_judge_module()
judge_q1 = _judge.judge_q1
judge_q2 = _judge.judge_q2
judge_q3 = _judge.judge_q3
judge_q4 = _judge.judge_q4
judge_q5 = _judge.judge_q5
CUTOFF_POINTS = _judge.CUTOFF_POINTS
CUTOFF_RATE = _judge.CUTOFF_RATE


# ======================================================================
# 合成データの組み立てヘルパ（measure_driving.py の出力の形を最小限で真似る）
# ======================================================================
def seeds_uniform(**kwargs) -> dict:
    """6 seed 分、全て同じ値を持つ per_seed_median の辞書を作る。"""
    return {f"seed{i}": dict(kwargs) for i in range(6)}


def seeds_list(key: str, values: list) -> dict:
    """1 個のキーについて、seed ごとに異なる値を割り当てた per_seed_median の辞書。"""
    return {f"seed{i}": {key: v} for i, v in enumerate(values)}


def measure(per_seed_median: dict) -> dict:
    """judge_q1/judge_q2 が読む最小構成（summary.per_seed_median だけを埋める）。"""
    return {"summary": {"per_seed_median": per_seed_median}}


def treat_p5(per_seed: dict, median_rate, verdict: str, excluded=None) -> dict:
    """judge_q4 が読む最小構成（t["summary"]["p5"] だけを埋める）。
    per_seed: {seed名: (rate, n_denominator)}"""
    return {"summary": {"p5": {
        "median_rate": median_rate,
        "verdict": verdict,
        "excluded_seeds": excluded or [],
        "per_seed": {k: {"rate": v[0], "n_denominator": v[1]} for k, v in per_seed.items()},
    }}}


def write_validation_history(dir_path: Path, entries: list) -> None:
    """judge_q3/judge_q5 が読む validation_history.json を書く。"""
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "validation_history.json").write_text(json.dumps(entries), encoding="utf-8")


# ======================================================================
# T1: Q1 の 1.25 倍境界
# ======================================================================
def t1_q1_threshold() -> bool:
    print("\n[T1] Q1: 介入群中央値 >= 対照群中央値 × 1.25 の境界")
    # 手計算: 対照 6 seed 全て 100.0 → 中央値 100.0 → 閾値 = 100.0 * 1.25 = 125.0
    c = measure(seeds_uniform(net_progress_per_1000=100.0))
    t_hit = measure(seeds_uniform(net_progress_per_1000=125.0))    # ちょうど 1.25 倍
    t_miss = measure(seeds_uniform(net_progress_per_1000=124.9))   # わずかに下回る
    r_hit = judge_q1(c, t_hit)
    r_miss = judge_q1(c, t_miss)
    print(f"  ちょうど 125.0（閾値と同値）: hit={r_hit['hit']}（期待 True）"
          f" threshold={r_hit['threshold']}")
    print(f"  わずかに下回る 124.9: hit={r_miss['hit']}（期待 False）")
    return (r_hit["hit"] is True) and (r_miss["hit"] is False) and (r_hit["threshold"] == 125.0)


# ======================================================================
# T2: Q2 の複合条件（リスポーン 0.80 倍以下 かつ 前進 0.90 倍以上）
# ======================================================================
def t2_q2_composite() -> bool:
    print("\n[T2] Q2: リスポーン 0.80 倍以下 かつ 前進 0.90 倍以上 の複合条件")
    # 手計算: 対照 respawn 中央値 100.0 → 閾値 80.0／対照 net_progress 中央値 100.0 → 保護下限 90.0
    c = measure(seeds_uniform(net_progress_per_1000=100.0, respawn_per_1000=100.0))
    # (a) リスポーン 80（<=80） かつ 前進 90（>=90） → 当たり
    t_a = measure(seeds_uniform(net_progress_per_1000=90.0, respawn_per_1000=80.0))
    # (b) リスポーンは下がった（70<=80）が前進が 90 を割った（85<90） → 外れ（要点の経路）
    t_b = measure(seeds_uniform(net_progress_per_1000=85.0, respawn_per_1000=70.0))
    # (c) 前進は保った（95>=90）がリスポーンが下がらない（85>80） → 外れ
    t_c = measure(seeds_uniform(net_progress_per_1000=95.0, respawn_per_1000=85.0))
    r_a, r_b, r_c = judge_q2(c, t_a), judge_q2(c, t_b), judge_q2(c, t_c)
    print(f"  (a) respawn=80,net=90: hit={r_a['hit']}（期待 True）"
          f" cond_respawn_ok={r_a['cond_respawn_ok']} cond_progress_not_degraded={r_a['cond_progress_not_degraded']}")
    print(f"  (b) respawn=70,net=85(<90 で失格): hit={r_b['hit']}（期待 False）"
          f" cond_respawn_ok={r_b['cond_respawn_ok']} cond_progress_not_degraded={r_b['cond_progress_not_degraded']}")
    print(f"  (c) respawn=85(>80 で失格),net=95: hit={r_c['hit']}（期待 False）"
          f" cond_respawn_ok={r_c['cond_respawn_ok']} cond_progress_not_degraded={r_c['cond_progress_not_degraded']}")
    ok_a = r_a["hit"] is True and r_a["cond_respawn_ok"] and r_a["cond_progress_not_degraded"]
    ok_b = r_b["hit"] is False and r_b["cond_respawn_ok"] and not r_b["cond_progress_not_degraded"]
    ok_c = r_c["hit"] is False and (not r_c["cond_respawn_ok"]) and r_c["cond_progress_not_degraded"]
    return ok_a and ok_b and ok_c


# ======================================================================
# T3: Q3 の境界（goal_rate 中央値が厳密に 0.05 なら外れ）
# ======================================================================
def t3_q3_boundary() -> bool:
    print("\n[T3] Q3: goal_rate の 6 seed 中央値が厳密に 0.05 なら外れ（境界）")
    with tempfile.TemporaryDirectory(prefix="judge_q3_") as root:
        root = Path(root)
        # 境界: 6 seed 全て最終 goal_rate=0.05 ちょうど → 中央値 0.05 → 外れ（`< 0.05` を満たさない）
        dirs_boundary = []
        for i in range(6):
            d = root / f"boundary_seed{i}"
            write_validation_history(d, [
                {"total_timesteps": 1_000_000, "goal_rate": 0.9},   # 途中経過（無視されるはず）
                {"total_timesteps": 2_000_000, "goal_rate": 0.05},  # 最終（total_timesteps 最大）
            ])
            dirs_boundary.append(d)
        r_boundary = judge_q3(dirs_boundary)

        # 中央値 0.025 → 当たり
        dirs_hit = []
        for i in range(6):
            d = root / f"hit_seed{i}"
            write_validation_history(d, [{"total_timesteps": 2_000_000, "goal_rate": 0.025}])
            dirs_hit.append(d)
        r_hit = judge_q3(dirs_hit)

    print(f"  中央値 0.05 ちょうど: hit={r_boundary['hit']}（期待 False）median={r_boundary['median']}")
    print(f"  中央値 0.025: hit={r_hit['hit']}（期待 True）median={r_hit['median']}")
    return (r_boundary["hit"] is False and r_boundary["median"] == 0.05
            and r_hit["hit"] is True and r_hit["median"] == 0.025)


# ======================================================================
# T4: Q4 の境界・分母 0 除外・全 seed 分母 0 で判定不能
# ======================================================================
def t4_q4_cases() -> bool:
    print("\n[T4] Q4: 立て直し成立割合の境界・分母 0 の除外・判定不能")
    # (a) 中央値 0.50 ちょうど → 当たり（>=）
    t_a = treat_p5({f"seed{i}": (0.5, 20) for i in range(6)}, median_rate=0.5, verdict="ok")
    r_a = judge_q4(t_a)
    print(f"  (a) 中央値 0.50 ちょうど: hit={r_a['hit']}（期待 True）")

    # (b) 1 seed が分母 0 で除外、残り 5 seed [0.3,0.4,0.5,0.6,0.7] の中央値 0.5 → 当たり
    remaining = [0.3, 0.4, 0.5, 0.6, 0.7]
    med_b = statistics.median(remaining)   # = 0.5
    per_seed_b = {f"seed{i}": (v, 20) for i, v in enumerate(remaining)}
    per_seed_b["seed_excluded"] = (None, 0)
    t_b = treat_p5(per_seed_b, median_rate=med_b, verdict="ok", excluded=["seed_excluded"])
    r_b = judge_q4(t_b)
    print(f"  (b) 1 seed 分母 0 で除外、残り中央値 {med_b}: hit={r_b['hit']}（期待 True）"
          f" excluded_seeds={r_b['excluded_seeds']} n_denominator[seed_excluded]="
          f"{r_b['n_denominator']['seed_excluded']}（期待 0）")

    # (c) 全 seed 分母 0 → 判定不能（hit は None）
    per_seed_c = {f"seed{i}": (None, 0) for i in range(6)}
    t_c = treat_p5(per_seed_c, median_rate=None, verdict="indeterminate")
    r_c = judge_q4(t_c)
    print(f"  (c) 全 seed 分母 0: hit={r_c['hit']!r}（期待 None） verdict={r_c['verdict']!r}")

    ok_a = r_a["hit"] is True
    ok_b = (r_b["hit"] is True and r_b["excluded_seeds"] == ["seed_excluded"]
            and r_b["n_denominator"]["seed_excluded"] == 0 and r_b["median"] == 0.5)
    ok_c = r_c["hit"] is None
    return ok_a and ok_b and ok_c


# ======================================================================
# T5: Q5 打ち切り条文の成立・不成立・欠測
# ======================================================================
def t5_q5_cases() -> bool:
    print("\n[T5] Q5: 打ち切り条文（10 点すべて goal_rate < 0.05）の成立・不成立・欠測")
    with tempfile.TemporaryDirectory(prefix="judge_q5_") as root:
        root = Path(root)

        # (a) 10 点すべて 0.03(<0.05) → 成立
        d_a = root / "seed_a"
        write_validation_history(d_a, [{"total_timesteps": s, "goal_rate": 0.03} for s in CUTOFF_POINTS])
        r_a = judge_q5([d_a])
        per_a = next(iter(r_a["per_seed"].values()))

        # (b) 50 万歩の 1 点だけ 0.10(>=0.05) → 不成立
        d_b = root / "seed_b"
        pts_b = {s: 0.03 for s in CUTOFF_POINTS}
        pts_b[CUTOFF_POINTS[4]] = 0.10   # 50 万歩の点だけ上振れ
        write_validation_history(d_b, [{"total_timesteps": s, "goal_rate": v} for s, v in pts_b.items()])
        r_b = judge_q5([d_b])
        per_b = next(iter(r_b["per_seed"].values()))

        # (c) 80 万歩の 1 点が記録に無い（欠測） → 成立としない
        d_c = root / "seed_c"
        pts_c = [s for s in CUTOFF_POINTS if s != CUTOFF_POINTS[7]]   # 80 万歩を欠落
        write_validation_history(d_c, [{"total_timesteps": s, "goal_rate": 0.03} for s in pts_c])
        r_c = judge_q5([d_c])
        per_c = next(iter(r_c["per_seed"].values()))

    print(f"  (a) 10 点すべて<0.05: fired={per_a['fired']}（期待 True）")
    print(f"  (b) 1 点だけ>=0.05: fired={per_b['fired']}（期待 False）")
    print(f"  (c) 1 点欠測: fired={per_c['fired']}（期待 False） missing={per_c['missing']}（期待 非空）")
    return (per_a["fired"] is True and per_b["fired"] is False
            and per_c["fired"] is False and len(per_c["missing"]) > 0)


# ======================================================================
# T6: main() の安全弁 4 条件（subprocess で実際に起動して確認する）
# ======================================================================
def base_control(num_timesteps=800_000, lags=None, n_seeds=6) -> dict:
    return {
        "models": [{"name": f"c{i}", "path": "dummy", "num_timesteps": num_timesteps}
                   for i in range(n_seeds)],
        "history_lags": list(lags) if lags is not None else [],
        "summary": {"per_seed_median": {f"seed{i}": {} for i in range(n_seeds)}},
    }


def base_treat(num_timesteps=800_000, lags=(1, 2, 5), n_seeds=6) -> dict:
    return {
        "models": [{"name": f"t{i}", "path": "dummy", "num_timesteps": num_timesteps}
                   for i in range(n_seeds)],
        "history_lags": list(lags) if lags is not None else [],
        "summary": {"per_seed_median": {f"seed{i}": {} for i in range(n_seeds)}},
    }


def run_judge_main(tmp_root: Path, control: dict, treat: dict, tag: str) -> subprocess.CompletedProcess:
    """judge.py を実際に subprocess で起動する（前提検査は main() の中にあるため、
    関数を直接呼ぶだけでは検査できない）。"""
    c_path, t_path, out_path = (tmp_root / f"{tag}_control.json",
                                 tmp_root / f"{tag}_treat.json",
                                 tmp_root / f"{tag}_out.json")
    c_path.write_text(json.dumps(control), encoding="utf-8")
    t_path.write_text(json.dumps(treat), encoding="utf-8")
    # ログディレクトリは前提検査が落ちた後は読まれないため、実在しなくてよい
    dummy_log = tmp_root / f"{tag}_dummy_log"
    cmd = [PYTHON, str(JUDGE_PY), "--control", str(c_path), "--treat", str(t_path),
           "--logs", str(dummy_log), "--out", str(out_path)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


def t6_safety_valves() -> bool:
    print("\n[T6] main() の安全弁 4 条件が非ゼロ終了コードで落ちること")
    results = []
    with tempfile.TemporaryDirectory(prefix="judge_safety_") as root:
        root = Path(root)

        # ① seed 数が 6 でない（対照だけ 5 seed にする）
        p1 = run_judge_main(root, base_control(n_seeds=5), base_treat(), "case1")
        msg1 = p1.stdout + p1.stderr
        want1 = "seed 数が 6 でない"
        results.append(("①seed 数不一致", p1.returncode != 0 and want1 in msg1, p1.returncode, want1 in msg1))

        # ② 学習量が seed 間で不揃い（介入群 models の num_timesteps をバラす）
        t2 = base_treat()
        t2["models"] = [{"name": f"t{i}", "path": "dummy",
                          "num_timesteps": 800_000 if i < 5 else 900_000} for i in range(6)]
        p2 = run_judge_main(root, base_control(), t2, "case2")
        msg2 = p2.stdout + p2.stderr
        want2 = "学習量が seed 間で揃っていない"
        results.append(("②学習量が seed 間で不揃い", p2.returncode != 0 and want2 in msg2, p2.returncode, want2 in msg2))

        # ③ 介入群と対照群の学習量が違う
        p3 = run_judge_main(root, base_control(num_timesteps=800_000),
                             base_treat(num_timesteps=900_000), "case3")
        msg3 = p3.stdout + p3.stderr
        want3 = "介入群と対照群の学習量が違う"
        results.append(("③対照と介入で学習量が違う", p3.returncode != 0 and want3 in msg3, p3.returncode, want3 in msg3))

        # ④a 介入群に遅れが無い
        p4a = run_judge_main(root, base_control(), base_treat(lags=()), "case4a")
        msg4a = p4a.stdout + p4a.stderr
        want4a = "観測履歴の遅れが記録されていない"
        results.append(("④a介入群に遅れ無し", p4a.returncode != 0 and want4a in msg4a, p4a.returncode, want4a in msg4a))

        # ④b 対照群に遅れが入っている
        p4b = run_judge_main(root, base_control(lags=(1, 2)), base_treat(), "case4b")
        msg4b = p4b.stdout + p4b.stderr
        want4b = "対照群の測定に観測履歴の遅れが入っている"
        results.append(("④b対照群に遅れ混入", p4b.returncode != 0 and want4b in msg4b, p4b.returncode, want4b in msg4b))

    for name, ok, rc, msg_ok in results:
        print(f"  {name}: returncode={rc}（非ゼロが期待）  該当メッセージを含む={msg_ok}")
    return all(ok for _, ok, _, _ in results)


# ======================================================================
# T7: 集約が中央値であってプール集計でないこと
# ======================================================================
def t7_median_not_pooled() -> bool:
    print("\n[T7] 集約が中央値であって、プール集計（全 seed をまとめた平均）でないこと")
    # 例: [1,1,1,1,1,10] は 中央値 1.0 と プール平均 2.5 が異なる並び
    #     （[1,1,1,9,9,9] のように中央値とプール平均が一致してしまう並びでは検査にならない）
    skewed = [1, 1, 1, 1, 1, 10]
    median_expected = statistics.median(skewed)   # (1+1)/2 = 1.0
    pooled_mean = statistics.mean(skewed)          # (1*5+10)/6 = 2.5
    print(f"  合成データ {skewed}: 中央値={median_expected}, プール平均={pooled_mean}"
          f"（両者が異なることを利用する）")
    c = measure(seeds_list("net_progress_per_1000", skewed))
    t = measure(seeds_uniform(net_progress_per_1000=100.0))   # 介入群側は何でもよい（対照側だけ見る）
    r = judge_q1(c, t)
    print(f"  judge_q1 が返した control_median = {r['control_median']}"
          f"（期待 中央値 {median_expected}、プール平均 {pooled_mean} ではない）")
    return r["control_median"] == median_expected and r["control_median"] != pooled_mean


# ======================================================================
# T8: 空振り防止の自己検査（合成データを壊すと判定が反転すること）
# ======================================================================
def t8_self_check_breaks() -> bool:
    print("\n[T8] 空振り防止の自己検査: 合成データを壊すと T1 の判定が反転すること")
    # T1(a) の当たりケースを複製し、介入群の値を「壊す」（対照の半分にする）。
    # 壊した後も判定関数が同じ「当たり」を返すなら、T1 は何も検査していないことになる。
    c = measure(seeds_uniform(net_progress_per_1000=100.0))
    t_correct = measure(seeds_uniform(net_progress_per_1000=125.0))   # 本来「当たり」
    t_broken = measure(seeds_uniform(net_progress_per_1000=50.0))     # 壊した値（対照の半分）
    r_correct = judge_q1(c, t_correct)
    r_broken = judge_q1(c, t_broken)
    print(f"  元データ（介入群=125.0）: hit={r_correct['hit']}（期待 True）")
    print(f"  壊したデータ（介入群=50.0）: hit={r_broken['hit']}（期待 False。反転すれば"
          f"T1 が実際に判定へ効いている証拠）")
    return r_correct["hit"] is True and r_broken["hit"] is False


# ======================================================================
def main() -> int:
    print("=" * 78)
    print("experiments/exp_021_observation_history/judge.py の単体テスト")
    print("=" * 78)
    tests = [
        ("T1 Q1 の 1.25 倍境界", t1_q1_threshold),
        ("T2 Q2 の複合条件（3 通り）", t2_q2_composite),
        ("T3 Q3 の 0.05 境界", t3_q3_boundary),
        ("T4 Q4 の境界・除外・判定不能", t4_q4_cases),
        ("T5 Q5 の成立・不成立・欠測", t5_q5_cases),
        ("T6 main() の安全弁 4 条件", t6_safety_valves),
        ("T7 集約が中央値（プール集計でない）", t7_median_not_pooled),
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
