"""
tests/test_seed_bands.py
========================
`common/seed_bands.py`（凍結帯の誤用を止める安全弁）の単体テスト。
出所: 裁定 R40 条件 4 → R11 バッチ項目 7。

計画書が挙げた 3 通り（**評価帯を素で指定 → 止まる／合言葉つき → 通る／学習帯 → 通る**）に
加えて、帯の分類・自己整合検査・理由なし gate の拒否も検査する。

実行:
    .venv/bin/python tests/test_seed_bands.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.seed_bands import (  # noqa: E402
    SeedBandViolation, assert_seeds_allowed, classify_seed, describe_seeds,
)


def _expect_raise(fn, label: str):
    try:
        fn()
    except SeedBandViolation:
        return True, f"  [PASS] {label} → 例外で止まった"
    except Exception as e:  # noqa: BLE001
        return False, f"  🔴 [FAIL] {label} → 別の例外: {type(e).__name__}: {e}"
    return False, f"  🔴 [FAIL] {label} → **止まらなかった**"


def _expect_ok(fn, label: str):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return False, f"  🔴 [FAIL] {label} → 例外: {type(e).__name__}: {e}"
    return True, f"  [PASS] {label} → 通った"


def main() -> int:
    results, lines = [], []

    # --- 帯の分類（正本と突き合わせた範囲であること） ---
    # 🔴 **namespace ごとに帯が違う**（同じ整数でも生成器が違えば別の迷路）。
    # とくに maze6 の学習帯 8000 以降は competition の候補プールと**数値が重なる**。
    cases = [("maze6", 6000, "eval"), ("maze6", 6019, "eval"),
             ("maze6", 7000, "validation"), ("maze6", 7019, "validation"),
             ("maze6", 8000, "free"), ("maze6", 8999, "free"),
             ("corridor", 3000, "eval"), ("corridor", 5019, "validation"),
             ("corridor", 8000, "free"),
             ("competition", 7837, "pool"), ("competition", 1000, "pool"),
             ("competition", 40999, "pool"), ("competition", 41000, "free"),
             ("competition", 999, "free"),
             # 同じ 8000 が namespace で別の帯になることの明示的な検査
             ("competition", 8000, "pool")]
    for ns, seed, want in cases:
        got = classify_seed(seed, ns)
        ok = (got == want)
        results.append(ok)
        lines.append(f"  {'[PASS]' if ok else '🔴 [FAIL]'} classify_seed({seed}, {ns!r})"
                     f" = {got}（期待 {want}）")

    # --- 計画書が挙げた 3 通り ---
    ok, msg = _expect_raise(
        lambda: assert_seeds_allowed(range(6000, 6020), namespace="maze6", purpose="train"),
        "maze6 の評価帯を素で学習に指定")
    results.append(ok); lines.append(msg)

    ok, msg = _expect_ok(
        lambda: assert_seeds_allowed(range(6000, 6020), namespace="maze6",
                                     purpose="gate", reason="M4 gate の本番判定"),
        "maze6 の評価帯を合言葉つき（purpose='gate' ＋ reason）で指定")
    results.append(ok); lines.append(msg)

    ok, msg = _expect_ok(
        lambda: assert_seeds_allowed(range(8000, 8020), namespace="maze6", purpose="train"),
        "maze6 の学習帯（8000 以降）を学習に指定")
    results.append(ok); lines.append(msg)

    # --- 追加: 理由なしの gate は拒否（合言葉の実体は理由の記録） ---
    ok, msg = _expect_raise(
        lambda: assert_seeds_allowed(range(6000, 6020), namespace="maze6",
                                     purpose="gate", reason=""),
        "purpose='gate' だが理由が空")
    results.append(ok); lines.append(msg)

    # --- 追加: 検証帯は validate では通り、train では止まる ---
    ok, msg = _expect_ok(
        lambda: assert_seeds_allowed(range(7000, 7020), namespace="maze6",
                                     purpose="validate"),
        "maze6 の検証帯を測定（validate）に指定")
    results.append(ok); lines.append(msg)
    ok, msg = _expect_raise(
        lambda: assert_seeds_allowed(range(7000, 7020), namespace="maze6", purpose="train"),
        "maze6 の検証帯を学習に指定")
    results.append(ok); lines.append(msg)

    # --- 追加: 競技の候補プール（採用 20 面でなくても禁止） ---
    ok, msg = _expect_raise(
        lambda: assert_seeds_allowed([7837], namespace="competition", purpose="train"),
        "競技の候補プールの seed（R40 で実際に使われた 7837）を学習に指定")
    results.append(ok); lines.append(msg)

    # --- 追加: 1 個でも混ざれば止まる（全部が凍結帯である必要はない） ---
    ok, msg = _expect_raise(
        lambda: assert_seeds_allowed([8000, 8001, 6000], namespace="maze6", purpose="train"),
        "maze6 の学習帯 2 個に評価帯 1 個が混ざる")
    results.append(ok); lines.append(msg)

    # --- 追加: namespace の指定が必須であること（既定値を置かない設計） ---
    try:
        assert_seeds_allowed([8000], purpose="train")   # namespace を渡していない
        results.append(False)
        lines.append("  🔴 [FAIL] namespace 未指定 → **通ってしまった**")
    except TypeError:
        results.append(True)
        lines.append("  [PASS] namespace 未指定 → TypeError で止まった（既定値なし）")
    try:
        classify_seed(8000, "maze16")
        results.append(False)
        lines.append("  🔴 [FAIL] 未知の namespace → 通ってしまった")
    except ValueError:
        results.append(True)
        lines.append("  [PASS] 未知の namespace → ValueError で止まった")

    # --- 帯の明示（ログ用の 1 行） ---
    text = describe_seeds([8000, 8001, 7000], namespace="maze6")
    ok = ("free" in text and "validation" in text and "n=3" in text
          and "namespace=maze6" in text)
    results.append(ok)
    lines.append(f"  {'[PASS]' if ok else '🔴 [FAIL]'} describe_seeds → {text}")

    print("=" * 74)
    print("common/seed_bands.py（凍結帯の安全弁）の単体テスト")
    print("=" * 74)
    for line in lines:
        print(line)
    n_ok = sum(1 for r in results if r)
    print("=" * 74)
    print(f"合計: {n_ok}/{len(results)} PASS")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
