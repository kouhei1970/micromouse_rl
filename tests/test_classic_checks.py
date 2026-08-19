"""tests/test_classic_checks.py — 再発防止の 4 検査そのものを検査する

🔴 **検査は「壊れたときに鳴る」ことを実測してから使う**（`docs/RESEARCH_PLAN.md` §12-9 (c)）。
本ファイルは各検査について **発火側（壊せば鳴る）と空振り側（正常なら鳴らない）を対で**確かめる。
片側だけの検査は、鳴らない検査と区別がつかないため認めない。
"""

from __future__ import annotations

import pytest

from classic.checks import (
    assert_same_callable,
    find_retracted_numbers,
    negative_control,
    plan_adherence,
    resolve_bound,
)


# ======================================================================
# 型 C: 計画に乗っていた割合
# ======================================================================

def test_plan_adherence_fires_on_the_real_case():
    """🔴 旧実装で実際に起きた比率（29,422 ティック中 69 = 0.23%）で鳴ること。"""
    plan_ids = ["diag"] * 69 + ["parent"] * (29422 - 69)
    got = plan_adherence(plan_ids, intended="diag")
    assert got.total_ticks == 29422
    assert got.intended_ticks == 69
    assert got.ratio == pytest.approx(0.0023, abs=1e-4)
    # どんな現実的な閾値でも落ちる
    assert not got.ok(0.5)
    assert not got.ok(0.1)
    assert "diag" in got.describe()


def test_plan_adherence_passes_when_the_plan_is_actually_used():
    """空振り側: 意図どおり乗っていれば通ること（恒偽の検査ではない）。"""
    got = plan_adherence(["diag"] * 950 + ["parent"] * 50, intended="diag")
    assert got.ratio == pytest.approx(0.95)
    assert got.ok(0.9)


def test_plan_adherence_rejects_empty_and_unspecified():
    """走行が無い／意図が空のときは、静かに 0 を返さずに落ちること。"""
    with pytest.raises(ValueError):
        plan_adherence([], intended="x")
    with pytest.raises(ValueError):
        plan_adherence(["a"], intended=set())


# ======================================================================
# 型 B: 実行経路の同一性
# ======================================================================

class _Base:
    def drive(self):
        return "base"


class _Mixin(_Base):
    def drive(self):            # 実際に走るのはこちら（旧実装で起きた形）
        return "mixin"


class _Policy(_Mixin):
    pass


def test_assert_same_callable_fires_when_the_copy_diverges():
    """発火側: 検査が _Base を想定しているが、実際に走るのは _Mixin のとき鳴ること。"""
    with pytest.raises(AssertionError) as e:
        assert_same_callable(_Policy, "drive", _Base)
    msg = str(e.value)
    assert "_Mixin" in msg            # 実際の定義元が示される
    assert "型 B" in msg              # どの型の事故かが分かる


def test_assert_same_callable_passes_on_the_true_owner():
    """空振り側: 実体を正しく指定すれば通ること。"""
    assert_same_callable(_Policy, "drive", _Mixin)
    assert resolve_bound(_Policy, "drive") is _Mixin.drive


def test_resolve_bound_raises_on_missing_method():
    with pytest.raises(AttributeError):
        resolve_bound(_Policy, "no_such_method")


# ======================================================================
# 否定対照
# ======================================================================

def test_negative_control_passes_when_truth_is_not_used():
    """合格: 壊しても変わらず（N1）、対照では変わる（N2）。"""
    got = negative_control(
        run_under_test=lambda broken: "same",              # 真値を見ていない
        run_control=lambda broken: "broken" if broken else "ok",
    )
    assert got.passed
    assert "合格" in got.verdict


def test_negative_control_fails_when_truth_leaks():
    """不合格: 壊すと変わる = まだ真値を読んでいる。"""
    got = negative_control(
        run_under_test=lambda broken: "broken" if broken else "ok",
        run_control=lambda broken: "broken" if broken else "ok",
    )
    assert not got.passed
    assert "不合格" in got.verdict


def test_negative_control_flags_a_toothless_check():
    """🔴 判定保留: 対照でも変わらない = 壊し方が効いていない。

    これを「合格」と読んでしまうのが、旧実装で最も危険だった読み違いである。
    """
    got = negative_control(
        run_under_test=lambda broken: "same",
        run_control=lambda broken: "same",                 # 壊し方が効いていない
    )
    assert not got.passed
    assert "判定保留" in got.verdict


# ======================================================================
# 型 E: 失効した数値の伝播
# ======================================================================

def test_find_retracted_numbers_finds_a_planted_string(tmp_path):
    """発火側: 失効した数値が文書に残っていれば見つかること。"""
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    doc = tmp_path / "summary.md"
    doc.write_text("時間の 69.6% が 0.45 m/s の斜め 37.6%\n", encoding="utf-8")
    subprocess.run(["git", "add", "summary.md"], cwd=tmp_path, check=True)

    hits = find_retracted_numbers(["37.6", "69.6"], repo_root=tmp_path)
    paths = {h.path for h in hits}
    assert "summary.md" in paths
    assert len(hits) >= 2


def test_find_retracted_numbers_respects_the_allow_list(tmp_path):
    """空振り側: 失効を宣言している文書自身は許す（そこには書いてあってよい）。"""
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "retraction.md").write_text("【失効】37.6% は引用禁止\n", encoding="utf-8")
    subprocess.run(["git", "add", "retraction.md"], cwd=tmp_path, check=True)

    assert find_retracted_numbers(["37.6"], repo_root=tmp_path, allow=["retraction.md"]) == []
    assert find_retracted_numbers(["37.6"], repo_root=tmp_path) != []   # 許さなければ鳴る
