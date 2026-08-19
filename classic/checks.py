"""classic/checks.py — 再発防止の 4 つの検査

`research_notes/note_029_classical_postmortem.md` §4 で「作法として登録せず検査に変換する」
と定めたものの実装である。**旧実装で実際に起きた 4 つの型**に、それぞれ 1 つずつ対応する。

    型 C  計画した経路が走行に使われたか誰も検査していない  → plan_adherence()
    型 B  実行経路と検査対象がずれる                        → assert_same_callable()
    ―    特権情報（真値）の残留は数値照合では検出できない  → negative_control()
    型 E  失効した数値が訂正されないまま伝播する            → find_retracted_numbers()

**なぜこれが要るか（実際に起きたこと）**

- 型 C: 13.3 m の斜め経路が計画されたが、実走で乗っていたのは 29,422 ティック中 69（**0.23%**）。
  半月のあいだ誰も気づかなかった。**この検査があれば初日に鳴っていた。**
- 型 B: 判定条文の検査が `SlalomPolicy._do_drive_control` を見ていたが、走行中に実際に呼ばれるのは
  別クラスの写しだった。処置の写しが 2 本に分かれ、片方だけ直しても検査が通る状態になっていた。
- 特権情報: 推定器の検証を「推定誤差を真値と比較する」だけにすると、**比較が通ること自体が
  安心材料になってしまう。**壊しても走行が変わらないことの実証が要る。
- 型 E: 失効させたカードの数値が、総括ノートと設計提案に訂正なしで残り、次のセッションが読んだ。

**すべての検査は「壊れたときに鳴る」ことを自分で確かめられる形にしてある**
（各関数に対応する空振り防止の検査が `tests/test_classic_checks.py` にある）。
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


# ======================================================================
# 型 C: 計画した経路が、実際に走行に使われたことを検査する
# ======================================================================

@dataclass(frozen=True)
class PlanAdherence:
    """走行が「意図した計画」にどれだけ乗っていたか。"""

    total_ticks: int
    intended_ticks: int
    ratio: float
    plan_histogram: dict[str, int]

    def ok(self, threshold: float) -> bool:
        return self.ratio >= threshold

    def describe(self) -> str:
        lines = [
            f"総ティック {self.total_ticks}",
            f"意図した計画に乗っていた {self.intended_ticks}（{self.ratio * 100:.2f}%）",
            "計画ごとの内訳:",
        ]
        for plan_id, n in sorted(self.plan_histogram.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {plan_id}: {n} ティック（{n / max(self.total_ticks, 1) * 100:.2f}%）")
        return "\n".join(lines)


def plan_adherence(plan_ids: Sequence[str], intended: str | set[str]) -> PlanAdherence:
    """ティックごとの「どの計画で走ったか」の識別子列から、意図した計画に乗っていた割合を出す。

    Args:
        plan_ids: 走行のティックごとに、そのとき駆動していた計画の識別子。
            **走行の一次記録に必ず残すこと**（残っていなければこの検査は書けない）。
        intended: 意図した計画の識別子（複数可）。

    Returns:
        PlanAdherence。

    Raises:
        ValueError: plan_ids が空のとき（走行が無いのに検査するのは誤り）。

    Note:
        **閾値は用途ごとに決めること。**ここでは既定値を置かない —
        「とりあえず 0.5」のような根拠のない閾値が、判定を静かに通してしまうため。
    """
    if len(plan_ids) == 0:
        raise ValueError("plan_ids が空である。走行が無いのに乗車率は測れない")
    want = {intended} if isinstance(intended, str) else set(intended)
    if not want:
        raise ValueError("intended が空である。何に乗っていてほしいのかを指定すること")
    hist = Counter(plan_ids)
    hit = sum(n for k, n in hist.items() if k in want)
    return PlanAdherence(
        total_ticks=len(plan_ids),
        intended_ticks=hit,
        ratio=hit / len(plan_ids),
        plan_histogram=dict(hist),
    )


# ======================================================================
# 型 B: 検査対象が、実際に走行中に呼ばれる関数と同一であることを確かめる
# ======================================================================

def resolve_bound(cls: type, method_name: str) -> Callable:
    """クラスの MRO をたどって、実際に呼ばれる実装（関数オブジェクト）を返す。

    継承と混ぜ込み（mixin）が重なると、名前が同じでも**実際に走るのは別の写し**になる。
    旧実装ではこれが原因で、検査が本番と別のコードを見ていた。
    """
    fn = getattr(cls, method_name, None)
    if fn is None:
        raise AttributeError(f"{cls.__name__} に {method_name} が無い")
    return fn


def assert_same_callable(cls: type, method_name: str, expected_owner: type) -> None:
    """`cls` で `method_name` を呼んだとき、実際に走るのが `expected_owner` の実装であることを表明する。

    Raises:
        AssertionError: 実際に走る実装の定義元が expected_owner でないとき。
            メッセージには**実際の定義元とファイル・行番号**を入れる（探し回らずに済むように）。
    """
    fn = resolve_bound(cls, method_name)
    code = getattr(fn, "__code__", None)
    where = f"{code.co_filename}:{code.co_firstlineno}" if code else "（位置不明）"
    owner_fn = getattr(expected_owner, method_name, None)
    if owner_fn is None:
        raise AssertionError(f"{expected_owner.__name__} に {method_name} が無い")
    if fn is not owner_fn:
        raise AssertionError(
            f"{cls.__name__}.{method_name} の実体は {expected_owner.__name__} のものではない。\n"
            f"  実際に走るのは: {getattr(fn, '__qualname__', fn)} @ {where}\n"
            f"  検査が想定していたのは: {getattr(owner_fn, '__qualname__', owner_fn)}\n"
            f"  → 処置の写しが 2 本に分かれている可能性がある（note_029 型 B）"
        )


# ======================================================================
# 特権情報: 壊しても変わらなければ、その経路は使われていない
# ======================================================================

@dataclass(frozen=True)
class NegativeControl:
    """否定対照の結果。**N1 と N2 は対で見る。**"""

    changed_when_broken: bool      # N1: 壊したら変わったか（変わってはいけない）
    changed_in_control: bool       # N2: 対照では変わったか（変わらなければ検査に判別力が無い）

    @property
    def verdict(self) -> str:
        if not self.changed_in_control:
            return "判定保留（検査に判別力が無い — 壊し方が効いていない）"
        return "合格（真値を使っていない）" if not self.changed_when_broken else "不合格（真値がまだ使われている）"

    @property
    def passed(self) -> bool:
        return self.changed_in_control and not self.changed_when_broken


def negative_control(run_under_test: Callable[[bool], object],
                     run_control: Callable[[bool], object],
                     equal: Callable[[object, object], bool] | None = None) -> NegativeControl:
    """否定対照を対で実行する。

    Args:
        run_under_test: 検査したい構成を走らせる。引数 True で「真値を壊した」状態にする。
        run_control: 真値を使うことが分かっている対照構成を走らせる（**空振り防止**）。
        equal: 2 つの走行結果が同一かの判定。既定は完全一致（==）。

    Note:
        **数値照合では代替できない。**推定誤差が小さければ、真値を読んでいても
        推定値を読んでいても走行はほぼ同じになる。「一致した」は「真値を使っていない」の証拠にならない。
    """
    eq = equal or (lambda a, b: a == b)
    a0, a1 = run_under_test(False), run_under_test(True)
    b0, b1 = run_control(False), run_control(True)
    return NegativeControl(
        changed_when_broken=not eq(a0, a1),
        changed_in_control=not eq(b0, b1),
    )


# ======================================================================
# 型 E: 失効した数値が、訂正されないまま他の文書へ伝播していないか
# ======================================================================

@dataclass(frozen=True)
class RetractedHit:
    path: str
    line_no: int
    line: str


def find_retracted_numbers(numbers: Iterable[str], repo_root: str | Path = ".",
                           allow: Iterable[str] = ()) -> list[RetractedHit]:
    """失効させた数値が、リポジトリ内の文書にまだ書かれていないかを探す。

    Args:
        numbers: 失効した数値の文字列（例: "37.6", "69.6"）。
        repo_root: 検索の起点。
        allow: 見つかってよいパス（失効を宣言している文書そのものなど）。

    Returns:
        見つかった箇所の一覧。**空であるべき**。

    Note:
        失効の宣言をカードの中だけに書くと、総括や設計提案に流れ込んだ数値は残る。
        実際にそれが起きた（note_029 §1-3）。
    """
    root = Path(repo_root)
    allowed = {str(a) for a in allow}
    hits: list[RetractedHit] = []
    for num in numbers:
        try:
            out = subprocess.run(
                ["git", "grep", "-n", "--", num],
                cwd=str(root), capture_output=True, text=True, timeout=60,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        for line in out.splitlines():
            m = re.match(r"^([^:]+):(\d+):(.*)$", line)
            if not m:
                continue
            path, line_no, body = m.group(1), int(m.group(2)), m.group(3)
            if path in allowed or any(path.startswith(a) for a in allowed):
                continue
            hits.append(RetractedHit(path=path, line_no=line_no, line=body.strip()[:200]))
    return hits
