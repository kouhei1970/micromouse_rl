"""
common/seed_bands.py
====================
**迷路 seed の帯（band）を明示し、凍結帯の誤用を止める安全弁。**

出所: 裁定 2026-08-13-R40 条件 4（学生A が exp_017 の実装直後に**凍結帯の
maze_7837 を 1 面スモークテストに使った**件の再発防止。即時・完全な自己申告により
実験は続行承認されたが、**道具の側に歯止めが無かった**ことが残った）。
実装は R11 バッチ項目 7（2026-08-14。学生B が共通ヘルパと単体テストを担当）。

--------------------------------------------------------------------------
設計の要点
--------------------------------------------------------------------------
0. **🔴 seed は「どの生成器の seed か」まで込みでないと帯が決まらない**（2026-08-14 に
   単体テストが捕らえた設計の欠陥）。M2（6x6・`mouse/maze6_gen.py`）の**学習帯 8000 以降**は、
   競技（16x16・`competition/maze_gen.py`）の**候補プール [1000, 40999] と数値が重なる**。
   同じ整数でも**生成器が違えば別の迷路**なので、`namespace` を必ず明示させる
   （`'maze6'` / `'competition'` / `'corridor'`）。**既定値は置かない** — 呼び出し側に
   帯を明示させること自体が R40 条件 4 の「帯の明示」でもある。
   （これは §9-17／R33 と同じ「同じ名前で別の量」の族である。**最初の実装は 1 つの
   名前空間だと思い込んでおり、M2 の学習をすべて止めてしまう誤りだった**）
1. **一律拒否にはしない。**gate 判定は評価帯を使うのが正しい用途なので、
   **「明示的な合言葉（`purpose='gate'` ＋ `reason` の文字列）でのみ許可」**にする。
   素の指定（既定）では**例外で止まる**。
2. **帯の定義をここで作らない。**コード側に正の定義元があるものは**それを読む**
   （`mouse.maze6_env._RESERVED_MAZE_SEEDS`）。**import 時に突き合わせ**、
   食い違ったら即座に落とす（正本を直したのにここが古い、という事故を防ぐ。
   `CLAUDE.md` の「是正が届かない経路を作らない」）。
3. **文書にしか正本が無い帯**（競技 16x16 の候補プール）は、出所を明記した上で
   ここに書く。**§9-7 が改訂されたらここも直す**（その旨をコメントに残す）。

--------------------------------------------------------------------------
帯の一覧（正本と対応）
--------------------------------------------------------------------------
| namespace | 帯 | 範囲 | 正本 | 学習に使ってよいか |
|---|---|---|---|---|
| `maze6` | `eval` | 6000-6019 | `mouse/maze6_env._RESERVED_MAZE_SEEDS`・§9-7 | ❌ gate 専用 |
| `maze6` | `validation` | 7000-7019 | 同上 | ❌ 日常判断・検証用 |
| `maze6` | `free` | 上記以外（学習は 8000 以降） | — | ✅ |
| `competition` | `eval` | **manifest の採用 20 seed** | `competition/mazes/eval/manifest.json` の `mazes[].seed` | ❌ gate 専用（reason つきで許可） |
| `competition` | `validation` | **manifest の採用 20 seed** | `competition/mazes/validation/manifest.json` | ❌ 日常判断・検証用 |
| `competition` | `pool` | 1000-40999 の**非採用 seed** | 研究計画書 §9-7（候補プール全体） | ❌ **全 purpose で不許可**（「選ばれなかった」だけで同じ生成過程にある） |
| `competition` | `free` | 上記以外 | — | ✅ |
| `corridor` | `eval` / `validation` | 3000-3019 / 5000-5019 | `mouse/corridor_env._RESERVED_COURSE_SEEDS` | ❌ M1 の評価・検証帯 |
| `corridor` | `free` | 上記以外 | — | ✅ |

使い方:
    from common.seed_bands import assert_seeds_allowed, describe_seeds

    # 学習・測定（凍結帯なら例外で止まる）。namespace は必須。
    assert_seeds_allowed(seeds, namespace="maze6", purpose="train")
    # 検証帯を意図して使う（測定・日常判断）
    assert_seeds_allowed(seeds, namespace="maze6", purpose="validate")
    # gate 判定だけは評価帯を許す（**合言葉と理由が要る**）
    assert_seeds_allowed(seeds, namespace="maze6", purpose="gate",
                         reason="M4 gate の本番判定")

    print(describe_seeds(seeds, namespace="maze6"))   # 帯の明示（ログの先頭に出す）
"""
from __future__ import annotations

# 競技（16x16）の候補プール。**正本は研究計画書 §9-7**（コード側に定義が無いので
# ここに書く。§9-7 が改訂されたら**ここも直すこと**）。
# 「学習への使用禁止の対象は採用 20 seed ではなく候補プール全体」（§9-7）。
_COMPETITION_POOL = range(1000, 41000)

# 競技帯の**採用 seed** は凍結 manifest から読む（**ここに列挙しない**。
# 2026-08-14 の是正: プール全体を一律 'pool' にすると、**採用済みの評価帯が
# purpose='gate' でも使えず、凍結帯の最終 1 回が実行できなかった**（学生A の実使用で発覚）。
# 設計原則「帯の定義を複製せず正本から読む」の適用先を凍結 manifest まで広げる）。
_COMPETITION_MANIFESTS = {
    "eval": "competition/mazes/eval/manifest.json",
    "validation": "competition/mazes/validation/manifest.json",
}
_COMPETITION_ADOPTED_CACHE = None


def _competition_adopted():
    """競技帯の採用 seed を凍結 manifest から読む（帯名 → seed の集合）。"""
    global _COMPETITION_ADOPTED_CACHE
    if _COMPETITION_ADOPTED_CACHE is None:
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        out = {}
        for band, rel in _COMPETITION_MANIFESTS.items():
            path = root / rel
            if not path.exists():
                raise AssertionError(
                    f"競技帯の manifest が無い: {path}。**帯の判定ができないので落とす**"
                    "（黙って 'pool' 扱いにすると凍結帯が gate で使えなくなる）")
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            out[band] = frozenset(int(m["seed"]) for m in data["mazes"])
        _COMPETITION_ADOPTED_CACHE = out
    return _COMPETITION_ADOPTED_CACHE


def _m2_reserved():
    """M2（6x6）の予約 seed を**実装の正本から**読む（ここで作らない）。"""
    from mouse.maze6_env import _RESERVED_MAZE_SEEDS
    return _RESERVED_MAZE_SEEDS


def _m1_reserved():
    """M1（廊下）の予約 seed を実装の正本から読む。"""
    from mouse.corridor_env import _RESERVED_COURSE_SEEDS
    return _RESERVED_COURSE_SEEDS


# 帯の判定順（先に当たったものを採る）。範囲はここで作らず、下の
# `_check_consistency_with_sources()` が実装の正本と突き合わせる。
_M2_EVAL = range(6000, 6020)
_M2_VALIDATION = range(7000, 7020)
_M1_EVAL = range(3000, 3020)
_M1_VALIDATION = range(5000, 5020)

#: 使える namespace（**既定値は置かない**。呼び出し側に明示させる）
NAMESPACES = ("maze6", "competition", "corridor")

#: `purpose` ごとに**許可する**帯（これ以外の帯の seed が混ざれば例外）。
#: 帯の名前は namespace の中でのローカル名（`eval` / `validation` / `pool` / `free`）。
_ALLOWED_BY_PURPOSE = {
    "train": ("free",),
    "validate": ("free", "validation"),
    "gate": ("free", "validation", "eval"),
}


class SeedBandViolation(RuntimeError):
    """凍結帯の seed を、許されない用途で使おうとした。"""


def _check_consistency_with_sources() -> None:
    """本モジュールの範囲が**実装の正本と一致している**ことを確かめる。

    食い違ったら即座に落とす。**正本を直したのにここが古い、という事故**
    （`CLAUDE.md`「是正が届かない経路を作らない」）を防ぐための自己検査である。
    """
    m2 = set(_m2_reserved())
    mine = set(_M2_EVAL) | set(_M2_VALIDATION)
    if m2 != mine:
        raise AssertionError(
            "common/seed_bands.py の M2 の帯が mouse/maze6_env._RESERVED_MAZE_SEEDS と "
            f"食い違っている（正本にのみ {sorted(m2 - mine)[:5]}… / "
            f"本ファイルにのみ {sorted(mine - m2)[:5]}…）。**正本に合わせて直すこと。**")
    m1 = set(_m1_reserved())
    mine1 = set(_M1_EVAL) | set(_M1_VALIDATION)
    if m1 != mine1:
        raise AssertionError(
            "common/seed_bands.py の M1 の帯が mouse/corridor_env._RESERVED_COURSE_SEEDS と "
            "食い違っている。**正本に合わせて直すこと。**")


def classify_seed(seed: int, namespace: str) -> str:
    """seed の属する帯のローカル名を返す（`free` なら学習に使ってよい）。

    **`namespace` は必須**（`'maze6'` / `'competition'` / `'corridor'`）。
    同じ整数でも生成器が違えば別の迷路であり、帯も別である（本モジュール冒頭 0.）。
    """
    if namespace not in NAMESPACES:
        raise ValueError(f"未知の namespace: {namespace!r}（{list(NAMESPACES)} のいずれか）")
    s = int(seed)
    if namespace == "maze6":
        if s in _M2_EVAL:
            return "eval"
        if s in _M2_VALIDATION:
            return "validation"
        return "free"
    if namespace == "corridor":
        if s in _M1_EVAL:
            return "eval"
        if s in _M1_VALIDATION:
            return "validation"
        return "free"
    # competition: **採用 seed は manifest から**帯を決め、残りの候補プールは 'pool'。
    # 'pool' は全 purpose で不許可（「選ばれなかった」だけで同じ生成過程にあるため。§9-7）。
    adopted = _competition_adopted()
    if s in adopted["eval"]:
        return "eval"
    if s in adopted["validation"]:
        return "validation"
    return "pool" if s in _COMPETITION_POOL else "free"


def describe_seeds(seeds, namespace: str) -> str:
    """帯の明示（**ログの先頭に出す**ための 1 行）。"""
    seeds = [int(s) for s in seeds]
    counts = {}
    for s in seeds:
        b = classify_seed(s, namespace)
        counts[b] = counts.get(b, 0) + 1
    parts = ", ".join(f"{k} × {v}" for k, v in sorted(counts.items()))
    lo, hi = (min(seeds), max(seeds)) if seeds else (None, None)
    return f"[seed 帯] namespace={namespace} n={len(seeds)}（{lo}〜{hi}）: {parts}"


def assert_seeds_allowed(seeds, namespace: str, purpose: str, reason: str = None) -> None:
    """`purpose` に対して許されない帯の seed が混ざっていたら例外を投げる。

    Args:
        seeds: 使う maze/course seed の列
        namespace: **必須**。`'maze6'`（6x6 の M2）／`'competition'`（16x16 の競技）／
                   `'corridor'`（M1 の廊下）。**同じ整数でも生成器が違えば別の迷路**
        purpose: `'train'`（学習・試作）／`'validate'`（日常判断・測定）／
                 `'gate'`（**評価帯を使う本番判定。`reason` が必須**）
        reason: `purpose='gate'` のときに必須の理由（記録に残す合言葉）

    Raises:
        SeedBandViolation: 許されない帯の seed が含まれていた／gate に理由が無い
        ValueError: 未知の purpose
    """
    if namespace not in NAMESPACES:
        raise ValueError(f"未知の namespace: {namespace!r}（{list(NAMESPACES)} のいずれか）")
    if purpose not in _ALLOWED_BY_PURPOSE:
        raise ValueError(f"未知の purpose: {purpose!r}（{list(_ALLOWED_BY_PURPOSE)} のいずれか）")
    if purpose == "gate" and not (reason and str(reason).strip()):
        raise SeedBandViolation(
            "purpose='gate' は評価帯の使用を許す唯一の経路なので、**理由（reason）が必須**です"
            "（何の判定でこの帯を使うのかを記録に残すため。裁定 R40 条件 4）")
    _check_consistency_with_sources()
    allowed = _ALLOWED_BY_PURPOSE[purpose]
    bad = {}
    for s in seeds:
        band = classify_seed(s, namespace)
        if band not in allowed:
            bad.setdefault(band, []).append(int(s))
    if bad:
        detail = "／".join(f"{k}: {v[:5]}{'…' if len(v) > 5 else ''}（{len(v)} 個）"
                           for k, v in sorted(bad.items()))
        raise SeedBandViolation(
            f"namespace={namespace!r} / purpose={purpose!r} では使えない帯の seed が"
            f"混ざっています → {detail}。\n"
            f"許可される帯: {allowed}。\n"
            "凍結帯（評価・検証・競技の候補プール）は学習・試作に使えません"
            "（研究計画書 §9-7・裁定 R40）。gate 判定で評価帯を使う場合のみ "
            "assert_seeds_allowed(..., purpose='gate', reason='…') を使ってください。")
