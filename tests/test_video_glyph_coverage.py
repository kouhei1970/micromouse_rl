"""動画の描画文字に、使うフォントの字形があるかを機械で検査する。

なぜ要るか（2026-08-22）: ヒラギノ角ゴシックは**太字 W7 の方が字形の数が少ない**。
`◀`(U+25C0) と `≈`(U+2248) は W4 にはあるが W7 には無く、**太字で描いた箇所だけが
豆腐（□）になって完成した動画に残った**。目視では見落とすので検査に変換する。
"""
import ast
import pathlib

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "research_notes" / "scripts"
FONTS = {
    False: "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",  # bold=False
    True: "/System/Library/Fonts/ヒラギノ角ゴシック W7.ttc",   # bold=True
}


def _cmap(path):
    from fontTools.ttLib import TTCollection

    return set(TTCollection(path).fonts[0].getBestCmap())


def _bold_of(node) -> bool | None:
    """描画呼び出しの中の font(...) から bold を読む。見つからなければ None。"""
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) == "font":
            for kw in n.keywords:
                if kw.arg == "bold" and isinstance(kw.value, ast.Constant):
                    return bool(kw.value.value)
            return False  # font(size) は既定で細字
    return None


def _drawn_strings():
    """`... .text(...)` 呼び出しの実引数にある文字列と、その太さを集める。"""
    for path in sorted(SCRIPTS.glob("*video_sensor*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in ast.walk(tree):
            if not (isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "text"):
                continue
            bold = _bold_of(call)
            if bold is None:
                continue  # フォント指定が読めない呼び出しは対象外
            for arg in call.args:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        yield path.name, call.lineno, sub.value, bold


@pytest.mark.skipif(
    not all(pathlib.Path(p).exists() for p in FONTS.values()), reason="ヒラギノが無い環境"
)
def test_drawn_glyphs_exist_in_the_weight_actually_used():
    cmaps = {b: _cmap(p) for b, p in FONTS.items()}
    missing = []
    for name, lineno, text, bold in _drawn_strings():
        for ch in text:
            if ord(ch) > 0x7F and ord(ch) not in cmaps[bold]:
                w = "W7(太字)" if bold else "W4(細字)"
                missing.append(f"{name}:{lineno} {w} に {ch!r} (U+{ord(ch):04X}) の字形が無い")
    assert not missing, "豆腐になる文字がある:\n  " + "\n  ".join(sorted(set(missing)))
