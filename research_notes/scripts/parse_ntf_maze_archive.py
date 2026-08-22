"""NTF 迷路集（1980-2003）のアスキー迷路を読み取る。

書式（実測）:
  1 行目  : 上端（最上段の区画の北壁）。`_` が壁
  以降 N 行: 各行が上から 1 段ぶん。
             列 2i   : 区画 (i, y) の西壁（`|`）
             列 2i+1 : 区画 (i, y) の南壁（`_`）
             列 2N   : 最東端の東壁
壁の表現は本リポジトリの規約に合わせる:
  v_walls[x, y] : 区画 (x-1,y)-(x,y) 間の縦壁。形状 (W+1, H)
  h_walls[x, y] : 区画 (x,y-1)-(x,y) 間の横壁。形状 (W, H+1)
座標は docs/COORDINATE_SYSTEM.md に従い x=東、y=北、(0,0)=南西。
"""
import re, sys, json
import numpy as np


def extract_blocks(html_text):
    """<pre> の中から「アスキー迷路 + 見出し」の組を取り出す。"""
    m = re.search(r"(?is)<pre>(.*?)</pre>", html_text)
    if not m:
        return []
    body = m.group(1)
    body = re.sub(r"(?is)<[^>]+>", "", body)          # タグだけ除去（空白は保つ）
    body = body.replace("&nbsp;", " ").replace("&amp;", "&")
    lines = body.split("\n")

    blocks, cur = [], []
    for raw in lines:
        line = raw.rstrip("\r")
        if re.match(r"^[|_ ]+$", line) and len(line.strip()) > 4:
            cur.append(line)
            continue
        title = re.search(r"The\s+(\d+)(?:st|nd|rd|th)\s+All Japan[^(]*\((\d{4})\)", line)
        if title and cur:
            blocks.append({"kai": int(title.group(1)), "year": int(title.group(2)),
                           "lines": cur})
            cur = []
        elif line.strip() and not line.strip().lower().startswith("start"):
            # 迷路でも見出しでもない行が来たら、溜めていたものは捨てる
            if cur and not title:
                cur = []
    return blocks


def parse(lines):
    """アスキー迷路 → (v_walls, h_walls, W, H)。読めなければ None。"""
    lines = [l for l in lines if l.strip()]
    if len(lines) < 3:
        return None
    width_chars = max(len(l) for l in lines)
    if width_chars % 2 == 0:
        width_chars += 1
    W = (width_chars - 1) // 2
    H = len(lines) - 1                      # 1 行目は上端
    if W < 4 or H < 4:
        return None
    pad = [l.ljust(width_chars) for l in lines]

    v = np.zeros((W + 1, H), dtype=np.uint8)
    h = np.zeros((W, H + 1), dtype=np.uint8)

    # 上端（最上段の北壁）: y = H
    for x in range(W):
        if pad[0][2 * x + 1] == "_":
            h[x, H] = 1
    # 各行。行 k (1..H) は上から k 段目 → y = H - k
    for k in range(1, H + 1):
        y = H - k
        row = pad[k]
        for x in range(W):
            if row[2 * x] == "|":
                v[x, y] = 1
            if row[2 * x + 1] == "_":
                h[x, y] = 1                  # その区画の南壁
        if row[2 * W] == "|":
            v[W, y] = 1
    return v, h, W, H


def render(v, h, W, H):
    """読み取った壁から、元と同じ形のアスキーを組み立て直す（往復検証用）。

    実測で確かめた作画の規則:
      - 奇数位置 2x+1 : 区画 x の横壁があれば `_`
      - 偶数位置 2x   : 縦壁があれば `|`。無ければ「区画 x の横壁」を映して `_`
                        （最東端 2W だけは区画 W-1 の横壁を映す）
      横壁は下線 2 文字ぶんで描かれるため、柱の位置にも `_` が出る。
    """
    def line_for(y, is_top):
        row = [" "] * (2 * W + 1)
        for x in range(W):
            hw = h[x, H] if is_top else h[x, y]
            if hw:
                row[2 * x + 1] = "_"
                row[2 * x] = "_"
        if (h[W - 1, H] if is_top else h[W - 1, y]):
            row[2 * W] = "_"
        if not is_top:
            for x in range(W + 1):
                if v[x, y]:
                    row[2 * x] = "|"
        return "".join(row)

    out = [line_for(None, True)]
    for k in range(1, H + 1):
        out.append(line_for(H - k, False))
    return out


if __name__ == "__main__":
    html_text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    blocks = extract_blocks(html_text)
    print(f"見つかった迷路: {len(blocks)} 面")
    ok = ng = 0
    results = []
    for b in blocks:
        p = parse(b["lines"])
        if p is None:
            print(f"  第{b['kai']}回({b['year']}): 読み取れず")
            ng += 1
            continue
        v, h, W, H = p
        back = render(v, h, W, H)
        src = [l.ljust(2 * W + 1)[: 2 * W + 1] for l in b["lines"] if l.strip()]
        same = (len(back) == len(src)) and all(a == b2 for a, b2 in zip(back, src))
        mark = "一致" if same else "★不一致"
        print(f"  第{b['kai']:2d}回({b['year']}) {W}x{H}  往復検証: {mark}")
        ok += same
        ng += (not same)
        results.append({"kai": b["kai"], "year": b["year"], "W": W, "H": H,
                        "roundtrip": bool(same),
                        "v": v.tolist(), "h": h.tolist()})
    print(f"\n往復検証: 一致 {ok} / 不一致 {ng}")
    json.dump(results, open("ntf_mazes.json", "w"), ensure_ascii=False)
