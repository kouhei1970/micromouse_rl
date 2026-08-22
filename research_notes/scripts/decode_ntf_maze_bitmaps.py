"""NTF ヒストリーアーカイブの迷路 BMP を読み取る。

書式（実測）: 194x194・24bit・色は黒と赤の 2 色のみ。
格子の間隔 12 画素、壁の太さ 2 画素。柱は格子点 (12i, 12j)。
画像の行 0 が北。壁は赤。
"""
import glob, json, re, os
import numpy as np
from PIL import Image

P, T = 12, 2   # 格子の間隔 / 壁の太さ


def decode(path):
    a = np.array(Image.open(path).convert("RGB"))
    if a.shape[0] != a.shape[1]:
        return None
    red = (a[:, :, 0] > 128) & (a[:, :, 1] < 128) & (a[:, :, 2] < 128)
    N = (a.shape[0] - T) // P
    if N * P + T != a.shape[0]:
        return None
    v = np.zeros((N + 1, N), dtype=np.uint8)   # 縦壁
    h = np.zeros((N, N + 1), dtype=np.uint8)   # 横壁
    for y in range(N):
        r_mid = P * (N - 1 - y) + P // 2       # その区画の中ほどの画素行
        for x in range(N + 1):
            v[x, y] = red[r_mid, P * x]
    for x in range(N):
        c_mid = P * x + P // 2
        for y in range(N + 1):
            h[x, y] = red[P * (N - y), c_mid]
    return v, h, N


def meta(path):
    b = os.path.basename(path)[:-4]
    m = re.match(r"(.+?)_(\d*)_?(\d{4})_classic_(\w*)_(\w*)_(\d+)x(\d+)(_maybe)?$", b)
    if not m:
        m2 = re.search(r"_(\d{4})_", b)
        return dict(name=b, series=b.split("_")[0], year=int(m2.group(1)) if m2 else None,
                    cls="", stage="", maybe="maybe" in b)
    return dict(name=b, series=m.group(1), year=int(m.group(3)),
                cls=m.group(4), stage=m.group(5), maybe=bool(m.group(8)))


if __name__ == "__main__":
    out = []
    for f in sorted(glob.glob("bmp/*.bmp")):
        d = decode(f)
        if d is None:
            print("読めず:", f); continue
        v, h, N = d
        info = meta(f)
        info.update(N=N, v=v.tolist(), h=h.tolist())
        out.append(info)
    json.dump(out, open("ntf_bmp_mazes.json", "w"), ensure_ascii=False)
    print("読み取り:", len(out), "面")
