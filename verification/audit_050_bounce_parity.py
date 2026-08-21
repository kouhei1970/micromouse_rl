"""
verification/audit_050_bounce_parity.py
================
光線追跡（`verification/audit_050_raycast.py`）の**多重反射の経路が正しいこと**を、
答えが分かっている配置で確かめる。

## なぜ要るか

実迷路で反射回数ごとの寄与を測ると、**3 回目の増分が 2 回目より大きい**姿勢が多数出た
（20 姿勢中 16 件）。拡散反射率 0.8 の受動的な系では 1 回増えるごとに減衰するはず、という
素朴な予想に反するため、教授セッションは当初これを実装の誤りと判断した（`AUDIT_050_RESULT` 参照）。

**その判断は誤りだった。**原因は「面の偶奇」である:

- LED が照らす点と PT が見る点は、**同じ平らな壁の上**にある（光軸がほぼ平行なため）
- **平らな面を出た光は、その面へ直接は戻れない**
- したがって「PT が見ている点」を間接的に照らすには
  **壁 → 床（または別の壁）→ 壁 → PT** の経路が要る。これは反射 **3 回**である
- 反射 2 回で PT に届くには、最後の反射点が PT の狭い視野（半値角 5°）に入っていなければ
  ならないが、床や側壁はそこに入らない。だから 2 回目は小さい

## この配置で何が起きるべきか（予測）

| 場面 | 2 回目 | 3 回目 | 4 回目 |
|---|---|---|---|
| 壁 1 枚のみ（床なし） | **ちょうど 0** | **ちょうど 0** | **ちょうど 0** |
| 壁 ＋ 床 | 小さい | **大きい** | **ほぼ 0** |
| 壁 ＋ 床 ＋ 左右の側壁 | 小さい | 大きい | 小さいが 0 ではない |

壁＋床で 4 回目がほぼ 0 になるのは、経路をたどると分かる。最後の反射点は必ず壁でなければ
ならず（PT の視野）、壁の直前は壁以外＝床、その前は床以外＝壁、その前は壁以外＝床 …と
交互になるので、**壁で終わる 4 回反射の経路が存在しない**。側壁を足すと
「壁 → 側壁 → 床 → 壁」が通るので、4 回目がわずかに立つ。

使い方:
  .venv/bin/python verification/audit_050_bounce_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classic.geometry import Rect
from verification.audit_050_raycast import Sensor, raycast_response

I_FULL = 0.8298934
N_RAYS = 120_000
SEED = 777001
SENSOR = Sensor(name="probe", pos=(0.0, 0.0, 0.010), axis=(1.0, 0.0, 0.0))


def bounce_increments(rects, include_floor):
    """反射 1 回の値と、2/3/4 回目の増分（満量比）を返す。"""
    vals = [raycast_response(SENSOR, (0.0, 0.0, 0.0), rects, n_rays=N_RAYS, seed=SEED,
                             max_bounces=k, include_floor=include_floor,
                             wall_height_m=0.05, max_range_m=0.35)
            for k in (1, 2, 3, 4)]
    return vals[0], [(vals[k] - vals[k - 1]) / I_FULL for k in (1, 2, 3)]


def scenes(distance_mm):
    """壁のみ / 壁＋床 / 通路（壁＋床＋側壁）の 3 つの場面を作る。"""
    wall = [Rect(cx=distance_mm / 1000.0 + 0.006, cy=0.0, hx=0.006, hy=0.084)]
    corridor = wall + [Rect(cx=0.0, cy=0.090, hx=0.30, hy=0.006),
                       Rect(cx=0.0, cy=-0.090, hx=0.30, hy=0.006)]
    return (("壁1枚のみ（床なし）", wall, False),
            ("壁1枚＋床", wall, True),
            ("壁＋床＋左右の側壁（通路）", corridor, True))


def main():
    print("=" * 78)
    print("多重反射の経路の検算: 面の偶奇（答えが分かっている配置）")
    print("=" * 78)
    ok = True
    for distance_mm in (40, 20):
        print(f"\n--- 壁まで {distance_mm}mm ---")
        for label, rects, floor in scenes(distance_mm):
            direct, inc = bounce_increments(rects, floor)
            print(f"  {label:30s} 1回={direct:.6f} | "
                  f"増分 2回目={inc[0]:+.6f} 3回目={inc[1]:+.6f} 4回目={inc[2]:+.6f}")
            if label.startswith("壁1枚のみ"):
                # 平面 1 枚では、面を出た光は戻れない。増分は厳密にゼロでなければならない。
                if any(abs(v) > 0.0 for v in inc):
                    print("    ✗ 平面 1 枚なのに多重反射の寄与が出た（光線が面に潜り込んでいる疑い）")
                    ok = False
            elif label == "壁1枚＋床":
                # 壁で終わる 4 回反射の経路は存在しないので、4 回目は 3 回目よりずっと小さい。
                if not (inc[1] > inc[0] and abs(inc[2]) < 0.1 * inc[1]):
                    print("    ✗ 壁＋床で予測した並び（3回目 > 2回目、4回目はほぼ0）にならなかった")
                    ok = False
    print("\n" + ("すべて予測どおり。多重反射の経路に誤りは見つからない。" if ok
                  else "予測と違う。多重反射の経路を疑うこと。"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
