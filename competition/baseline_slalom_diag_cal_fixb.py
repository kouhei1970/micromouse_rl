#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**是正 (B)** — 斜め方策が経路と逆を向いたまま走り出すのを直す
（`experiments/exp_016_diagonal/card_016diag_fixB.md`）。

2026-08-14 新設（教授裁定 = 是正案 (B) の検収合格・実装 GO）。

--------------------------------------------------------------------------
何を直すのか（**実測で特定した 1 点**）
--------------------------------------------------------------------------
**斜め経路を張った瞬間の状態**（maze_41003・4 件とも同じ）:

| 区画 | 機体の方位 | 経路の最初の方位 | **食い違い** |
|---|---|---|---|
| (0, 0) | **S（270°）** | **N（90°）** | **180°** |

**機体は帰路で南を向いてスタート区画へ入る。そこで斜めの分岐が成立し、
北を指す経路がそのまま設置され、機体は逆を向いたまま追従を始めて壁に衝突する。**

**`SlalomPolicy` は 180° 折返しを「その場旋回」で処理する実装を持っているのに、
`SlalomDiagPolicy._replan` は斜めの分岐に入ると親を呼ばないので、そこを迂回していた。**

**⇒ 欠陥は「斜め経路の作り方」ではなく「斜め経路へ入る前の向き合わせを飛ばしたこと」。**

--------------------------------------------------------------------------
どう直すのか — **親に任せる**（カード §1-1）
--------------------------------------------------------------------------
**機体の方位と経路の最初の方位が食い違うときは、斜めの分岐に入らず親へ委ねる。**
**親がその場旋回で向きを合わせ、揃った次の再計画で斜め経路が張られる。**

**⚠️ 自前でその場旋回は書かない**（書くと「その場旋回の実装」という 2 つ目の変更になる）。
**親の `_replan` も 1 行も写さない** — `_use_diag()` を False に見せるだけで、
**親の既存の分岐がそのまま働く**（`baseline_slalom_diag_cal.py` の委譲と同じ思想）。

--------------------------------------------------------------------------
閾値（**実装より前に固定した。探索していない**）
--------------------------------------------------------------------------
**採用 = カード §1-2 の案 (i)「完全一致を要求」。**

**根拠は構成上の事実**: **`build_diagonal_path` は先頭節点に接続の円弧を置かない**
（`tan_len[0] = 0`）。**したがって、どんな食い違いも追従則だけで吸収することになる。**
**案 (ii) の「45° まで許す」は見立てだけなので採らない。**
**この閾値を緩める方向へは動かさない**（カード §1-2 の宣言）。

--------------------------------------------------------------------------
無害性（カード §3 の手順 2）
--------------------------------------------------------------------------
- **(a) `align_check=False` にすると、是正前と全走行がビット単位で一致する**
- **(b) 向きが最初から一致している場合に、挙動が従来と変わらない**
  （**委譲を足したことが、一致ケースの経路を変えていない**ことの確認・教授指示）

どちらも `experiments/exp_016_diagonal/check_diag_fixb.py` が実測する。
"""
import sys

from competition.baseline_slalom_diag import SlalomDiagPolicy
from competition.baseline_slalom_diag_cal import SlalomDiagCalPolicy

from diagonal_model import DELTA8  # noqa: E402


class SlalomDiagCalFixBPolicy(SlalomDiagCalPolicy):
    """斜めの分岐へ入る前に**向きが揃っているか**を確かめる。

    揃っていなければ斜めの分岐へ入らず、**親のその場旋回に任せる**。
    """

    name = "L0-c+E1T+TR+DIAG+F0+F0b+cal0.75+clothoid45+alignB"

    def __init__(self, *args, align_check: bool = True, **kw):
        super().__init__(*args, **kw)
        # False にすると是正前とビット単位で同じ（無害性の確認用）
        self.align_check = bool(align_check)
        self._suppress_diag = False
        self._plan_cache = None
        # 報告用のカウンタ（**読むだけ。挙動に影響しない**）
        self.n_align_ok = 0        # 向きが揃っていて斜めへ入った回数
        self.n_align_defer = 0     # 向きが食い違って親へ委ねた回数

    # ------------------------------------------------------------------
    def _plan_diag(self, start_cell, heading):
        """**同じ再計画の中でだけ**結果を使い回す（経路を 2 度引かないため）。

        ⚠️ **キャッシュの寿命は 1 回の `_replan` の中だけ**である。
        `_replan` の入口で捨てるので、**地図が育っても古い経路を使うことはない**。
        """
        key = (tuple(start_cell), heading)
        if self._plan_cache is not None and self._plan_cache[0] == key:
            return self._plan_cache[1]
        out = super()._plan_diag(start_cell, heading)
        self._plan_cache = (key, out)
        return out

    # ------------------------------------------------------------------
    def _use_diag(self) -> bool:
        """親の条件に「向きが揃っていること」を足す。"""
        return super()._use_diag() and not self._suppress_diag

    # ------------------------------------------------------------------
    def _aligned(self, x: float, y: float) -> bool:
        """**経路の最初の方位が、機体の向いている方位と完全に一致するか。**

        一致の判定は**離散の方位どうしの一致**で行う（閾値を持たない）。
        判定できない場合（方位が 8 方位に無い・経路が引けない）は
        **食い違い扱い**にして親へ委ねる — **安全側**である。
        """
        from competition.baseline_slalom import pos_to_cell

        d_in = getattr(self, "_heading_dir", None)
        if d_in not in DELTA8:
            return False
        cell = pos_to_cell(x, y, self.width, self.height, self.cell_size)
        try:
            _nodes, dirs, _tr = self._plan_diag(cell, d_in)
        except Exception:
            return False
        if len(dirs) < 2:
            # 親の既存の条件（`len(dirs) < 2` なら親へ委ねる）と同じ扱い。
            # ここで False を返しても挙動は変わらない
            return False
        return dirs[0] == d_in

    # ------------------------------------------------------------------
    def _replan(self, x: float, y: float, yaw: float) -> None:
        self._plan_cache = None            # このティックの間だけ有効
        try:
            if self.align_check and super()._use_diag():
                ok = self._aligned(x, y)
                self._suppress_diag = not ok
                self.n_align_ok += int(ok)
                self.n_align_defer += int(not ok)
            return super()._replan(x, y, yaw)
        finally:
            self._suppress_diag = False
            self._plan_cache = None
