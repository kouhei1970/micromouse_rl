#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""実戦帯を **クラシック規格 19 面** へ揃える（教授裁定 2026-08-15・GO 済み）。

`competition/mazes/contest_reference/` は 18 面だったが、内訳は クラシック 17 面 ＋
**ハーフサイズ規格 1 面**（`16MM2021H_Kansai`）で、クラシック 2 面（`16MM2020CX`・
`16MM2017C_Chubu`）が欠けていた。本スクリプトで 19 面へ揃える。

  - 追加: `16MM2020CX`・`16MM2017C_Chubu`（npz を複製し XML を新規生成）
  - 除去: `16MM2021H_Kansai`（ハーフサイズ規格。クラシックの評価に使えない）
  - `manifest.json` を書き換える

**既存 17 面のファイルは 1 バイトも触らない**（実行前後のハッシュで確認する）。

    .venv/bin/python research_notes/scripts/place_contest_band.py [--apply]
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from mouse.mjcf import build_maze_robot_xml   # noqa: E402
from mouse.params import RobotParams          # noqa: E402

SRC = ROOT / "competition" / "reference_mazes" / "contest"
DST = ROOT / "competition" / "mazes" / "contest_reference"
AUDIT = ROOT / "research_notes" / "scripts" / "contest_class_audit.json"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="実際に書き換える（既定は下見のみ）")
    args = ap.parse_args()

    audit = {r["name"]: r for r in json.load(open(AUDIT, encoding="utf-8"))}
    want = sorted(n for n, r in audit.items() if r["spec"] == "classic")
    have = sorted(p.stem.replace("maze_", "") for p in DST.glob("maze_*.npz"))
    add = [n for n in want if n not in have]
    remove = [n for n in have if n not in want]
    keep = [n for n in have if n in want]

    print(f"目標 = クラシック規格 {len(want)} 面 ／ 現状 {len(have)} 面")
    print(f"  追加 {len(add)}: {add}")
    print(f"  除去 {len(remove)}: {remove}  （理由: "
          + ", ".join(f"{n} は {audit[n]['klass']} = ハーフサイズ規格" for n in remove) + "）")
    print(f"  据え置き {len(keep)} 面（**1 バイトも触らない**）")
    if not args.apply:
        print("\n下見のみ。実行するには --apply を付ける。")
        return 0

    before = {p.name: sha(p) for p in sorted(DST.iterdir()) if p.name != "manifest.json"}

    params = RobotParams()
    for n in add:
        src_npz = SRC / f"contest_{n}.npz"
        dst_npz = DST / f"maze_{n}.npz"
        shutil.copyfile(src_npz, dst_npz)
        d = np.load(dst_npz)
        build_maze_robot_xml(d["v_walls"], d["h_walls"], str(DST / f"maze_{n}.xml"),
                             model_name=f"maze_{n}", params=params)
        print(f"  + {dst_npz.name} / {n}.xml")
    for n in remove:
        for suf in (".npz", ".xml"):
            p = DST / f"maze_{n}{suf}"
            if p.exists():
                p.unlink()
                print(f"  − {p.name}")

    manifest = {
        "purpose": "実戦帯（大会実迷路）。章の完了確認の補助と、斜め・経路単位選択の実測の題材。"
                   "**学習・調整には使わない。**最終の完成判定には評価帯 20 迷路と併せて用いる"
                   "（ユーザ合意 2026-08-15）",
        "selection": "competition/reference_mazes/contest/ のうち **クラシック規格**のもの全部"
                     "（ファイル名の CX = 全日本クラシックエキスパート決勝、C = 地区・学生大会等）。"
                     "H / HX は**ハーフサイズ規格**（区画 90 mm）なので除外した",
        "n": len(want),
        "spec_breakdown": {"CX": sum(1 for n in want if audit[n]["klass"] == "CX"),
                           "C": sum(1 for n in want if audit[n]["klass"] == "C")},
        "mazes": [{"npz": f"maze_{n}.npz", "source": f"{n}.maze",
                   "klass": audit[n]["klass"], "d_true": audit[n]["d_true"]} for n in want],
        "provenance": "kerikun11/micromouse-maze-data (MIT, (c) 2020 Ryotaro Onuki), commit 762ed2b6。"
                      "詳細は competition/reference_mazes/README.md",
        "verification": "往復変換の全数検証 = tests/test_contest_maze_roundtrip.py（127 項目 PASS）。"
                        "クラス分けと難度指標 = research_notes/scripts/check_contest_maze_classes.py",
        "note": "ファイル名を maze_<面名>.npz にしてあるのは、評価器 evaluate_all が maze_*.npz しか"
                "拾わないため。中身は competition/reference_mazes/contest/ の contest_<面名>.npz と同一"
                "（seed=-1、start/goals のメタデータ付き）。"
                "全 19 面が 16×16・ゴール中央 2×2・スタート (0,0) で評価器の前提を満たす",
    }
    (DST / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    after = {p.name: sha(p) for p in sorted(DST.iterdir()) if p.name != "manifest.json"}
    untouched = all(before[k] == after[k] for k in before if k in after)
    print(f"\n据え置き分のハッシュ一致: {untouched}")
    print(f"配置後: npz {len(list(DST.glob('maze_*.npz')))} 面 / "
          f"xml {len(list(DST.glob('maze_*.xml')))} 面")
    return 0


if __name__ == "__main__":
    sys.exit(main())
