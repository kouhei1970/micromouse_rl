# 大会実迷路の参照データ（基準データ）

本ディレクトリは、**実際のマイクロマウス大会で使われた 16x16 迷路 42 面**と、その変換・
指標計算スクリプト一式を保管する。生成迷路（`competition/mazes/`）の難度が実競技の
分布に入っているかを判定するための**基準データ**であり、追加検査・Step 0（敵対的迷路
設計）・将来の再検証で繰り返し参照する。

2026-08-11 に、揮発する一時領域（セッション固有 scratchpad）からリポジトリ内へ移設した
（教授裁定）。移設にあたり全スクリプトを新しい配置で再実行し、`contest_stats.csv` が
**バイト単位で同一に再生成される**ことと、`validate.py` の往復検証が**全 42 面 OK** で
あることを確認済み。

---

## 1. 出所とライセンス

| 項目 | 内容 |
|---|---|
| 上流リポジトリ | **kerikun11/micromouse-maze-data** |
| URL | https://github.com/kerikun11/micromouse-maze-data |
| 取得コミット | `762ed2b68735ea29148c6a1251a90ed0651ff26b`（`add MM2023HX maze`, 2024-02-22） |
| 取得日 | 2026-08-10（前任の学生 A セッション） |
| ライセンス | **MIT License** |
| 原著作者 | **Copyright (c) 2020 Ryotaro Onuki** |
| ライセンス全文 | `micromouse-maze-data/LICENSE`（上流のものをそのまま同梱） |
| 上流 README | `micromouse-maze-data/README.md`（`.maze` テキスト形式の仕様が書かれている。パーサの実装根拠） |

MIT ライセンスの条件に従い、**著作権表示とライセンス全文を同梱**している。再配布・改変時も
`micromouse-maze-data/LICENSE` を必ず添えること。`micromouse-maze-data/data/*.maze`
（80 本）は上流の原本を**無改変で**そのまま置いている。

---

## 2. ディレクトリ構成

```
competition/reference_mazes/
├── README.md                       ← 本ファイル
├── micromouse-maze-data/
│   ├── LICENSE                     ← 上流 MIT ライセンス全文（必須同梱）
│   ├── README.md                   ← 上流 README（.maze 形式の仕様）
│   └── data/*.maze                 ← 上流の原本 80 本（無改変）
├── contest/
│   ├── contest_<面名>.npz          ← 変換済み 42 面（v_walls (17,16) / h_walls (16,17)）
│   └── manifest.json               ← 変換の記録（総数・除外理由・スタート/ゴール座標）
├── contest_stats.csv               ← 42 面の指標（1 面 1 行・36 列）
├── contest_stats_detail.json       ← 同・分布や集計を含む詳細
├── parse_maze.py                   ← .maze テキストのパーサ
├── convert_all.py                  ← .maze → npz 一括変換（16x16 かつ壁完全既知のみ）
├── validate.py                     ← 往復検証（npz → .maze テキスト逆生成 → 原文と1文字単位比較）
├── compute_stats.py                ← 42 面の指標計算（contest_stats.csv/json を生成）
├── analyze_ours.py                 ← 生成迷路側の同一指標を計算（比較用）
├── crosscheck_eval.py              ← 生成迷路の指標を既知の実測値・rule_audit.json と突合
└── prototype/                      ← 案 3（経路保護型除去）の予備実験スクリプト（記録用）
```

### 実行方法

すべて本ディレクトリを作業ディレクトリとして実行する（パスは `__file__` 基準）。

```bash
cd competition/reference_mazes
../../.venv/bin/python convert_all.py     # .maze → npz（contest/ を作り直す）
../../.venv/bin/python validate.py        # 往復検証（全 42 面 OK が期待値）
../../.venv/bin/python compute_stats.py   # contest_stats.csv / _detail.json を再生成
```

---

## 3. 42 面がどう選ばれたか

上流 `data/` の **80 ファイル**から、次の 2 条件で絞り込んだ結果が 42 面である
（`convert_all.py`、除外の内訳は `contest/manifest.json` に記録）。

1. **16x16 であること** — 80 本のうち 16x16 は 45 本（他は 4x4・8x8・9x9・32x32 等）
2. **壁が完全既知であること** — 45 本のうち未知壁 `.` を含む 3 本（`Cheese` 系）を除外

除外した 3 面の難度が系統的に異なる可能性は否定できない（`docs/MAZE_DIFFICULTY_REPORT.md`
§7-2 の未確認事項）。

---

## 4. 基準となる主要な指標（42 面）

`contest_stats.csv` から。**生成迷路がこの分布に入っているかを判定する。**

| 指標 | 中央値 | 範囲 |
|---|---|---|
| 真の最短距離 D_true [区画] | **63** | 40〜249 |
| 迂回率 D_true / マンハッタン距離 | **4.80** | 1.82〜17.79 |
| 独立閉路数 β = E−V+C | **20** | 1〜40 |
| 行き止まり数（次数 1 の区画） | 24 | 1〜63 |
| 最短経路の本数 | 5 | 1〜265 |
| ゴール周囲 12 区画の平均次数 | 2.17 | 1.75〜2.50 |

規定 6 項目の適合は **28/42 面（67%）**。つまり**実大会迷路の 3 分の 1 は、我々が採用して
いる 6 項目のいずれかを満たさない**（壁づたいで解ける 5 面、ゴール入口 2 箇所の 6 面、
孤立柱ありの 3 面）。したがって**実大会迷路をそのまま評価セットには使えない**。本データは
あくまで「難度の分布を合わせるための基準」として使う。

詳細な比較と考察は `docs/MAZE_DIFFICULTY_REPORT.md`（正本）を参照。

---

## 5. `prototype/` について

`docs/MAZE_DIFFICULTY_REPORT.md` §5 案 3（経路保護型の内壁除去＋D0 受理窓）を
**リポジトリに実装する前**に、scratchpad 上で試作・実測したスクリプト群である。

| ファイル | 内容 |
|---|---|
| `proto2.py` | 案 3 の試作本体（`gen_pathaware`）。§4.2 の反証実験（β を 15→30 に倍増しても D_true が動かない）もここ |
| `final.py` | 案 3 の 6 項目規定監査（20/20 適合を確認） |
| `final2.py` | 案 3 を 3 つの独立 seed ブロック（各 20 面）で実測。§5 の表の出所 |
| `decomp.py` | 生成手順の寄与分解（どの手順が最短距離を何区画縮めているか、300 seed） |
| `dist.py` / `sweep.py` | 案 1（除去枚数のみ削減）の掃引 |
| `proto.py` | 案 2（除去削減＋受理窓）の試作 |
| `ring.py` | ゴール周囲 12 区画の平均次数 |
| `forced.py` | 手順 4（強制開放 12 枚）の影響測定 |

**これらは記録であって正本ではない。** 案 3 の正本の実装は
`competition/maze_gen_v2.py`（2026-08-11 改修）である。`final.py` / `final2.py` は
`proto2.py` を `exec` で読み込むため、本ディレクトリを作業ディレクトリにして実行すること。

---

## 6. 移設にあたっての改変（1 箇所のみ）

- `analyze_ours.py` の `OUTDIR`: セッション固有 scratchpad の絶対パスがハードコード
  されていたため、`os.path.dirname(os.path.abspath(__file__))` に変更した。
  **これ以外のスクリプト・データは無改変。**
