# mazes/ — 迷路データベース

マイクロマウス（クラシック競技）の迷路を、**引ける形**で集めた置き場である。
評価ハーネス（`competition/`）専用の資産ではなく、学習・解析・歴史照合のどれからも
参照される共有の資源として `competition/` の外（リポジトリ直下）に置いてある。

設計の経緯は `research_notes/note_036_maze_database.md` を、座標系の取り決めは
`docs/COORDINATE_SYSTEM.md` を見ること。

## 何が入っているか

| 種別（ディレクトリ） | 面数 | 中身 |
|---|---|---|
| `contest/` | 155 | 実在した競技迷路（NTF 由来 102 面 ＋ kerikun11 由来 53 面。下記「出所」参照） |

`reserved/`（確保済みの評価用迷路など、ファイルそのものを正とするもの）は
今回はまだ入れていない。

`INDEX.tsv` は生成物（属性 ＋ 指標を迷路 1 面につき 1 行）。手で編集しない。
`research_notes/scripts/build_maze_index.py` で作り直せる。

## 出所

### 1) NTF ヒストリーアーカイブ（102 面・`source_type: bmp`）

- **公益財団法人ニューテクノロジー振興財団（NTF）の「全日本マイクロマウス大会
  ヒストリーアーカイブ事業」**が公開している迷路図（BMP 画像）から読み取った
- 出所ページ: https://www.ntf.or.jp/mouse/history/index.html
  （「過去の迷路図(編集中)」— 出所ページ自身が編集中と明記している）
- 取得日: 2026-08-23
- 変換元・読み取り方の詳細: `competition/mazes/contest_historical/manifest.json`
  （変換元の npz 一式。**本データベースはそこからの変換であり、変換元は変更していない**）
  ／`research_notes/scripts/decode_ntf_maze_bitmaps.py`（画像→壁配列の読み取り機）
- 取り込みの経緯・迷路の難度分析: `research_notes/note_035_historical_contest_mazes.md`

### 2) kerikun11/micromouse-maze-data（53 面・`source_type: ascii`）

- **MIT License / Copyright (c) 2020 Ryotaro Onuki**
- リポジトリ: https://github.com/kerikun11/micromouse-maze-data
- 取得コミット: `762ed2b68735ea29148c6a1251a90ed0651ff26b`（2024-02-22）。
  本リポジトリでは `competition/reference_mazes/micromouse-maze-data/`（80 面・無改変）に
  ベンダリング済み。2026-08-23 に GitHub 上流の `master` HEAD と突き合わせ、同一コミットで
  あることを確認した（上流に新しい面は増えていない）
- 全 80 面のうち、実在の競技結果と判断した 53 面を取り込んだ。除いた 27 面の内訳・
  取り込みの判断・id の付け方（既存の規則に揃えられない台湾・特別迷路の扱いを含む）は
  `research_notes/scripts/import_kerikun11_mazes.py` の冒頭コメントに詳しく書いてある
- 謝辞: **`micromouseonline/mazefiles`**（https://github.com/micromouseonline/mazefiles ）は
  データとしては取り込んでいないが、「何年の何が存在するか」を把握する索引として調査に用いた
  （`note_036` §8・§9 のユーザ決定）。ライセンス表記が無く、編者自身も「決定版とみなすべきでない」
  と明記しているため

各 `.maze` ファイルの前書きにも、その面ごとの出所（`source`）・出所 URL
（`source_url`）・取得日（`retrieved`）を必ず記す。

### 資料の形式と確からしさは別の軸

前書きの `source_type` は資料そのものの見た目（bmp・ascii・pdf・binary・generated）、
`confidence` は確からしさの評価（confirmed・single-source・disputed）である。
**この 2 つは別物**（`source_type` が変わっても `confidence` が自動的に決まるわけではない）。

同じ迷路が形式の違う複数の資料に載っていることが分かっている
（`note_035`: NTF `taikai/map.html` のアスキー版 15 面と、本データベースの BMP 版は
1980〜1994 年の 15 面中 5 面で壁が食い違う。この 15 面の取り込みは段4で別途行う）。

### 🔴 指紋の突き合わせで分かったこと（NTF ⇔ kerikun11、2026-08-23）

`content_sha256` で 155 面を機械的に突き合わせたところ、一致する組が 2 つ見つかった
（検査: `tests/test_maze_db.py::test_fingerprints_can_be_grouped_mechanically_and_find_known_matches`）。

1. **`AllJapan_033_2012_exp_fin`（NTF）と `AllJapan_033_2012_exp_fin__kerikun11`**:
   同じ「全日本2012年エキスパート決勝」で、独立した2つの出所の壁配置が完全一致した。
   両方とも `confidence: confirmed` に格上げした
2. 🔴 **`APEC2002_2002`（NTF）と `AllJapan_039_2018_exp_fin`（kerikun11）**:
   **別の大会・別の年のはずなのに壁配置が完全一致した。** 2002年APECの迷路と、
   kerikun11 側で2018年全日本エキスパート決勝と推定した迷路が同一である。同一迷路の
   意図的な再利用か、どちらかのラベル誤りかは判断できていない。**両方の `notes` に
   相互参照を記して confidence は変えていない**（未解決のまま報告する）

なお、**既存 102 面の内部重複**（`note_035` の追記: `AllJapan_013_1992_exp_fin` と
`AllJapan_016_1995_exp_fin` が同一）については、kerikun11 の全日本データが 2012 年以降しか
無いため、どちらが正しいかを裏付ける材料は今回得られなかった。

取り込んだ 53 面はすべて `source_type: ascii`・`confidence: single-source`
（上記2件で個別に confirmed へ格上げしたものを除く）。

## 形式 — 1 迷路 1 ファイル（`.maze`）

前書き（簡易 YAML 風）と、アスキー図（`+---+` 形式）の 2 段構成。

```
---
id: AllJapan_015_1994_exp_fin
size: [16, 16]
start: [0, 0]
start_heading: N
goal: [[7, 7], [8, 7], [7, 8], [8, 8]]
series: AllJapan
edition: 15
year: 1994
class: expert
stage: final
source_type: bmp
source: "NTF ヒストリーアーカイブ MazeImage/AllJapan_015_1994_classic_exp_fin_16x16.bmp"
source_url: "https://www.ntf.or.jp/mouse/history/index.html"
retrieved: 2026-08-23
confidence: single-source
content_sha256: 238387610d36b0d747a4f266e7949856e8d612bc294a7451fe4180c279979ee2
---
+---+---+---+---+ ...
|               | ...
+   +---+   +   + ...
...
```

### 前書きの項目

| キー | 意味 |
|---|---|
| `id` | 迷路の識別子（ファイル名から拡張子を除いたもの） |
| `size` | `[width, height]`（区画数） |
| `start` | スタート区画 `[x, y]`（クラシックは南西の隅 `[0, 0]`） |
| `start_heading` | 発進方向（`N`/`E`/`S`/`W`）。競技規約では必ず `N`（`docs/COORDINATE_SYSTEM.md` §1） |
| `goal` | ゴール区画の一覧（クラシックは中央 2×2 の 4 区画） |
| `series`/`edition`/`year`/`class`/`stage` | 大会の系列・回・年・クラス（`expert`/`freshman`/空）・段階（`final`/`preliminary`/空） |
| `source_type` | 資料の形式。`bmp`/`ascii`/`pdf`/`binary`/`generated` のいずれか |
| `source`/`source_url`/`retrieved` | 出所の詳細・取得日 |
| `confidence` | 確からしさ。`confirmed`（複数資料が一致）/`single-source`（1 資料のみ）/`disputed`（資料間で食い違う） |
| `disputes` | （`confidence: disputed` のときのみ）食い違う相手の `id` 一覧 |
| `notes` | 補足（自由記述、無ければ省略） |
| `content_sha256` | **壁の並びだけ**から決まる指紋（年・出所・確度は含めない）。同じ迷路が別の出所から入っても同じ値になる |

### アスキー図の読み方

- 柱 `+`、横壁 `---`、縦壁 `|`、開通は空白、**未知の壁は `?`**（3 文字の区間は `???`）
- **図は上が北**（`docs/COORDINATE_SYSTEM.md` の「上から見て北が上」の言い換え）
- 区画の内側 3 文字にスタート `S`・ゴール `G` を表示する（前書きの `start`/`goal` が正。
  図の表示はそれと必ず一致する — 読み込み器が食い違いを検査で弾く）
- 詰めた書き方（`|` と `_`）は採らない。**壁 1 枚と文字位置が 1 対 1 で対応する**書き方だけを使う
  （`note_035` で実際に踏んだ落とし穴のため。詳細は `research_notes/note_036_maze_database.md` §2-2）

## 引き方

```python
from common.maze_db import MazeDB

db = MazeDB()                                   # mazes/ を読み込む
rec = db.get("AllJapan_015_1994_exp_fin")        # 1 面
finals = db.query(kind="contest", stage="final", confidence="single-source")
old = db.query(year=range(1980, 1995), series="AllJapan")
v_walls, h_walls = db.walls(rec)                 # 既存コード（0=壁なし・1=壁あり）へそのまま渡せる
```

`v_walls`/`h_walls` の添字規約は `classic/maze_map.py`・`competition/evaluator.py` と同じ:

- `v_walls[x, y]`: 区画 `(x-1,y)` と `(x,y)` の間の縦壁。形状 `(width+1, height)`
- `h_walls[x, y]`: 区画 `(x,y-1)` と `(x,y)` の間の横壁。形状 `(width, height+1)`

`walls()` は未知の壁（`?`）が 1 枚でも残っている迷路には使えない（例外を返す）。
0/1 表現には未知を表す枠が無く、黙って丸めると情報が失われるため。

未知壁を含む面（`Cheese_2017_k11h` など kerikun11 の Cheese 系 3 面。下記参照）を扱うときは
`unknown` 引数で丸め方を明示する:

```python
v, h = db.walls(rec, unknown="wall")   # 未知を壁ありとみなす（悲観・保守的）
v, h = db.walls(rec, unknown="open")   # 未知を壁なしとみなす（楽観）
v, h = db.walls(rec)                   # 省略時は従来どおり例外（振る舞いは変えていない）
```

## 未知の壁（`?`）を含む面

kerikun11 の原資料は未知壁を `.`（横は ` . `・縦は `.`）で表す。本データベースでは
そのまま `?`／`???` に写している（意味は「未知」のまま 1 対 1 で対応する）。

- 未知壁を含む面は **`Cheese_2017_k11h`・`Cheese_2019_k11h`・`Cheese_2019_k11h_cand`** の
  3 面のみ（kerikun11 の「Cheese」系。ゴールが中央でない特別な迷路で、実際の競技結果か
  候補案かは未確認）
- **外周・スタート区画の開口・ゴール区画内部はすべて既知**であり、本データベースの
  構造検査（外周が完全・スタートから到達できる・開口が1つ 等）はこの3面でも成立する
  （悲観的な丸め＝未知を壁とみなしても到達可能なことを確認済み）

## 検査

`tests/test_maze_db.py`（`.venv/bin/python -m pytest tests/test_maze_db.py -v`）。

1. 往復: すべての面で `dumps(load(f))` が元のファイルと 1 文字単位で一致する
   （未知壁を含む Cheese 系 3 面も同じ検査に含まれる）
2. 構造: 外周が完全・ゴール 2×2 の内側に壁が無い・スタートから到達できる・
   スタート区画の開口が 1 つだけ
3. 索引の同期: `INDEX.tsv` を生成し直したものが版管理下のものと一致する
4. NTF 由来 102 面すべてで、変換元の npz（`competition/mazes/contest_historical/`）と
   壁配列がバイト単位で一致する
5. `content_sha256` が壁だけから決まること（前書きの他の項目を変えても指紋が変わらない）
6. アスキーの文字位置の取り違えを捕まえる検査（`+---+` の各文字がどの壁に対応するかを固定する）
7. kerikun11 取り込み（段3）固有の検査: `walls(rec, unknown=...)` の3通りの振る舞い・
   `disputes` の相互参照・指紋の機械的な洗い出しが既知の3組（本README「指紋の突き合わせ」
   参照）を見つけること
