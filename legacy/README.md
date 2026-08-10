# legacy/ — 凍結・変更禁止

本ディレクトリは 2026-08-10 のプログラム再編で凍結された旧 Phase1〜4 系の実装一式である。
**過去実験の再現性保全のためにのみ残しており、新規作業（実装・調査・報告書執筆いずれも）で
参照・流用しないこと。** 詳細な再編方針・現行プログラムの正本は `docs/RESEARCH_PLAN.md` を参照。

## 凍結理由: 既知の重大欠陥

`docs/MODEL_VERIFICATION_PLAN.md` §2 の反証検証（独立再実装による定量確認）で確定した、
このロボットモデル（`legacy/assets/micromouse_*.xml`）の物理的な欠陥一覧。C1〜C6 はいずれも
`critical`/`major` 判定で、旧 Phase1〜3 の学習結果はこれらの欠陥を織り込んだ数値であり、
現行 `mouse/`・`assets/mouse_v2.xml` の結果とは比較できない。

| # | 欠陥 | 深刻度 |
|---|------|--------|
| C1 | 車輪・モータ質量が非現実的（車輪 50 g・密度が鉛(11.34 g/cm³)超の 12.475 g/cm³、モータ箱 50 g）。総質量 233.25 g のうち車輪+モータが 85.75% を占める | critical |
| C2 | 接触パラメータ（solref/solimp）が MuJoCo 既定値のまま機体スケールに対して過大に柔らかく、壁への貫入・すり抜け（トンネリング）が発生（3.6 m/s で 62%、3.8 m/s 以上で 100%） | critical |
| C3 | キャスタの `friction="0 0 0"` 指定が MuJoCo の摩擦結合則（優先度同位では要素ごと最大値を採用）により床既定値 μ=1.0 で上書きされ、摩擦ゼロという設計意図が全 Phase で無効化されていた | critical |
| C4 | 車輪の回転慣性（並進換算等価質量）が直進ダイナミクスの過半（有効慣性質量の 52.96%）を支配し、車体本来の運動から乖離 | major |
| C5 | ヨー慣性の 72.9% が車輪・モータ質量に由来し、旋回応答が実機と本質的に乖離 | major |
| C6 | 最大トルク印加時、車輪-床の摩擦限界がトルク上限要求推進力の 40% しかなく、発進がスリップ律速（トルク律速の設計値通りに加速しない） | major |

（要約。定量値・検証根拠・反証プロトコルの詳細は `docs/MODEL_VERIFICATION_PLAN.md` §2 を参照）

## 移設前後のパス対応表

| 旧パス（凍結前） | 新パス（凍結後） |
|---|---|
| `phase1_open/` | `legacy/phase1_open/` |
| `phase2_slalom/` | `legacy/phase2_slalom/` |
| `phase3_maze/` | `legacy/phase3_maze/` |
| `phase4_speed/` | `legacy/phase4_speed/` |
| `assets/micromouse_open.xml` | `legacy/assets/micromouse_open.xml` |
| `assets/micromouse_random_3x3.xml` | `legacy/assets/micromouse_random_3x3.xml` |
| `assets/micromouse_random_4x4.xml` | `legacy/assets/micromouse_random_4x4.xml` |
| `assets/micromouse_random_5x5.xml` | `legacy/assets/micromouse_random_5x5.xml` |
| `assets/micromouse_random_7x7.xml` | `legacy/assets/micromouse_random_7x7.xml` |
| `assets/micromouse_slalom.xml` | `legacy/assets/micromouse_slalom.xml` |
| `models/phase1_open.zip` | `legacy/models/phase1_open.zip` |
| `models/phase1_test_model.zip` | `legacy/models/phase1_test_model.zip` |
| `models/phase2_slalom.zip` | `legacy/models/phase2_slalom.zip` |
| `common/robot_builder.py` | `legacy/common/robot_builder.py` |
| `common/maze_generator.py` | `legacy/common/maze_generator.py` |
| `common/mjcf_builder.py` | `legacy/common/mjcf_builder.py` |
| `common/maze_assets.py` | `legacy/common/maze_assets.py` |
| `common/demo_xml_builder.py` | `legacy/common/demo_xml_builder.py` |

`common/output_manager.py` は新系統（`competition/`・`mouse/`）でも継続使用するため移設していない。
`common/visualization.py`・`common/extract_learning_curve.py` は新系統から未使用だが、
特定 Phase に依存しない汎用ユーティリティのため `common/` に残している
（`extract_learning_curve.py` は `phase3_maze.env` を直接 import しており、本移設後にそのまま
実行するとインポートエラーになる。新系統で使う場合は import パスの更新が必要）。

## 新系統との関係

現行の研究プログラムは `mouse/`（ロボット v2）・`competition/`（凍結評価ハーネス）・
`experiments/`（実験カード）・`research_notes/`（研究ノート）で構成される、本ディレクトリとは
独立した新規実装である。`competition/maze_gen.py` の DFS 迷路生成ロジックは
`legacy/phase3_maze/maze_generator.py` の該当部分を移植したものであり、その移植元として
一度だけ参照された（移植後は独立、`legacy/` への実行時 import 依存なし）。

読む順序・現行の正本は `docs/RESEARCH_PLAN.md` → `docs/MODEL_VERIFICATION_PLAN.md` →
`docs/ROBOT_SPEC.md` → `research_notes/` を参照。

## 追記: レガシー専用スクリプトの移設（2026-08-10）

以下は旧 Phase1〜4 のコードにのみ依存する解析・デモ用スクリプトであり、
本ディレクトリへ併せて移設した（移設前のパスからは import が解決できないため）:

| 移設前 | 移設後 | 依存 |
|---|---|---|
| `common/extract_learning_curve.py` | `legacy/common/extract_learning_curve.py` | `phase3_maze.env` |
| `docs/generate_demo.py` | `legacy/generate_demo.py` | `common.demo_xml_builder`、`models/phase{1,2,3}_*.zip` |

いずれも凍結対象であり、実行するには sys.path の調整が必要（新系統からは使用しない）。
