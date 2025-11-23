# Micromouse Reinforcement Learning Project

階層型強化学習を用いたマイクロマウスの自律ナビゲーション学習プロジェクト

## 概要

MuJoCo物理エンジンとStable-Baselines3 (PPO)を使用して、マイクロマウスロボットが迷路を自律的にナビゲートする能力を段階的に学習します。

### 階層型アプローチ

```
Phase 3 (High-Level Policy)
  ↓ 目標速度指令
Phase 1 (Low-Level Controller)
  ↓ モーター制御
MuJoCo Physics Simulation
```

## プロジェクト構造

```
micromouse_rl/
├── models/                     # 訓練済みモデル
│   ├── phase1_open.zip         # Phase 1: 低レベル速度制御
│   ├── phase2_slalom.zip       # Phase 2: スラローム (HRL)
│   └── phase3_maze.zip         # Phase 3: 迷路ナビゲーション (HRL)
│
├── phase1_open/                # Phase 1: 低レベル制御
│   ├── env.py                  # 環境定義
│   ├── train.py                # 訓練スクリプト
│   └── README.md
│
├── phase2_slalom/              # Phase 2: スラローム
│   ├── env.py                  # 環境定義
│   ├── train.py                # 訓練スクリプト
│   └── README.md
│
├── phase3_maze/                # Phase 3: 迷路ナビゲーション
│   ├── env.py                  # 環境定義
│   ├── train.py                # 訓練スクリプト
│   ├── analyze_final_results.py
│   ├── diagnose_behavior.py
│   ├── evaluate.py
│   └── README.md
│
├── common/                     # 共通ユーティリティ
│   ├── output_manager.py       # 出力管理
│   ├── visualization.py        # 可視化ツール
│   ├── maze_assets.py          # 迷路生成
│   └── README.md
│
├── outputs/                    # 実験結果
│   ├── phase1_open/
│   │   ├── latest/            # 最新の実験結果
│   │   └── archive/           # 過去の実験
│   ├── phase2_slalom/
│   └── phase3_maze/
│
├── assets/                     # MuJoCo XMLアセット
├── OUTPUT_STRUCTURE.md         # 出力管理仕様
├── QUICKSTART_OUTPUT.md        # 出力管理クイックガイド
└── README.md                   # このファイル
```

## 開発フェーズ

### Phase 1: 低レベル速度制御 ✅ 完了
オープンフィールドでの基本的な速度制御を学習

**タスク**: 目標速度（線速度・角速度）を正確に追従

**特徴**:
- 環境: オープンフィールド（壁なし）
- 観測: ロボット状態（位置、速度、姿勢）+ 目標速度
- 行動: モーター指令（左右輪）
- 報酬: 速度追従誤差の最小化

**成果**:
- 訓練ステップ: 1,000,000
- 速度追従精度: 高精度（詳細はphase1_open/README.md参照）
- モデル: `models/phase1_open.zip`

### Phase 2: スラローム（階層型RL） ✅ 完了
L字カーブのスラローム走行を学習

**タスク**: Phase 1の低レベル制御を使用してスラロームコースを高速走行

**特徴**:
- 高レベルポリシーが目標速度を決定
- Phase 1の低レベル制御がモーター指令を生成
- 階層型アーキテクチャの実証

**成果**:
- スラローム走行の習得
- 階層型RLの有効性を確認

### Phase 3: 迷路ナビゲーション（階層型RL） 🚧 進行中
ランダム迷路での2セルナビゲーション

**タスク**: 7x7迷路内で中間セル経由でゴールに到達

**特徴**:
- 環境: ランダム生成7x7迷路
- 高レベルポリシーがナビゲーション戦略を学習
- Phase 1の低レベル制御を活用
- 距離センサーによる壁検出

**現在の性能** (1M steps訓練):
- 成功率: 85%
- 平均ステップ数: 510
- 課題: 蛇行による経路効率の低下

**改善計画**:
- 報酬構造の調整（時間ペナルティ強化）
- スムーズネスペナルティの追加

## クイックスタート

### 環境セットアップ

```bash
# 仮想環境作成
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存パッケージインストール
pip install gymnasium mujoco stable-baselines3 matplotlib opencv-python
```

### Phase 1: 低レベル制御の訓練

```bash
python phase1_open/train.py
```

出力: `outputs/phase1_open/latest/`

### Phase 3: 迷路ナビゲーションの訓練

```bash
# Phase 1モデルが必要
python phase3_maze/train.py
```

出力: `outputs/phase3_maze/latest/`

### 評価・分析

```bash
# 性能評価
python phase3_maze/evaluate.py

# 詳細分析
python phase3_maze/analyze_final_results.py

# 動作診断
python phase3_maze/diagnose_behavior.py
```

## 出力管理

すべてのPhaseで統一的な出力構造を使用:

```
outputs/phase{X}/
├── latest/              # 最新の実験結果（すぐにアクセス可能）
│   ├── learning_curve.png
│   ├── evaluation.mp4
│   ├── metrics.json
│   └── model_info.txt
└── archive/             # 過去の実験（タイムスタンプ付き）
    └── YYYYMMDD_HHMMSS/
```

詳細: [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md), [QUICKSTART_OUTPUT.md](QUICKSTART_OUTPUT.md)

## 技術スタック

- **物理シミュレーション**: MuJoCo
- **強化学習**: Stable-Baselines3 (PPO)
- **環境インターフェース**: Gymnasium
- **可視化**: Matplotlib, OpenCV
- **言語**: Python 3.10+

## 主要な成果

1. **階層型RLの実装**: 低レベル制御と高レベルナビゲーションの分離
2. **ランダム迷路生成**: DFSアルゴリズムによる多様な訓練環境
3. **統一的な出力管理**: 実験結果の体系的な管理・比較
4. **包括的な診断ツール**: 学習挙動の詳細分析

## ドキュメント

- [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md) - 出力ディレクトリ構造の詳細仕様
- [QUICKSTART_OUTPUT.md](QUICKSTART_OUTPUT.md) - 出力管理のクイックガイド
- [common/README.md](common/README.md) - 共通ユーティリティの説明
- [phase1_open/README.md](phase1_open/README.md) - Phase 1詳細
- [phase2_slalom/README.md](phase2_slalom/README.md) - Phase 2詳細
- [phase3_maze/README.md](phase3_maze/README.md) - Phase 3詳細

## トラブルシューティング

### MuJoCo関連エラー
```bash
# MuJoCo正常インストール確認
python -c "import mujoco; print(mujoco.__version__)"
```

### 訓練が進まない
1. Phase 1モデルが訓練済みか確認
2. GPU/CPUリソース確認
3. ログファイル確認

### 出力が見つからない
```bash
# 最新結果は常にlatest/にある
ls outputs/phase3_maze/latest/
```

## 今後の展開

- [ ] Phase 3の蛇行問題の解決
- [ ] より大きな迷路サイズへの拡張
- [ ] 実機への転移学習
- [ ] マルチゴールタスクへの拡張

## ライセンス

このプロジェクトは研究・教育目的で開発されています。

## 謝辞

- MuJoCo physics engine
- Stable-Baselines3 team
- Gymnasium project
