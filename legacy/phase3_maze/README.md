# Phase 3: Maze Navigation (High-Level Policy)

階層型強化学習の高レベルポリシー - ランダム迷路内での2セルナビゲーション

## 概要

Phase 3では、Phase 1で訓練した低レベル速度制御を使用して、7x7のランダム迷路内で正確に2セル移動するタスクを学習します。

### タスク仕様
- **環境**: 7x7ランダム迷路（DFS生成）
- **スポーンエリア**: 内側5x5セル（壁際を避ける）
- **目標**: 中間セル → ゴールセル（各1セル移動）
- **成功条件**:
  - ゴールセル到達
  - 正しい向き（±15°以内）でボーナス報酬

## ファイル構成

```
phase3_maze/
├── env.py                      # MicromouseMazeEnv環境定義
├── train.py                    # 訓練スクリプト（OutputManager統合）
├── evaluate.py                 # モデル評価スクリプト
├── analyze_final_results.py   # 総合性能分析
├── diagnose_behavior.py        # 動作診断ツール
├── maze_generator.py           # DFS迷路生成
├── visualize_maze.py           # 迷路可視化
└── README.md                   # このファイル
```

## 環境仕様

### 観測空間 (8次元)
```python
[
    distances[4],      # 距離センサー (前/右/後/左)
    linear_velocity,   # 線速度 (m/s)
    angular_velocity,  # 角速度 (rad/s)
    dist_to_goal,      # ゴールまでの距離 (m)
    rel_angle          # ゴールへの相対角度 (rad, -π to π)
]
```

### 行動空間 (2次元・連続)
```python
[
    target_linear,     # 目標線速度 (-0.3 to 0.3 m/s)
    target_angular     # 目標角速度 (-3.0 to 3.0 rad/s)
]
```

### 報酬構造
```python
r_goal = 1000.0              # ゴール到達
r_goal_orientation = 200.0   # 正しい向き (±15°)
r_intermediate = 100.0       # 中間セル到達
r_time = -1.0                # 時間ペナルティ（毎ステップ）
r_collision = -10.0          # 壁衝突
r_tipover = -100.0           # 転倒
```

## 使用方法

### 1. 訓練

```bash
python phase3_maze/train.py
```

**訓練設定:**
- アルゴリズム: PPO
- 総ステップ数: 1,000,000
- 学習率: 3e-4
- バッチサイズ: 64
- n_steps: 2048
- 迷路再生成: 20エピソードごと

**出力:**
- モデル: `models/phase3_maze.zip`
- 出力: `outputs/phase3_maze/latest/`
  - `learning_curve.png` - 学習曲線
  - `evaluation.mp4` - 評価動画
  - `metrics.json` - 定量的指標
  - `model_info.txt` - モデル情報

### 2. 評価

```bash
python phase3_maze/evaluate.py
```

指定エピソード数でモデルを評価し、成功率・平均ステップ数などを表示。

### 3. 性能分析

```bash
python phase3_maze/analyze_final_results.py
```

20エピソードの詳細分析を実行:
- 成功/失敗の分類
- 報酬分布
- エピソード長分布
- 軌跡可視化

### 4. 動作診断

```bash
python phase3_maze/diagnose_behavior.py
```

蛇行などの問題を診断:
- 経路効率の計算
- 角速度の振動解析
- 観測空間の分析

## 性能指標

### 現在の性能 (1M steps訓練後)
- **成功率**: 85%
- **平均ステップ数**: 510 (成功時)
- **中間セル到達率**: 90%
- **正しい向き率**: 25%

### 課題
- 蛇行が多い（経路効率: 58-125%）
- 最適経路の1.5-2倍のステップ数

### 改善案
[diagnose_behavior.py](diagnose_behavior.py) の診断結果に基づく:
1. 時間ペナルティ強化: -1.0 → -2.0
2. スムーズネスペナルティ追加: -0.1 × |Δω|
3. 観測空間への方向ベクトル追加を検討

## 迷路生成

```bash
python phase3_maze/visualize_maze.py --size 7
```

7x7のランダム迷路を生成・可視化。DFSアルゴリズムを使用して連結性を保証。

## 階層構造

```
Phase 3 (High-Level)
    ↓ 目標速度 (linear, angular)
Phase 1 (Low-Level)
    ↓ モーター指令 (left_motor, right_motor)
MuJoCo Physics
```

Phase 3のポリシーは、Phase 1で訓練済みの低レベル制御を呼び出して速度指令を実現します。

## トラブルシューティング

### モデルが見つからない
```bash
# Phase 1モデルが必要
ls models/phase1_open.zip
```

### 迷路XMLが生成されない
```bash
# 手動で迷路生成
python phase3_maze/visualize_maze.py --size 7
```

### 訓練が進まない
- Phase 1モデルが正しく訓練されているか確認
- 迷路生成が正常に機能しているか確認
- 報酬構造を確認（`env.py`）

## 参考

- [OUTPUT_STRUCTURE.md](../OUTPUT_STRUCTURE.md) - 出力管理の仕様
- [common/README.md](../common/README.md) - 共通ユーティリティ
- Phase 1の低レベル制御訓練が前提
