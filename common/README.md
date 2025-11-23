# Common Utilities

このディレクトリには、全Phaseで共通して使用するユーティリティモジュールが含まれています。

## モジュール一覧

### 1. output_manager.py
出力ファイル管理の標準化ツール

**主な機能:**
- タイムスタンプ付きアーカイブディレクトリの自動作成
- `latest/`ディレクトリの自動更新
- 標準化されたメトリクスの保存（JSON形式）
- モデル情報の保存（テキスト形式）

**使用例:**
```python
from common.output_manager import OutputManager

# 初期化
output_mgr = OutputManager("phase3_maze")

# ファイルパス取得
learning_curve_path = output_mgr.get_path("learning_curve.png")
save_plot(learning_curve_path)

# メトリクス保存
output_mgr.save_metrics(
    metrics={"success_rate": 0.85, "avg_steps": 510},
    phase_specific={"intermediate_rate": 0.90}
)

# モデル情報保存
output_mgr.save_model_info({
    "Algorithm": "PPO",
    "Training Steps": 1000000,
    # ...
})

# 完了処理（latest/を更新）
output_mgr.finalize(summary="Training complete!")
```

### 2. visualization.py
可視化ツール

**主な機能:**
- 学習曲線のプロット
- 動画記録
- 制御性能グラフの生成

### 3. maze_assets.py
迷路アセット生成ツール

**主な機能:**
- MuJoCo用XMLファイルの生成
- DFS迷路生成アルゴリズム

### 4. mjcf_builder.py
MuJoCo XMLビルダー

**主な機能:**
- XMLファイルの動的生成
- 壁、床、ゴールの配置

### 5. robot_builder.py
ロボットモデルビルダー

**主な機能:**
- ロボット本体のXML生成
- センサー配置

### 6. extract_learning_curve.py
学習曲線抽出ツール

**主な機能:**
- チェックポイントモデルから学習曲線を再構築
- monitor.csvがない場合の代替手段

**使用例:**
```bash
python common/extract_learning_curve.py \
  --checkpoint-pattern 'logs/ppo_phase3_maze_*_steps.zip' \
  --output 'outputs/phase3_maze/learning_curve_from_checkpoints.png'
```

### 7. live_monitor.py
リアルタイム訓練モニター（テキストベース）

**主な機能:**
- 訓練プロセスの標準出力をリアルタイム表示
- エピソード情報、報酬、成功率の追跡

### 8. monitor_training.py
リアルタイム訓練モニター（グラフベース）

**主な機能:**
- Matplotlibでの動的グラフ更新
- リアルタイム学習曲線表示

## 統一的な出力管理

すべての訓練スクリプトは`OutputManager`を使用して、以下の標準構造に従います:

```
outputs/
├── phase{X}/
│   ├── latest/              # 最新の実験結果
│   │   ├── model_info.txt
│   │   ├── learning_curve.png
│   │   ├── evaluation.mp4
│   │   └── metrics.json
│   └── archive/             # 過去の実験（タイムスタンプ付き）
│       └── YYYYMMDD_HHMMSS/
│           └── ...
```

詳細は[OUTPUT_STRUCTURE.md](../OUTPUT_STRUCTURE.md)を参照してください。

## メトリクスフォーマット

すべてのPhaseで共通のJSON形式:

```json
{
  "timestamp": "20251123_123456",
  "phase": "phase3_maze",
  "training_steps": 1000000,
  "success_rate": 0.85,
  "avg_reward": 699.2,
  "avg_steps": 510,
  "phase_specific": {
    "intermediate_rate": 0.90
  }
}
```

## ファイル命名規則

### 標準ファイル（全Phase共通）
- `model_info.txt` - モデル設定情報
- `learning_curve.png` - 学習曲線
- `learning_curve.csv` - 学習曲線データ
- `evaluation.mp4` - 評価動画
- `metrics.json` - 評価指標

### Phase固有ファイル
- Phase 1/2: `control_performance.png`
- Phase 2: `trajectory.png`, `velocity_tracking.png`
- Phase 3: `performance_analysis.png`, `maze_visualization.png`
