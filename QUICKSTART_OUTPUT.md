# 出力管理クイックスタート

## 概要

プロジェクト全体で統一的な出力管理システムを導入しました。
すべてのPhaseで同じ構造を使用し、最新の実験結果に簡単にアクセスできます。

## ディレクトリ構造

```
outputs/
├── phase{X}/
│   ├── latest/              # 最新の実験結果（常にここを参照）
│   └── archive/             # 過去の実験（タイムスタンプ付き）
│       └── YYYYMMDD_HHMMSS/
```

## 訓練スクリプトでの使用方法

### 基本的な使い方

```python
from common.output_manager import OutputManager

# 1. OutputManagerを初期化
output_mgr = OutputManager("phase3_maze")

# 2. 訓練実行
model.learn(total_timesteps=1000000)

# 3. 出力ファイルの保存
learning_curve_path = output_mgr.get_path("learning_curve.png")
plot_learning_curve(learning_curve_path)

video_path = output_mgr.get_path("evaluation.mp4")
record_video(video_path)

# 4. メトリクスの保存
output_mgr.save_metrics(
    metrics={
        "training_steps": 1000000,
        "success_rate": 0.85,
        "avg_steps": 510,
    },
    phase_specific={
        "intermediate_rate": 0.90,
    }
)

# 5. モデル情報の保存
output_mgr.save_model_info({
    "Algorithm": "PPO",
    "Training Steps": 1000000,
    "Learning Rate": 3e-4,
})

# 6. 完了処理（latest/を自動更新）
output_mgr.finalize(summary="Training complete!")
```

### 実行結果

訓練完了後、以下のファイルが生成されます:

```
outputs/phase3_maze/
├── latest/                          # 最新結果（自動更新）
│   ├── learning_curve.png
│   ├── evaluation.mp4
│   ├── metrics.json
│   └── model_info.txt
└── archive/
    └── 20251123_143025/            # この実行のアーカイブ
        ├── learning_curve.png
        ├── evaluation.mp4
        ├── metrics.json
        └── model_info.txt
```

## 分析スクリプトでの使用方法

### 最新結果の読み込み

```python
from common.output_manager import OutputManager
import json
from pathlib import Path

# 最新のメトリクスを読み込み
metrics = OutputManager.load_latest_metrics("phase3_maze")
print(f"Success Rate: {metrics['success_rate']}")
print(f"Avg Steps: {metrics['avg_steps']}")

# 最新のファイルを読み込み
latest_dir = Path("outputs/phase3_maze/latest")
learning_curve = plt.imread(latest_dir / "learning_curve.png")
plt.imshow(learning_curve)
```

### 複数実験の比較

```python
from common.output_manager import OutputManager
from pathlib import Path
import json

# すべてのアーカイブを取得
archives = OutputManager.list_archives("phase3_maze")
print(f"Found {len(archives)} experiments")

# 各実験のメトリクスを比較
for timestamp in archives[-5:]:  # 最新5実験
    metrics = OutputManager.load_archive_metrics("phase3_maze", timestamp)
    print(f"{timestamp}: Success={metrics['success_rate']:.2f}")
```

## 標準ファイル一覧

### すべてのPhaseで共通
- `model_info.txt` - モデル設定情報
- `learning_curve.png` - 学習曲線グラフ
- `learning_curve.csv` - 学習曲線データ（CSV）
- `evaluation.mp4` - 評価動画
- `metrics.json` - 定量的評価指標

### Phase固有ファイル
#### Phase 1 (Low-Level Control)
- `control_performance.png` - 速度追従性能

#### Phase 2 (Slalom)
- `trajectory.png` - 軌跡可視化
- `velocity_tracking.png` - 速度追従詳細

#### Phase 3 (Maze Navigation)
- `performance_analysis.png` - 総合性能分析
- `maze_visualization.png` - 迷路可視化
- `control_performance.png` - 制御性能

## metrics.json フォーマット

```json
{
  "timestamp": "20251123_143025",
  "phase": "phase3_maze",
  "training_steps": 1000000,
  "success_rate": 0.85,
  "avg_reward": 699.2,
  "avg_steps": 510,
  "phase_specific": {
    "intermediate_rate": 0.90,
    "maze_size": "7x7",
    "spawn_area": "5x5"
  }
}
```

## 既存ファイルの移行

既存の出力ファイルは自動的に新しい構造に移行済みです:

```bash
# 移行されたファイルの確認
ls outputs/phase3_maze/latest/
ls outputs/phase3_maze/archive/20251123_125952_migrated/
```

## 今後の実験

新しい訓練スクリプトは自動的に新しい構造を使用します:

```bash
# Phase 3の訓練を実行
python phase3_maze/train.py

# 実行後、latest/が自動更新される
ls outputs/phase3_maze/latest/
```

## トラブルシューティング

### Q: latest/が更新されない
A: `output_mgr.finalize()`を必ず呼び出してください

### Q: 古いファイルを削除したい
A: `archive/`内の古いタイムスタンプディレクトリを削除してください
```bash
# 古いアーカイブを削除（例）
rm -rf outputs/phase3_maze/archive/20251123_125952_migrated
```

### Q: 手動で最新結果を更新したい
A: OutputManagerの`update_latest()`を使用
```python
output_mgr = OutputManager("phase3_maze")
output_mgr.update_latest()
```

## 詳細ドキュメント

- [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md) - 詳細な構造仕様
- [common/README.md](common/README.md) - ユーティリティモジュール一覧
