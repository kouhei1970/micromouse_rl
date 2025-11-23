# Output Directory Structure Specification

## Directory Structure

```
outputs/
├── phase1_open/
│   ├── latest/              # 最新の実験結果（シンボリックリンク or コピー）
│   │   ├── model_info.txt
│   │   ├── learning_curve.png
│   │   ├── learning_curve.csv
│   │   ├── control_performance.png
│   │   ├── evaluation.mp4
│   │   └── metrics.json
│   └── archive/             # 過去の実験結果（タイムスタンプ付き）
│       ├── 20251123_123456/
│       │   ├── model_info.txt
│       │   ├── learning_curve.png
│       │   ├── learning_curve.csv
│       │   ├── control_performance.png
│       │   ├── evaluation.mp4
│       │   └── metrics.json
│       └── 20251123_145678/
│           └── ...
│
├── phase2_slalom/
│   ├── latest/
│   │   ├── model_info.txt
│   │   ├── learning_curve.png
│   │   ├── learning_curve.csv
│   │   ├── trajectory.png
│   │   ├── velocity_tracking.png
│   │   ├── evaluation.mp4
│   │   └── metrics.json
│   └── archive/
│       └── YYYYMMDD_HHMMSS/
│           └── ...
│
├── phase3_maze/
│   ├── latest/
│   │   ├── model_info.txt
│   │   ├── learning_curve.png
│   │   ├── learning_curve.csv
│   │   ├── performance_analysis.png
│   │   ├── maze_visualization.png
│   │   ├── evaluation.mp4
│   │   ├── control_performance.png
│   │   └── metrics.json
│   └── archive/
│       └── YYYYMMDD_HHMMSS/
│           └── ...
│
└── README.md                # このドキュメント
```

## File Naming Convention

### Standard Files (すべてのPhaseで共通)
- `model_info.txt` - モデル情報（ハイパーパラメータ、訓練ステップ数など）
- `learning_curve.png` - 学習曲線グラフ
- `learning_curve.csv` - 学習曲線データ（CSVフォーマット）
- `evaluation.mp4` - 評価動画
- `metrics.json` - 定量的評価指標（成功率、平均ステップ数など）

### Phase-Specific Files
#### Phase 1 & 2 (Low-Level Control)
- `control_performance.png` - 速度追従性能グラフ

#### Phase 2 (Slalom)
- `trajectory.png` - 軌跡可視化
- `velocity_tracking.png` - 速度追従詳細

#### Phase 3 (Maze Navigation)
- `performance_analysis.png` - 総合性能分析
- `maze_visualization.png` - 迷路可視化
- `control_performance.png` - 制御性能（低レベル制御の使用状況）

## Usage Guidelines

### 1. Training Scripts
訓練スクリプトは以下の手順で出力を管理:

```python
import datetime
import json
import shutil
from pathlib import Path

# タイムスタンプ生成
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# アーカイブディレクトリ作成
phase_dir = Path("outputs/phase3_maze")
archive_dir = phase_dir / "archive" / timestamp
archive_dir.mkdir(parents=True, exist_ok=True)

# 出力ファイル保存（アーカイブに直接保存）
save_learning_curve(archive_dir / "learning_curve.png")
save_video(archive_dir / "evaluation.mp4")
# ... その他のファイル

# メトリクス保存
metrics = {
    "success_rate": 0.85,
    "avg_steps": 510,
    "timestamp": timestamp,
    # ...
}
with open(archive_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# latestディレクトリを更新
latest_dir = phase_dir / "latest"
if latest_dir.exists():
    shutil.rmtree(latest_dir)
shutil.copytree(archive_dir, latest_dir)
```

### 2. Analysis Scripts
分析スクリプトは`latest/`を読み込む:

```python
from pathlib import Path

phase_dir = Path("outputs/phase3_maze")
latest_dir = phase_dir / "latest"

# 最新の結果を読み込み
learning_curve = plt.imread(latest_dir / "learning_curve.png")
with open(latest_dir / "metrics.json") as f:
    metrics = json.load(f)
```

### 3. Comparison Scripts
複数の実験を比較する場合はアーカイブから読み込む:

```python
archive_dir = Path("outputs/phase3_maze/archive")
experiments = sorted(archive_dir.iterdir())

for exp_dir in experiments[-5:]:  # 最新5実験
    with open(exp_dir / "metrics.json") as f:
        metrics = json.load(f)
    # 比較処理
```

## Migration Plan

### Step 1: Create Directory Structure
```bash
# 各Phaseに latest/ と archive/ を作成
mkdir -p outputs/phase{1_open,2_slalom,3_maze}/{latest,archive}
```

### Step 2: Archive Existing Files
既存ファイルをタイムスタンプ付きアーカイブに移動

### Step 3: Update Scripts
- `phase1_open/train.py`
- `phase2_slalom/train.py`
- `phase3_maze/train.py`
- 各種分析スクリプト

### Step 4: Create Utility Module
共通の出力管理ユーティリティ (`common/output_manager.py`) を作成

## Benefits

1. **一貫性**: すべてのPhaseで同じ構造
2. **利便性**: `latest/`で常に最新結果にアクセス
3. **履歴管理**: アーカイブで過去の実験を保持
4. **比較容易**: タイムスタンプで実験を識別
5. **自動化**: スクリプトが標準的な場所を参照

## Standard Metrics Format

すべてのPhaseで共通のメトリクスフォーマット:

```json
{
  "timestamp": "20251123_123456",
  "phase": "phase3_maze",
  "training_steps": 1000000,
  "success_rate": 0.85,
  "avg_reward": 699.2,
  "avg_steps": 510,
  "phase_specific": {
    "proper_orientation_rate": 0.25,
    "intermediate_rate": 0.90
  }
}
```
