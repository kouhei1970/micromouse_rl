# Phase 4: Speed Learning

センサーパターンから最適な速度を学習する強化学習環境。

## 概要

Phase 3では「ゴールに到達する」ことが目標でしたが、Phase 4では**「可能な限り高速にゴールに到達する」**ことを目標とします。

### 学習目標

- **直進路**: 加速して高速走行
- **ターン前**: 適切に減速
- **ターン中**: スムーズに旋回
- **ターン後**: 再加速

センサー値（連続値）から「安全な最高速度」を推論することを学習します。

## アーキテクチャ

```
Phase 4 Policy (Speed Controller)
  ↓ 目標速度 [v, omega]
Phase 1 Controller (Low-Level)
  ↓ モーター電圧 [left, right]
MuJoCo Physics Simulation
```

## 観測空間（10次元）

| Index | 内容 | 範囲 | 説明 |
|-------|------|------|------|
| 0-3 | 距離センサー | 0.0-0.15m | LF, LS, RF, RS |
| 4 | 線速度 | m/s | 現在の速度 |
| 5 | 角速度 | rad/s | 現在の角速度 |
| 6 | ゴール距離 | m | ゴールまでの距離 |
| 7 | ゴール角度 | rad | ゴールへの相対角度 |
| 8 | **前方開放度** | 0-1 | 直進可能性の指標 |
| 9 | **左右バランス** | -1〜1 | ターン方向の指標 |

### 追加特徴量の意味

- **前方開放度** `(LF + RF) / 0.30`: 大きいほど前方が開いており、加速可能
- **左右バランス** `(LS - RS) / 0.15`: 正なら左に壁が近い、負なら右に壁が近い

## 行動空間（2次元）

| Index | 内容 | 範囲 |
|-------|------|------|
| 0 | 目標線速度 | 0.0 - 1.0 m/s |
| 1 | 目標角速度 | -5.0 - 5.0 rad/s |

## 報酬設計

| 要素 | 報酬値 | 説明 |
|------|--------|------|
| ゴール到達 | +500 | 固定報酬 |
| 中間セル | +100 | 中間地点到達 |
| **速度報酬** | +3.0 × v | 高速走行を促進 |
| **時間ペナルティ** | -0.5/step | 早期到達を促進 |
| 衝突ペナルティ | -100 | 衝突時 |
| スムーズネス | -0.3×|Δv| - 0.2×|Δω| | 急な速度変化を抑制 |
| 壁接近 | -5.0 (2cm以下) | 危険回避 |

### Phase 3 との比較

| 要素 | Phase 3 | Phase 4 |
|------|---------|---------|
| 時間ペナルティ | -0.03/step | **-0.5/step** (17倍) |
| 速度報酬 | 0.2 × v | **3.0 × v** (15倍) |
| 進行報酬 | 25 × Δdist | なし（速度報酬に統合） |

## 使用方法

### 訓練

```bash
cd /path/to/micromouse_rl
python phase4_speed/train.py
```

### 評価

```bash
# 基本評価
python phase4_speed/evaluate.py

# 動画付き評価
python phase4_speed/evaluate.py --video

# リアルタイム表示
python phase4_speed/evaluate.py --render

# エピソード数指定
python phase4_speed/evaluate.py --episodes 20
```

## 出力

訓練後、以下が生成されます：

- `models/phase4_speed.zip`: 訓練済みモデル
- `outputs/phase4_speed/latest/`: 最新の出力
  - `learning_curve.png`: 学習曲線
  - `evaluation.mp4`: 評価動画
  - `metrics.json`: 評価指標

評価時：
- `outputs/phase4_speed/evaluation/`
  - `speed_profile.png`: 速度プロファイル分析
  - `evaluation.mp4`: 評価動画

## 期待される動作

訓練が成功すると、エージェントは以下の動作を学習します：

1. **直進区間**: センサーが前方開放を検出 → 加速
2. **ターン接近**: 前方センサーが壁を検出 → 減速
3. **ターン実行**: スムーズに旋回
4. **ターン完了**: 前方が開放 → 再加速

## 技術的な詳細

### Phase 1 との連携

Phase 4 は Phase 1 の低レベルコントローラを使用します。高レベルポリシーは目標速度 `(v, omega)` を出力し、Phase 1 がそれをモーター電圧に変換します。

```python
# 低レベルコントローラの使用
ll_obs = self._get_low_level_obs(target_v, target_omega)
ll_action, _ = self.low_level_model.predict(ll_obs, deterministic=True)
self.data.ctrl[0] = ll_action[0] * 3.0  # 左モーター
self.data.ctrl[1] = ll_action[1] * 3.0  # 右モーター
```

### 迷路環境

Phase 3 と同じランダム迷路生成を使用：
- 7×7 迷路
- 5×5 スポーンエリア
- 20エピソードごとに迷路再生成

## トラブルシューティング

### モデルが見つからない

```
Error: Model file models/phase4_speed.zip not found.
```

→ まず訓練を実行してください: `python phase4_speed/train.py`

### Phase 1 モデルが見つからない

```
FileNotFoundError: Model file models/phase1_open.zip not found.
```

→ Phase 1 の訓練が必要です: `python phase1_open/train.py`
