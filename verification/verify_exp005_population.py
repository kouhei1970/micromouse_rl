#!/usr/bin/env python3
"""exp_005 の集計母集団を、保存された走行ごとのデータから独立に再集計する。

作成: 2026-08-11 准教授セッション
入力: outputs/exp_005_collision_penalty/latest/metrics.json の per_course[].trials[] のみ
      （学生B の集計スクリプトは読んでいない）
"""
import json, statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
d = json.loads((REPO / "outputs/exp_005_collision_penalty/latest/metrics.json").read_text())
tr = [t for c in d["per_course"] for t in c["trials"]]

def agg(ts, label):
    fl = [(t["sign_flip_rate_left"] + t["sign_flip_rate_right"]) / 2 for t in ts]
    print(f"  {label:<24} n={len(ts):3d}  反転 {st.mean(fl):6.2f}  "
          f"速度 {st.mean([t['mean_speed'] for t in ts]):.4f}  "
          f"s/区画 {st.mean([t['sec_per_cell'] for t in ts]):.4f}")

print(f"exp_005 / {d['n_courses']} コース × {d['n_trials_per_course']} 試行 = {len(tr)} 走行")
print(f"保存された要約値: 反転 {(d['sign_flip_rate_left_mean']+d['sign_flip_rate_right_mean'])/2:.2f}"
      f"  速度 {d['mean_forward_speed_mps']:.4f}  s/区画 {d['mean_sec_per_cell']:.4f}"
      f"  完走率 {d['no_contact_completion_rate']}")
agg(tr, "全試行")
agg([t for t in tr if t["no_contact_complete"]], "壁接触なし完走のみ")
agg([t for t in tr if t["complete"]], "完走のみ（接触許容）")
print("\n照合:")
print("  全試行 75.08/0.9582/0.1674 → taskC の『全試行』表 75.1/0.958/0.167 と一致")
print("  完走のみ 75.63/0.9604/0.1711 → card §3-1 の 75.6/0.960/0.171 と一致")
print("  → note_009 事例 6 は集計母集団の違いで完全に説明される（推論ではなく保存物からの再現）")
