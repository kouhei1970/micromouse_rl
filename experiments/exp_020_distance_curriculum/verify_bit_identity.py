#!/usr/bin/env python3
"""exp_020 の「不活性の bit 一致」検証（カード §2-3 の条文の実装）。

Verify that the new code, with the curriculum disabled, reproduces exp_019's records.

## 何を照合するか（**量を名指しする** — §9-17）

**照合するのは `episode_seeds.jsonl` の【記録列】である。軌跡そのものではない**
（exp_019 に残っているのはエピソード終端の 1 行だけで、軌跡は保存されていない）。

> **強さの限定**: **力学が同じなら記録列は必ず一致する。しかし逆は必ずしも真ではない**
> （記録列に現れない差が残る可能性は排除できない）。
> **ただし `reset()` は `np_random` から初期擾乱（横 ±0.02 m・方位 ±10°）を引くので、
> 乱数の消費が 1 回でもずれれば初期姿勢がまるごと変わり、`maze_seed` 以外の全列が動く。**
> **したがって実務上の検出力は高い。**

**新設の 3 列（`min_d_since_respawn` / `path_len_m` / `visit_steps`）は
exp_019 側に存在しない**ので、**照合対象から外す**（カード §2-3 の条文どおり）。

## 使い方（**准教授が独立に再現できる形**）

```bash
# 1. 新版をカリキュラム無効（フラグを渡さない）で 10 万歩だけ回す
.venv/bin/python experiments/exp_012_continuous_potential/train.py \
    --condition E --env-version v2 --action-highpass-penalty 0 --seed 1 \
    --total-steps 100000 --log-dir logs/exp_020_bitcheck \
    --model-out models/exp_020_bitcheck.zip

# 2. exp_019 の seed1 の同区間と照合する
.venv/bin/python experiments/exp_020_distance_curriculum/verify_bit_identity.py \
    --new logs/exp_020_bitcheck/episode_seeds.jsonl \
    --ref logs/exp_019_v2_seed1/episode_seeds.jsonl
```
"""
import argparse
import json
import sys
from pathlib import Path

#: 照合する列。**新設の 3 列は ref 側に無いので入れない**（カード §2-3）。
COMPARE_KEYS = ("step", "maze_seed", "outcome", "n_visited", "n_respawn",
                "odom_error_m", "goal_contained_rule", "delta_t_containment")


def load(path: Path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def main() -> int:
    p = argparse.ArgumentParser(description="不活性の bit 一致検証（exp_020 カード §2-3）")
    p.add_argument("--new", required=True, help="新版（カリキュラム無効）の episode_seeds.jsonl")
    p.add_argument("--ref", required=True, help="exp_019 の同 seed の episode_seeds.jsonl")
    p.add_argument("--out", default="outputs/exp_020_bit_identity.json")
    args = p.parse_args()

    new, ref = load(Path(args.new)), load(Path(args.ref))
    n = min(len(new), len(ref))
    if n == 0:
        print("🔴 照合できるエピソードが無い")
        return 1

    mismatches, checked_keys = [], set()
    for i in range(n):
        for k in COMPARE_KEYS:
            in_new, in_ref = k in new[i], k in ref[i]
            if not in_new and not in_ref:
                continue                     # 双方に無い列は対象外（条件で出ない列がある）
            checked_keys.add(k)
            if in_new != in_ref or new[i][k] != ref[i][k]:
                mismatches.append(dict(episode_index=i, key=k,
                                       new=new[i].get(k, "<欠>"), ref=ref[i].get(k, "<欠>")))

    # 空振り防止の確認: **照合に意味のある事象が実際に含まれているか**
    n_goal = sum(1 for r in ref[:n] if r["outcome"] == "goal")
    n_respawn = sum(1 for r in ref[:n] if r.get("n_respawn", 0) >= 1)
    new_only = sorted(set(new[0]) - set(ref[0])) if new and ref else []

    print("=" * 78)
    print("exp_020 不活性の bit 一致検証（照合対象 = episode_seeds.jsonl の記録列）")
    print("=" * 78)
    print(f"  新版 {len(new)} エピソード / 参照 {len(ref)} エピソード → 先頭 {n} 本を照合")
    print(f"  照合した列: {sorted(checked_keys)}")
    print(f"  照合から外した列（新設・参照側に無い）: {new_only}")
    print(f"  参照側の内訳: ゴール {n_goal} 件・リスポーン経験 {n_respawn} 本"
          f"（**0 だと照合が空振りになる**）")
    print()
    if mismatches:
        print(f"  🔴 不一致 {len(mismatches)} 件。最初の 5 件:")
        for m in mismatches[:5]:
            print(f"    ep#{m['episode_index']} {m['key']}: 新 {m['new']} / 参照 {m['ref']}")
        verdict = "MISMATCH"
    else:
        print(f"  ✅ 先頭 {n} エピソードの全列が一致した")
        verdict = "IDENTICAL"

    out = dict(verdict=verdict, n_compared=n, compare_keys=sorted(checked_keys),
               excluded_new_only_keys=new_only, n_goal_in_ref=n_goal,
               n_respawn_episodes_in_ref=n_respawn, n_mismatch=len(mismatches),
               mismatches=mismatches[:50],
               note=("照合対象は記録列であり軌跡そのものではない。"
                     "力学が同じなら記録列は必ず一致するが、逆は必ずしも真ではない。"))
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  → {op}")
    return 0 if verdict == "IDENTICAL" else 1


if __name__ == "__main__":
    sys.exit(main())
