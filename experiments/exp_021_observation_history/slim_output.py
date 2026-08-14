#!/usr/bin/env python3
"""測定出力から生データ（`raw`）を落とした軽量版を作る（版管理下に置くため）。

Strip the per-step raw records from a measurement output so the judged values
themselves can live under version control.

## なぜ要るか（准教授 AUDIT_042 §3 の (2)・2026-08-14 採択）

**判定が乗っている値そのもの（`summary`・`metrics`・`p5`）は 39 KB 程度**しかないのに、
**`outputs/**` は版管理から除外**されているため**残らない**。
**生データ入りの完全版は 7 MB 超**あって版管理には重い。
**そこで「判定の値だけの版」を分離して版管理下に置く**。

**完全版（生データ入り）は版管理下に置かず、SHA-256 をカードに記録する**
（**モデルが残っていれば決定的に再生成できる** — 測り直しで値が 1 つも動かないことは
`card.md` §4-5 で確認済み）。

## 使い方

```bash
.venv/bin/python experiments/exp_021_observation_history/slim_output.py \
    outputs/exp_021_driving_control_final.json \
    outputs/exp_021_driving_control_800k.json
```

**入力ファイルは読むだけで書き換えない。**出力は `<入力>.slim.json`。
**完全版の SHA-256 も印字する**（カードへの記録用）。
"""
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slim(path: Path) -> Path:
    d = json.loads(path.read_text(encoding="utf-8"))
    # `raw`（毎歩の d_hist・resp_hist）だけを落とす。他は 1 つも触らない。
    for v in d.get("detail", {}).values():
        v.pop("raw", None)
    d["_slim_note"] = (
        "毎歩の生データ（detail[*].raw）を落とした版。判定に使う値（summary・metrics・p5）は"
        "完全版と同一である。完全版は outputs/ 配下（版管理外）にあり、SHA-256 を"
        "experiments/exp_021_observation_history/card.md に記録している。"
        "モデルが残っていれば measure_driving.py で決定的に再生成できる。")
    out = path.with_suffix(".slim.json")
    out.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for a in sys.argv[1:]:
        p = Path(a)
        out = slim(p)
        print(f"{p}")
        print(f"  SHA-256（完全版）= {sha256(p)}")
        print(f"  → {out}（{out.stat().st_size / 1024:.1f} KB）")


if __name__ == "__main__":
    main()
