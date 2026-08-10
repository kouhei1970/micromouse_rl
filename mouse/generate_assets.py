"""
mouse/generate_assets.py
================
実行すると assets/mouse_v2.xml（平原 + ロボット v2）を生成するスクリプト。

実行方法（リポジトリルートで）:
    .venv/bin/python -m mouse.generate_assets
"""
import os

from mouse.mjcf import build_open_floor_xml


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, 'assets')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'mouse_v2.xml')
    build_open_floor_xml(out_path)
    print(f'Generated {out_path}')


if __name__ == '__main__':
    main()
