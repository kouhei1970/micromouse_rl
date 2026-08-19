"""tests/ の pytest 収集設定。

## collect_ignore（実行スクリプトを収集対象から外す）

以下の 2 本は **pytest のテストファイルではなく、import された時点で
本体の処理（学習・判定）をその場で走らせて `sys.exit()` する実行スクリプト**である。
pytest がこれらを収集しようとすると、import 時の副作用（学習の起動・判定の実行・
`SystemExit` の送出）がそのまま走ってしまい、収集段階で `pytest` 全体が
`INTERNALERROR` で止まる（2026-08-19 実測）。

削除も改変もしない。**実行するときは pytest ではなく直接 python で走らせる**
（各ファイルの docstring に実行方法が書いてある）:

    .venv/bin/python tests/test_judge_recurrent.py
    .venv/bin/python tests/test_run_summary_argv.py
"""
collect_ignore = [
    "test_judge_recurrent.py",
    "test_run_summary_argv.py",
]
