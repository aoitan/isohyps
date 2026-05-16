# Chunk 03: REPL helper と観測性を整える

## 目的

- モデルが import なしで探索しやすい REPL helper を揃える
- step ごとの失敗原因と状態遷移を追いやすくする

## 含める作業

- helper の最小追加を検討する
- helper の root escape 防止を維持する
- `ExecutionObservation` とレポート出力の観測項目を見直す
- 必要なら step 要約を `analysis_report.md` に出力する

## この chunk でやらないこと

- `llm_query` の親子 budget 契約の変更
- controller runtime の標準化
- `analysis_docs` の文書 schema 再設計
- 既存 README の全面更新
- tree-sitter ロジックの大規模拡張

## 他 chunk でやること

- controller runtime の標準化: `01-controller-default-runtime.md`
- child query 契約: `02-child-query-contract.md`
- doc 出力契約: `04-analysis-docs-contract.md`
- テスト移行: `05-test-migration.md`

## レビュー観点

- helper 追加が sandbox 境界を緩めていないか
- helper が多すぎず、用途が重複していないか
- 失敗時の情報量が十分か
- 観測情報がノイズ過多になっていないか

## 主な変更対象

- `rlm_runtime.py`
- `analysis_report.md` の生成ロジック
- `test_rlm_runtime.py`

## 依存

- `01-controller-default-runtime.md`
- `02-child-query-contract.md` の後だと調整しやすい

## 完了条件

- helper の追加 / 非追加理由が説明できる
- 失敗レポートだけ見て停止理由を追える
- helper 境界に関するテストが維持される
