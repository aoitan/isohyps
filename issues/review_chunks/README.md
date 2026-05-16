# Review Chunks

`issues/rlm_refactoring_task_breakdown.md` を、レビューしやすい単位に分割したファイル群。

## 方針

- 1ファイル = 1回のレビューで判断しやすい責務のまとまり
- 依存が強い変更は同じ chunk に寄せる
- 各 chunk に「レビュー観点」を付ける
- 各 chunk に「この chunk でやらないこと」を書いてスコープ逸脱を防ぐ

## Chunk List

1. `01-controller-default-runtime.md`
2. `02-child-query-contract.md`
3. `03-repl-helpers-and-observability.md`
4. `04-analysis-docs-contract.md`
5. `05-test-migration.md`
6. `06-readme-and-ops-docs.md`

## 推奨レビュー順

1. controller の標準化
2. child query 契約
3. doc 出力契約
4. テスト移行
5. helper / observability
6. README / 運用文書
