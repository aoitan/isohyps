# Chunk 06: README と運用制約を更新する

## 目的

- README の主張を現状コードと合わせる
- 実行制約や degrade 動作を事前に分かるようにする

## 含める作業

- controller / sandbox / child query 中心に説明を書き換える
- `legacy` が残るなら制限事項として明記する
- timeout, max steps, max depth, token budget を整理する
- tree-sitter 未導入時の挙動を書く

## この chunk でやらないこと

- controller の実装変更
- `llm_query` 契約の仕様変更
- REPL helper の追加
- doc builder の実装変更
- テスト構成の整理そのもの

## 他 chunk でやること

- controller runtime の標準化: `01-controller-default-runtime.md`
- child query 契約: `02-child-query-contract.md`
- helper / observability: `03-repl-helpers-and-observability.md`
- doc 出力契約: `04-analysis-docs-contract.md`
- テスト移行: `05-test-migration.md`

## レビュー観点

- README が marketing 文書ではなく実装説明になっているか
- 利用者が失敗条件を事前に理解できるか
- コマンド例が最新の既定値と一致しているか

## 主な変更対象

- `README.md`

## 依存

- `01-controller-default-runtime.md`
- `02-child-query-contract.md`
- `04-analysis-docs-contract.md`

## 完了条件

- README の説明と既定動作が一致する
- 実運用で詰まりやすい制約が明記される
