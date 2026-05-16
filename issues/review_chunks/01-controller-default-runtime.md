# Chunk 01: Controller Runtime を標準化する

## 目的

- `legacy` 依存を薄め、controller runtime を主系にする
- CLI と README の標準経路を現状の実装方針に揃える

## 含める作業

- `--runtime` のデフォルトを `controller` に変更する
- `RLMRuntimeAnalyzer` を標準の実行経路として扱う
- `RLMAnalyzer` を残す場合は非推奨扱いにする
- controller 側の goal / result の基本契約を整理する

## この chunk でやらないこと

- `llm_query` の親子設定分離
- REPL helper の追加や sandbox 挙動の変更
- `analysis_docs` 生成ロジックの再設計
- controller runtime 向けテストの大規模追加
- README の細かな運用制約追記

## 他 chunk でやること

- child query 契約の整理: `02-child-query-contract.md`
- doc 出力契約の整理: `04-analysis-docs-contract.md`
- テスト移行: `05-test-migration.md`
- README / 運用制約の更新: `06-readme-and-ops-docs.md`

## レビュー観点

- 既存 CLI 利用者への破壊的変更が許容範囲か
- `legacy` を残す理由が明確か
- デフォルト変更後のコードパスが一貫しているか
- result schema の最低限が曖昧でないか

## 主な変更対象

- `analyzer.py`
- `README.md`
- 必要なら controller の result schema 定義

## 依存

- 先行依存なし

## 完了条件

- 標準実行が controller runtime を通る
- README の実行例が controller 基準になる
- `legacy` の位置づけがコードか文書で分かる
