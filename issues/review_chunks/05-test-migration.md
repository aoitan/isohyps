# Chunk 05: テストを新 runtime 基準へ寄せる

## 目的

- 旧 `legacy` 前提テストから、controller runtime 前提のテスト構成へ寄せる
- 新アーキテクチャの正常系 / 異常系 / budget 系を主系列で守る

## 含める作業

- `test_analyzer.py` の legacy 寄りテストを整理する
- runtime 中心のテスト構成へ寄せる
- child query, timeout, invalid code, docs fallback などを追加する
- Phase 1-4 を最低1件ずつカバーする

## この chunk でやらないこと

- controller 本体の新機能設計
- `llm_query` 契約の仕様決定そのもの
- REPL helper の追加判断
- README の説明刷新
- doc 出力仕様の新規設計

## 他 chunk でやること

- controller runtime の標準化: `01-controller-default-runtime.md`
- child query 契約: `02-child-query-contract.md`
- helper / observability: `03-repl-helpers-and-observability.md`
- doc 出力契約: `04-analysis-docs-contract.md`
- README / 運用文書: `06-readme-and-ops-docs.md`

## レビュー観点

- テスト構成だけで現行アーキテクチャが読み取れるか
- legacy の保守コストが高すぎないか
- モック依存が強すぎて本質を外していないか
- budget / timeout の境界条件が押さえられているか

## 主な変更対象

- `test_analyzer.py`
- `test_rlm_runtime.py`

## 依存

- `01-controller-default-runtime.md`
- `02-child-query-contract.md`
- `04-analysis-docs-contract.md`

## 完了条件

- controller runtime の主要挙動をテストで説明できる
- legacy テストの位置づけが明確になる
