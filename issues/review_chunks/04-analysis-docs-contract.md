# Chunk 04: `analysis_docs` 生成契約を整理する

## 目的

- `result["documents"]` への依存を弱める
- REPL の最終状態や summary から一貫した doc 出力ができるようにする

## 含める作業

- `documents` が空でも `index.md` と report を成立させる
- host 側の doc builder 責務を整理する
- path sanitize と衝突時挙動を定義する
- 出力 path 規約を明文化する

## この chunk でやらないこと

- CLI デフォルトの変更
- `llm_query` の親子設定分離
- REPL helper の追加
- controller の step 観測ログ強化
- README の全面改稿

## 他 chunk でやること

- controller runtime の標準化: `01-controller-default-runtime.md`
- child query 契約: `02-child-query-contract.md`
- helper / observability: `03-repl-helpers-and-observability.md`
- テスト移行: `05-test-migration.md`
- README / 運用文書: `06-readme-and-ops-docs.md`

## レビュー観点

- LLM の返却値に過度依存していないか
- state ベース拡張に耐える構造か
- path traversal / 同名衝突の扱いが妥当か
- 文書生成の責務が REPL と host で混線していないか

## 主な変更対象

- `rlm_runtime.py`
- `test_rlm_runtime.py`
- 必要なら `README.md`

## 依存

- `01-controller-default-runtime.md`

## 完了条件

- `documents` 未返却でも最低限の docs が出る
- sanitize と衝突時動作が決まる
- doc builder の責務境界が説明できる
