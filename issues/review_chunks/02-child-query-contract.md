# Chunk 02: `llm_query` の子クエリ契約を整理する

## 目的

- `llm_query` を単なる再帰呼び出しから、子クエリ API として整理する
- 親クエリと子クエリの設定や失敗時挙動を明確化する

## 含める作業

- child query 用の設定を `RLMController` に注入できるようにする
- 親と子で budget や prompt builder を分離可能にする
- 子クエリの返却値と失敗時の扱いを決める
- 親 prompt へ子レスポンス全文を混ぜないことを保証する

## この chunk でやらないこと

- CLI デフォルト変更や `legacy` の整理
- REPL helper の種類追加
- `analysis_docs` の path sanitize / 衝突解決の変更
- README 全体の書き換え
- controller 以外の backend 実装追加

## 他 chunk でやること

- controller runtime の標準化: `01-controller-default-runtime.md`
- helper / observability: `03-repl-helpers-and-observability.md`
- doc 出力契約: `04-analysis-docs-contract.md`
- README / 運用文書: `06-readme-and-ops-docs.md`

## レビュー観点

- API の責務が小さく保たれているか
- 将来、親子で別 model / backend を使える設計余地があるか
- 子クエリ失敗時の親側挙動がデバッグしやすいか
- 予算共有 / 分離のルールが理解しやすいか

## 主な変更対象

- `rlm_runtime.py`
- `test_rlm_runtime.py`

## 依存

- `01-controller-default-runtime.md`

## 完了条件

- child query 設定を差し替えられる
- 子クエリの成功系 / 失敗系テストがある
- 親 prompt 汚染を避ける契約が明文化される
