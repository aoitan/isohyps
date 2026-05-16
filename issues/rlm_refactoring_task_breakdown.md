# RLM リファクタリング実装タスク分解

`issues/rlm_refactoring_tasks.md` を、現状実装との差分を踏まえて着手単位に分解した作業リスト。

## 目的

- `legacy` 依存の静的解析器から controller runtime へ主系を移す
- `llm_query` を「単なる再帰」ではなく RLM のサブクエリ API として整理する
- REPL の最終状態から `analysis_docs` を再構築できるようにする
- テストを新アーキテクチャ基準へ寄せる

## 優先順位

1. controller runtime を主系にする
2. `llm_query` の責務を明確化する
3. `analysis_docs` の生成源を REPL 側に寄せる
4. テストと README を移行する

## Task Group A: ランタイム統合

### A1. `legacy` / `controller` の責務整理

- `RLMAnalyzer` を互換レイヤに留めるか削除するか決める
- `RLMRuntimeAnalyzer` を CLI の標準経路にする
- `--runtime` のデフォルトを `controller` に変更する
- `legacy` を残す場合は非推奨扱いにする

完了条件:
- `analyzer.py` の標準実行が controller runtime を通る
- README の実行例が controller 基準になっている

### A2. 解析ゴールと結果契約の固定

- controller に渡す goal 文面を定型化する
- `finish()` の返却フォーマットを明文化する
- `summary` / `documents` / 任意メタデータの扱いを決める

完了条件:
- result schema がコード上の定数または説明コメントで追える
- テストで schema の最低限が保証される

## Task Group B: `llm_query` の再設計

### B1. 親クエリと子クエリの設定分離

- `llm_query` が親と同じ `client` をそのまま使う設計を見直す
- サブクエリ専用の model / budget / prompt builder を差し込めるようにする
- 将来的に親子で別 backend を選べる余地を残す

完了条件:
- `RLMController` が child query 用設定を受け取れる
- 少なくとも「親と別 budget」をテストできる

### B2. 子クエリの返却値と失敗時挙動の明確化

- `llm_query` 成功時に REPL へ返す値の制約を決める
- 子クエリ失敗時に例外化するか、構造化エラーを返すか決める
- 親プロンプトに子の全文を混ぜないことを保証する

完了条件:
- 子クエリの失敗ケースに対するテストがある
- 親 prompt へ不要な子レスポンス文字列を埋め戻さない

## Task Group C: REPL と観測可能性

### C1. REPL 初期 helper の拡充

- `list_dir` / `read_text` / `extract_symbols` の責務を見直す
- 必要なら `path_exists`, `is_dir`, `read_json` など最小 helper を追加する
- 「import せず探索できる」状態を強める

完了条件:
- モデルが素朴な探索で詰まりにくい helper セットになる
- helper の root escape 防止が維持される

### C2. step 観測ログの構造化

- `ExecutionObservation` の保存内容を見直す
- `analysis_report.md` に step ごとの要約を出すか決める
- デバッグ時に失敗原因を追えるようにする

完了条件:
- 少なくとも `kind`, `error`, `finished`, `state` の遷移が追える
- 失敗時レポートだけ見て止まり方が分かる

## Task Group D: `analysis_docs` の再構築

### D1. REPL 最終状態から文書を生成できるようにする

- `result["documents"]` 依存を弱める
- REPL state から host 側が `index.md` や個別 doc を合成できる形を検討する
- 最低でも「documents が空でも root summary は成立する」を保証する

完了条件:
- `documents` 未返却でもレポートと `index.md` が一貫して出る
- 将来、state ベースの doc builder へ拡張しやすい構造になる

### D2. 出力パスと文書構造のルール化

- `path` の sanitize ルールを明文化する
- 同名衝突時の扱いを決める
- ディレクトリ風 path と単一ファイル path の出力規約を決める

完了条件:
- path traversal 防止に加えて衝突時動作も定義される
- テストで少なくとも 1 件カバーする

## Task Group E: テスト移行

### E1. 旧テストの位置づけ整理

- `test_analyzer.py` のうち legacy 専用のものを分離する
- controller runtime の標準挙動テストを主系列にする
- 必要なら legacy テストを `deprecated` 扱いで残す

完了条件:
- 新 runtime の成功系・失敗系・budget 系が主テストとして読める
- 旧 runtime のテストが混ざって意図を曖昧にしない

### E2. 追加すべき controller テスト

- 子クエリ budget 共有 or 分離の挙動
- step timeout 後の state reset
- `documents` なし結果の doc 出力
- 子クエリ失敗時の親 step の観測内容
- invalid code 連続時の打ち切り

完了条件:
- Issue の Phase 1-4 を最低1件ずつ検証するテストがある

## Task Group F: ドキュメント移行

### F1. README の主張を実装に合わせる

- 「再帰的要約」中心の説明から controller / sandbox / child query 中心へ更新する
- `legacy` が残るなら制限事項として明記する

完了条件:
- README の特徴説明が現状コードとずれない

### F2. 運用上の制約を書き出す

- step timeout
- max steps / max depth
- 近似 token 管理
- tree-sitter 未導入時の degrade 動作

完了条件:
- 実行前に分かるべき制約が README にある

## 推奨実装順

1. A1
2. A2
3. B1
4. B2
5. D1
6. E1
7. E2
8. F1
9. F2
10. C1
11. C2
12. D2

## 最小マイルストーン

### Milestone 1

- controller runtime を既定化
- README 更新
- 既存テストが通る

### Milestone 2

- `llm_query` 設定分離
- child query 失敗系テスト追加

### Milestone 3

- `analysis_docs` の生成契約整理
- legacy テスト整理

## 保留事項

- 「別のLLM」を本当に別 model / 別 backend にするか
- sandbox の安全性をどこまで強化するか
- helper をどこまで増やすか
- structured result schema を dataclass / TypedDict 化するか
