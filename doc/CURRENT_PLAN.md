# Implementation Plan: Machine Scan Accuracy & Report Format Synchronization

## 1. 概要とゴール (Summary & Goal)
- **Must**: 
  - 静的マシンスキャン（`machine_scan`）において、不要な設定ファイルやテスト設定ファイルが「解説対象（`kind == "source"`）」と誤判定され、カバレッジ評価や `index.md` の要解説リストに混入するノイズを極小化する。
  - 生成される `analysis_report.md` のフォーマット（ヘッダー行末尾の改行スペース、セクション構成等）を、本体コントローラー（`project_analysis.py`）の生成形式と完全に一致させる。
- **Want**:
  - 特になし。

## 2. スコープ定義 (Scope Definition)
### ✅ In-Scope (やること)
1. **ファイル判定ロジック (`kind`) の強化**:
   - `isohyps/machine_analysis.py` 内の `extract_file_metadata` を修正。
   - `Dockerfile`, `docker-compose.yml`, `compose.yml`, `compose.yaml` などの一般的なコンテナ・デプロイ設定ファイルを `kind = "config"` とする。
   - `conftest.py` を `kind = "config"` とする。
   - `.github/` 配下にあるファイルなど、特定のフォルダ・ドットファイル群を `kind = "config"` または `kind = "other"` とする。
   - 拡張子がプログラムコードではない特定のデータ/設定ファイル（`.yaml`, `.yml`, `.json`, `.toml`, `.xml`, `.ini`, `.cfg` 等）は、デフォルトで `"source"` にせず `"config"`（あるいはすでに binary であれば `"other"` / `"doc"`）にする。
2. **レポートフォーマットの完全同期**:
   - `isohyps/machine_analysis.py` の `analysis_report.md` 生成処理を修正。
   - ヘッダーメタデータ行の末尾に、`project_analysis.py` と同様に半角スペース2つ `  ` を付与する。
   - `### Weak Or Failed Docs` セクション（内容は常に `- (none)`）を追加し、他のセクションの空行構成も完全に同期させる。
3. **TDD用テストの追加と検証**:
   - `tests/test_machine_analysis.py` にテストケースを追加。
   - 境界値ファイル（`conftest.py`, `Dockerfile`, `.github/workflows/main.yml`）の `kind` 判定が正しく `"config"` になることを検証。
   - 生成された `analysis_report.md` に `Weak Or Failed Docs` が含まれ、ヘッダーにスペース2つがあることを検証。

### ⛔ Non-Goals (やらないこと/スコープ外)
- LLM呼び出し機能（Level 1以上）の実装・検証。
- 今回のタスクに無関係なファイル（CLIパーサー等）のリファクタリング。

## 3. 実装ステップ (Implementation Steps)

### Step 1: テストの作成 (Red)
- *Action*: `tests/test_machine_analysis.py` に境界値ファイル（`conftest.py` や `Dockerfile` 等）の判定テスト、および `analysis_report.md` のフォーマット完全一致を検証するアサーションを追加。
- *Validation*: テストを実行し、新しく追加したアサーションが期待通り失敗（Red）することを確認。

### Step 2: ファイル種別判定 (`kind`) の改善 (Green)
- *Action*: `isohyps/machine_analysis.py` の `extract_file_metadata` に分類ルールを追加。
- *Validation*: 新規追加した `kind` 分類テストがパス（Green）することを確認。

### Step 3: レポートフォーマットの同期 (Green)
- *Action*: `isohyps/machine_analysis.py` の `analysis_report.md` 出力テキストを修正し、行末スペースや `Weak Or Failed Docs` セクションを追加。
- *Validation*: レポート検証テストがパス（Green）することを確認。

### Step 4: 全体動作確認と整理 (Refactor)
- *Action*: コードをクリーンアップし、テストを実行。
- *Validation*: `pytest` で全テストが合格することを確認。

## 4. 検証プラン (Verification Plan)
- `pytest` がすべて Green であること。
- 実際のダミープロジェクト等でのマシンスキャン出力が、`conftest.py` や `Dockerfile` を `needs_explanation` から正しく除外することを確認。
