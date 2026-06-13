# Isohyps

> *等高線（Isohyps）* — コードの抽象度という「高さ」を地図のように行き来しながら、プロジェクト全体の地形を把握するツール。

LLM がファイルシステムを自律的に探索し、各ディレクトリ・ファイルの解析結果を段階的に集約することで、プロジェクト全体の設計と機能をMarkdownドキュメントとして出力するツールです。

## 目次

1. [概要](#概要)
2. [セットアップ](#セットアップ)
3. [アーキテクチャと実行モデル](#アーキテクチャと実行モデル)
4. [基本的な使い方](#基本的な使い方)
5. [運用上の制約とパラメータ仕様](#運用上の制約とパラメータ仕様)
6. [依存関係とフォールバック](#依存関係とフォールバック)
7. [出力結果](#出力結果)
8. [制限事項と非推奨機能](#制限事項と非推奨機能)

---

## 概要

Isohyps は、LLM を使ったコードベース解析ツールです。Controller ランタイムが LLM エージェントとしてディレクトリを探索・解析し、構造化された Markdown ドキュメントを生成します。

主な用途:
- 大規模リポジトリの全体像の把握
- コードベースのドキュメント自動生成
- 外部レビュアーやオンボーディング向けの設計ドキュメント作成

対応バックエンド: Google Gemini API および Ollama（ローカル/リモート）

---

## セットアップ

### 1. 環境構築
Python 3.10以降と [uv](https://docs.astral.sh/uv/) が必要です。

```bash
# リポジトリのクローン（またはディレクトリへ移動）
cd isohype

# ロックファイルに基づいて依存ライブラリを同期
uv sync
```

構造化シンボル抽出を有効にしたい場合のみ、追加依存を入れます。
`tree-sitter-languages` は Python 3.13 では利用できない環境があるため、標準依存からは外しています。

```bash
uv sync --extra symbols
```

`requirements.txt` は pip 互換が必要な環境向けの補助ファイルです。通常の開発・実行では `uv sync` / `uv run` を使ってください。

### 2. 環境変数の設定 (Gemini を使用する場合)
`.env` ファイルを作成し、Google Gemini API キーを設定します。

```env
GOOGLE_API_KEY=your_api_key_here
```

### 3. Ollama を使用する場合の注意事項

Ollama バックエンドを使う場合、`--num-ctx` でコンテキストサイズを設定できます（デフォルト: `8192`）。大規模プロジェクトを解析する際にコンテキストが溢れてエラーになる場合は、この値を増やしてください。

```bash
uv run python analyzer.py . --backend ollama --model llama3 --num-ctx 32768
```

---

## アーキテクチャと実行モデル

### Controller
デフォルトのランタイムです。LLM エージェントが以下のヘルパーツールを使ってディレクトリを自律的に探索します。

- `list_dir`: ディレクトリ一覧の取得
- `read_text`: ファイルのテキスト読み込み
- `extract_symbols`: 構造化シンボル（クラス・関数など）の抽出
- `llm_query`: 下位ディレクトリへの再帰的サブクエリ（Child Query）
- `finish`: 探索完了・結果の確定

探索が完了すると、LLM は `finish({'summary': str, 'documents': [...]})` を呼び出します。`documents` の各要素には `path` / `title` / `content` を指定できます（省略時はホスト側が補完）。

### Sandbox
Controller の各ステップはサンドボックスプロセス内で実行されます。タイムアウトやリソース超過が発生した場合、プロセスはリセットされ、次のステップで再試行が可能です。

### Child Query
`llm_query` ヘルパーを使って、サブディレクトリやファイル群を別の Controller インスタンスに委任する再帰的サブクエリです。`--depth` パラメータで再帰の深さを制限できます。

---

## 基本的な使い方

### 基本コマンド
```bash
uv run python analyzer.py [解析対象ディレクトリ] [オプション]
```

### 実行例

#### Gemini API を使用する場合（デフォルト）
```bash
uv run python analyzer.py . --depth 2
```

#### ローカルの Ollama を使用する場合
```bash
uv run python analyzer.py . --backend ollama --model llama3
```

#### Ollama（ネットワーク上のサーバー）を使用する場合
```bash
uv run python analyzer.py . \
  --backend ollama \
  --ollama-url http://192.168.1.10:11434 \
  --num-ctx 16384 \
  --model llama3
```

### 主なオプション

| オプション | デフォルト値 | 説明 |
|---|---|---|
| `root` | `.` | 解析を開始するルートディレクトリ |
| `--depth` | `2` | Child Query による再帰探索の最大深さ |
| `--max-steps` | `30` | Controller の最大ステップ数 |
| `--step-timeout` | `15.0` | 1ステップあたりのタイムアウト秒数 |
| `--llm-timeout` | `120.0` | 1回の LLM バックエンド呼び出しに対するタイムアウト秒数 |
| `--max-total-tokens` | `90000` | Controller 全体の共有トークン予算 |
| `--backend` | `gemini` | LLM バックエンド（`gemini` または `ollama`） |
| `--model` | — | 使用するモデル名 |
| `--out` | `analysis_docs` | 出力先ディレクトリ |
| `--ollama-url` | — | Ollama API のエンドポイント URL |
| `--num-ctx` | `8192` | Ollama のコンテキストサイズ |
| `--runtime` | `controller` | `controller`（推奨）または `legacy`（非推奨） |

---

## 運用上の制約とパラメータ仕様

Controller ランタイムは、無限ループや過剰な API 課金を防ぐために以下の実行時制約を持ちます。各制約に達した場合の**システムの振る舞い**を理解しておくことが重要です。

### `--max-steps`（デフォルト: `30`）
Controller が LLM ツールコールを実行できる最大ループ回数です。

**上限到達時の挙動:** 上限に達すると `BudgetExceededError` が発生し、探索は即座に打ち切られます。ツール全体はクラッシュせず、**そこまでに収集できた情報を使って出力ドキュメントを生成**し、ステータス `budget_exceeded` で正常終了します。

### `--max-total-tokens`（デフォルト: `90000`）
Controller 全体（すべての LLM 呼び出しの合計）で消費可能なトークン数の上限です。

**上限到達時の挙動:** `--max-steps` と同様に `BudgetExceededError` として扱われます。**そこまでの結果で出力を生成**し、`budget_exceeded` ステータスで終了します。出力ドキュメントの `analysis_report.md` にステータスと使用済みトークン数が記録されます。

### `--step-timeout`（デフォルト: `15.0` 秒）
Sandbox での 1 ステップの実行に許容する最大時間です。

**タイムアウト時の挙動:** タイムアウトに達すると **Sandbox プロセスがリセット**されます。そのステップは失敗として扱われますが、残りのステップ数に余裕がある場合は Controller が次のステップで別の手段を試行できます。

### `--llm-timeout`（デフォルト: `120.0` 秒）
バックエンド LLM への 1 回の問い合わせに許容する最大時間です。

**タイムアウト時の挙動:** タイムアウトに達するとその問い合わせは `model_error` として観測されます。応答が返らない、または Ollama / Gemini 側が完了後にクライアントへ復帰しない場合でも、Controller は永久に待ち続けず次のステップまたは打ち切りへ進みます。詳細は `analysis_report.md` に記録されます。

### `--depth`（デフォルト: `2`）
Child Query（`llm_query` ヘルパー）による再帰的サブクエリの最大深度です。

**上限到達時の挙動:** 深度制限を超えた Controller は `budget_exceeded` として終了します。Child Query 内で発生した場合、呼び出し元の `llm_query` はエラーとして観測され、そのステップは失敗扱いになります。残りのステップ数に余裕があれば、Controller は次のステップで別の手段を試行できます。

> **注意:** `budget_exceeded` ステータスで終了した場合、出力されるドキュメントは**不完全**です。完全な解析が必要な場合は `--max-total-tokens` または `--max-steps` を増やして再実行してください。

---

## 依存関係とフォールバック

### tree-sitter

`extract_symbols` ヘルパーはコードの構造化解析（クラス・関数などのシンボル抽出）に、任意依存の `tree-sitter-languages` パッケージを使用します。標準の `requirements.txt` には含めていません。

**`tree-sitter` が利用できない場合（未インストール、またはパース失敗時）:**
シンボル抽出は行われず、ファイル先頭部分の単純なテキスト抽出（`read_text_excerpt`、デフォルト最大 1000 文字）にデグレードします。このデグレードは自動的に行われ、ツールの実行は継続されますが、**解析精度が大きく低下する**可能性があります。

Python 3.10 / 3.11 など対応環境で精度を重視する場合は、追加依存のインストールを推奨します。

```bash
uv sync --extra symbols
```

### 初期コンテキストと探索品質

Controller runtime では、探索の初期ガイドとして**リポジトリ全体の浅い階層のディレクトリマップ (`repo_map`) が初期環境のグローバル変数として注入されます**。これにより、LLM は探索の初手でプロジェクト全体の大まかな構造（`src`, `tests`, `docs` など）を安価に把握でき、初期探索での無駄打ちを防ぎます。LLM は `repo_map` で全体像を掴んだ後、必要に応じて `list_dir` や `read_text` などの helper を併用して詳細な探索を行います。

`repo_map` は本文を含まない部分的なマップです。デフォルトでは深さ 2、最大 500 ノードまでに制限され、`.git`, `node_modules`, `__pycache__`, `dist`, `build`, `venv` などの生成物・依存ディレクトリとシンボリックリンクは含めません。上限に達した場合は `truncated: true` になり、深い階層や省略された領域は helper で確認する前提です。

---

## 出力結果

`--out` で指定したディレクトリ（デフォルト: `analysis_docs`）に、以下の構造で Markdown ファイルが生成されます。

```text
analysis_docs/
├── analysis_report.md   # 実行サマリー（ステータス・トークン使用量等）
├── index.md             # プロジェクト全体の要約
├── src/
│   ├── index.md         # src ディレクトリの要約
│   ├── main.md          # main.py の解析結果
│   └── utils.md         # utils.py の解析結果
└── ...
```

`analysis_report.md` には以下の情報が含まれます:
- 実行バックエンドとモデル名
- ランタイムステータス（`finished` / `budget_exceeded` / `error` 等）
- 使用済みステップ数・トークン数
- エグゼクティブサマリー

---

## 制限事項と非推奨機能

### `--runtime legacy`（非推奨）

`--runtime legacy` は旧来の `RLMAnalyzer` クラスを使った実行モードです。互換性のために残されていますが、**現在は非推奨であり将来のバージョンで削除予定**です。

Legacy モードは Controller ランタイムと以下の点で異なります:
- トークン予算管理（`--max-total-tokens`）が機能しません。
- ステップタイムアウト（`--step-timeout`）が適用されません。
- `budget_exceeded` によるフェイルセーフな部分出力の仕組みがありません。

新規の利用は推奨しません。Controller ランタイムを使用してください。
