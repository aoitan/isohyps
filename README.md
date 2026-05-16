# Isohyps

> *等高線（Isohyps）* — コードの抽象度という「高さ」を地図のように行き来しながら、プロジェクト全体の地形を把握するツール。

Recursive Language Models (RLM) の概念に基づき、大規模なプロジェクトを「抽象から具体へと段階的に」再帰下降解析するスクリプトです。

## 概要

このツールは、LLMがファイルシステムを自律的に探索し、各階層で情報を要約・集約することで、トークン制限を回避しながらプロジェクト全体の設計と機能を把握することを目的としています。

### 特徴
- **再帰的要約:** 下位ディレクトリやファイルの解析結果を上位に積み上げ、プロジェクト全体の「エグゼクティブ・サマリー」を構築します。
- **動的探索:** LLMがディレクトリ内の重要度を判断し、解析すべきパスを自ら選択します。
- **構造化ドキュメンテーション:** 解析結果をプロジェクトのディレクトリ構造を模したMarkdownファイル群として出力します。
- **マルチバックエンド:** Google Gemini API および Ollama (ローカル/リモート) に対応しています。

## セットアップ

### 1. 環境構築
Python 3.10以降を推奨します。

```bash
# リポジトリのクローン（またはディレクトリへ移動）
cd project-analyzer-rlm/poc

# 仮想環境の作成
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# 依存ライブラリのインストール
pip install -r requirements.txt
```

### 2. 環境変数の設定 (Geminiを使用する場合のみ)
`.env` ファイルを作成し、Google Gemini APIキーを設定します。

```env
GOOGLE_API_KEY=your_api_key_here
```

## 使い方

### 基本コマンド
```bash
python analyzer.py [解析対象ディレクトリ] [オプション]
```

### 実行例

#### 1. Gemini API を使用する場合 (デフォルト)
```bash
python analyzer.py . --depth 3
```

#### 2. ローカルの Ollama を使用する場合
```bash
python analyzer.py . --backend ollama --model llama3
```

#### 3. ネットワーク上の Ollama サーバーを使用する場合
```bash
python analyzer.py . \
  --backend ollama \
  --ollama-url http://192.168.1.10:11434 \
  --num-ctx 16384 \
  --model llama3
```

#### 4. Controller runtime を使用する場合 (デフォルト)
```bash
python analyzer.py . \
  --depth 2 \
  --max-steps 8 \
  --max-total-tokens 30000 \
  --step-timeout 15
```

### 主なオプション
- `root`: 解析を開始するルートディレクトリ（デフォルト: `.`）
- `--depth`: 再帰解析の最大深さ（デフォルト: `2`）
- `--runtime`: `controller` (推奨) または `legacy` (非推奨 / 将来削除予定)
- `--max-steps`: controller runtime の最大 step 数 (デフォルト: `8`)
- `--step-timeout`: controller runtime の 1 step あたりタイムアウト秒数 (デフォルト: `15.0`)
- `--max-total-tokens`: controller runtime の共有トークン予算 (デフォルト: `30000`)
- `--backend`: 使用するLLMバックエンド (`gemini` または `ollama`)
- `--model`: 使用するモデル名
- `--out`: 構造化ドキュメントの出力先ディレクトリ（デフォルト: `analysis_docs`）
- `--ollama-url`: Ollama APIのエンドポイントURL
- `--num-ctx`: Ollamaのコンテキストサイズ

`controller` runtime はトップレベルの解析完了時に `finish({'summary': str})` を最低契約として要求し、必要に応じて
`documents` list を受け取って `analysis_docs` を再構築します。`documents` の各要素では `path` / `title` / `content`
を指定できますが、省略時はホスト側が補完します。`legacy` runtime は互換用途として残っていますが、新規利用は推奨しません。
`legacy` を使う場合は、controller と実行時間やトークン消費の特性が異なる点に注意してください。

## 出力結果

`--out` で指定したディレクトリ（デフォルト: `analysis_docs`）に、以下の構造でMarkdownファイルが生成されます。

```text
analysis_docs/
├── index.md             # プロジェクト全体の要約 (Root)
├── analysis_report.md   # 全解析結果を集約したレポート
├── src/
│   ├── index.md         # srcディレクトリの要約
│   ├── main.md          # main.pyの解析結果
│   └── utils.md         # utils.pyの解析結果
└── ...
```

各ファイルには、LLMによって生成された機能説明、クラス・関数の責務、およびモジュールの役割が記述されます。

`controller` runtime では、LLM は helper (`list_dir`, `read_text`, `extract_symbols`, `llm_query`, `finish`) を介して探索し、最終結果から `analysis_docs` を再構築します。
