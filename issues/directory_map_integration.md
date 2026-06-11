# ディレクトリマップを RLM 環境へ追加する

## 背景

現状の controller runtime は `list_dir`, `read_text`, `extract_symbols`, `read_json` などの helper を使ってリポジトリを逐次探索する構成になっている。

この構成でも探索は可能だが、LLM は毎回「どこから見るべきか」を局所観測だけで決める必要があり、初期探索での無駄打ちが起きやすい。特にプロジェクト解析では、コード本体に入る前に `src`, `tests`, `docs`, `.github`, 設定ファイル群などの大まかな地図が分かっているだけで、探索順序と `llm_query` の分割単位が安定する。

RLM の考え方とも矛盾しない。入力全体をプロンプトに押し込むのではなく、環境内オブジェクトとして保持し、必要に応じてコードで参照させればよい。

## 問題

現状には以下がない。

1. リポジトリ全体の圧縮された構造を表す初期オブジェクト
2. LLM が探索前に「何系統の情報が存在するか」を安価に把握する手段
3. ディレクトリツリーを helper 群とは別の一次情報として扱う契約

その結果、LLM は毎回 `list_dir('.')` から始めるか、運が悪いと重要でない領域を先に掘る。

## 目的

- REPL 初期環境にディレクトリマップを追加する
- LLM がコード解析前に全体構造を安価に把握できるようにする
- 既存 helper ベース探索を壊さず、探索の初期ガイドを与える

## 非目的

- 全ファイル本文を事前に環境へロードすること
- ベクタ検索や BM25 など別種の検索基盤を同時に導入すること
- `context` 全体を全面再設計して論文実装へ完全追従すること
- ディレクトリマップだけを唯一の真実のソースにすること

## 提案

controller 起動時に、ルート配下から機械的にディレクトリマップを生成し、REPL へ `repo_map` として注入する。

`repo_map` は全文字列ではなく、探索判断に使う圧縮表現とする。LLM はまず `repo_map` を参照して全体像を把握し、必要になった時だけ既存 helper で詳細を読む。

想定する使い方:

- `repo_map` で `src`, `tests`, `docs`, `config` 相当の位置を把握する
- 重要そうな領域だけ `list_dir` / `read_text` / `extract_symbols` で深掘る
- 子クエリの分割単位を `repo_map` を基準に決める

## 追加方式

第一候補:

- `_sandbox_worker` の初期 `globals_dict` に `repo_map` を追加する
- `PromptBuilder.SYSTEM_PROMPT` に `repo_map` の存在と推奨利用法を短く追記する

補助案:

- `get_repo_map()` helper を追加し、巨大マップの再構築や lazy 取得に備える

まずは REPL 変数として直接見える形を優先する。

## ディレクトリマップの最小要件

- 深さ 2 ないし 3 までの構造を持つ
- 除外対象を含めない
  - `.git`, `node_modules`, `__pycache__`, `dist`, `build`, `venv` など
- ルート直下の重要ファイルは個別に見える
- ノードごとに最低限の種別を持つ
  - `dir`, `file`
- 可能なら簡易カテゴリを持つ
  - `code`, `test`, `doc`, `config`, `ci`, `generated`, `unknown`

## 最小 schema 案

```json
{
  "root": ".",
  "max_depth": 2,
  "nodes": [
    {
      "path": "README.md",
      "node_type": "file",
      "category": "doc"
    },
    {
      "path": "src",
      "node_type": "dir",
      "category": "code"
    },
    {
      "path": "tests",
      "node_type": "dir",
      "category": "test"
    }
  ]
}
```

ここでは本文を埋め込まない。本文は既存 helper で読む。

## 実装方針

### 1. マップ生成関数を追加する

- ルートから shallow なディレクトリマップを生成する
- 生成物は Python `dict` か `list[dict]`
- 文字数とノード数の上限を持たせる

### 2. REPL 初期状態へ注入する

- `_sandbox_worker` の `globals_dict` に `repo_map` を追加する
- state snapshot に巨大に出すぎないよう要約表示で扱う

### 3. プロンプトを最小更新する

- `repo_map` が探索の出発点であることを明示する
- ただし `repo_map` を信じ切らず、必要時は helper で確認するよう誘導する

### 4. テストを追加する

- `repo_map` が REPL から参照できる
- root escape 防止の既存仕様を壊さない
- 除外対象が map に含まれない
- 深さ制限が効く

## 受け入れ条件

- controller runtime で `repo_map` が REPL から参照できる
- 初回ステップで `list_dir('.')` を呼ばずに `repo_map` だけで探索方針を立てられる
- 既存 helper による詳細探索は引き続き利用できる
- テストでマップ生成と REPL 注入が検証される
- README または運用ドキュメントに `repo_map` の位置づけが追記される

## リスク

- マップが粗すぎると誤誘導になる
- マップが細かすぎると RLM の軽量性を損なう
- 自動カテゴリ分類が雑だと探索の優先順位を誤らせる
- 生成コストと維持コストのわりに効果が薄い可能性がある

## 最初のスコープ

最初は以下に限定する。

- shallow なマップ生成
- `repo_map` の REPL 注入
- prompt 1行追加
- 基本テスト

カテゴリ推定の高度化や Issue/PR/設計文書の統合は後続 issue で扱う。
