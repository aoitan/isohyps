# `machine_index.json` の導入と公開API・内部ヘルパーの分離

## 背景

`isohyps` のスキャン結果に対する評価（`eval_persistence_review.md`）において、複数イテレーションを経ても「公開APIと内部ヘルパーの混在」や「モジュール責務・依存関係の根拠不足」といった根本的な問題が残り続けていることが判明した。
これは、最終的な `index.md` を直接生成・修正しようとしているためであり、中間の機械成果物を固定していないことに起因する。

## 問題

1. **公開APIと内部ヘルパーの混在**
   - `_` 始まりの内部関数や内部クラスが、公開APIと同じ Outline に並び、主要なインターフェースが埋もれている。
2. **依存関係・読む順序の根拠不足**
   - トポロジカルソート等により「順序」は提示されるが、その依存先や方向、なぜその順序なのかという根拠（理由）が表示されないため、読者が納得して読み進められない。
3. **モジュール責務の1行説明不足**
   - ファイル名と関数一覧はあるが、「このモジュールは何を担当するのか」がインデックス上で一目で分からない。
4. **Attention Points の優先度（Severity）不足**
   - 警告（ファイルサイズ大、テスト不在など）は検出されるが、重み付け（Severity）がないため、どれを優先的に確認すべきか判断しにくい。
5. **stale / missing / changed 判定の検査可能性不足**
   - 状態判定が行われても、判定の根拠（ソースハッシュ、ドキュメントハッシュ、判定理由など）が独立した成果物として出力されず、読者が機械的に検査できない。

これまでの改善は、`<details>` タグで隠すなどの「偽改善」にとどまっており、構造的なノイズの排除や検査可能性の向上には至っていない。

## 目的

- `index.md` の直接生成をやめ、中間の機械成果物である `machine_index.json` を導入し、データ保持と表示ロジックを分離する。
- データの構造化と出力の安定性を高め、表示上の再発を防止する。
- 各ファイル内のシンボルを機械的に分類し、公開APIと内部ヘルパーを整理して出力できるようにする。

## 提案（構造的解決策）

### 1. 中間の機械成果物 `machine_index.json` の固定
表示の改善に直接取り組むのではなく、先にメタデータを構造化した JSON を出力する。
**含める情報案:**
- `file path`
- `module summary`
- `public symbols`
- `internal symbols`
- `imports`
- `reverse imports / fan-in`
- `test presence`
- `source hash`
- `doc hash`
- `stale status`
- `warning severity`

### 2. シンボル分類の独立フェーズ化
Outline 抽出と表示ロジックを分離し、先にシンボルを分類する。
**分類カテゴリ例:**
- `public_api`
- `internal_helper`
- `entrypoint`
- `test_only`
- `generated_or_config`

### 3. 読む順序を独立成果物にする
`dependency_graph.json` と `dependency_order.md` を出力し、以下の情報を明示する。
- `file` / `depends_on` / `depended_by`
- `fan_in` / `fan_out`
- `cycle detected` (循環参照の有無)
- `recommended order` (推奨順序)
- `order reason` (順序の根拠)

### 4. Attention Points への Severity（重大度）導入
警告に以下の評価軸（Severity）を導入してトリアージを容易にする。
- `critical`: 高 fan-in かつ大きい、または entrypoint かつ doc missing
- `high`: 高 fan-in、stale、責務不明
- `medium`: test missing
- `low`: 単独の軽微な欠損

### 5. stale 判定の独立成果物化
検査可能な `doc_freshness.json` を出力し、以下の判定根拠を明示する。
- `source file` / `doc file`
- `current source hash` / `recorded source hash` / `doc hash`
- `status` / `reason`

## 次に試す最小変更（最初に着手するタスク）

**`machine_index.json` を追加し、各ファイルごとに `public_symbols` と `internal_symbols` を分離して出力する。**

最も長く残っている問題である「公開APIと内部ヘルパーの混在」を解決するため、まずはこの分類結果を保持する中間成果物スキーマを実装する。これにより、将来的な表示改善、ノイズ制御、優先度判定などの基盤を整える。
