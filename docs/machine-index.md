# `machine_index.json` 公開契約

この文書は、Isohyps が生成する `machine_index.json` v1 の利用者向け契約です。言語非依存の型定義は [`schemas/machine-index.schema.json`](../schemas/machine-index.schema.json)、producer／reader の実装は [`isohyps/machine_index.py`](../isohyps/machine_index.py) にあります。

## 1. 責務とバージョン

`machine_index.json` は、後段の LLM、表示生成、追加成果物が参照するための公開機械成果物です。内部解析や履歴比較のための `machine_analysis.json` とは責務を分けています。

- 現行 producer version: `schema_version: "1.0"`
- version format: 整数の `major.minor` を表す文字列（例: `1.0`, `1.7`）
- v1 producer の top-level field: 下表の4項目のみ
- v1 producer の file entry field: 下表の9項目のみ
- v1.0 で producer が出力する任意 field はない。将来の任意 field は、supported major 内の拡張枠として扱う

`schema_version` は浮動小数点数として比較しません。reader は現在 major `1` をサポートし、`1.x` を受理します。未知の major（例: `2.0`）は、未知 field の追加とは別の変更として拒否します。

## 2. Top-level fields

| Field | Required | Type | 契約 |
| --- | --- | --- | --- |
| `schema_version` | yes | string | `^[0-9]+\.[0-9]+$`。producer は `"1.0"` |
| `files` | yes | array of objects | repository-relative `path` の昇順。一つの path は一度だけ現れる |
| `dependency_graph` | yes | object: path -> array of path | `file -> direct internal dependency`。key と隣接配列は path の昇順 |
| `dependency_order` | yes | array of path | 依存先を先に置く決定的な順序。各 graph key をちょうど一度含む |

JSON の object key は canonical serialization の `sort_keys=true` により昇順で出力されます。配列の順序は object key の順序とは別に、上表と file entry の規則で固定されます。

### Dependency の向きと order

たとえば `src/app.py` が `src/config.py` に依存する場合、次のようになります。

```text
dependency_graph["src/app.py"] == ["src/config.py"]
dependency_order == ["src/config.py", "src/app.py", ...]
```

graph の key は `kind == "source"` かつ `language != "unknown"` の解析可能な source file と一致します。隣接先は graph key に存在する path で、self edge と重複 edge はありません。

循環がない場合、すべての edge で依存先が依存元より前になります。循環がある場合は graph を保持し、循環を含む部分は既存の決定的 fallback（path 昇順）で埋めます。その場合の `dependency_order` は一意の厳密なトポロジカル順序を意味しません。

## 3. File entry fields

`files` の各要素は次の9項目を必須とします。

| Field | Required | Type | 契約 |
| --- | --- | --- | --- |
| `path` | yes | string | scan root 相対の正規化済み POSIX path。空、absolute path、Windows drive path、backslash、NUL、空／`.`／`..` segment、末尾 slash は不可 |
| `hash` | yes | string | 読み取り可能な text は lowercase 64桁 SHA-256。binary は `binary_skipped`、読み取り失敗は `error` |
| `size` | yes | non-negative integer | scan 時点の byte size |
| `language` | yes | string | 既存 language detector の値。判定不能は `unknown` |
| `kind` | yes | enum | `source`、`test`、`config`、`doc`、`other` のいずれか |
| `public_symbols` | yes | array of string | 公開シンボル。現行の classes、続く top-level functions の抽出順を維持 |
| `internal_symbols` | yes | array of string | `_` で始まるシンボル。同じ抽出順を維持し、名前順への並べ替えはしない |
| `fan_in` | yes | non-negative integer | graph 上の直接 incoming edge 数 |
| `fan_out` | yes | non-negative integer | `dependency_graph[path]` の直接依存数 |

`hash` は現在の file content identity であり、freshness 判定結果ではありません。`binary_skipped` と `error` は通常の SHA-256 digest として比較してはいけません。

`fan_in` は全 adjacency を逆向きに数えた値、`fan_out` はその file の adjacency 長と一致します。graph 対象外の file の `fan_out` は0です。`fan_in`／`fan_out` の不整合は contract violation です。

## 4. 順序と決定性

同じ入力とは、正規化された scan root、対象 file の path と bytes、および同じ ignore・classification・symbol/dependency extraction 規則を指します。次の状態は public index の暗黙入力にしません。

- 前回の `machine_analysis.json` やその他の出力
- file の mtime
- Git history、worktree status
- 生成済み Markdown の存在や更新時刻
- coverage／attention の履歴比較結果

producer は次の規則で値と bytes を正規化します。

- `files`、dependency graph の key、各 adjacency は path 昇順
- `public_symbols`／`internal_symbols` は現行の classes-then-functions 抽出順
- `dependency_order` は決定的な dependency-first topological order。cycle 時は決定的 fallback
- JSON は UTF-8、`ensure_ascii=false`、2-space indent、object key の `sort_keys=true`
- JSON の末尾には LF を一つだけ付ける

そのため、同じ入力を同じ scan 条件で繰り返した場合、`machine_index.json` の内容と bytes は一致します。`output_dir` が scan root 配下の場合は output subtree を走査から除外します。`output_dir == root` は無効で、書込み前にエラーになります。

## 5. Unknown fields と version compatibility

reader は supported major `1` の index について、未知の追加 field を無視して既知の契約を検証します。Python reader は未知 field を保持した object を返す場合がありますが、未知 field に意味を依存しないことが consumer の責務です。

この方針は producer が内部 object をそのまま公開してよいという意味ではありません。producer は明示的な allowlist projection を使い、v1.0 で出力する key set を固定します。

受理される拡張と拒否される変更は次のとおりです。

| 変更 | 扱い |
| --- | --- |
| major `1` の optional top-level／file field の追加 | reader は未知 field として無視する |
| 説明の明確化や、既存 required field の意味を変えない minor 更新 | `1.x` の範囲で許容 |
| required field の追加、削除、rename、型変更 | major version を上げる |
| path、symbol order、dependency の向き／order など既存 field の意味変更 | major version を上げる |
| malformed version、未知 major、required 欠落、既知 field の型違い、path／graph invariant 違反 | contract error として拒否 |

JSON Schema の `additionalProperties: true` は、supported major 内で reader が未知 field を許容する方針を表します。path の一意性・配列順・graph/order/fan の相互整合性・supported major 判定は JSON Schema だけではなく、Python contract validator の責務です。

## 6. 内部成果物との境界と migration

旧形式の `machine_index.json` は version がなく、`machine_analysis.json` と同じ内部 result の複製でした。v1 ではこの偶然の field 公開を正式契約にしません。旧形式を読む consumer は v1 reader の前提を満たさないため、再 scan で v1 index を生成するか、必要な間だけ個別の migration shim を用意してください。

v1 index に含めない主な情報と所有先は次のとおりです。

| 旧／内部情報 | v1 の扱い | 参照先・理由 |
| --- | --- | --- |
| `symbols`、`repo_map`、file の `classes`／`functions`／`imports` | public index には出さない | 詳細な内部解析が必要なら `machine_analysis.json` |
| `attention` | 出さない | 前回結果との差分に依存する診断情報 |
| `coverage_targets`、`coverage_summary`、`coverage_contract` | 出さない | docs、mtime、status に依存する coverage／表示情報 |
| file の `mtime`、`status`、`git_status`、`last_seen_commit`、`todo_count` | 出さない | filesystem、Git、worktree、previous output に依存 |

v1 の `hash` は安定した content digest のみを提供します。freshness の hash comparison／判定、severity 判定、LLM による module summary は v1 の責務ではありません。将来追加する場合は、現在状態だけから導出できる optional snapshot または別 artifact として設計し、既存 field の意味を変更しないことが必要です。

repo 外の旧形式 consumer はリポジトリからは観測できません。そうした consumer がある場合、v1 への切替前にこの契約を共有し、旧 field の移行先を決めてください。

## 7. 書込みと利用上の注意

producer は一時ファイルを destination と同じ directory に作成し、flush／fsync 後に `os.replace()` で `machine_index.json` を置き換えます。通常の reader は書込み途中の truncated JSON ではなく、置換前の完全な成果物か置換後の完全な成果物を読みます。

同時に複数の producer が同じ output directory を更新する運用は契約していません。writer は単一 writer 前提です。

## 8. 実装上の確認入口

契約を利用・検証する入口は次のとおりです。

- `build_machine_index_v1(analysis)`: 内部 analysis から allowlist projection を作る
- `validate_machine_index(data)`: version、型、path、graph/order、fan metrics を検証する
- `load_machine_index(path)`: UTF-8 JSON を読み、supported major と契約を検証する
- `serialize_machine_index(data)`: canonical JSON text を返す
- `write_machine_index_atomic(path, data)`: canonical JSON を atomic replace で書く

契約の回帰テストは [`tests/test_machine_analysis.py`](../tests/test_machine_analysis.py) にあります。ここでは v1 minor と unknown field、未知 major／不正 field、symbol／dependency の既存意味、cycle、stale-doc を含む連続 scan の byte equality、root 内 output の自己混入防止を確認します。
