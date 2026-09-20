# `machine_index.json` 公開契約

この文書は、Isohyps が生成する `machine_index.json` v2 の利用者向け契約です。言語非依存の型定義は [`schemas/machine-index.schema.json`](../schemas/machine-index.schema.json)、producer／reader の実装は [`isohyps/machine_index.py`](../isohyps/machine_index.py)、source summary の生成・表示は [`isohyps/module_summary.py`](../isohyps/module_summary.py) にあります。

## 1. 責務とバージョン

`machine_index.json` は、後段の LLM、表示生成、追加成果物が参照するための、軽量な公開機械成果物です。内部解析や履歴比較のための `machine_analysis.json` とは責務を分けています。

- 現行 producer version: `schema_version: "2.2"`
- version format: 浮動小数点数ではない整数の `major.minor` を表す文字列（例: `1.0`、`2.10`）
- 全 version で必須の top-level field: `schema_version`、`files`、`dependency_graph`、`dependency_order`
- v2 の必須 field: 上記に加えて `attention`
- v2.1 以降の必須 field: 上記に加えて `doc_freshness`
- v2.2 以降の追加: source file entry に任意の `module_summary`

通常の v2 producer は、`attention`、`doc_freshness`、および `kind == "source"` の全 file entry の `module_summary` を生成します。公開契約上、`module_summary` は旧 analysis や移行中の入力を読めるよう optional のままです。field がないことと、summary が生成された結果 `method == "unknown"` であることは別の状態です。

`build_machine_index_v1()` は従来の source-only projection を維持します。`build_machine_index_v2()` は v1 projection に v2 の `attention`、`doc_freshness`、および存在する source summary を明示的に追加します。内部 analysis object をそのまま公開しません。

## 2. Top-level fields

| Field | Required | Type | 契約 |
| --- | --- | --- | --- |
| `schema_version` | yes | string | `^[0-9]+\.[0-9]+$`。現行 producer は `"2.2"` |
| `files` | yes | array of objects | repository-relative POSIX `path` の昇順。一つの path は一度だけ現れる |
| `dependency_graph` | yes | object: path -> array of path | `file -> direct internal dependency`。key と adjacency 配列は path の昇順 |
| `dependency_order` | yes | array of path | 依存先を先に置く決定的な順序 |
| `attention` | v2 yes | array | severity 順、同一 severity 内は `(path, kind)` 順の構造化 entry |
| `doc_freshness` | v2.1+ yes | object | `doc_freshness.json` の path、schema version、SHA-256、status counts を束縛する参照 |

### Dependency の向きと order

たとえば `src/app.py` が `src/config.py` に依存する場合、次のようになります。

~~~text
dependency_graph["src/app.py"] == ["src/config.py"]
dependency_order == ["src/config.py", "src/app.py", ...]
~~~

graph の key は `kind == "source"` かつ `language != "unknown"` の解析可能な source file と一致します。隣接先は graph key に存在する path で、self edge と重複 edge はありません。循環がある場合も graph は保持され、循環部分は決定的な path 昇順 fallback になります。

### `attention` と `doc_freshness`

`attention` entry は `severity`、`kind`、`path`、`reason`、`evidence` を持ちます。severity は自動修正可否ではなく、確認すべき順番を示します。主な signal は large file、high fan-in／fan-out、missing／stale doc、test missing、TODO increase です。

`doc_freshness` は `path`、`schema_version`、`sha256`、`counts` を持ち、`counts` は `missing`、`fresh`、`stale`、`unknown` の status count です。reader は参照 artifact の path が index directory 外へ出る場合、symlink の場合、存在しない場合、digest が一致しない場合に受理しません。

## 3. File entry

各 file entry は次の9項目を必須とします。reader と schema は additive な未知 field を許容しますが、producer は明示的な allowlist projection を使用します。

| Field | Required | Type | 契約 |
| --- | --- | --- | --- |
| `path` | yes | string | scan root 相対の正規化済み POSIX path。absolute path、backslash、NUL、空／`.`／`..` segment、末尾 slash は不可 |
| `hash` | yes | string | readable text は lowercase 64桁 SHA-256。binary は `binary_skipped`、読取失敗は `error` |
| `size` | yes | non-negative integer | scan 時点の byte size |
| `language` | yes | string | 既存 language detector の値。判定不能は `unknown` |
| `kind` | yes | enum | `source`、`test`、`config`、`doc`、`other` のいずれか |
| `public_symbols` | yes | array of string | 現行の classes、続く top-level functions の抽出順 |
| `internal_symbols` | yes | array of string | `_` で始まるシンボル。同じ抽出順を維持 |
| `fan_in` | yes | non-negative integer | graph 上の直接 incoming edge 数 |
| `fan_out` | yes | non-negative integer | その file の直接依存数 |

`module_summary` は v2.2 以降の source file entry に付く optional field です。通常 producer は `kind == "source"` の全 entry（`language == "unknown"` を含む）へ付けます。`test`、`config`、`doc`、`other`、および現在の分類で config となる `__init__.py` には、summary を責務情報として追加しません。

summary 自体に `path` や `hash` はありません。summary は親 file entry の `path` と `hash` に束縛されます。`hash` は現在の file content identity であり freshness 判定結果ではありません。機械解析が metadata と異なる byte snapshot から根拠を得た場合は、根拠を公開せず `reason: "source_changed"` の `unknown` にします。

## 4. `module_summary` 契約

`module_summary` は自由文の業務要約ではなく、source から直接観測できた記述または構造的事実の bounded projection です。object 内の次の7項目はすべて必須です。

| Field | Type / limit | 意味 |
| --- | --- | --- |
| `text` | non-empty string、最大240 Unicode code points | 表示・選別用の限定的な文面 |
| `method` | `module_docstring` / `structural_facts` / `unknown` | summary の生成方法 |
| `reason` | known では `null`。unknown では後述の enum | 欠測・解析失敗の理由 |
| `parser` | `python_ast` / `tree_sitter` / `regex` / `none` | facts を得た parser |
| `evidence` | array、最大6件 | summary に採用した根拠だけ |
| `omitted_evidence_count` | 0〜`2^63-1` の integer | 上限により記録しなかった定義根拠の数 |
| `text_truncated` | boolean | text または表示名が上限で省略されたか |

各 evidence は次の6項目を必須とします。

| Field | Type / 値 | 意味 |
| --- | --- | --- |
| `kind` | `module_docstring` / `definition` / `entrypoint_candidate` | 根拠の種類 |
| `origin` | `python_ast` / `tree_sitter` / `regex` / `attention_entrypoint_resolver_v1` | 根拠を観測した経路 |
| `line` | 正の integer または `null` | source line。不明または entrypoint 候補では `null` |
| `value` | non-empty string、最大160 Unicode code points | 根拠の bounded excerpt |
| `value_truncated` | boolean | value が省略されたか |
| `paragraphs_omitted` | boolean | docstring の後続段落を省略したか |

method ごとの不変条件は次のとおりです。

- `module_docstring`: `reason == null`、`parser == "python_ast"`。evidence は module docstring ちょうど1件で、`text == evidence.value`。
- `structural_facts`: `reason == null`。evidence に `definition` または `entrypoint_candidate` を少なくとも1件含めます。definition の origin は parser と一致します。
- `unknown`: `text == "unknown"`、evidence は空、`omitted_evidence_count == 0`、`text_truncated == false`。`reason` は `read_error`、`decode_error`、`parse_error`、`unsupported`、`binary_skipped`、`source_changed`、`insufficient_evidence` のいずれかです。

正常な docstring 由来の file entry の関連部分は次のとおりです。他の required file field は説明のため省略しています（hash は説明用の値です）。

~~~json
{
  "path": "src/contours.py",
  "hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "kind": "source",
  "module_summary": {
    "text": "Read contour records.",
    "method": "module_docstring",
    "reason": null,
    "parser": "python_ast",
    "evidence": [
      {
        "kind": "module_docstring",
        "origin": "python_ast",
        "line": 1,
        "value": "Read contour records.",
        "value_truncated": false,
        "paragraphs_omitted": false
      }
    ],
    "omitted_evidence_count": 0,
    "text_truncated": false
  }
}
~~~

docstring がない場合の限定的な例は、名前を観測事実として列挙するだけです。

~~~json
{
  "text": "Non-underscore top-level definitions: ContourReader, load_contours.",
  "method": "structural_facts",
  "reason": null,
  "parser": "python_ast",
  "evidence": [
    {
      "kind": "definition",
      "origin": "python_ast",
      "line": 4,
      "value": "ContourReader",
      "value_truncated": false,
      "paragraphs_omitted": false
    },
    {
      "kind": "definition",
      "origin": "python_ast",
      "line": 12,
      "value": "load_contours",
      "value_truncated": false,
      "paragraphs_omitted": false
    }
  ],
  "omitted_evidence_count": 0,
  "text_truncated": false
}
~~~

根拠がない、または解析に失敗した場合は、架空の責務を作らず次のようにします。

~~~json
{
  "text": "unknown",
  "method": "unknown",
  "reason": "insufficient_evidence",
  "parser": "python_ast",
  "evidence": [],
  "omitted_evidence_count": 0,
  "text_truncated": false
}
~~~

`module_summary` field 自体が存在しない旧 input は、明示的な `unknown` ではありません。人間可読 report では `not available (legacy input)` と表示されます。

## 5. 生成方法と根拠の境界

summary 生成は deterministic な pure rule です。LLM、外部 API、import 名からの業務責務推測は行いません。選択順は次のとおりです。

1. read／decode／parse／binary／snapshot mismatch の失敗は根拠を捨てて `unknown` にします。`unsupported` は entrypoint 候補がある場合に限り、手元にある parser facts による限定的な structural summary を許します。
2. Python の module-level docstring があれば最優先します。AST で得た先頭の非空 paragraph だけを引用し、後続 paragraph があれば `paragraphs_omitted: true` にします。class／function docstring は module summary の代用にしません。docstring は source 記述の抜粋であり、意味の正しさを検証済みとは扱いません。
3. docstring がなければ、entrypoint candidate と定義候補から限定的な structural summary を作ります。Python では非 underscore の top-level class／function／async function を `Non-underscore top-level definitions: ...` と表示します。非 Python では parser 名を含む `Definition candidates (tree_sitter): ...` または `Definition candidates (regex): ...` の形式で、top-level や業務責務を過度に断定しません。
4. entrypoint evidence は `kind: "entrypoint_candidate"`、`origin: "attention_entrypoint_resolver_v1"`、`line: null`、`value: "Entrypoint candidate detected."` です。これは resolver が候補と検出したことだけを示し、設定宣言済み、関数が実在する、または実行可能であることを保証しません。
5. imports は summary 本文にも採用 evidence にも複製しません。`billing` や `authentication` を import するだけの module は、その業務を担当すると断定せず、他の根拠がなければ `unknown` になります。非 underscore 定義も公開 API の完全な契約や再 export の意味を保証するものではありません。

正規化と容量制限は次のとおりです。

- Unicode whitespace を ASCII space に畳み、連続する space を一つにします。残る Unicode category `Cc`、`Cf`、`Cs` は U+FFFD に置換します。Unicode normalization は適用しません。
- `text` は最大240 code points、evidence の `value` は最大160 code points です。長い定義名を本文へ表示する場合は最大24 code pointsです。上限超過時は末尾を `…` にし、対応する `text_truncated`／`value_truncated` を true にします。
- 定義 evidence は deterministic に正規化した `(name, kind, line)` で重複排除・整列し、最大5件を採用します。未採用分は本文の `(+N more)` と `omitted_evidence_count` に反映します。entrypoint evidence は先頭に置きます。
- summary object の canonical JSON は UTF-8、`ensure_ascii=false`、2-space indent、`sort_keys=true`、末尾 LF です。summary 追加分の bytes は16 KiB以下に制限し、超過時は末尾の definition evidence を減らして省略数を更新します。この上限は index 全体の容量上限ではありません。
- reader は未知の summary key／evidence key を無視できます。producer は既知の7 field／6 evidence fieldだけを projection します。入力 mapping は変更しません。

## 6. 順序と決定性

v2 で同じ入力とは、正規化された scan root、対象 file の path と bytes、同じ解析規則、同じ parser／resolver 条件、および正規化済み analysis snapshot を指します。mtime や例外文そのものを summary の値や sort key に含めません。

producer は次の順序を固定します。

- `files`、dependency graph の key、各 adjacency は path 昇順
- `public_symbols`／`internal_symbols` は現行の classes-then-functions 抽出順。summary 用の定義候補の並べ替えがこの公開配列を変更することはありません
- `dependency_order` は dependency-first の deterministic topological order。cycle 時は deterministic fallback
- `attention` は `critical`、`high`、`medium`、`low` 順、同一 severity 内は `(path, kind)` 順
- JSON は UTF-8、`ensure_ascii=false`、2-space indent、`sort_keys=true`、末尾 LF 一つ

同じ snapshot を繰り返し処理した場合、各 source の summary と `machine_index.json` の canonical bytes は一致します。通常の連続 scan では TODO／freshness の履歴が変わる場合があるため、index bytes の決定性を検証するときはその analysis snapshot も固定してください。

## 7. 人間可読 index と非信頼データ

`index.md` と `machine_report.md` の両方に `## Module Summaries` を持ち、公開 machine-index projection を共通 renderer へ渡して、source file 全件を path 昇順で表示します。unchanged file、`language == "unknown"`、summary が明示的に `unknown` の file も一覧から除外しません。表示時に summary を再生成せず、JSON の `text`、`method`、`reason`、`parser`、evidence、省略情報をそのまま対応づけます。

source 由来の path、docstring、定義名、evidence value は非信頼データです。renderer は固定 HTML の text node 内だけに値を配置し、HTML escape と Markdown punctuation の numeric character reference 化を行います。任意値を tag、attribute、URL、Markdown link として解釈しません。docstring の method label は `Module docstring excerpt (unverified)` です。

escape は表示構造を守るためのもので、source 記述の意味や LLM に対する命令性を検証するものではありません。`method` と evidence を併読し、summary 単独でファイルを除外する判定を確定しないでください。`unknown` は「情報がない／解析できない」を表すだけで、「読む必要がない」ことを表しません。

## 8. Unknown fields と version compatibility

reader は明示された supported major の index を検証し、同じ major の未知 field は保持していても意味に依存しません。required field の追加、削除、rename、型変更、既存 field の意味変更は major version を上げる必要があります。

| 入力 version | `module_summary` の扱い |
| --- | --- |
| v1.x | v1 projection の契約外。reader は additive な未知 field を無視し、v1 builder は summary を投影しません |
| v2.0 | `attention` は required、`doc_freshness` は不要。summary は旧版の未知/additive field として扱い、summary contract は適用しません |
| v2.1 | `attention` と `doc_freshness` は required。summary は存在しても旧版 field として扱い、summary contract は適用しません |
| v2.2+ | `module_summary` は optional。存在する場合は object、enum、上限、method/evidence 整合性を検証し、欠落は受理します |

minor は文字列を浮動小数点へ変換せず整数として比較します。そのため `2.2`、`2.9`、`2.10`、`2.100` は summary contract の対象であり、`2.01` のような非 canonical version は拒否されます。v2.2 の summary を外部の旧 reader が扱う場合、未知 additive field を無視する実装であっても、summary を必須と仮定しないでください。外部 consumer 全体の互換性はこの repository からは保証できません。

旧 input に含まれる、`module_summary` がまだない file entry の例:

~~~json
{
  "path": "src/legacy.py",
  "hash": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "size": 12,
  "language": "python",
  "kind": "source",
  "public_symbols": [],
  "internal_symbols": [],
  "fan_in": 0,
  "fan_out": 0
}
~~~

この file entry の summary 欠落は、生成結果 `unknown` とは異なります。旧 input を人間可読表示へ渡す場合だけ legacy label を表示し、reader が架空の `unknown` summary を付け足すことはありません。

## 9. 内部成果物との境界と書込み

v2 index に含めない主な情報と参照先は次のとおりです。

| 内部情報 | 公開 index | 参照先・理由 |
| --- | --- | --- |
| `symbols`、`repo_map`、file の classes／functions／imports | 出さない | 詳細な内部解析が必要なら `machine_analysis.json` |
| `attention_diagnostics` | 出さない | 検出不能と no-finding を区別する内部診断 |
| `coverage_targets`、`coverage_summary`、`coverage_contract` | 出さない | docs、mtime、status に依存する内部情報 |
| file の `mtime`、`status`、`git_status`、`last_seen_commit`、`todo_count` | 出さない | filesystem、Git、worktree、previous output に依存する情報 |

producer は一時 file を destination と同じ directory に作成し、flush／fsync 後に `os.replace()` で `machine_index.json` を置き換えます。reader は書込み途中の truncated JSON ではなく、置換前または置換後の完全な成果物を読みます。producer は単一 writer 前提です。

## 10. 実装・検証の入口

- `build_machine_index_v1(analysis)`: 内部 analysis から従来の allowlist projection を作る
- `build_machine_index_v2(analysis)`: v2.2 の `attention`、`doc_freshness`、存在する source summary を含む projection を作る
- `validate_machine_index(data, supported_major=...)`: version、型、path、graph/order、fan metrics、および v2.2+ summary を検証する
- `load_machine_index(path, supported_major=...)`: UTF-8 JSON を読み、supported major と契約を検証する
- `serialize_machine_index(data, supported_major=...)`: canonical JSON text を返す
- `write_machine_index_atomic(path, data, supported_major=...)`: canonical JSON を atomic replace で書く

契約の回帰テストは [tests/test_machine_analysis.py](../tests/test_machine_analysis.py) にあります。v1 preservation、v2.2 summary の version gate、docstring／structural／unknown の代表例、snapshot mismatch、path 順、決定的順序、両 Markdown の全 source 表示、非信頼値の escape を確認します。
