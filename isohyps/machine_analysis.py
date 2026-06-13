from __future__ import annotations

import os
import ast
import re
import json
import fnmatch
import hashlib
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from isohyps.analysis_helpers import detect_language, is_probably_binary, extract_symbols

# 簡易的なYAML出力のためのシリアライザ
def simple_yaml_dump(data: Any, indent_level: int = 0) -> str:
    spacing = "  " * indent_level
    if isinstance(data, dict):
        lines = []
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{spacing}{k}:")
                lines.append(simple_yaml_dump(v, indent_level + 1))
            else:
                lines.append(f"{spacing}{k}: {v}")
        return "\n".join(lines)
    elif isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{spacing}-")
                lines.append(simple_yaml_dump(item, indent_level + 1))
            else:
                lines.append(f"{spacing}- {item}")
        return "\n".join(lines)
    else:
        return f"{spacing}{data}"


def get_git_commit_hash(path: Path) -> str | None:
    try:
        res = subprocess.run(
            ["git", "log", "-n", "1", "--pretty=format:%H", "--", str(path)],
            capture_output=True,
            text=True,
            check=False,
            cwd=path.parent,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return None


def get_git_status_info(root: Path) -> dict[str, str]:
    status_map = {}
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            cwd=root,
        )
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                if len(line) > 3:
                    state = line[:2].strip()
                    file_path = line[3:].strip()
                    # git の状態から status をマッピング
                    if "M" in state:
                        status_map[file_path] = "changed"
                    elif "A" in state or "??" in state:
                        status_map[file_path] = "added"
                    elif "D" in state:
                        status_map[file_path] = "deleted"
    except Exception:
        pass
    return status_map


def extract_file_metadata(path: Path, root: Path, previous_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    abs_path = path.resolve()
    rel_path = abs_path.relative_to(root.resolve()).as_posix()
    
    # ハッシュの計算
    sha256_hash = hashlib.sha256()
    try:
        if not is_probably_binary(abs_path):
            content = abs_path.read_bytes()
            sha256_hash.update(content)
            file_hash = sha256_hash.hexdigest()
        else:
            file_hash = "binary_skipped"
    except Exception:
        file_hash = "error"

    stat = abs_path.stat()
    language = detect_language(abs_path) or "unknown"

    # ファイル種別の判定 (kind)
    kind = "source"
    parts = Path(rel_path).parts
    if any(p == "tests" or p == "test" or p.startswith("test_") for p in parts) or abs_path.stem.startswith("test_") or abs_path.stem.endswith("_test"):
        kind = "test"
    elif abs_path.name in ("pyproject.toml", "requirements.txt", "package.json", "Makefile", "setup.py", "uv.lock", "Cargo.toml", "CMakeLists.txt", "go.mod", "go.sum", ".gitignore", "pnpm-lock.yaml", "yarn.lock"):
        kind = "config"
    elif any(p in ("docs", "doc", "wiki") for p in parts) or abs_path.suffix.lower() in (".md", ".rst", ".txt", ".pdf") or abs_path.name == "LICENSE":
        kind = "doc"
    elif is_probably_binary(abs_path):
        kind = "other"

    # Gitコミットハッシュの取得
    last_commit = get_git_commit_hash(abs_path)

    # 変更ステータスの判定 (status)
    status = "added"
    if previous_meta and rel_path in previous_meta:
        prev = previous_meta[rel_path]
        if prev.get("hash") == file_hash:
            status = "unchanged"
        else:
            status = "changed"
    else:
        # previous_meta がない場合は git status も参考にする
        git_status_map = get_git_status_info(root)
        if rel_path in git_status_map:
            status = git_status_map[rel_path]
        else:
            status = "added"

    todo_count = 0
    if kind == "source" and not is_probably_binary(abs_path):
        try:
            content = abs_path.read_text(encoding="utf-8", errors="ignore")
            todo_count = len(re.findall(r'(?:TODO|FIXME)[:\s]+(.*)', content, re.IGNORECASE))
        except Exception:
            pass

    return {
        "path": rel_path,
        "hash": file_hash,
        "mtime": int(stat.st_mtime),
        "size": stat.st_size,
        "language": language,
        "kind": kind,
        "last_seen_commit": last_commit,
        "status": status,
        "todo_count": todo_count,
    }


def _extract_python_symbols_and_imports(code: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    symbols = []
    imports = []
    exports = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return symbols, imports, exports

    # インポートの解析
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"module": alias.name, "internal": False})
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # 相対インポートか、またはパッケージ名がある場合
                is_internal = node.level > 0 or node.module.startswith(("src", "isohyps")) # 今回のプロジェクト依存の簡易チェック
                imports.append({"module": node.module, "internal": is_internal})

    # クラス・関数の解析
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.append({"name": node.name, "kind": "class", "line": node.lineno})
            # クラス内メソッド
            for subnode in node.body:
                if isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append({"name": f"{node.name}.{subnode.name}", "kind": "method", "line": subnode.lineno})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append({"name": node.name, "kind": "function", "line": node.lineno})
        elif isinstance(node, ast.Assign):
            # __all__ 定義の解析
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                exports.append(elt.value)

    # exports が __all__ で定義されていなかった場合のデフォルト（アンダースコアで始まらないグローバルな名前）
    if not exports:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                exports.append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                exports.append(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        exports.append(target.id)

    return symbols, imports, exports


def extract_file_symbols(path: Path, root: Path) -> dict[str, Any]:
    abs_path = path.resolve()
    rel_path = abs_path.relative_to(root.resolve()).as_posix()
    language = detect_language(abs_path)

    result = {
        "path": rel_path,
        "symbols": [],
        "imports": [],
        "exports": [],
    }

    if is_probably_binary(abs_path):
        return result

    try:
        code = abs_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return result

    # Python の場合は AST を使用
    if language == "python":
        symbols, imports, exports = _extract_python_symbols_and_imports(code)
        result["symbols"] = symbols
        result["imports"] = imports
        result["exports"] = exports
        return result

    # Python 以外は、tree_sitter があれば試み、なければ簡易的な正規表現
    try:
        from tree_sitter_languages import get_language, get_parser
        parser = get_parser(language)
        tree = parser.parse(code.encode("utf-8"))
        
        # 簡易的なシンボル抽出（既存のクエリと対応）
        from isohyps.analysis_helpers import SYMBOL_QUERIES
        query_str = SYMBOL_QUERIES.get(language, "")
        if query_str:
            lang_obj = get_language(language)
            query = lang_obj.query(query_str)
            captures = query.captures(tree.root_node)
            
            symbol_nodes = []
            if isinstance(captures, dict):
                symbol_nodes = captures.get("symbol", [])
            else:
                symbol_nodes = [node for node, name in captures if name == "symbol"]

            seen = set()
            for node in symbol_nodes:
                if node.start_byte in seen:
                    continue
                seen.add(node.start_byte)
                
                # ノードタイプから種別判定
                kind = "function"
                if "class" in node.type:
                    kind = "class"
                elif "method" in node.type:
                    kind = "method"
                elif "interface" in node.type:
                    kind = "class"

                # 簡易的な名前抽出（最初の1行からキーワードを探す）
                line_text = code.splitlines()[node.start_point[0]].strip()
                name_match = re.search(r'(?:class|def|function|func|fn|interface)\s+([a-zA-Z0-9_]+)', line_text)
                name = name_match.group(1) if name_match else line_text[:40]

                result["symbols"].append({
                    "name": name,
                    "kind": kind,
                    "line": node.start_point[0] + 1
                })
    except Exception:
        # 正規表現による簡易フォールバック
        lines = code.splitlines()
        for i, line in enumerate(lines):
            line_strip = line.strip()
            # 関数、メソッド、クラス定義の簡易パターン
            class_match = re.match(r'^\s*(?:class|struct|interface)\s+([a-zA-Z0-9_]+)', line_strip)
            if class_match:
                result["symbols"].append({"name": class_match.group(1), "kind": "class", "line": i + 1})
                continue
            
            fn_match = re.match(r'^\s*(?:def|function|func|fn)\s+([a-zA-Z0-9_]+)', line_strip)
            if fn_match:
                result["symbols"].append({"name": fn_match.group(1), "kind": "function", "line": i + 1})

    # インポートの簡易正規表現抽出
    for line in code.splitlines()[:250]:  # 冒頭250行に限定
        line_strip = line.strip()
        # ES6 import, CommonJS require, Go import, Rust use, Java import
        import_match = re.match(r'^(?:import|from|use|require)\s+[\'"]?([a-zA-Z0-9_\-\.\/]+)[\'"]?', line_strip)
        if import_match:
            module_name = import_match.group(1)
            is_internal = module_name.startswith((".", "/")) or any(p in module_name for p in ("src", "isohyps"))
            result["imports"].append({"module": module_name, "internal": is_internal})

    return result


def build_repo_map_summary(root: Path, files_meta: list[dict[str, Any]]) -> dict[str, Any]:
    directories = {}
    tests = []
    entrypoints = []

    for meta in files_meta:
        path_str = meta["path"]
        kind = meta["kind"]
        language = meta["language"]

        # ディレクトリ情報の集計
        parts = Path(path_str).parent.as_posix()
        dir_key = parts if parts != "." else "./"
        if dir_key not in directories:
            directories[dir_key] = {"files": 0, "languages": set()}
        
        directories[dir_key]["files"] += 1
        if language != "unknown":
            directories[dir_key]["languages"].add(language)

        # テストの集計
        if kind == "test":
            tests.append(path_str)

        # エントリポイントの簡易抽出（pyproject.toml, package.json などのパース）
        if meta["path"] == "pyproject.toml":
            try:
                content = (root / "pyproject.toml").read_text(encoding="utf-8")
                # 簡単な正規表現によるスクリプト抽出
                scripts = re.findall(r'([a-zA-Z0-9_\-]+)\s*=\s*[\'"]([a-zA-Z0-9_\.\:]+)[\'"]', content)
                for name, target in scripts:
                    entrypoints.append(f"pyproject.toml: {name} -> {target}")
            except Exception:
                pass
        elif meta["path"] == "package.json":
            try:
                content = (root / "package.json").read_text(encoding="utf-8")
                data = json.loads(content)
                bin_info = data.get("bin", {})
                if isinstance(bin_info, dict):
                    for k, v in bin_info.items():
                        entrypoints.append(f"package.json: bin.{k} -> {v}")
                elif isinstance(bin_info, str):
                    entrypoints.append(f"package.json: bin -> {bin_info}")
                scripts = data.get("scripts", {})
                for k, v in scripts.items():
                    entrypoints.append(f"package.json: script.{k} -> {v}")
            except Exception:
                pass
        elif Path(path_str).name in ("main.py", "app.py", "index.js", "main.go", "lib.rs"):
            entrypoints.append(f"Detected main file: {path_str}")

    # Set を List に変換
    for d in directories.values():
        d["languages"] = sorted(list(d["languages"]))

    return {
        "directories": directories,
        "entrypoints": entrypoints,
        "tests": sorted(tests),
    }


def detect_attention_points(
    root: Path,
    files_meta: list[dict[str, Any]],
    symbols_list: list[dict[str, Any]],
    previous_meta: dict[str, Any] | None = None
) -> list[str]:
    attention = []
    
    # マッピング情報の整理
    meta_by_path = {meta["path"]: meta for meta in files_meta}
    symbols_by_path = {sym["path"]: sym for sym in symbols_list}

    # インポートの集計 (fan-in / fan-out)
    fan_in = {}  # モジュールがインポートされている回数
    fan_out = {} # 各ファイルがインポートしている内部モジュールの数

    for path, sym in symbols_by_path.items():
        fan_out[path] = 0
        for imp in sym["imports"]:
            mod = imp["module"]
            is_internal = imp["internal"]
            
            if is_internal:
                fan_out[path] += 1
                # インポート元のモジュール名を簡易判定
                for other_path in meta_by_path.keys():
                    other_mod_name = Path(other_path).stem
                    if other_mod_name == mod.split(".")[-1]:
                        fan_in[other_path] = fan_in.get(other_path, 0) + 1

    # 1. 巨大ファイルの検出 (Large files: しきい値300行)
    for path, meta in meta_by_path.items():
        if meta["kind"] == "source":
            try:
                line_count = len((root / path).read_text(encoding="utf-8", errors="ignore").splitlines())
                if line_count > 300:
                    attention.append(f"file is large: {path}, {line_count} lines")
            except Exception:
                pass

    # 2. テストの有無の判定 (no tests found - __init__.py とサイズ50B未満の極小ファイルを除外)
    for path, meta in meta_by_path.items():
        if meta["kind"] == "source" and meta["language"] == "python":
            if Path(path).name == "__init__.py" or meta["size"] < 50:
                continue
            stem = Path(path).stem
            test_exists = False
            for test_path in symbols_by_path.keys():
                test_stem = Path(test_path).stem
                if test_stem in (f"test_{stem}", f"{stem}_test"):
                    test_exists = True
                    break
            if not test_exists:
                attention.append(f"no tests found for {path}")

    # 3. TODO/FIXME の検出 (過去のメタデータがあれば増加分だけを報告)
    for path, meta in meta_by_path.items():
        if meta["kind"] == "source" and not is_probably_binary(root / path):
            try:
                content = (root / path).read_text(encoding="utf-8", errors="ignore")
                matches = re.findall(r'(?:TODO|FIXME)[:\s]+(.*)', content, re.IGNORECASE)
                if matches:
                    prev_count = 0
                    has_previous = False
                    if previous_meta and path in previous_meta:
                        prev_todo = previous_meta[path].get("todo_count")
                        if prev_todo is not None:
                            prev_count = prev_todo
                            has_previous = True
                            
                    if has_previous:
                        if len(matches) > prev_count:
                            attention.append(f"TODO/FIXME count increased in {path}: {len(matches)} items")
                    else:
                        attention.append(f"TODO/FIXME count increased in {path}: {len(matches)} items")
            except Exception:
                pass

    # 4. high fan-in / fan-out
    for path, count in fan_in.items():
        if count >= 10:
            attention.append(f"high fan-in: {path} imported by {count} files")
            
    for path, count in fan_out.items():
        if count >= 15:
            attention.append(f"high fan-out: {path} imports {count} internal modules")

    return attention


def parse_gitignore(root: Path) -> list[str]:
    gitignore_path = root / ".gitignore"
    if not gitignore_path.exists():
        return []
    patterns = []
    with open(gitignore_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 空行、コメント、および否定パターンは簡易除外では処理しないため無視リストから除外
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            patterns.append(line)
    return patterns


def should_ignore(path: Path, root: Path, gitignore_patterns: list[str]) -> bool:
    try:
        rel_path = path.resolve().relative_to(root.resolve())
    except ValueError:
        return True
    rel_path_str = rel_path.as_posix()
    
    # ハードコードされた無視ディレクトリ
    ignore_dirs = {".git", "__pycache__", "venv", ".venv", ".env", "node_modules", ".vscode", ".idea", "dist", "build"}
    # 明示的に無視したい一時・キャッシュディレクトリ
    extra_ignore_dirs = {".pytest_cache", ".serena", ".kelpie"}
    
    for part in rel_path.parts:
        if part in ignore_dirs or part in extra_ignore_dirs or part.endswith(".egg-info"):
            return True

    # .gitignore パターンマッチ
    for pattern in gitignore_patterns:
        # ディレクトリ制限の処理
        if pattern.endswith("/"):
            p = pattern.rstrip("/")
            if any(fnmatch.fnmatch(part, p) for part in rel_path.parts):
                return True
        else:
            # 任意のファイル・ディレクトリ名にマッチ
            if fnmatch.fnmatch(rel_path_str, pattern) or any(fnmatch.fnmatch(part, pattern) for part in rel_path.parts):
                return True
            # ルートからの絶対的な指定
            if pattern.startswith("/"):
                p = pattern.lstrip("/")
                if fnmatch.fnmatch(rel_path_str, p):
                    return True
    return False


def generate_machine_report(
    root: Path,
    files_meta: list[dict[str, Any]],
    repo_map: dict[str, Any],
    attention: list[str]
) -> str:
    lines = [
        "# Project Machine Analysis Report",
        "",
        f"**Root Directory:** `{root.resolve()}`",
        f"**Total Files Discovered:** {len(files_meta)}",
        "",
        "## Repo Map Summary",
        "",
        "### Directories",
        "",
    ]

    for dir_path, info in repo_map["directories"].items():
        langs = ", ".join(info["languages"]) or "none"
        lines.append(f"- `{dir_path}`: {info['files']} files (languages: {langs})")

    lines.extend([
        "",
        "### Entrypoints",
        "",
    ])
    if repo_map["entrypoints"]:
        for ep in repo_map["entrypoints"]:
            lines.append(f"- {ep}")
    else:
        lines.append("- (none detected)")

    lines.extend([
        "",
        "### Tests",
        "",
    ])
    if repo_map["tests"]:
        for t in repo_map["tests"]:
            lines.append(f"- `{t}`")
    else:
        lines.append("- (none)")

    lines.extend([
        "",
        "## Attention Points",
        "",
    ])
    if attention:
        for att in attention:
            lines.append(f"- {att}")
    else:
        lines.append("- No critical risks or attention points detected.")

    lines.extend([
        "",
        "## File Inventory",
        "",
        "| Path | Kind | Language | Size (Bytes) | Hash | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for meta in files_meta:
        lines.append(
            f"| `{meta['path']}` | {meta['kind']} | {meta['language']} | {meta['size']} | `{meta['hash'][:10]}` | {meta['status']} |"
        )

    return "\n".join(lines)


def analyze_machine_level(root_path: Path, output_dir: Path) -> dict[str, Any]:
    root = root_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 前回の結果をロードして変更履歴を比較
    json_path = output_dir / "machine_analysis.json"
    previous_meta = None
    if json_path.exists():
        try:
            prev_data = json.loads(json_path.read_text(encoding="utf-8"))
            previous_meta = {f["path"]: f for f in prev_data.get("files", [])}
        except Exception:
            pass

    # .gitignore パターンの取得
    gitignore_patterns = parse_gitignore(root)

    # ファイルの走査
    all_files = []
    ignored_count = 0
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            if should_ignore(p, root, gitignore_patterns):
                ignored_count += 1
                continue
            all_files.append(p)

    # メタデータ抽出
    files_meta = []
    for f in all_files:
        files_meta.append(extract_file_metadata(f, root, previous_meta))

    # シンボル抽出
    symbols_list = []
    for f in all_files:
        symbols_list.append(extract_file_symbols(f, root))

    # repo_map サマリー作成
    repo_map = build_repo_map_summary(root, files_meta)

    # アテンションポイント検出
    attention = detect_attention_points(root, files_meta, symbols_list, previous_meta)

    # 最終データの統合
    result = {
        "files": files_meta,
        "symbols": symbols_list,
        "repo_map": repo_map,
        "attention": attention,
    }

    # 機械向け JSON 書き出し
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 機械向け YAML 書き出し (PyYAML非依存)
    yaml_path = output_dir / "machine_analysis.yaml"
    yaml_path.write_text(simple_yaml_dump(result), encoding="utf-8")

    # 人間向け Markdown 書き出し
    report_path = output_dir / "machine_report.md"
    report_content = generate_machine_report(root, files_meta, repo_map, attention)
    report_path.write_text(report_content, encoding="utf-8")

    # ドキュメントの存在有無・更新チェックとカバレッジの算出
    missing_docs = []
    stale_docs = []
    valid_docs = []
    
    # 対象ファイル： kind が source または test であり、バイナリやエラーでないもの
    coverage_targets = [m for m in files_meta if m["kind"] in ("source", "test") and m["hash"] not in ("binary_skipped", "error")]
    
    for meta in coverage_targets:
        rel_path = meta["path"]
        doc_candidates = [
            output_dir / f"{rel_path}.md",
            output_dir / Path(rel_path).with_suffix(".md")
        ]
        
        found_doc = None
        for candidate in doc_candidates:
            if candidate.exists() and candidate.is_file():
                found_doc = candidate
                break
                
        if not found_doc:
            missing_docs.append(rel_path)
        else:
            source_mtime = meta["mtime"]
            doc_mtime = found_doc.stat().st_mtime
            is_unchanged = (meta["status"] == "unchanged")
            
            # ソースコード更新日時がドキュメント更新日時 + 2.0秒より新しく、かつ内容に変更がある場合を stale と判定
            if source_mtime > doc_mtime + 2.0 and not is_unchanged:
                stale_docs.append(rel_path)
            else:
                valid_docs.append(rel_path)
                
    total_targets = len(coverage_targets)
    documented_count = total_targets - len(missing_docs)
    coverage_percent = (documented_count / total_targets * 100) if total_targets > 0 else 100.0

    # analysis_report.md の自動合成
    analysis_report = (
        f"# Project Analysis Report: {root.name}\n\n"
        f"**Root Directory:** `{root}`  \n"
        f"**Backend:** none (machine scan only)  \n"
        f"**Runtime:** controller  \n"
        f"**Status:** success  \n"
        f"**Total Files Scanned:** {len(all_files)}  \n"
        f"**Total Steps Used:** 0  \n"
        f"**Total Tokens Used:** 0  \n\n"
        f"## Executive Summary\n"
        f"This report summarizes the static machine scan of the project. No LLM resources were utilized.\n\n"
        f"## Machine Analysis Status\n"
        f"- **Source Coverage:** {coverage_percent:.1f}%\n"
        f"- **Attention Points Detected:** {len(attention)} items\n"
        f"- **Active Files Count:** {len(all_files)}\n"
        f"- **Ignored Files Count:** {ignored_count}\n\n"
        f"Refer to [machine_report.md](./machine_report.md) for the detailed inventory and symbol details.\n"
    )
    (output_dir / "analysis_report.md").write_text(analysis_report, encoding="utf-8")

    # index.md の自動合成
    changed_files = [m["path"] for m in files_meta if m["status"] in ("changed", "added")]
    # 解説が必要なファイル（新規・変更されたファイル、解説欠損、陳腐化ドキュメント）
    needs_explanation = set(changed_files) | set(missing_docs) | set(stale_docs)
    
    index_content = (
        f"# Directory: {root.name}\n\n"
        f"Welcome to the static analysis index for `{root.name}`.\n\n"
        f"## Project Summary\n"
        f"- **Total Source Files:** {len(all_files)}\n"
        f"- **Status:** Static Scan Complete\n\n"
        f"## Stale or Newly Added Files (Need Explanation)\n"
        f"These files are either new or modified, and their corresponding documentation (if any) may be stale.\n"
    )
    if needs_explanation:
        for cf in sorted(list(needs_explanation)):
            reasons = []
            if cf in missing_docs:
                reasons.append("missing doc")
            elif cf in stale_docs:
                reasons.append("stale doc")
            
            m_status = next((m["status"] for m in files_meta if m["path"] == cf), None)
            if m_status in ("changed", "added"):
                reasons.append(f"source {m_status}")
                
            reason_str = f" ({', '.join(reasons)})" if reasons else ""
            index_content += f"- [ ] `{cf}`{reason_str}\n"
    else:
        index_content += "- No modified or added files detected. All files are unchanged.\n"
        
    index_content += "\n## High Priority Files to Inspect (Attention Points)\nThese files have warnings or high complexity:\n"
    if attention:
        for att in attention:
            index_content += f"- {att}\n"
    else:
        index_content += "- No critical attention points.\n"
        
    index_content += f"\n## Directory Structure Overview\nRefer to [machine_report.md](./machine_report.md) for the full inventory.\n"
    (output_dir / "index.md").write_text(index_content, encoding="utf-8")

    return result
