from __future__ import annotations

import os
import ast
import re
import json
import fnmatch
import hashlib
import stat as stat_module
import subprocess
import tomllib
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from isohyps.analysis_helpers import detect_language, is_probably_binary, extract_symbols
from isohyps.attention import (
    AttentionContractError,
    AttentionDiagnostic,
    AttentionSignalSnapshot,
    DOC_STATUSES,
    DocStatus,
    AttentionEntry,
    SEVERITY_ORDER,
    classify_attention,
    validate_repository_relative_path,
)
from isohyps.machine_index import (
    build_machine_index_v2,
    validate_machine_index,
    write_machine_index_atomic,
)

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


def _build_coverage_targets(files_meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        meta
        for meta in files_meta
        if meta["kind"] == "source"
        and meta["language"] != "unknown"
        and meta["hash"] not in ("binary_skipped", "error")
    ]


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
    parent_parts = parts[:-1]
    is_test_dir = any(p in ("tests", "test") or p.startswith("test_") for p in parent_parts)
    is_test_file = abs_path.stem.startswith("test_") or abs_path.stem.endswith("_test")
    is_helper_file = any(w in abs_path.stem.lower() for w in ("helper", "util", "fixture", "mock", "stub"))

    # よくある設定ファイル・デプロイファイル等のリスト
    config_names = {
        "pyproject.toml", "requirements.txt", "package.json", "Makefile", "setup.py", 
        "uv.lock", "Cargo.toml", "CMakeLists.txt", "go.mod", "go.sum", ".gitignore", 
        "pnpm-lock.yaml", "yarn.lock", "Dockerfile", "docker-compose.yml", "compose.yml", 
        "compose.yaml", ".dockerignore", ".editorconfig", "conftest.py", "__init__.py"
    }

    is_github_asset = any(p == ".github" for p in parent_parts)
    is_config_extension = abs_path.suffix.lower() in (".yaml", ".yml", ".json", ".toml", ".xml", ".ini", ".cfg")

    if is_test_dir or (is_test_file and not is_helper_file):
        kind = "test"
    elif abs_path.name in config_names or is_github_asset or is_config_extension:
        kind = "config"
    elif any(p in ("docs", "doc", "wiki") for p in parent_parts) or abs_path.suffix.lower() in (".md", ".rst", ".txt", ".pdf") or abs_path.name == "LICENSE":
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

    readable = file_hash not in ("binary_skipped", "error")
    line_count: int | None = None
    todo_count = 0
    if kind == "source" and readable and not is_probably_binary(abs_path):
        try:
            content = abs_path.read_text(encoding="utf-8", errors="ignore")
            line_count = len(content.splitlines())
            todo_count = len(re.findall(r'(?:TODO|FIXME)[:\s]+(.*)', content, re.IGNORECASE))
        except Exception:
            readable = False

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
        "line_count": line_count,
        "readable": readable,
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
        "classes": [],
        "functions": [],
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
        result["classes"] = [s["name"] for s in symbols if s.get("kind") == "class"]
        result["functions"] = [s["name"] for s in symbols if s.get("kind") == "function"]
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

    result["classes"] = [s["name"] for s in result.get("symbols", []) if s.get("kind") == "class"]
    result["functions"] = [s["name"] for s in result.get("symbols", []) if s.get("kind") == "function"]

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

    # 4. high fan-in / fan-out (ファイルの厳密な依存関係メトリクスに基づく)
    for path, meta in meta_by_path.items():
        fi_count = meta.get("fan_in", 0)
        fo_count = meta.get("fan_out", 0)
        if fi_count >= 5:
            attention.append(f"high fan-in: {path} imported by {fi_count} files")
        if fo_count >= 15:
            attention.append(f"high fan-out: {path} imports {fo_count} internal modules")

    return attention


_MISSING = object()


def _append_attention_diagnostic(
    diagnostics: list[AttentionDiagnostic],
    detector: str,
    code: str,
    path: str | None = None,
) -> None:
    """Append one stable diagnostic unless the same failure was already seen."""

    diagnostic = AttentionDiagnostic(detector=detector, code=code, path=path)  # type: ignore[arg-type]
    if diagnostic not in diagnostics:
        diagnostics.append(diagnostic)


def _sort_attention_diagnostics(
    diagnostics: Sequence[AttentionDiagnostic],
) -> list[AttentionDiagnostic]:
    return sorted(
        set(diagnostics),
        key=lambda diagnostic: (
            diagnostic.detector,
            diagnostic.path or "",
            diagnostic.code,
        ),
    )


def _normalise_metadata_path(
    value: Any,
    diagnostics: list[AttentionDiagnostic],
    *,
    detector: str = "metadata",
) -> str | None:
    if isinstance(value, Path):
        value = value.as_posix()
    if not isinstance(value, str):
        _append_attention_diagnostic(diagnostics, detector, "invalid_path")
        return None
    try:
        return validate_repository_relative_path(value)
    except AttentionContractError:
        _append_attention_diagnostic(diagnostics, detector, "invalid_path", value)
        return None


def _ordered_attention_metadata(
    files_meta: Sequence[Mapping[str, Any]],
    diagnostics: list[AttentionDiagnostic],
) -> list[tuple[str, Mapping[str, Any]]]:
    entries: list[tuple[str, Mapping[str, Any]]] = []
    for metadata in files_meta:
        if not isinstance(metadata, Mapping):
            _append_attention_diagnostic(diagnostics, "metadata", "invalid_metadata")
            continue
        path = _normalise_metadata_path(metadata.get("path"), diagnostics)
        if path is not None:
            entries.append((path, metadata))

    entries.sort(key=lambda entry: entry[0])
    unique_entries: list[tuple[str, Mapping[str, Any]]] = []
    seen_paths: set[str] = set()
    for path, metadata in entries:
        if path in seen_paths:
            _append_attention_diagnostic(diagnostics, "metadata", "duplicate_path", path)
            continue
        seen_paths.add(path)
        unique_entries.append((path, metadata))
    return unique_entries


def _metadata_integer(
    metadata: Mapping[str, Any],
    key: str,
    diagnostics: list[AttentionDiagnostic],
    path: str,
    *,
    missing_is_none: bool = True,
    missing_code: str | None = None,
) -> int | None:
    value = metadata.get(key, _MISSING)
    if value is _MISSING or (value is None and missing_is_none):
        if missing_code is not None:
            _append_attention_diagnostic(diagnostics, "metadata", missing_code, path)
        return None
    if type(value) is not int or value < 0:
        _append_attention_diagnostic(diagnostics, "metadata", f"invalid_{key}", path)
        return None
    return value


def _metadata_boolean(
    metadata: Mapping[str, Any],
    key: str,
    diagnostics: list[AttentionDiagnostic],
    path: str,
) -> bool | None:
    value = metadata.get(key, _MISSING)
    if value is _MISSING:
        return None
    if type(value) is not bool:
        _append_attention_diagnostic(diagnostics, "metadata", f"invalid_{key}", path)
        return None
    return value


def _previous_attention_metadata(
    previous_meta: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    diagnostics: list[AttentionDiagnostic],
) -> dict[str, Mapping[str, Any]]:
    """Normalize both the old path map and a previous ``files`` array."""

    if previous_meta is None:
        return {}

    raw_entries: list[tuple[Any, Any]] = []
    if isinstance(previous_meta, Mapping):
        files = previous_meta.get("files")
        if isinstance(files, Sequence) and not isinstance(files, (str, bytes, bytearray)):
            raw_entries = [
                (entry.get("path") if isinstance(entry, Mapping) else None, entry)
                for entry in files
            ]
        else:
            raw_entries = list(previous_meta.items())
    elif isinstance(previous_meta, Sequence) and not isinstance(
        previous_meta, (str, bytes, bytearray)
    ):
        raw_entries = [
            (entry.get("path") if isinstance(entry, Mapping) else None, entry)
            for entry in previous_meta
        ]
    else:
        _append_attention_diagnostic(diagnostics, "metadata", "invalid_previous_metadata")
        return {}

    result: dict[str, Mapping[str, Any]] = {}
    for raw_path, entry in raw_entries:
        path = _normalise_metadata_path(raw_path, diagnostics)
        if path is None:
            continue
        if not isinstance(entry, Mapping):
            _append_attention_diagnostic(
                diagnostics, "metadata", "invalid_previous_metadata", path
            )
            continue
        if path in result:
            _append_attention_diagnostic(diagnostics, "metadata", "duplicate_previous_path", path)
            continue
        result[path] = entry
    return result


def _attention_coverage_target(metadata: Mapping[str, Any]) -> bool:
    return (
        metadata.get("kind") == "source"
        and metadata.get("language") != "unknown"
        and metadata.get("hash") not in ("binary_skipped", "error")
        and metadata.get("readable", True) is not False
    )


def _doc_status_snapshot_for_entries(
    root: Path | None,
    entries: Sequence[tuple[str, Mapping[str, Any]]],
    output_dir: Path,
    diagnostics: list[AttentionDiagnostic],
) -> dict[str, DocStatus]:
    """Calculate deterministic document status for already-normalized metadata."""

    statuses: dict[str, DocStatus] = {path: "current" for path, _ in entries}
    output_root = Path(output_dir).resolve()

    for path, metadata in entries:
        if metadata.get("kind") == "source" and (
            metadata.get("readable") is False
            or metadata.get("hash") in ("binary_skipped", "error")
        ):
            statuses[path] = "unavailable"
            _append_attention_diagnostic(
                diagnostics,
                "metadata",
                "binary_skipped"
                if metadata.get("hash") == "binary_skipped"
                else "read_unavailable",
                path,
            )
            continue
        if not _attention_coverage_target(metadata):
            continue

        found_doc_stat: os.stat_result | None = None
        for candidate in (
            output_root / f"{path}.md",
            output_root / Path(path).with_suffix(".md"),
        ):
            try:
                candidate_stat = candidate.stat()
            except FileNotFoundError:
                continue
            except OSError:
                statuses[path] = "unavailable"
                _append_attention_diagnostic(
                    diagnostics, "doc_status", "stat_failed", path
                )
                break
            if stat_module.S_ISREG(candidate_stat.st_mode):
                found_doc_stat = candidate_stat
                break
        else:
            statuses[path] = "missing"

        if statuses[path] == "unavailable" or found_doc_stat is None:
            continue

        # Preserve the existing behavior: an unchanged source is not stale
        # merely because the previous documentation has an older mtime.
        if metadata.get("status") == "unchanged":
            continue

        source_mtime = metadata.get("mtime", _MISSING)
        if type(source_mtime) not in (int, float):
            if root is not None:
                try:
                    source_mtime = (root / path).stat().st_mtime
                except OSError:
                    source_mtime = _MISSING
            else:
                source_mtime = _MISSING

        if source_mtime is _MISSING or type(source_mtime) is bool:
            statuses[path] = "unavailable"
            _append_attention_diagnostic(
                diagnostics, "doc_status", "source_mtime_unavailable", path
            )
            continue

        doc_mtime = found_doc_stat.st_mtime
        if type(doc_mtime) is bool or type(doc_mtime) not in (int, float):
            statuses[path] = "unavailable"
            _append_attention_diagnostic(
                diagnostics, "doc_status", "doc_mtime_unavailable", path
            )
            continue

        if source_mtime > doc_mtime + 2.0:
            statuses[path] = "stale"

    return statuses


def build_doc_status_snapshot(
    root: Path,
    files_meta: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> tuple[dict[str, DocStatus], list[AttentionDiagnostic]]:
    """Return per-path doc status and stable diagnostics for a scan."""

    diagnostics: list[AttentionDiagnostic] = []
    try:
        metadata = list(files_meta)
    except TypeError:
        _append_attention_diagnostic(diagnostics, "metadata", "invalid_metadata")
        return {}, _sort_attention_diagnostics(diagnostics)
    entries = _ordered_attention_metadata(metadata, diagnostics)
    statuses = _doc_status_snapshot_for_entries(
        Path(root).resolve(), entries, Path(output_dir), diagnostics
    )
    return statuses, _sort_attention_diagnostics(diagnostics)


def _normalise_doc_statuses(
    entries: Sequence[tuple[str, Mapping[str, Any]]],
    supplied: Mapping[str, Any],
    diagnostics: list[AttentionDiagnostic],
) -> dict[str, DocStatus]:
    statuses: dict[str, DocStatus] = {path: "current" for path, _ in entries}
    known_paths = set(statuses)
    for raw_path, raw_status in supplied.items():
        path = _normalise_metadata_path(raw_path, diagnostics, detector="doc_status")
        if path is None or path not in known_paths:
            continue
        if raw_status not in DOC_STATUSES:
            statuses[path] = "unavailable"
            _append_attention_diagnostic(
                diagnostics, "doc_status", "invalid_status", path
            )
            continue
        statuses[path] = raw_status
        if raw_status == "unavailable":
            _append_attention_diagnostic(
                diagnostics, "doc_status", "status_unavailable", path
            )
    return statuses


def _metadata_doc_statuses(
    entries: Sequence[tuple[str, Mapping[str, Any]]],
    diagnostics: list[AttentionDiagnostic],
) -> dict[str, DocStatus]:
    statuses: dict[str, DocStatus] = {path: "current" for path, _ in entries}
    for path, metadata in entries:
        if "doc_status" not in metadata:
            continue
        raw_status = metadata["doc_status"]
        if raw_status not in DOC_STATUSES:
            statuses[path] = "unavailable"
            _append_attention_diagnostic(
                diagnostics, "doc_status", "invalid_status", path
            )
            continue
        statuses[path] = raw_status
        if raw_status == "unavailable":
            _append_attention_diagnostic(
                diagnostics, "doc_status", "status_unavailable", path
            )
    return statuses


def _normalise_entrypoint_paths(
    entrypoint_paths: Collection[str] | None,
    diagnostics: list[AttentionDiagnostic],
) -> set[str]:
    if entrypoint_paths is None:
        return set()
    values: Collection[Any] = entrypoint_paths
    if isinstance(entrypoint_paths, str):
        values = (entrypoint_paths,)
    result: set[str] = set()
    for value in values:
        path = _normalise_metadata_path(value, diagnostics, detector="entrypoint")
        if path is not None:
            result.add(path)
    return result


def _source_metrics(
    root: Path | None,
    path: str,
    readable: bool,
    line_count: int | None,
    todo_count: int | None,
    diagnostics: list[AttentionDiagnostic],
) -> tuple[bool, int | None, int]:
    """Reuse metadata counts and fill missing counts with one text read."""

    if not readable:
        return False, None, 0

    needs_line_count = line_count is None
    needs_todo_count = todo_count is None
    if not needs_line_count and not needs_todo_count:
        assert line_count is not None
        assert todo_count is not None
        return True, line_count, todo_count

    if root is None:
        if needs_line_count:
            _append_attention_diagnostic(
                diagnostics, "metadata", "line_count_unavailable", path
            )
        if needs_todo_count:
            _append_attention_diagnostic(
                diagnostics, "metadata", "todo_count_unavailable", path
            )
        return True, line_count, todo_count or 0

    try:
        content = (root / path).read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeError):
        _append_attention_diagnostic(diagnostics, "metadata", "read_failed", path)
        return False, None, 0

    if needs_line_count:
        line_count = len(content.splitlines())
    if needs_todo_count:
        todo_count = len(
            re.findall(r"(?:TODO|FIXME)[:\s]+(.*)", content, re.IGNORECASE)
        )
    return True, line_count, todo_count or 0


def build_attention_snapshots(
    root: Path | None,
    files_meta: Sequence[Mapping[str, Any]],
    previous_meta: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    output_dir: Path | None = None,
    *,
    doc_status_by_path: Mapping[str, DocStatus] | None = None,
    entrypoint_paths: Collection[str] | None = None,
) -> tuple[list[AttentionSignalSnapshot], list[AttentionDiagnostic]]:
    """Build one normalized attention snapshot per repository-relative path.

    The builder consumes scan metadata and optional precomputed doc/entrypoint
    facts.  It does not classify findings or read Git/history.  When a count
    is absent from metadata, at most one source read is used to fill the
    missing line/TODO values; an unavailable read is returned as a stable
    diagnostic and does not become a false finding.
    """

    diagnostics: list[AttentionDiagnostic] = []
    try:
        metadata = list(files_meta)
    except TypeError:
        _append_attention_diagnostic(diagnostics, "metadata", "invalid_metadata")
        return [], _sort_attention_diagnostics(diagnostics)

    entries = _ordered_attention_metadata(metadata, diagnostics)
    root_path = Path(root).resolve() if root is not None else None
    previous_by_path = _previous_attention_metadata(previous_meta, diagnostics)
    test_stems: set[str] = set()
    for path, item in entries:
        if item.get("kind") == "test":
            test_stems.add(PurePosixPath(path).stem)

    if doc_status_by_path is not None:
        doc_statuses = _normalise_doc_statuses(
            entries, doc_status_by_path, diagnostics
        )
    elif isinstance(output_dir, Mapping):
        # Accept the compact positional form used by callers that already
        # computed statuses: ``build_attention_snapshots(root, meta, prev,
        # statuses)``.  A real output directory is always path-like.
        doc_statuses = _normalise_doc_statuses(entries, output_dir, diagnostics)
        output_dir = None
    elif output_dir is not None:
        if root_path is None:
            doc_statuses = {path: "current" for path, _ in entries}
            for path, item in entries:
                if _attention_coverage_target(item):
                    doc_statuses[path] = "unavailable"
                    _append_attention_diagnostic(
                        diagnostics, "doc_status", "root_unavailable", path
                    )
        else:
            doc_statuses = _doc_status_snapshot_for_entries(
                root_path, entries, Path(output_dir), diagnostics
            )
    else:
        doc_statuses = _metadata_doc_statuses(entries, diagnostics)

    # A source whose contents could not be read cannot have a trustworthy
    # coverage status, even when a caller supplied a stale/missing override.
    for path, item in entries:
        if item.get("kind") == "source" and (
            item.get("readable") is False
            or item.get("hash") in ("binary_skipped", "error")
        ):
            doc_statuses[path] = "unavailable"
            _append_attention_diagnostic(
                diagnostics,
                "metadata",
                "binary_skipped"
                if item.get("hash") == "binary_skipped"
                else "read_unavailable",
                path,
            )

    resolved_entrypoints = _normalise_entrypoint_paths(
        entrypoint_paths, diagnostics
    )
    snapshots: list[AttentionSignalSnapshot] = []

    for path, item in entries:
        kind = item.get("kind", "other")
        if not isinstance(kind, str) or not kind:
            _append_attention_diagnostic(diagnostics, "metadata", "invalid_kind", path)
            kind = "other"
        language = item.get("language", "unknown")
        if not isinstance(language, str) or not language:
            _append_attention_diagnostic(
                diagnostics, "metadata", "invalid_language", path
            )
            language = "unknown"

        readable_value = _metadata_boolean(item, "readable", diagnostics, path)
        if readable_value is None:
            readable = item.get("hash") not in ("binary_skipped", "error")
        else:
            readable = readable_value
        if item.get("hash") in ("binary_skipped", "error"):
            readable = False
            if kind == "source":
                _append_attention_diagnostic(
                    diagnostics,
                    "metadata",
                    "binary_skipped" if item.get("hash") == "binary_skipped" else "read_unavailable",
                    path,
                )
        elif kind == "source" and not readable:
            _append_attention_diagnostic(
                diagnostics, "metadata", "read_unavailable", path
            )

        line_count = _metadata_integer(item, "line_count", diagnostics, path)
        todo_count = _metadata_integer(item, "todo_count", diagnostics, path)
        if kind == "source":
            readable, line_count, todo_count = _source_metrics(
                root_path,
                path,
                readable,
                line_count,
                todo_count,
                diagnostics,
            )
        else:
            line_count = None
            todo_count = 0

        previous_count: int | None = None
        previous_item = previous_by_path.get(path)
        previous_entry_exists = previous_item is not None
        if previous_entry_exists:
            previous_count = _metadata_integer(
                previous_item,
                "todo_count",
                diagnostics,
                path,
                missing_code="previous_todo_count_unavailable",
            )
        todo_increased = False
        if kind == "source" and readable:
            current_count = todo_count or 0
            if previous_entry_exists:
                # An existing but unavailable baseline is not a first scan.
                todo_increased = (
                    previous_count is not None and current_count > previous_count
                )
            else:
                todo_increased = current_count > 0
        else:
            current_count = 0

        test_missing = False
        if kind == "source" and language == "python":
            if PurePosixPath(path).name not in ("__init__.py",):
                size = _metadata_integer(item, "size", diagnostics, path)
                if size is None and root_path is not None:
                    try:
                        size = (root_path / path).stat().st_size
                    except OSError:
                        _append_attention_diagnostic(
                            diagnostics, "test_index", "size_unavailable", path
                        )
                elif size is None:
                    _append_attention_diagnostic(
                        diagnostics, "test_index", "size_unavailable", path
                    )
                if size is not None and size >= 50:
                    source_stem = PurePosixPath(path).stem
                    test_missing = not {
                        f"test_{source_stem}",
                        f"{source_stem}_test",
                    }.intersection(test_stems)

        fan_in = _metadata_integer(
            item,
            "fan_in",
            diagnostics,
            path,
            missing_code="fan_in_unavailable",
        )
        fan_out = _metadata_integer(
            item,
            "fan_out",
            diagnostics,
            path,
            missing_code="fan_out_unavailable",
        )
        if fan_in is None or fan_out is None:
            # The snapshot contract uses integers, so an unavailable graph
            # metric cannot safely be represented as zero.
            continue
        entrypoint_flag = _metadata_boolean(item, "is_entrypoint", diagnostics, path)
        is_entrypoint = bool(entrypoint_flag) if entrypoint_flag is not None else False
        is_entrypoint = is_entrypoint or path in resolved_entrypoints

        snapshots.append(
            AttentionSignalSnapshot(
                path=path,
                kind=kind,
                language=language,
                readable=readable,
                line_count=line_count,
                fan_in=fan_in,
                fan_out=fan_out,
                test_missing=test_missing,
                todo_current=current_count,
                todo_previous=previous_count,
                todo_increased=todo_increased,
                is_entrypoint=is_entrypoint,
                doc_status=doc_statuses.get(path, "current"),
            )
        )

    return snapshots, _sort_attention_diagnostics(diagnostics)


# Name used by callers that describe the result as a map of statuses.
build_doc_statuses = build_doc_status_snapshot


def resolve_attention_entrypoints(
    root: Path, files_meta: Sequence[Mapping[str, Any]]
) -> tuple[set[str], list[AttentionDiagnostic]]:
    """Resolve supported configuration entrypoints to known source paths."""

    diagnostics: list[AttentionDiagnostic] = []
    known_sources = {
        str(item.get("path"))
        for item in files_meta
        if item.get("kind") == "source" and isinstance(item.get("path"), str)
    }
    resolved = {
        path
        for path in known_sources
        if PurePosixPath(path).name
        in ("main.py", "app.py", "index.js", "main.go", "lib.rs")
    }

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            project = data.get("project", {})
            targets: list[Any] = []
            if isinstance(project, Mapping):
                for section in ("scripts", "gui-scripts"):
                    values = project.get(section, {})
                    if isinstance(values, Mapping):
                        targets.extend(values.values())
            for target in targets:
                if not isinstance(target, str):
                    _append_attention_diagnostic(
                        diagnostics, "entrypoint", "target_unresolved", "pyproject.toml"
                    )
                    continue
                module = target.split(":", 1)[0]
                relative = module.replace(".", "/")
                candidates = {
                    candidate
                    for candidate in (
                        f"{relative}.py",
                        f"{relative}/__init__.py",
                        f"src/{relative}.py",
                        f"src/{relative}/__init__.py",
                    )
                    if candidate in known_sources
                }
                if len(candidates) == 1:
                    resolved.update(candidates)
                else:
                    _append_attention_diagnostic(
                        diagnostics,
                        "entrypoint",
                        "target_ambiguous" if candidates else "target_unresolved",
                        "pyproject.toml",
                    )
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            _append_attention_diagnostic(
                diagnostics, "entrypoint", "parse_failed", "pyproject.toml"
            )

    package_json = root / "package.json"
    if package_json.is_file():
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
            bin_value = data.get("bin") if isinstance(data, Mapping) else None
            targets = list(bin_value.values()) if isinstance(bin_value, Mapping) else [bin_value]
            for target in targets:
                if not isinstance(target, str):
                    continue
                candidate = target.removeprefix("./")
                try:
                    validate_repository_relative_path(candidate)
                except AttentionContractError:
                    candidate = ""
                if candidate in known_sources:
                    resolved.add(candidate)
                else:
                    _append_attention_diagnostic(
                        diagnostics, "entrypoint", "target_unresolved", "package.json"
                    )
        except (OSError, UnicodeError, json.JSONDecodeError):
            _append_attention_diagnostic(
                diagnostics, "entrypoint", "parse_failed", "package.json"
            )

    return resolved, _sort_attention_diagnostics(diagnostics)



def resolve_module_to_path(module_name: str, src_files: list[str], current_file: str) -> str | None:
    # 1. 相対インポートの解決
    if module_name.startswith("."):
        dots_count = 0
        for char in module_name:
            if char == ".":
                dots_count += 1
            else:
                break
        
        curr_parts = current_file.split("/")
        if len(curr_parts) > dots_count:
            base_dir_parts = curr_parts[:-dots_count]
            rem_mod = module_name[dots_count:]
            if rem_mod:
                resolved_parts = base_dir_parts + rem_mod.split(".")
            else:
                resolved_parts = base_dir_parts
            
            possible_rel_path = "/".join(resolved_parts)
            for src in src_files:
                src_no_ext = src.rsplit(".", 1)[0]
                if src_no_ext == possible_rel_path:
                    return src
                if src_no_ext + "/__init__" == possible_rel_path:
                    return src
        return None

    # 2. 絶対インポートの解決
    for src in src_files:
        src_no_ext = src.rsplit(".", 1)[0]
        dotted_src = src_no_ext.replace("/", ".")
        if dotted_src == module_name:
            return src
        if src_no_ext.endswith("/__init__"):
            package_dotted = src_no_ext[:-9].replace("/", ".")
            if package_dotted == module_name:
                return src
        # パッケージの階層構造（サブモジュールやクラスなどのインポート）に対応
        if module_name.startswith(dotted_src + "."):
            return src

    return None


def build_dependency_graph(files_meta: list[dict[str, Any]], symbols_list: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    # kind == "source" かつ language != "unknown" の通常ソースファイルのみを対象とする
    files = [m["path"] for m in files_meta if m["kind"] == "source" and m["language"] != "unknown"]
    symbols_by_path = {sym["path"]: sym for sym in symbols_list}
    
    forward_graph = {f: [] for f in files}
    backward_graph = {f: [] for f in files}
    
    for f in files:
        if f not in symbols_by_path:
            continue
        sym_info = symbols_by_path[f]
        imports = sym_info.get("imports", [])
        
        for imp in imports:
            mod_name = imp["module"]
            resolved_path = resolve_module_to_path(mod_name, files, f)
            if resolved_path and resolved_path != f:
                if resolved_path not in forward_graph[f]:
                    forward_graph[f].append(resolved_path)
                if f not in backward_graph[resolved_path]:
                    backward_graph[resolved_path].append(f)
                    
    return forward_graph, backward_graph


def topological_sort(graph: dict[str, list[str]]) -> list[str]:
    # 入次数を計算
    in_degree = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] = in_degree.get(v, 0) + 1
            
    # 入次数0のノードをキューに追加 (決定論的な順序を保つためソート)
    queue = sorted([u for u in graph if in_degree[u] == 0])
    order = []
    
    while queue:
        u = queue.pop(0)
        order.append(u)
        
        for v in sorted(graph.get(u, [])):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                queue.sort() # キューの中身を常にソートして決定論的な取り出し順を保つ
                
    # 循環参照などによりソートしきれなかったノードを決定論的順序で追加 (無限ループ防止ガード)
    remaining = sorted([u for u in graph if u not in order])
    if remaining:
        order.extend(remaining)
        
    return order


def generate_mermaid_graph(forward_graph: dict[str, list[str]]) -> str:
    # 描画制限: ノード数が 50 を超える場合は描画をスキップして警告
    total_nodes = len(forward_graph)
    if total_nodes > 50:
        return (
            "```text\n"
            "Dependency graph is too large to display as a Mermaid diagram.\n"
            f"Total files: {total_nodes} (limit: 50)\n"
            "```"
        )
        
    lines = ["```mermaid", "graph TD"]
    
    # 依存関係（エッジ）があるノード、または依存されているノードのみを描画対象とする
    active_nodes = set()
    for u, vs in forward_graph.items():
        if vs:
            active_nodes.add(u)
            for v in vs:
                active_nodes.add(v)
                
    if not active_nodes:
        return "*(No internal module dependencies detected)*"
        
    # ノードの定義 (IDとラベル)
    node_ids = {}
    for i, node in enumerate(sorted(list(active_nodes))):
        node_ids[node] = f"node_{i}"
        label = node
        parts = Path(node).parts
        if len(parts) > 1:
            label = f"{parts[-2]}/{parts[-1]}"
        lines.append(f"  {node_ids[node]}[\"{label}\"]")
        
    # エッジの定義
    for u in sorted(forward_graph.keys()):
        if u not in node_ids:
            continue
        for v in sorted(forward_graph[u]):
            if v not in node_ids:
                continue
            lines.append(f"  {node_ids[u]} --> {node_ids[v]}")
            
    lines.append("```")
    return "\n".join(lines)


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


def _is_path_within(path: Path, directory: Path) -> bool:
    """Return whether ``path`` is inside ``directory`` after resolution."""

    try:
        path.resolve().relative_to(directory)
    except ValueError:
        return False
    return True


def generate_machine_report(
    root: Path,
    files_meta: list[dict[str, Any]],
    repo_map: dict[str, Any],
    attention: list[AttentionEntry],
    forward_graph: dict[str, list[str]]
) -> str:
    mermaid_diag = generate_mermaid_graph(forward_graph)

    lines = [
        "# Project Machine Analysis Report",
        "",
        f"**Root Directory:** `{root.resolve()}`",
        f"**Total Files Discovered:** {len(files_meta)}",
        "",
        "## Dependency Graph Map",
        "",
        mermaid_diag,
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
    lines.extend(render_attention_markdown(attention).splitlines())

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


def _markdown_code(value: object) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").replace("`", "\\`")
    return f"`{text}`"


def render_attention_markdown(attention: Sequence[Mapping[str, Any]]) -> str:
    """Render canonical attention entries without reclassifying or reordering."""

    grouped = {severity: [] for severity in SEVERITY_ORDER}
    for entry in attention:
        grouped[entry["severity"]].append(entry)
    lines: list[str] = []
    for severity in SEVERITY_ORDER:
        lines.extend([f"### {severity.title()}", ""])
        entries = grouped[severity]
        if not entries:
            lines.append("- (none)")
        else:
            for entry in entries:
                evidence = json.dumps(entry["evidence"], ensure_ascii=False, sort_keys=True)
                reason = str(entry["reason"]).replace("\r", " ").replace("\n", " ")
                lines.append(
                    f"- {_markdown_code(entry['path'])} [{entry['kind']}]: "
                    f"{reason} — {_markdown_code(evidence)}"
                )
        lines.append("")
    return "\n".join(lines).rstrip()


def analyze_machine_level(root_path: Path, output_dir: Path) -> dict[str, Any]:
    root = root_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir == root:
        raise ValueError("output_dir must be different from the scan root")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_is_within_root = _is_path_within(output_dir, root)

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
            if output_is_within_root and _is_path_within(p, output_dir):
                continue
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

    # シンボルとアウトライン情報を files_meta にマージ
    symbols_by_path = {sym["path"]: sym for sym in symbols_list}
    for meta in files_meta:
        rel_path = meta["path"]
        sym_info = symbols_by_path.get(rel_path, {})
        classes = sym_info.get("classes", [])
        functions = sym_info.get("functions", [])
        meta["classes"] = classes
        meta["functions"] = functions

        # 公開・内部シンボルの分類
        public_symbols = []
        internal_symbols = []
        for sym in classes + functions:
            if sym.startswith('_'):
                internal_symbols.append(sym)
            else:
                public_symbols.append(sym)
        meta["public_symbols"] = public_symbols
        meta["internal_symbols"] = internal_symbols

    # Gitステータスの取得とマッピング
    git_status_map = get_git_status_info(root)
    for meta in files_meta:
        rel_path = meta["path"]
        meta["git_status"] = git_status_map.get(rel_path, None)

    # repo_map サマリー作成
    repo_map = build_repo_map_summary(root, files_meta)

    # 依存関係グラフの構築とトポロジカルソート
    forward_graph, backward_graph = build_dependency_graph(files_meta, symbols_list)
    dependency_order = topological_sort(backward_graph)

    # ファンイン・ファンアウトメトリクスの算出
    for meta in files_meta:
        rel_path = meta["path"]
        meta["fan_in"] = len(backward_graph.get(rel_path, []))
        meta["fan_out"] = len(forward_graph.get(rel_path, []))

    coverage_targets = _build_coverage_targets(files_meta)

    # ドキュメントの存在有無・更新チェックとカバレッジの算出
    missing_docs = []
    stale_docs = []
    valid_docs = []

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
    coverage_summary = {
        "all_files_discovered": len(all_files),
        "ignored_files": ignored_count,
        "coverage_target_files": total_targets,
        "documented_files": documented_count,
        "missing_docs": len(missing_docs),
        "stale_docs": len(stale_docs),
        "coverage_percent": coverage_percent,
    }
    coverage_contract = {
        "include": [
            "kind == source",
            "language != unknown",
            "hash not in ('binary_skipped', 'error')",
        ],
        "exclude": [
            "kind in test/config/doc/other",
            "ignored by .gitignore",
        ],
        "counts": coverage_summary,
    }

    doc_status_by_path = {path: "missing" for path in missing_docs}
    doc_status_by_path.update({path: "stale" for path in stale_docs})
    doc_status_by_path.update({path: "current" for path in valid_docs})
    resolved_entrypoints, entrypoint_diagnostics = resolve_attention_entrypoints(
        root, files_meta
    )
    snapshots, attention_diagnostics = build_attention_snapshots(
        root,
        files_meta,
        previous_meta,
        doc_status_by_path=doc_status_by_path,
        entrypoint_paths=resolved_entrypoints,
    )
    attention_diagnostics = _sort_attention_diagnostics(
        [*attention_diagnostics, *entrypoint_diagnostics]
    )
    attention = classify_attention(snapshots)

    # 最終データの統合
    result = {
        "files": files_meta,
        "symbols": symbols_list,
        "repo_map": repo_map,
        "attention": attention,
        "attention_diagnostics": [
            {"detector": item.detector, "code": item.code, "path": item.path}
            for item in attention_diagnostics
        ],
        "dependency_graph": forward_graph,
        "dependency_order": dependency_order,
        "coverage_targets": [meta["path"] for meta in coverage_targets],
        "coverage_summary": coverage_summary,
        "coverage_contract": coverage_contract,
    }

    # 公開機械 index は内部解析結果の allowlist 投影として生成する。
    # 投影は履歴・mtime・Git 状態・coverage/attention を参照せず、契約検証後に
    # 決定的 serializer と atomic writer へ渡す。
    machine_index = build_machine_index_v2(result)
    validate_machine_index(machine_index, supported_major=2)

    # 機械向け JSON 書き出し
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # machine_index.json の書き出し
    index_json_path = output_dir / "machine_index.json"
    write_machine_index_atomic(index_json_path, machine_index, supported_major=2)

    # 機械向け YAML 書き出し (PyYAML非依存)
    yaml_path = output_dir / "machine_analysis.yaml"
    yaml_path.write_text(simple_yaml_dump(result), encoding="utf-8")

    # 人間向け Markdown 書き出し
    report_path = output_dir / "machine_report.md"
    report_content = generate_machine_report(root, files_meta, repo_map, attention, forward_graph)
    report_path.write_text(report_content, encoding="utf-8")

    # analysis_report.md の自動合成 (コントローラー互換フォーマット)
    percent_str = f"{coverage_percent:.1f}%"
    analysis_report = (
        f"# Project Analysis Report: {root.name}\n\n"
        f"**Root Directory:** `{root}`  \n"
        f"**Backend:** none (machine scan only)  \n"
        f"**Runtime:** controller  \n"
        f"**Status:** finished  \n"
        f"**Steps Used:** 0  \n"
        f"**Approx Tokens:** 0  \n"
        f"**Worker Budget:** 0 tokens, 0 LLM calls  \n"
        f"**Synthesis RLM Budget:** 0 tokens, 0 LLM calls  \n"
        f"**Global Budget:** 0 tokens, 0 LLM calls  \n\n"
        f"## Executive Summary\n\n"
        f"This report summarizes the static machine scan of the project. No LLM resources were utilized.\n\n"
        f"## Coverage Contract\n\n"
        f"- Included in coverage: source files with known language and non-binary hashes\n"
        f"- Excluded from coverage: tests, config, docs, other, ignored paths, binary files, unknown-language files\n"
        f"- All files discovered: {coverage_summary['all_files_discovered']}\n"
        f"- Coverage target files: {coverage_summary['coverage_target_files']}\n\n"
        f"## Source Coverage\n\n"
        f"- Source files discovered: {total_targets}\n"
        f"- Source files with matching docs: {documented_count}\n"
        f"- Source files missing matching docs: {len(missing_docs)}\n"
        f"- Extra docs without matching source: 0\n"
        f"- Weak or failed docs: 0\n"
        f"- Fallback docs generated: 0\n"
        f"- Coverage: {percent_str}\n\n"
        f"### Fallback Generated Source Docs\n\n"
        f"- (none)\n\n"
        f"### Missing Source Docs\n\n"
    )
    if missing_docs:
        analysis_report += "\n".join(f"- `{path}`" for path in sorted(missing_docs)) + "\n\n"
    else:
        analysis_report += "- (none)\n\n"
        
    analysis_report += (
        f"### Extra Docs Without Matching Source\n\n"
        f"- (none)\n\n"
        f"### Weak Or Failed Docs\n\n"
        f"- (none)\n\n"
        f"## Step History\n\n"
        f"| Step | Kind | Status | Summary |\n"
        f"| :--- | :--- | :--- | :--- |\n"
        f"| 1 | machine_scan | OK | Static machine scan completed successfully. |\n"
    )
    (output_dir / "analysis_report.md").write_text(analysis_report, encoding="utf-8")

    # index.md の自動合成
    changed_files = [m["path"] for m in files_meta if m["status"] in ("changed", "added") and m["kind"] == "source" and m["language"] != "unknown"]
    git_changed = [m["path"] for m in files_meta if m.get("git_status") is not None and m["kind"] == "source" and m["language"] != "unknown"]
    # 解説が必要なファイル（新規・変更されたファイル、解説欠損、陳腐化ドキュメント、Git変更検知されたファイル）
    needs_explanation = set(changed_files) | set(missing_docs) | set(stale_docs) | set(git_changed)
    
    # トポロジカルソート順（Bottom-up）で並べ替え
    sorted_needs_explanation = []
    for f in dependency_order:
        if f in needs_explanation:
            sorted_needs_explanation.append(f)
    # ソート順に含まれなかったものを決定論的に末尾に追加
    for f in sorted(list(needs_explanation)):
        if f not in sorted_needs_explanation:
            sorted_needs_explanation.append(f)
            
    index_content = (
        f"# Directory: {root.name}\n\n"
        f"Welcome to the static analysis index for `{root.name}`.\n\n"
        f"## Project Summary\n"
        f"- **Total Files Discovered:** {coverage_summary['all_files_discovered']}\n"
        f"- **Coverage Target Files:** {coverage_summary['coverage_target_files']}\n"
        f"- **Status:** Static Scan Complete\n\n"
        f"## Stale or Newly Added Files (Need Explanation)\n"
        f"These files are either new or modified, and their corresponding documentation (if any) may be stale.\n"
    )
    if sorted_needs_explanation:
        for cf in sorted_needs_explanation:
            reasons = []
            if cf in missing_docs:
                reasons.append("missing doc")
            elif cf in stale_docs:
                reasons.append("stale doc")
            
            m_status = next((m["status"] for m in files_meta if m["path"] == cf), None)
            if m_status in ("changed", "added"):
                reasons.append(f"source {m_status}")

            m_git_status = next((m.get("git_status") for m in files_meta if m["path"] == cf), None)
            if m_git_status:
                if m_git_status == "M":
                    reasons.append("Modified in Git")
                elif m_git_status in ("A", "??"):
                    reasons.append("Added in Git")
                
            reason_str = f" ({', '.join(reasons)})" if reasons else ""
            index_content += f"- [ ] `{cf}`{reason_str}\n"

            # アウトライン（クラス・関数）があれば折りたたみで表示
            file_meta_info = next((m for m in files_meta if m["path"] == cf), None)
            if file_meta_info:
                pub_syms = file_meta_info.get("public_symbols", [])
                int_syms = file_meta_info.get("internal_symbols", [])
                if pub_syms or int_syms:
                    index_content += "  <details>\n"
                    index_content += "    <summary>Outline (Symbols)</summary>\n"
                    if pub_syms:
                        index_content += "    <strong>Public API:</strong>\n"
                        index_content += "    <ul>\n"
                        for s in pub_syms:
                            index_content += f"      <li><code>{s}</code></li>\n"
                        index_content += "    </ul>\n"
                    if int_syms:
                        index_content += "    <strong>Internal Helpers:</strong>\n"
                        index_content += "    <ul>\n"
                        for s in int_syms:
                            index_content += f"      <li><code>{s}</code></li>\n"
                        index_content += "    </ul>\n"
                    index_content += "  </details>\n"
    else:
        index_content += "- No modified or added files detected. All files are unchanged.\n"
        
    index_content += "\n## High Priority Files to Inspect (Attention Points)\nThese files have warnings or high complexity:\n"
    if attention:
        index_content += render_attention_markdown(attention) + "\n"
    else:
        index_content += "- (none)\n"
        
    index_content += f"\n## Directory Structure Overview\nRefer to [machine_report.md](./machine_report.md) for the full inventory.\n"
    (output_dir / "index.md").write_text(index_content, encoding="utf-8")

    return result
