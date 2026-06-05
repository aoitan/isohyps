from __future__ import annotations

from pathlib import Path
from typing import Any

SUPPORTED_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".kt": "kotlin",
    ".swift": "swift",
}

SYMBOL_QUERIES: dict[str, str] = {
    "python": """
        [(function_definition) @symbol
         (class_definition) @symbol]
    """,
    "javascript": """
        [(function_declaration) @symbol
         (class_declaration) @symbol
         (lexical_declaration
           (variable_declarator value: [(arrow_function) (function_expression)])) @symbol]
    """,
    "typescript": """
        [(function_declaration) @symbol
         (class_declaration) @symbol
         (interface_declaration) @symbol
         (type_alias_declaration) @symbol]
    """,
    "tsx": """
        [(function_declaration) @symbol
         (class_declaration) @symbol
         (interface_declaration) @symbol]
    """,
    "go": """
        [(function_declaration) @symbol
         (method_declaration) @symbol
         (type_declaration) @symbol]
    """,
    "rust": """
        [(function_item) @symbol
         (struct_item) @symbol
         (enum_item) @symbol
         (impl_item) @symbol
         (trait_item) @symbol]
    """,
    "java": """
        [(class_declaration) @symbol
         (interface_declaration) @symbol
         (method_declaration) @symbol
         (constructor_declaration) @symbol]
    """,
    "ruby": """
        [(method) @symbol
         (class) @symbol
         (module) @symbol]
    """,
    "c": """
        [(function_definition) @symbol
         (struct_specifier) @symbol]
    """,
    "cpp": """
        [(function_definition) @symbol
         (class_specifier) @symbol
         (struct_specifier) @symbol]
    """,
    "c_sharp": """
        [(class_declaration) @symbol
         (interface_declaration) @symbol
         (method_declaration) @symbol]
    """,
    "kotlin": """
        [(function_declaration) @symbol
         (class_declaration) @symbol
         (object_declaration) @symbol]
    """,
    "swift": """
        [(function_declaration) @symbol
         (class_declaration) @symbol
         (struct_declaration) @symbol
         (protocol_declaration) @symbol]
    """,
}

BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".exe", ".pyc", ".o", ".a", ".so"}


def detect_language(path: Path) -> str | None:
    return SUPPORTED_LANGUAGES.get(path.suffix.lower())


def is_probably_binary(path: Path) -> bool:
    return path.suffix.lower() in BINARY_SUFFIXES


def read_text_excerpt(path: Path, limit: int = 5000) -> str:
    if is_probably_binary(path):
        return "（バイナリファイルのため解析をスキップしました）"
    return path.read_text(encoding="utf-8", errors="ignore")[:limit]


def extract_symbols(
    path: Path,
    lang: str | None = None,
    max_symbols: int = 10,
    max_lines_per_symbol: int = 16,
    fallback_limit: int = 1000,
) -> dict[str, Any]:
    language_name = lang or detect_language(path)
    result: dict[str, Any] = {
        "path": str(path),
        "language": language_name,
        "symbols": [],
        "fallback_excerpt": "",
        "error": None,
    }

    if not path.exists() or not path.is_file():
        result["error"] = f"file not found: {path}"
        return result

    if is_probably_binary(path):
        result["fallback_excerpt"] = "（バイナリファイルのため解析をスキップしました）"
        return result

    result["fallback_excerpt"] = read_text_excerpt(path, limit=fallback_limit)

    if not language_name:
        return result

    query_str = SYMBOL_QUERIES.get(language_name, "")
    if not query_str:
        return result

    try:
        from tree_sitter_languages import get_language, get_parser

        code = path.read_text(encoding="utf-8", errors="ignore")
        parser = get_parser(language_name)
        tree = parser.parse(code.encode("utf-8"))
        language = get_language(language_name)
        query = language.query(query_str)
        raw = query.captures(tree.root_node)
        if isinstance(raw, dict):
            symbol_nodes = raw.get("symbol", [])
        else:
            symbol_nodes = [node for node, capture_name in raw if capture_name == "symbol"]

        seen: set[int] = set()
        lines = code.splitlines()
        snippets: list[str] = []
        for node in symbol_nodes[:max_symbols]:
            if node.start_byte in seen:
                continue
            seen.add(node.start_byte)
            start = node.start_point[0]
            end = min(node.end_point[0], start + max_lines_per_symbol - 1)
            snippets.append("\n".join(lines[start : end + 1]))
        result["symbols"] = snippets
    except Exception as exc:
        result["error"] = str(exc)

    return result
