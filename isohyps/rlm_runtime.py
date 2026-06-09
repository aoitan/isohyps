from __future__ import annotations

import ast
import concurrent.futures
import io
import json
import math
import multiprocessing
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field, replace
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable, Protocol

from isohyps.analysis_helpers import detect_language, extract_symbols, is_probably_binary, read_text_excerpt

_DEBUG = bool(os.getenv("DEBUG"))


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        print(f"[DBG {tag}] {msg}", file=sys.stderr, flush=True)


class QueryClient(Protocol):
    def query(self, prompt: str) -> str:
        ...


SAFE_BUILTINS = {
    "AssertionError": AssertionError,
    "AttributeError": AttributeError,
    "Exception": Exception,
    "False": False,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "NameError": NameError,
    "None": None,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "True": True,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "dir": dir,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "getattr": getattr,
    "globals": globals,
    "hasattr": hasattr,
    "id": id,
    "int": int,
    "isinstance": isinstance,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "print": print,
    "range": range,
    "repr": repr,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "super": super,
    "tuple": tuple,
    "type": type,
    "vars": vars,
    "zip": zip,
}


def _approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _cap_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


_READ_SIZE_LIMIT = 2 * 1024 * 1024  # 2 MB


def _safe_repr(value: Any, limit: int, depth: int = 0, seen: set[int] | None = None) -> str:
    if seen is None:
        seen = set()

    value_id = id(value)
    if isinstance(value, (list, tuple, dict, set)):
        if value_id in seen:
            return "..."
        seen.add(value_id)

    if isinstance(value, str) and len(value) > limit:
        return f"str(len={len(value)}) {value[:limit - 3]}..."
    if depth >= 2:
        if isinstance(value, (list, tuple, set, dict)):
            return f"{type(value).__name__}(len={len(value)})"
        rendered = repr(value)
        return _cap_text(rendered, limit)
    if isinstance(value, list):
        if len(value) > 100:
            first = _safe_repr(value[0], 20, depth + 1, seen) if value else ""
            return f"list(len={len(value)}) [{first}, ...]"
        return "[" + ", ".join(_safe_repr(item, limit, depth + 1, seen) for item in value) + "]"
    if isinstance(value, tuple):
        rendered_items = ", ".join(_safe_repr(item, limit, depth + 1, seen) for item in value)
        if len(value) == 1:
            rendered_items += ","
        return f"({rendered_items})"
    if isinstance(value, set):
        if len(value) > 100:
            return f"set(len={len(value)}) {{...}}"
        return "{" + ", ".join(_safe_repr(item, limit, depth + 1, seen) for item in value) + "}"
    if isinstance(value, dict):
        if len(value) > 50:
            return f"dict(len={len(value)}) {{...}}"
        items = [
            f"{_safe_repr(key, limit, depth + 1, seen)}: {_safe_repr(item, limit, depth + 1, seen)}"
            for key, item in value.items()
        ]
        return "{" + ", ".join(items) + "}"

    return repr(value)


def _summarize_value(value: Any, limit: int) -> str:
    # Early-exit for large collections to avoid expensive repr() calls (OOM guard)
    if isinstance(value, list) and len(value) > 100:
        first = _safe_repr(value[0], 20) if value else ""
        return f"list(len={len(value)}) [{first}, ...]"
    if isinstance(value, dict) and len(value) > 50:
        return f"dict(len={len(value)}) {{...}}"
    if isinstance(value, str) and len(value) > limit:
        return f"str(len={len(value)}) {value[:limit - 3]}..."
    rendered = _safe_repr(value, limit)
    if len(rendered) > limit:
        rendered = rendered[: limit - 3] + "..."
    return f"{type(value).__name__} {rendered}"


def _sanitize_md_table_cell(text: str) -> str:
    """Sanitize a string for safe insertion into a Markdown table cell."""
    text = text.replace("|", "&#124;")
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    if text.endswith("\\"):
        text += "\\"
    return text


def _summarize_child_error(error: str | None, limit: int = 200) -> str:
    if not error:
        return "no detail provided"
    lines = [line.strip() for line in error.splitlines() if line.strip()]
    detail = lines[-1] if lines else error.strip()
    return _cap_text(detail, limit)


def _child_query_fallback_result(goal: str, child_context: Any, status: str, error: str | None) -> str:
    detail = _summarize_child_error(error)
    card = child_context.get("file") if isinstance(child_context, dict) else None
    if not isinstance(card, dict) and isinstance(child_context, dict):
        card = child_context

    if not isinstance(card, dict):
        return (
            "## Child Query Fallback\n\n"
            f"The child query could not complete ({status}: {detail}). "
            "No file card was provided, so only the parent-visible context is available.\n\n"
            "## Focus\n\n"
            f"{_cap_text(goal, 500)}"
        )

    info = card.get("info") if isinstance(card.get("info"), dict) else {}
    symbols = card.get("symbols") if isinstance(card.get("symbols"), dict) else {}
    path = card.get("path") or info.get("path") or "unknown"
    reasons = card.get("reasons")
    if not isinstance(reasons, list):
        reasons = []
    excerpt = card.get("excerpt")
    if not isinstance(excerpt, str):
        excerpt = symbols.get("fallback_excerpt") if isinstance(symbols.get("fallback_excerpt"), str) else ""
    symbol_items = symbols.get("symbols")

    return (
        "## Child Query Fallback\n\n"
        f"The deep-dive child query for `{path}` could not complete ({status}: {detail}). "
        "This document is a mechanical fallback built from the file card so the parent can continue.\n\n"
        "## Responsibility\n\n"
        f"File-card-only responsibility estimate for `{path}` based on language, size, symbols, and excerpt.\n\n"
        "## Main Elements\n\n"
        f"- Language: {info.get('language')}\n"
        f"- Lines: {info.get('line_count')}\n"
        f"- Selection reasons: {', '.join(str(reason) for reason in reasons) if reasons else 'not recorded'}\n"
        f"- Symbols: {_cap_text(str(symbol_items), 800)}\n\n"
        "## Inputs and Outputs\n\n"
        "Inputs and outputs were not model-expanded because the child query failed; infer them from the listed symbols and excerpt.\n\n"
        "## Dependencies and Caveats\n\n"
        "This fallback is less reliable than a completed deep dive and should be reviewed if the file is critical.\n\n"
        "## Excerpt\n\n"
        "```\n"
        f"{_cap_text(excerpt, 1200)}\n"
        "```"
    )


def _merge_recorded_documents(result: Any, documents: list[dict[str, str]]) -> Any:
    if not documents or not isinstance(result, dict):
        return result

    merged = dict(result)
    raw_documents = merged.get("documents")
    existing_documents = raw_documents if isinstance(raw_documents, list) else []
    documents_by_path: dict[str, dict[str, str]] = {}

    for document in existing_documents:
        if not isinstance(document, dict):
            continue
        path = document.get("path")
        title = document.get("title")
        content = document.get("content")
        if isinstance(path, str) and isinstance(title, str) and isinstance(content, str):
            documents_by_path[path] = {"path": path, "title": title, "content": content}

    for document in documents:
        path = document.get("path")
        title = document.get("title")
        content = document.get("content")
        if isinstance(path, str) and isinstance(title, str) and isinstance(content, str):
            documents_by_path[path] = {"path": path, "title": title, "content": content}

    merged["documents"] = list(documents_by_path.values())
    return merged


@dataclass
class BudgetLimits:
    max_steps: int = 30
    max_depth: int = 1
    max_total_tokens: int = 90000
    max_local_tokens: int = 90000
    step_timeout_seconds: float = 15.0
    llm_timeout_seconds: float = 120.0
    max_stdout_chars: int = 2000
    max_state_items: int = 20
    max_state_value_chars: int = 160


@dataclass
class PartialBudgetLimits:
    max_steps: int | None = None
    max_total_tokens: int | None = None
    max_local_tokens: int | None = None


@dataclass
class ChildQueryConfig:
    prompt_builder: PromptBuilder | None = None
    limits: PartialBudgetLimits | None = None


@dataclass
class BudgetSnapshot:
    steps_used: int
    llm_calls: int
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    global_llm_calls: int = 0
    global_prompt_tokens: int = 0
    global_response_tokens: int = 0
    global_total_tokens: int = 0


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class RunContext:
    limits: BudgetLimits
    steps_used: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    response_tokens: int = 0
    global_llm_calls: int = 0
    global_prompt_tokens: int = 0
    global_response_tokens: int = 0

    def reserve_step(self) -> None:
        if self.steps_used >= self.limits.max_steps:
            raise BudgetExceededError(f"max_steps={self.limits.max_steps} reached")
        self.steps_used += 1
        _dbg("budget", f"step reserved: {self.steps_used}/{self.limits.max_steps}")

    def ensure_depth(self, depth: int) -> None:
        if depth > self.limits.max_depth:
            raise BudgetExceededError(f"max_depth={self.limits.max_depth} reached")

    def ensure_prompt_budget(self, prompt: str) -> None:
        prompt_tokens = _approx_tokens(prompt)
        if self.total_tokens + prompt_tokens > self.limits.max_local_tokens:
            raise BudgetExceededError(
                f"max_local_tokens={self.limits.max_local_tokens} would be exceeded by prompt "
                f"(prompt_tokens={prompt_tokens}, current_local_tokens={self.total_tokens})"
            )
        if self.global_total_tokens + prompt_tokens > self.limits.max_total_tokens:
            raise BudgetExceededError(
                f"max_total_tokens={self.limits.max_total_tokens} would be exceeded by prompt "
                f"(prompt_tokens={prompt_tokens}, current_global_tokens={self.global_total_tokens})"
            )

    def record_query(self, prompt: str, response: str) -> None:
        prompt_tokens = _approx_tokens(prompt)
        response_tokens = _approx_tokens(response)
        
        self.llm_calls += 1
        self.prompt_tokens += prompt_tokens
        self.response_tokens += response_tokens
        
        self.global_llm_calls += 1
        self.global_prompt_tokens += prompt_tokens
        self.global_response_tokens += response_tokens
        
        _dbg("budget", f"llm_call #{self.llm_calls}: total_tokens={self.total_tokens}/{self.limits.max_local_tokens} (prompt_cumul={self.prompt_tokens} resp_cumul={self.response_tokens})")
        if self.total_tokens > self.limits.max_local_tokens:
            raise BudgetExceededError(f"max_local_tokens={self.limits.max_local_tokens} reached")
        if self.global_total_tokens > self.limits.max_total_tokens:
            raise BudgetExceededError(f"max_total_tokens={self.limits.max_total_tokens} reached")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.response_tokens

    @property
    def global_total_tokens(self) -> int:
        return self.global_prompt_tokens + self.global_response_tokens

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            steps_used=self.steps_used,
            llm_calls=self.llm_calls,
            prompt_tokens=self.prompt_tokens,
            response_tokens=self.response_tokens,
            total_tokens=self.total_tokens,
            global_llm_calls=self.global_llm_calls,
            global_prompt_tokens=self.global_prompt_tokens,
            global_response_tokens=self.global_response_tokens,
            global_total_tokens=self.global_total_tokens,
        )

    def accumulate(self, other: BudgetSnapshot) -> None:
        self.global_llm_calls = other.global_llm_calls
        self.global_prompt_tokens = other.global_prompt_tokens
        self.global_response_tokens = other.global_response_tokens
        if self.global_total_tokens > self.limits.max_total_tokens:
            raise BudgetExceededError(f"max_total_tokens={self.limits.max_total_tokens} reached")


@dataclass
class ExecutionObservation:
    kind: str
    stdout: str
    error: str | None
    state: dict[str, str]
    finished: bool
    result: Any
    partial_documents: list[dict[str, str]] = field(default_factory=list)

    def to_prompt(self) -> str:
        state_lines = "\n".join(f"- {name}: {value}" for name, value in sorted(self.state.items()))
        return (
            f"kind: {self.kind}\n"
            f"stdout:\n{self.stdout or '(empty)'}\n\n"
            f"error:\n{self.error or '(none)'}\n\n"
            f"state:\n{state_lines or '(empty)'}\n\n"
            f"finished: {self.finished}\n"
            f"result: {_summarize_value(self.result, 120)}"
        )

    @classmethod
    def issue(cls, kind: str, error: str) -> "ExecutionObservation":
        return cls(kind=kind, stdout="", error=error, state={}, finished=False, result=None)


@dataclass
class ControllerResult:
    status: str
    result: Any
    steps: list[ExecutionObservation]
    error: str | None
    budget: BudgetSnapshot
    final_state: dict[str, str]


@dataclass
class ValidatedCode:
    kind: str
    code: str | None
    error: str | None


def query_with_timeout(client: QueryClient, prompt: str, timeout_seconds: float) -> str:
    """Run a model query with a wall-clock timeout.

    The runtime previously had a timeout only for sandbox execution. When the
    backend HTTP call wedges after model generation, the controller could hang
    indefinitely. This wrapper bounds the wait and lets the controller recover.
    """
    if timeout_seconds <= 0:
        return client.query(prompt)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(client.query, prompt)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        raise TimeoutError(f"LLM query timed out after {timeout_seconds:.1f}s") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


class CodeResponseValidator:
    MODEL_ERROR_PREFIXES = ("[Gemini Error:", "[Ollama Error:", "[Error:")

    def _strip_blockquote_markers(self, text: str) -> str:
        lines = text.splitlines()
        non_empty = [line for line in lines if line.strip()]
        if not non_empty:
            return text
        if not all(line.lstrip().startswith(">") for line in non_empty):
            return text

        stripped_lines = []
        for line in lines:
            if not line.strip():
                stripped_lines.append("")
                continue
            stripped = line.lstrip()
            stripped_lines.append(stripped[1:].lstrip())
        return "\n".join(stripped_lines).strip()

    def _extract_fenced_code(self, text: str) -> str:
        lines = text.splitlines()
        start = None
        for index, line in enumerate(lines):
            if line.strip().startswith("```"):
                start = index
                break
        if start is None:
            return text

        end = None
        for index in range(start + 1, len(lines)):
            if lines[index].strip() == "```":
                end = index
                break
        if end is None:
            return text
        return "\n".join(lines[start + 1 : end]).strip()

    def _unwrap_string_literal_code(self, code: str) -> str:
        try:
            parsed = ast.parse(code)
        except SyntaxError:
            return code
        if (
            len(parsed.body) == 1
            and isinstance(parsed.body[0], ast.Expr)
            and isinstance(parsed.body[0].value, ast.Constant)
            and isinstance(parsed.body[0].value.value, str)
        ):
            candidate = parsed.body[0].value.value.strip()
            if candidate:
                try:
                    ast.parse(candidate)
                except SyntaxError:
                    return code
                return candidate
        return code

    def _reject_disallowed_syntax(self, parsed: ast.AST) -> str | None:
        for node in ast.walk(parsed):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return "Import statements are not allowed. Use sandbox helpers instead."
        return None

    def normalize(self, response: str) -> ValidatedCode:
        stripped = self._strip_blockquote_markers(response.strip())
        if not stripped:
            return ValidatedCode(kind="invalid_code", code=None, error="Model returned an empty response.")
        if stripped.startswith(self.MODEL_ERROR_PREFIXES):
            return ValidatedCode(kind="model_error", code=None, error=stripped)

        code = self._extract_fenced_code(stripped)
        code = self._unwrap_string_literal_code(code)

        if not code:
            return ValidatedCode(kind="invalid_code", code=None, error="No Python code remained after normalization.")

        try:
            parsed = ast.parse(code)
        except SyntaxError as exc:
            return ValidatedCode(kind="invalid_code", code=None, error=f"SyntaxError: {exc}")

        disallowed_error = self._reject_disallowed_syntax(parsed)
        if disallowed_error:
            return ValidatedCode(kind="invalid_code", code=None, error=disallowed_error)

        return ValidatedCode(kind="code", code=code, error=None)


class PromptBuilder:
    SYSTEM_PROMPT = (
        "You are operating a Recursive Language Model runtime.\n"
        "Return only Python code and no prose.\n"
        "State persists across steps inside a Python sandbox.\n"
        "Use helpers instead of imports or direct OS access.\n"
        "Available helpers:\n"
        "- list_dir(path='.') -> list[str]\n"
        "- read_text(path, offset=0, limit=2000) -> str\n"
        "- file_info(path) -> dict(path, exists, is_file, is_dir, size_bytes, line_count, char_count, approx_tokens, language, binary)\n"
        "- search_text(path, pattern, max_results=10, context_chars=160) -> list[dict(offset, line, match, excerpt)]\n"
        "- read_json(path) -> parsed json value\n"
        "- path_exists(path) -> bool\n"
        "- is_dir(path) -> bool\n"
        "- extract_symbols(path) -> dict(language, symbols, fallback_excerpt, error)\n"
        "- llm_query(prompt, context=None) -> child result value\n"
        "- record_document(path, title, content) -> persist one completed Markdown source document for partial output\n"
        "- finish(value) -> immediately end the run and return value to the caller.\n"
        "Rules:\n"
        "- Do not import modules.\n"
        "- Do not attempt network, subprocess, or filesystem mutation.\n"
        "- All helper paths are relative to the analysis root. Use '.' for the root; do not prefix paths with the root directory name.\n"
        "- Prefer helper calls over large string constants.\n"
        "- A global variable `repo_map` is available in your environment. It is a partial map (up to depth 2, capped at 500 nodes) and includes `repo_map['source_worklist']`, the source files that need explanations. Use it as a starting point to understand the project structure, but always use helpers (like list_dir) to confirm details or explore deeper paths.\n"
        "- When you are done, call finish(value).\n"
    )

    def build(self, goal: str, step: int, max_steps: int, previous: str, parent_context: Any | None) -> str:
        context_line = "(none)" if parent_context is None else _summarize_value(parent_context, 400)
        return (
            f"{self.SYSTEM_PROMPT}\n"
            f"Goal: {goal}\n"
            f"Current step: {step}/{max_steps}\n"
            f"Parent context: {context_line}\n\n"
            f"Previous observation:\n{previous}\n"
        )


class _FinishSignal(Exception):
    def __init__(self, value: Any):
        super().__init__("finish")
        self.value = value


class _CappedWriter(io.StringIO):
    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit
        self._size = 0
        self._truncated = False

    def write(self, s: str) -> int:
        if self._size >= self.limit:
            self._truncated = True
            return len(s)
        remaining = self.limit - self._size
        chunk = s[:remaining]
        self._size += len(chunk)
        if len(chunk) < len(s):
            self._truncated = True
        return super().write(chunk)

    def get_capped_value(self) -> str:
        value = self.getvalue().strip()
        if self._truncated:
            value = f"{value}\n...[stdout truncated]".strip()
        return value


def _generate_repo_map(root_path: Path, max_depth: int = 2, max_nodes: int = 500) -> dict[str, Any]:
    ignore_dirs = {
        ".git", "node_modules", "__pycache__", "dist", "build", "venv", ".venv",
        ".pytest_cache", ".kelpie", ".serena", "target", "out", ".idea", ".vscode",
        ".mypy_cache", ".tox", "coverage", ".eggs", "htmlcov",
    }
    ignore_files = {
        "bun.lock",
        "cargo.lock",
        "composer.lock",
        "gemfile.lock",
        "package-lock.json",
        "pnpm-lock.yaml",
        "poetry.lock",
        "uv.lock",
        "yarn.lock",
    }

    def get_category(path: Path) -> str:
        name = path.name.lower()
        if path.is_dir():
            if name in ["src", "lib", "app", "cmd", "pkg", "internal"]:
                return "code"
            if name in ["test", "tests", "spec", "specs"]:
                return "test"
            if name in ["doc", "docs"]:
                return "doc"
            if name in ["config", "conf", ".github", ".vscode"]:
                return "config"
            if name in ["scripts", "ci", "ops"]:
                return "ci"
        else:
            if name.endswith((".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".rb")):
                if "test" in name:
                    return "test"
                return "code"
            if name.endswith((".md", ".txt", ".rst", ".pdf")):
                return "doc"
            if name.endswith((".json", ".yaml", ".yml", ".toml", ".ini", ".conf", ".cfg")):
                return "config"
            if name in ["dockerfile", "makefile", "build.gradle"]:
                return "ci"
        return "unknown"

    nodes: list[dict[str, Any]] = []
    truncated = False

    def is_ignored_path(path: Path) -> bool:
        relative_parts = path.relative_to(root_path).parts
        if any(part in ignore_dirs for part in relative_parts):
            return True
        return path.is_file() and path.name.lower() in ignore_files

    def collect_source_worklist(max_files: int = 1000) -> tuple[list[str], bool]:
        source_paths: list[str] = []
        source_truncated = False
        try:
            for item in sorted(root_path.rglob("*")):
                if len(source_paths) >= max_files:
                    source_truncated = True
                    break
                if item.is_symlink() or not item.is_file():
                    continue
                if is_ignored_path(item):
                    continue
                if is_probably_binary(item) or detect_language(item) is None:
                    continue
                source_paths.append(item.relative_to(root_path).as_posix())
        except PermissionError:
            pass
        return source_paths, source_truncated

    def _traverse(current_path: Path, current_depth: int) -> None:
        nonlocal truncated
        if current_depth > max_depth or truncated:
            return

        try:
            for item in sorted(current_path.iterdir(), key=lambda p: p.name):
                if truncated:
                    return
                # Skip symlinks to avoid traversal anomalies and security issues
                if item.is_symlink():
                    continue
                # Skip non-regular files and non-directories (FIFOs, sockets, etc.)
                if not item.is_file() and not item.is_dir():
                    continue
                if is_ignored_path(item):
                    continue

                if len(nodes) >= max_nodes:
                    truncated = True
                    return

                rel_path = item.relative_to(root_path).as_posix()
                node_type = "dir" if item.is_dir() else "file"
                category = get_category(item)

                nodes.append({
                    "path": rel_path,
                    "node_type": node_type,
                    "category": category,
                })

                if item.is_dir() and current_depth < max_depth:
                    _traverse(item, current_depth + 1)
        except PermissionError:
            pass

    _traverse(root_path, 1)
    source_worklist, source_worklist_truncated = collect_source_worklist()

    return {
        "_meta": {
            "note": (
                f"This is a partial map up to depth {max_depth}, capped at {max_nodes} nodes. "
                "Use list_dir for deeper exploration."
            ),
        },
        "root": ".",
        "max_depth": max_depth,
        "truncated": truncated,
        "source_worklist": source_worklist,
        "source_worklist_truncated": source_worklist_truncated,
        "nodes": nodes,
    }


def _sandbox_worker(connection: Connection, root: str, limits: BudgetLimits) -> None:
    root_path = Path(root).resolve()
    helper_names = {
        "list_dir",
        "read_text",
        "file_info",
        "search_text",
        "extract_symbols",
        "llm_query",
        "finish",
        "path_exists",
        "is_dir",
        "read_json",
        "record_document",
    }
    
    try:
        repo_map = _generate_repo_map(root_path)
    except Exception as e:
        _dbg("sandbox", f"failed to generate repo map: {e}")
        repo_map = {"_meta": {"note": "repo_map unavailable"}, "root": ".", "max_depth": 0, "truncated": False, "nodes": []}

    globals_dict: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "__name__": "__main__",
        "repo_map": repo_map,
        "analysis_documents": [],
    }

    def resolve_path(path: str | Path) -> Path:
        raw_path = Path(path)
        path_str = str(raw_path)
        if path_str in {"", "."}:
            candidate = root_path
        else:
            relative_parts = raw_path.parts
            if relative_parts and relative_parts[0] == root_path.name:
                relative_parts = relative_parts[1:]
            candidate = (root_path / Path(*relative_parts)).resolve()
        if candidate != root_path and root_path not in candidate.parents:
            raise ValueError(f"path escapes root: {path}")
        return candidate

    def list_dir(path: str = ".") -> list[str]:
        _dbg("sandbox", f"list_dir({path!r})")
        target = resolve_path(path)
        return sorted(item.name for item in target.iterdir())

    def _file_size_guard(target: Path, helper_name: str) -> int:
        file_size = target.stat().st_size
        if file_size > _READ_SIZE_LIMIT:
            raise ValueError(
                f"{helper_name}: file too large ({file_size} bytes > {_READ_SIZE_LIMIT} bytes limit)"
            )
        return file_size

    def read_text(path: str, offset: int = 0, limit: int = 2000) -> str:
        _dbg("sandbox", f"read_text({path!r}, offset={offset}, limit={limit})")
        target = resolve_path(path)
        _file_size_guard(target, "read_text")
        if offset < 0:
            raise ValueError("read_text: offset must be >= 0")
        if limit < 0:
            raise ValueError("read_text: limit must be >= 0")
        capped_limit = min(limit, 2000)
        if is_probably_binary(target):
            return "（バイナリファイルのため解析をスキップしました）"
        content = target.read_text(encoding="utf-8", errors="ignore")
        return content[offset : offset + capped_limit]

    def file_info(path: str) -> dict[str, Any]:
        _dbg("sandbox", f"file_info({path!r})")
        target = resolve_path(path)
        if not target.exists():
            return {
                "path": str(target.relative_to(root_path)) if target != root_path else ".",
                "exists": False,
                "is_file": False,
                "is_dir": False,
                "size_bytes": None,
                "line_count": None,
                "char_count": None,
                "approx_tokens": None,
                "language": None,
                "binary": False,
            }
        stat = target.stat()
        is_file = target.is_file()
        line_count = None
        char_count = None
        if is_file and not is_probably_binary(target) and stat.st_size <= _READ_SIZE_LIMIT:
            content = target.read_text(encoding="utf-8", errors="ignore")
            line_count = content.count("\n") + (1 if content else 0)
            char_count = len(content)
        return {
            "path": str(target.relative_to(root_path)) if target != root_path else ".",
            "exists": target.exists(),
            "is_file": is_file,
            "is_dir": target.is_dir(),
            "size_bytes": stat.st_size,
            "line_count": line_count,
            "char_count": char_count,
            "approx_tokens": math.ceil((char_count if char_count is not None else stat.st_size) / 4),
            "language": detect_language(target) if is_file else None,
            "binary": is_probably_binary(target) if is_file else False,
        }

    def search_text(path: str, pattern: str, max_results: int = 10, context_chars: int = 160) -> list[dict[str, Any]]:
        _dbg("sandbox", f"search_text({path!r}, pattern={pattern!r}, max_results={max_results}, context_chars={context_chars})")
        target = resolve_path(path)
        _file_size_guard(target, "search_text")
        if not pattern:
            raise ValueError("search_text: pattern must not be empty")
        max_results = max(0, min(max_results, 50))
        context_chars = max(0, min(context_chars, 1000))
        if is_probably_binary(target):
            return []
        content = target.read_text(encoding="utf-8", errors="ignore")
        lower_content = content.lower()
        lower_pattern = pattern.lower()
        results: list[dict[str, Any]] = []
        start = 0
        while len(results) < max_results:
            index = lower_content.find(lower_pattern, start)
            if index < 0:
                break
            excerpt_start = max(0, index - context_chars)
            excerpt_end = min(len(content), index + len(pattern) + context_chars)
            line_number = content.count("\n", 0, index) + 1
            results.append(
                {
                    "offset": index,
                    "line": line_number,
                    "match": content[index : index + len(pattern)],
                    "excerpt": content[excerpt_start:excerpt_end],
                }
            )
            start = index + max(1, len(pattern))
        return results

    def extract_symbols_helper(path: str) -> dict[str, Any]:
        _dbg("sandbox", f"extract_symbols({path!r})")
        target = resolve_path(path)
        info = extract_symbols(target, lang=detect_language(target))
        info["path"] = str(target.relative_to(root_path))
        return info

    def llm_query(prompt: str, context: Any = None) -> Any:
        _dbg("sandbox", f"llm_query {len(prompt)} chars: {prompt[:80]!r}...")
        connection.send({"type": "llm_query", "prompt": prompt, "context": context})
        while True:
            message = connection.recv()
            if message["type"] == "llm_query_result":
                _dbg("sandbox", f"llm_query result: {str(message['value'])[:80]!r}")
                return message["value"]
            if message["type"] == "llm_query_error":
                raise RuntimeError(message["error"])

    def finish(value: Any) -> None:
        _dbg("sandbox", f"finish({type(value).__name__}): {repr(value)[:80]!r}")
        raise _FinishSignal(value)

    def record_document(path: str, title: str, content: str) -> dict[str, str]:
        if not isinstance(path, str) or not path.strip():
            raise ValueError("record_document: path must be a non-empty string")
        if not isinstance(title, str):
            raise ValueError("record_document: title must be a string")
        if not isinstance(content, str):
            raise ValueError("record_document: content must be a string")
        document = {"path": path, "title": title, "content": content}
        globals_dict["analysis_documents"].append(document)
        return {"path": path, "title": title, "content_chars": str(len(content))}

    def path_exists(path: str) -> bool:
        _dbg("sandbox", f"path_exists({path!r})")
        target = resolve_path(path)
        return target.exists()

    def is_dir(path: str) -> bool:
        _dbg("sandbox", f"is_dir({path!r})")
        target = resolve_path(path)
        return target.is_dir()

    def read_json(path: str) -> Any:
        _dbg("sandbox", f"read_json({path!r})")
        target = resolve_path(path)
        _file_size_guard(target, "read_json")
        return json.loads(target.read_text(encoding="utf-8"))

    globals_dict.update(
        {
            "list_dir": list_dir,
            "read_text": read_text,
            "file_info": file_info,
            "search_text": search_text,
            "extract_symbols": extract_symbols_helper,
            "llm_query": llm_query,
            "finish": finish,
            "path_exists": path_exists,
            "is_dir": is_dir,
            "read_json": read_json,
            "record_document": record_document,
        }
    )

    def partial_documents() -> list[dict[str, str]]:
        documents = globals_dict.get("analysis_documents")
        if not isinstance(documents, list):
            return []
        sanitized_documents: list[dict[str, str]] = []
        for document in documents:
            if not isinstance(document, dict):
                continue
            path = document.get("path")
            title = document.get("title")
            content = document.get("content")
            if isinstance(path, str) and isinstance(title, str) and isinstance(content, str):
                sanitized_documents.append({"path": path, "title": title, "content": content})
        return sanitized_documents

    def should_prune_transient_global(name: str, value: Any, pre_existing_names: set[str]) -> bool:
        if (
            name in pre_existing_names
            or name.startswith("_")
            or name == "__builtins__"
            or name in helper_names
            or name == "analysis_documents"
            or name == "file_cards"
            or callable(value)
        ):
            return False
        if isinstance(value, str):
            return len(value) > 4000
        if isinstance(value, (list, tuple, set)):
            if len(value) <= 25:
                return False
            return any(
                isinstance(item, (dict, list, tuple, set)) or (isinstance(item, str) and len(item) > 500)
                for item in value
            )
        if isinstance(value, dict):
            if len(value) > 25:
                return True
            return any(
                isinstance(item, (dict, list, tuple, set)) or (isinstance(item, str) and len(item) > 500)
                for item in value.values()
            )
        return False

    def prune_transient_globals(pre_existing_names: set[str]) -> None:
        for name in list(globals_dict):
            if should_prune_transient_global(name, globals_dict[name], pre_existing_names):
                del globals_dict[name]

    def snapshot_state() -> dict[str, str]:
        state: dict[str, str] = {}
        for name, value in globals_dict.items():
            if name.startswith("_") or name == "__builtins__" or name in helper_names or name == "analysis_documents":
                continue
            state[name] = _summarize_value(value, limits.max_state_value_chars)
            if len(state) >= limits.max_state_items:
                state["..."] = f"state truncated at {limits.max_state_items} items"
                break
        return state

    while True:
        try:
            command = connection.recv()
        except EOFError:
            break

        if command["type"] == "shutdown":
            break
        if command["type"] != "exec":
            connection.send({"type": "result", "kind": "execution_error", "stdout": "", "error": "Unknown command.", "state": snapshot_state(), "finished": False, "result": None})
            continue

        stream = _CappedWriter(limits.max_stdout_chars)
        error = None
        finished = False
        result = None
        kind = "ok"
        pre_existing_names = set(globals_dict)
        with redirect_stdout(stream):
            try:
                exec(command["code"], globals_dict, globals_dict)
            except _FinishSignal as finish_signal:
                finished = True
                result = finish_signal.value
                kind = "finished"
            except Exception:
                error = _cap_text(traceback.format_exc(), limits.max_stdout_chars)
                kind = "execution_error"
        prune_transient_globals(pre_existing_names)
        connection.send(
            {
                "type": "result",
                "kind": kind,
                "stdout": stream.get_capped_value(),
                "error": error,
                "state": snapshot_state(),
                "finished": finished,
                "result": result,
                "partial_documents": partial_documents(),
            }
        )


class IsolatedREPL:
    def __init__(self, root: Path, limits: BudgetLimits):
        self.root = root.resolve()
        self.limits = limits
        self._process: multiprocessing.Process | None = None
        self._connection: Connection | None = None

    def _ensure_worker(self) -> None:
        if self._process is not None and self._process.is_alive() and self._connection is not None:
            return
        if self._connection is not None:
            self._connection.close()
        parent_conn, child_conn = multiprocessing.Pipe()
        process = multiprocessing.Process(
            target=_sandbox_worker,
            args=(child_conn, str(self.root), self.limits),
            daemon=True,
        )
        process.start()
        child_conn.close()
        self._process = process
        self._connection = parent_conn
        _dbg("repl", f"sandbox worker started pid={process.pid} root={self.root}")

    def _restart_worker(self) -> None:
        _dbg("repl", "restarting sandbox worker")
        self.close()
        self._ensure_worker()

    def execute(
        self,
        code: str,
        llm_query_handler,
        finish_validator: Callable[[Any], list[str]] | None = None,
        finish_normalizer: Callable[[Any], Any] | None = None,
        finish_error_formatter: Callable[[list[str]], str] | None = None,
    ) -> ExecutionObservation:
        self._ensure_worker()
        assert self._connection is not None
        assert self._process is not None

        first_line = code.split("\n")[0][:60]
        _dbg("repl", f"execute ({len(code)} chars): {first_line!r}...")
        self._connection.send({"type": "exec", "code": code})
        deadline = time.monotonic() + self.limits.step_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _dbg("repl", f"sandbox step timed out after {self.limits.step_timeout_seconds:.1f}s — restarting worker")
                self._restart_worker()
                return ExecutionObservation.issue(
                    kind="execution_error",
                    error=f"Sandbox step timed out after {self.limits.step_timeout_seconds:.1f}s; sandbox state was reset.",
                )
            if not self._connection.poll(min(remaining, 0.1)):
                continue
            message = self._connection.recv()
            if message["type"] == "llm_query":
                _dbg("repl", f"sandbox requests llm_query: {message['prompt'][:80]!r}...")
                try:
                    value = llm_query_handler(message["prompt"], message.get("context"))
                    _dbg("repl", f"llm_query_handler returned: {str(value)[:80]!r}")
                    self._connection.send({"type": "llm_query_result", "value": value})
                except Exception as exc:
                    _dbg("repl", f"llm_query_handler error: {exc}")
                    self._connection.send({"type": "llm_query_error", "error": str(exc)})
                deadline = time.monotonic() + self.limits.step_timeout_seconds
                continue
            if message["type"] == "result":
                obs = ExecutionObservation(
                    kind=message["kind"],
                    stdout=message["stdout"],
                    error=message["error"],
                    state=message["state"],
                    finished=message["finished"],
                    result=message["result"],
                    partial_documents=message.get("partial_documents") or [],
                )
                _dbg("repl", f"result kind={obs.kind} finished={obs.finished} stdout={len(obs.stdout)}chars state_keys={list(obs.state.keys())[:5]}")
                if obs.error:
                    _dbg("repl", f"result error: {obs.error[:120]!r}")
                if obs.finished and finish_validator is not None:
                    normalized_result = finish_normalizer(obs.result) if finish_normalizer is not None else obs.result
                    normalized_result = _merge_recorded_documents(normalized_result, obs.partial_documents)
                    validation_errors = finish_validator(normalized_result)
                    if validation_errors:
                        if finish_error_formatter is not None:
                            validation_error = finish_error_formatter(validation_errors)
                        else:
                            validation_error = "Invalid finish result: " + " ".join(validation_errors)
                        _dbg("repl", f"finish validation failed: {validation_error[:120]!r}")
                        return ExecutionObservation(
                            kind="execution_error",
                            stdout=obs.stdout,
                            error=validation_error,
                            state=obs.state,
                            finished=False,
                            result=None,
                            partial_documents=obs.partial_documents,
                        )
                    obs.result = normalized_result
                elif obs.finished:
                    obs.result = _merge_recorded_documents(obs.result, obs.partial_documents)
                return obs

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.send({"type": "shutdown"})
            except Exception:
                pass
            self._connection.close()
            self._connection = None
        if self._process is not None:
            self._process.join(timeout=0.2)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=0.2)
            self._process = None

    def __enter__(self) -> "IsolatedREPL":
        self._ensure_worker()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class RLMController:
    def __init__(
        self,
        client: QueryClient,
        root: Path,
        run_context: RunContext,
        prompt_builder: PromptBuilder | None = None,
        validator: CodeResponseValidator | None = None,
        child_config: ChildQueryConfig | None = None,
    ):
        self.client = client
        self.root = root.resolve()
        self.run_context = run_context
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.validator = validator or CodeResponseValidator()
        self.child_config = child_config or ChildQueryConfig()

    def run(
        self,
        goal: str,
        depth: int = 0,
        parent_context: Any | None = None,
        finish_validator: Callable[[Any], list[str]] | None = None,
        finish_normalizer: Callable[[Any], Any] | None = None,
        finish_error_formatter: Callable[[list[str]], str] | None = None,
    ) -> ControllerResult:
        _dbg("ctrl", f"run start depth={depth} goal={goal[:80]!r}")
        steps: list[ExecutionObservation] = []
        previous = "No previous observation."
        final_state: dict[str, str] = {}

        try:
            self.run_context.ensure_depth(depth)
        except BudgetExceededError as exc:
            _dbg("ctrl", f"depth budget exceeded: {exc}")
            return ControllerResult(
                status="budget_exceeded",
                result=None,
                steps=steps,
                error=str(exc),
                budget=self.run_context.snapshot(),
                final_state=final_state,
            )

        with IsolatedREPL(self.root, self.run_context.limits) as repl:
            while True:
                try:
                    self.run_context.reserve_step()
                except BudgetExceededError as exc:
                    _dbg("ctrl", f"step budget exceeded: {exc}")
                    partial_documents = self._collect_partial_documents(steps)
                    return ControllerResult(
                        status="budget_exceeded",
                        result=self._partial_analysis_result(partial_documents),
                        steps=steps,
                        error=str(exc),
                        budget=self.run_context.snapshot(),
                        final_state=final_state,
                    )

                prompt = self.prompt_builder.build(
                    goal=goal,
                    step=self.run_context.steps_used,
                    max_steps=self.run_context.limits.max_steps,
                    previous=previous,
                    parent_context=parent_context,
                )
                _dbg("ctrl", f"step {self.run_context.steps_used}/{self.run_context.limits.max_steps}: sending prompt ({len(prompt)} chars)")

                try:
                    self.run_context.ensure_prompt_budget(prompt)
                    response = query_with_timeout(
                        self.client,
                        prompt,
                        timeout_seconds=self.run_context.limits.llm_timeout_seconds,
                    )
                    _dbg("ctrl", f"llm response ({len(response)} chars): {response[:100]!r}...")
                    self.run_context.record_query(prompt, response)
                except BudgetExceededError as exc:
                    _dbg("ctrl", f"token budget exceeded during query: {exc}")
                    observation = ExecutionObservation.issue("model_error", str(exc))
                    steps.append(observation)
                    final_state = observation.state or final_state
                    partial_documents = self._collect_partial_documents(steps)
                    return ControllerResult(
                        status="budget_exceeded",
                        result=self._partial_analysis_result(partial_documents),
                        steps=steps,
                        error=str(exc),
                        budget=self.run_context.snapshot(),
                        final_state=final_state,
                    )
                except Exception as exc:
                    _dbg("ctrl", f"llm query error: {type(exc).__name__}: {exc}")
                    observation = ExecutionObservation.issue("model_error", f"{type(exc).__name__}: {exc}")
                else:
                    validation = self.validator.normalize(response)
                    _dbg("ctrl", f"validation kind={validation.kind}" + (f" error={validation.error!r}" if validation.error else ""))
                    if validation.kind != "code":
                        observation = ExecutionObservation.issue(validation.kind, validation.error or validation.kind)
                    else:
                        observation = repl.execute(
                            validation.code,
                            llm_query_handler=lambda child_goal, child_context: self._run_subquery(
                                child_goal,
                                child_context,
                                depth=depth + 1,
                            ),
                            finish_validator=finish_validator,
                            finish_normalizer=finish_normalizer,
                            finish_error_formatter=finish_error_formatter,
                        )

                steps.append(observation)
                final_state = observation.state or final_state
                _dbg("ctrl", f"observation: kind={observation.kind} finished={observation.finished} steps_so_far={len(steps)}")
                if observation.finished:
                    _dbg("ctrl", f"finish at step {self.run_context.steps_used}: result={type(observation.result).__name__}")
                    return ControllerResult(
                        status="finished",
                        result=observation.result,
                        steps=steps,
                        error=None,
                        budget=self.run_context.snapshot(),
                        final_state=final_state,
                    )
                previous = observation.to_prompt()

    def _child_limits(self) -> BudgetLimits:
        child_limits = replace(self.run_context.limits)
        overrides = self.child_config.limits
        if overrides is None:
            return child_limits
        if overrides.max_steps is not None:
            child_limits.max_steps = overrides.max_steps
        if overrides.max_total_tokens is not None:
            child_limits.max_total_tokens = overrides.max_total_tokens
        if overrides.max_local_tokens is not None:
            child_limits.max_local_tokens = overrides.max_local_tokens
        return child_limits

    def _run_subquery(self, goal: str, child_context: Any, depth: int) -> Any:
        _dbg("ctrl", f"subquery depth={depth} goal={goal[:80]!r}")
        child_run_context = RunContext(
            limits=self._child_limits(),
            global_llm_calls=self.run_context.global_llm_calls,
            global_prompt_tokens=self.run_context.global_prompt_tokens,
            global_response_tokens=self.run_context.global_response_tokens,
        )
        child_controller = type(self)(
            client=self.client,
            root=self.root,
            run_context=child_run_context,
            prompt_builder=self.child_config.prompt_builder or self.prompt_builder,
            validator=self.validator,
            child_config=self.child_config,
        )
        try:
            child_result = child_controller.run(goal=goal, depth=depth, parent_context=child_context)
        except Exception as exc:
            return _child_query_fallback_result(
                goal,
                child_context,
                "execution_error",
                f"{type(exc).__name__}: {exc}",
            )
        finally:
            self.run_context.accumulate(child_run_context.snapshot())
        _dbg(
            "ctrl",
            f"subquery done status={child_result.status} steps_used={child_result.budget.steps_used} tokens={child_result.budget.total_tokens}",
        )
        if child_result.status != "finished":
            return _child_query_fallback_result(
                goal,
                child_context,
                child_result.status,
                child_result.error,
            )
        return child_result.result

    def _collect_partial_documents(self, steps: list[ExecutionObservation]) -> list[dict[str, str]]:
        documents_by_path: dict[str, dict[str, str]] = {}
        for step in steps:
            for document in step.partial_documents:
                path = document.get("path")
                title = document.get("title")
                content = document.get("content")
                if isinstance(path, str) and isinstance(title, str) and isinstance(content, str):
                    documents_by_path[path] = {"path": path, "title": title, "content": content}
        return list(documents_by_path.values())

    def _partial_analysis_result(self, documents: list[dict[str, str]]) -> dict[str, Any] | None:
        if not documents:
            return None
        return {
            "summary": "Partial project analysis assembled from documents recorded before the controller stopped.",
            "documents": documents,
        }
