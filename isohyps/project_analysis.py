from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypedDict

from isohyps.analysis_helpers import detect_language, extract_symbols, is_probably_binary
from isohyps.rlm_runtime import (
    BudgetLimits,
    ChildQueryConfig,
    ControllerResult,
    ExecutionObservation,
    PartialBudgetLimits,
    PromptBuilder,
    RLMController,
    RunContext,
    _sanitize_md_table_cell,
    _summarize_value,
)


PROJECT_ANALYSIS_CHILD_MAX_STEPS = 3


class AnalysisDocumentDict(TypedDict, total=False):
    path: str
    title: str
    content: str


class _AnalysisResultOptionalDict(TypedDict, total=False):
    documents: list[AnalysisDocumentDict]


class AnalysisResultDict(_AnalysisResultOptionalDict):
    summary: str


ANALYSIS_RESULT_SCHEMA_TEXT = (
    "  {\n"
    "    'summary': str,        # High-level overview of the findings\n"
    "    'documents': [         # Optional list; omit it when no detailed docs are needed\n"
    "      {\n"
    "        'path': str,       # Optional relative path (e.g., 'auth.md')\n"
    "        'title': str,      # Optional document title\n"
    "        'content': str     # Optional Markdown content\n"
    "      }, ...\n"
    "    ]\n"
    "  }\n"
)


DEFAULT_GOAL_TEMPLATE = (
    "Analyze the project rooted at '{root_name}'. "
    "Inspect the repository using helpers, use llm_query only for focused subproblems, "
    "consume the explicit source worklist in repo_map['source_worklist'], "
    "build mechanical file cards for every source file, select a small justified deep-dive set from those cards, "
    "use llm_query only for the selected deep-dive files, "
    "and finish with a dict containing a string 'summary'. "
    "Do not repeat recorded document contents in finish(); the runtime will attach them."
)


def validate_analysis_result(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return [f"Expected finish(value) to receive a dict, got {type(value).__name__}."]

    errors: list[str] = []
    if "summary" not in value:
        errors.append("Missing required key 'summary'.")
    elif not isinstance(value["summary"], str):
        errors.append(f"Expected 'summary' to be str, got {type(value['summary']).__name__}.")

    documents = value.get("documents")
    if documents is None:
        return errors
    if not isinstance(documents, list):
        errors.append(f"Expected 'documents' to be list, got {type(documents).__name__}.")
        return errors

    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            errors.append(f"Expected documents[{index}] to be dict, got {type(document).__name__}.")
            continue
        for field in ("path", "title", "content"):
            field_value = document.get(field)
            if field_value is not None and not isinstance(field_value, str):
                errors.append(
                    f"Expected documents[{index}]['{field}'] to be str, got {type(field_value).__name__}."
                )
    return errors


def _documented_analysis_paths(value: Any) -> set[str]:
    if not isinstance(value, dict):
        return set()
    documents = value.get("documents")
    if not isinstance(documents, list):
        return set()

    paths: set[str] = set()
    for document in documents:
        if not isinstance(document, dict):
            continue
        path = document.get("path")
        if not isinstance(path, str):
            continue
        normalized = PurePosixPath(path.replace("\\", "/")).as_posix().lstrip("./")
        if normalized and normalized != "index.md":
            paths.add(normalized)
    return paths


def _matching_source_doc_candidates(source_path: str) -> set[str]:
    source = PurePosixPath(source_path)
    return {
        source.as_posix(),
        source.with_suffix(".md").as_posix(),
        f"{source.as_posix()}.md",
    }


def validate_project_analysis_finish(value: Any, source_files: list[str] | None = None) -> list[str]:
    errors = validate_analysis_result(value)
    if errors:
        return errors

    summary = value["summary"].strip()
    lowered_summary = summary.lower()
    if len(summary) < 40:
        errors.append("Expected 'summary' to contain a substantive project analysis, got a very short summary.")
    for shallow_phrase in (
        "initial exploration",
        "exploration of the root directory",
        "exploration started",
        "root directory only",
    ):
        if shallow_phrase in lowered_summary:
            errors.append(
                "Summary appears to describe an initial/root-only exploration. Continue inspecting important directories before finish()."
            )
            break

    documents = value.get("documents")
    if not isinstance(documents, list) or not documents:
        errors.append("Expected 'documents' to contain at least one analysis document.")
        return errors

    has_substantive_document = False
    for document in documents:
        if not isinstance(document, dict):
            continue
        content = document.get("content")
        if isinstance(content, str) and len(content.strip()) >= 40:
            has_substantive_document = True
            break
    if not has_substantive_document:
        errors.append("Expected at least one document with substantive 'content'.")

    if source_files:
        documented_paths = _documented_analysis_paths(value)
        missing_sources = [
            source_path
            for source_path in source_files
            if not (_matching_source_doc_candidates(source_path) & documented_paths)
        ]
        if missing_sources:
            preview = ", ".join(missing_sources[:10])
            if len(missing_sources) > 10:
                preview += f", ... ({len(missing_sources)} total)"
            errors.append(
                "Expected 'documents' to include one Markdown document for every source file in "
                f"repo_map['source_worklist']; missing: {preview}."
            )

    return errors


def normalize_analysis_result(value: Any) -> Any:
    """Best-effort coercion for common model mistakes in finish(value)."""
    if not isinstance(value, dict):
        return value

    normalized = dict(value)
    summary = normalized.get("summary")
    if isinstance(summary, dict):
        nested_summary = summary.get("summary")
        if isinstance(nested_summary, str):
            normalized["summary"] = nested_summary
            if "documents" not in normalized and isinstance(summary.get("documents"), list):
                normalized["documents"] = summary["documents"]
    return normalized


def format_analysis_result_errors(errors: list[str]) -> str:
    detail = " ".join(errors)
    return (
        f"Invalid analysis result format: {detail} "
        "Call finish() with a dict containing a substantive string 'summary' and a non-empty "
        "'documents' list of dicts with string 'path', 'title', and 'content' fields."
    )


class ProjectAnalysisPromptBuilder(PromptBuilder):
    SYSTEM_PROMPT = (
        "You are operating a Recursive Language Model runtime for project analysis.\n"
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
        "- finish(value) -> immediately end the run. For top-level project analysis, the value MUST be a dict matching:\n"
        f"{ANALYSIS_RESULT_SCHEMA_TEXT}"
        "Rules:\n"
        "- Do not import modules.\n"
        "- Do not attempt network, subprocess, or filesystem mutation.\n"
        "- Do not assign to helper names such as list_dir, read_text, file_info, search_text, read_json, path_exists, is_dir, extract_symbols, llm_query, or finish.\n"
        "- All helper paths are relative to the analysis root. Use '.' for the root; do not prefix paths with the root directory name.\n"
        "- Prefer helper calls over large string constants.\n"
        "- A global variable `repo_map` is available in your environment. It is a partial map (up to depth 2, capped at 500 nodes) and includes `repo_map['source_worklist']`, the source files that need explanations. Use it as a starting point to understand the project structure, but always use helpers (like list_dir) to confirm details or explore deeper paths.\n"
        "Minimum exploration before finish:\n"
        "- Do not finish after only inspecting the root directory or README.\n"
        "- Inspect repo_map first, then confirm important areas with helpers.\n"
        "- Treat `repo_map['source_worklist']` as the explicit analysis worklist. Keep a pending list in sandbox state and reduce it as each source file is inspected.\n"
        "- Before spending child-query budget, create a mechanical file card for each pending source file with file_info(), extract_symbols(), and a short read_text() excerpt only when useful.\n"
        "- Select a small deep-dive set from those cards before calling llm_query. Prefer entrypoints, CLI files, application/controller/runtime modules, import hubs, large or symbol-rich source files, and files explicitly named or implied by the user goal.\n"
        "- Deprioritize tests, __init__.py, generated/cache/build files, snapshots, fixtures, and simple config/data files unless the user goal targets them directly.\n"
        "- Keep the child-query count controlled: call llm_query for one selected deep-dive file at a time with only that file's card, excerpt, symbols, and selection reasons. Do not call llm_query for files selected as machine-card-only.\n"
        "- For every source file, record exactly one source document. Deep-dive documents must include the selection reason. Machine-card-only documents must state why no deep dive was used and summarize the mechanical card.\n"
        "- Do not bundle multiple source files into one child query; large multi-file child prompts are likely to time out before finish().\n"
        "- For each inspected source file, add a document whose path matches either `<source>.md` or the source path with a `.md` suffix, so coverage does not rely on fallback generation.\n"
        "- As soon as one source document is complete, call record_document(path, title, content). Do not repeat recorded document contents in the final finish() value.\n"
        "- Do not call finish() while pending source files remain. End the step without finish() and continue from sandbox state on the next step.\n"
        "- Before finish, inspect at least two important non-root directories with list_dir(), file_info(), read_text(), extract_symbols(), or llm_query().\n"
        "- Prioritize code, test, document, and configuration areas when present.\n"
        "- Use llm_query for focused subproblems, such as summarizing a large directory or a cluster of files.\n"
        "- The final summary should describe observed components and responsibilities, not just say exploration started.\n"
        "- The final value MUST include a substantive summary. It may omit documents when every source document was already recorded with record_document().\n"
        "Valid controller-code example:\n"
        "if 'pending' not in globals():\n"
        "    pending = list(repo_map['source_worklist'])\n"
        "    file_cards = []\n"
        "    for path in pending:\n"
        "        info = file_info(path)\n"
        "        symbols = extract_symbols(path)\n"
        "        excerpt = read_text(path, 0, 1200) if info.get('size_bytes', 0) <= 6000 else ''\n"
        "        reasons = []\n"
        "        if path.endswith('/cli.py') or path.endswith('__main__.py') or path.endswith('application.py'):\n"
        "            reasons.append('entrypoint_or_application')\n"
        "        if info.get('line_count', 0) >= 120:\n"
        "            reasons.append('large_source')\n"
        "        if isinstance(symbols.get('symbols'), list) and len(symbols.get('symbols')) >= 4:\n"
        "            reasons.append('symbol_rich')\n"
        "        if path.startswith('tests/') or path.endswith('/__init__.py') or path == '__init__.py':\n"
        "            reasons.append('noise_or_low_priority')\n"
        "        score = len([reason for reason in reasons if reason != 'noise_or_low_priority'])\n"
        "        if 'noise_or_low_priority' in reasons:\n"
        "            score = 0\n"
        "        file_cards.append({'path': path, 'info': info, 'symbols': symbols, 'excerpt': excerpt, 'reasons': reasons, 'score': score})\n"
        "    deep_cards = sorted([card for card in file_cards if card['score'] > 0], key=lambda card: card['score'], reverse=True)[:5]\n"
        "    deep_paths = [card['path'] for card in deep_cards]\n"
        "    pending = list(deep_paths) + [card['path'] for card in file_cards if card['path'] not in deep_paths]\n"
        "target_path = pending[0] if pending else None\n"
        "target_card = next((card for card in file_cards if card['path'] == target_path), None) if target_path else None\n"
        "if target_card:\n"
        "    path = target_card['path']\n"
        "    if path in deep_paths:\n"
        "        target_doc = llm_query(\n"
        "            'Write one concrete Markdown explanation for the single source file in Parent context. Include: responsibility, main classes/functions, inputs/outputs, dependencies, and caveats. Call finish(markdown_text) in the first child step. Do not use generic placeholder text.',\n"
        "            {'file': target_card},\n"
        "        )\n"
        "        doc = '# Analysis of ' + path + '\\n\\n## Deep-dive decision\\n\\nSelected for: ' + ', '.join(target_card['reasons']) + '\\n\\n' + str(target_doc)\n"
        "    else:\n"
        "        doc = '# Analysis of ' + path + '\\n\\n## Deep-dive decision\\n\\nMachine-card only: lower priority for limited budget. Reasons: ' + ', '.join(target_card['reasons'] or ['not selected']) + '\\n\\n## File Card\\n\\n- Language: ' + str(target_card['info'].get('language')) + '\\n- Lines: ' + str(target_card['info'].get('line_count')) + '\\n- Symbols: ' + str(target_card['symbols'].get('symbols'))[:600] + '\\n\\n## Excerpt\\n\\n```\\n' + str(target_card['excerpt'])[:800] + '\\n```'\n"
        "    record_document(path + '.md', 'Analysis of ' + path, doc)\n"
        "    pending = pending[1:]\n"
        "if pending:\n"
        "    progress_note = 'Recorded one source document; continue next step with remaining pending files.'\n"
        "if not pending:\n"
        "    summary = '# Project Index\\n\\nAnalyzed all source files from repo_map source_worklist.\\n\\n## Deep-dive files\\n\\n' + '\\n'.join('- ' + path for path in deep_paths) + '\\n\\n## Coverage\\n\\nRecorded one Markdown document for every source file; low-priority files kept machine-card only.'\n"
        "    finish({'summary': summary})\n"
        "Do not write `import`, `from ... import ...`, or finish(None). Do not finish with a bare string.\n"
        "- When you are done, call finish(value).\n"
    )


class ProjectAnalysisChildPromptBuilder(PromptBuilder):
    SYSTEM_PROMPT = (
        "You are a child Recursive Language Model query for project analysis.\n"
        "Return only Python code and no prose.\n"
        "Your job is to answer the focused child goal using the provided Parent context.\n"
        "When the child goal asks for one Markdown explanation, produce a Python string and call finish(markdown_text).\n"
        "When the Parent context contains a `file` card, produce one Markdown explanation for that exact file and call finish(markdown_text).\n"
        "For a `file` card, do not call helper functions to gather more context; use only the path, source excerpt, symbols, and selection reasons already shown in Parent context, then finish in the first step.\n"
        "Prefer finishing in one step. Do not create documents, do not call record_document(), and do not return a dict for single-file Markdown.\n"
        "Available helpers:\n"
        "- list_dir(path='.') -> list[str]\n"
        "- read_text(path, offset=0, limit=2000) -> str\n"
        "- file_info(path) -> dict(path, exists, is_file, is_dir, size_bytes, line_count, char_count, approx_tokens, language, binary)\n"
        "- search_text(path, pattern, max_results=10, context_chars=160) -> list[dict(offset, line, match, excerpt)]\n"
        "- read_json(path) -> parsed json value\n"
        "- path_exists(path) -> bool\n"
        "- is_dir(path) -> bool\n"
        "- extract_symbols(path) -> dict(language, symbols, fallback_excerpt, error)\n"
        "- finish(value) -> immediately end the child run and return value to the parent.\n"
        "Rules:\n"
        "- Do not import modules.\n"
        "- Do not attempt network, subprocess, or filesystem mutation.\n"
        "- Do not call llm_query from child analysis unless the child goal explicitly requires deeper recursion.\n"
        "- All helper paths are relative to the analysis root.\n"
        "- If Parent context contains source text or symbols, base the Markdown on that context instead of re-reading large files.\n"
        "- For a `file` card, the only valid first action is building `markdown_text` and calling finish(markdown_text).\n"
        "Valid child-code example:\n"
        "markdown_text = '## Responsibility\\n\\nThis file is responsible for the behavior described in the child goal and Parent context.\\n\\n## Main Elements\\n\\n- Describe concrete functions/classes seen in context.\\n\\n## Inputs and Outputs\\n\\n- Describe inputs and outputs visible in context.\\n\\n## Dependencies and Caveats\\n\\n- Note imports, callers, or limits visible in context.'\n"
        "finish(markdown_text)\n"
    )


@dataclass
class AnalysisDocument:
    path: str
    title: str
    content: str


@dataclass
class StructuredAnalysis:
    summary: str
    documents: list[AnalysisDocument]


@dataclass(frozen=True)
class SourceCoverage:
    source_files: list[str]
    documented_files: list[str]
    missing_files: list[str]
    extra_document_paths: list[str]
    weak_document_paths: list[str]

    @property
    def total_count(self) -> int:
        return len(self.source_files)

    @property
    def documented_count(self) -> int:
        return len(self.documented_files)

    @property
    def missing_count(self) -> int:
        return len(self.missing_files)

    @property
    def extra_document_count(self) -> int:
        return len(self.extra_document_paths)

    @property
    def weak_document_count(self) -> int:
        return len(self.weak_document_paths)

    @property
    def percent(self) -> float:
        if not self.source_files:
            return 100.0
        return (self.documented_count / self.total_count) * 100


def _normalize_structured_analysis(root_name: str, result: Any) -> StructuredAnalysis:
    if isinstance(result, dict):
        summary = str(result.get("summary") or result.get("result") or "")
        raw_documents = result.get("documents") or []
    else:
        summary = str(result)
        raw_documents = []

    documents: list[AnalysisDocument] = []
    for item in raw_documents:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "index.md")
        title = str(item.get("title") or Path(path).stem or root_name)
        content = str(item.get("content") or "")
        documents.append(AnalysisDocument(path=path, title=title, content=content))

    if not any(doc.path == "index.md" for doc in documents):
        documents.insert(0, AnalysisDocument(path="index.md", title=f"Directory: {root_name}", content=summary))

    return StructuredAnalysis(summary=summary, documents=documents)


def _summarize_observation(obs: ExecutionObservation) -> str:
    if obs.error:
        lines = [line.strip() for line in obs.error.splitlines() if line.strip()]
        return lines[-1] if lines else obs.error
    if obs.stdout:
        return obs.stdout.strip().splitlines()[0]
    return f"Result: {_summarize_value(obs.result, 100)}"


class AnalysisDocBuilder:
    """Generate project-analysis documents from a controller result."""

    RESERVED_NAMES: frozenset[str] = frozenset({"analysis_report.md"})
    MAX_FILENAME_LENGTH = 255
    IGNORED_SOURCE_DIRS: frozenset[str] = frozenset(
        {
            ".git",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "venv",
            "node_modules",
            "dist",
            "build",
        }
    )

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self._written_paths: set[Path] = set()

    def build(
        self,
        root_path: Path,
        controller_result: ControllerResult,
        backend: str,
        model: str,
    ) -> StructuredAnalysis:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        report_path = self.output_dir / "analysis_report.md"
        self._written_paths.add(report_path)

        structured = _normalize_structured_analysis(root_path.name, controller_result.result)

        if controller_result.status != "finished":
            detail = controller_result.error or structured.summary or str(controller_result.result)
            guidance = []
            if controller_result.status == "budget_exceeded":
                guidance.append("Try increasing --max-total-tokens or --max-steps.")
            guidance.append("If the controller keeps failing, retry with --runtime legacy.")
            failure_summary = f"[{controller_result.status}] {detail} {' '.join(guidance)}".strip()
            partial_documents = [document for document in structured.documents if document.path != "index.md"]
            structured = StructuredAnalysis(
                summary=failure_summary,
                documents=[
                    AnalysisDocument(
                        path="index.md",
                        title=f"Analysis stopped: {controller_result.status}",
                        content=failure_summary,
                    ),
                    *partial_documents,
                ],
            )

        for document in structured.documents:
            sanitized = self._sanitize_path(document.path)
            target = self._avoid_collision(sanitized)
            self._written_paths.add(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            body = document.content
            if body and not body.lstrip().startswith("#"):
                body = f"# {document.title}\n\n{body}"
            target.write_text(body or f"# {document.title}\n", encoding="utf-8")
            document.path = str(target.relative_to(self.output_dir))

        coverage = self._build_source_coverage(root_path, structured.documents)
        fallback_generated_files = list(coverage.missing_files)
        if fallback_generated_files:
            self._write_fallback_source_docs(root_path, structured.documents, fallback_generated_files)
            coverage = self._build_source_coverage(root_path, structured.documents)
        report_path.write_text(
            "\n".join(
                [
                    f"# Project Analysis Report: {root_path.name}",
                    "",
                    f"**Root Directory:** `{root_path}`  ",
                    f"**Backend:** {backend} ({model})  ",
                    f"**Runtime:** controller  ",
                    f"**Status:** {controller_result.status}  ",
                    f"**Steps Used:** {controller_result.budget.steps_used}  ",
                    f"**Approx Tokens:** {controller_result.budget.total_tokens}  ",
                    "",
                    "## Executive Summary",
                    "",
                    structured.summary or str(controller_result.result),
                    "",
                    *self._render_coverage_section(coverage, fallback_generated_files),
                    "",
                    "## Final State",
                    "",
                    "\n".join(f"- {name}: {value}" for name, value in sorted(controller_result.final_state.items())) or "- (empty)",
                    "",
                    "## Step History",
                    "",
                    "| Step | Kind | Status | Summary |",
                    "| :--- | :--- | :--- | :--- |",
                    *[
                        f"| {i+1} | {step.kind} | {'OK' if not step.error else 'ERR'} | {_sanitize_md_table_cell(_summarize_observation(step)[:100])} |"
                        for i, step in enumerate(controller_result.steps)
                    ],
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return structured

    def _write_fallback_source_docs(
        self,
        root_path: Path,
        documents: list[AnalysisDocument],
        missing_files: list[str],
    ) -> None:
        for source_path in missing_files:
            source_doc = AnalysisDocument(
                path=f"{source_path}.md",
                title=f"Source: {source_path}",
                content=self._render_fallback_source_doc(root_path, source_path),
            )
            target = self._avoid_collision(self._sanitize_path(source_doc.path))
            self._written_paths.add(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source_doc.content, encoding="utf-8")
            source_doc.path = str(target.relative_to(self.output_dir))
            documents.append(source_doc)

    def _render_fallback_source_doc(self, root_path: Path, source_path: str) -> str:
        absolute_path = root_path / source_path
        language = detect_language(absolute_path) or "unknown"
        try:
            line_count = len(absolute_path.read_text(encoding="utf-8", errors="ignore").splitlines())
        except OSError:
            line_count = 0

        symbol_info = extract_symbols(absolute_path, lang=language, max_symbols=8, fallback_limit=1200)
        symbols = symbol_info.get("symbols") if isinstance(symbol_info, dict) else []
        fallback_excerpt = symbol_info.get("fallback_excerpt") if isinstance(symbol_info, dict) else ""
        symbol_lines = []
        if isinstance(symbols, list):
            for symbol in symbols:
                if isinstance(symbol, str) and symbol.strip():
                    symbol_lines.append(f"- `{symbol.strip().splitlines()[0]}`")

        lines = [
            f"# Source: {source_path}",
            "",
            "## File Summary",
            "",
            f"- Path: `{source_path}`",
            f"- Language: `{language}`",
            f"- Lines: {line_count}",
            "",
            "This fallback document was generated because the controller did not return a document mapped to this source file.",
            "",
            "## Detected Symbols",
            "",
        ]
        lines.extend(symbol_lines or ["- (none detected)"])
        lines.extend(["", "## Excerpt", "", "```"])
        if isinstance(fallback_excerpt, str):
            lines.append(fallback_excerpt[:1200])
        lines.extend(["```", ""])
        return "\n".join(lines)

    def _build_source_coverage(self, root_path: Path, documents: list[AnalysisDocument]) -> SourceCoverage:
        source_files = self._collect_source_files(root_path)
        documented_paths = self._documented_paths(documents)
        documented_files = [source for source in source_files if self._source_has_matching_doc(source, documented_paths)]
        documented_set = set(documented_files)
        missing_files = [source for source in source_files if source not in documented_set]
        matched_document_paths = {
            document_path
            for source in source_files
            for document_path in self._matching_doc_candidates(source)
            if document_path in documented_paths
        }
        extra_document_paths = sorted(documented_paths - matched_document_paths)
        weak_document_paths = sorted(
            self._document_path(document)
            for document in documents
            if self._document_path(document) in documented_paths and self._is_weak_document(document)
        )
        return SourceCoverage(
            source_files=source_files,
            documented_files=documented_files,
            missing_files=missing_files,
            extra_document_paths=extra_document_paths,
            weak_document_paths=weak_document_paths,
        )

    def _collect_source_files(self, root_path: Path) -> list[str]:
        root = root_path.resolve()
        output_dir = self.output_dir.resolve()
        source_files: list[str] = []

        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            if self._is_ignored_source_path(path, root, output_dir):
                continue
            if is_probably_binary(path) or detect_language(path) is None:
                continue
            source_files.append(path.relative_to(root).as_posix())

        return source_files

    def _is_ignored_source_path(self, path: Path, root: Path, output_dir: Path) -> bool:
        try:
            path.relative_to(output_dir)
            return True
        except ValueError:
            pass

        relative_parts = path.relative_to(root).parts
        return any(part in self.IGNORED_SOURCE_DIRS for part in relative_parts)

    def _documented_paths(self, documents: list[AnalysisDocument]) -> set[str]:
        paths: set[str] = set()
        for document in documents:
            doc_path = PurePosixPath(str(document.path).replace("\\", "/"))
            normalized = doc_path.as_posix().lstrip("./")
            if not normalized or normalized == "index.md":
                continue
            paths.add(normalized)
        return paths

    def _source_has_matching_doc(self, source_path: str, documented_paths: set[str]) -> bool:
        return bool(self._matching_doc_candidates(source_path) & documented_paths)

    def _matching_doc_candidates(self, source_path: str) -> set[str]:
        source = PurePosixPath(source_path)
        return {
            source.as_posix(),
            source.with_suffix(".md").as_posix(),
            f"{source.as_posix()}.md",
        }

    def _document_path(self, document: AnalysisDocument) -> str:
        return PurePosixPath(str(document.path).replace("\\", "/")).as_posix().lstrip("./")

    def _is_weak_document(self, document: AnalysisDocument) -> bool:
        content = document.content.strip()
        if len(content) < 40:
            return True
        lowered = content.lower()
        failure_markers = (
            "analysis failed",
            "failed to analyze",
            "model_error",
            "budget_exceeded",
            "invalid analysis result",
        )
        return any(marker in lowered for marker in failure_markers)

    def _render_coverage_section(self, coverage: SourceCoverage, fallback_generated_files: list[str] | None = None) -> list[str]:
        fallback_generated_files = fallback_generated_files or []
        percent = f"{coverage.percent:.1f}%"
        lines = [
            "## Source Coverage",
            "",
            f"- Source files discovered: {coverage.total_count}",
            f"- Source files with matching docs: {coverage.documented_count}",
            f"- Source files missing matching docs: {coverage.missing_count}",
            f"- Extra docs without matching source: {coverage.extra_document_count}",
            f"- Weak or failed docs: {coverage.weak_document_count}",
            f"- Fallback docs generated: {len(fallback_generated_files)}",
            f"- Coverage: {percent}",
            "",
        ]

        if fallback_generated_files:
            lines.extend(
                [
                    "### Fallback Generated Source Docs",
                    "",
                    *[f"- `{path}`" for path in fallback_generated_files],
                    "",
                ]
            )
        else:
            lines.extend(["### Fallback Generated Source Docs", "", "- (none)", ""])

        if coverage.missing_files:
            lines.extend(
                [
                    "### Missing Source Docs",
                    "",
                    *[f"- `{path}`" for path in coverage.missing_files],
                    "",
                ]
            )
        else:
            lines.extend(["### Missing Source Docs", "", "- (none)", ""])

        if coverage.extra_document_paths:
            lines.extend(
                [
                    "### Extra Docs Without Matching Source",
                    "",
                    *[f"- `{path}`" for path in coverage.extra_document_paths],
                    "",
                ]
            )
        else:
            lines.extend(["### Extra Docs Without Matching Source", "", "- (none)", ""])

        if coverage.weak_document_paths:
            lines.extend(
                [
                    "### Weak Or Failed Docs",
                    "",
                    *[f"- `{path}`" for path in coverage.weak_document_paths],
                    "",
                ]
            )
        else:
            lines.extend(["### Weak Or Failed Docs", "", "- (none)", ""])

        return lines

    def _sanitize_path(self, raw_path: str) -> Path:
        normalized = str(raw_path).replace("\\", "/")
        parts = PurePosixPath(normalized).parts
        safe_parts: list[str] = []
        for part in parts:
            if part.startswith("/") or part == "..":
                continue
            if part == ".":
                continue
            if part.endswith(":"):
                continue
            if len(part) >= 2 and part[1] == ":":
                part = part[2:]
                if not part:
                    continue
            if len(part) > self.MAX_FILENAME_LENGTH:
                part = part[: self.MAX_FILENAME_LENGTH]
            safe_parts.append(part)

        if not safe_parts:
            return self.output_dir / "index.md"

        return self._ensure_markdown_path(self.output_dir / Path(*safe_parts))

    def _ensure_markdown_path(self, target_path: Path) -> Path:
        if target_path.suffix == ".md":
            return target_path
        return target_path.with_name(f"{target_path.name}.md")

    def _avoid_collision(self, target_path: Path) -> Path:
        stem = target_path.stem
        ext = target_path.suffix
        parent = target_path.parent

        if len(target_path.name) > self.MAX_FILENAME_LENGTH:
            max_stem_len = self.MAX_FILENAME_LENGTH - len(ext)
            target_path = parent / f"{stem[: max(1, max_stem_len)]}{ext}"

        if target_path not in self._written_paths:
            return target_path

        i = 1
        while True:
            suffix_str = f"_{i}"
            max_stem_len = self.MAX_FILENAME_LENGTH - len(ext) - len(suffix_str)
            truncated_stem = stem[: max(1, max_stem_len)]
            candidate = parent / f"{truncated_stem}{suffix_str}{ext}"
            if candidate not in self._written_paths:
                return candidate
            i += 1


def write_analysis_docs(
    output_dir: Path,
    root_path: Path,
    controller_result: ControllerResult,
    backend: str,
    model: str,
) -> StructuredAnalysis:
    return AnalysisDocBuilder(output_dir).build(root_path, controller_result, backend, model)


class RLMRuntimeAnalyzer:
    def __init__(
        self,
        client,
        max_depth: int = 2,
        max_steps: int = 30,
        output_dir: Path | None = None,
        step_timeout_seconds: float = 15.0,
        llm_timeout_seconds: float = 120.0,
        max_total_tokens: int = 90000,
        backend_name: str = "unknown",
        model_name: str = "unknown",
    ):
        self.client = client
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.output_dir = output_dir
        self.step_timeout_seconds = step_timeout_seconds
        self.llm_timeout_seconds = llm_timeout_seconds
        self.max_total_tokens = max_total_tokens
        self.backend_name = backend_name
        self.model_name = model_name
        self.last_result = None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, path: Path) -> str:
        root = path.resolve()
        limits = BudgetLimits(
            max_steps=self.max_steps,
            max_depth=self.max_depth,
            max_total_tokens=self.max_total_tokens,
            step_timeout_seconds=self.step_timeout_seconds,
            llm_timeout_seconds=self.llm_timeout_seconds,
        )
        controller = RLMController(
            client=self.client,
            root=root,
            run_context=RunContext(limits=limits),
            prompt_builder=ProjectAnalysisPromptBuilder(),
            child_config=ChildQueryConfig(
                prompt_builder=ProjectAnalysisChildPromptBuilder(),
                limits=PartialBudgetLimits(max_steps=PROJECT_ANALYSIS_CHILD_MAX_STEPS),
            ),
        )
        source_files = AnalysisDocBuilder(self.output_dir or (root / ".isohyps-output-validation"))._collect_source_files(root)

        def validate_finish(value: Any) -> list[str]:
            return validate_project_analysis_finish(value, source_files=source_files)

        goal = DEFAULT_GOAL_TEMPLATE.format(root_name=root.name)
        result = controller.run(
            goal=goal,
            finish_validator=validate_finish,
            finish_normalizer=normalize_analysis_result,
            finish_error_formatter=format_analysis_result_errors,
        )
        self.last_result = result

        summary = result.error or str(result.result)
        if result.status == "finished" and isinstance(result.result, dict):
            summary = str(result.result.get("summary") or summary)
        elif result.status != "finished":
            guidance = []
            if result.status == "budget_exceeded":
                guidance.append("Try increasing --max-total-tokens or --max-steps.")
            guidance.append("If the controller keeps failing, retry with --runtime legacy.")
            summary = f"[{result.status}] {summary} {' '.join(guidance)}".strip()

        if self.output_dir:
            write_analysis_docs(
                output_dir=self.output_dir,
                root_path=root,
                controller_result=result,
                backend=self.backend_name,
                model=self.model_name,
            )
        return summary
