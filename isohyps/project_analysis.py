from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypedDict

from isohyps.rlm_runtime import (
    BudgetLimits,
    ControllerResult,
    ExecutionObservation,
    PromptBuilder,
    RLMController,
    RunContext,
    _sanitize_md_table_cell,
    _summarize_value,
)


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
    "and finish with a dict containing a string 'summary'. "
    "Optionally include a 'documents' list; document path, title, and content fields are optional strings."
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
        "Call finish() with a dict containing at least a string 'summary' and, if provided, "
        "a 'documents' list of dicts with string 'path', 'title', and 'content' fields."
    )


class ProjectAnalysisPromptBuilder(PromptBuilder):
    SYSTEM_PROMPT = (
        "You are operating a Recursive Language Model runtime for project analysis.\n"
        "Return only Python code and no prose.\n"
        "State persists across steps inside a Python sandbox.\n"
        "Use helpers instead of imports or direct OS access.\n"
        "Available helpers:\n"
        "- list_dir(path='.') -> list[str]\n"
        "- read_text(path, limit=2000) -> str\n"
        "- read_json(path) -> parsed json value\n"
        "- path_exists(path) -> bool\n"
        "- is_dir(path) -> bool\n"
        "- extract_symbols(path) -> dict(language, symbols, fallback_excerpt, error)\n"
        "- llm_query(prompt, context=None) -> child result value\n"
        "- finish(value) -> immediately end the run. For top-level project analysis, the value MUST be a dict matching:\n"
        f"{ANALYSIS_RESULT_SCHEMA_TEXT}"
        "Rules:\n"
        "- Do not import modules.\n"
        "- Do not attempt network, subprocess, or filesystem mutation.\n"
        "- All helper paths are relative to the analysis root. Use '.' for the root; do not prefix paths with the root directory name.\n"
        "- Prefer helper calls over large string constants.\n"
        "- A global variable `repo_map` is available in your environment. It is a partial map (up to depth 2, capped at 500 nodes). Use it as a starting point to understand the project structure, but always use helpers (like list_dir) to confirm details or explore deeper paths.\n"
        "- When you are done, call finish(value).\n"
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
        return obs.error.splitlines()[0]
    if obs.stdout:
        return obs.stdout.strip().splitlines()[0]
    return f"Result: {_summarize_value(obs.result, 100)}"


class AnalysisDocBuilder:
    """Generate project-analysis documents from a controller result."""

    RESERVED_NAMES: frozenset[str] = frozenset({"analysis_report.md"})
    MAX_FILENAME_LENGTH = 255

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
            structured = StructuredAnalysis(
                summary=failure_summary,
                documents=[
                    AnalysisDocument(
                        path="index.md",
                        title=f"Analysis stopped: {controller_result.status}",
                        content=failure_summary,
                    )
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

        return self.output_dir / Path(*safe_parts)

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
        max_steps: int = 8,
        output_dir: Path | None = None,
        step_timeout_seconds: float = 15.0,
        llm_timeout_seconds: float = 120.0,
        max_total_tokens: int = 30000,
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
        )
        goal = DEFAULT_GOAL_TEMPLATE.format(root_name=root.name)
        result = controller.run(
            goal=goal,
            finish_validator=validate_analysis_result,
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
