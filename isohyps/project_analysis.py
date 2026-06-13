from __future__ import annotations

import json
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
    _summarize_parent_context,
    _summarize_value,
)


PROJECT_ANALYSIS_CHILD_MAX_STEPS = 5


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

REQUIRED_PROJECT_SUMMARY_SECTIONS = (
    "## Major directories",
    "## Important files",
    "## Relationships",
    "## Uncertainties",
)

DEFAULT_GOAL_TEMPLATE_PHASE1 = (
    "Phase 1: Survey the project rooted at '{root_name}' and draft an initial high-level architecture overview. "
    "Examine repo_map and README to identify uncertainties. "
    "Write a concise architecture description in Japanese. "
    "Finish with a dict containing a string 'summary' under the key 'summary'. "
    "Do not explain relationships or detailed file responsibilities yet."
)

DEFAULT_GOAL_TEMPLATE_PHASE2 = (
    "Phase 2: Analyze relationships and dependency graph of components in the repository rooted at '{root_name}'. "
    "Use this initial architecture overview: {phase1_summary}. "
    "Examine relationships.json and import hubs to map component interactions. "
    "In the 'summary' output, you MUST include a Mermaid diagram (```mermaid) showing component interactions, "
    "followed by detailed text explanations in Japanese. "
    "Build the Phase 2 summary with this Python shape: "
    "summary_lines = ['## Relationships', '', '```mermaid', 'graph TD', '  CLI --> Runtime', '  Runtime --> Workers', '```', '', '- Explain the real observed relationships.']; "
    "summary = '\\n'.join(summary_lines); finish({{'summary': summary}}). "
    "Replace the example nodes with observed components. Do not use triple-quoted strings or f-strings for this summary. "
    "Finish with a dict containing a string 'summary' containing the Mermaid diagram and explanations."
)

DEFAULT_GOAL_TEMPLATE_PHASE3 = (
    "Phase 3: Synthesize the final project analysis report for the repository rooted at '{root_name}'. "
    "Combine Architecture Overview: {phase1_summary} "
    "and Relationships with Mermaid diagram: {phase2_summary}. "
    "Before synthesis, inspect the artifact environment with list_artifacts('.'), read_artifact_json('manifest.json'), "
    "read_artifact_json('relationships.json'), and selected read_artifact('source-docs/...') snippets for important files. "
    "Use those artifact observations as the source of file-level information; do not rely only on Phase 1 or Phase 2 text. "
    "Integrate uncertainties and file-level artifact information to build the final report. "
    "The final summary MUST include exactly these Markdown sections: "
    "## Major directories, ## Important files, ## Relationships, and ## Uncertainties. "
    "Write the text in Japanese. "
    "Finish with a dict containing a string 'summary' matching the final output format."
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
    summary_lines = {line.strip() for line in summary.splitlines()}
    missing_sections = [section for section in REQUIRED_PROJECT_SUMMARY_SECTIONS if section not in summary_lines]
    if missing_sections:
        errors.append(
            "Expected 'summary' to include required Markdown sections: " + ", ".join(missing_sections) + "."
        )
    if "## Relationships" in summary_lines and "```mermaid" not in lowered_summary:
        errors.append("Expected 'summary' Relationships section to include a Mermaid diagram (```mermaid).")
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
    for failed_report_phrase in (
        "## child query fallback",
        "[directory name]",
        "[file path]",
        "[role]",
        "[description]",
        "[uncertainty 1]",
        "graph td\n  a --> b",
        "budget_exceeded",
        "execution_error",
        "invalid_code",
        "model_error",
        "this section was not separated in the model output",
        "予算制限により中断",
        "予算制限により中断された",
        "フォールバック情報",
        "具体的なプロジェクトの構造に関する情報は含まれていません",
        "コンテキスト内にディレクトリの情報は見当たりません",
        "解析失敗のため詳細不明",
        "明示的な依存関係のグラフは生成されませんでした",
        "定義された関係性は存在しません",
        "解析が中断されたため",
    ):
        if failed_report_phrase in lowered_summary:
            errors.append(
                "Summary appears to contain child-query fallback, execution failure, or placeholder text. "
                "Regenerate a substantive project analysis before finish()."
            )
            break

    documents = value.get("documents")
    if documents is not None:
        if not isinstance(documents, list):
            errors.append(f"Expected 'documents' to be list, got {type(documents).__name__}.")
            return errors
        if documents:
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
    summary = normalized.get("summary")
    if isinstance(summary, str) and len(summary.strip()) >= 40:
        summary_lines = {line.strip() for line in summary.splitlines()}
        missing_sections = [section for section in REQUIRED_PROJECT_SUMMARY_SECTIONS if section not in summary_lines]
        if missing_sections:
            additions = [
                f"{section}\n\n- This section was not separated in the model output; review the surrounding summary for available evidence."
                for section in missing_sections
            ]
            normalized["summary"] = summary.rstrip() + "\n\n" + "\n\n".join(additions)
    return normalized


def format_analysis_result_errors(errors: list[str]) -> str:
    detail = " ".join(errors)
    return (
        f"Invalid analysis result format: {detail} "
        "Call finish() with a dict containing a substantive Markdown string 'summary'. "
        "The summary must include ## Major directories, ## Important files, ## Relationships, "
        "and ## Uncertainties. You may omit 'documents' because worker source documents are "
        "attached by the runtime."
    )


class ProjectAnalysisPromptBuilder(PromptBuilder):
    # Class constant for backward compatibility with tests checking raw SYSTEM_PROMPT attribute
    SYSTEM_PROMPT = (
        "You are operating a Recursive Language Model runtime for project analysis.\n"
        "Return only Python code and no prose.\n"
        "CRITICAL SYNTAX RULES:\n"
        "- Do not use triple-quoted strings (`\"\"\"` or `'''`) anywhere in generated code.\n"
        "- Do not use f-strings anywhere in generated code.\n"
        "- Build multi-line Markdown as a list of ordinary quoted strings, then join it with '\\n'.\n"
        "- Example: summary = '\\n'.join(['## Major directories', '', '- `src/`: application code'])\n"
        "State persists across steps inside a Python sandbox.\n"
        "Use helpers instead of imports or direct OS access.\n"
        "Available helpers:\n"
        "- list_dir(path='.') -> list[str]\n"
        "- read_text(path, offset=0, limit=2000) -> str\n"
        "- file_info(path) -> dict(path, exists, is_file, is_dir, size_bytes, line_count, char_count, approx_tokens, language, binary)\n"
        "- search_text(path, pattern, max_results=10, context_chars=160) -> list[dict(offset, line, match, excerpt)]\n"
        "- list_artifacts(path='.') -> list[str]\n"
        "- read_artifact(path, offset=0, limit=2000) -> str\n"
        "- read_artifact_json(path) -> parsed artifact json value\n"
        "- grep_artifacts(pattern, max_results=10, context_chars=160) -> list[dict(path, offset, match, excerpt)]\n"
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
        "- Do not assign to helper names such as list_dir, read_text, file_info, search_text, list_artifacts, read_artifact, read_artifact_json, grep_artifacts, read_json, path_exists, is_dir, extract_symbols, llm_query, or finish.\n"
        "- Do not call unavailable helpers such as is_file(path). To test whether a path is a file, call info = file_info(path) and read info['is_file'].\n"
        "- `globals` and `locals` are functions, not dict variables. Do not write globals[...] or locals[...]; call globals() or locals() before membership checks or subscripting, such as globals()['name'].\n"
        "- Avoid triple-quoted strings and f-strings in generated controller code. Build Markdown with ordinary quoted strings joined by '\\n' so the code remains syntactically valid.\n"
        "- All helper paths are relative to the analysis root. Use '.' for the root; do not prefix paths with the root directory name.\n"
        "- Prefer helper calls over large string constants.\n"
        "- A global variable `repo_map` is available in your environment. It is a partial map (up to depth 2, capped at 500 nodes) and includes `repo_map['source_worklist']`, the source files that need explanations. Use it as a starting point to understand the project structure, but always use helpers (like list_dir) to confirm details or explore deeper paths.\n"
        "Minimum exploration before finish:\n"
        "- Do not finish after only inspecting the root directory or README.\n"
        "- Inspect repo_map first, then confirm important areas with helpers.\n"
        "- Treat `repo_map['source_worklist']` as compact coverage context, not as a per-file documentation queue.\n"
        "- File-level source documents are pre-generated by source-document workers outside the controller as artifacts. Start with list_artifacts('.'), read_artifact_json('manifest.json'), and read_artifact_json('relationships.json'), then selectively read source-doc artifacts needed for synthesis. Use read_artifact() only for Markdown snippets. Do not loop over every source file to write source documents.\n"
        "- Treat artifact paths such as `manifest.json`, `relationships.json`, and `source-docs/` as evidence indexes, not as project components to report under Major directories or Important files.\n"
        "- Do not rely on large artifact JSON objects persisting across steps. Re-read artifacts when needed, or keep only compact derived samples in state.\n"
        "- Select a small important path set only for project-level synthesis. Prefer entrypoints, CLI files, application/controller/runtime modules, import hubs, large or symbol-rich source files, and files explicitly named or implied by the user goal.\n"
        "- Deprioritize tests, __init__.py, generated/cache/build files, snapshots, fixtures, and simple config/data files unless the user goal targets them directly.\n"
        "- For final project synthesis, do not call llm_query. Build the report from artifact observations and compact helper results in the sandbox.\n"
        "- The final value MUST include a substantive project-level summary. It may omit documents because worker source documents are attached by the runtime.\n"
        "- Before finish, inspect at least two important non-root directories with list_dir(), file_info(), read_text(), or extract_symbols().\n"
        "- Prioritize code, test, document, and configuration areas when present.\n"
        "- The final summary should describe observed components and responsibilities, not just say exploration started.\n"
        "- Include project-specific top-level source directories from `repo_map['source_worklist']` in `observed_dirs`, even when their repo_map node category is `unknown`, so project packages are not replaced by artifact metadata directories.\n"
        "- When artifact_documents or source_doc_snippets contain concrete paths, the `## Important files` section MUST name those observed paths and their roles. Do not say concrete file paths are unavailable when the manifest or snippets list them.\n"
        "- The final summary MUST start with one or two Japanese prose sentences with no Markdown heading. This leading paragraph is the Executive Summary body.\n"
        "- The final summary MUST include these Markdown sections: `## Major directories`, `## Important files`, `## Relationships`, and `## Uncertainties`.\n"
        "- Before calling finish(), make a best-effort check that summary contains each required heading exactly once. If the generated summary is incomplete, revise the Markdown locally; do not call llm_query to repair it and do not raise your own AssertionError for summary heading validation because the host validator performs final validation.\n"
        "Invalid controller-code examples:\n"
        "import os\n"
        "from pathlib import Path\n"
        "is_file('src/app.py')\n"
        "summary = f'''## Major directories\n"
        "finish({'summary': 'Initial Survey Complete.'})\n"
        "finish('## Major directories\\n\\n- Only a bare string result.')\n"
        "Valid controller-code example:\n"
        "artifact_roots = list_artifacts('.')\n"
        "artifact_manifest = read_artifact_json('manifest.json')\n"
        "source_paths = list(repo_map['source_worklist'])\n"
        "artifact_docs = artifact_manifest.get('documents', [])\n"
        "relationships = read_artifact_json('relationships.json').get('relationships', [])\n"
        "important_paths = []\n"
        "for path in source_paths:\n"
        "    if path.endswith('/cli.py') or path.endswith('__main__.py') or path.endswith('application.py'):\n"
        "        important_paths.append(path)\n"
        "important_paths = important_paths[:5]\n"
        "dirs = []\n"
        "for node in repo_map.get('nodes', []):\n"
        "    path = node.get('path')\n"
        "    if node.get('node_type') == 'dir' and node.get('category') in ['code', 'test'] and path:\n"
        "        dirs.append(path)\n"
        "for path in source_paths:\n"
        "    top_dir = path.split('/')[0] if '/' in path else None\n"
        "    if top_dir and top_dir not in dirs:\n"
        "        dirs.append(top_dir)\n"
        "relation_sample = relationships[:20]\n"
        "important_path_set = set(important_paths)\n"
        "observed_dirs = []\n"
        "for dirname in dirs[:3]:\n"
        "    observed_dirs.append({'path': dirname, 'items': list_dir(dirname)})\n"
        "source_doc_snippets = []\n"
        "for doc in artifact_docs:\n"
        "    artifact_path = doc.get('artifact_path')\n"
        "    document_path = doc.get('source_path') or doc.get('document_path')\n"
        "    source_path = document_path[:-3] if isinstance(document_path, str) and document_path.endswith('.md') else document_path\n"
        "    if artifact_path and source_path in important_path_set:\n"
        "        source_doc_snippets.append({'path': source_path, 'artifact_path': artifact_path, 'snippet': read_artifact(artifact_path, 0, 800)})\n"
        "    if len(source_doc_snippets) >= 8:\n"
        "        break\n"
        "details = []\n"
        "for path in important_paths:\n"
        "    details.append({'path': path, 'info': file_info(path), 'symbols': extract_symbols(path)})\n"
        "summary_lines = ['このレポートは worker が生成したファイル解析 artifact と関係 artifact をもとに、主要な構造と不確実性を統合したものです。', 'RLM controller は必要な artifact だけを読み、全文を state に保持せずに最終要約を作成します。', '', '## Major directories']\n"
        "for observed in observed_dirs:\n"
        "    summary_lines.append('- `' + observed['path'] + '`: ' + str(len(observed['items'])) + ' entries observed.')\n"
        "summary_lines.extend(['', '## Important files'])\n"
        "for detail in details:\n"
        "    info = detail.get('info', {})\n"
        "    symbols = detail.get('symbols', {}).get('symbols', [])\n"
        "    role = 'symbols=' + str(len(symbols)) + ', lines=' + str(info.get('line_count'))\n"
        "    summary_lines.append('- `' + detail['path'] + '`: ' + role)\n"
        "for snippet in source_doc_snippets:\n"
        "    summary_lines.append('- Artifact `' + snippet['artifact_path'] + '` informs `' + snippet['path'] + '`.')\n"
        "summary_lines.extend(['', '## Relationships'])\n"
        "if relation_sample:\n"
        "    for relation in relation_sample[:10]:\n"
        "        source = str(relation.get('source') or relation.get('from') or relation.get('importer') or 'unknown')\n"
        "        target = str(relation.get('target') or relation.get('to') or relation.get('imported') or 'unknown')\n"
        "        summary_lines.append('- `' + source + '` -> `' + target + '`')\n"
        "else:\n"
        "    summary_lines.append('- relationships.json に明示的な関係が少ないため、重要ファイルと import hub を追加確認する必要があります。')\n"
        "summary_lines.extend(['', '## Uncertainties', '- worker artifact の要約粒度に依存するため、重要ファイルの責務説明は source-doc artifact の品質確認が必要です。'])\n"
        "summary = '\\n'.join(summary_lines)\n"
        "required_headings = ['## Major directories', '## Important files', '## Relationships', '## Uncertainties']\n"
        "has_required_headings = isinstance(summary, str) and all(summary.count(heading) == 1 for heading in required_headings)\n"
        "if not has_required_headings:\n"
        "    summary = 'worker artifact をもとに主要構造を統合しました。\\n\\n## Major directories\\n\\n- artifact と repo_map から確認した主要ディレクトリを記録します。\\n\\n## Important files\\n\\n- 選択した重要ファイルと artifact の対応を記録します。\\n\\n## Relationships\\n\\n- relationships.json から確認した依存関係を記録します。\\n\\n## Uncertainties\\n\\n- 追加確認が必要な点を記録します。'\n"
        "finish({'summary': summary})\n"
        "Do not write `import`, `from ... import ...`, or finish(None). Do not finish with a bare string.\n"
        "- When you are done, call finish(value).\n"
    )

    def __init__(self, phase: int = 1):
        super().__init__()
        self.phase = phase

    def build(self, goal: str, step: int, max_steps: int, previous: str, parent_context: Any | None) -> str:
        system_prompt = self._get_system_prompt()
        context_line = "(none)" if parent_context is None else _summarize_parent_context(parent_context)
        return (
            f"{system_prompt}\n"
            f"Goal: {goal}\n"
            f"Current step: {step}/{max_steps}\n"
            f"Parent context: {context_line}\n\n"
            f"Previous observation:\n{previous}\n"
        )

    def _get_system_prompt(self) -> str:
        if self.phase == 1:
            return self.SYSTEM_PROMPT + (
                "\nPhase 1 Specific Guidelines:\n"
                "- Focus on drafting the high-level architecture overview. Outline the system overall layout, major directories, and general purpose.\n"
                "- Do not loop over source files to write detailed descriptions yet.\n"
                "- The final value MUST include a 'summary' containing: ## Major directories (and optionally ## Uncertainties).\n"
            )
        elif self.phase == 2:
            return self.SYSTEM_PROMPT + (
                "\nPhase 2 Specific Guidelines:\n"
                "- Focus on analyzing component relationships, interactions, and dependency graphs.\n"
                "- Read relationships.json and check import statements to understand data flow.\n"
                "- The final value MUST include a 'summary' containing a Mermaid diagram (```mermaid) showing the component interactions, and detailed text explanations under the section '## Relationships'.\n"
                "Phase 2 valid controller-code example:\n"
                "relationships = read_artifact_json('relationships.json').get('relationships', [])\n"
                "relationship_sample = relationships[:15]\n"
                "del relationships\n"
                "summary_lines = ['## Relationships', '', '```mermaid', 'graph TD']\n"
                "for relation in relationship_sample[:8]:\n"
                "    source = relation.get('source') or relation.get('from') or relation.get('importer') or 'ComponentA'\n"
                "    target = relation.get('target') or relation.get('to') or relation.get('imported') or 'ComponentB'\n"
                "    summary_lines.append('  ' + str(source).replace('/', '_').replace('.', '_') + ' --> ' + str(target).replace('/', '_').replace('.', '_'))\n"
                "summary_lines.extend(['```', '', '- relationships.json と import 情報から主要コンポーネント間の依存を要約する。'])\n"
                "summary = '\\n'.join(summary_lines)\n"
                "finish({'summary': summary})\n"
            )
        else:
            return self.SYSTEM_PROMPT + (
                "\nPhase 3 Specific Guidelines:\n"
                "- Focus on synthesizing the final report by merging Phase 1 and Phase 2 summaries and resolving uncertainties.\n"
                "- Before final synthesis, read manifest.json, relationships.json, and selected source-doc artifacts so the report names concrete directories, important files, and observed relationships.\n"
                "- If Phase 1 or Phase 2 text is generic, prefer concrete evidence from artifact_documents, relationship_sample, and source_doc_snippets.\n"
                "- Ensure the output follows the final report format.\n"
                "- The final summary MUST include exactly these Markdown sections: ## Major directories, ## Important files, ## Relationships (containing the Mermaid diagram), and ## Uncertainties.\n"
            )


class ProjectAnalysisChildPromptBuilder(PromptBuilder):
    SYSTEM_PROMPT = (
        "You are a child Recursive Language Model query for project analysis.\n"
        "Return only Python code and no prose.\n"
        "Your job is to answer the focused child goal using the provided Parent context.\n"
        "When the child goal asks for one Markdown explanation, produce a Python string and call finish(markdown_text).\n"
        "When the child goal asks for a project index, synthesis, or integration summary, do not explore with helpers; synthesize only from Parent context, build one Markdown string, and call finish(markdown_text) in the first step.\n"
        "For synthesis, the Markdown MUST include exactly these sections: `## Major directories`, `## Important files`, `## Relationships`, and `## Uncertainties`.\n"
        "For synthesis, never include `## Child Query Fallback`, Step History, execution_error, invalid_code, model_error, raw Python repr, or raw JSON fragments in markdown_text.\n"
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
        "- For synthesis, the only valid first action is building `markdown_text` from Parent context and calling finish(markdown_text).\n"
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


def _merge_analysis_documents(result: Any, documents: list[AnalysisDocument]) -> dict[str, Any]:
    if isinstance(result, dict):
        merged: dict[str, Any] = dict(result)
        summary = str(merged.get("summary") or merged.get("result") or "")
        raw_documents = merged.get("documents")
    else:
        merged = {}
        summary = "" if result is None else str(result)
        raw_documents = []

    documents_by_path: dict[str, dict[str, str]] = {
        document.path: {"path": document.path, "title": document.title, "content": document.content}
        for document in documents
    }
    if isinstance(raw_documents, list):
        for item in raw_documents:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            title = item.get("title")
            content = item.get("content")
            if isinstance(path, str) and isinstance(title, str) and isinstance(content, str):
                documents_by_path[path] = {"path": path, "title": title, "content": content}

    merged["summary"] = summary
    merged["documents"] = list(documents_by_path.values())
    return merged


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
                    f"**Worker Budget:** {controller_result.budget.worker_total_tokens} tokens, {controller_result.budget.worker_llm_calls} LLM calls  ",
                    f"**Synthesis RLM Budget:** {controller_result.budget.synthesis_total_tokens} tokens, {controller_result.budget.synthesis_llm_calls} LLM calls  ",
                    f"**Global Budget:** {controller_result.budget.global_total_tokens} tokens, {controller_result.budget.global_llm_calls} LLM calls  ",
                    "",
                    "## Executive Summary",
                    "",
                    structured.summary or str(controller_result.result),
                    "",
                    *self._render_coverage_section(coverage, fallback_generated_files),
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


class SourceDocumentWorker:
    """Build per-source Markdown documents outside the controller."""

    REQUIRED_SECTIONS = (
        "## Responsibility",
        "## Main Functions / Classes",
        "## Inputs and Outputs",
        "## Dependencies",
        "## Caveats",
    )

    def __init__(self, client: Any | None = None):
        self.client = client

    def build_documents(
        self,
        root_path: Path,
        source_files: list[str],
        run_context: RunContext | None = None,
    ) -> list[AnalysisDocument]:
        return [self._build_document(root_path, source_path, run_context=run_context) for source_path in source_files]

    def build_artifact_files(self, documents: list[AnalysisDocument]) -> dict[str, str]:
        relationships = [
            relationship
            for document in documents
            for relationship in self._document_relationships(document)
        ]
        manifest = {
            "documents": [
                {
                    "artifact_path": f"source-docs/{document.path}",
                    "document_path": document.path,
                    "title": document.title,
                    "content_chars": len(document.content),
                }
                for document in documents
            ],
            "relationships_artifact": "relationships.json",
        }
        artifacts = {
            "manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2),
            "relationships.json": json.dumps({"relationships": relationships}, ensure_ascii=False, indent=2),
        }
        for document in documents:
            artifacts[f"source-docs/{document.path}"] = document.content
        return artifacts

    def _build_document(
        self,
        root_path: Path,
        source_path: str,
        run_context: RunContext | None = None,
    ) -> AnalysisDocument:
        absolute_path = root_path / source_path
        language = detect_language(absolute_path) or "unknown"
        try:
            content = absolute_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            content = ""
        line_count = content.count("\n") + (1 if content else 0)
        symbol_info = extract_symbols(absolute_path, lang=language, max_symbols=12, fallback_limit=1200)
        symbols = symbol_info.get("symbols") if isinstance(symbol_info, dict) else []
        fallback_excerpt = symbol_info.get("fallback_excerpt") if isinstance(symbol_info, dict) else ""
        excerpt = fallback_excerpt if isinstance(fallback_excerpt, str) else content[:1200]
        dependency_lines = self._dependency_lines(content, language)
        symbol_lines = self._symbol_lines(symbols)

        markdown = self._build_llm_markdown(
            source_path=source_path,
            language=language,
            line_count=line_count,
            symbol_lines=symbol_lines,
            dependency_lines=dependency_lines,
            excerpt=excerpt,
        )
        if markdown is None:
            markdown = self._build_deterministic_markdown(
                source_path=source_path,
                language=language,
                line_count=line_count,
                symbol_lines=symbol_lines,
                dependency_lines=dependency_lines,
                excerpt=excerpt,
            )
            if run_context is not None:
                run_context.record_worker_usage(content, markdown)
        elif run_context is not None:
            worker_prompt = self._llm_prompt(
                source_path,
                language,
                line_count,
                symbol_lines,
                dependency_lines,
                excerpt,
            )
            run_context.record_worker_usage(worker_prompt, markdown, llm_calls=1)
        return AnalysisDocument(path=f"{source_path}.md", title=f"Analysis of {source_path}", content=markdown)

    def _build_llm_markdown(
        self,
        source_path: str,
        language: str,
        line_count: int,
        symbol_lines: list[str],
        dependency_lines: list[str],
        excerpt: str,
    ) -> str | None:
        if self.client is None:
            return None

        prompt = self._llm_prompt(source_path, language, line_count, symbol_lines, dependency_lines, excerpt)
        try:
            response = self.client.query(prompt)
        except Exception:
            return None
        markdown = response.strip()
        if not self._is_usable_llm_markdown(markdown):
            return None
        return markdown

    def _llm_prompt(
        self,
        source_path: str,
        language: str,
        line_count: int,
        symbol_lines: list[str],
        dependency_lines: list[str],
        excerpt: str,
    ) -> str:
        symbols = "\n".join(symbol_lines or ["- (none detected)"])
        dependencies = "\n".join(dependency_lines or ["- (none detected)"])
        return "\n".join(
            [
                "You are a normal LLM worker generating one bounded source-file analysis artifact.",
                "Return Markdown only. Do not mention the RLM controller.",
                "Write Japanese explanations grounded only in the provided metadata and excerpt.",
                "Include exactly these sections: ## Responsibility, ## Main Functions / Classes, ## Inputs and Outputs, ## Dependencies, ## Caveats.",
                "",
                f"Path: {source_path}",
                f"Language: {language}",
                f"Lines: {line_count}",
                "",
                "Detected symbols:",
                symbols,
                "",
                "Detected dependencies:",
                dependencies,
                "",
                "Bounded source excerpt:",
                "```",
                excerpt[:4000],
                "```",
            ]
        )

    def _is_usable_llm_markdown(self, markdown: str) -> bool:
        if len(markdown.strip()) < 120:
            return False
        lowered = markdown.lower()
        if "```python" in lowered and "finish(" in lowered:
            return False
        return all(section in markdown for section in self.REQUIRED_SECTIONS)

    def _build_deterministic_markdown(
        self,
        source_path: str,
        language: str,
        line_count: int,
        symbol_lines: list[str],
        dependency_lines: list[str],
        excerpt: str,
    ) -> str:
        return "\n".join(
            [
                f"# Analysis of {source_path}",
                "",
                "## Responsibility",
                "",
                f"`{source_path}` is a `{language}` source file with {line_count} lines. "
                "This deterministic worker document summarizes the file from static metadata, detected symbols, and a bounded source excerpt.",
                "",
                "## Main Functions / Classes",
                "",
                *(symbol_lines or ["- No functions or classes were detected by the static symbol extractor."]),
                "",
                "## Inputs and Outputs",
                "",
                "- Inputs: values accepted by the detected functions/classes or module-level configuration visible in the source excerpt.",
                "- Outputs: return values, side effects, CLI behavior, tests, or exported definitions implied by the detected symbols.",
                "",
                "## Dependencies",
                "",
                *(dependency_lines or ["- No import/include dependency lines were detected in the bounded scan."]),
                "",
                "## Caveats",
                "",
                "- This file-level document is generated by a deterministic worker, not by the RLM controller.",
                "- Nuanced behavior beyond the static symbols and excerpt should be verified against the source when this file is critical.",
                "",
                "## Excerpt",
                "",
                "```",
                excerpt[:1200],
                "```",
                "",
            ]
        )

    def _symbol_lines(self, symbols: Any) -> list[str]:
        if not isinstance(symbols, list):
            return []
        lines: list[str] = []
        for symbol in symbols:
            if isinstance(symbol, str) and symbol.strip():
                lines.append(f"- `{symbol.strip().splitlines()[0]}`")
        return lines

    def _dependency_lines(self, content: str, language: str) -> list[str]:
        prefixes_by_language = {
            "python": ("import ", "from "),
            "javascript": ("import ", "require("),
            "typescript": ("import ", "require("),
            "go": ("import ",),
            "rust": ("use ", "extern crate "),
            "java": ("import ",),
        }
        prefixes = prefixes_by_language.get(language, ("import ", "from ", "use ", "#include"))
        lines: list[str] = []
        for raw_line in content.splitlines()[:200]:
            stripped = raw_line.strip()
            if stripped.startswith(prefixes):
                lines.append(f"- `{stripped[:160]}`")
            if len(lines) >= 12:
                break
        return lines

    def _document_relationships(self, document: AnalysisDocument) -> list[dict[str, str]]:
        relationships: list[dict[str, str]] = []
        in_dependencies = False
        for line in document.content.splitlines():
            if line == "## Dependencies":
                in_dependencies = True
                continue
            if in_dependencies and line.startswith("## "):
                break
            if not in_dependencies or not line.startswith("- `") or not line.endswith("`"):
                continue
            statement = line[3:-1].strip()
            if not statement or statement.startswith("No import/include"):
                continue
            relationships.append(
                {
                    "source": document.path.removesuffix(".md"),
                    "target": self._dependency_target(statement),
                    "statement": statement,
                }
            )
        return relationships

    def _dependency_target(self, statement: str) -> str:
        if statement.startswith("from "):
            parts = statement.split()
            return parts[1] if len(parts) > 1 else statement
        if statement.startswith("import "):
            return statement.removeprefix("import ").split()[0].rstrip(",")
        if statement.startswith("use "):
            return statement.removeprefix("use ").rstrip(";")
        if statement.startswith("extern crate "):
            return statement.removeprefix("extern crate ").rstrip(";")
        if statement.startswith("#include"):
            return statement.removeprefix("#include").strip()
        return statement


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
        run_context = RunContext(limits=limits)
        source_files = AnalysisDocBuilder(self.output_dir or (root / ".isohyps-output-validation"))._collect_source_files(root)
        source_worker = SourceDocumentWorker(client=self.client)
        worker_documents = source_worker.build_documents(root, source_files, run_context=run_context)
        artifact_files = source_worker.build_artifact_files(worker_documents)

        # ----------------------------------------------------
        # Phase 1: Survey & Hypothesis (Architecture Overview)
        # ----------------------------------------------------
        run_context.steps_used = 0
        controller_p1 = RLMController(
            client=self.client,
            root=root,
            run_context=run_context,
            prompt_builder=ProjectAnalysisPromptBuilder(phase=1),
            child_config=ChildQueryConfig(
                prompt_builder=ProjectAnalysisChildPromptBuilder(),
                limits=PartialBudgetLimits(max_steps=PROJECT_ANALYSIS_CHILD_MAX_STEPS),
            ),
            artifact_files=artifact_files,
        )
        goal_p1 = DEFAULT_GOAL_TEMPLATE_PHASE1.format(root_name=root.name)

        def validate_p1(value: Any) -> list[str]:
            if not isinstance(value, dict) or "summary" not in value or not isinstance(value["summary"], str):
                return ["Expected finish(value) to receive a dict with 'summary' key."]
            summary = value["summary"]
            if "## Major directories" not in summary:
                return ["Expected 'summary' to include '## Major directories'."]
            return []

        result_p1 = controller_p1.run(
            goal=goal_p1,
            finish_validator=validate_p1,
            finish_normalizer=normalize_analysis_result,
            finish_error_formatter=format_analysis_result_errors,
        )

        if result_p1.status != "finished":
            return self._handle_failure(root, result_p1, worker_documents)

        phase1_summary = result_p1.result.get("summary", "")

        # ----------------------------------------------------
        # Phase 2: Probe Relationships (Mermaid and Explanations)
        # ----------------------------------------------------
        run_context.steps_used = 0
        controller_p2 = RLMController(
            client=self.client,
            root=root,
            run_context=run_context,
            prompt_builder=ProjectAnalysisPromptBuilder(phase=2),
            child_config=ChildQueryConfig(
                prompt_builder=ProjectAnalysisChildPromptBuilder(),
                limits=PartialBudgetLimits(max_steps=PROJECT_ANALYSIS_CHILD_MAX_STEPS),
            ),
            artifact_files=artifact_files,
        )
        goal_p2 = DEFAULT_GOAL_TEMPLATE_PHASE2.format(root_name=root.name, phase1_summary=phase1_summary)

        def validate_p2(value: Any) -> list[str]:
            if not isinstance(value, dict) or "summary" not in value or not isinstance(value["summary"], str):
                return ["Expected finish(value) to receive a dict with 'summary' key."]
            summary = value["summary"]
            if "## Relationships" not in summary:
                return ["Expected 'summary' to include '## Relationships'."]
            if "```mermaid" not in summary:
                return ["Expected 'summary' to include a Mermaid diagram (```mermaid)."]
            return []

        result_p2 = controller_p2.run(
            goal=goal_p2,
            finish_validator=validate_p2,
            finish_normalizer=normalize_analysis_result,
            finish_error_formatter=format_analysis_result_errors,
        )

        if result_p2.status != "finished":
            return self._handle_failure(root, result_p2, worker_documents)

        phase2_summary = result_p2.result.get("summary", "")

        # ----------------------------------------------------
        # Phase 3: Synthesis & Report Generation
        # ----------------------------------------------------
        run_context.steps_used = 0
        controller_p3 = RLMController(
            client=self.client,
            root=root,
            run_context=run_context,
            prompt_builder=ProjectAnalysisPromptBuilder(phase=3),
            child_config=ChildQueryConfig(
                prompt_builder=ProjectAnalysisChildPromptBuilder(),
                limits=PartialBudgetLimits(max_steps=PROJECT_ANALYSIS_CHILD_MAX_STEPS),
            ),
            artifact_files=artifact_files,
        )
        goal_p3 = DEFAULT_GOAL_TEMPLATE_PHASE3.format(
            root_name=root.name,
            phase1_summary=phase1_summary,
            phase2_summary=phase2_summary,
        )

        def validate_p3(value: Any) -> list[str]:
            merged_value = _merge_analysis_documents(value, worker_documents)
            return validate_project_analysis_finish(merged_value, source_files=source_files)

        result_p3 = controller_p3.run(
            goal=goal_p3,
            finish_validator=validate_p3,
            finish_normalizer=normalize_analysis_result,
            finish_error_formatter=format_analysis_result_errors,
        )

        result_p3.result = _merge_analysis_documents(result_p3.result, worker_documents)
        self.last_result = result_p3

        summary = result_p3.error or str(result_p3.result)
        if result_p3.status == "finished" and isinstance(result_p3.result, dict):
            summary = str(result_p3.result.get("summary") or summary)
        elif result_p3.status != "finished":
            return self._handle_failure(root, result_p3, worker_documents)

        if self.output_dir:
            write_analysis_docs(
                output_dir=self.output_dir,
                root_path=root,
                controller_result=result_p3,
                backend=self.backend_name,
                model=self.model_name,
            )
        return summary

    def _handle_failure(self, root: Path, result: ControllerResult, worker_documents: list[AnalysisDocument]) -> str:
        result.result = _merge_analysis_documents(result.result, worker_documents)
        self.last_result = result
        summary = result.error or str(result.result)
        if isinstance(result.result, dict) and "summary" in result.result and result.result["summary"]:
            summary = str(result.result["summary"])

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
