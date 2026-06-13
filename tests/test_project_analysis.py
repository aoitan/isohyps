import shutil
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from analyzer import BaseLLMClient
from isohyps.project_analysis import (
    AnalysisDocBuilder,
    DEFAULT_GOAL_TEMPLATE_PHASE1,
    DEFAULT_GOAL_TEMPLATE_PHASE2,
    DEFAULT_GOAL_TEMPLATE_PHASE3,
    PROJECT_ANALYSIS_CHILD_MAX_STEPS,
    ProjectAnalysisChildPromptBuilder,
    ProjectAnalysisPromptBuilder,
    RLMRuntimeAnalyzer,
    SourceDocumentWorker,
    format_analysis_result_errors,
    normalize_analysis_result,
    validate_analysis_result,
    validate_project_analysis_finish,
    write_analysis_docs,
)
from isohyps.rlm_runtime import BudgetSnapshot, ChildQueryConfig, ControllerResult, ExecutionObservation
from tests.test_utils import ScriptedClient


def _summary_with_required_sections(intro: str) -> str:
    return (
        f"{intro}\n\n"
        "## Major directories\n\n"
        "- `isohyps/`: Runtime and analysis implementation.\n\n"
        "## Important files\n\n"
        "- `isohyps/project_analysis.py`: Project analysis orchestration.\n\n"
        "## Relationships\n\n"
        "```mermaid\n"
        "graph TD\n"
        "  CLI --> Runtime\n"
        "```\n\n"
        "- The CLI delegates project analysis to the runtime controller.\n\n"
        "## Uncertainties\n\n"
        "- Exact deployment constraints are outside this summary."
    )


REQUIRED_SUMMARY_SECTIONS = _summary_with_required_sections("")
WORKER_LLM_DOCUMENT = "\n".join(
    [
        "# Analysis of app.py",
        "",
        "## Responsibility",
        "",
        "`app.py` は fixture の Python ソースで、hello 関数の責務を説明する worker LLM artifact です。",
        "",
        "## Main Functions / Classes",
        "",
        "- `hello`: fixture の関数。",
        "",
        "## Inputs and Outputs",
        "",
        "- Inputs: なし。",
        "- Outputs: なし。",
        "",
        "## Dependencies",
        "",
        "- (none detected)",
        "",
        "## Caveats",
        "",
        "- fixture 用の短い LLM worker 出力です。",
    ]
)


class TestProjectAnalysisContract(unittest.TestCase):
    def _summary_with_required_sections(self) -> str:
        return _summary_with_required_sections(
            "The project contains a CLI layer, runtime controller, tests, and documentation areas."
        )

    def test_validate_analysis_result_rejects_invalid_payloads(self):
        self.assertEqual(
            validate_analysis_result("done"),
            ["Expected finish(value) to receive a dict, got str."],
        )
        self.assertEqual(validate_analysis_result({"summary": "ok"}), [])
        self.assertEqual(validate_analysis_result({"summary": "ok", "documents": [{}]}), [])
        self.assertEqual(
            validate_analysis_result({"summary": "ok", "documents": "README.md"}),
            ["Expected 'documents' to be list, got str."],
        )

    def test_validate_project_analysis_finish_rejects_shallow_finish(self):
        errors = validate_project_analysis_finish(
            {
                "summary": "Initial exploration of the root directory completed.",
                "documents": [{"path": "a.md", "title": "A", "content": "Too short"}],
            }
        )

        self.assertTrue(any("initial/root-only exploration" in error for error in errors))
        self.assertTrue(any("substantive 'content'" in error for error in errors))

    def test_validate_project_analysis_finish_accepts_omitted_documents(self):
        errors_none = validate_project_analysis_finish(
            {
                "summary": self._summary_with_required_sections(),
            }
        )
        self.assertEqual(errors_none, [])

        errors_empty = validate_project_analysis_finish(
            {
                "summary": self._summary_with_required_sections(),
                "documents": [],
            }
        )
        self.assertEqual(errors_empty, [])

    def test_validate_project_analysis_finish_accepts_substantive_documents(self):
        errors = validate_project_analysis_finish(
            {
                "summary": self._summary_with_required_sections(),
                "documents": [
                    {
                        "path": "index.md",
                        "title": "Overview",
                        "content": "The repository includes a CLI entrypoint, core runtime modules, tests, and documentation.",
                    }
                ],
            }
        )

        self.assertEqual(errors, [])

    def test_validate_project_analysis_finish_rejects_missing_required_summary_sections(self):
        errors = validate_project_analysis_finish(
            {
                "summary": "The project contains a CLI layer, runtime controller, tests, and documentation areas.",
                "documents": [
                    {
                        "path": "index.md",
                        "title": "Overview",
                        "content": "The repository includes a CLI entrypoint, core runtime modules, tests, and documentation.",
                    }
                ],
            }
        )

        self.assertTrue(any("required Markdown sections" in error for error in errors))
        self.assertTrue(any("## Major directories" in error for error in errors))
        self.assertTrue(any("## Important files" in error for error in errors))
        self.assertTrue(any("## Relationships" in error for error in errors))
        self.assertTrue(any("## Uncertainties" in error for error in errors))

    def test_validate_project_analysis_finish_rejects_report_placeholders(self):
        errors = validate_project_analysis_finish(
            {
                "summary": (
                    "## Major directories\n\n"
                    "- [Directory name]: [Description]\n\n"
                    "## Important files\n\n"
                    "- [File path]: [Role]\n\n"
                    "## Relationships\n\n"
                    "```mermaid\n"
                    "graph TD\n"
                    "  A --> B\n"
                    "```\n\n"
                    "[Description]\n\n"
                    "## Uncertainties\n\n"
                    "- [Uncertainty 1]"
                )
            }
        )

        self.assertTrue(any("placeholder text" in error for error in errors))

    def test_validate_project_analysis_finish_rejects_empty_relationships_without_mermaid(self):
        errors = validate_project_analysis_finish(
            {
                "summary": (
                    "The project contains a CLI layer, runtime controller, tests, and documentation areas.\n\n"
                    "## Major directories\n\n"
                    "- `isohyps/`: Runtime and analysis implementation.\n\n"
                    "## Important files\n\n"
                    "- `isohyps/project_analysis.py`: Project analysis orchestration.\n\n"
                    "## Relationships\n\n"
                    "明示的な依存関係のグラフは生成されませんでした。\n\n"
                    "## Uncertainties\n\n"
                    "- Exact deployment constraints are outside this summary."
                )
            }
        )

        self.assertTrue(any("Mermaid diagram" in error for error in errors))
        self.assertTrue(any("placeholder text" in error for error in errors))

    def test_validate_project_analysis_finish_accepts_missing_source_documents(self):
        errors = validate_project_analysis_finish(
            {
                "summary": self._summary_with_required_sections(),
                "documents": [
                    {
                        "path": "index.md",
                        "title": "Overview",
                        "content": "The repository includes a CLI entrypoint, core runtime modules, tests, and documentation.",
                    }
                ],
            },
            source_files=["app.py"],
        )

        self.assertEqual(errors, [])

    def test_validate_project_analysis_finish_accepts_source_path_markdown_document(self):
        errors = validate_project_analysis_finish(
            {
                "summary": self._summary_with_required_sections(),
                "documents": [
                    {
                        "path": "app.py.md",
                        "title": "App",
                        "content": "The app module exposes a small CLI-facing function and its runtime behavior.",
                    }
                ],
            },
            source_files=["app.py"],
        )

        self.assertEqual(errors, [])

    def test_normalize_analysis_result_flattens_nested_summary_dict(self):
        normalized = normalize_analysis_result(
            {
                "summary": {
                    "summary": "child-summary",
                    "documents": [{"path": "child.md", "content": "x"}],
                }
            }
        )

        self.assertEqual(normalized["summary"], "child-summary")
        self.assertEqual(normalized["documents"][0]["path"], "child.md")

    def test_validate_project_analysis_finish_rejects_normalized_placeholder_sections(self):
        normalized = normalize_analysis_result(
            {
                "summary": (
                    "The project contains a CLI layer, runtime controller, tests, and documentation areas.\n\n"
                    "## Major directories\n\n"
                    "- `isohyps/`: Runtime and analysis implementation."
                )
            }
        )

        errors = validate_project_analysis_finish(normalized)

        self.assertTrue(any("placeholder text" in error for error in errors))
        self.assertIn("## Important files", normalized["summary"])
        self.assertIn("## Relationships", normalized["summary"])
        self.assertIn("## Uncertainties", normalized["summary"])

    def test_validate_project_analysis_finish_rejects_child_query_fallback_summary(self):
        errors = validate_project_analysis_finish(
            {
                "summary": (
                    "## Child Query Fallback\n\n"
                    "The deep-dive child query for `unknown` could not complete "
                    "(budget_exceeded: max_steps reached).\n\n"
                    "## Major directories\n\n"
                    "- Unavailable because fallback text replaced the project summary.\n\n"
                    "## Important files\n\n"
                    "- Unavailable because fallback text replaced the project summary.\n\n"
                    "## Relationships\n\n"
                    "- Unavailable because fallback text replaced the project summary.\n\n"
                    "## Uncertainties\n\n"
                    "- The project synthesis failed."
                )
            }
        )

        self.assertTrue(any("child-query fallback" in error for error in errors))

    def test_validate_project_analysis_finish_rejects_japanese_fallback_summary(self):
        errors = validate_project_analysis_finish(
            {
                "summary": (
                    "このドキュメントは、解析が予算制限により中断されたことによるフォールバック情報です。"
                    "具体的なプロジェクトの構造に関する情報は含まれていません。\n\n"
                    "## Major directories\n\n"
                    "コンテキスト内にディレクトリの情報は見当たりません。\n\n"
                    "## Important files\n\n"
                    "`unknown` (解析失敗のため詳細不明)\n\n"
                    "## Relationships\n\n"
                    "定義された関係性は存在しません。\n\n"
                    "## Uncertainties\n\n"
                    "解析が中断されたため、プロジェクトの全体像に関する不確実性が非常に高い状態です。"
                )
            }
        )

        self.assertTrue(any("placeholder text" in error for error in errors))

    def test_project_analysis_prompt_contains_strict_schema(self):
        prompt = ProjectAnalysisPromptBuilder.SYSTEM_PROMPT
        self.assertIn("'summary': str", prompt)
        self.assertIn("'documents': [", prompt)
        self.assertIn("'path': str", prompt)
        self.assertIn("'title': str", prompt)
        self.assertIn("'content': str", prompt)

    def test_project_analysis_prompt_builder_build_includes_system_prompt_rules(self):
        builder = ProjectAnalysisPromptBuilder(phase=1)
        prompt = builder.build(
            goal="Test goal",
            step=1,
            max_steps=3,
            previous="Previous observations",
            parent_context=None
        )
        self.assertIn("globals` and `locals` are functions, not dict variables", prompt)
        self.assertIn("artifact_roots = list_artifacts('.')", prompt)

    def test_project_analysis_prompt_builder_build_with_parent_context_does_not_raise(self):
        builder = ProjectAnalysisPromptBuilder(phase=1)
        prompt = builder.build(
            goal="Test goal",
            step=1,
            max_steps=3,
            previous="Previous observations",
            parent_context="Some parent context"
        )
        self.assertIn("Parent context:", prompt)

    def test_phase2_prompt_contains_phase_specific_controller_code_example(self):
        prompt = ProjectAnalysisPromptBuilder(phase=2).build(
            goal="Phase 2 goal",
            step=1,
            max_steps=3,
            previous="",
            parent_context=None,
        )

        self.assertIn("Phase 2 valid controller-code example", prompt)
        self.assertIn("relationships = read_artifact_json('relationships.json').get('relationships', [])", prompt)
        self.assertIn("relationship_sample = relationships[:15]", prompt)
        self.assertIn("del relationships", prompt)
        self.assertIn("summary_lines = ['## Relationships', '', '```mermaid', 'graph TD']", prompt)
        self.assertIn("summary = '\\n'.join(summary_lines)", prompt)
        self.assertIn("finish({'summary': summary})", prompt)

    def test_default_goal_keeps_file_card_state_compact(self):
        goal = DEFAULT_GOAL_TEMPLATE_PHASE1
        self.assertIn("Phase 1: Survey the project rooted at", goal)
        self.assertIn("identify uncertainties", goal)
        self.assertIn("concise architecture description", goal)

    def test_phase2_goal_shows_mermaid_summary_construction_example(self):
        goal = DEFAULT_GOAL_TEMPLATE_PHASE2.format(root_name="repo", phase1_summary="phase one")
        self.assertIn("Build the Phase 2 summary with this Python shape", goal)
        self.assertIn("summary_lines = [", goal)
        self.assertIn("'## Relationships'", goal)
        self.assertIn("'```mermaid'", goal)
        self.assertIn("'```'", goal)
        self.assertIn("summary = '\\n'.join(summary_lines)", goal)
        self.assertIn("finish({'summary': summary})", goal)
        self.assertIn("Do not use triple-quoted strings or f-strings", goal)

    def test_phase3_goal_requires_artifact_driven_synthesis(self):
        goal = DEFAULT_GOAL_TEMPLATE_PHASE3.format(
            root_name="repo",
            phase1_summary="phase one",
            phase2_summary="phase two",
        )

        self.assertIn("list_artifacts('.')", goal)
        self.assertIn("read_artifact_json('manifest.json')", goal)
        self.assertIn("read_artifact_json('relationships.json')", goal)
        self.assertIn("read_artifact('source-docs/...')", goal)
        self.assertIn("do not rely only on Phase 1 or Phase 2 text", goal)

    def test_project_analysis_prompt_contains_minimum_exploration_rules(self):
        prompt = ProjectAnalysisPromptBuilder.SYSTEM_PROMPT
        self.assertLess(prompt.index("CRITICAL SYNTAX RULES"), prompt.index("Available helpers:"))
        self.assertIn("Do not use triple-quoted strings (`\"\"\"` or `'''`) anywhere in generated code.", prompt)
        self.assertIn("Do not use f-strings anywhere in generated code.", prompt)
        self.assertIn("Build multi-line Markdown as a list of ordinary quoted strings", prompt)
        self.assertIn("Minimum exploration before finish", prompt)
        self.assertIn("Do not finish after only inspecting the root directory or README.", prompt)
        self.assertIn("at least two important non-root directories", prompt)
        self.assertIn("file_info", prompt)
        self.assertIn("search_text", prompt)
        self.assertIn("list_artifacts", prompt)
        self.assertIn("read_artifact", prompt)
        self.assertIn("read_artifact_json", prompt)
        self.assertIn("grep_artifacts", prompt)
        self.assertIn("Do not assign to helper names", prompt)
        self.assertIn("Do not call unavailable helpers such as is_file(path)", prompt)
        self.assertIn("call info = file_info(path) and read info['is_file']", prompt)
        self.assertIn("Do not write globals[...] or locals[...]", prompt)
        self.assertIn("call globals() or locals() before membership checks or subscripting", prompt)
        self.assertIn("Avoid triple-quoted strings and f-strings", prompt)
        self.assertIn("ordinary quoted strings joined by '\\n'", prompt)
        self.assertIn("repo_map['source_worklist']", prompt)
        self.assertIn("compact coverage context", prompt)
        self.assertIn("pre-generated by source-document workers outside the controller as artifacts", prompt)
        self.assertIn("read_artifact_json('manifest.json')", prompt)
        self.assertIn("read_artifact_json('relationships.json')", prompt)
        self.assertIn("Use read_artifact() only for Markdown snippets", prompt)
        self.assertIn("Do not loop over every source file to write source documents", prompt)
        self.assertIn("Treat artifact paths such as `manifest.json`, `relationships.json`, and `source-docs/` as evidence indexes", prompt)
        self.assertIn("not as project components to report under Major directories or Important files", prompt)
        self.assertIn("Do not rely on large artifact JSON objects persisting across steps", prompt)
        self.assertIn("Select a small important path set only for project-level synthesis", prompt)
        self.assertIn("Prefer entrypoints, CLI files, application/controller/runtime modules", prompt)
        self.assertIn("Deprioritize tests, __init__.py", prompt)
        self.assertIn("For final project synthesis, do not call llm_query.", prompt)
        self.assertIn("Build the report from artifact observations and compact helper results", prompt)
        self.assertIn("worker source documents are attached by the runtime", prompt)
        self.assertIn("Include project-specific top-level source directories", prompt)
        self.assertIn("even when their repo_map node category is `unknown`", prompt)
        self.assertIn("When artifact_documents or source_doc_snippets contain concrete paths", prompt)
        self.assertIn("Do not say concrete file paths are unavailable", prompt)
        self.assertIn("start with one or two Japanese prose sentences with no Markdown heading", prompt)
        self.assertIn("This leading paragraph is the Executive Summary body.", prompt)
        self.assertIn("check that summary contains each required heading exactly once", prompt)
        self.assertIn("Invalid controller-code examples", prompt)
        self.assertIn("import os", prompt)
        self.assertIn("from pathlib import Path", prompt)
        self.assertIn("is_file('src/app.py')", prompt)
        self.assertIn("summary = f'''## Major directories", prompt)
        self.assertIn("Initial Survey Complete.", prompt)
        self.assertIn("## Major directories", prompt)
        self.assertIn("## Important files", prompt)
        self.assertIn("## Relationships", prompt)
        self.assertIn("## Uncertainties", prompt)

    def test_project_analysis_prompt_contains_valid_controller_code_example(self):
        prompt = ProjectAnalysisPromptBuilder.SYSTEM_PROMPT
        self.assertIn("Valid controller-code example", prompt)
        self.assertNotIn("if 'important_paths' not in globals():", prompt)
        self.assertIn("artifact_roots = list_artifacts('.')", prompt)
        self.assertIn("artifact_manifest = read_artifact_json('manifest.json')", prompt)
        self.assertIn("artifact_docs = artifact_manifest.get('documents', [])", prompt)
        self.assertIn("relationships = read_artifact_json('relationships.json').get('relationships', [])", prompt)
        self.assertIn("source_paths = list(repo_map['source_worklist'])", prompt)
        self.assertIn("important_paths.append(path)", prompt)
        self.assertIn("path = node.get('path')", prompt)
        self.assertIn("node.get('category') in ['code', 'test'] and path", prompt)
        self.assertIn("top_dir = path.split('/')[0] if '/' in path else None", prompt)
        self.assertIn("if top_dir and top_dir not in dirs:", prompt)
        self.assertIn("dirs.append(top_dir)", prompt)
        self.assertIn("relation_sample = relationships[:20]", prompt)
        self.assertIn("important_path_set = set(important_paths)", prompt)
        self.assertIn("source_doc_snippets = []", prompt)
        self.assertIn("document_path = doc.get('source_path') or doc.get('document_path')", prompt)
        self.assertIn("source_path = document_path[:-3] if isinstance(document_path, str) and document_path.endswith('.md') else document_path", prompt)
        self.assertIn("if artifact_path and source_path in important_path_set:", prompt)
        self.assertIn("'artifact_path': artifact_path", prompt)
        self.assertIn("read_artifact(artifact_path, 0, 800)", prompt)
        self.assertIn("if len(source_doc_snippets) >= 8:", prompt)
        self.assertIn("observed_dirs = []", prompt)
        self.assertIn("details.append({'path': path, 'info': file_info(path), 'symbols': extract_symbols(path)})", prompt)
        self.assertIn("For final project synthesis, do not call llm_query.", prompt)
        self.assertIn("summary_lines = ['このレポートは worker が生成したファイル解析 artifact", prompt)
        self.assertIn("start with one or two Japanese prose sentences with no Markdown heading", prompt)
        self.assertIn("for observed in observed_dirs:", prompt)
        self.assertIn("for relation in relation_sample[:10]:", prompt)
        self.assertIn("RLM controller は必要な artifact だけを読み", prompt)
        self.assertNotIn("summary = llm_query(", prompt)
        self.assertIn("required_headings = ['## Major directories', '## Important files', '## Relationships', '## Uncertainties']", prompt)
        self.assertIn("summary.count(heading) == 1", prompt)
        self.assertIn("do not raise your own AssertionError for summary heading validation", prompt)
        self.assertIn("has_required_headings = isinstance(summary, str)", prompt)
        self.assertIn("do not call llm_query to repair it", prompt)
        self.assertNotIn("Revise this project index into valid Markdown", prompt)
        self.assertNotIn("Invalid summary headings", prompt)
        self.assertNotIn("Describe observed source directories.", prompt)
        self.assertIn("## Relationships", prompt)
        self.assertIn("## Uncertainties", prompt)
        self.assertIn("finish({'summary': summary})", prompt)
        self.assertNotIn("str(artifact_roots)", prompt)
        self.assertNotIn("str(observed_dirs)", prompt)
        self.assertNotIn("str(details)", prompt)
        self.assertNotIn("artifact_relationships =", prompt)
        self.assertNotIn("+ artifact_relationships +", prompt)
        self.assertNotIn("pending = []", prompt)
        self.assertIn("Do not write `import`, `from ... import ...`, or finish(None).", prompt)
        self.assertIn("Do not finish with a bare string.", prompt)

    def test_phase3_prompt_prefers_concrete_artifact_evidence(self):
        prompt = ProjectAnalysisPromptBuilder(phase=3).build(
            goal="Phase 3 goal",
            step=1,
            max_steps=3,
            previous="",
            parent_context=None,
        )

        self.assertIn("read manifest.json, relationships.json, and selected source-doc artifacts", prompt)
        self.assertIn("names concrete directories, important files, and observed relationships", prompt)
        self.assertIn("If Phase 1 or Phase 2 text is generic", prompt)
        self.assertIn("artifact_documents, relationship_sample, and source_doc_snippets", prompt)

    def test_finish_error_guidance_matches_worker_document_runtime(self):
        message = format_analysis_result_errors(["Expected 'summary' to contain a substantive project analysis."])

        self.assertIn("substantive Markdown string 'summary'", message)
        self.assertIn("## Major directories", message)
        self.assertIn("## Important files", message)
        self.assertIn("## Relationships", message)
        self.assertIn("## Uncertainties", message)
        self.assertIn("You may omit 'documents'", message)
        self.assertNotIn("non-empty 'documents' list", message)

    def test_project_analysis_child_prompt_requires_python_finish_for_markdown(self):
        prompt = ProjectAnalysisChildPromptBuilder.SYSTEM_PROMPT
        self.assertIn("Return only Python code and no prose.", prompt)
        self.assertIn("finish(markdown_text)", prompt)
        self.assertIn("When the Parent context contains a `file` card", prompt)
        self.assertIn("When the child goal asks for a project index, synthesis, or integration summary", prompt)
        self.assertIn("synthesize only from Parent context", prompt)
        self.assertIn("## Major directories", prompt)
        self.assertIn("## Important files", prompt)
        self.assertIn("## Relationships", prompt)
        self.assertIn("## Uncertainties", prompt)
        self.assertIn("never include `## Child Query Fallback`", prompt)
        self.assertIn("execution_error, invalid_code, model_error", prompt)
        self.assertIn("raw Python repr, or raw JSON fragments", prompt)
        self.assertIn("do not return a dict for single-file Markdown", prompt)
        self.assertIn("do not call helper functions to gather more context", prompt)
        self.assertIn("the only valid first action is building `markdown_text`", prompt)
        self.assertIn("For synthesis, the only valid first action is building `markdown_text`", prompt)
        self.assertIn("Prefer finishing in one step.", prompt)
        self.assertIn("Do not create documents", prompt)
        self.assertIn("do not call record_document()", prompt)


class TestSourceDocumentWorker(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.root = Path(self.test_dir) / "repo"
        self.root.mkdir()
        (self.root / "app.py").write_text("import os\n\ndef hello(name):\n    return name\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_builds_deterministic_source_document(self):
        documents = SourceDocumentWorker().build_documents(self.root, ["app.py"])

        self.assertEqual(len(documents), 1)
        document = documents[0]
        self.assertEqual(document.path, "app.py.md")
        self.assertIn("## Responsibility", document.content)
        self.assertIn("## Main Functions / Classes", document.content)
        self.assertIn("- `def hello(name):`", document.content)
        self.assertIn("## Inputs and Outputs", document.content)
        self.assertIn("## Dependencies", document.content)
        self.assertIn("- `import os`", document.content)
        self.assertIn("deterministic worker", document.content)
        self.assertNotIn("This fallback document was generated", document.content)

    def test_builds_artifact_files_for_controller_selection(self):
        worker = SourceDocumentWorker()
        documents = worker.build_documents(self.root, ["app.py"])

        artifacts = worker.build_artifact_files(documents)

        self.assertIn("manifest.json", artifacts)
        self.assertIn("relationships.json", artifacts)
        self.assertIn("source-docs/app.py.md", artifacts)
        self.assertIn("source-docs/app.py.md", artifacts["manifest.json"])
        self.assertIn('"source": "app.py"', artifacts["relationships.json"])
        self.assertIn('"target": "os"', artifacts["relationships.json"])
        self.assertIn('"statement": "import os"', artifacts["relationships.json"])
        self.assertIn("## Responsibility", artifacts["source-docs/app.py.md"])

    def test_counts_worker_document_generation_in_worker_budget(self):
        from isohyps.rlm_runtime import BudgetLimits, RunContext

        context = RunContext(limits=BudgetLimits())

        SourceDocumentWorker().build_documents(self.root, ["app.py"], run_context=context)

        snapshot = context.snapshot()
        self.assertGreater(snapshot.worker_total_tokens, 0)
        self.assertEqual(snapshot.worker_llm_calls, 0)
        self.assertEqual(snapshot.global_total_tokens, 0)
        self.assertEqual(snapshot.synthesis_total_tokens, 0)

    def test_uses_llm_worker_when_client_is_available(self):
        from isohyps.rlm_runtime import BudgetLimits, RunContext

        client = MagicMock(spec=BaseLLMClient)
        client.query.return_value = WORKER_LLM_DOCUMENT
        context = RunContext(limits=BudgetLimits())

        document = SourceDocumentWorker(client=client).build_documents(
            self.root,
            ["app.py"],
            run_context=context,
        )[0]

        self.assertIn("worker LLM artifact", document.content)
        self.assertNotIn("deterministic worker", document.content)
        self.assertEqual(context.snapshot().worker_llm_calls, 1)
        client.query.assert_called_once()


class TestPackagingDependencies(unittest.TestCase):
    def test_symbols_extra_includes_tree_sitter_on_current_python(self):
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

        dependencies = pyproject["project"]["optional-dependencies"]["symbols"]

        tree_sitter_dependencies = [
            dependency for dependency in dependencies if dependency.startswith("tree-sitter")
        ]
        self.assertEqual(
            tree_sitter_dependencies,
            ["tree-sitter>=0.21.0,<0.22", "tree-sitter-languages"],
        )
        self.assertTrue(all("python_version" not in dependency for dependency in tree_sitter_dependencies))


class TestRLMRuntimeAnalyzer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.test_dir) / "docs"
        self.root = Path(self.test_dir) / "repo"
        self.root.mkdir()
        (self.root / "README.md").write_text("# demo\n", encoding="utf-8")
        self.client = MagicMock(spec=BaseLLMClient)
        self.summary_text = _summary_with_required_sections(
            "runtime summary with project components and responsibilities"
        )
        self.summary_text_p1 = "## Major directories\n\n- `isohyps`: runtime implementation."
        self.summary_text_p2 = "## Relationships\n\n```mermaid\ngraph TD\nA --> B\n```\nSome relations."
        self.client.query.side_effect = [
            f"finish({{'summary': {self.summary_text_p1!r}}})",
            f"finish({{'summary': {self.summary_text_p2!r}}})",
            f"finish({{'summary': {self.summary_text!r}, "
            "'documents': [{'path': 'index.md', 'title': 'Root', 'content': 'runtime summary with enough project detail for validation'}, "
            "{'path': 'README.md', 'title': 'README', 'content': 'readme details'}]})"
        ]

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_runtime_analyzer_writes_structured_docs(self):
        analyzer = RLMRuntimeAnalyzer(
            self.client,
            max_depth=2,
            max_steps=2,
            output_dir=self.output_dir,
            backend_name="test",
            model_name="fake-model",
        )

        summary = analyzer.analyze(self.root)

        self.assertEqual(summary, self.summary_text)
        self.assertTrue((self.output_dir / "index.md").exists())
        self.assertTrue((self.output_dir / "README.md").exists())
        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("runtime summary with project components and responsibilities", report)
        self.assertIn("Runtime:** controller", report)

    def test_runtime_analyzer_defaults_to_expanded_controller_budget(self):
        analyzer = RLMRuntimeAnalyzer(self.client)
        self.assertEqual(analyzer.max_depth, 2)
        self.assertEqual(analyzer.max_steps, 30)
        self.assertEqual(analyzer.max_total_tokens, 90000)
        self.assertEqual(analyzer.step_timeout_seconds, 15.0)
        self.assertEqual(analyzer.llm_timeout_seconds, 120.0)

    def test_runtime_analyzer_adds_recovery_guidance(self):
        fake_result = SimpleNamespace(
            status="budget_exceeded",
            error="max_total_tokens=10 reached",
            result=None,
            budget=SimpleNamespace(steps_used=1, llm_calls=1, total_tokens=10),
        )
        fake_controller = MagicMock()
        fake_controller.run.return_value = fake_result

        with patch("isohyps.project_analysis.RLMController", return_value=fake_controller):
            analyzer = RLMRuntimeAnalyzer(self.client)
            summary = analyzer.analyze(self.root)

        self.assertIn("Try increasing --max-total-tokens or --max-steps.", summary)
        self.assertIn("--runtime legacy", summary)

    def test_runtime_analyzer_configures_child_project_analysis_prompt(self):
        fake_controller = MagicMock()
        fake_controller.run.return_value = ControllerResult(
            status="finished",
            result={
                "summary": _summary_with_required_sections(
                    "runtime summary with project components and responsibilities"
                ),
                "documents": [
                    {
                        "path": "README.md",
                        "title": "README",
                        "content": "README source document with enough detail for validation.",
                    }
                ],
            },
            steps=[],
            error=None,
            budget=BudgetSnapshot(steps_used=1, llm_calls=1, prompt_tokens=10, response_tokens=5, total_tokens=15),
            final_state={},
        )

        with patch("isohyps.project_analysis.RLMController", return_value=fake_controller) as controller_cls:
            analyzer = RLMRuntimeAnalyzer(self.client)
            analyzer.analyze(self.root)

        child_config = controller_cls.call_args.kwargs["child_config"]
        self.assertIsInstance(child_config, ChildQueryConfig)
        self.assertIsInstance(child_config.prompt_builder, ProjectAnalysisChildPromptBuilder)
        self.assertEqual(child_config.limits.max_steps, PROJECT_ANALYSIS_CHILD_MAX_STEPS)

    def test_runtime_analyzer_machine_phase(self):
        analyzer = RLMRuntimeAnalyzer(
            self.client,
            output_dir=self.output_dir,
            phase="machine",
        )
        summary = analyzer.analyze(self.root)
        self.assertIn("# Project Machine Analysis Report", summary)
        self.assertIn("## Repo Map Summary", summary)
        self.client.query.assert_not_called()

    def test_runtime_analyzer_abstract_phase(self):
        analyzer = RLMRuntimeAnalyzer(
            self.client,
            output_dir=self.output_dir,
            phase="abstract",
        )
        summary = analyzer.analyze(self.root)
        self.assertIn("isohyps`: runtime implementation", summary)
        self.assertEqual(self.client.query.call_count, 1)

    def test_runtime_analyzer_relation_phase(self):
        analyzer = RLMRuntimeAnalyzer(
            self.client,
            output_dir=self.output_dir,
            phase="relation",
        )
        summary = analyzer.analyze(self.root)
        self.assertIn("Some relations", summary)
        self.assertEqual(self.client.query.call_count, 2)


class TestRLMRuntimeAnalyzerIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.test_dir) / "docs"
        self.root = Path(self.test_dir) / "repo"
        self.root.mkdir()
        (self.root / "app.py").write_text("def hello(): pass\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _make_analyzer(self, responses, max_steps=5, step_timeout_seconds=15.0):
        client = ScriptedClient([WORKER_LLM_DOCUMENT, *responses])
        return RLMRuntimeAnalyzer(
            client,
            output_dir=self.output_dir,
            max_steps=max_steps,
            max_depth=2,
            step_timeout_seconds=step_timeout_seconds,
        ), client

    def test_child_query_integration(self):
        responses = [
            "finish({'summary': '## Major directories\\n\\n- `.`: fixture project root.'})",
            "finish({'summary': '## Relationships\\n\\n- `app.py` is the main file.\\n\\n```mermaid\\ngraph TD;\\napp-->Root\\n```'})",
            "child_result = llm_query('Analyze app.py', {'path': 'app.py'})\n"
            "summary = f'Parent saw: {child_result}; app.py was inspected through a child query.' + "
            f"{REQUIRED_SUMMARY_SECTIONS!r}\n"
            "finish({'summary': summary, "
            "'documents': [{'path': 'index.md', 'title': 'Overview', 'content': f'Parent saw: {child_result}; app.py was inspected through a child query.'}, "
            "{'path': 'app.py.md', 'title': 'app.py', 'content': f'app.py was inspected through a child query. {child_result}'}]})",
            "finish('Child summary of app.py')",
        ]
        analyzer, _ = self._make_analyzer(responses, max_steps=5)

        summary = analyzer.analyze(self.root)

        self.assertIn("Parent saw: Child summary of app.py", summary)
        self.assertTrue((self.output_dir / "index.md").exists())
        self.assertTrue((self.output_dir / "analysis_report.md").exists())

    def test_budget_exceeded_fallback(self):
        analyzer, _ = self._make_analyzer(["x = 1"], max_steps=1)

        summary = analyzer.analyze(self.root)

        self.assertIn("budget_exceeded", summary)
        self.assertIn("--runtime legacy", summary)
        self.assertIn("max_steps=1", summary)

    def test_budget_exceeded_uses_worker_source_documents_before_fallback(self):
        analyzer, _ = self._make_analyzer(["x = 1"], max_steps=1)

        analyzer.analyze(self.root)

        doc = (self.output_dir / "app.py.md").read_text(encoding="utf-8")
        self.assertIn("worker LLM artifact", doc)
        self.assertNotIn("This fallback document was generated", doc)
        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("Fallback docs generated: 0", report)
        self.assertIn("Source files missing matching docs: 0", report)
        self.assertRegex(report, r"\*\*Worker Budget:\*\* \d+ tokens, 1 LLM calls")

    def test_summary_only_finish_uses_worker_source_documents(self):
        summary_text = _summary_with_required_sections(
            "Controller synthesized project-level responsibilities from selected artifacts."
        )
        analyzer, _ = self._make_analyzer(
            [
                "finish({'summary': '## Major directories\\n\\n- `.`: fixture project root.'})",
                "finish({'summary': '## Relationships\\n\\n- `app.py` is main.\\n\\n```mermaid\\ngraph TD;\\napp-->Root\\n```'})",
                f"finish({{'summary': {summary_text!r}}})",
            ],
            max_steps=1,
        )

        summary = analyzer.analyze(self.root)

        self.assertIn("project-level responsibilities", summary)
        doc = (self.output_dir / "app.py.md").read_text(encoding="utf-8")
        self.assertIn("worker LLM artifact", doc)
        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("**Status:** finished", report)
        self.assertIn("**Worker Budget:**", report)
        self.assertIn("**Synthesis RLM Budget:**", report)
        self.assertIn("**Global Budget:**", report)
        self.assertIn("Fallback docs generated: 0", report)
        self.assertIn("Source files missing matching docs: 0", report)

    def test_controller_reads_worker_artifacts_for_synthesis(self):
        analyzer, _ = self._make_analyzer(
            [
                "finish({'summary': '## Major directories\\n\\n- `.`: fixture project root.'})",
                "finish({'summary': '## Relationships\\n\\n- `app.py` is main.\\n\\n```mermaid\\ngraph TD;\\napp-->Root\\n```'})",
                "roots = list_artifacts('.')\n"
                "manifest = read_artifact_json('manifest.json')\n"
                "hit = grep_artifacts('worker LLM artifact')[0]\n"
                "summary = 'Controller synthesized from artifact ' + hit['path'] + ' after reading manifest with source-docs present: ' + str(manifest['documents'][0]['artifact_path'] == 'source-docs/app.py.md') + "
                f"{REQUIRED_SUMMARY_SECTIONS!r}\n"
                "finish({'summary': summary})"
            ],
            max_steps=1,
        )

        summary = analyzer.analyze(self.root)

        self.assertIn("Controller synthesized from artifact source-docs/app.py.md", summary)
        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("**Status:** finished", report)
        self.assertIn("Fallback docs generated: 0", report)

    def test_budget_exceeded_preserves_recorded_source_documents(self):
        responses = [
            "src = read_text('app.py')\n"
            "record_document('app.py.md', 'app.py', 'app.py defines hello and this recorded analysis survives step budget exhaustion. ' + src)"
        ]
        analyzer, _ = self._make_analyzer(responses, max_steps=1)

        summary = analyzer.analyze(self.root)

        self.assertIn("budget_exceeded", summary)
        doc = (self.output_dir / "app.py.md").read_text(encoding="utf-8")
        self.assertIn("recorded analysis survives step budget exhaustion", doc)
        self.assertNotIn("This fallback document was generated", doc)
        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("Fallback docs generated: 0", report)
        self.assertIn("Source files missing matching docs: 0", report)

    def test_recorded_source_documents_satisfy_summary_only_finish(self):
        summary_text = _summary_with_required_sections("Finished with recorded source documents attached by the runtime.")
        responses = [
            "finish({'summary': '## Major directories\\n\\n- `.`: fixture project root.'})",
            "finish({'summary': '## Relationships\\n\\n- `app.py` is main.\\n\\n```mermaid\\ngraph TD;\\napp-->Root\\n```'})",
            "record_document('app.py.md', 'app.py', 'app.py defines hello and is covered by a recorded source document.')\n"
            f"finish({{'summary': {summary_text!r}}})",
        ]
        analyzer, _ = self._make_analyzer(responses, max_steps=2)

        summary = analyzer.analyze(self.root)

        self.assertIn("Finished with recorded source documents", summary)
        doc = (self.output_dir / "app.py.md").read_text(encoding="utf-8")
        self.assertIn("recorded source document", doc)
        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("**Status:** finished", report)
        self.assertIn("Fallback docs generated: 0", report)

    def test_invalid_code_retry(self):
        analyzer, client = self._make_analyzer(
            [
                "This is not python code.",
                "finish({'summary': '## Major directories\\n\\n- `.`: fixture project root.'})",
                "finish({'summary': '## Relationships\\n\\n- `app.py` is main.\\n\\n```mermaid\\ngraph TD;\\napp-->Root\\n```'})",
                "summary = 'Recovered after invalid code and produced a substantive project analysis.' + "
                f"{REQUIRED_SUMMARY_SECTIONS!r}\n"
                "finish({'summary': summary, "
                "'documents': [{'path': 'index.md', 'title': 'Overview', 'content': 'Recovered after invalid code and produced a substantive project analysis.'}, "
                "{'path': 'app.py.md', 'title': 'app.py', 'content': 'app.py has a small hello function and is covered by a source document.'}]})",
            ],
            max_steps=3,
        )

        summary = analyzer.analyze(self.root)

        self.assertIn("Recovered after invalid code", summary)
        self.assertIn("invalid_code", client.prompts[2])


class TestAnalysisDocBuilder(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _make_result(self, status="finished", result=None, error=None):
        return ControllerResult(
            status=status,
            result=result,
            steps=[],
            error=error,
            budget=BudgetSnapshot(steps_used=1, llm_calls=1, prompt_tokens=10, response_tokens=5, total_tokens=15),
            final_state={},
        )

    def test_build_step_history_reports_traceback_cause(self):
        builder = AnalysisDocBuilder(self.output_dir)
        result = self._make_result(result={"summary": "all good", "documents": []})
        result.steps.append(
            ExecutionObservation(
                kind="execution_error",
                stdout="",
                error="Traceback (most recent call last):\n  File '<stdin>', line 1, in <module>\nValueError: boom",
                state={},
                finished=False,
                result=None,
            )
        )

        builder.build(Path(self.temp_dir), result, backend="test", model="fake")

        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("| 1 | execution_error | ERR | ValueError: boom |", report)
        self.assertNotIn("| 1 | execution_error | ERR | Traceback", report)

    def test_build_omits_final_state_from_public_report(self):
        builder = AnalysisDocBuilder(self.output_dir)
        result = self._make_result(result={"summary": "all good", "documents": []})
        result.final_state = {
            "observed_dirs": "list(len=3)",
            "repo_map": "dict {'nodes': list(len=500)}",
        }

        builder.build(Path(self.temp_dir), result, backend="test", model="fake")

        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertNotIn("## Final State", report)
        self.assertNotIn("observed_dirs", report)
        self.assertNotIn("repo_map", report)

    def test_sanitize_path_strips_parent_traversal(self):
        builder = AnalysisDocBuilder(self.output_dir)
        result = builder._sanitize_path("../escape.md")
        self.assertEqual(result, self.output_dir.resolve() / "escape.md")

    def test_sanitize_path_strips_absolute_path(self):
        builder = AnalysisDocBuilder(self.output_dir)
        result = builder._sanitize_path("/etc/passwd")
        self.assertTrue(str(result).startswith(str(self.output_dir.resolve())))
        self.assertNotIn("/etc", str(result.relative_to(self.output_dir.resolve())))

    def test_sanitize_path_strips_multiple_traversals(self):
        builder = AnalysisDocBuilder(self.output_dir)
        result = builder._sanitize_path("foo/../../bar.md")
        self.assertEqual(result, self.output_dir.resolve() / "foo" / "bar.md")

    def test_sanitize_path_strips_windows_parent_traversal(self):
        builder = AnalysisDocBuilder(self.output_dir)
        result = builder._sanitize_path(r"..\escape.md")
        self.assertEqual(result, self.output_dir.resolve() / "escape.md")

    def test_sanitize_path_strips_windows_drive_prefix(self):
        builder = AnalysisDocBuilder(self.output_dir)
        result = builder._sanitize_path(r"C:\temp\escape.md")
        self.assertEqual(result, self.output_dir.resolve() / "temp" / "escape.md")

    def test_sanitize_path_strips_windows_drive_relative_prefix(self):
        builder = AnalysisDocBuilder(self.output_dir)
        result = builder._sanitize_path(r"C:temp\escape.md")
        self.assertEqual(result, self.output_dir.resolve() / "temp" / "escape.md")

    def test_sanitize_path_strips_unc_root_marker(self):
        builder = AnalysisDocBuilder(self.output_dir)
        result = builder._sanitize_path(r"\\server\share\escape.md")
        self.assertEqual(result, self.output_dir.resolve() / "server" / "share" / "escape.md")

    def test_sanitize_path_empty_string_returns_index_md(self):
        builder = AnalysisDocBuilder(self.output_dir)
        result = builder._sanitize_path("")
        self.assertEqual(result, self.output_dir.resolve() / "index.md")

    def test_avoid_collision_adds_suffix(self):
        builder = AnalysisDocBuilder(self.output_dir)
        path = self.output_dir / "doc.md"
        builder._written_paths.add(path)
        result = builder._avoid_collision(path)
        self.assertEqual(result, self.output_dir / "doc_1.md")

    def test_avoid_collision_truncates_long_stem_without_conflict(self):
        builder = AnalysisDocBuilder(self.output_dir)
        long_stem = "a" * 300
        path = self.output_dir / f"{long_stem}.md"
        result = builder._avoid_collision(path)
        self.assertLessEqual(len(result.name), AnalysisDocBuilder.MAX_FILENAME_LENGTH)

    def test_build_creates_index_and_report(self):
        builder = AnalysisDocBuilder(self.output_dir)
        result = self._make_result(result={"summary": "all good", "documents": []})
        builder.build(Path(self.temp_dir), result, backend="test", model="fake")
        self.assertTrue((self.output_dir / "index.md").exists())
        self.assertTrue((self.output_dir / "analysis_report.md").exists())

    def test_build_generates_fallback_docs_for_missing_source_coverage(self):
        root = Path(self.temp_dir) / "repo"
        root.mkdir()
        (root / "app.py").write_text("def hello(): pass\n", encoding="utf-8")
        (root / "README.md").write_text("# demo\n", encoding="utf-8")
        builder = AnalysisDocBuilder(self.output_dir)
        result = self._make_result(result={"summary": "all good", "documents": []})

        builder.build(root, result, backend="test", model="fake")

        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("## Source Coverage", report)
        self.assertIn("Source files discovered: 1", report)
        self.assertIn("Source files with matching docs: 1", report)
        self.assertIn("Source files missing matching docs: 0", report)
        self.assertIn("Fallback docs generated: 1", report)
        self.assertIn("Coverage: 100.0%", report)
        self.assertIn("### Fallback Generated Source Docs", report)
        self.assertIn("- `app.py`", report)
        self.assertNotIn("README.md", report)
        fallback_doc = (self.output_dir / "app.py.md").read_text(encoding="utf-8")
        self.assertIn("# Source: app.py", fallback_doc)
        self.assertIn("def hello", fallback_doc)

    def test_build_counts_source_doc_with_md_suffix_as_covered(self):
        root = Path(self.temp_dir) / "repo"
        root.mkdir()
        (root / "app.py").write_text("def hello(): pass\n", encoding="utf-8")
        builder = AnalysisDocBuilder(self.output_dir)
        result = self._make_result(
            result={
                "summary": "all good",
                "documents": [{"path": "app.md", "title": "App", "content": "app details"}],
            }
        )

        builder.build(root, result, backend="test", model="fake")

        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("Source files discovered: 1", report)
        self.assertIn("Source files with matching docs: 1", report)
        self.assertIn("Source files missing matching docs: 0", report)
        self.assertIn("Fallback docs generated: 0", report)
        self.assertIn("- (none)", report)

    def test_build_appends_md_suffix_to_source_path_document(self):
        root = Path(self.temp_dir) / "repo"
        root.mkdir()
        (root / "app.py").write_text("def hello(): pass\n", encoding="utf-8")
        builder = AnalysisDocBuilder(self.output_dir)
        result = self._make_result(
            result={
                "summary": "all good",
                "documents": [{"path": "app.py", "title": "App", "content": "app details"}],
            }
        )

        builder.build(root, result, backend="test", model="fake")

        self.assertFalse((self.output_dir / "app.py").exists())
        self.assertTrue((self.output_dir / "app.py.md").exists())
        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("Source files with matching docs: 1", report)
        self.assertIn("Fallback docs generated: 0", report)

    def test_build_reports_extra_docs_without_matching_source(self):
        root = Path(self.temp_dir) / "repo"
        root.mkdir()
        (root / "app.py").write_text("def hello(): pass\n", encoding="utf-8")
        builder = AnalysisDocBuilder(self.output_dir)
        result = self._make_result(
            result={
                "summary": "all good",
                "documents": [
                    {"path": "app.py.md", "title": "App", "content": "app details with enough content to pass weak output detection"},
                    {"path": "notes.md", "title": "Notes", "content": "extra details with enough content to pass weak output detection"},
                ],
            }
        )

        builder.build(root, result, backend="test", model="fake")

        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("Extra docs without matching source: 1", report)
        self.assertIn("### Extra Docs Without Matching Source", report)
        self.assertIn("- `notes.md`", report)

    def test_build_reports_weak_or_failed_docs(self):
        root = Path(self.temp_dir) / "repo"
        root.mkdir()
        (root / "app.py").write_text("def hello(): pass\n", encoding="utf-8")
        builder = AnalysisDocBuilder(self.output_dir)
        result = self._make_result(
            result={
                "summary": "all good",
                "documents": [{"path": "app.py.md", "title": "App", "content": ""}],
            }
        )

        builder.build(root, result, backend="test", model="fake")

        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("Weak or failed docs: 1", report)
        self.assertIn("### Weak Or Failed Docs", report)
        self.assertIn("- `app.py.md`", report)

    def test_build_llm_cannot_claim_analysis_report(self):
        builder = AnalysisDocBuilder(self.output_dir)
        result = self._make_result(
            result={
                "summary": "done",
                "documents": [{"path": "analysis_report.md", "title": "My Report", "content": "LLM content"}],
            }
        )
        builder.build(Path(self.temp_dir), result, backend="test", model="fake")
        self.assertTrue((self.output_dir / "analysis_report_1.md").exists())
        report_text = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("Project Analysis Report", report_text)

    def test_build_sanitizes_traversal_path(self):
        builder = AnalysisDocBuilder(self.output_dir)
        outside = Path(self.temp_dir) / "escape.md"
        result = self._make_result(
            result={
                "summary": "done",
                "documents": [{"path": "../escape.md", "title": "Escape", "content": "blocked"}],
            }
        )
        builder.build(Path(self.temp_dir), result, backend="test", model="fake")
        self.assertFalse(outside.exists())
        self.assertTrue((self.output_dir / "escape.md").exists())

    def test_build_budget_exceeded_produces_minimal_output(self):
        builder = AnalysisDocBuilder(self.output_dir)
        result = self._make_result(status="budget_exceeded", error="max_steps=1 reached")
        structured = builder.build(Path(self.temp_dir), result, backend="test", model="fake")
        self.assertIn("[budget_exceeded]", structured.summary)
        self.assertTrue((self.output_dir / "index.md").exists())
        self.assertTrue((self.output_dir / "analysis_report.md").exists())
        index_text = (self.output_dir / "index.md").read_text(encoding="utf-8")
        self.assertIn("Try increasing --max-total-tokens or --max-steps.", index_text)

    def test_build_budget_exceeded_still_generates_source_fallback_docs(self):
        root = Path(self.temp_dir) / "repo"
        root.mkdir()
        (root / "app.py").write_text("def hello(): pass\n", encoding="utf-8")
        builder = AnalysisDocBuilder(self.output_dir)
        result = self._make_result(status="budget_exceeded", error="max_steps=1 reached")

        builder.build(root, result, backend="test", model="fake")

        self.assertTrue((self.output_dir / "app.py.md").exists())
        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("Source files with matching docs: 1", report)
        self.assertIn("Fallback docs generated: 1", report)
        self.assertIn("- `app.py`", report)
        self.assertIn("Coverage: 100.0%", report)

    def test_write_analysis_docs_sanitizes_parent_traversal_paths(self):
        result = self._make_result(
            result={
                "summary": "done",
                "documents": [{"path": "../escape.md", "title": "Escape", "content": "blocked"}],
            }
        )
        outside = Path(self.temp_dir) / "escape.md"

        write_analysis_docs(self.output_dir, Path(self.temp_dir), result, backend="test", model="fake")

        self.assertFalse(outside.exists())
        self.assertTrue((self.output_dir / "escape.md").exists())

    def test_write_analysis_docs_step_history_sanitizes_pipe_chars(self):
        result = ControllerResult(
            status="finished",
            result={"summary": "done"},
            steps=[
                ExecutionObservation(
                    kind="ok",
                    stdout="col1 | col2",
                    error=None,
                    state={},
                    finished=False,
                    result=None,
                )
            ],
            error=None,
            budget=BudgetSnapshot(steps_used=1, llm_calls=1, prompt_tokens=10, response_tokens=5, total_tokens=15),
            final_state={},
        )

        write_analysis_docs(self.output_dir, Path(self.temp_dir), result, backend="test", model="fake")

        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertNotIn("col1 | col2", report)
        self.assertIn("&#124;", report)


if __name__ == "__main__":
    unittest.main()
