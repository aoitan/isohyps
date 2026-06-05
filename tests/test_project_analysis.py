import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from analyzer import BaseLLMClient
from isohyps.project_analysis import (
    AnalysisDocBuilder,
    ProjectAnalysisPromptBuilder,
    RLMRuntimeAnalyzer,
    normalize_analysis_result,
    validate_analysis_result,
    write_analysis_docs,
)
from isohyps.rlm_runtime import BudgetSnapshot, ControllerResult, ExecutionObservation
from tests.test_utils import ScriptedClient


class TestProjectAnalysisContract(unittest.TestCase):
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

    def test_project_analysis_prompt_contains_strict_schema(self):
        prompt = ProjectAnalysisPromptBuilder.SYSTEM_PROMPT
        self.assertIn("'summary': str", prompt)
        self.assertIn("'documents': [", prompt)
        self.assertIn("'path': str", prompt)
        self.assertIn("'title': str", prompt)
        self.assertIn("'content': str", prompt)


class TestRLMRuntimeAnalyzer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.test_dir) / "docs"
        self.root = Path(self.test_dir) / "repo"
        self.root.mkdir()
        (self.root / "README.md").write_text("# demo\n", encoding="utf-8")
        self.client = MagicMock(spec=BaseLLMClient)
        self.client.query.return_value = (
            "finish({'summary': 'runtime summary', "
            "'documents': [{'path': 'index.md', 'title': 'Root', 'content': 'runtime summary'}, "
            "{'path': 'README.md', 'title': 'README', 'content': 'readme details'}]})"
        )

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

        self.assertEqual(summary, "runtime summary")
        self.assertTrue((self.output_dir / "index.md").exists())
        self.assertTrue((self.output_dir / "README.md").exists())
        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("runtime summary", report)
        self.assertIn("Runtime:** controller", report)

    def test_runtime_analyzer_defaults_to_30000_tokens(self):
        analyzer = RLMRuntimeAnalyzer(self.client)
        self.assertEqual(analyzer.max_depth, 2)
        self.assertEqual(analyzer.max_total_tokens, 30000)
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
        client = ScriptedClient(responses)
        return RLMRuntimeAnalyzer(
            client,
            output_dir=self.output_dir,
            max_steps=max_steps,
            max_depth=2,
            step_timeout_seconds=step_timeout_seconds,
        ), client

    def test_child_query_integration(self):
        responses = [
            "child_result = llm_query('Analyze app.py', {'path': 'app.py'})\n"
            "finish({'summary': f'Parent saw: {child_result}', 'documents': []})",
            "finish('Child summary of app.py')",
        ]
        analyzer, _ = self._make_analyzer(responses, max_steps=5)

        summary = analyzer.analyze(self.root)

        self.assertEqual(summary, "Parent saw: Child summary of app.py")
        self.assertTrue((self.output_dir / "index.md").exists())
        self.assertTrue((self.output_dir / "analysis_report.md").exists())

    def test_budget_exceeded_fallback(self):
        analyzer, _ = self._make_analyzer(["x = 1"], max_steps=1)

        summary = analyzer.analyze(self.root)

        self.assertIn("budget_exceeded", summary)
        self.assertIn("--runtime legacy", summary)
        self.assertIn("max_steps=1", summary)

    def test_invalid_code_retry(self):
        analyzer, client = self._make_analyzer(
            [
                "This is not python code.",
                "finish({'summary': 'recovered', 'documents': []})",
            ],
            max_steps=3,
        )

        summary = analyzer.analyze(self.root)

        self.assertEqual(summary, "recovered")
        self.assertIn("invalid_code", client.prompts[1])


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
