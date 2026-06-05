"""Integration tests for RLMRuntimeAnalyzer (new controller runtime).

Migrated from test_analyzer.py: TestRLMRuntimeAnalyzer, TestCLIParser.
New integration scenarios: child query, budget exceeded, invalid code retry,
docs fallback generation.
"""
import time
import tempfile
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from analyzer import RLMRuntimeAnalyzer, BaseLLMClient, build_parser
from tests.test_utils import ScriptedClient


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


class TestCLIParser(unittest.TestCase):
    def test_parser_defaults_to_controller_runtime(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.root, ".")
        self.assertEqual(args.runtime, "controller")
        self.assertEqual(args.depth, 2)
        self.assertEqual(args.max_total_tokens, 30000)
        self.assertEqual(args.step_timeout, 15.0)
        self.assertEqual(args.llm_timeout, 120.0)


class TestRLMRuntimeAnalyzerIntegration(unittest.TestCase):
    """Integration scenarios for RLMRuntimeAnalyzer using ScriptedClient."""

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

    # ------------------------------------------------------------------
    # Task 3.1: Child query integration
    # ------------------------------------------------------------------
    def test_child_query_integration(self):
        """Parent issues a child query; child result is reflected in the summary."""
        responses = [
            # Parent step 1: issue child query then finish with the result
            "child_result = llm_query('Analyze app.py', {'path': 'app.py'})\n"
            "finish({'summary': f'Parent saw: {child_result}', 'documents': []})",
            # Child step 1: analyse and finish
            "finish('Child summary of app.py')",
        ]
        analyzer, _ = self._make_analyzer(responses, max_steps=5)

        summary = analyzer.analyze(self.root)

        self.assertEqual(summary, "Parent saw: Child summary of app.py")
        self.assertTrue((self.output_dir / "index.md").exists())
        self.assertTrue((self.output_dir / "analysis_report.md").exists())
        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("Parent saw: Child summary of app.py", report)

    # ------------------------------------------------------------------
    # Task 3.2: Budget exceeded fallback
    # ------------------------------------------------------------------
    def test_budget_exceeded_fallback(self):
        """When max_steps is hit the summary contains budget_exceeded guidance."""
        responses = [
            # Only one response; model never calls finish() so budget is exceeded
            "x = 1",
        ]
        analyzer, _ = self._make_analyzer(responses, max_steps=1)

        summary = analyzer.analyze(self.root)

        self.assertIn("budget_exceeded", summary)
        self.assertIn("--runtime legacy", summary)
        self.assertIn("max_steps=1", summary)

    # ------------------------------------------------------------------
    # Task 3.3: Invalid code retry
    # ------------------------------------------------------------------
    def test_invalid_code_retry(self):
        """Model returns invalid Python first; after retry it finishes normally."""
        responses = [
            "This is not python code.",
            "finish({'summary': 'recovered', 'documents': []})",
        ]
        analyzer, client = self._make_analyzer(responses, max_steps=3)

        summary = analyzer.analyze(self.root)

        self.assertEqual(summary, "recovered")
        # Second prompt must contain feedback about the invalid code
        self.assertIn("invalid_code", client.prompts[1])

    def test_step_timeout_is_reported(self):
        """A timed out sandbox step is surfaced to later controller prompts and reports."""
        responses = [
            "while True:\n    pass",
            "finish({'summary': 'recovered from timeout', 'documents': []})",
        ]
        analyzer, client = self._make_analyzer(
            responses,
            max_steps=3,
            step_timeout_seconds=0.05,
        )

        summary = analyzer.analyze(self.root)

        self.assertIn("budget_exceeded", summary)
        self.assertIn("timed out", client.prompts[1])
        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("timed out", report)

    def test_llm_timeout_turns_into_model_error_and_budget_exit(self):
        class SlowClient:
            def query(self, prompt: str) -> str:
                time.sleep(0.05)
                return "finish({'summary': 'too late', 'documents': []})"

        analyzer = RLMRuntimeAnalyzer(
            SlowClient(),
            output_dir=self.output_dir,
            max_steps=1,
            max_depth=2,
            llm_timeout_seconds=0.01,
        )

        summary = analyzer.analyze(self.root)

        self.assertIn("budget_exceeded", summary)
        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("LLM query timed out", report)

    # ------------------------------------------------------------------
    # Task 3.4: Docs fallback generation
    # ------------------------------------------------------------------
    def test_docs_fallback_generation(self):
        """Even with an empty documents list, index.md and analysis_report.md are generated."""
        responses = [
            "finish({'summary': 'minimal summary', 'documents': []})",
        ]
        analyzer, _ = self._make_analyzer(responses, max_steps=3)

        summary = analyzer.analyze(self.root)

        self.assertEqual(summary, "minimal summary")
        self.assertTrue((self.output_dir / "index.md").exists())
        self.assertTrue((self.output_dir / "analysis_report.md").exists())
        report = (self.output_dir / "analysis_report.md").read_text(encoding="utf-8")
        self.assertIn("minimal summary", report)


if __name__ == "__main__":
    unittest.main()
