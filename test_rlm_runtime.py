import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlm_runtime import (
    BudgetExceededError,
    BudgetLimits,
    CodeResponseValidator,
    ControllerResult,
    BudgetSnapshot,
    ChildQueryConfig,
    IsolatedREPL,
    PartialBudgetLimits,
    PromptBuilder,
    RLMController,
    RunContext,
    validate_analysis_result,
    write_analysis_docs,
)


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def query(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("No more scripted responses left.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class TaggedPromptBuilder(PromptBuilder):
    def __init__(self, tag: str):
        self.tag = tag

    def build(self, goal: str, step: int, max_steps: int, previous: str, parent_context):
        return f"{self.tag}\n" + super().build(goal, step, max_steps, previous, parent_context)


class TestCodeResponseValidator(unittest.TestCase):
    def test_rejects_backend_error_strings(self):
        validator = CodeResponseValidator()
        validated = validator.normalize("[Gemini Error: boom]")
        self.assertEqual(validated.kind, "model_error")

    def test_rejects_non_python_prose(self):
        validator = CodeResponseValidator()
        validated = validator.normalize("Here is the plan: inspect the files first.")
        self.assertEqual(validated.kind, "invalid_code")

    def test_accepts_fenced_python(self):
        validator = CodeResponseValidator()
        validated = validator.normalize("```python\nx = 1\n```")
        self.assertEqual(validated.kind, "code")
        self.assertEqual(validated.code, "x = 1")


class TestRLMRuntime(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text("def greet(name):\n    return f'Hello, {name}'\n", encoding="utf-8")
        (self.root / "README.md").write_text("# demo\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _controller(self, responses, **limits_kwargs):
        client = ScriptedClient(responses)
        prompt_builder = limits_kwargs.pop("prompt_builder", None)
        validator = limits_kwargs.pop("validator", None)
        child_config = limits_kwargs.pop("child_config", None)
        limits = BudgetLimits(max_steps=limits_kwargs.pop("max_steps", 4), max_depth=limits_kwargs.pop("max_depth", 2), **limits_kwargs)
        controller = RLMController(
            client=client,
            root=self.root,
            run_context=RunContext(limits=limits),
            prompt_builder=prompt_builder,
            validator=validator,
            child_config=child_config,
        )
        return controller, client

    def test_controller_keeps_state_across_steps(self):
        controller, client = self._controller(
            [
                "files = list_dir('.')\nprint(files)",
                "source = read_text('src/app.py')\nfinish({'summary': 'done', 'documents': [], 'files': files, 'has_greet': 'greet' in source})",
            ]
        )

        result = controller.run("Inspect the repository and report whether app.py defines greet.")

        self.assertEqual(result.status, "finished")
        self.assertTrue(result.result["has_greet"])
        self.assertEqual(result.result["files"], ["README.md", "src"])
        self.assertIn("files:", client.prompts[1])

    def test_controller_recovers_from_invalid_code_and_execution_error(self):
        controller, client = self._controller(
            [
                "This is not code.",
                "raise ValueError('boom')",
                "finish('recovered')",
            ],
            max_steps=3,
        )

        result = controller.run("Recover after invalid code and execution error.")

        self.assertEqual(result.status, "finished")
        self.assertEqual(result.result, "recovered")
        self.assertIn("invalid_code", client.prompts[1])
        self.assertIn("ValueError: boom", client.prompts[2])

    def test_budget_exceeded_when_controller_never_finishes(self):
        controller, _ = self._controller(["x = 1"], max_steps=1)
        result = controller.run("Never finish.")
        self.assertEqual(result.status, "budget_exceeded")
        self.assertIn("max_steps=1", result.error)

    def test_budget_limits_defaults_match_controller_runtime_defaults(self):
        limits = BudgetLimits()

        self.assertEqual(limits.max_total_tokens, 30000)
        self.assertEqual(limits.step_timeout_seconds, 15.0)

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

    def test_controller_retries_after_invalid_structured_finish(self):
        controller, client = self._controller(
            [
                "finish('not-structured')",
                "finish({'summary': 'fixed', 'documents': []})",
            ],
            max_steps=2,
        )

        result = controller.run("Analyze the repository.", require_structured_finish=True)

        self.assertEqual(result.status, "finished")
        self.assertEqual(result.result["summary"], "fixed")
        self.assertIn("Invalid analysis result format", client.prompts[1])

    def test_finish_stops_following_side_effects(self):
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            observation = repl.execute("finish('done')\nvalue = 1\nprint('after')", lambda prompt, context: None)

        self.assertTrue(observation.finished)
        self.assertEqual(observation.result, "done")
        self.assertNotIn("value", observation.state)
        self.assertNotIn("after", observation.stdout)

    def test_repl_blocks_root_escape(self):
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            observation = repl.execute("read_text('../outside.txt')", lambda prompt, context: None)

        self.assertEqual(observation.kind, "execution_error")
        self.assertIn("escapes root", observation.error)

    def test_extract_symbols_helper_available_without_tree_sitter(self):
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            observation = repl.execute("info = extract_symbols('src/app.py')\nfinish(info['language'])", lambda prompt, context: None)

        self.assertTrue(observation.finished)
        self.assertEqual(observation.result, "python")

    def test_llm_query_returns_child_value(self):
        controller, client = self._controller(
            [
                "child = llm_query('Summarize src/app.py', {'path': 'src/app.py'})\nfinish({'summary': child, 'documents': []})",
                "finish('child-summary')",
            ],
            max_steps=4,
            max_depth=2,
        )

        result = controller.run("Use a child query.")

        self.assertEqual(result.status, "finished")
        self.assertEqual(result.result["summary"], "child-summary")
        self.assertEqual(len(client.prompts), 2)
        self.assertIn("Parent context: dict {'path': 'src/app.py'}", client.prompts[1])

    def test_llm_query_uses_independent_child_step_limit(self):
        controller, _ = self._controller(
            [
                "try:\n    llm_query('Need another step')\nexcept RuntimeError as exc:\n    child_error = str(exc)",
                "x = 1",
                "finish({'summary': child_error, 'documents': []})",
            ],
            max_steps=2,
            child_config=ChildQueryConfig(limits=PartialBudgetLimits(max_steps=1)),
        )

        result = controller.run("Handle a child budget failure.")

        self.assertEqual(result.status, "finished")
        self.assertEqual(result.budget.steps_used, 2)
        self.assertIn("Child query failed (budget_exceeded): max_steps=1 reached", result.result["summary"])

    def test_run_context_accumulates_child_token_usage(self):
        controller, _ = self._controller(
            ["finish('child-summary')"],
            child_config=ChildQueryConfig(limits=PartialBudgetLimits(max_total_tokens=2000)),
        )

        child_value = controller._run_subquery("Summarize src/app.py", {"path": "src/app.py"}, depth=1)

        self.assertEqual(child_value, "child-summary")
        self.assertEqual(controller.run_context.llm_calls, 1)
        self.assertGreater(controller.run_context.total_tokens, 0)

    def test_llm_query_child_tokens_can_trip_parent_budget(self):
        controller, _ = self._controller(
            ["finish('child-summary')"],
            max_total_tokens=1,
            child_config=ChildQueryConfig(limits=PartialBudgetLimits(max_total_tokens=2000)),
        )

        with self.assertRaises(BudgetExceededError):
            controller._run_subquery("Summarize src/app.py", None, depth=1)

        self.assertEqual(controller.run_context.llm_calls, 1)
        self.assertGreater(controller.run_context.total_tokens, controller.run_context.limits.max_total_tokens)

    def test_llm_query_can_override_child_prompt_builder(self):
        child_prompt_builder = TaggedPromptBuilder("CHILD-PROMPT")
        controller, client = self._controller(
            [
                "child = llm_query('Summarize src/app.py')\nfinish({'summary': child, 'documents': []})",
                "finish('child-summary')",
            ],
            child_config=ChildQueryConfig(prompt_builder=child_prompt_builder),
        )

        result = controller.run("Use a child query with a custom prompt.")

        self.assertEqual(result.status, "finished")
        self.assertNotIn("CHILD-PROMPT", client.prompts[0])
        self.assertIn("CHILD-PROMPT", client.prompts[1])

    def test_llm_query_sanitizes_child_failure_details(self):
        controller, _ = self._controller([])
        child_result = ControllerResult(
            status="execution_error",
            result=None,
            steps=[],
            error="Traceback (most recent call last):\n  File '<stdin>', line 1, in <module>\nValueError: boom",
            budget=BudgetSnapshot(
                steps_used=1,
                llm_calls=0,
                prompt_tokens=0,
                response_tokens=0,
                total_tokens=0,
            ),
            final_state={},
        )

        with patch.object(RLMController, "run", return_value=child_result):
            with self.assertRaises(RuntimeError) as exc_info:
                controller._run_subquery("Summarize src/app.py", None, depth=1)

        self.assertEqual(str(exc_info.exception), "Child query failed (execution_error): ValueError: boom")

    def test_llm_query_sanitizes_unexpected_child_exception(self):
        controller, _ = self._controller([])

        with patch.object(RLMController, "run", side_effect=RuntimeError("Traceback\nValueError: noisy detail")):
            with self.assertRaises(RuntimeError) as exc_info:
                controller._run_subquery("Summarize src/app.py", None, depth=1)

        self.assertEqual(
            str(exc_info.exception),
            "Child query failed (execution_error): ValueError: noisy detail",
        )

    def test_write_analysis_docs_sanitizes_parent_traversal_paths(self):
        controller, _ = self._controller(["finish({'summary': 'done', 'documents': []})"])
        result = controller.run("Return a minimal document payload.")
        result.result = {
            "summary": "done",
            "documents": [{"path": "../escape.md", "title": "Escape", "content": "blocked"}],
        }

        output_dir = self.root / "docs"
        outside = self.root / "escape.md"
        write_analysis_docs(output_dir, self.root, result, backend="test", model="fake")

        self.assertFalse(outside.exists())
        self.assertTrue((output_dir / "escape.md").exists())

    def test_write_analysis_docs_marks_nonfinished_runs(self):
        result = ControllerResult(
            status="budget_exceeded",
            result=None,
            steps=[],
            error="max_total_tokens=10 reached",
            budget=BudgetSnapshot(
                steps_used=1,
                llm_calls=1,
                prompt_tokens=8,
                response_tokens=4,
                total_tokens=12,
            ),
            final_state={},
        )

        output_dir = self.root / "docs"
        structured = write_analysis_docs(output_dir, self.root, result, backend="test", model="fake")
        index = (output_dir / "index.md").read_text(encoding="utf-8")
        report = (output_dir / "analysis_report.md").read_text(encoding="utf-8")

        self.assertIn("[budget_exceeded]", structured.summary)
        self.assertIn("Try increasing --max-total-tokens or --max-steps.", index)
        self.assertIn("--runtime legacy", index)
        self.assertIn("**Status:** budget_exceeded", report)

    def test_system_prompt_contains_strict_schema(self):
        from rlm_runtime import PromptBuilder
        prompt = PromptBuilder.SYSTEM_PROMPT
        self.assertIn("'summary': str", prompt)
        self.assertIn("'documents': [", prompt)
        self.assertIn("'path': str", prompt)
        self.assertIn("'title': str", prompt)
        self.assertIn("'content': str", prompt)


if __name__ == "__main__":
    unittest.main()
