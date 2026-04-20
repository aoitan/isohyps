import shutil
import tempfile
import unittest
from pathlib import Path

from rlm_runtime import (
    BudgetLimits,
    CodeResponseValidator,
    IsolatedREPL,
    RLMController,
    RunContext,
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
        limits = BudgetLimits(max_steps=limits_kwargs.pop("max_steps", 4), max_depth=limits_kwargs.pop("max_depth", 2), **limits_kwargs)
        controller = RLMController(client=client, root=self.root, run_context=RunContext(limits=limits))
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


if __name__ == "__main__":
    unittest.main()
