import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from isohyps.rlm_runtime import (
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
)
from tests.test_utils import ScriptedClient


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

    def test_rejects_import_statements_before_execution(self):
        validator = CodeResponseValidator()
        validated = validator.normalize("import os\nfinish('done')")
        self.assertEqual(validated.kind, "invalid_code")
        self.assertIn("Import statements are not allowed", validated.error)

    def test_rejects_from_import_statements_before_execution(self):
        validator = CodeResponseValidator()
        validated = validator.normalize("from pathlib import Path\nfinish('done')")
        self.assertEqual(validated.kind, "invalid_code")
        self.assertIn("Import statements are not allowed", validated.error)

    def test_accepts_fenced_python(self):
        validator = CodeResponseValidator()
        validated = validator.normalize("```python\nx = 1\n```")
        self.assertEqual(validated.kind, "code")
        self.assertEqual(validated.code, "x = 1")

    def test_accepts_blockquoted_fenced_python(self):
        validator = CodeResponseValidator()
        validated = validator.normalize("> ```python\n> x = 1\n> ```")
        self.assertEqual(validated.kind, "code")
        self.assertEqual(validated.code, "x = 1")

    def test_accepts_fenced_python_with_surrounding_prose(self):
        validator = CodeResponseValidator()
        validated = validator.normalize("Here is the code:\n```python\nx = 1\n```\nDone.")
        self.assertEqual(validated.kind, "code")
        self.assertEqual(validated.code, "x = 1")

    def test_unwraps_python_code_returned_as_string_literal(self):
        validator = CodeResponseValidator()
        validated = validator.normalize("\"finish('ok')\"")
        self.assertEqual(validated.kind, "code")
        self.assertEqual(validated.code, "finish('ok')")

    def test_unwraps_fenced_python_code_returned_as_string_literal(self):
        validator = CodeResponseValidator()
        validated = validator.normalize("```python\n\"finish('ok')\"\n```")
        self.assertEqual(validated.kind, "code")
        self.assertEqual(validated.code, "finish('ok')")


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

    def test_budget_exceeded_before_oversized_prompt_is_sent(self):
        controller, client = self._controller(
            ["finish('should not be used')"],
            max_steps=8,
            max_total_tokens=100,
        )

        result = controller.run("A" * 2000)

        self.assertEqual(result.status, "budget_exceeded")
        self.assertEqual(len(client.prompts), 0)
        self.assertEqual(result.budget.llm_calls, 0)
        self.assertIn("would be exceeded by prompt", result.error)
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].kind, "model_error")

    def test_budget_exceeded_after_query_stops_without_retrying_same_prompt(self):
        controller, client = self._controller(
            ["x = '" + ("A" * 2000) + "'"],
            max_steps=8,
            max_total_tokens=500,
        )

        result = controller.run("Return a small result.")

        self.assertEqual(result.status, "budget_exceeded")
        self.assertEqual(len(client.prompts), 1)
        self.assertEqual(result.budget.llm_calls, 1)
        self.assertIn("max_total_tokens=500 reached", result.error)
        self.assertEqual(len(result.steps), 1)

    def test_budget_limits_defaults_match_controller_runtime_defaults(self):
        limits = BudgetLimits()

        self.assertEqual(limits.max_total_tokens, 30000)
        self.assertEqual(limits.step_timeout_seconds, 15.0)

    def test_controller_retries_after_invalid_finish_callback(self):
        def validate_string(value):
            return [] if isinstance(value, str) else ["Expected string result."]

        controller, client = self._controller(
            [
                "finish({'not': 'string'})",
                "finish('fixed')",
            ],
            max_steps=2,
        )

        result = controller.run("Return a string result.", finish_validator=validate_string)

        self.assertEqual(result.status, "finished")
        self.assertEqual(result.result, "fixed")
        self.assertIn("Invalid finish result: Expected string result.", client.prompts[1])

    def test_finish_stops_following_side_effects(self):
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            observation = repl.execute("finish('done')\nvalue = 1\nprint('after')", lambda prompt, context: None)

        self.assertTrue(observation.finished)
        self.assertEqual(observation.result, "done")
        self.assertNotIn("value", observation.state)
        self.assertNotIn("after", observation.stdout)

    def test_repl_defines_main_module_name(self):
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            observation = repl.execute(
                "def main():\n    finish('ran-main')\n\nif __name__ == '__main__':\n    main()",
                lambda prompt, context: None,
            )

        self.assertTrue(observation.finished)
        self.assertEqual(observation.result, "ran-main")

    def test_repl_blocks_root_escape(self):
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            observation = repl.execute("read_text('../outside.txt')", lambda prompt, context: None)

        self.assertEqual(observation.kind, "execution_error")
        self.assertIn("escapes root", observation.error)

    def test_repl_allows_root_name_prefix_in_paths(self):
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            observation = repl.execute("finish(list_dir('src'))", lambda prompt, context: None)
            prefixed = repl.execute(f"finish(list_dir('{self.root.name}/src'))", lambda prompt, context: None)

        self.assertTrue(observation.finished)
        self.assertTrue(prefixed.finished)
        self.assertEqual(prefixed.result, observation.result)

    def test_extract_symbols_helper_available_without_tree_sitter(self):
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            observation = repl.execute("info = extract_symbols('src/app.py')\nfinish(info['language'])", lambda prompt, context: None)

        self.assertTrue(observation.finished)
        self.assertEqual(observation.result, "python")

    def test_repo_map_exposes_source_worklist(self):
        (self.root / "src" / "nested").mkdir()
        (self.root / "src" / "nested" / "worker.py").write_text("class Worker:\n    pass\n", encoding="utf-8")
        (self.root / "uv.lock").write_text("# lock\n", encoding="utf-8")

        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            observation = repl.execute("finish(repo_map['source_worklist'])", lambda prompt, context: None)

        self.assertTrue(observation.finished)
        self.assertEqual(observation.result, ["src/app.py", "src/nested/worker.py"])

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

    def test_llm_query_wait_time_does_not_trip_parent_step_timeout(self):
        with IsolatedREPL(self.root, BudgetLimits(step_timeout_seconds=0.2)) as repl:
            observation = repl.execute(
                "child = llm_query('slow child')\nfinish(child)",
                lambda prompt, context: (time.sleep(0.3) or "child-summary"),
            )

        self.assertTrue(observation.finished)
        self.assertEqual(observation.result, "child-summary")

    def test_short_text_can_finish_without_divide_and_conquer(self):
        short_text = "短い文章です。分割せずに要点を返せる入力です。"
        controller, client = self._controller(
            [
                "finish({'strategy': 'direct', 'summary': '短文なので直接要約した'})",
            ],
            max_steps=3,
            max_depth=2,
        )

        result = controller.run(f"Summarize this short text directly when possible:\n{short_text}")

        self.assertEqual(result.status, "finished")
        self.assertEqual(result.result["strategy"], "direct")
        self.assertEqual(result.result["summary"], "短文なので直接要約した")
        self.assertEqual(len(client.prompts), 1)
        self.assertEqual(result.budget.llm_calls, 1)
        self.assertIn(short_text, client.prompts[0])

    def test_very_long_text_can_be_processed_with_divide_and_conquer(self):
        first_chunk = "A" * 5000
        second_chunk = "B" * 5000
        long_text = f"{first_chunk}\n--- SPLIT ---\n{second_chunk}"
        controller, client = self._controller(
            [
                "\n".join(
                    [
                        "left = llm_query('Summarize chunk A', {'chunk_id': 'A', 'text': 'A' * 5000})",
                        "right = llm_query('Summarize chunk B', {'chunk_id': 'B', 'text': 'B' * 5000})",
                        "finish({'strategy': 'divide-and-conquer', 'summary': left + ' / ' + right})",
                    ]
                ),
                "finish('summary-A')",
                "finish('summary-B')",
            ],
            max_steps=6,
            max_depth=2,
        )

        result = controller.run(f"Summarize this very long text by splitting it if useful:\n{long_text}")

        self.assertEqual(result.status, "finished")
        self.assertEqual(result.result["strategy"], "divide-and-conquer")
        self.assertEqual(result.result["summary"], "summary-A / summary-B")
        self.assertEqual(len(client.prompts), 3)
        self.assertEqual(result.budget.llm_calls, 3)
        self.assertGreater(len(client.prompts[0]), 10000)
        self.assertIn("Parent context: dict {'chunk_id': 'A', 'text': str(len=5000)", client.prompts[1])
        self.assertIn("Parent context: dict {'chunk_id': 'B', 'text': str(len=5000)", client.prompts[2])

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

    def test_system_prompt_contains_new_helpers(self):
        prompt = PromptBuilder.SYSTEM_PROMPT
        self.assertIn("path_exists", prompt)
        self.assertIn("is_dir", prompt)
        self.assertIn("read_json", prompt)
        self.assertIn("file_info", prompt)
        self.assertIn("search_text", prompt)
        self.assertIn("repo_map", prompt)

    def test_repl_path_exists_and_is_dir(self):
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            obs = repl.execute(
                "result = {'exists_src': path_exists('src'), 'is_dir_src': is_dir('src'), 'exists_missing': path_exists('missing.txt')}\nfinish(result)",
                lambda prompt, context: None,
            )
        self.assertTrue(obs.finished)
        self.assertTrue(obs.result["exists_src"])
        self.assertTrue(obs.result["is_dir_src"])
        self.assertFalse(obs.result["exists_missing"])

    def test_repl_path_exists_blocks_root_escape(self):
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            obs = repl.execute("path_exists('../outside')", lambda prompt, context: None)
        self.assertEqual(obs.kind, "execution_error")
        self.assertIn("escapes root", obs.error)

    def test_repl_is_dir_blocks_root_escape(self):
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            obs = repl.execute("is_dir('../outside')", lambda prompt, context: None)
        self.assertEqual(obs.kind, "execution_error")
        self.assertIn("escapes root", obs.error)

    def test_repl_read_json(self):
        (self.root / "config.json").write_text('{"key": "value", "num": 42}', encoding="utf-8")
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            obs = repl.execute("data = read_json('config.json')\nfinish(data['key'])", lambda prompt, context: None)
        self.assertTrue(obs.finished)
        self.assertEqual(obs.result, "value")

    def test_repl_read_text_supports_offset_and_capped_limit(self):
        (self.root / "long.txt").write_text("abcdef" * 500, encoding="utf-8")
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            obs = repl.execute(
                "finish({'slice': read_text('long.txt', offset=3, limit=4), 'capped_len': len(read_text('long.txt', limit=3000))})",
                lambda prompt, context: None,
            )

        self.assertTrue(obs.finished)
        self.assertEqual(obs.result["slice"], "defa")
        self.assertEqual(obs.result["capped_len"], 2000)

    def test_repl_file_info_reports_text_metadata(self):
        (self.root / "notes.txt").write_text("one\ntwo\nthree", encoding="utf-8")
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            obs = repl.execute("finish(file_info('notes.txt'))", lambda prompt, context: None)

        self.assertTrue(obs.finished)
        self.assertTrue(obs.result["exists"])
        self.assertTrue(obs.result["is_file"])
        self.assertEqual(obs.result["line_count"], 3)
        self.assertEqual(obs.result["char_count"], len("one\ntwo\nthree"))
        self.assertEqual(obs.result["language"], None)
        self.assertFalse(obs.result["binary"])

    def test_repl_file_info_reports_missing_path(self):
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            obs = repl.execute("finish(file_info('missing.txt'))", lambda prompt, context: None)

        self.assertTrue(obs.finished)
        self.assertFalse(obs.result["exists"])
        self.assertIsNone(obs.result["size_bytes"])

    def test_repl_search_text_returns_offsets_lines_and_excerpts(self):
        (self.root / "rfc.txt").write_text(
            "first stream mention\nsecond line\nanother STREAM mention",
            encoding="utf-8",
        )
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            obs = repl.execute(
                "finish(search_text('rfc.txt', 'stream', max_results=5, context_chars=8))",
                lambda prompt, context: None,
            )

        self.assertTrue(obs.finished)
        self.assertEqual(len(obs.result), 2)
        self.assertEqual(obs.result[0]["line"], 1)
        self.assertEqual(obs.result[0]["match"], "stream")
        self.assertEqual(obs.result[1]["line"], 3)
        self.assertEqual(obs.result[1]["match"], "STREAM")
        self.assertIn("mention", obs.result[1]["excerpt"])

    def test_repl_search_text_blocks_root_escape(self):
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            obs = repl.execute("search_text('../outside.txt', 'x')", lambda prompt, context: None)
        self.assertEqual(obs.kind, "execution_error")
        self.assertIn("escapes root", obs.error)

    def test_repl_read_json_blocks_root_escape(self):
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            obs = repl.execute("read_json('../outside.json')", lambda prompt, context: None)
        self.assertEqual(obs.kind, "execution_error")
        self.assertIn("escapes root", obs.error)

    def test_repl_read_json_rejects_oversized_file(self):
        large_path = self.root / "big.json"
        large_path.write_bytes(b'{"x": "' + b"a" * (2 * 1024 * 1024 + 1) + b'"}')
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            obs = repl.execute("read_json('big.json')", lambda prompt, context: None)
        self.assertEqual(obs.kind, "execution_error")
        self.assertIn("too large", obs.error)

    def test_repl_read_text_rejects_oversized_file(self):
        large_path = self.root / "big.txt"
        large_path.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            obs = repl.execute("read_text('big.txt')", lambda prompt, context: None)
        self.assertEqual(obs.kind, "execution_error")
        self.assertIn("too large", obs.error)

    def test_repl_repo_map_injected(self):
        (self.root / "src").mkdir(exist_ok=True)
        (self.root / "src" / "main.py").write_text("print('hello')", encoding="utf-8")
        (self.root / "README.md").write_text("Hello", encoding="utf-8")
        with IsolatedREPL(self.root, BudgetLimits()) as repl:
            obs = repl.execute(
                "finish(repo_map)", 
                lambda prompt, context: None
            )
        self.assertTrue(obs.finished, f"Execution failed: {obs.error}")
        self.assertIsInstance(obs.result, dict)
        self.assertEqual(obs.result["root"], ".")
        self.assertIn("nodes", obs.result)
        # New schema fields
        self.assertIn("_meta", obs.result)
        self.assertIn("note", obs.result["_meta"])
        self.assertIn("truncated", obs.result)
        self.assertFalse(obs.result["truncated"])
        nodes = obs.result["nodes"]
        paths = [node["path"] for node in nodes]
        self.assertIn("src", paths)
        self.assertIn("src/main.py", paths)
        self.assertIn("README.md", paths)
        
        readme_node = next(n for n in nodes if n["path"] == "README.md")
        self.assertEqual(readme_node["node_type"], "file")
        self.assertEqual(readme_node["category"], "doc")
        
        src_node = next(n for n in nodes if n["path"] == "src")
        self.assertEqual(src_node["node_type"], "dir")
        self.assertEqual(src_node["category"], "code")

    def test_generate_repo_map_truncated_at_max_nodes(self):
        from isohyps.rlm_runtime import _generate_repo_map
        # Create 10 files and set max_nodes=5 to force truncation
        for i in range(10):
            (self.root / f"file_{i:02d}.py").write_text("", encoding="utf-8")
        result = _generate_repo_map(self.root, max_depth=1, max_nodes=5)
        self.assertTrue(result["truncated"])
        self.assertLessEqual(len(result["nodes"]), 5)
        self.assertIn("_meta", result)

    def test_generate_repo_map_no_truncation_within_limit(self):
        from isohyps.rlm_runtime import _generate_repo_map
        (self.root / "a.py").write_text("", encoding="utf-8")
        (self.root / "b.py").write_text("", encoding="utf-8")
        result = _generate_repo_map(self.root, max_depth=1, max_nodes=500)
        self.assertFalse(result["truncated"])

    def test_generate_repo_map_respects_depth_limit_and_ignores_generated_dirs(self):
        from isohyps.rlm_runtime import _generate_repo_map

        (self.root / "src" / "pkg").mkdir(parents=True)
        (self.root / "src" / "pkg" / "deep.py").write_text("", encoding="utf-8")
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules" / "package.js").write_text("", encoding="utf-8")
        (self.root / "build").mkdir()
        (self.root / "build" / "artifact.py").write_text("", encoding="utf-8")

        result = _generate_repo_map(self.root, max_depth=2)
        paths = [node["path"] for node in result["nodes"]]

        self.assertIn("src", paths)
        self.assertIn("src/pkg", paths)
        self.assertNotIn("src/pkg/deep.py", paths)
        self.assertNotIn("node_modules", paths)
        self.assertNotIn("node_modules/package.js", paths)
        self.assertNotIn("build", paths)
        self.assertNotIn("build/artifact.py", paths)

    def test_generate_repo_map_ignores_lockfiles(self):
        from isohyps.rlm_runtime import _generate_repo_map

        (self.root / "uv.lock").write_text("package = []\n", encoding="utf-8")
        (self.root / "package-lock.json").write_text("{}", encoding="utf-8")
        (self.root / "src" / "app.py").write_text("", encoding="utf-8")

        result = _generate_repo_map(self.root, max_depth=2)
        paths = [node["path"] for node in result["nodes"]]

        self.assertIn("src/app.py", paths)
        self.assertNotIn("uv.lock", paths)
        self.assertNotIn("package-lock.json", paths)

    def test_generate_repo_map_excludes_symlinks(self):
        import os
        from isohyps.rlm_runtime import _generate_repo_map
        real_dir = self.root / "real_dir"
        real_dir.mkdir()
        (real_dir / "real_file.py").write_text("", encoding="utf-8")
        link_path = self.root / "link_to_real"
        os.symlink(str(real_dir), str(link_path))
        result = _generate_repo_map(self.root, max_depth=2)
        paths = [n["path"] for n in result["nodes"]]
        self.assertNotIn("link_to_real", paths)
        self.assertIn("real_dir", paths)

    def test_summarize_value_large_list(self):
        from isohyps.rlm_runtime import _summarize_value
        large_list = list(range(200))
        result = _summarize_value(large_list, 160)
        self.assertIn("list(len=200)", result)
        self.assertNotIn(repr(large_list), result)

    def test_summarize_value_large_dict(self):
        from isohyps.rlm_runtime import _summarize_value
        large_dict = {str(i): i for i in range(100)}
        result = _summarize_value(large_dict, 160)
        self.assertIn("dict(len=100)", result)

    def test_summarize_value_nested_large_collection(self):
        from isohyps.rlm_runtime import _summarize_value
        value = {"items": list(range(200))}
        result = _summarize_value(value, 160)
        self.assertIn("dict {'items': list(len=200)", result)
        self.assertNotIn(repr(value), result)

    def test_summarize_value_self_reference(self):
        from isohyps.rlm_runtime import _summarize_value
        value = []
        value.append(value)
        result = _summarize_value(value, 160)
        self.assertIn("list [...]", result)

    def test_summarize_value_large_str(self):
        from isohyps.rlm_runtime import _summarize_value
        long_str = "a" * 500
        result = _summarize_value(long_str, 160)
        self.assertIn("str(len=500)", result)

    def test_sanitize_md_table_cell(self):
        from isohyps.rlm_runtime import _sanitize_md_table_cell
        self.assertEqual(_sanitize_md_table_cell("hello | world"), "hello &#124; world")
        self.assertEqual(_sanitize_md_table_cell("line1\nline2"), "line1 line2")
        self.assertEqual(_sanitize_md_table_cell("ok"), "ok")
        self.assertEqual(_sanitize_md_table_cell("ends\\"), "ends\\\\")


if __name__ == "__main__":
    unittest.main()
