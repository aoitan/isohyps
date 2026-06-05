import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import rlm_cli


class TestRLMCLI(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        (self.root / "README.md").write_text("# demo\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_scripted_short_text_finishes_directly(self):
        args = rlm_cli.build_parser().parse_args(
            [
                str(self.root),
                "--backend",
                "scripted",
                "--goal",
                "短い文章です。分割せずに要点を返せる入力です。",
                "--scripted-response",
                "finish({'strategy': 'direct', 'summary': '短文なので直接要約した'})",
            ]
        )

        payload = rlm_cli.run(args)

        self.assertEqual(payload["status"], "finished")
        self.assertEqual(payload["result"]["strategy"], "direct")
        self.assertEqual(payload["budget"]["llm_calls"], 1)
        self.assertEqual(len(payload["steps"]), 1)

    def test_scripted_long_text_can_use_child_queries(self):
        first_chunk = "A" * 5000
        second_chunk = "B" * 5000
        long_text = f"{first_chunk}\n--- SPLIT ---\n{second_chunk}"
        parent_response = "\n".join(
            [
                "left = llm_query('Summarize chunk A', {'chunk_id': 'A', 'text': 'A' * 5000})",
                "right = llm_query('Summarize chunk B', {'chunk_id': 'B', 'text': 'B' * 5000})",
                "finish({'strategy': 'divide-and-conquer', 'summary': left + ' / ' + right})",
            ]
        )
        args = rlm_cli.build_parser().parse_args(
            [
                str(self.root),
                "--backend",
                "scripted",
                "--goal",
                f"Summarize this very long text by splitting it if useful:\n{long_text}",
                "--scripted-response",
                parent_response,
                "--scripted-response",
                "finish('summary-A')",
                "--scripted-response",
                "finish('summary-B')",
                "--max-steps",
                "6",
            ]
        )

        payload = rlm_cli.run(args)

        self.assertEqual(payload["status"], "finished")
        self.assertEqual(payload["result"]["strategy"], "divide-and-conquer")
        self.assertEqual(payload["result"]["summary"], "summary-A / summary-B")
        self.assertEqual(payload["budget"]["llm_calls"], 3)

    def test_main_prints_json(self):
        argv = [
            str(self.root),
            "--backend",
            "scripted",
            "--goal",
            "Return ok.",
            "--scripted-response",
            "finish('ok')",
            "--compact",
        ]

        with patch("sys.stdout") as stdout:
            exit_code = rlm_cli.main(argv)

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.write.call_args_list[0].args[0])
        self.assertEqual(payload["status"], "finished")
        self.assertEqual(payload["result"], "ok")

    def test_main_rejects_scripted_backend_without_responses(self):
        exit_code = rlm_cli.main([str(self.root), "--backend", "scripted", "--goal", "Return ok."])

        self.assertEqual(exit_code, 2)

    def test_oversized_goal_stops_before_scripted_llm_call(self):
        args = rlm_cli.build_parser().parse_args(
            [
                str(self.root),
                "--backend",
                "scripted",
                "--goal",
                "A" * 2000,
                "--scripted-response",
                "finish('should not be used')",
                "--max-total-tokens",
                "100",
            ]
        )

        payload = rlm_cli.run(args)

        self.assertEqual(payload["status"], "budget_exceeded")
        self.assertEqual(payload["budget"]["llm_calls"], 0)
        self.assertIn("would be exceeded by prompt", payload["error"])

    def test_rlm_cli_does_not_import_project_analysis(self):
        source = Path(rlm_cli.__file__).read_text(encoding="utf-8")

        self.assertNotIn("project_analysis", source)
        self.assertNotIn("write_analysis_docs", source)
        self.assertNotIn("RLMRuntimeAnalyzer", source)


if __name__ == "__main__":
    unittest.main()
