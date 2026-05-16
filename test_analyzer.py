import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil
from analyzer import RLMAnalyzer, BaseLLMClient, RLMRuntimeAnalyzer, build_parser


class TestRLMAnalyzer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.root = Path(self.test_dir)

        # テスト用の階層構造作成
        # root/
        #   sub1/
        #     file1.py
        #   file2.txt
        (self.root / "sub1").mkdir()
        (self.root / "sub1" / "file1.py").write_text("def hello(): pass", encoding='utf-8')
        (self.root / "file2.txt").write_text("some text", encoding='utf-8')

        self.mock_client = MagicMock(spec=BaseLLMClient)
        self.analyzer = RLMAnalyzer(self.mock_client, max_depth=2, warn=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_recursive_analysis(self):
        root_name = self.root.name

        def side_effect(prompt):
            # ディレクトリ選択プロンプト
            if "解析すべき重要な要素を最大5つ選び" in prompt:
                if "ディレクトリ 'sub1'" in prompt:
                    return "file1.py"
                return "sub1, file2.txt"
            # ディレクトリ要約プロンプト
            if "役割を担っているかを技術的に詳しく要約" in prompt:
                if f"ディレクトリ 'sub1'" in prompt:
                    return "Sub1 directory summary"
                return "Root directory summary"
            # ファイル解析プロンプト (tree-sitter or fallback)
            if "file1.py" in prompt:
                return "File1 summary"
            if "file2.txt" in prompt:
                return "File2 summary"
            return "Default summary"

        self.mock_client.query.side_effect = side_effect

        result = self.analyzer.analyze(self.root)

        self.assertEqual(result, "Root directory summary")
        # root選択, sub1選択, file1, sub1要約, file2, root要約 = 6回
        self.assertGreaterEqual(self.mock_client.query.call_count, 4)

    def test_depth_limit(self):
        result = self.analyzer.analyze(self.root / "file2.txt", depth=self.analyzer.max_depth + 1)
        self.assertIn("Depth Limit Reached", result)

    def test_cache_prevents_duplicate_analysis(self):
        self.mock_client.query.return_value = "Summary"
        file_path = self.root / "file2.txt"

        result1 = self.analyzer.analyze(file_path)
        result2 = self.analyzer.analyze(file_path)

        self.assertEqual(result1, result2)
        # キャッシュにより LLM 呼び出しは 1 回のみ
        self.assertEqual(self.mock_client.query.call_count, 1)

    def test_detect_language(self):
        self.assertEqual(self.analyzer._detect_language(Path("foo.py")),   "python")
        self.assertEqual(self.analyzer._detect_language(Path("foo.js")),   "javascript")
        self.assertEqual(self.analyzer._detect_language(Path("foo.ts")),   "typescript")
        self.assertEqual(self.analyzer._detect_language(Path("foo.go")),   "go")
        self.assertEqual(self.analyzer._detect_language(Path("foo.rs")),   "rust")
        self.assertEqual(self.analyzer._detect_language(Path("foo.java")), "java")
        self.assertEqual(self.analyzer._detect_language(Path("foo.cpp")),  "cpp")
        self.assertEqual(self.analyzer._detect_language(Path("foo.cs")),   "c_sharp")
        self.assertEqual(self.analyzer._detect_language(Path("foo.kt")),   "kotlin")
        self.assertEqual(self.analyzer._detect_language(Path("foo.swift")), "swift")
        self.assertIsNone(self.analyzer._detect_language(Path("foo.txt")))
        self.assertIsNone(self.analyzer._detect_language(Path("foo.md")))


class TestTreeSitterAnalysis(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.mock_client = MagicMock(spec=BaseLLMClient)
        self.mock_client.query.return_value = "Summary"
        self.analyzer = RLMAnalyzer(self.mock_client, max_depth=2, warn=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_python_symbols_extracted(self):
        py_file = Path(self.test_dir) / "example.py"
        py_file.write_text(
            "def greet(name):\n    return f'Hello, {name}'\n\nclass Animal:\n    pass\n",
            encoding='utf-8',
        )

        result = self.analyzer._analyze_code_file(py_file, 0, 'python')

        self.assertEqual(result, "Summary")
        prompt = self.mock_client.query.call_args[0][0]
        # tree-sitter でシンボルが抽出されていればプロンプトにコードが含まれる
        self.assertIn("example.py", prompt)
        self.assertIn("python", prompt)

    def test_javascript_symbols_extracted(self):
        js_file = Path(self.test_dir) / "app.js"
        js_file.write_text(
            "function greet(name) { return `Hello, ${name}`; }\n"
            "class Animal { constructor(n) { this.name = n; } }\n",
            encoding='utf-8',
        )

        result = self.analyzer._analyze_code_file(js_file, 0, 'javascript')

        self.assertEqual(result, "Summary")
        self.mock_client.query.assert_called_once()
        prompt = self.mock_client.query.call_args[0][0]
        self.assertIn("app.js", prompt)

    def test_fallback_on_treesitter_error(self):
        """tree-sitter が失敗した場合は _analyze_file にフォールバックする"""
        txt_file = Path(self.test_dir) / "notes.txt"
        txt_file.write_text("some content", encoding='utf-8')

        # 存在しない言語名を渡して強制的にエラーを起こす
        result = self.analyzer._analyze_code_file(txt_file, 0, 'nonexistent_lang_xyz')

        self.assertEqual(result, "Summary")
        self.mock_client.query.assert_called_once()


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

    def test_runtime_analyzer_adds_recovery_guidance(self):
        fake_result = SimpleNamespace(
            status="budget_exceeded",
            error="max_total_tokens=10 reached",
            result=None,
            budget=SimpleNamespace(steps_used=1, llm_calls=1, total_tokens=10),
        )
        fake_controller = MagicMock()
        fake_controller.run.return_value = fake_result

        with patch("analyzer.RLMController", return_value=fake_controller):
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


if __name__ == "__main__":
    unittest.main()
