import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import MagicMock

# 機械解析（Level 0）モジュールの各機能を検証します
from isohyps.machine_analysis import (
    analyze_machine_level,
    extract_file_metadata,
    extract_file_symbols,
    build_repo_map_summary,
    detect_attention_points,
)

class TestMachineAnalysis(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.output_dir = Path(tempfile.mkdtemp())

        # テスト用のファイル群を作成
        self.src_dir = self.test_dir / "src"
        self.src_dir.mkdir()
        self.test_files_dir = self.test_dir / "tests"
        self.test_files_dir.mkdir()

        # 1. 普通のPythonソースファイル
        self.runner_file = self.src_dir / "runner.py"
        self.runner_file.write_text(
            "import os\n"
            "import sys\n"
            "from src.config import Config\n"
            "\n"
            "class Runner:\n"
            "    def __init__(self):\n"
            "        self.config = Config()\n"
            "\n"
            "    def run(self):\n"
            "        # TODO: Implement run details\n"
            "        print('running')\n"
            "\n"
            "def main():\n"
            "    r = Runner()\n"
            "    r.run()\n",
            encoding="utf-8"
        )

        # 2. 設定ファイル (config.py) - 複数のファイルからインポートされる想定
        self.config_file = self.src_dir / "config.py"
        self.config_file.write_text(
            "class Config:\n"
            "    def __init__(self):\n"
            "        self.debug = True\n",
            encoding="utf-8"
        )

        # 3. テストファイル
        self.runner_test_file = self.test_files_dir / "test_runner.py"
        self.runner_test_file.write_text(
            "import unittest\n"
            "from src.runner import Runner\n"
            "\n"
            "class TestRunner(unittest.TestCase):\n"
            "    def test_run(self):\n"
            "        r = Runner()\n"
            "        self.assertIsNotNone(r)\n",
            encoding="utf-8"
        )

        # 4. 大きなレガシーファイル (100行以上にして large file 警告をトリガーする)
        self.legacy_file = self.src_dir / "legacy.py"
        large_content = "\n".join([f"line_{i} = {i}" for i in range(150)])
        self.legacy_file.write_text(large_content, encoding="utf-8")

        # 5. pyproject.toml (config/entrypoint)
        self.toml_file = self.test_dir / "pyproject.toml"
        self.toml_file.write_text(
            "[project]\n"
            "name = 'test-project'\n"
            "[project.scripts]\n"
            "test-cli = 'src.runner:main'\n",
            encoding="utf-8"
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.output_dir)

    def test_extract_file_metadata(self):
        # 1ファイルのメタデータ抽出テスト
        meta = extract_file_metadata(self.runner_file, self.test_dir)
        self.assertEqual(meta["path"], "src/runner.py")
        self.assertEqual(meta["language"], "python")
        self.assertEqual(meta["kind"], "source")
        self.assertTrue(len(meta["hash"]) > 0)
        self.assertEqual(meta["size"], self.runner_file.stat().st_size)

        meta_test = extract_file_metadata(self.runner_test_file, self.test_dir)
        self.assertEqual(meta_test["kind"], "test")

        meta_toml = extract_file_metadata(self.toml_file, self.test_dir)
        self.assertEqual(meta_toml["kind"], "config")

    def test_extract_file_symbols(self):
        # シンボル抽出テスト
        symbols_info = extract_file_symbols(self.runner_file, self.test_dir)
        self.assertEqual(symbols_info["path"], "src/runner.py")

        # class Runner, def run, def main の抽出を確認
        symbol_names = [sym["name"] for sym in symbols_info["symbols"]]
        self.assertIn("Runner", symbol_names)
        self.assertIn("Runner.run", symbol_names)
        self.assertIn("main", symbol_names)

        # imports 抽出の確認
        imports = [imp["module"] for imp in symbols_info["imports"]]
        self.assertIn("os", imports)
        self.assertIn("sys", imports)
        self.assertIn("src.config", imports)

        # exports 抽出の確認 (Python のデフォルトは all 以外の public シンボル等)
        self.assertIn("Runner", symbols_info["exports"])
        self.assertIn("main", symbols_info["exports"])

    def test_build_repo_map_summary(self):
        # repo_map サマリー作成のテスト
        files_meta = [
            extract_file_metadata(self.runner_file, self.test_dir),
            extract_file_metadata(self.config_file, self.test_dir),
            extract_file_metadata(self.runner_test_file, self.test_dir),
            extract_file_metadata(self.toml_file, self.test_dir),
        ]
        
        summary = build_repo_map_summary(self.test_dir, files_meta)
        
        # ディレクトリサマリーの確認
        self.assertIn("src", summary["directories"])
        self.assertEqual(summary["directories"]["src"]["files"], 2)
        self.assertIn("python", summary["directories"]["src"]["languages"])

        # エントリポイントの確認
        self.assertIn("pyproject.toml: test-cli -> src.runner:main", summary["entrypoints"])

        # テストファイルの確認
        self.assertIn("tests/test_runner.py", summary["tests"])

    def test_detect_attention_points(self):
        # アテンションポイント（リスクや警告）の検出テスト
        files_meta = [
            extract_file_metadata(self.runner_file, self.test_dir),
            extract_file_metadata(self.config_file, self.test_dir),
            extract_file_metadata(self.legacy_file, self.test_dir),
            extract_file_metadata(self.toml_file, self.test_dir),
        ]
        
        symbols_list = [
            extract_file_symbols(self.runner_file, self.test_dir),
            extract_file_symbols(self.config_file, self.test_dir),
            extract_file_symbols(self.legacy_file, self.test_dir),
            extract_file_symbols(self.toml_file, self.test_dir),
        ]

        attention = detect_attention_points(self.test_dir, files_meta, symbols_list)
        
        # 注意項目の検出を確認
        attention_texts = [att for att in attention]
        
        # 1. legacy.py は 100 行以上のため large file であること
        self.assertTrue(any("large" in text and "legacy.py" in text for text in attention_texts))
        # 2. config.py にはテストがない（config.py に対応する test_config.py がない）
        self.assertTrue(any("no tests" in text and "config.py" in text for text in attention_texts))
        # 3. runner.py に TODO が含まれる
        self.assertTrue(any("TODO/FIXME" in text and "runner.py" in text for text in attention_texts))

    def test_analyze_machine_level(self):
        # level 0 全体プロセスのテスト
        analyze_machine_level(self.test_dir, self.output_dir)
        
        # 出力ファイルの存在確認
        json_path = self.output_dir / "machine_analysis.json"
        yaml_path = self.output_dir / "machine_analysis.yaml"
        report_path = self.output_dir / "machine_report.md"

        self.assertTrue(json_path.exists())
        self.assertTrue(yaml_path.exists())
        self.assertTrue(report_path.exists())

        # JSON の中身の簡易的な検証
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("files", data)
            self.assertIn("repo_map", data)
            self.assertIn("attention", data)

        # YAML の中身の簡易的な検証
        yaml_content = yaml_path.read_text(encoding="utf-8")
        self.assertIn("files:", yaml_content)
        self.assertIn("repo_map:", yaml_content)

        # Markdown レポートの検証
        report_content = report_path.read_text(encoding="utf-8")
        self.assertIn("# Project Machine Analysis Report", report_content)
        self.assertIn("## Repo Map Summary", report_content)
        self.assertIn("## Attention Points", report_content)

    def test_gitignore_filtering(self):
        # .gitignore ファイルの作成
        gitignore_file = self.test_dir / ".gitignore"
        gitignore_file.write_text(
            "# ignore python caches\n"
            "__pycache__/\n"
            "*.pyc\n"
            "# ignore build artifacts\n"
            "dist/\n"
            "build/\n"
            "# ignore test temp directory\n"
            "temp_to_ignore/\n"
            "# neglect complex negation pattern for safety\n"
            "!src/keep_me.py\n",
            encoding="utf-8"
        )
        
        # 除外されるべきフォルダとファイルの作成
        ignored_dir = self.test_dir / "temp_to_ignore"
        ignored_dir.mkdir()
        ignored_file = ignored_dir / "should_be_ignored.py"
        ignored_file.write_text("print('ignored')", encoding="utf-8")
        
        # 除外されるべきキャッシュフォルダの作成
        pytest_cache_dir = self.test_dir / ".pytest_cache"
        pytest_cache_dir.mkdir()
        pytest_cache_file = pytest_cache_dir / "nodeids"
        pytest_cache_file.write_text("nodeid_data", encoding="utf-8")
        
        serena_dir = self.test_dir / ".serena"
        serena_dir.mkdir()
        serena_file = serena_dir / "document_symbols.pkl"
        serena_file.write_text("pickle_data", encoding="utf-8")

        egg_info_dir = self.test_dir / "test_project.egg-info"
        egg_info_dir.mkdir()
        egg_info_file = egg_info_dir / "PKG-INFO"
        egg_info_file.write_text("pkg_info_data", encoding="utf-8")
        
        # 無視されない通常ファイル
        kept_file = self.src_dir / "keep_me.py"
        kept_file.write_text("print('keep')", encoding="utf-8")
        
        result = analyze_machine_level(self.test_dir, self.output_dir)
        files_paths = [f["path"] for f in result["files"]]
        
        # 無視されるべきファイルが含まれていないことを確認
        self.assertNotIn("temp_to_ignore/should_be_ignored.py", files_paths)
        self.assertNotIn(".pytest_cache/nodeids", files_paths)
        self.assertNotIn(".serena/document_symbols.pkl", files_paths)
        self.assertNotIn("test_project.egg-info/PKG-INFO", files_paths)
        
        # 通常ファイルが維持されていることを確認
        self.assertIn("src/keep_me.py", files_paths)

    def test_machine_synthesized_reports(self):
        analyze_machine_level(self.test_dir, self.output_dir)
        
        index_path = self.output_dir / "index.md"
        report_path = self.output_dir / "analysis_report.md"
        
        self.assertTrue(index_path.exists())
        self.assertTrue(report_path.exists())
        
        # analysis_report.md の検証
        report_content = report_path.read_text(encoding="utf-8")
        self.assertIn("Status:** success", report_content)
        self.assertIn("Source Coverage:** 0%", report_content)
        self.assertIn("Backend:** none (machine scan only)", report_content)
        
        # index.md の検証
        index_content = index_path.read_text(encoding="utf-8")
        self.assertIn("Directory: " + self.test_dir.name, index_content)
        self.assertIn("Stale or Newly Added Files", index_content)
        self.assertIn("High Priority Files to Inspect", index_content)

if __name__ == "__main__":
    unittest.main()

