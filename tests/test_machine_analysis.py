import unittest
import tempfile
import shutil
import json
import os
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

# 機械解析（Level 0）モジュールの各機能を検証します
from isohyps.machine_analysis import (
    analyze_machine_level,
    build_attention_snapshots,
    build_doc_status_snapshot,
    extract_file_metadata,
    extract_file_symbols,
    build_repo_map_summary,
    detect_attention_points,
    resolve_attention_entrypoints,
)
from isohyps.machine_index import (
    MACHINE_INDEX_FILE_FIELDS,
    MACHINE_INDEX_SCHEMA_VERSION,
    MACHINE_INDEX_TOP_LEVEL_FIELDS,
    MachineIndexContractError,
    build_machine_index_v1,
    load_machine_index,
    serialize_machine_index,
    validate_machine_index,
)
from isohyps.attention import (
    AttentionContractError,
    AttentionSignalSnapshot,
    ATTENTION_ENTRY_FIELDS,
    ATTENTION_KINDS,
    canonical_attention_bytes,
    classify_attention,
    serialize_attention,
    validate_attention,
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

        # 4. 大きなレガシーファイル (300行以上にして large file 警告をトリガーする)
        self.legacy_file = self.src_dir / "legacy.py"
        large_content = "\n".join([f"line_{i} = {i}" for i in range(350)])
        self.legacy_file.write_text(large_content, encoding="utf-8")

        # 4b. 空の __init__.py (no tests 警告から除外されるべきファイル)
        self.init_file = self.src_dir / "__init__.py"
        self.init_file.write_text("", encoding="utf-8")

        # 5. pyproject.toml (config/entrypoint)
        self.toml_file = self.test_dir / "pyproject.toml"
        self.toml_file.write_text(
            "[project]\n"
            "name = 'test-project'\n"
            "[project.scripts]\n"
            "test-cli = 'src.runner:main'\n",
            encoding="utf-8"
        )

        # 6. 境界値・設定ファイル群 (ノイズ削減テスト用)
        self.conftest_file = self.src_dir / "conftest.py"
        self.conftest_file.write_text("# Test fixtures", encoding="utf-8")

        self.dockerfile_file = self.test_dir / "Dockerfile"
        self.dockerfile_file.write_text("FROM python:3.9", encoding="utf-8")

        self.github_dir = self.test_dir / ".github/workflows"
        self.github_dir.mkdir(parents=True, exist_ok=True)
        self.workflow_file = self.github_dir / "deploy.yml"
        self.workflow_file.write_text("name: deploy", encoding="utf-8")

        self.yaml_config_file = self.src_dir / "settings.yaml"
        self.yaml_config_file.write_text("debug: false", encoding="utf-8")

    @staticmethod
    def _valid_machine_index_fixture():
        """Return a small v1 fixture with a dependency edge and non-source file."""

        return {
            "schema_version": MACHINE_INDEX_SCHEMA_VERSION,
            "files": [
                {
                    "path": "README.md",
                    "hash": "b" * 64,
                    "size": 12,
                    "language": "unknown",
                    "kind": "doc",
                    "public_symbols": [],
                    "internal_symbols": [],
                    "fan_in": 0,
                    "fan_out": 0,
                },
                {
                    "path": "src/app.py",
                    "hash": "a" * 64,
                    "size": 128,
                    "language": "python",
                    "kind": "source",
                    "public_symbols": ["App", "run"],
                    "internal_symbols": ["_helper"],
                    "fan_in": 0,
                    "fan_out": 1,
                },
                {
                    "path": "src/config.py",
                    "hash": "c" * 64,
                    "size": 64,
                    "language": "python",
                    "kind": "source",
                    "public_symbols": ["Config"],
                    "internal_symbols": [],
                    "fan_in": 1,
                    "fan_out": 0,
                },
            ],
            "dependency_graph": {
                "src/app.py": ["src/config.py"],
                "src/config.py": [],
            },
            "dependency_order": ["src/config.py", "src/app.py"],
        }

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
        self.assertEqual(meta["line_count"], len(self.runner_file.read_text(encoding="utf-8").splitlines()))
        self.assertTrue(meta["readable"])

        meta_test = extract_file_metadata(self.runner_test_file, self.test_dir)
        self.assertEqual(meta_test["kind"], "test")

        meta_toml = extract_file_metadata(self.toml_file, self.test_dir)
        self.assertEqual(meta_toml["kind"], "config")

        # 通常ソースだが「test」を含むファイル名の境界値テスト
        helpers_file = self.src_dir / "test_helpers.py"
        helpers_file.write_text("def helper(): pass", encoding="utf-8")
        meta_helpers = extract_file_metadata(helpers_file, self.test_dir)
        self.assertEqual(meta_helpers["kind"], "source")

        tester_file = self.src_dir / "auth_tester.py"
        tester_file.write_text("class AuthTester: pass", encoding="utf-8")
        meta_tester = extract_file_metadata(tester_file, self.test_dir)
        self.assertEqual(meta_tester["kind"], "source")

        # 独自パッケージフォルダ配下のテストヘルパーの境界値テスト
        kuroko_helpers = self.test_dir / "kuroko/test_utils.py"
        kuroko_helpers.parent.mkdir(parents=True, exist_ok=True)
        kuroko_helpers.write_text("def helper(): pass", encoding="utf-8")
        meta_kuroko_helpers = extract_file_metadata(kuroko_helpers, self.test_dir)
        self.assertEqual(meta_kuroko_helpers["kind"], "source")

        # 新しい設定・境界ファイルの判定テスト
        meta_conftest = extract_file_metadata(self.conftest_file, self.test_dir)
        self.assertEqual(meta_conftest["kind"], "config")

        meta_docker = extract_file_metadata(self.dockerfile_file, self.test_dir)
        self.assertEqual(meta_docker["kind"], "config")

        meta_workflow = extract_file_metadata(self.workflow_file, self.test_dir)
        self.assertEqual(meta_workflow["kind"], "config")

        meta_yaml = extract_file_metadata(self.yaml_config_file, self.test_dir)
        self.assertEqual(meta_yaml["kind"], "config")

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
            extract_file_metadata(self.init_file, self.test_dir),
            extract_file_metadata(self.toml_file, self.test_dir),
        ]
        
        symbols_list = [
            extract_file_symbols(self.runner_file, self.test_dir),
            extract_file_symbols(self.config_file, self.test_dir),
            extract_file_symbols(self.legacy_file, self.test_dir),
            extract_file_symbols(self.init_file, self.test_dir),
            extract_file_symbols(self.toml_file, self.test_dir),
        ]

        attention = detect_attention_points(self.test_dir, files_meta, symbols_list)
        
        # 注意項目の検出を確認
        attention_texts = [att for att in attention]
        
        # 1. legacy.py は 300 行以上のため large file であること
        self.assertTrue(any("large" in text and "legacy.py" in text for text in attention_texts))
        # 2. config.py にはテストがない
        self.assertTrue(any("no tests" in text and "config.py" in text for text in attention_texts))
        # 3. runner.py に TODO が含まれる
        self.assertTrue(any("TODO/FIXME" in text and "runner.py" in text for text in attention_texts))
        # 4. __init__.py はテスト不足警告から除外されていること
        self.assertFalse(any("no tests" in text and "__init__.py" in text for text in attention_texts))

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
        # 1回目：ドキュメントなし（カバレッジ0%）の検証
        analyze_machine_level(self.test_dir, self.output_dir)
        
        index_path = self.output_dir / "index.md"
        report_path = self.output_dir / "analysis_report.md"
        
        self.assertTrue(index_path.exists())
        self.assertTrue(report_path.exists())
        
        report_content = report_path.read_text(encoding="utf-8")
        # コントローラー互換フォーマットの検証
        self.assertIn("Status:** finished  \n", report_content)
        self.assertIn("Steps Used:** 0  \n", report_content)
        self.assertIn("Approx Tokens:** 0  \n", report_content)
        self.assertIn("## Source Coverage", report_content)
        self.assertIn("Source files discovered:", report_content)
        self.assertIn("Source files missing matching docs:", report_content)
        
        # 詳細な行末スペースとセクションの完全同期テスト
        self.assertIn("Root Directory:** `" + str(self.test_dir.resolve()) + "`  \n", report_content)
        self.assertIn("### Weak Or Failed Docs\n\n- (none)", report_content)
        self.assertIn("### Extra Docs Without Matching Source\n\n- (none)", report_content)
        
        index_content = index_path.read_text(encoding="utf-8")
        self.assertIn("Directory: " + self.test_dir.name, index_content)
        self.assertIn("Stale or Newly Added Files", index_content)
        # 通常のソースファイルは要説明として検出されること
        self.assertIn("src/runner.py", index_content)
        # テストファイルや設定ファイルは要説明に入っていないこと
        self.assertNotIn("tests/test_runner.py", index_content)
        self.assertNotIn("pyproject.toml", index_content)
        
        # 境界・ノイズ設定ファイルが除外されていることの検証
        self.assertNotIn("conftest.py", index_content)
        self.assertNotIn("Dockerfile", index_content)
        self.assertNotIn("deploy.yml", index_content)
        self.assertNotIn("settings.yaml", index_content)

    def test_coverage_and_stale_detection(self):
        # 2回目：ダミー解説ドキュメントを配置してカバレッジやStaleを検証するテスト
        
        # validなドキュメントを作成 (runner.py.md)
        doc_dir = self.output_dir / "src"
        doc_dir.mkdir(parents=True, exist_ok=True)
        runner_doc = doc_dir / "runner.py.md"
        runner_doc.write_text("# Runner Doc", encoding="utf-8")
        import os
        # ソースコードより新しく更新時刻を設定
        runner_mtime = self.runner_file.stat().st_mtime
        os.utime(runner_doc, (runner_mtime + 10.0, runner_mtime + 10.0))

        # staleなドキュメントを作成 (config.py.md)
        config_doc = doc_dir / "config.py.md"
        config_doc.write_text("# Config Doc", encoding="utf-8")
        config_mtime = self.config_file.stat().st_mtime
        # ソースコードより古く更新時刻を設定
        os.utime(config_doc, (config_mtime - 10.0, config_mtime - 10.0))

        # 再度分析を実行
        analyze_machine_level(self.test_dir, self.output_dir)

        index_path = self.output_dir / "index.md"
        report_path = self.output_dir / "analysis_report.md"
        
        report_content = report_path.read_text(encoding="utf-8")
        # runner.py と config.py にドキュメントが存在するため、
        # カバレッジが 0% より大きくなること
        self.assertNotIn("Coverage: 0.0%", report_content)
        
        index_content = index_path.read_text(encoding="utf-8")
        # config.py は stale としてリストされること
        self.assertTrue(any("config.py" in line and "stale" in line for line in index_content.splitlines()))
        # runner.py は valid なので、git modified などの別の理由がない限り index.md の「要説明」から除外されること
        self.assertFalse(any("src/runner.py" in line and "missing" in line for line in index_content.splitlines()))
        # テストファイルは missing doc と判定されないため、リストされないこと
        self.assertNotIn("tests/test_runner.py", index_content)

    def test_dependency_graph_building(self):
        # 依存関係抽出と Mermaid グラフ生成、トポロジカルソートの統合検証テスト
        # setUp にて runner.py が src.config に依存している
        result = analyze_machine_level(self.test_dir, self.output_dir)
        
        # 1. JSON の出力内容に symbols と imports があり、逆引き解決されていることの検証
        json_path = self.output_dir / "machine_analysis.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("dependency_graph", data)
            dep_graph = data["dependency_graph"]
            # runner.py -> config.py の依存関係
            self.assertIn("src/config.py", dep_graph.get("src/runner.py", []))

        # 2. machine_report.md に Mermaid グラフが出力されていることの検証
        report_path = self.output_dir / "machine_report.md"
        report_content = report_path.read_text(encoding="utf-8")
        self.assertIn("## Dependency Graph Map", report_content)
        self.assertIn("graph TD", report_content)
        # エスケープされたノード名が定義されていること
        self.assertIn("src/runner.py", report_content)
        self.assertIn("src/config.py", report_content)

        # 3. index.md の Stale or Newly Added Files がボトムアップ推奨読解順（config.py が runner.py より先）に並んでいることの検証
        index_path = self.output_dir / "index.md"
        index_content = index_path.read_text(encoding="utf-8")
        lines = index_content.splitlines()
        
        runner_idx = -1
        config_idx = -1
        for i, line in enumerate(lines):
            if "src/runner.py" in line:
                runner_idx = i
            elif "src/config.py" in line:
                config_idx = i
                
        self.assertTrue(config_idx != -1 and runner_idx != -1)
        # config.py は runner.py の依存先なので、先に読むべき（インデックスの上位）
        self.assertTrue(config_idx < runner_idx, f"Expected config.py (idx {config_idx}) to be before runner.py (idx {runner_idx})")

    def test_dependency_cycles_and_determinism(self):
        # 循環参照が存在する場合のハング防止、および決定論的なアルファベット順ソートの検証テスト
        
        # 相互参照するダミーファイルを作成
        cycle_a = self.src_dir / "cycle_a.py"
        cycle_a.write_text("from src.cycle_b import B\nclass A: pass", encoding="utf-8")
        cycle_b = self.src_dir / "cycle_b.py"
        cycle_b.write_text("from src.cycle_a import A\nclass B: pass", encoding="utf-8")
        
        # 無限ループせずに正常終了すること
        result = analyze_machine_level(self.test_dir, self.output_dir)
        
        # JSON内の dependency_graph に循環が記録、あるいは安全に処理されていること
        json_path = self.output_dir / "machine_analysis.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("dependency_graph", data)
            
        # index.md が一意に出力されていることの確認
        index_path = self.output_dir / "index.md"
        index_content = index_path.read_text(encoding="utf-8")
        self.assertIn("src/cycle_a.py", index_content)
        self.assertIn("src/cycle_b.py", index_content)

    def test_mermaid_limit_guard(self):
        # 大規模プロジェクト時の Mermaid ノード数制限ガードの検証テスト
        # 制限（50件）を超えるように55個のダミーソースファイルを作成
        for i in range(55):
            dummy_file = self.src_dir / f"dummy_{i:02d}.py"
            # 決定論的な依存関係を少し作っておく
            if i > 0:
                dummy_file.write_text(f"from src.dummy_{i-1:02d} import X", encoding="utf-8")
            else:
                dummy_file.write_text("pass", encoding="utf-8")
                
        analyze_machine_level(self.test_dir, self.output_dir)
        
        report_path = self.output_dir / "machine_report.md"
        report_content = report_path.read_text(encoding="utf-8")
        
        # Mermaid グラフがスキップされ、注意警告文になっていること
        self.assertIn("Dependency graph is too large to display as a Mermaid diagram", report_content)
        self.assertNotIn("graph TD", report_content)

    def test_git_status_extraction(self):
        # Gitステータス抽出と例外安全、JSON出力の検証
        # ダミーのGit変更ファイルを模擬するため、一時ディレクトリ上にGitリポジトリを初期化してテストすることも可能だが、
        # ここでは get_git_modified_files の戻り値が analyze_machine_level の処理を通じて
        # JSON内の files_meta 各項目の git_status キーに格納されるかを検証。
        result = analyze_machine_level(self.test_dir, self.output_dir)
        json_path = self.output_dir / "machine_analysis.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 各ファイルのメタデータに git_status がデフォルト（"?" や "unchanged" や Noneなど）で含まれていること
            for file_info in data["files"]:
                if file_info["kind"] == "source":
                    self.assertIn("git_status", file_info)

    def test_ast_outline_extraction(self):
        # ASTを用いたクラス・主要関数の抽出と例外安全の検証
        # テスト対象ファイルの1つにダミーのクラスと関数を定義しておく
        dummy_code = (
            "class MyDummyClass:\n"
            "    def method(self):\n"
            "        pass\n"
            "def my_dummy_function():\n"
            "    pass\n"
        )
        dummy_py = self.src_dir / "dummy_ast.py"
        dummy_py.write_text(dummy_code, encoding="utf-8")

        # 構文エラーのファイルも作成して例外安全（クラッシュしないこと）を検証
        bad_py = self.src_dir / "bad_syntax.py"
        bad_py.write_text("class Unfinished:", encoding="utf-8")

        result = analyze_machine_level(self.test_dir, self.output_dir)
        
        json_path = self.output_dir / "machine_analysis.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # files_meta 内の各ファイルに classes / functions フィールドが存在すること
            found_dummy = False
            found_bad = False
            for file_info in data["files"]:
                if file_info["path"] == "src/dummy_ast.py":
                    found_dummy = True
                    self.assertIn("classes", file_info)
                    self.assertIn("functions", file_info)
                    self.assertEqual(file_info["classes"], ["MyDummyClass"])
                    self.assertEqual(file_info["functions"], ["my_dummy_function"])
                elif file_info["path"] == "src/bad_syntax.py":
                    found_bad = True
                    self.assertIn("classes", file_info)
                    self.assertIn("functions", file_info)
                    self.assertEqual(file_info["classes"], [])
                    self.assertEqual(file_info["functions"], [])
            self.assertTrue(found_dummy)
            self.assertTrue(found_bad)

    def test_fan_in_fan_out_metrics(self):
        # ファンイン・ファンアウトメトリクスの算出、JSON保存、およびAttention Pointsへの追加検証
        # setUpにより runner.py が src/config.py をインポートしているため、
        # config.py のファンインは >= 1, runner.py のファンアウトは >= 1 になるはず。
        result = analyze_machine_level(self.test_dir, self.output_dir)
        
        json_path = self.output_dir / "machine_analysis.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # files 各項目の fan_in / fan_out キーの存在と値の検証
            for file_info in data["files"]:
                if file_info["path"] == "src/config.py":
                    self.assertIn("fan_in", file_info)
                    self.assertGreaterEqual(file_info["fan_in"], 1)
                elif file_info["path"] == "src/runner.py":
                    self.assertIn("fan_out", file_info)
                    self.assertGreaterEqual(file_info["fan_out"], 1)

    def test_markdown_noise_reduction_details(self):
        # クラス・主要関数数が閾値以上、または常に index.md 等で折りたたみ（details）構造になっているかの検証
        result = analyze_machine_level(self.test_dir, self.output_dir)
        index_path = self.output_dir / "index.md"
        index_content = index_path.read_text(encoding="utf-8")
        
        # HTMLの details タグによる折りたたみがマークダウンに含まれていること
        self.assertIn("<details>", index_content)
        self.assertIn("</details>", index_content)

    def test_machine_index_json_generation_and_symbol_classification(self):
        # machine_index.json の生成と、public/internal シンボル分類の検証
        # テスト対象ファイルの1つに、公開および内部（_始まり）のクラスと関数を定義しておく
        dummy_code = (
            "class PublicClass:\n"
            "    pass\n"
            "class _InternalClass:\n"
            "    pass\n"
            "def public_function():\n"
            "    pass\n"
            "def _internal_function():\n"
            "    pass\n"
        )
        dummy_py = self.src_dir / "dummy_symbols.py"
        dummy_py.write_text(dummy_code, encoding="utf-8")

        result = analyze_machine_level(self.test_dir, self.output_dir)

        index_json_path = self.output_dir / "machine_index.json"
        self.assertTrue(index_json_path.exists())

        with open(index_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            found_dummy = False
            for file_info in data.get("files", []):
                if file_info["path"] == "src/dummy_symbols.py":
                    found_dummy = True
                    self.assertIn("public_symbols", file_info)
                    self.assertIn("internal_symbols", file_info)
                    
                    # 公開シンボル（先頭が _ でないもの）の検証
                    self.assertIn("PublicClass", file_info["public_symbols"])
                    self.assertIn("public_function", file_info["public_symbols"])
                    self.assertNotIn("_InternalClass", file_info["public_symbols"])
                    self.assertNotIn("_internal_function", file_info["public_symbols"])
                    
                    # 内部シンボル（先頭が _ で始まるもの）の検証
                    self.assertIn("_InternalClass", file_info["internal_symbols"])
                    self.assertIn("_internal_function", file_info["internal_symbols"])
                    self.assertNotIn("PublicClass", file_info["internal_symbols"])
                    self.assertNotIn("public_function", file_info["internal_symbols"])

            self.assertTrue(found_dummy)

    def test_machine_index_preserves_exact_symbol_order(self):
        # classes -> top-level functions の既存抽出順と public/internal 分類を固定する。
        root = self.test_dir / "symbol-order-root"
        source = root / "src" / "symbols.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "class PublicClass:\n"
            "    pass\n"
            "class _InternalClass:\n"
            "    pass\n"
            "def public_first():\n"
            "    pass\n"
            "def _internal_helper():\n"
            "    pass\n"
            "class PublicSecond:\n"
            "    pass\n"
            "def public_second():\n"
            "    pass\n",
            encoding="utf-8",
        )

        analyze_machine_level(root, self.output_dir)
        data = json.loads(
            (self.output_dir / "machine_index.json").read_text(encoding="utf-8")
        )
        entry = next(file for file in data["files"] if file["path"] == "src/symbols.py")

        self.assertEqual(
            entry["public_symbols"],
            ["PublicClass", "PublicSecond", "public_first", "public_second"],
        )
        self.assertEqual(
            entry["internal_symbols"], ["_InternalClass", "_internal_helper"]
        )

    def test_machine_index_preserves_dependency_semantics_for_branch_and_cycle(self):
        # 分岐、同一 source からの複数 edge、cycle を含む isolated fixture を使う。
        root = self.test_dir / "dependency-contract-root"
        files = {
            "src/app.py": (
                "from src.beta import Beta\n"
                "from src.alpha import Alpha\n"
                "class App: pass\n"
            ),
            "src/worker.py": (
                "from src.beta import Beta\n"
                "from src.alpha import Alpha\n"
                "def work(): pass\n"
            ),
            "src/alpha.py": (
                "from src.shared import Shared\n"
                "class Alpha: pass\n"
            ),
            "src/beta.py": (
                "from src.shared import Shared\n"
                "class Beta: pass\n"
            ),
            "src/shared.py": "class Shared: pass\n",
            "src/cycle_a.py": (
                "from src.cycle_b import B\n"
                "class A: pass\n"
            ),
            "src/cycle_b.py": (
                "from src.cycle_a import A\n"
                "class B: pass\n"
            ),
        }
        for relative_path, content in files.items():
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        output = self.output_dir / "dependency-contract"
        analyze_machine_level(root, output)
        index_path = output / "machine_index.json"
        data = json.loads(index_path.read_text(encoding="utf-8"))
        first_bytes = index_path.read_bytes()

        self.assertEqual(
            data["dependency_graph"],
            {
                "src/alpha.py": ["src/shared.py"],
                "src/app.py": ["src/alpha.py", "src/beta.py"],
                "src/beta.py": ["src/shared.py"],
                "src/cycle_a.py": ["src/cycle_b.py"],
                "src/cycle_b.py": ["src/cycle_a.py"],
                "src/shared.py": [],
                "src/worker.py": ["src/alpha.py", "src/beta.py"],
            },
        )
        self.assertEqual(
            data["dependency_order"],
            [
                "src/shared.py",
                "src/alpha.py",
                "src/beta.py",
                "src/app.py",
                "src/worker.py",
                "src/cycle_a.py",
                "src/cycle_b.py",
            ],
        )
        self.assertEqual(
            {
                file["path"]: (file["fan_in"], file["fan_out"])
                for file in data["files"]
            },
            {
                "src/alpha.py": (2, 1),
                "src/app.py": (0, 2),
                "src/beta.py": (2, 1),
                "src/cycle_a.py": (1, 1),
                "src/cycle_b.py": (1, 1),
                "src/shared.py": (2, 0),
                "src/worker.py": (0, 2),
            },
        )

        analyze_machine_level(root, output)
        self.assertEqual(first_bytes, index_path.read_bytes())

    def test_machine_index_is_byte_stable_across_prior_analysis_and_stale_docs(self):
        # 初回の added/stale 状態と、前回解析を読んだ2回目の unchanged 状態を跨いでも、
        # history-aware な内部 result が公開 index の bytes を変えないことを確認する。
        root = self.test_dir / "repeat-scan-root"
        source = root / "src" / "app.py"
        source.parent.mkdir(parents=True)
        source.write_text("def app():\n    return 'stable'\n", encoding="utf-8")

        doc = self.output_dir / "src" / "app.py.md"
        doc.parent.mkdir(parents=True)
        doc.write_text("# App", encoding="utf-8")
        source_mtime = source.stat().st_mtime
        os.utime(doc, (source_mtime - 10.0, source_mtime - 10.0))

        first_result = analyze_machine_level(root, self.output_dir)
        index_path = self.output_dir / "machine_index.json"
        first_bytes = index_path.read_bytes()
        first_file = next(
            file for file in first_result["files"] if file["path"] == "src/app.py"
        )
        self.assertNotEqual(first_file["status"], "unchanged")
        self.assertGreater(first_result["coverage_summary"]["stale_docs"], 0)

        second_result = analyze_machine_level(root, self.output_dir)
        second_bytes = index_path.read_bytes()
        second_file = next(
            file for file in second_result["files"] if file["path"] == "src/app.py"
        )

        self.assertEqual(second_file["status"], "unchanged")
        self.assertNotEqual(first_file["status"], second_file["status"])
        # v2 includes normalized history/doc attention.  The first snapshot is
        # stale while the second is current, so the public bytes intentionally
        # differ; determinism is guaranteed for an identical analysis snapshot.
        self.assertNotEqual(first_bytes, second_bytes)
        first_index = json.loads(first_bytes)
        second_index = json.loads(second_bytes)
        self.assertEqual(first_index["schema_version"], "2.0")
        self.assertTrue(any(entry["kind"] == "doc_stale" for entry in first_index["attention"]))
        self.assertFalse(any(entry["kind"] == "doc_stale" for entry in second_index["attention"]))

    def test_machine_index_is_byte_stable_when_files_are_created_in_different_orders(self):
        logical_files = [
            (
                "src/root.py",
                "from src.zed import Zed\nfrom src.alpha import Alpha\n",
            ),
            ("src/zed.py", "class Zed: pass\n"),
            ("src/alpha.py", "class Alpha: pass\n"),
            ("README.md", "# Fixture\n"),
        ]
        outputs = []

        for suffix, creation_order in (
            ("forward", logical_files),
            ("reverse", list(reversed(logical_files))),
        ):
            root = self.test_dir / f"creation-order-{suffix}"
            for relative_path, content in creation_order:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            output = self.output_dir / f"creation-order-{suffix}"
            analyze_machine_level(root, output)
            outputs.append((output / "machine_index.json").read_bytes())

        self.assertEqual(outputs[0], outputs[1])

    def test_machine_scan_excludes_root_contained_output_and_rejects_same_root(self):
        root = self.test_dir / "contained-output-root"
        source = root / "src" / "app.py"
        source.parent.mkdir(parents=True)
        source.write_text("def app():\n    return 1\n", encoding="utf-8")

        contained_output = root / "analysis"
        (contained_output / "nested").mkdir(parents=True)
        (contained_output / "ghost.py").write_text("def ghost(): pass\n", encoding="utf-8")
        (contained_output / "nested" / "ghost.py").write_text(
            "def nested_ghost(): pass\n", encoding="utf-8"
        )

        first_result = analyze_machine_level(root, contained_output)
        second_result = analyze_machine_level(root, contained_output)
        for result in (first_result, second_result):
            paths = [file["path"] for file in result["files"]]
            self.assertEqual(paths, ["src/app.py"])
            self.assertNotIn("analysis/ghost.py", paths)
            self.assertNotIn("analysis/nested/ghost.py", paths)

            public_data = json.loads(
                (contained_output / "machine_index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [file["path"] for file in public_data["files"]], ["src/app.py"]
            )

        same_root = self.test_dir / "same-root-output"
        same_root.mkdir()
        (same_root / "source.py").write_text("value = 1\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            analyze_machine_level(same_root, same_root)
        self.assertFalse((same_root / "machine_analysis.json").exists())

    def test_machine_index_schema_matches_python_contract(self):
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "schemas"
            / "machine-index.schema.json"
        )
        with schema_path.open(encoding="utf-8") as handle:
            schema = json.load(handle)

        self.assertEqual(
            schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
        )
        self.assertEqual(
            schema["required"], list(MACHINE_INDEX_TOP_LEVEL_FIELDS)
        )
        self.assertTrue(schema["additionalProperties"])

        top_level_types = {
            "schema_version": "string",
            "files": "array",
            "dependency_graph": "object",
            "dependency_order": "array",
        }
        for field, expected_type in top_level_types.items():
            with self.subTest(top_level_field=field):
                self.assertEqual(schema["properties"][field]["type"], expected_type)

        file_schema = schema["$defs"]["fileEntry"]
        self.assertEqual(file_schema["required"], list(MACHINE_INDEX_FILE_FIELDS))
        self.assertTrue(file_schema["additionalProperties"])

        file_types = {
            "hash": "string",
            "size": "integer",
            "language": "string",
            "kind": "string",
            "public_symbols": "array",
            "internal_symbols": "array",
            "fan_in": "integer",
            "fan_out": "integer",
        }
        for field, expected_type in file_types.items():
            with self.subTest(file_field=field):
                self.assertEqual(
                    file_schema["properties"][field]["type"], expected_type
                )
        self.assertEqual(
            file_schema["properties"]["path"]["$ref"],
            "#/$defs/repositoryRelativePath",
        )

    def test_machine_index_minor_versions_and_unknown_fields_are_compatible(self):
        fixture = self._valid_machine_index_fixture()
        fixture["schema_version"] = "1.7"
        fixture["future_top_level"] = {"owner": "downstream"}
        fixture["files"][1]["future_file_field"] = ["kept for a newer reader"]

        index_path = self.output_dir / "machine-index-v1-minor.json"
        index_path.write_text(serialize_machine_index(fixture), encoding="utf-8")
        loaded = load_machine_index(index_path)

        self.assertEqual(loaded["schema_version"], "1.7")
        self.assertEqual(loaded["future_top_level"], {"owner": "downstream"})
        self.assertEqual(
            loaded["files"][1]["future_file_field"], ["kept for a newer reader"]
        )
        self.assertEqual(
            {field: loaded[field] for field in MACHINE_INDEX_TOP_LEVEL_FIELDS},
            {field: fixture[field] for field in MACHINE_INDEX_TOP_LEVEL_FIELDS},
        )
        for loaded_entry, expected_entry in zip(loaded["files"], fixture["files"]):
            self.assertEqual(
                {field: loaded_entry[field] for field in MACHINE_INDEX_FILE_FIELDS},
                {field: expected_entry[field] for field in MACHINE_INDEX_FILE_FIELDS},
            )

    def test_machine_index_rejects_invalid_versions_required_fields_and_types(self):
        fixture = self._valid_machine_index_fixture()
        invalid_cases = []

        malformed_version = deepcopy(fixture)
        malformed_version["schema_version"] = "1"
        invalid_cases.append(("malformed version", malformed_version, "schema_version"))

        non_string_version = deepcopy(fixture)
        non_string_version["schema_version"] = 1.0
        invalid_cases.append(("version type", non_string_version, "schema_version"))

        unknown_major = deepcopy(fixture)
        unknown_major["schema_version"] = "2.0"
        invalid_cases.append(("unknown major", unknown_major, "schema_version"))

        missing_top_level = deepcopy(fixture)
        del missing_top_level["files"]
        invalid_cases.append(("missing top-level field", missing_top_level, "root.files"))

        missing_file_field = deepcopy(fixture)
        del missing_file_field["files"][1]["hash"]
        invalid_cases.append(("missing file field", missing_file_field, "files[1].hash"))

        wrong_file_type = deepcopy(fixture)
        wrong_file_type["files"][1]["size"] = True
        invalid_cases.append(("file field type", wrong_file_type, "files[1].size"))

        wrong_nested_type = deepcopy(fixture)
        wrong_nested_type["files"][1]["public_symbols"] = ["App", 3]
        invalid_cases.append(
            ("symbol item type", wrong_nested_type, "files[1].public_symbols[1]")
        )

        wrong_graph_type = deepcopy(fixture)
        wrong_graph_type["dependency_graph"] = []
        invalid_cases.append(("graph type", wrong_graph_type, "dependency_graph"))

        for label, data, location in invalid_cases:
            with self.subTest(case=label):
                with self.assertRaises(MachineIndexContractError) as context:
                    validate_machine_index(data)
                self.assertIn(location, str(context.exception))

    def test_machine_index_rejects_invalid_and_duplicate_paths(self):
        invalid_paths = [
            ("absolute path", "/README.md", "files[0].path"),
            ("traversal path", "../README.md", "files[0].path"),
            ("empty path segment", "src//app.py", "files[1].path"),
            ("windows path", r"src\\app.py", "files[1].path"),
        ]

        for label, path, location in invalid_paths:
            with self.subTest(case=label):
                data = self._valid_machine_index_fixture()
                target_index = 0 if location == "files[0].path" else 1
                data["files"][target_index]["path"] = path
                with self.assertRaises(MachineIndexContractError) as context:
                    validate_machine_index(data)
                self.assertIn(location, str(context.exception))

        duplicate_path = self._valid_machine_index_fixture()
        duplicate_path["files"][1]["path"] = duplicate_path["files"][0]["path"]
        with self.assertRaises(MachineIndexContractError) as context:
            validate_machine_index(duplicate_path)
        self.assertIn("files[1].path", str(context.exception))

        invalid_dependency_path = self._valid_machine_index_fixture()
        invalid_dependency_path["dependency_graph"]["src/app.py"] = [
            "../src/config.py"
        ]
        with self.assertRaises(MachineIndexContractError) as context:
            validate_machine_index(invalid_dependency_path)
        self.assertIn("dependency_graph['src/app.py'][0]", str(context.exception))

    def test_machine_index_rejects_graph_order_and_fan_invariant_violations(self):
        invalid_cases = []

        missing_graph_key = self._valid_machine_index_fixture()
        del missing_graph_key["dependency_graph"]["src/config.py"]
        invalid_cases.append(("missing graph key", missing_graph_key, "dependency_graph"))

        unknown_dependency = self._valid_machine_index_fixture()
        unknown_dependency["dependency_graph"]["src/app.py"] = ["src/missing.py"]
        invalid_cases.append(
            (
                "unknown dependency",
                unknown_dependency,
                "dependency_graph['src/app.py'][0]",
            )
        )

        wrong_order = self._valid_machine_index_fixture()
        wrong_order["dependency_order"] = ["src/app.py", "src/config.py"]
        invalid_cases.append(("dependency order", wrong_order, "dependency_order"))

        wrong_fan_out = self._valid_machine_index_fixture()
        wrong_fan_out["files"][1]["fan_out"] = 0
        invalid_cases.append(("fan out", wrong_fan_out, "files[1].fan_out"))

        wrong_fan_in = self._valid_machine_index_fixture()
        wrong_fan_in["files"][2]["fan_in"] = 0
        invalid_cases.append(("fan in", wrong_fan_in, "files[2].fan_in"))

        cycle_with_wrong_fallback = {
            "schema_version": "1.0",
            "files": [
                {
                    "path": path,
                    "hash": "a" * 64,
                    "size": 1,
                    "language": "python",
                    "kind": "source",
                    "public_symbols": [],
                    "internal_symbols": [],
                    "fan_in": 1,
                    "fan_out": 1,
                }
                for path in ("a.py", "b.py")
            ],
            "dependency_graph": {"a.py": ["b.py"], "b.py": ["a.py"]},
            "dependency_order": ["b.py", "a.py"],
        }
        invalid_cases.append(
            ("cycle fallback order", cycle_with_wrong_fallback, "dependency_order")
        )

        for label, data, location in invalid_cases:
            with self.subTest(case=label):
                with self.assertRaises(MachineIndexContractError) as context:
                    validate_machine_index(data)
                self.assertIn(location, str(context.exception))

    def test_machine_index_projection_is_allowlisted_and_does_not_mutate_analysis(self):
        analysis = self._valid_machine_index_fixture()
        analysis.pop("schema_version")
        analysis["attention"] = ["history-dependent diagnostic"]
        analysis["coverage"] = {"stale_docs": 1}
        analysis["files"][1]["status"] = "added"
        analysis["files"][1]["mtime"] = 123.0
        original_analysis = deepcopy(analysis)

        projected = build_machine_index_v1(analysis)

        self.assertEqual(analysis, original_analysis)
        self.assertEqual(set(projected), set(MACHINE_INDEX_TOP_LEVEL_FIELDS))
        for entry in projected["files"]:
            self.assertEqual(set(entry), set(MACHINE_INDEX_FILE_FIELDS))
        self.assertNotIn("attention", projected)
        self.assertNotIn("coverage", projected)
        self.assertNotIn("status", projected["files"][1])
        self.assertEqual(projected["schema_version"], MACHINE_INDEX_SCHEMA_VERSION)


class TestAttentionSnapshotBuilder(unittest.TestCase):
    @staticmethod
    def metadata(path: str, **overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "path": path,
            "hash": "a" * 64,
            "size": 100,
            "language": "python",
            "kind": "source",
            "status": "changed",
            "todo_count": 0,
            "line_count": 10,
            "readable": True,
            "fan_in": 0,
            "fan_out": 0,
            "mtime": 100,
        }
        values.update(overrides)
        return values

    def test_builds_one_sorted_snapshot_with_all_attention_signals(self):
        files_meta = [
            self.metadata(
                "src/app.py",
                line_count=301,
                todo_count=2,
                fan_in=5,
                fan_out=15,
            ),
            self.metadata("src/untested.py", size=100),
            self.metadata("tests/test_app.py", kind="test", language="python"),
            self.metadata("README.md", kind="doc", language="unknown"),
        ]

        snapshots, diagnostics = build_attention_snapshots(
            None,
            list(reversed(files_meta)),
            {"src/app.py": {"todo_count": 1}},
            doc_status_by_path={
                "src/app.py": "missing",
                "src/untested.py": "stale",
            },
            entrypoint_paths={"src/app.py"},
        )

        self.assertEqual(diagnostics, [])
        self.assertEqual([snapshot.path for snapshot in snapshots], [
            "README.md",
            "src/app.py",
            "src/untested.py",
            "tests/test_app.py",
        ])
        by_path = {snapshot.path: snapshot for snapshot in snapshots}

        app = by_path["src/app.py"]
        self.assertEqual(app.line_count, 301)
        self.assertEqual(app.fan_in, 5)
        self.assertEqual(app.fan_out, 15)
        self.assertFalse(app.test_missing)
        self.assertEqual(app.todo_current, 2)
        self.assertEqual(app.todo_previous, 1)
        self.assertTrue(app.todo_increased)
        self.assertTrue(app.is_entrypoint)
        self.assertEqual(app.doc_status, "missing")

        untested = by_path["src/untested.py"]
        self.assertTrue(untested.test_missing)
        self.assertEqual(untested.doc_status, "stale")

    def test_only_kind_test_paths_are_used_for_test_lookup(self):
        source_files = [
            self.metadata("src/widget.py"),
            # A source-shaped filename must not count as a test candidate.
            self.metadata("src/test_widget.py"),
        ]
        snapshots, diagnostics = build_attention_snapshots(None, source_files)
        self.assertEqual(diagnostics, [])
        self.assertTrue(next(s for s in snapshots if s.path == "src/widget.py").test_missing)

        test_files = source_files + [
            self.metadata("tests/test_widget.py", kind="test"),
        ]
        snapshots, diagnostics = build_attention_snapshots(None, test_files)
        self.assertEqual(diagnostics, [])
        self.assertFalse(next(s for s in snapshots if s.path == "src/widget.py").test_missing)

    def test_todo_previous_count_and_first_snapshot_boundaries(self):
        files_meta = [
            self.metadata("src/increased.py", todo_count=2),
            self.metadata("src/unchanged.py", todo_count=1),
            self.metadata("src/decreased.py", todo_count=1),
            self.metadata("src/first.py", todo_count=1),
            self.metadata("src/empty-first.py", todo_count=0),
        ]
        previous = {
            "src/increased.py": {"todo_count": 1},
            "src/unchanged.py": {"todo_count": 1},
            "src/decreased.py": {"todo_count": 2},
        }

        snapshots, diagnostics = build_attention_snapshots(None, files_meta, previous)
        self.assertEqual(diagnostics, [])
        by_path = {snapshot.path: snapshot for snapshot in snapshots}
        self.assertTrue(by_path["src/increased.py"].todo_increased)
        self.assertFalse(by_path["src/unchanged.py"].todo_increased)
        self.assertFalse(by_path["src/decreased.py"].todo_increased)
        self.assertTrue(by_path["src/first.py"].todo_increased)
        self.assertFalse(by_path["src/empty-first.py"].todo_increased)
        self.assertIsNone(by_path["src/first.py"].todo_previous)

    def test_unavailable_previous_todo_baselines_do_not_trigger_first_snapshot(self):
        files_meta = [
            self.metadata("src/invalid-previous.py", todo_count=1, size=40),
            self.metadata("src/missing-previous.py", todo_count=1, size=40),
        ]
        previous = {
            "src/invalid-previous.py": {"todo_count": "bad"},
            "src/missing-previous.py": {},
        }

        snapshots, diagnostics = build_attention_snapshots(None, files_meta, previous)
        by_path = {snapshot.path: snapshot for snapshot in snapshots}

        self.assertFalse(by_path["src/invalid-previous.py"].todo_increased)
        self.assertFalse(by_path["src/missing-previous.py"].todo_increased)
        self.assertEqual(
            {(diagnostic.detector, diagnostic.code, diagnostic.path) for diagnostic in diagnostics},
            {
                ("metadata", "invalid_todo_count", "src/invalid-previous.py"),
                (
                    "metadata",
                    "previous_todo_count_unavailable",
                    "src/missing-previous.py",
                ),
            },
        )
        self.assertEqual(classify_attention(snapshots), [])

    def test_missing_graph_metrics_are_diagnostics_and_valid_zero_is_preserved(self):
        missing_fan_in = self.metadata("src/missing-in.py", size=40)
        missing_fan_in.pop("fan_in")
        null_fan_out = self.metadata("src/null-out.py", size=40, fan_out=None)
        valid_zero = self.metadata("src/zero.py", size=40, fan_in=0, fan_out=0)

        snapshots, diagnostics = build_attention_snapshots(
            None, [missing_fan_in, null_fan_out, valid_zero]
        )
        by_path = {snapshot.path: snapshot for snapshot in snapshots}

        self.assertEqual(set(by_path), {"src/zero.py"})
        self.assertEqual(by_path["src/zero.py"].fan_in, 0)
        self.assertEqual(by_path["src/zero.py"].fan_out, 0)
        self.assertEqual(
            {(diagnostic.detector, diagnostic.code, diagnostic.path) for diagnostic in diagnostics},
            {
                ("metadata", "fan_in_unavailable", "src/missing-in.py"),
                ("metadata", "fan_out_unavailable", "src/null-out.py"),
            },
        )

    def test_doc_status_snapshot_keeps_missing_current_stale_and_two_second_boundary(self):
        root = Path(tempfile.mkdtemp())
        output = Path(tempfile.mkdtemp())
        try:
            files_meta = [
                self.metadata("src/exact.py", mtime=100),
                self.metadata("src/stale.py", mtime=100),
                self.metadata("src/missing.py", mtime=100),
                self.metadata("src/unchanged.py", mtime=100, status="unchanged"),
            ]
            for path, mtime in (
                ("src/exact.py.md", 98.0),
                ("src/stale.py.md", 97.99),
                ("src/unchanged.py.md", 0.0),
            ):
                doc = output / path
                doc.parent.mkdir(parents=True, exist_ok=True)
                doc.write_text("# doc", encoding="utf-8")
                os.utime(doc, (mtime, mtime))

            statuses, diagnostics = build_doc_status_snapshot(root, files_meta, output)
            self.assertEqual(diagnostics, [])
            self.assertEqual(statuses["src/exact.py"], "current")
            self.assertEqual(statuses["src/stale.py"], "stale")
            self.assertEqual(statuses["src/missing.py"], "missing")
            self.assertEqual(statuses["src/unchanged.py"], "current")

            snapshots, diagnostics = build_attention_snapshots(
                root, files_meta, output_dir=output
            )
            self.assertEqual(diagnostics, [])
            by_path = {snapshot.path: snapshot for snapshot in snapshots}
            self.assertEqual(by_path["src/exact.py"].doc_status, "current")
            self.assertEqual(by_path["src/stale.py"].doc_status, "stale")
            self.assertEqual(by_path["src/missing.py"].doc_status, "missing")
            self.assertEqual(by_path["src/unchanged.py"].doc_status, "current")
        finally:
            shutil.rmtree(root)
            shutil.rmtree(output)

    def test_unavailable_metadata_and_doc_status_are_diagnostics_not_findings(self):
        unavailable = self.metadata(
            "src/unavailable.py",
            hash="error",
            readable=False,
            line_count=None,
            todo_count=None,
            size=40,
            doc_status="unavailable",
        )
        snapshots, diagnostics = build_attention_snapshots(
            None,
            [unavailable],
            doc_status_by_path={"src/unavailable.py": "unavailable"},
        )

        self.assertEqual(len(snapshots), 1)
        self.assertFalse(snapshots[0].readable)
        self.assertIsNone(snapshots[0].line_count)
        self.assertEqual(
            {(diagnostic.detector, diagnostic.code, diagnostic.path) for diagnostic in diagnostics},
            {
                ("metadata", "read_unavailable", "src/unavailable.py"),
                ("doc_status", "status_unavailable", "src/unavailable.py"),
            },
        )
        self.assertEqual(classify_attention(snapshots), [])

    def test_resolves_structured_python_and_node_entrypoints(self):
        root = Path(tempfile.mkdtemp())
        try:
            (root / "pyproject.toml").write_text(
                '[project.scripts]\ncli = "pkg.cli:main"\n', encoding="utf-8"
            )
            (root / "package.json").write_text(
                json.dumps({"bin": {"tool": "bin/tool.js"}, "scripts": {"fake": "src/fake.js"}}),
                encoding="utf-8",
            )
            metadata = [
                self.metadata("src/pkg/cli.py"),
                self.metadata("bin/tool.js", language="javascript"),
                self.metadata("src/fake.js", language="javascript"),
            ]
            paths, diagnostics = resolve_attention_entrypoints(root, metadata)
            self.assertEqual(paths, {"src/pkg/cli.py", "bin/tool.js"})
            self.assertEqual(diagnostics, [])
        finally:
            shutil.rmtree(root)


class TestAttentionClassifier(unittest.TestCase):
    @staticmethod
    def snapshot(path: str, **overrides: object) -> AttentionSignalSnapshot:
        values: dict[str, object] = {
            "path": path,
            "kind": "source",
            "language": "python",
            "readable": True,
            "line_count": 10,
            "fan_in": 0,
            "fan_out": 0,
            "test_missing": False,
            "todo_current": 0,
            "todo_previous": None,
            "todo_increased": False,
            "is_entrypoint": False,
            "doc_status": "current",
        }
        values.update(overrides)
        return AttentionSignalSnapshot(**values)

    @staticmethod
    def entry_by_identity(entries: list[dict[str, object]], path: str, kind: str):
        return next(entry for entry in entries if entry["path"] == path and entry["kind"] == kind)

    def test_classifier_emits_all_kinds_and_severity_levels(self):
        snapshots = [
            self.snapshot("src/critical_hub.py", line_count=301, fan_in=5),
            self.snapshot(
                "src/critical_cli.py",
                is_entrypoint=True,
                doc_status="missing",
            ),
            self.snapshot("src/high_hub.py", fan_in=5),
            self.snapshot("src/high_cli.py", is_entrypoint=True, doc_status="stale"),
            self.snapshot("src/testless.py", test_missing=True),
            self.snapshot("src/wide.py", fan_out=15),
            self.snapshot("src/undocumented.py", doc_status="missing"),
            self.snapshot("src/stale.py", doc_status="stale"),
            self.snapshot(
                "src/todo.py",
                todo_current=1,
                todo_previous=None,
                todo_increased=True,
            ),
        ]

        entries = classify_attention(snapshots)

        self.assertEqual({entry["kind"] for entry in entries}, set(ATTENTION_KINDS))
        self.assertEqual(
            {entry["severity"] for entry in entries},
            {"critical", "high", "medium", "low"},
        )
        for entry in entries:
            self.assertEqual(set(entry), set(ATTENTION_ENTRY_FIELDS))
            self.assertTrue(entry["reason"])
            self.assertIsInstance(entry["evidence"], dict)

        self.assertEqual(
            self.entry_by_identity(entries, "src/critical_hub.py", "large_file")["severity"],
            "critical",
        )
        self.assertEqual(
            self.entry_by_identity(entries, "src/critical_hub.py", "high_fan_in")["severity"],
            "critical",
        )
        self.assertEqual(
            self.entry_by_identity(entries, "src/critical_cli.py", "doc_missing")["severity"],
            "critical",
        )
        self.assertEqual(
            self.entry_by_identity(entries, "src/high_hub.py", "high_fan_in")["severity"],
            "high",
        )
        self.assertEqual(
            self.entry_by_identity(entries, "src/high_cli.py", "doc_stale")["severity"],
            "high",
        )
        for path, kind in (
            ("src/testless.py", "test_missing"),
            ("src/wide.py", "high_fan_out"),
            ("src/undocumented.py", "doc_missing"),
            ("src/stale.py", "doc_stale"),
        ):
            with self.subTest(path=path, kind=kind):
                self.assertEqual(self.entry_by_identity(entries, path, kind)["severity"], "medium")
        self.assertEqual(
            self.entry_by_identity(entries, "src/todo.py", "todo_increase")["severity"],
            "low",
        )

    def test_classifier_enforces_threshold_boundaries_and_non_promotions(self):
        entries = classify_attention(
            [
                self.snapshot("boundary/fan-in-4.py", fan_in=4),
                self.snapshot("boundary/fan-in-5.py", fan_in=5),
                self.snapshot("boundary/fan-out-14.py", fan_out=14),
                self.snapshot("boundary/fan-out-15.py", fan_out=15),
                self.snapshot("boundary/lines-300.py", line_count=300),
                self.snapshot("boundary/lines-301.py", line_count=301),
                self.snapshot("single/missing.py", doc_status="missing"),
                self.snapshot("single/stale.py", doc_status="stale"),
                self.snapshot("single/large.py", line_count=301),
                self.snapshot("single/testless.py", test_missing=True),
            ]
        )
        found = {(entry["path"], entry["kind"]): entry for entry in entries}

        self.assertNotIn(("boundary/fan-in-4.py", "high_fan_in"), found)
        self.assertEqual(found[("boundary/fan-in-5.py", "high_fan_in")]["severity"], "high")
        self.assertNotIn(("boundary/fan-out-14.py", "high_fan_out"), found)
        self.assertEqual(found[("boundary/fan-out-15.py", "high_fan_out")]["severity"], "medium")
        self.assertNotIn(("boundary/lines-300.py", "large_file"), found)
        self.assertEqual(found[("boundary/lines-301.py", "large_file")]["severity"], "medium")
        self.assertEqual(found[("single/missing.py", "doc_missing")]["severity"], "medium")
        self.assertEqual(found[("single/stale.py", "doc_stale")]["severity"], "medium")
        self.assertEqual(found[("single/large.py", "large_file")]["severity"], "medium")
        self.assertEqual(found[("single/testless.py", "test_missing")]["severity"], "medium")

    def test_classifier_applies_compound_promotions_without_overpromoting(self):
        entries = classify_attention(
            [
                self.snapshot("compound/fan-in-doc.py", fan_in=5, doc_status="missing"),
                self.snapshot(
                    "compound/entrypoint-doc.py",
                    fan_in=5,
                    is_entrypoint=True,
                    doc_status="missing",
                ),
                self.snapshot(
                    "compound/entrypoint-stale.py",
                    fan_in=5,
                    is_entrypoint=True,
                    doc_status="stale",
                ),
            ]
        )
        found = {(entry["path"], entry["kind"]): entry for entry in entries}

        self.assertEqual(
            found[("compound/fan-in-doc.py", "high_fan_in")]["severity"], "high"
        )
        self.assertEqual(
            found[("compound/fan-in-doc.py", "doc_missing")]["severity"], "high"
        )
        self.assertEqual(
            found[("compound/entrypoint-doc.py", "high_fan_in")]["severity"], "high"
        )
        self.assertEqual(
            found[("compound/entrypoint-doc.py", "doc_missing")]["severity"], "critical"
        )
        self.assertEqual(
            found[("compound/entrypoint-stale.py", "high_fan_in")]["severity"], "high"
        )
        self.assertEqual(
            found[("compound/entrypoint-stale.py", "doc_stale")]["severity"], "high"
        )
        self.assertNotEqual(
            found[("compound/fan-in-doc.py", "doc_missing")]["severity"], "critical"
        )

    def test_classifier_is_deterministic_for_permutations_and_canonical_bytes(self):
        snapshots = [
            self.snapshot("z/low.py", todo_current=2, todo_previous=1, todo_increased=True),
            self.snapshot("a/critical.py", fan_in=5, line_count=301),
            self.snapshot("m/medium.py", fan_out=15),
            self.snapshot("b/high.py", fan_in=5),
        ]
        variants = (snapshots, list(reversed(snapshots)), snapshots[2:] + snapshots[:2])
        expected = classify_attention(variants[0])
        expected_bytes = canonical_attention_bytes(expected)

        for variant in variants:
            with self.subTest(order=[snapshot.path for snapshot in variant]):
                actual = classify_attention(variant)
                self.assertEqual(actual, expected)
                self.assertEqual(canonical_attention_bytes(actual), expected_bytes)
                self.assertEqual(serialize_attention(actual).encode("utf-8"), expected_bytes)

        self.assertEqual(
            [(entry["severity"], entry["path"], entry["kind"]) for entry in expected],
            [
                ("critical", "a/critical.py", "high_fan_in"),
                ("critical", "a/critical.py", "large_file"),
                ("high", "b/high.py", "high_fan_in"),
                ("medium", "m/medium.py", "high_fan_out"),
                ("low", "z/low.py", "todo_increase"),
            ],
        )

    def test_classifier_deduplicates_identical_snapshots_and_rejects_conflicts(self):
        snapshot = self.snapshot("src/duplicate.py", fan_in=5)
        self.assertEqual(classify_attention([snapshot, snapshot]), classify_attention([snapshot]))

        with self.assertRaises(AttentionContractError):
            classify_attention(
                [
                    self.snapshot("src/conflict.py", fan_in=5),
                    self.snapshot("src/conflict.py", fan_in=6),
                ]
            )

    def test_attention_entry_validator_rejects_noncanonical_or_invalid_shape(self):
        entries = classify_attention([self.snapshot("src/a.py", fan_in=5), self.snapshot("src/b.py", fan_out=15)])
        validate_attention(entries)

        with self.assertRaises(AttentionContractError):
            validate_attention(list(reversed(entries)))

        invalid = [dict(entry) for entry in entries]
        invalid[0]["evidence"] = dict(invalid[0]["evidence"])
        invalid[0]["evidence"]["fan_in"] = True
        with self.assertRaises(AttentionContractError):
            validate_attention(invalid)

if __name__ == "__main__":
    unittest.main()
