import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil
import os
from analyzer import RLMAnalyzer, LLMClient

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
        (self.root / "sub1" / "file1.py").write_text("print('hello')", encoding='utf-8')
        (self.root / "file2.txt").write_text("some text", encoding='utf-8')
        
        self.mock_client = MagicMock(spec=LLMClient)
        self.analyzer = RLMAnalyzer(self.mock_client, max_depth=2)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_recursive_analysis(self):
        # 1. ルートディレクトリでの重要要素選択をモック
        # 2. サブディレクトリでの重要要素選択をモック
        # 3. ファイル解析をモック
        
        def side_effect(prompt, context=""):
            if "ディレクトリ 'root'" in prompt or f"'{self.root.name}'" in prompt:
                if "重要なファイルやディレクトリ" in prompt:
                    return "sub1, file2.txt"
                if "役割を簡潔に要約" in prompt:
                    return "Root directory summary"
            if "ディレクトリ 'sub1'" in prompt:
                if "重要なファイルやディレクトリ" in prompt:
                    return "file1.py"
                if "役割を簡潔に要約" in prompt:
                    return "Sub1 directory summary"
            if "ファイル 'file1.py'" in prompt:
                return "File1 summary"
            if "ファイル 'file2.txt'" in prompt:
                return "File2 summary"
            return "Default summary"

        self.mock_client.query.side_effect = side_effect
        
        result = self.analyzer.analyze(self.root)
        
        self.assertEqual(result, "Root directory summary")
        # 呼び出し回数の確認（root x2, sub1 x2, file1 x1, file2 x1 = 6回程度）
        self.assertGreaterEqual(self.mock_client.query.call_count, 4)

if __name__ == "__main__":
    unittest.main()
