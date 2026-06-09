import os
import sys
import argparse
import warnings
from pathlib import Path
from abc import ABC, abstractmethod

from isohyps.analysis_helpers import detect_language, extract_symbols, is_probably_binary
from isohyps.project_analysis import RLMRuntimeAnalyzer

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv():
        return False

load_dotenv()

_DEBUG = bool(os.getenv("DEBUG"))


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        print(f"[DBG {tag}] {msg}", file=sys.stderr, flush=True)


class BaseLLMClient(ABC):
    @abstractmethod
    def query(self, prompt: str) -> str:
        pass

class GeminiClient(BaseLLMClient):
    def __init__(self, model_name="gemini-1.5-flash"):
        try:
            from google import genai
        except ModuleNotFoundError as exc:
            raise ValueError("google-genai package is required for the Gemini backend.") from exc
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def query(self, prompt: str) -> str:
        _dbg("llm", f"gemini query model={self.model_name} prompt={len(prompt)} chars")
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            result = response.text.strip()
            _dbg("llm", f"gemini response={len(result)} chars: {result[:80]!r}...")
            return result
        except Exception as e:
            _dbg("llm", f"gemini error: {e}")
            return f"[Gemini Error: {e}]"

class OllamaClient(BaseLLMClient):
    def __init__(self, model_name="llama3", base_url=None, num_ctx=4096):
        try:
            import ollama
        except ModuleNotFoundError as exc:
            raise ValueError("ollama package is required for the Ollama backend.") from exc
        self.model_name = model_name
        self.client = ollama.Client(host=base_url) if base_url else ollama
        self.options = {
            'num_ctx': num_ctx,
            'temperature': 0.2, # 解析の安定性を高めるために低めに設定
        }

    def query(self, prompt: str) -> str:
        _dbg("llm", f"ollama query model={self.model_name} prompt={len(prompt)} chars")
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options=self.options
            )
            result = response['message']['content'].strip()
            _dbg("llm", f"ollama response={len(result)} chars: {result[:80]!r}...")
            return result
        except Exception as e:
            _dbg("llm", f"ollama error: {e}")
            return f"[Ollama Error: {e}]"


class RLMAnalyzer:
    def __init__(self, client: BaseLLMClient, max_depth: int = 3, output_dir: Path = None, warn: bool = True):
        if warn:
            warnings.warn(
                "RLMAnalyzer (legacy) is deprecated and will be removed in a future version. "
                "Use RLMRuntimeAnalyzer (controller) instead; controller may use different token "
                "budgets and execution times.",
                FutureWarning,
                stacklevel=2,
            )
        self.client = client
        self.max_depth = max_depth
        self.cache = {}
        self.output_dir = output_dir
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, path: Path, depth: int = 0, rel_path: Path = Path(".")) -> str:
        if depth > self.max_depth:
            return f"- [Depth Limit Reached] {path.name}"

        if path in self.cache:
            return self.cache[path]

        print(f"{'  ' * depth}Analyzing: {path.name}")

        if path.is_dir():
            result = self._analyze_directory(path, depth, rel_path)
            if self.output_dir:
                out_file = self.output_dir / rel_path / "index.md"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(f"# Directory: {path.name}\n\n{result}")
        else:
            lang = self._detect_language(path)
            if lang:
                result = self._analyze_code_file(path, depth, lang)
            else:
                result = self._analyze_file(path, depth)
            
            if self.output_dir:
                out_file = self.output_dir / rel_path.with_suffix(".md")
                out_file.parent.mkdir(parents=True, exist_ok=True)
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(f"# File: {path.name}\n\n{result}")

        self.cache[path] = result
        return result

    def _detect_language(self, path: Path) -> str | None:
        return detect_language(path)

    def _analyze_code_file(self, path: Path, depth: int, lang: str) -> str:
        """tree-sitter でシンボルを抽出し、LLM に要約させる（多言語対応）"""
        symbol_info = extract_symbols(path, lang=lang)
        if symbol_info["error"]:
            print(f"Warning: tree-sitter analysis failed for {path} ({lang}): {symbol_info['error']}")
        if not symbol_info["symbols"]:
            return self._analyze_file(path, depth, lang)

        snippets = [f"```{lang}\n{snippet}\n```" for snippet in symbol_info["symbols"]]
        symbols_str = '\n\n'.join(snippets)
        prompt = (
            f"ファイル '{path.name}' ({lang}) には以下のシンボル定義があります：\n\n"
            f"{symbols_str}\n\n"
            "これらの定義から、このファイルが提供している主要な機能と設計意図を技術的に詳しく要約してください。"
            "日本語で回答してください。"
        )
        return self.client.query(prompt)

    def _analyze_directory(self, path: Path, depth: int, rel_path: Path) -> str:
        ignore_list = {".git", "__pycache__", "venv", ".env", "node_modules", ".vscode", ".idea"}
        items = [item for item in path.iterdir() if item.name not in ignore_list]
        
        if not items:
            return "空のディレクトリです。"

        item_names = [f"{'[DIR] ' if item.is_dir() else '[FILE]'} {item.name}" for item in items]
        items_str = "\n".join(item_names)

        selection_prompt = (
            f"あなたはシニアエンジニアとして、プロジェクト構造を解析しています。\n"
            f"ディレクトリ '{path.name}' の中身は以下の通りです：\n"
            f"{items_str}\n\n"
            "この中で、プロジェクトの機能や責務を理解するために解析すべき重要な要素を最大5つ選び、"
            "カンマ区切りで名前だけを回答してください（例: src, main.py, utils.py）。"
            "優先度の高い順にお願いします。もし重要そうなものがなければ 'None' と回答してください。"
            "回答はカンマ区切りのリスト、または 'None' という単語のみにしてください。"
        )
        selected_items_str = self.client.query(selection_prompt)
        
        if "None" in selected_items_str or not selected_items_str or "[Error" in selected_items_str:
            return f"ディレクトリ '{path.name}' の構造解析を完了しました。"

        selected_names = [name.strip().strip('`').strip('"') for name in selected_items_str.split(",") if name.strip()]

        sub_results = []
        for name in selected_names:
            sub_path = path / name
            if sub_path.exists():
                # 相対パスを更新して再帰呼び出し
                res = self.analyze(sub_path, depth + 1, rel_path / name)
                sub_results.append(f"#### {name}\n{res}")

        summary_prompt = (
            f"ディレクトリ '{path.name}' 内の主要な要素の解析結果は以下の通りです：\n\n"
            f"{os.linesep.join(sub_results)}\n\n"
            "これらの情報を統合し、このディレクトリがプロジェクト全体でどのような役割"
            "を担っているかを技術的に詳しく要約してください。日本語で回答してください。"
        )
        return self.client.query(summary_prompt)

    def _analyze_file(self, path: Path, depth: int, lang: str | None = None) -> str:
        try:
            if is_probably_binary(path):
                return "（バイナリファイルのため解析をスキップしました）"

            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(5000)
        except Exception as e:
            return f"[Error reading file: {e}]"

        language_hint = f"言語は {lang} です。\n" if lang else ""
        prompt = (
            f"ファイル '{path.name}' の内容（冒頭部分）を解析してください：\n"
            f"{language_hint}"
            f"```\n{content}\n```\n\n"
            "このファイルが提供している主要な機能、主要なクラス、エクスポートされている関数、"
            "およびその責務を技術的に要約してください。日本語で回答してください。"
        )
        return self.client.query(prompt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recursive Project Analyzer (RLM style)")
    parser.add_argument("root", help="Analysis root directory", default=".", nargs="?")
    parser.add_argument("--depth", type=int, default=2, help="Max recursion depth")
    parser.add_argument("--runtime", choices=["legacy", "controller"], default="controller", help="Runtime to use (default: controller)")
    parser.add_argument("--max-steps", type=int, default=30, help="Max controller steps for the controller runtime")
    parser.add_argument("--step-timeout", type=float, default=15.0, help="Per-step timeout for the controller runtime")
    parser.add_argument("--llm-timeout", type=float, default=120.0, help="Per-query timeout for backend LLM calls")
    parser.add_argument("--max-total-tokens", type=int, default=90000, help="Approximate shared token budget for the controller runtime")
    parser.add_argument("--backend", choices=["gemini", "ollama"], default="gemini", help="LLM backend to use")
    parser.add_argument("--model", help="LLM model name (defaults: gemini-1.5-flash or llama3)")
    parser.add_argument("--out", default="analysis_docs", help="Output directory for structured documentation")
    parser.add_argument("--ollama-url", help="Base URL for Ollama API (e.g. http://192.168.1.10:11434)")
    parser.add_argument("--num-ctx", type=int, default=8192, help="Context size (num_ctx) for Ollama")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.runtime == "legacy":
        warnings.warn(
            "'legacy' runtime is deprecated. Using 'controller' is recommended; "
            "controller may use different token budgets and execution times.",
            FutureWarning,
            stacklevel=1,
        )

    root_path = Path(args.root).resolve()
    output_path = Path(args.out).resolve()
    
    if args.backend == "gemini":
        model = args.model if args.model else "gemini-1.5-flash"
        try:
            client = GeminiClient(model_name=model)
        except ValueError as e:
            print(f"Error: {e}")
            return
    else:
        model = args.model if args.model else "llama3"
        client = OllamaClient(model_name=model, base_url=args.ollama_url, num_ctx=args.num_ctx)

    # 出力先ディレクトリを指定して初期化
    if args.runtime == "controller":
        analyzer = RLMRuntimeAnalyzer(
            client,
            max_depth=args.depth,
            max_steps=args.max_steps,
            output_dir=output_path,
            step_timeout_seconds=args.step_timeout,
            llm_timeout_seconds=args.llm_timeout,
            max_total_tokens=args.max_total_tokens,
            backend_name=args.backend,
            model_name=model,
        )
    else:
        analyzer = RLMAnalyzer(client, max_depth=args.depth, output_dir=output_path, warn=False)

    print(f"Starting RLM analysis from: {root_path}")
    print(f"Runtime: {args.runtime}")
    print(f"Backend: {args.backend}, Model: {model}")
    if args.runtime == "controller":
        print(
            "Agentic controller runtime is active "
            f"(depth={args.depth}, max_steps={args.max_steps}, token_budget={args.max_total_tokens}, llm_timeout={args.llm_timeout}s)."
        )
    if args.backend == "ollama":
        print(f"Ollama URL: {args.ollama_url if args.ollama_url else 'local'}")
        print(f"Context Size: {args.num_ctx}")
    print(f"Output directory: {output_path}")
    print("-" * 30)

    final_summary = analyzer.analyze(root_path)

    if args.runtime == "legacy":
        report_path = output_path / "analysis_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Project Analysis Report: {root_path.name}\n\n")
            f.write(f"**Root Directory:** `{root_path}`  \n")
            f.write(f"**Backend:** {args.backend} ({model})  \n")
            f.write(f"**Runtime:** legacy  \n")
            f.write(f"**Max Depth:** {args.depth}  \n\n")
            f.write("## Executive Summary\n\n")
            f.write(final_summary)
    else:
        report_path = output_path / "analysis_report.md"

    print(f"\nAnalysis complete.")
    print(f"Structured docs: {output_path}")
    print(f"Full report:     {report_path}")

if __name__ == "__main__":
    main()
