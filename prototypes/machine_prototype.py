import os
import sys
import fnmatch
import json
from pathlib import Path

# 親ディレクトリをパスに追加して isohyps のモジュールをロードできるようにする
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isohyps.machine_analysis import (
    extract_file_metadata,
    extract_file_symbols,
    build_repo_map_summary,
    detect_attention_points,
    generate_machine_report,
    simple_yaml_dump
)

def parse_gitignore(root: Path) -> list[str]:
    gitignore_path = root / ".gitignore"
    if not gitignore_path.exists():
        return []
    patterns = []
    with open(gitignore_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line)
    return patterns

def should_ignore(path: Path, root: Path, gitignore_patterns: list[str]) -> bool:
    try:
        rel_path = path.resolve().relative_to(root.resolve())
    except ValueError:
        return True
    rel_path_str = rel_path.as_posix()
    
    # ハードコードされた無視ディレクトリ
    ignore_dirs = {".git", "__pycache__", "venv", ".venv", ".env", "node_modules", ".vscode", ".idea", "dist", "build"}
    # 明示的に無視したい一時・キャッシュディレクトリ
    extra_ignore_dirs = {".pytest_cache", ".serena", ".kelpie"}
    
    for part in rel_path.parts:
        if part in ignore_dirs or part in extra_ignore_dirs or part.endswith(".egg-info"):
            return True

    # .gitignore パターンマッチ
    for pattern in gitignore_patterns:
        # ディレクトリ制限の処理
        if pattern.endswith("/"):
            p = pattern.rstrip("/")
            if any(fnmatch.fnmatch(part, p) for part in rel_path.parts):
                return True
        else:
            # 任意のファイル・ディレクトリ名にマッチ
            if fnmatch.fnmatch(rel_path_str, pattern) or any(fnmatch.fnmatch(part, pattern) for part in rel_path.parts):
                return True
            # ルートからの絶対的な指定
            if pattern.startswith("/"):
                p = pattern.lstrip("/")
                if fnmatch.fnmatch(rel_path_str, p):
                    return True
    return False

def run_prototype_scan(root_path: str, output_dir_path: str):
    root = Path(root_path).resolve()
    output_dir = Path(output_dir_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning target: {root}")
    print(f"Output to: {output_dir}")
    
    # .gitignore のパース
    gitignore_patterns = parse_gitignore(root)
    print(f"Loaded gitignore patterns: {gitignore_patterns}")
    
    # ファイルの走査
    all_files = []
    ignored_count = 0
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            if should_ignore(p, root, gitignore_patterns):
                ignored_count += 1
                continue
            all_files.append(p)
            
    print(f"Total files found (active): {len(all_files)}")
    print(f"Total files ignored: {ignored_count}")
    
    # 検出されたファイルの一覧表示
    print("\n--- Active Files ---")
    for f in all_files[:15]:
        print(f"  {f.relative_to(root)}")
    if len(all_files) > 15:
        print(f"  ... and {len(all_files) - 15} more files")

    # メタデータとシンボル抽出
    files_meta = [extract_file_metadata(f, root) for f in all_files]
    symbols_list = [extract_file_symbols(f, root) for f in all_files]
    repo_map = build_repo_map_summary(root, files_meta)
    attention = detect_attention_points(root, files_meta, symbols_list)
    
    # 1. machine_report.md の作成
    report_content = generate_machine_report(root, files_meta, repo_map, attention)
    (output_dir / "machine_report.md").write_text(report_content, encoding="utf-8")
    
    # 2. analysis_report.md の合成（機械用サマリー）
    # コントローラー実行としてのStatusは machine_scan_only、トークン使用量は 0
    analysis_report = f"""# Project Analysis Report: {root.name}

**Root Directory:** `{root}`  
**Backend:** none (machine scan only)  
**Runtime:** controller  
**Status:** success  
**Total Files Scanned:** {len(all_files)}  
**Total Steps Used:** 0  
**Total Tokens Used:** 0  

## Executive Summary
This report summarizes the static machine scan of the project. No LLM resources were utilized.

## Machine Analysis Status
- **Source Coverage:** 0% (LLM analysis not executed)
- **Attention Points Detected:** {len(attention)} items
- **Active Files Count:** {len(all_files)}
- **Ignored Files Count:** {ignored_count}

Refer to [machine_report.md](./machine_report.md) for the detailed inventory and symbol details.
"""
    (output_dir / "analysis_report.md").write_text(analysis_report, encoding="utf-8")
    
    # 3. index.md の合成
    # staleファイル（変更されたファイル）や、アテンションポイントをインデックスとして提示
    changed_files = [m["path"] for m in files_meta if m["status"] in ("changed", "added")]
    
    index_content = f"""# Directory: {root.name}

Welcome to the static analysis index for `{root.name}`.

## Project Summary
- **Total Source Files:** {len(all_files)}
- **Status:** Static Scan Complete

## Stale or Newly Added Files (Need Explanation)
These files are either new or modified, and their corresponding documentation (if any) may be stale.
"""
    if changed_files:
        for cf in changed_files:
            index_content += f"- [ ] `{cf}` (Status: {next(m['status'] for m in files_meta if m['path'] == cf)})\n"
    else:
        index_content += "- No modified or added files detected. All files are unchanged.\n"
        
    index_content += """
## High Priority Files to Inspect (Attention Points)
These files have warnings or high complexity:
"""
    if attention:
        for att in attention:
            index_content += f"- {att}\n"
    else:
        index_content += "- No critical attention points.\n"
        
    index_content += f"""
## Directory Structure Overview
Refer to [machine_report.md](./machine_report.md) for the full inventory.
"""
    (output_dir / "index.md").write_text(index_content, encoding="utf-8")
    print("\nGenerated machine_report.md, analysis_report.md, and index.md successfully!")

if __name__ == "__main__":
    target_repo = "/Users/aoitan/workspace/kuroko/repo"
    out_dir = "/Users/aoitan/workspace/project-analyzer-rlm/tomoe_works/isohyps/scan/analysis_docs_proto"
    run_prototype_scan(target_repo, out_dir)
