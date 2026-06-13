import os
import sys
import shutil
from pathlib import Path

# Add parent dir to path to load isohyps modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isohyps.machine_analysis import (
    extract_file_metadata,
    extract_file_symbols,
    parse_gitignore,
)

def prototype_report_generation():
    # Setup dummy directory tree to simulate the project
    test_dir = Path("./temp_proto_project")
    output_dir = Path("./temp_proto_output")
    
    if test_dir.exists():
        shutil.rmtree(test_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
        
    test_dir.mkdir()
    output_dir.mkdir()
    
    # Create files
    (test_dir / "src").mkdir()
    (test_dir / "tests").mkdir()
    
    runner_file = test_dir / "src/runner.py"
    runner_file.write_text("print('hello')", encoding="utf-8")
    
    test_file = test_dir / "tests/test_runner.py"
    test_file.write_text("def test(): pass", encoding="utf-8")
    
    config_file = test_dir / "pyproject.toml"
    config_file.write_text("[project]\nname = 'proto'", encoding="utf-8")
    
    # 1. Run metadata extraction
    gitignore_patterns = parse_gitignore(test_dir)
    all_files = [runner_file, test_file, config_file]
    
    files_meta = []
    for f in all_files:
        files_meta.append(extract_file_metadata(f, test_dir))
        
    # 2. Extract symbols
    symbols_list = []
    for f in all_files:
        symbols_list.append(extract_file_symbols(f, test_dir))
        
    # Compare coverage targets (source vs test)
    # Original targets: kind in ("source", "test")
    coverage_targets_orig = [m for m in files_meta if m["kind"] in ("source", "test") and m["hash"] not in ("binary_skipped", "error")]
    # Improved targets: kind == "source" only
    coverage_targets_improved = [m for m in files_meta if m["kind"] == "source" and m["hash"] not in ("binary_skipped", "error")]
    
    print("Coverage Targets Comparison:")
    print(f"Original targets: {[t['path'] for t in coverage_targets_orig]}")
    print(f"Improved targets: {[t['path'] for t in coverage_targets_improved]}")
    
    # Calculate coverage with dummy docs
    # Let's say we have runner.py.md in output_dir
    (output_dir / "src").mkdir(parents=True, exist_ok=True)
    (output_dir / "src/runner.py.md").write_text("# Runner Doc", encoding="utf-8")
    
    def calculate_coverage(targets):
        missing = []
        valid = []
        for meta in targets:
            rel_path = meta["path"]
            doc_candidates = [
                output_dir / f"{rel_path}.md",
                output_dir / Path(rel_path).with_suffix(".md")
            ]
            found_doc = any(c.exists() for c in doc_candidates)
            if not found_doc:
                missing.append(rel_path)
            else:
                valid.append(rel_path)
        return missing, valid

    missing_orig, valid_orig = calculate_coverage(coverage_targets_orig)
    missing_imp, valid_imp = calculate_coverage(coverage_targets_improved)
    
    print("\nDocumentation Status:")
    print(f"Original - Missing: {missing_orig}, Valid: {valid_orig}")
    print(f"Improved - Missing: {missing_imp}, Valid: {valid_imp}")
    
    # Simulate compat report content
    coverage_percent = (len(valid_imp) / len(coverage_targets_improved) * 100) if coverage_targets_improved else 100.0
    
    report_lines = [
        f"# Project Analysis Report: temp_proto_project",
        "",
        f"**Root Directory:** `{test_dir.resolve()}`  ",
        f"**Backend:** none (machine scan only)  ",
        f"**Runtime:** controller  ",
        f"**Status:** finished  ",  # Changed from success to finished
        f"**Steps Used:** 0  ",      # Changed from Total Steps Used
        f"**Approx Tokens:** 0  ",   # Changed from Total Tokens Used
        f"**Worker Budget:** 0 tokens, 0 LLM calls  ",
        f"**Synthesis RLM Budget:** 0 tokens, 0 LLM calls  ",
        f"**Global Budget:** 0 tokens, 0 LLM calls  ",
        "",
        "## Executive Summary",
        "",
        "This report summarizes the static machine scan of the project. No LLM resources were utilized.",
        "",
        "## Source Coverage",  # Matches the format of project_analysis.py
        "",
        f"- Source files discovered: {len(coverage_targets_improved)}",
        f"- Source files with matching docs: {len(valid_imp)}",
        f"- Source files missing matching docs: {len(missing_imp)}",
        f"- Extra docs without matching source: 0",
        f"- Weak or failed docs: 0",
        f"- Fallback docs generated: 0",
        f"- Coverage: {coverage_percent:.1f}%",
        "",
        "### Fallback Generated Source Docs",
        "",
        "- (none)",
        "",
        "### Missing Source Docs",
        "",
    ]
    if missing_imp:
        for path in missing_imp:
            report_lines.append(f"- `{path}`")
    else:
        report_lines.append("- (none)")
    report_lines.append("")
    
    report_lines.extend([
        "### Extra Docs Without Matching Source",
        "",
        "- (none)",
        "",
        "## Step History",
        "",
        "| Step | Kind | Status | Summary |",
        "| :--- | :--- | :--- | :--- |",
        "| 1 | machine_scan | OK | Static machine scan completed successfully. |",
        ""
    ])
    
    print("\n--- Generated compatible report structure: ---")
    print("\n".join(report_lines[:15]))
    print("...")
    print("\n".join(report_lines[15:28]))
    
    # Cleanup
    shutil.rmtree(test_dir)
    shutil.rmtree(output_dir)

if __name__ == "__main__":
    prototype_report_generation()
