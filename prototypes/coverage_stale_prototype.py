import os
import sys
import fnmatch
import json
import re
from pathlib import Path
from typing import Any

# Add parent dir to path to load isohyps modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isohyps.machine_analysis import (
    extract_file_metadata,
    extract_file_symbols,
    build_repo_map_summary,
    parse_gitignore,
    should_ignore
)

def check_coverage_and_stale(root: Path, output_dir: Path, active_files: list[Path]):
    missing_docs = []
    stale_docs = []
    valid_docs = []
    
    # Check for each active file
    for f in active_files:
        rel_path = f.relative_to(root).as_posix()
        
        # Candidate paths for the documentation file in the output directory
        # e.g., output_dir / "kanpe/brief_formatter.py.md"
        doc_candidates = [
            output_dir / f"{rel_path}.md",
            output_dir / Path(rel_path).with_suffix(".md")
        ]
        
        found_doc = None
        for candidate in doc_candidates:
            if candidate.exists() and candidate.is_file():
                found_doc = candidate
                break
                
        if not found_doc:
            missing_docs.append(rel_path)
        else:
            # Compare mtime
            source_mtime = f.stat().st_mtime
            doc_mtime = found_doc.stat().st_mtime
            
            # Allow some tolerance for file copy / extraction delays
            if source_mtime > doc_mtime + 1.0:
                stale_docs.append(rel_path)
            else:
                valid_docs.append(rel_path)
                
    total_count = len(active_files)
    documented_count = total_count - len(missing_docs)
    coverage_percent = (documented_count / total_count * 100) if total_count > 0 else 100.0
    
    return {
        "missing": sorted(missing_docs),
        "stale": sorted(stale_docs),
        "valid": sorted(valid_docs),
        "coverage_percent": coverage_percent
    }

def detect_attention_points_improved(
    root: Path,
    files_meta: list[dict[str, Any]],
    symbols_list: list[dict[str, Any]],
    previous_meta: dict[str, Any] | None = None
) -> list[str]:
    attention = []
    
    meta_by_path = {meta["path"]: meta for meta in files_meta}
    symbols_by_path = {sym["path"]: sym for sym in symbols_list}

    # Gather fan-in / fan-out
    fan_in = {}
    fan_out = {}

    for path, sym in symbols_by_path.items():
        fan_out[path] = 0
        for imp in sym["imports"]:
            mod = imp["module"]
            is_internal = imp["internal"]
            
            if is_internal:
                fan_out[path] += 1
                for other_path in meta_by_path.keys():
                    other_mod_name = Path(other_path).stem
                    if other_mod_name == mod.split(".")[-1]:
                        fan_in[other_path] = fan_in.get(other_path, 0) + 1

    # 1. Large files check (Threshold: 300 lines)
    for path, meta in meta_by_path.items():
        if meta["kind"] == "source":
            try:
                line_count = len((root / path).read_text(encoding="utf-8", errors="ignore").splitlines())
                if line_count > 300: # Changed from 100 to 300
                    attention.append(f"file is large: {path}, {line_count} lines")
            except Exception:
                pass

    # 2. No tests found check (Exclude __init__.py and empty files)
    for path, meta in meta_by_path.items():
        if meta["kind"] == "source" and meta["language"] == "python":
            # Exclude __init__.py and empty/very small files
            if Path(path).name == "__init__.py" or meta["size"] < 50:
                continue
                
            stem = Path(path).stem
            test_exists = False
            for test_path in symbols_by_path.keys():
                test_stem = Path(test_path).stem
                if test_stem in (f"test_{stem}", f"{stem}_test"):
                    test_exists = True
                    break
            if not test_exists:
                attention.append(f"no tests found for {path}")

    # 3. TODO/FIXME detection (Compare with previous if available)
    for path, meta in meta_by_path.items():
        if meta["kind"] == "source" and not is_probably_binary(root / path):
            try:
                content = (root / path).read_text(encoding="utf-8", errors="ignore")
                matches = re.findall(r'(?:TODO|FIXME)[:\s]+(.*)', content, re.IGNORECASE)
                if matches:
                    prev_count = 0
                    has_previous = False
                    if previous_meta and path in previous_meta:
                        # Best effort: check if we stored todo count in previous run
                        # (Note: we need to write todo_count to files_meta)
                        prev_todo = previous_meta[path].get("todo_count")
                        if prev_todo is not None:
                            prev_count = prev_todo
                            has_previous = True
                            
                    if has_previous:
                        if len(matches) > prev_count:
                            attention.append(f"TODO/FIXME count increased in {path}: {len(matches)} items (previously {prev_count})")
                    else:
                        attention.append(f"TODO/FIXME count: {len(matches)} items in {path}")
            except Exception:
                pass

    # 4. high fan-in / fan-out
    for path, count in fan_in.items():
        if count >= 10:
            attention.append(f"high fan-in: {path} imported by {count} files")
            
    for path, count in fan_out.items():
        if count >= 15:
            attention.append(f"high fan-out: {path} imports {count} internal modules")

    return attention

def is_probably_binary(path: Path) -> bool:
    try:
        with open(path, 'rb') as f:
            chunk = f.read(1024)
            return b'\x00' in chunk
    except Exception:
        return True

def run_prototype_coverage_scan(root_path: str, output_dir_path: str):
    root = Path(root_path).resolve()
    output_dir = Path(output_dir_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== PROTOTYPE SCAN ===")
    print(f"Target: {root}")
    print(f"Output: {output_dir}")
    
    # 1. Load previous metadata to compare (needed for TODO comparison)
    json_path = output_dir / "machine_analysis.json"
    previous_meta = None
    if json_path.exists():
        try:
            prev_data = json.loads(json_path.read_text(encoding="utf-8"))
            previous_meta = {f["path"]: f for f in prev_data.get("files", [])}
            print(f"Loaded previous metadata for {len(previous_meta)} files.")
        except Exception as e:
            print(f"Could not load previous metadata: {e}")
            
    # 2. Get active files
    gitignore_patterns = parse_gitignore(root)
    all_files = []
    ignored_count = 0
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            if should_ignore(p, root, gitignore_patterns):
                ignored_count += 1
                continue
            all_files.append(p)
            
    print(f"Active Files: {len(all_files)}, Ignored Files: {ignored_count}")
    
    # 3. Simulate coverage and stale docs check
    # Let's create dummy documentation files for some files to test the logic
    # e.g., dummy docs for kuroko/application.py (up-to-date) and kanpe/cli.py (stale)
    print("\n--- Simulating Documentation Environment ---")
    test_doc_dir = output_dir / "kanpe"
    test_doc_dir.mkdir(parents=True, exist_ok=True)
    
    # Create an up-to-date doc
    app_py = root / "kuroko" / "application.py"
    if app_py.exists():
        app_doc = output_dir / "kuroko" / "application.py.md"
        app_doc.parent.mkdir(parents=True, exist_ok=True)
        app_doc.write_text("# Dummy App Doc", encoding="utf-8")
        # Ensure its mtime is newer than source
        os.utime(app_doc, (app_py.stat().st_atime, app_py.stat().st_mtime + 10.0))
        print("Created up-to-date doc for kuroko/application.py")
        
    # Create a stale doc
    cli_py = root / "kanpe" / "cli.py"
    if cli_py.exists():
        cli_doc = output_dir / "kanpe" / "cli.py.md"
        cli_doc.write_text("# Dummy CLI Doc", encoding="utf-8")
        # Ensure its mtime is older than source
        os.utime(cli_doc, (cli_py.stat().st_atime, cli_py.stat().st_mtime - 10.0))
        print("Created stale doc for kanpe/cli.py")
        
    cov_results = check_coverage_and_stale(root, output_dir, all_files)
    print("\n--- Coverage Results ---")
    print(f"Coverage: {cov_results['coverage_percent']:.1f}%")
    print(f"Missing Docs Count: {len(cov_results['missing'])}")
    print(f"Stale Docs: {cov_results['stale']}")
    print(f"Valid Docs: {cov_results['valid']}")
    
    # 4. Extract metadata (including todo_count for future runs)
    files_meta = []
    for f in all_files:
        meta = extract_file_metadata(f, root, previous_meta)
        # Add todo_count to meta
        try:
            if meta["kind"] == "source" and not is_probably_binary(f):
                content = f.read_text(encoding="utf-8", errors="ignore")
                meta["todo_count"] = len(re.findall(r'(?:TODO|FIXME)[:\s]+(.*)', content, re.IGNORECASE))
            else:
                meta["todo_count"] = 0
        except Exception:
            meta["todo_count"] = 0
        files_meta.append(meta)
        
    symbols_list = [extract_file_symbols(f, root) for f in all_files]
    repo_map = build_repo_map_summary(root, files_meta)
    
    # Detect attention points with improved logic
    attention = detect_attention_points_improved(root, files_meta, symbols_list, previous_meta)
    print(f"\n--- Attention Points Improved (Count: {len(attention)}) ---")
    for att in attention[:10]:
        print(f"  - {att}")
    if len(attention) > 10:
        print(f"  ... and {len(attention) - 10} more")

    # 5. Generate synthesis outputs locally
    # index.md should list Stale OR Newly Added OR Missing explanation files
    changed_or_missing_or_stale = []
    
    # Compile files that need explanation (missing or stale docs)
    # Plus new/modified files (according to git or hashes)
    needs_explanation = set(cov_results["missing"]) | set(cov_results["stale"])
    for m in files_meta:
        if m["status"] in ("changed", "added"):
            needs_explanation.add(m["path"])
            
    print(f"\nFiles needing explanation (Count: {len(needs_explanation)}):")
    for ne in sorted(list(needs_explanation))[:10]:
        # Determine specific reason
        reasons = []
        if ne in cov_results["missing"]:
            reasons.append("missing doc")
        if ne in cov_results["stale"]:
            reasons.append("stale doc")
        m_status = next((m["status"] for m in files_meta if m["path"] == ne), None)
        if m_status in ("changed", "added"):
            reasons.append(f"source {m_status}")
        print(f"  - `{ne}` ({', '.join(reasons)})")

if __name__ == "__main__":
    target_repo = "/Users/aoitan/workspace/kuroko/repo"
    out_dir = "/Users/aoitan/workspace/project-analyzer-rlm/tomoe_works/isohyps/scan/analysis_docs_proto"
    run_prototype_scan = run_prototype_coverage_scan(target_repo, out_dir)
