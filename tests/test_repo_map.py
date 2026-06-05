import os
from pathlib import Path

def generate_repo_map(root_path: Path, max_depth: int = 2) -> dict:
    ignore_dirs = {".git", "node_modules", "__pycache__", "dist", "build", "venv", ".venv", ".pytest_cache", ".kelpie", ".serena"}
    
    def get_category(path: Path) -> str:
        name = path.name.lower()
        if path.is_dir():
            if name in ["src", "lib", "app", "cmd", "pkg", "internal"]:
                return "code"
            if name in ["test", "tests", "spec", "specs"]:
                return "test"
            if name in ["doc", "docs"]:
                return "doc"
            if name in ["config", "conf", ".github", ".vscode"]:
                return "config"
            if name in ["scripts", "ci", "ops"]:
                return "ci"
        else:
            if name.endswith((".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".rb")):
                if "test" in name:
                    return "test"
                return "code"
            if name.endswith((".md", ".txt", ".rst", ".pdf")):
                return "doc"
            if name.endswith((".json", ".yaml", ".yml", ".toml", ".ini", ".conf", ".cfg")):
                return "config"
            if name in ["dockerfile", "makefile", "build.gradle"]:
                return "ci"
        return "unknown"

    nodes = []

    def _traverse(current_path: Path, current_depth: int):
        if current_depth > max_depth:
            return
            
        try:
            for item in current_path.iterdir():
                if item.is_dir() and item.name in ignore_dirs:
                    continue
                
                rel_path = item.relative_to(root_path).as_posix()
                node_type = "dir" if item.is_dir() else "file"
                category = get_category(item)
                
                nodes.append({
                    "path": rel_path,
                    "node_type": node_type,
                    "category": category
                })
                
                if item.is_dir() and current_depth < max_depth:
                    _traverse(item, current_depth + 1)
        except PermissionError:
            pass

    _traverse(root_path, 1)
    
    return {
        "root": ".",
        "max_depth": max_depth,
        "nodes": nodes
    }

if __name__ == "__main__":
    repo_map = generate_repo_map(Path("."))
    import json
    print(json.dumps(repo_map, indent=2))
