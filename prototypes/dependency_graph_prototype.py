import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

def resolve_module_to_path(module_name: str, src_files: List[str], current_file: str) -> str | None:
    """
    モジュール名（例: 'kuroko.application' または '.db'）から実際のソースファイルパス（relative）を特定する。
    """
    # 逆引き用：ファイル名 -> モジュール名候補
    # 例: "kuroko/application.py" -> ["kuroko.application"]
    # 例: "kuroko/__init__.py" -> ["kuroko"]
    
    # 1. 絶対インポートの解決
    # ドット区切りの名前をパスに変換してチェック
    mod_path_parts = module_name.split(".")
    
    # 相対インポートの処理 (例: node.level > 0 のときなど、先頭がドット)
    if module_name.startswith("."):
        # ドットの数を数える
        dots_count = 0
        for char in module_name:
            if char == ".":
                dots_count += 1
            else:
                break
        
        # current_file の親ディレクトリから相対的に解決
        curr_parts = current_file.split("/")
        if len(curr_parts) > dots_count:
            base_dir_parts = curr_parts[:-dots_count]
            # 残りのモジュール名部分
            rem_mod = module_name[dots_count:]
            if rem_mod:
                resolved_parts = base_dir_parts + rem_mod.split(".")
            else:
                resolved_parts = base_dir_parts
            
            # 解決されたパスのパターンを作成
            possible_rel_path = "/".join(resolved_parts)
            # パスリストから一致するものを探す
            for src in src_files:
                src_no_ext = src.rsplit(".", 1)[0]
                if src_no_ext == possible_rel_path:
                    return src
                if src_no_ext + "/__init__" == possible_rel_path:
                    return src
        return None

    # 絶対インポートの探索
    for src in src_files:
        src_no_ext = src.rsplit(".", 1)[0]
        # パス区切りをドットに変換したものと一致するか
        dotted_src = src_no_ext.replace("/", ".")
        if dotted_src == module_name:
            return src
        # __init__.py の場合は親パッケージ名とも一致する
        if src_no_ext.endswith("/__init__"):
            package_dotted = src_no_ext[:-9].replace("/", ".")
            if package_dotted == module_name:
                return src
            
        # インポート名が長くてファイルが親モジュールの場合（例: shinko.llm.LLMClient をインポートして、ファイルが shinko/llm.py）
        if module_name.startswith(dotted_src + "."):
            return src

    return None


def build_dependency_graph(analysis_data: Dict[str, Any]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    analysis_data (json) からファイル間の有向グラフを構築する。
    returns: (adjacency_list_forward, adjacency_list_backward)
    """
    files = [f["path"] for f in analysis_data["files"] if f["kind"] == "source" and f["language"] != "unknown"]
    symbols_by_path = {sym["path"]: sym for sym in analysis_data["symbols"]}
    
    forward_graph = {f: [] for f in files}
    backward_graph = {f: [] for f in files}
    
    for f in files:
        if f not in symbols_by_path:
            continue
        sym_info = symbols_by_path[f]
        imports = sym_info.get("imports", [])
        
        for imp in imports:
            mod_name = imp["module"]
            resolved_path = resolve_module_to_path(mod_name, files, f)
            if resolved_path and resolved_path != f:
                if resolved_path not in forward_graph[f]:
                    forward_graph[f].append(resolved_path)
                if f not in backward_graph[resolved_path]:
                    backward_graph[resolved_path].append(f)
                    
    return forward_graph, backward_graph


def detect_cycles_dfs(graph: Dict[str, List[str]]) -> List[List[str]]:
    """
    DFSで依存関係内の循環（Cycles）をすべて検出する。
    """
    cycles = []
    visited = {} # 0: unvisited, 1: visiting, 2: visited
    path = []
    
    def dfs(node):
        visited[node] = 1
        path.append(node)
        
        for neighbor in graph.get(node, []):
            if visited.get(neighbor, 0) == 1:
                # 循環を検出
                cycle_start_idx = path.index(neighbor)
                cycle = path[cycle_start_idx:] + [neighbor]
                cycles.append(cycle)
            elif visited.get(neighbor, 0) == 0:
                dfs(neighbor)
                
        path.pop()
        visited[node] = 2

    for node in graph:
        if visited.get(node, 0) == 0:
            dfs(node)
            
    return cycles


def topological_sort(graph: Dict[str, List[str]]) -> List[str]:
    """
    Kahnのアルゴリズムを用いてトポロジカルソートを実行する。
    循環がある場合は、循環以外の部分を部分ソートする。
    """
    # 入次数を計算
    in_degree = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] = in_degree.get(v, 0) + 1
            
    # 入次数0のノードをキューに追加
    queue = [u for u in graph if in_degree[u] == 0]
    order = []
    
    while queue:
        # 決定論的に安定した順序を保つためソート（またはアルファベット順）
        queue.sort()
        u = queue.pop(0)
        order.append(u)
        
        for v in graph.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # ソートしきれなかったノード（循環に含まれるノードなど）を追加
    remaining = [u for u in graph if u not in order]
    if remaining:
        # アルファベット順で追加
        remaining.sort()
        order.extend(remaining)
        
    return order


def generate_mermaid_graph(graph: Dict[str, List[str]]) -> str:
    """
    依存関係グラフを Mermaid フォーマットに変換する。
    ノイズを減らすため、依存関係があるノードのみを描画する。
    """
    lines = ["graph TD"]
    
    # 描画対象のノードを絞り込む
    active_nodes = set()
    for u, vs in graph.items():
        if vs:
            active_nodes.add(u)
            for v in vs:
                active_nodes.add(v)
                
    # ノード定義 (IDとラベル)
    # パスが長いので、最後のファイル名だけをラベルにし、IDをエスケープする
    node_ids = {}
    for i, node in enumerate(sorted(list(active_nodes))):
        node_ids[node] = f"node_{i}"
        label = Path(node).name
        # ディレクトリ構造がわかりやすいように親ディレクトリも含める (例: kuroko/cli.py)
        parts = Path(node).parts
        if len(parts) > 1:
            label = f"{parts[-2]}/{parts[-1]}"
        lines.append(f"  {node_ids[node]}[\"{label}\"]")
        
    # エッジ定義
    for u in sorted(graph.keys()):
        if u not in node_ids:
            continue
        for v in sorted(graph[u]):
            if v not in node_ids:
                continue
            lines.append(f"  {node_ids[u]} --> {node_ids[v]}")
            
    return "\n".join(lines)


def run_prototype():
    # 前回の実験結果のJSONファイルを読み込む
    json_path = Path("/Users/aoitan/workspace/project-analyzer-rlm/tomoe_works/isohyps/scan/analysis_docs/machine_analysis.json")
    if not json_path.exists():
        print(f"Error: {json_path} does not exist.")
        return
        
    print(f"Loading analysis data from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print("\n--- 1. Building Dependency Graph ---")
    forward_graph, backward_graph = build_dependency_graph(data)
    
    print(f"Total source files: {len(forward_graph)}")
    for f, deps in sorted(forward_graph.items()):
        if deps:
            print(f"  {f} depends on: {deps}")
            
    print("\n--- 2. Cycle Detection ---")
    cycles = detect_cycles_dfs(forward_graph)
    if cycles:
        print(f"Found {len(cycles)} cycle(s):")
        for cycle in cycles:
            print(f"  Cycle: {' -> '.join(cycle)}")
    else:
        print("No cycles detected.")
        
    print("\n--- 3. Topological Sort (Recommended Reading Order: Bottom-up) ---")
    # 依存先から依存元への辺（backward_graph）を使ってトポロジカルソートを実行することで、
    # 前提知識のいらない独立したモジュールからエントリーポイントへ向かう順序が得られます。
    order = topological_sort(backward_graph)
    print("Recommended Order (Bottom-up: start reading from these):")
    for i, path in enumerate(order):
        deps_count = len(forward_graph[path])
        dep_by_count = len(backward_graph[path])
        print(f"  {i+1}. {path} (dependencies: {deps_count}, imported_by: {dep_by_count})")
        
    print("\n--- 4. Mermaid Graph ---")
    mermaid = generate_mermaid_graph(forward_graph)
    print(mermaid)


if __name__ == "__main__":
    run_prototype()
