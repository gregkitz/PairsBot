#!/usr/bin/env python
"""
Codebase Audit Script

This script analyzes the codebase to identify potential AI blindspot issues,
focusing especially on duplicated implementations and disconnected components.
"""

import os
import sys
import re
import ast
import json
import subprocess
from collections import defaultdict, Counter
from pathlib import Path
import importlib
import inspect
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Any, Optional

# Configuration
SRC_DIR = "src"
AUDIT_DIR = "audit/reports"
FILE_EXTENSIONS = [".py"]
MIN_DUPLICATE_LINES = 6
MAX_RECOMMENDED_FILE_LINES = 500
MAX_RECOMMENDED_FUNCTION_LINES = 100
MAX_RECOMMENDED_FUNCTION_COMPLEXITY = 10
MAX_NESTED_BLOCKS = 4

# Ensure audit directory exists
os.makedirs(AUDIT_DIR, exist_ok=True)


def count_lines_in_file(file_path: str) -> int:
    """Count the number of non-empty lines in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def get_all_python_files() -> List[str]:
    """Get all Python files in the project."""
    python_files = []
    for root, _, files in os.walk(SRC_DIR):
        for file in files:
            if any(file.endswith(ext) for ext in FILE_EXTENSIONS):
                python_files.append(os.path.join(root, file))
    return python_files


class FunctionVisitor(ast.NodeVisitor):
    """AST visitor to extract function and method information."""
    
    def __init__(self):
        self.functions = []
        self.current_class = None
        self.imports = []
        
    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        # Visit children (including methods)
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_FunctionDef(self, node):
        # Calculate complexity (simplified version)
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)):
                complexity += 1
        
        # Get function lines
        start_line = node.lineno
        end_line = 0
        for child in ast.walk(node):
            if hasattr(child, 'lineno'):
                end_line = max(end_line, child.lineno)
        
        # Get function body as code (for similarity comparison)
        function_body = ast.get_source_segment(
            open(self.current_file, 'r').read(), 
            node
        )
        
        # Capture function details
        function_info = {
            'name': node.name,
            'class': self.current_class,
            'file': self.current_file,
            'start_line': start_line,
            'end_line': end_line,
            'complexity': complexity,
            'body': function_body,
            'params': [arg.arg for arg in node.args.args],
            'args_count': len(node.args.args)
        }
        
        self.functions.append(function_info)
        self.generic_visit(node)
    
    def visit_Import(self, node):
        for name in node.names:
            self.imports.append(name.name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        for name in node.names:
            if node.module:
                self.imports.append(f"{node.module}.{name.name}")
            else:
                self.imports.append(name.name)
        self.generic_visit(node)


def extract_functions_from_file(file_path: str) -> Tuple[List[Dict], List[str]]:
    """Extract function information from a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read())
            visitor = FunctionVisitor()
            visitor.current_file = file_path
            visitor.visit(tree)
            return visitor.functions, visitor.imports
        except SyntaxError:
            print(f"Syntax error in {file_path}")
            return [], []


def find_large_files(python_files: List[str]) -> List[Dict]:
    """Find files exceeding the recommended size limit."""
    large_files = []
    
    for file_path in python_files:
        line_count = count_lines_in_file(file_path)
        if line_count > MAX_RECOMMENDED_FILE_LINES:
            large_files.append({
                'file': file_path,
                'lines': line_count,
                'excess': line_count - MAX_RECOMMENDED_FILE_LINES
            })
    
    # Sort by line count descending
    return sorted(large_files, key=lambda x: x['lines'], reverse=True)


def find_complex_functions(functions: List[Dict]) -> List[Dict]:
    """Find functions that are overly complex."""
    complex_functions = []
    
    for func in functions:
        lines = func['end_line'] - func['start_line'] + 1
        if (lines > MAX_RECOMMENDED_FUNCTION_LINES or 
            func['complexity'] > MAX_RECOMMENDED_FUNCTION_COMPLEXITY):
            complex_functions.append({
                'name': func['name'],
                'class': func['class'],
                'file': func['file'],
                'lines': lines,
                'complexity': func['complexity'],
                'start_line': func['start_line'],
                'params': len(func['params'])
            })
    
    # Sort by complexity descending
    return sorted(complex_functions, key=lambda x: x['complexity'], reverse=True)


def detect_similar_functions(functions: List[Dict]) -> List[Dict]:
    """Detect functions with similar implementations."""
    similar_functions = []
    
    # Group functions by parameter count for more meaningful comparison
    functions_by_param_count = defaultdict(list)
    for func in functions:
        functions_by_param_count[func['args_count']].append(func)
    
    # For each group, compare function bodies
    for param_count, funcs in functions_by_param_count.items():
        if len(funcs) < 2:
            continue
        
        for i, func1 in enumerate(funcs):
            for func2 in funcs[i+1:]:
                # Skip functions in the same file that are close to each other (likely overloads)
                if (func1['file'] == func2['file'] and 
                    abs(func1['start_line'] - func2['start_line']) < 20):
                    continue
                
                # Simple similarity check based on function body
                # A more sophisticated approach would use a proper code similarity metric
                body1 = func1['body'].split('\n')
                body2 = func2['body'].split('\n')
                
                # Strip whitespace for better comparison
                body1 = [line.strip() for line in body1 if line.strip()]
                body2 = [line.strip() for line in body2 if line.strip()]
                
                # Skip very small functions
                if len(body1) < MIN_DUPLICATE_LINES or len(body2) < MIN_DUPLICATE_LINES:
                    continue
                
                # Calculate similarity (Jaccard similarity of lines)
                set1 = set(body1)
                set2 = set(body2)
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                similarity = intersection / union if union > 0 else 0
                
                if similarity > 0.5:  # Threshold for similarity
                    similar_functions.append({
                        'func1': f"{func1['class']}.{func1['name']}" if func1['class'] else func1['name'],
                        'func2': f"{func2['class']}.{func2['name']}" if func2['class'] else func2['name'],
                        'file1': func1['file'],
                        'file2': func2['file'],
                        'similarity': similarity,
                        'lines1': func1['end_line'] - func1['start_line'] + 1,
                        'lines2': func2['end_line'] - func2['start_line'] + 1,
                    })
    
    # Sort by similarity descending
    return sorted(similar_functions, key=lambda x: x['similarity'], reverse=True)


def analyze_imports(all_imports: Dict[str, List[str]]) -> Dict:
    """Analyze import patterns to detect potential disconnected components."""
    # Create a directed graph of imports
    G = nx.DiGraph()
    
    # Add nodes (files)
    for file_path in all_imports:
        module_name = file_path.replace('/', '.').replace('\\', '.').replace('.py', '')
        G.add_node(module_name)
    
    # Add edges (imports)
    for file_path, imports in all_imports.items():
        source_module = file_path.replace('/', '.').replace('\\', '.').replace('.py', '')
        for imp in imports:
            if imp in G:
                G.add_edge(source_module, imp)
    
    # Analyze connectivity
    connected_components = list(nx.weakly_connected_components(G))
    isolated_modules = [component for component in connected_components if len(component) == 1]
    
    # Save the dependency graph visualization
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', 
            font_size=8, font_weight='bold', arrows=True, 
            arrowsize=10, width=0.5, alpha=0.7)
    plt.savefig(os.path.join(AUDIT_DIR, 'dependency_graph.png'), dpi=300, bbox_inches='tight')
    
    return {
        'total_modules': len(G.nodes),
        'connected_components': len(connected_components),
        'isolated_modules': [list(m)[0] for m in isolated_modules],
        'largest_component_size': max(len(c) for c in connected_components),
        'graph_density': nx.density(G)
    }


def main():
    print("Starting codebase audit...")
    
    # Get all Python files
    python_files = get_all_python_files()
    print(f"Found {len(python_files)} Python files")
    
    # Extract functions and imports from all files
    all_functions = []
    all_imports = {}
    
    for file_path in python_files:
        functions, imports = extract_functions_from_file(file_path)
        all_functions.extend(functions)
        all_imports[file_path] = imports
    
    print(f"Extracted {len(all_functions)} functions and methods")
    
    # Find large files
    large_files = find_large_files(python_files)
    print(f"Found {len(large_files)} large files")
    
    # Find complex functions
    complex_functions = find_complex_functions(all_functions)
    print(f"Found {len(complex_functions)} complex functions")
    
    # Detect similar functions
    similar_functions = detect_similar_functions(all_functions)
    print(f"Found {len(similar_functions)} similar function pairs")
    
    # Analyze imports
    import_analysis = analyze_imports(all_imports)
    print(f"Import analysis: {len(import_analysis['isolated_modules'])} isolated modules found")
    
    # Save results
    results = {
        'large_files': large_files,
        'complex_functions': complex_functions,
        'similar_functions': similar_functions,
        'import_analysis': import_analysis
    }
    
    with open(os.path.join(AUDIT_DIR, 'audit_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary report
    with open(os.path.join(AUDIT_DIR, 'audit_summary.md'), 'w') as f:
        f.write("# Codebase Audit Summary\n\n")
        
        f.write("## Large Files\n\n")
        f.write("| File | Lines | Excess |\n")
        f.write("|------|-------|--------|\n")
        for file in large_files[:10]:  # Top 10
            f.write(f"| {file['file']} | {file['lines']} | +{file['excess']} |\n")
        
        f.write("\n## Complex Functions\n\n")
        f.write("| Function | File | Lines | Complexity | Parameters |\n")
        f.write("|----------|------|-------|------------|------------|\n")
        for func in complex_functions[:10]:  # Top 10
            class_prefix = f"{func['class']}." if func['class'] else ""
            f.write(f"| {class_prefix}{func['name']} | {func['file']}:{func['start_line']} | {func['lines']} | {func['complexity']} | {func['params']} |\n")
        
        f.write("\n## Similar Function Pairs\n\n")
        f.write("| Function 1 | Function 2 | Similarity | Lines |\n")
        f.write("|------------|------------|------------|-------|\n")
        for pair in similar_functions[:10]:  # Top 10
            f.write(f"| {pair['func1']} ({pair['file1']}) | {pair['func2']} ({pair['file2']}) | {pair['similarity']:.2f} | {pair['lines1']}/{pair['lines2']} |\n")
        
        f.write("\n## Import Graph Analysis\n\n")
        f.write(f"- Total modules: {import_analysis['total_modules']}\n")
        f.write(f"- Connected components: {import_analysis['connected_components']}\n")
        f.write(f"- Largest component size: {import_analysis['largest_component_size']}\n")
        f.write(f"- Graph density: {import_analysis['graph_density']:.4f}\n")
        f.write(f"- Isolated modules: {len(import_analysis['isolated_modules'])}\n")
        
        if import_analysis['isolated_modules']:
            f.write("\n### Isolated Modules\n\n")
            for module in import_analysis['isolated_modules'][:10]:  # Top 10
                f.write(f"- {module}\n")
    
    print(f"Audit complete. Results saved to {AUDIT_DIR}/")
    print(f"Summary report: {AUDIT_DIR}/audit_summary.md")
    print(f"Dependency graph: {AUDIT_DIR}/dependency_graph.png")


if __name__ == "__main__":
    main() 