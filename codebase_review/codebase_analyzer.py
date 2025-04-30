#!/usr/bin/env python3
"""
Codebase Analyzer

This script analyzes the quant-trader codebase and generates reports about
its structure, components, and relationships.
"""

import os
import sys
import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

class CodebaseAnalyzer:
    """Analyzes the codebase and generates reports."""
    
    def __init__(self, root_dir: str):
        """Initialize with the root directory of the codebase."""
        self.root_dir = Path(root_dir)
        self.exclude_dirs = {
            '.git', '.venv', 'venv', 'data', '__pycache__', 
            'logs', 'output', 'paper_trading_data', '.pytest_cache'
        }
        self.python_files: List[Path] = []
        self.modules: Dict[str, Dict] = {}
        self.classes: Dict[str, Dict] = {}
        self.functions: Dict[str, Dict] = {}
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.imports: Dict[str, Set[str]] = defaultdict(set)
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the codebase."""
        if self.python_files:
            return self.python_files
            
        for root, dirs, files in os.walk(self.root_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    self.python_files.append(Path(os.path.join(root, file)))
        
        return self.python_files
    
    def _get_docstring(self, node: ast.AST) -> str:
        """Extract docstring from an AST node."""
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            return ""
            
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            return node.body[0].value.s.strip()
        return ""
    
    def _is_method(self, node: ast.FunctionDef, tree: ast.Module) -> bool:
        """Check if a function definition is a method (inside a class)."""
        # Simple heuristic: traverse the tree and check if the function is in a class body
        for potential_class in ast.walk(tree):
            if isinstance(potential_class, ast.ClassDef):
                for child in potential_class.body:
                    if isinstance(child, ast.FunctionDef) and child.name == node.name:
                        return True
        return False
    
    def parse_file(self, file_path: Path) -> Dict:
        """Parse a Python file and extract its components."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = []
            classes = []
            functions = []
            
            # Extract imports
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for name in node.names:
                            imports.append(f"{node.module}.{name.name}")
            
            # Extract classes
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    base_classes = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            base_classes.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            base_classes.append(f"{base.value.id}.{base.attr}" if hasattr(base.value, 'id') else base.attr)
                    
                    methods = []
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            methods.append(child.name)
                    
                    class_info = {
                        'name': node.name,
                        'docstring': self._get_docstring(node),
                        'base_classes': base_classes,
                        'methods': methods,
                        'file_path': str(file_path.relative_to(self.root_dir))
                    }
                    classes.append(class_info)
                    self.classes[f"{file_path.stem}.{node.name}"] = class_info
            
            # Extract top-level functions (not methods)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                    args = []
                    for arg in node.args.args:
                        args.append(arg.arg)
                    
                    function_info = {
                        'name': node.name,
                        'docstring': self._get_docstring(node),
                        'args': args,
                        'file_path': str(file_path.relative_to(self.root_dir))
                    }
                    functions.append(function_info)
                    self.functions[f"{file_path.stem}.{node.name}"] = function_info
            
            # Update dependencies and imports
            relative_path = str(file_path.relative_to(self.root_dir))
            self.imports[relative_path] = set(imports)
            
            module_info = {
                'path': relative_path,
                'docstring': self._get_docstring(tree),
                'classes': classes,
                'functions': functions,
                'imports': imports
            }
            
            self.modules[relative_path] = module_info
            return module_info
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return {}
    
    def analyze_codebase(self) -> None:
        """Analyze the entire codebase."""
        files = self.find_python_files()
        for file in files:
            self.parse_file(file)
        
        # Build dependency graph
        for path, imports in self.imports.items():
            for imp in imports:
                parts = imp.split('.')
                # Check if this is likely an internal import
                if parts[0] in ('src', 'tests') or any(
                    Path(self.root_dir, *parts[0:i]).is_dir() 
                    for i in range(1, len(parts))
                ):
                    self.dependencies[path].add(imp)
    
    def generate_component_inventory(self) -> Dict:
        """Generate a component inventory."""
        inventory = defaultdict(list)
        
        for path, module in self.modules.items():
            parts = path.split(os.sep)
            if len(parts) > 1:
                component_type = parts[0]
                if component_type == 'src' and len(parts) > 2:
                    component_type = parts[1]
                
                inventory[component_type].append({
                    'name': Path(path).stem,
                    'path': path,
                    'docstring': module['docstring'],
                    'classes': len(module['classes']),
                    'functions': len(module['functions']),
                    'dependencies': list(self.dependencies.get(path, set()))
                })
        
        return dict(inventory)
    
    def generate_class_inventory(self) -> Dict:
        """Generate an inventory of classes."""
        inventory = defaultdict(list)
        
        for class_path, class_info in self.classes.items():
            file_path = class_info['file_path']
            parts = file_path.split(os.sep)
            
            if len(parts) > 1:
                component_type = parts[0]
                if component_type == 'src' and len(parts) > 2:
                    component_type = parts[1]
                
                inventory[component_type].append({
                    'name': class_info['name'],
                    'path': file_path,
                    'docstring': class_info['docstring'],
                    'base_classes': class_info['base_classes'],
                    'methods': len(class_info['methods'])
                })
        
        return dict(inventory)
    
    def save_report(self, filename: str, data: Any) -> None:
        """Save a report to a file."""
        output_dir = Path(self.root_dir, 'codebase_review')
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            if filename.endswith('.json'):
                json.dump(data, f, indent=2)
            else:
                f.write(data)
        
        print(f"Report saved to {output_dir / filename}")
    
    def generate_all_reports(self) -> None:
        """Generate all reports."""
        self.analyze_codebase()
        
        # Generate component inventory
        component_inventory = self.generate_component_inventory()
        self.save_report('component_inventory.json', component_inventory)
        
        # Generate class inventory
        class_inventory = self.generate_class_inventory()
        self.save_report('class_inventory.json', class_inventory)
        
        # Generate dependency graph
        self.save_report('dependencies.json', {k: list(v) for k, v in self.dependencies.items()})
        
        # Generate markdown summary
        summary = self._generate_markdown_summary(component_inventory, class_inventory)
        self.save_report('codebase_summary.md', summary)
    
    def _generate_markdown_summary(self, component_inventory: Dict, class_inventory: Dict) -> str:
        """Generate a markdown summary of the codebase."""
        lines = ["# Quant-Trader Codebase Summary\n\n"]
        
        # Summary statistics
        lines.append("## Summary Statistics\n")
        lines.append(f"- Python Files: {len(self.python_files)}")
        lines.append(f"- Modules: {len(self.modules)}")
        lines.append(f"- Classes: {len(self.classes)}")
        lines.append(f"- Functions: {len(self.functions)}")
        lines.append(f"- Components: {sum(len(v) for v in component_inventory.values())}\n")
        
        # Component overview
        lines.append("## Component Overview\n")
        for component_type, components in sorted(component_inventory.items()):
            lines.append(f"### {component_type} ({len(components)})\n")
            for component in sorted(components, key=lambda x: x['name']):
                docstring_summary = component['docstring'].split('\n')[0] if component['docstring'] else "No description"
                lines.append(f"- **{component['name']}** - {docstring_summary}")
                lines.append(f"  - Path: `{component['path']}`")
                lines.append(f"  - Classes: {component['classes']}, Functions: {component['functions']}")
                if component['dependencies']:
                    lines.append(f"  - Dependencies: {', '.join(sorted(component['dependencies'][:5]))}")
                    if len(component['dependencies']) > 5:
                        lines.append(f"    and {len(component['dependencies']) - 5} more...")
                lines.append("")
        
        # Key classes
        lines.append("## Key Classes\n")
        for component_type, classes in sorted(class_inventory.items()):
            # Only include components with classes
            if not classes:
                continue
                
            lines.append(f"### {component_type} ({len(classes)})\n")
            for class_info in sorted(classes, key=lambda x: x['name']):
                docstring_summary = class_info['docstring'].split('\n')[0] if class_info['docstring'] else "No description"
                lines.append(f"- **{class_info['name']}** - {docstring_summary}")
                lines.append(f"  - Path: `{class_info['path']}`")
                if class_info['base_classes']:
                    lines.append(f"  - Inherits from: {', '.join(class_info['base_classes'])}")
                lines.append(f"  - Methods: {class_info['methods']}")
                lines.append("")
        
        return "\n".join(lines)

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = os.getcwd()
    
    analyzer = CodebaseAnalyzer(root_dir)
    analyzer.generate_all_reports()

if __name__ == "__main__":
    main() 