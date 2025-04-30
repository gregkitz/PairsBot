#!/usr/bin/env python3
"""
Script to analyze code quality metrics.

This script analyzes the codebase to extract metrics like:
- Cyclomatic complexity
- Lines of code
- Documentation coverage
- Duplicate code
- Type hints usage

It generates a JSON report with the results.
"""

import os
import sys
import json
import ast
import re
import time
import argparse
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor to calculate cyclomatic complexity."""
    
    def __init__(self):
        """Initialize the visitor."""
        self.complexity = 1  # Start with 1 for the function itself
        self.functions = {}
    
    def visit_FunctionDef(self, node):
        """Visit function definition."""
        old_complexity = self.complexity
        self.complexity = 1
        
        # Visit the function body
        for child in node.body:
            self.visit(child)
        
        # Store the complexity for this function
        self.functions[node.name] = self.complexity
        
        # Restore the parent complexity
        self.complexity = old_complexity
    
    def visit_If(self, node):
        """Visit if statement."""
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_While(self, node):
        """Visit while loop."""
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_For(self, node):
        """Visit for loop."""
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_Try(self, node):
        """Visit try block."""
        # Add 1 for try and 1 for each except handler
        self.complexity += 1 + len(node.handlers)
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node):
        """Visit except handler."""
        # Already counted in visit_Try
        self.generic_visit(node)
    
    def visit_BoolOp(self, node):
        """Visit boolean operation."""
        # Add complexity for boolean operations (and, or)
        self.complexity += len(node.values) - 1
        self.generic_visit(node)
    
    def visit_BinOp(self, node):
        """Visit binary operation."""
        self.generic_visit(node)
    
    def visit_Lambda(self, node):
        """Visit lambda function."""
        self.complexity += 1
        self.generic_visit(node)


class TypeHintVisitor(ast.NodeVisitor):
    """AST visitor to check for type hints usage."""
    
    def __init__(self):
        """Initialize the visitor."""
        self.has_hints = 0
        self.missing_hints = 0
        self.functions = 0
        self.classes = 0
        self.all_functions = set()
        self.typed_functions = set()
    
    def visit_FunctionDef(self, node):
        """Visit function definition."""
        self.functions += 1
        self.all_functions.add(node.name)
        
        # Check return type annotation
        has_return_annotation = node.returns is not None
        
        # Check argument annotations
        args_with_annotation = sum(1 for arg in node.args.args if arg.annotation is not None)
        total_args = len(node.args.args)
        
        # A function is considered typed if it has a return annotation and all arguments are annotated
        if has_return_annotation and (args_with_annotation == total_args) and total_args > 0:
            self.has_hints += 1
            self.typed_functions.add(node.name)
        else:
            self.missing_hints += 1
        
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Visit class definition."""
        self.classes += 1
        self.generic_visit(node)


class DocstringVisitor(ast.NodeVisitor):
    """AST visitor to check for docstrings."""
    
    def __init__(self):
        """Initialize the visitor."""
        self.has_docstring = 0
        self.missing_docstring = 0
        self.modules = 0
        self.functions = 0
        self.classes = 0
        self.all_functions = set()
        self.documented_functions = set()
        self.all_classes = set()
        self.documented_classes = set()
    
    def visit_Module(self, node):
        """Visit module."""
        self.modules += 1
        if ast.get_docstring(node):
            self.has_docstring += 1
        else:
            self.missing_docstring += 1
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Visit function definition."""
        self.functions += 1
        self.all_functions.add(node.name)
        
        # Skip dunder methods except for __init__
        is_dunder = node.name.startswith("__") and node.name.endswith("__") and node.name != "__init__"
        
        if ast.get_docstring(node) and not is_dunder:
            self.has_docstring += 1
            self.documented_functions.add(node.name)
        else:
            self.missing_docstring += 1
        
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Visit class definition."""
        self.classes += 1
        self.all_classes.add(node.name)
        
        if ast.get_docstring(node):
            self.has_docstring += 1
            self.documented_classes.add(node.name)
        else:
            self.missing_docstring += 1
        
        self.generic_visit(node)


def count_lines(file_path: str) -> Dict[str, int]:
    """
    Count lines in a file, categorized by type.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with line counts by category
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        code_lines = 0
        blank_lines = 0
        comment_lines = 0
        docstring_lines = 0
        
        in_docstring = False
        docstring_delimiter = None
        
        for line in lines:
            stripped = line.strip()
            
            # Handle docstrings
            if in_docstring:
                docstring_lines += 1
                if stripped.endswith(docstring_delimiter):
                    in_docstring = False
                    docstring_delimiter = None
                continue
            
            # Check for docstring start
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = True
                docstring_delimiter = stripped[0:3]
                docstring_lines += 1
                if stripped.endswith(docstring_delimiter) and len(stripped) > 3:
                    in_docstring = False
                    docstring_delimiter = None
                continue
            
            # Blank lines
            if not stripped:
                blank_lines += 1
                continue
            
            # Comments
            if stripped.startswith('#'):
                comment_lines += 1
                continue
            
            # Code lines
            code_lines += 1
        
        return {
            'total': len(lines),
            'code': code_lines,
            'blank': blank_lines,
            'comment': comment_lines,
            'docstring': docstring_lines
        }
    except Exception as e:
        print(f"Error counting lines in {file_path}: {e}")
        return {
            'total': 0,
            'code': 0,
            'blank': 0,
            'comment': 0,
            'docstring': 0,
            'error': str(e)
        }


def analyze_file(file_path: str) -> Dict[str, Any]:
    """
    Analyze a single Python file for metrics.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Dictionary with file metrics
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Parse the AST
        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError as e:
            return {
                'path': file_path,
                'error': f"Syntax error: {str(e)}",
                'traceback': traceback.format_exc()
            }
        
        # Analyze complexity
        complexity_visitor = ComplexityVisitor()
        complexity_visitor.visit(tree)
        functions_complexity = complexity_visitor.functions
        
        # Calculate max and average complexity
        if functions_complexity:
            max_complexity = max(functions_complexity.values())
            avg_complexity = sum(functions_complexity.values()) / len(functions_complexity)
        else:
            max_complexity = 0
            avg_complexity = 0
        
        # Analyze type hints
        type_hint_visitor = TypeHintVisitor()
        type_hint_visitor.visit(tree)
        
        # Calculate type hint usage percentage
        if type_hint_visitor.functions > 0:
            type_hint_percentage = (type_hint_visitor.has_hints / type_hint_visitor.functions) * 100
        else:
            type_hint_percentage = 0
        
        # Analyze docstrings
        docstring_visitor = DocstringVisitor()
        docstring_visitor.visit(tree)
        
        # Calculate docstring coverage percentage
        total_documentable = docstring_visitor.modules + docstring_visitor.functions + docstring_visitor.classes
        if total_documentable > 0:
            docstring_percentage = (docstring_visitor.has_docstring / total_documentable) * 100
        else:
            docstring_percentage = 0
        
        # Count lines
        line_counts = count_lines(file_path)
        
        # Return the metrics
        return {
            'path': file_path,
            'size': os.path.getsize(file_path),
            'line_counts': line_counts,
            'complexity': {
                'max': max_complexity,
                'average': avg_complexity,
                'functions': functions_complexity
            },
            'type_hints': {
                'percentage': type_hint_percentage,
                'functions_with_hints': type_hint_visitor.has_hints,
                'functions_missing_hints': type_hint_visitor.missing_hints,
                'total_functions': type_hint_visitor.functions
            },
            'docstrings': {
                'percentage': docstring_percentage,
                'has_docstring': docstring_visitor.has_docstring,
                'missing_docstring': docstring_visitor.missing_docstring,
                'total_documentable': total_documentable
            }
        }
    except Exception as e:
        return {
            'path': file_path,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def find_python_files(directory: str, exclude_dirs: List[str] = None) -> List[str]:
    """
    Find all Python files in a directory tree.
    
    Args:
        directory: Root directory to search
        exclude_dirs: List of directories to exclude
        
    Returns:
        List of Python file paths
    """
    if exclude_dirs is None:
        exclude_dirs = []
    
    # Convert to absolute paths for proper comparison
    exclude_dirs = [os.path.abspath(os.path.join(directory, d)) for d in exclude_dirs]
    
    python_files = []
    
    try:
        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(root, d)) not in exclude_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
    except Exception as e:
        print(f"Error walking directory {directory}: {e}")
    
    return python_files


def find_duplicate_code(file_list: List[str], min_lines: int = 5) -> List[Dict[str, Any]]:
    """
    Find duplicate code blocks in the given files.
    
    Args:
        file_list: List of Python file paths
        min_lines: Minimum number of lines to consider a duplicate
        
    Returns:
        List of dictionaries with duplicate code blocks
    """
    try:
        # Try to use pmd-cpd if available
        try:
            result = subprocess.run(
                ['pmd', 'cpd', '--minimum-tokens', str(min_lines * 5), '--files'] + file_list,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse PMD output
            duplicates = []
            current_duplicate = None
            
            for line in result.stdout.splitlines():
                if line.startswith('Found a '):
                    if current_duplicate:
                        duplicates.append(current_duplicate)
                    
                    tokens = line.split()
                    lines = int(tokens[3])
                    
                    current_duplicate = {
                        'lines': lines,
                        'tokens': int(tokens[6]),
                        'occurrences': []
                    }
                
                elif line.startswith('Starting at line'):
                    tokens = line.split()
                    file_path = ' '.join(tokens[6:])
                    line_number = int(tokens[3])
                    
                    if current_duplicate:
                        current_duplicate['occurrences'].append({
                            'file': file_path,
                            'line': line_number
                        })
            
            if current_duplicate:
                duplicates.append(current_duplicate)
            
            return duplicates
        
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"PMD copy-paste detector not available: {e}")
            print("Falling back to built-in duplicate code detection")
    
    except Exception as e:
        print(f"Error using PMD: {e}")
        print("Falling back to built-in duplicate code detection")
    
    # Fallback implementation of duplicate code detection
    duplicates = []
    file_contents = {}
    
    # Read all files
    for file_path in file_list:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                file_contents[file_path] = lines
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    # Find duplicates
    line_signatures = {}
    
    for file_path, lines in file_contents.items():
        for i in range(len(lines) - min_lines + 1):
            # Create a signature for this block of code
            block = ''.join(lines[i:i+min_lines]).strip()
            if len(block) < 10:  # Skip very short blocks
                continue
            
            signature = hash(block)
            
            if signature in line_signatures:
                # Found a duplicate
                line_signatures[signature]['occurrences'].append({
                    'file': file_path,
                    'line': i + 1
                })
            else:
                line_signatures[signature] = {
                    'lines': min_lines,
                    'tokens': len(block),
                    'occurrences': [{
                        'file': file_path,
                        'line': i + 1
                    }]
                }
    
    # Filter out unique blocks
    for signature, data in line_signatures.items():
        if len(data['occurrences']) > 1:
            duplicates.append(data)
    
    return duplicates


def generate_metrics_report(directory: str, output_file: str, exclude_dirs: List[str] = None, 
                           min_duplicate_lines: int = 5, parallel: bool = True):
    """
    Generate a comprehensive code metrics report.
    
    Args:
        directory: Root directory to analyze
        output_file: Path to save the report
        exclude_dirs: List of directories to exclude
        min_duplicate_lines: Minimum lines to consider a duplicate
        parallel: Whether to use parallel processing
    """
    start_time = time.time()
    
    try:
        # Find all Python files
        python_files = find_python_files(directory, exclude_dirs)
        
        if not python_files:
            print(f"No Python files found in {directory}")
            return {
                'timestamp': datetime.now().isoformat(),
                'directory': directory,
                'files_analyzed': 0,
                'error': 'No Python files found',
                'duration_seconds': time.time() - start_time
            }
        
        # Analyze files
        file_metrics = []
        
        if parallel:
            try:
                # Use parallel processing
                with ProcessPoolExecutor() as executor:
                    file_metrics = list(executor.map(analyze_file, python_files))
            except Exception as e:
                print(f"Error in parallel processing: {e}")
                print("Falling back to sequential processing")
                # Process sequentially if parallel fails
                file_metrics = [analyze_file(file_path) for file_path in python_files]
        else:
            # Process sequentially
            file_metrics = [analyze_file(file_path) for file_path in python_files]
        
        # Count files with errors
        files_with_errors = sum(1 for m in file_metrics if 'error' in m)
        if files_with_errors > 0:
            print(f"Warning: {files_with_errors} files had errors during analysis")
        
        # Find duplicate code
        duplicates = find_duplicate_code(
            [m['path'] for m in file_metrics if 'error' not in m], 
            min_duplicate_lines
        )
        
        # Calculate aggregate metrics
        valid_metrics = [m for m in file_metrics if 'error' not in m]
        if not valid_metrics:
            print("Warning: No files could be analyzed successfully")
            return {
                'timestamp': datetime.now().isoformat(),
                'directory': directory,
                'files_analyzed': len(python_files),
                'files_with_errors': files_with_errors,
                'error': 'No files could be analyzed successfully',
                'file_metrics': file_metrics,
                'duration_seconds': time.time() - start_time
            }
        
        total_lines = sum(m.get('line_counts', {}).get('total', 0) for m in valid_metrics)
        code_lines = sum(m.get('line_counts', {}).get('code', 0) for m in valid_metrics)
        
        # Calculate complexity metrics
        complexity_values = [m.get('complexity', {}).get('max', 0) for m in valid_metrics]
        max_complexity = max(complexity_values) if complexity_values else 0
        
        avg_complexity_values = [m.get('complexity', {}).get('average', 0) for m in valid_metrics]
        avg_complexity = sum(avg_complexity_values) / len(avg_complexity_values) if avg_complexity_values else 0
        
        # Calculate docstring coverage
        total_documentable = sum(m.get('docstrings', {}).get('total_documentable', 0) for m in valid_metrics)
        total_documented = sum(m.get('docstrings', {}).get('has_docstring', 0) for m in valid_metrics)
        docstring_coverage = (total_documented / total_documentable) * 100 if total_documentable > 0 else 0
        
        # Calculate type hint coverage
        total_functions = sum(m.get('type_hints', {}).get('total_functions', 0) for m in valid_metrics)
        functions_with_hints = sum(m.get('type_hints', {}).get('functions_with_hints', 0) for m in valid_metrics)
        type_hint_coverage = (functions_with_hints / total_functions) * 100 if total_functions > 0 else 0
        
        # Create the report
        report = {
            'timestamp': datetime.now().isoformat(),
            'directory': directory,
            'files_analyzed': len(python_files),
            'files_with_errors': files_with_errors,
            'summary': {
                'total_lines': total_lines,
                'code_lines': code_lines,
                'max_complexity': max_complexity,
                'avg_complexity': avg_complexity,
                'docstring_coverage': docstring_coverage,
                'type_hint_coverage': type_hint_coverage,
                'duplicate_blocks': len(duplicates)
            },
            'file_metrics': file_metrics,
            'duplicates': duplicates,
            'duration_seconds': time.time() - start_time
        }
        
        # Save the report
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            print(f"Error saving report to {output_file}: {e}")
            print("Report will be returned but not saved.")
            report['save_error'] = str(e)
        
        return report
    
    except Exception as e:
        error_report = {
            'timestamp': datetime.now().isoformat(),
            'directory': directory,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'duration_seconds': time.time() - start_time
        }
        
        # Try to save error report
        try:
            with open(output_file, 'w') as f:
                json.dump(error_report, f, indent=2)
        except Exception:
            pass
        
        return error_report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate code quality metrics report')
    parser.add_argument('--directory', '-d', default='.', help='Directory to analyze')
    parser.add_argument('--output', '-o', default='code_metrics_report.json', help='Output report file')
    parser.add_argument('--exclude', '-e', action='append', default=['venv', '.git', '__pycache__'], 
                        help='Directories to exclude')
    parser.add_argument('--min-duplicate-lines', '-m', type=int, default=5, 
                        help='Minimum lines to consider a duplicate')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    
    try:
        args = parser.parse_args()
        
        print(f"Analyzing code in {args.directory}...")
        report = generate_metrics_report(
            args.directory,
            args.output,
            args.exclude,
            args.min_duplicate_lines,
            not args.no_parallel
        )
        
        if 'error' in report and 'summary' not in report:
            print(f"\nError during analysis: {report['error']}")
            return 1
        
        # Print summary
        print("\nAnalysis complete!")
        print(f"Files analyzed: {report['files_analyzed']}")
        
        if 'files_with_errors' in report and report['files_with_errors'] > 0:
            print(f"Files with errors: {report['files_with_errors']}")
        
        if 'summary' in report:
            print(f"Total lines: {report['summary']['total_lines']}")
            print(f"Code lines: {report['summary']['code_lines']}")
            print(f"Max complexity: {report['summary']['max_complexity']:.1f}")
            print(f"Avg complexity: {report['summary']['avg_complexity']:.1f}")
            print(f"Docstring coverage: {report['summary']['docstring_coverage']:.1f}%")
            print(f"Type hint coverage: {report['summary']['type_hint_coverage']:.1f}%")
            print(f"Duplicate blocks: {report['summary']['duplicate_blocks']}")
        
        if 'save_error' in report:
            print(f"Warning: Could not save report to {args.output}: {report['save_error']}")
        else:
            print(f"Report saved to {args.output}")
        
        print(f"Duration: {report['duration_seconds']:.2f} seconds")
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main()) 