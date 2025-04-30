#!/usr/bin/env python3
"""
Script to check for docstrings in Python files.

This pre-commit hook checks that functions, classes, and methods have docstrings.
It reports missing docstrings and exits with a non-zero status if any are found.
"""

import ast
import sys
from typing import List, Set, Tuple


class DocstringVisitor(ast.NodeVisitor):
    """AST visitor that checks for missing docstrings in functions, classes, and methods."""

    def __init__(self):
        """Initialize visitor with empty sets of missing docstrings."""
        self.missing_module_docstring = False
        self.missing_class_docstrings: Set[Tuple[str, int]] = set()
        self.missing_function_docstrings: Set[Tuple[str, int]] = set()
        self.missing_method_docstrings: Set[Tuple[str, int]] = set()
        self.current_class = None

    def visit_Module(self, node: ast.Module) -> None:
        """Check if module has a docstring."""
        if not ast.get_docstring(node):
            self.missing_module_docstring = True
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Check if class has a docstring and visit its methods."""
        old_class = self.current_class
        self.current_class = node.name

        if not ast.get_docstring(node):
            self.missing_class_docstrings.add((node.name, node.lineno))

        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check if function or method has a docstring."""
        # Skip __init__ methods with no body except for calling super().__init__
        is_simple_init = (
            node.name == "__init__"
            and len(node.body) <= 1
            and (
                len(node.body) == 0
                or (
                    isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Call)
                    and isinstance(node.body[0].value.func, ast.Attribute)
                    and node.body[0].value.func.attr == "__init__"
                    and isinstance(node.body[0].value.func.value, ast.Call)
                    and isinstance(node.body[0].value.func.value.func, ast.Name)
                    and node.body[0].value.func.value.func.id == "super"
                )
            )
        )

        # Skip dunder methods
        is_dunder = node.name.startswith("__") and node.name.endswith("__") and node.name != "__init__"

        if not ast.get_docstring(node) and not is_simple_init and not is_dunder:
            if self.current_class:
                self.missing_method_docstrings.add((f"{self.current_class}.{node.name}", node.lineno))
            else:
                self.missing_function_docstrings.add((node.name, node.lineno))

        self.generic_visit(node)


def check_file(filename: str) -> int:
    """
    Check a Python file for missing docstrings.

    Args:
        filename: Path to the Python file to check

    Returns:
        0 if all required docstrings are present, 1 otherwise
    """
    with open(filename, "r", encoding="utf-8") as file:
        try:
            tree = ast.parse(file.read(), filename)
        except SyntaxError as e:
            print(f"Syntax error in {filename}: {e}")
            return 1

    visitor = DocstringVisitor()
    visitor.visit(tree)

    # Only show issues for user-defined files
    has_issues = False

    if visitor.missing_module_docstring:
        print(f"{filename}: Missing module docstring")
        has_issues = True

    for name, lineno in sorted(visitor.missing_class_docstrings, key=lambda x: x[1]):
        print(f"{filename}:{lineno}: Missing docstring for class '{name}'")
        has_issues = True

    for name, lineno in sorted(visitor.missing_function_docstrings, key=lambda x: x[1]):
        print(f"{filename}:{lineno}: Missing docstring for function '{name}'")
        has_issues = True

    for name, lineno in sorted(visitor.missing_method_docstrings, key=lambda x: x[1]):
        print(f"{filename}:{lineno}: Missing docstring for method '{name}'")
        has_issues = True

    return 1 if has_issues else 0


def main(filenames: List[str]) -> int:
    """
    Check multiple Python files for missing docstrings.

    Args:
        filenames: List of file paths to check

    Returns:
        0 if all required docstrings are present in all files, 1 otherwise
    """
    return_code = 0
    for filename in filenames:
        if not filename.endswith(".py"):
            continue
        file_return_code = check_file(filename)
        if file_return_code != 0:
            return_code = file_return_code
    return return_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:])) 