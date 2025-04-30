@echo off
echo Running Enhanced ES Futures Trading System Codebase Audit
echo ========================================================
echo.

:: Create audit directory
if not exist audit\reports mkdir audit\reports

:: Install required dependencies
echo Installing dependencies...
pip install networkx matplotlib pylint

:: Run the audit script
echo.
echo Running audit script...
python scripts\codebase_audit.py

echo.
echo Audit complete! Check the audit/reports directory for results.
echo Summary report: audit/reports/audit_summary.md
echo Dependency graph: audit/reports/dependency_graph.png 