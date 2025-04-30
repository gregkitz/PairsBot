#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run code quality metrics analysis and generate dashboard.
.DESCRIPTION
    This script analyzes the codebase for quality metrics and generates
    an HTML dashboard with the results.
.PARAMETER Directory
    The directory to analyze. Defaults to the current directory.
.PARAMETER OutputDir
    The directory to save the report files. Defaults to "reports" in the current directory.
.PARAMETER ExcludeDirs
    Comma-separated list of directories to exclude from analysis.
.PARAMETER ProjectName
    The name of the project to display in the dashboard.
.EXAMPLE
    ./scripts/quality/run_metrics.ps1 -Directory "src" -OutputDir "reports/quality" -ProjectName "Trading System"
#>

param (
    [string]$Directory = ".",
    [string]$OutputDir = "reports",
    [string]$ExcludeDirs = "venv,.git,__pycache__,.idea",
    [string]$ProjectName = "Trading System"
)

# Stop on any error
$ErrorActionPreference = "Stop"

# Define colors for output
$colors = @{
    Success = "Green"
    Error = "Red"
    Warning = "Yellow"
    Info = "Cyan"
    Title = "Magenta"
}

# Get scripts directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = (Get-Item (Split-Path -Parent $scriptDir)).Parent.FullName

# Display banner
Write-Host "=================================================================" -ForegroundColor $colors.Title
Write-Host "                CODE QUALITY METRICS ANALYSIS                    " -ForegroundColor $colors.Title
Write-Host "=================================================================" -ForegroundColor $colors.Title
Write-Host ""

# Ensure output directory exists
if (-not (Test-Path -Path $OutputDir)) {
    Write-Host "Creating output directory: $OutputDir" -ForegroundColor $colors.Info
    New-Item -Path $OutputDir -ItemType Directory -Force | Out-Null
}

# Compute timestamp for filenames
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$metricsFile = Join-Path -Path $OutputDir -ChildPath "code_metrics_$timestamp.json"
$dashboardFile = Join-Path -Path $OutputDir -ChildPath "code_metrics_dashboard_$timestamp.html"

# Convert exclude directories to list
$excludeDirsList = $ExcludeDirs -split ','

# Run metrics analysis
try {
    Write-Host "Running code metrics analysis on $Directory..." -ForegroundColor $colors.Info
    
    # Build arguments list
    $args = @(
        Join-Path -Path $scriptDir -ChildPath "code_metrics.py"
        "--directory", $Directory
        "--output", $metricsFile
    )
    
    # Add exclude directories
    foreach ($dir in $excludeDirsList) {
        $args += "--exclude"
        $args += $dir
    }
    
    Write-Host "Executing: python $($args -join ' ')" -ForegroundColor $colors.Info
    
    # Run the command
    & python $args
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error running code metrics analysis. Exit code: $LASTEXITCODE" -ForegroundColor $colors.Error
        exit $LASTEXITCODE
    }
    
    Write-Host "Code metrics analysis completed." -ForegroundColor $colors.Success
    Write-Host "Report saved to: $metricsFile" -ForegroundColor $colors.Success
}
catch {
    Write-Host "Error running code metrics analysis: $_" -ForegroundColor $colors.Error
    exit 1
}

# Generate dashboard
try {
    Write-Host ""
    Write-Host "Generating dashboard..." -ForegroundColor $colors.Info
    
    # Build arguments
    $args = @(
        Join-Path -Path $scriptDir -ChildPath "code_metrics_dashboard.py"
        "--metrics", $metricsFile
        "--output", $dashboardFile
        "--project", $ProjectName
    )
    
    # Run the command
    & python $args
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error generating dashboard. Exit code: $LASTEXITCODE" -ForegroundColor $colors.Error
        exit $LASTEXITCODE
    }
    
    Write-Host "Dashboard generation completed." -ForegroundColor $colors.Success
    Write-Host "Dashboard saved to: $dashboardFile" -ForegroundColor $colors.Success
    
    # Try to open the dashboard in the default browser
    try {
        Write-Host ""
        Write-Host "Opening dashboard in browser..." -ForegroundColor $colors.Info
        Start-Process $dashboardFile
    }
    catch {
        Write-Host "Could not open dashboard in browser: $_" -ForegroundColor $colors.Warning
        Write-Host "Please open the file manually: $dashboardFile" -ForegroundColor $colors.Info
    }
}
catch {
    Write-Host "Error generating dashboard: $_" -ForegroundColor $colors.Error
    exit 1
}

# Create the latest symlinks/copies
try {
    Write-Host ""
    Write-Host "Creating links to latest reports..." -ForegroundColor $colors.Info
    
    $latestMetricsFile = Join-Path -Path $OutputDir -ChildPath "code_metrics_latest.json"
    $latestDashboardFile = Join-Path -Path $OutputDir -ChildPath "code_metrics_dashboard_latest.html"
    
    # On Windows, we can't create symlinks easily, so copy the files
    Copy-Item -Path $metricsFile -Destination $latestMetricsFile -Force
    Copy-Item -Path $dashboardFile -Destination $latestDashboardFile -Force
    
    Write-Host "Latest links created:" -ForegroundColor $colors.Success
    Write-Host "Latest metrics: $latestMetricsFile" -ForegroundColor $colors.Success
    Write-Host "Latest dashboard: $latestDashboardFile" -ForegroundColor $colors.Success
}
catch {
    Write-Host "Error creating latest links: $_" -ForegroundColor $colors.Warning
}

# Summary
Write-Host ""
Write-Host "=================================================================" -ForegroundColor $colors.Title
Write-Host "                CODE QUALITY ANALYSIS COMPLETE                   " -ForegroundColor $colors.Title
Write-Host "=================================================================" -ForegroundColor $colors.Title
Write-Host ""
Write-Host "To view the results, open: $dashboardFile" -ForegroundColor $colors.Info
Write-Host "Or the latest version: $(Join-Path -Path $OutputDir -ChildPath 'code_metrics_dashboard_latest.html')" -ForegroundColor $colors.Info
Write-Host ""

exit 0 