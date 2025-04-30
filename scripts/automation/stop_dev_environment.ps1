#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Stops the development environment.
.DESCRIPTION
    This script stops all running Docker containers and performs cleanup.
.EXAMPLE
    ./scripts/automation/stop_dev_environment.ps1
.NOTES
    Requires PowerShell 5.1+ and Docker to be installed.
#>

#Requires -Version 5.1

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

# Define paths
$rootDir = (Get-Item (Split-Path -Parent $MyInvocation.MyCommand.Path)).Parent.Parent.FullName

# Display banner
Write-Host "=================================================================" -ForegroundColor $colors.Title
Write-Host "             STOPPING DEVELOPMENT ENVIRONMENT                    " -ForegroundColor $colors.Title
Write-Host "=================================================================" -ForegroundColor $colors.Title
Write-Host ""

# Check if Docker is running
try {
    $dockerRunning = $null -ne (Get-Command "docker" -ErrorAction SilentlyContinue)
    if (-not $dockerRunning) {
        Write-Host "Docker is not installed or not in PATH. Nothing to stop." -ForegroundColor $colors.Warning
        exit 0
    }
    
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker is not running. Nothing to stop." -ForegroundColor $colors.Warning
        exit 0
    }
}
catch {
    Write-Host "Error checking Docker status: $_" -ForegroundColor $colors.Error
    Write-Host "Docker might not be running. Nothing to stop." -ForegroundColor $colors.Warning
    exit 0
}

# Check if docker-compose.yml exists
$dockerComposeFile = Join-Path -Path $rootDir -ChildPath "docker-compose.yml"
if (-not (Test-Path -Path $dockerComposeFile)) {
    Write-Host "docker-compose.yml not found. Cannot stop containers." -ForegroundColor $colors.Error
    exit 1
}

# Get list of running containers
Write-Host "Checking for running containers..." -ForegroundColor $colors.Info
$runningContainers = docker-compose ps | Select-String "Up"
$containerCount = $runningContainers.Count

if ($containerCount -eq 0) {
    Write-Host "No running containers found." -ForegroundColor $colors.Info
    exit 0
}

Write-Host "Found $containerCount running containers." -ForegroundColor $colors.Info

# Stop containers
Write-Host "Stopping containers..." -ForegroundColor $colors.Info
cd $rootDir
docker-compose down

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to stop containers. You may need to stop them manually:" -ForegroundColor $colors.Error
    Write-Host "docker-compose down --remove-orphans" -ForegroundColor $colors.Info
    exit 1
}

# Verify containers are stopped
$runningContainers = docker-compose ps | Select-String "Up"
if ($runningContainers.Count -gt 0) {
    Write-Host "Some containers are still running. Attempting to force stop..." -ForegroundColor $colors.Warning
    docker-compose down --remove-orphans
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to force stop containers. You may need to stop them manually:" -ForegroundColor $colors.Error
        Write-Host "docker-compose down --remove-orphans" -ForegroundColor $colors.Info
        exit 1
    }
}

# Optional cleanup
$cleanup = $false
if ($cleanup) {
    Write-Host "Performing cleanup..." -ForegroundColor $colors.Info
    
    # Remove unused Docker resources (optional)
    docker system prune -f
    
    # Clean up temporary files
    $tempFiles = @(
        (Join-Path -Path $rootDir -ChildPath "logs\*.log"),
        (Join-Path -Path $rootDir -ChildPath "celery_optimization_report.json")
    )
    
    foreach ($pattern in $tempFiles) {
        Remove-Item -Path $pattern -Force -ErrorAction SilentlyContinue
    }
}

Write-Host ""
Write-Host "=================================================================" -ForegroundColor $colors.Title
Write-Host "          DEVELOPMENT ENVIRONMENT STOPPED SUCCESSFULLY           " -ForegroundColor $colors.Title
Write-Host "=================================================================" -ForegroundColor $colors.Title
Write-Host ""
Write-Host "To restart the environment, run:" -ForegroundColor $colors.Info
Write-Host "./scripts/automation/start_dev_environment.ps1" -ForegroundColor $colors.Info
exit 0 