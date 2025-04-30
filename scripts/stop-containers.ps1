# PowerShell script to stop Docker containers for the quantitative trading system
# This script safely stops all containerized services

$ErrorActionPreference = "Stop"
$ScriptName = $MyInvocation.MyCommand.Name

# Move to the root directory of the project
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootPath = Split-Path -Parent $scriptPath
Set-Location $rootPath

# Check if docker-compose.yml exists
if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "Error: docker-compose.yml not found in $rootPath" -ForegroundColor Red
    exit 1
}

Write-Host "Stopping containers..." -ForegroundColor Yellow

try {
    # Stop the containers
    docker-compose down

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error stopping containers. See the output above for details." -ForegroundColor Red
        exit 1
    }

    Write-Host "Containers stopped successfully." -ForegroundColor Green
    
} catch {
    Write-Host "Error stopping containers: $_" -ForegroundColor Red
    
    # Try to force stop if regular stop fails
    Write-Host "Attempting to force stop containers..." -ForegroundColor Yellow
    try {
        docker-compose down -v --remove-orphans
    } catch {
        Write-Host "Failed to force stop containers. You may need to stop them manually." -ForegroundColor Red
        exit 1
    }
}

# Optionally check if any containers are still running
try {
    $runningContainers = docker ps --filter "name=quant-trader" -q
    if ($runningContainers) {
        Write-Host "Warning: Some containers may still be running. Check with 'docker ps'." -ForegroundColor Yellow
    }
} catch {
    Write-Host "Error checking for running containers: $_" -ForegroundColor Red
}

Write-Host "Shutdown complete!" -ForegroundColor Green 