# PowerShell script to start Docker containers for the quantitative trading system
# This script ensures Docker is running and starts the containerized services

$ErrorActionPreference = "Stop"
$ScriptName = $MyInvocation.MyCommand.Name

# Check if Docker is running
try {
    $dockerStatus = docker ps > $null 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker is not running. Please start Docker Desktop and try again." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Error checking Docker status: $_" -ForegroundColor Red
    Write-Host "Please ensure Docker Desktop is installed and running." -ForegroundColor Red
    exit 1
}

Write-Host "Docker is running. Starting containers..." -ForegroundColor Green

# Move to the root directory of the project
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootPath = Split-Path -Parent $scriptPath
Set-Location $rootPath

# Check if docker-compose.yml exists
if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "Error: docker-compose.yml not found in $rootPath" -ForegroundColor Red
    exit 1
}

try {
    # Start the containers in detached mode
    Write-Host "Starting containers with docker-compose..." -ForegroundColor Yellow
    docker-compose up -d

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error starting containers. See the output above for details." -ForegroundColor Red
        exit 1
    }

    # Display container status
    Write-Host "`nContainers are now running. Status:" -ForegroundColor Green
    docker-compose ps

    # Display service URLs
    Write-Host "`nServices available at:" -ForegroundColor Cyan
    Write-Host " - API: http://localhost:8000" -ForegroundColor Cyan
    Write-Host " - Flower Dashboard: http://localhost:5555" -ForegroundColor Cyan

    Write-Host "`nUse 'docker-compose logs -f' to view logs" -ForegroundColor Yellow
    Write-Host "Use 'scripts/stop-containers.ps1' to stop the containers" -ForegroundColor Yellow
    
} catch {
    Write-Host "Error starting containers: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`nStartup complete!" -ForegroundColor Green 