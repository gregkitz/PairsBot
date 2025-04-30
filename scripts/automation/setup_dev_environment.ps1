#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Sets up the development environment for the trading system.
.DESCRIPTION
    This script installs required tools, sets up pre-commit hooks,
    and configures the development environment.
.NOTES
    Requires Python 3.10+ and pip to be installed.
#>

# Stop on any error
$ErrorActionPreference = "Stop"

# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Display header
Write-Host "====================================================="
Write-Host "   Trading System Development Environment Setup"
Write-Host "====================================================="
Write-Host ""

# Check Python version
Write-Host "Checking Python version..."
$pythonVersion = python --version
if (-not $?) {
    Write-Host "Python not found. Please install Python 3.10 or higher." -ForegroundColor Red
    exit 1
}
Write-Host "Found $pythonVersion" -ForegroundColor Green

# Check if virtual environment exists, create if not
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if (-not $?) {
        Write-Host "Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
if ($PSVersionTable.PSEdition -eq "Core") {
    # PowerShell Core
    & ./venv/bin/Activate.ps1
} else {
    # Windows PowerShell
    & ./venv/Scripts/Activate.ps1
}

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if (-not $?) {
    Write-Host "Failed to install dependencies." -ForegroundColor Red
    exit 1
}
Write-Host "Dependencies installed." -ForegroundColor Green

# Install development tools
Write-Host "Installing development tools..." -ForegroundColor Yellow
pip install pre-commit black flake8 mypy pytest pytest-cov
if (-not $?) {
    Write-Host "Failed to install development tools." -ForegroundColor Red
    exit 1
}
Write-Host "Development tools installed." -ForegroundColor Green

# Install pre-commit hooks
Write-Host "Setting up pre-commit hooks..." -ForegroundColor Yellow
pre-commit install
if (-not $?) {
    Write-Host "Failed to install pre-commit hooks." -ForegroundColor Red
    exit 1
}
Write-Host "Pre-commit hooks installed." -ForegroundColor Green

# Configure Docker desktop if available
if (Get-Command "docker" -ErrorAction SilentlyContinue) {
    Write-Host "Configuring Docker for optimal performance..." -ForegroundColor Yellow
    
    # Check if Docker is using WSL2 backend
    $dockerInfo = docker info | Select-String "Operating System"
    if ($dockerInfo -like "*WSL2*") {
        Write-Host "Docker is using WSL2 backend - Good!" -ForegroundColor Green
    } else {
        Write-Host "Docker might not be using WSL2 backend. For best performance, configure Docker Desktop to use WSL2." -ForegroundColor Yellow
    }

    # Check available memory
    $dockerInfo = docker info | Select-String "Total Memory"
    Write-Host "Docker memory configuration: $dockerInfo" -ForegroundColor Cyan
    Write-Host "Recommended Docker Desktop settings for this machine:" -ForegroundColor Cyan
    Write-Host "  - CPUs: 12-16 cores" -ForegroundColor Cyan
    Write-Host "  - Memory: 32-48 GB" -ForegroundColor Cyan
    Write-Host "  - Swap: 4-8 GB" -ForegroundColor Cyan
    Write-Host "  - Disk image size: 100+ GB" -ForegroundColor Cyan
}

# Success message
Write-Host ""
Write-Host "====================================================="
Write-Host "   Development environment setup complete!"
Write-Host "====================================================="
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Run 'docker-compose up' to start the services"
Write-Host "2. Access the API at http://localhost:8000"
Write-Host "3. Monitor tasks at http://localhost:5555"
Write-Host ""
Write-Host "Happy coding!" -ForegroundColor Green 