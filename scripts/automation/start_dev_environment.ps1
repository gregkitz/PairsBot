#!/usr/bin/env pwsh
<#
.SYNOPSIS
    One-command startup script for the development environment.
.DESCRIPTION
    This script starts all required services, checks GPU availability,
    and ensures the development environment is ready for use.
.EXAMPLE
    ./scripts/automation/start_dev_environment.ps1
.NOTES
    Requires PowerShell 5.1+ and Docker to be installed.
#>

#Requires -Version 5.1

# Stop on any error
$ErrorActionPreference = "Stop"

# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

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
$scriptsDir = Join-Path -Path $rootDir -ChildPath "scripts"
$configDir = Join-Path -Path $rootDir -ChildPath "config"
$dataDir = Join-Path -Path $rootDir -ChildPath "data"
$outputDir = Join-Path -Path $rootDir -ChildPath "output"
$logsDir = Join-Path -Path $rootDir -ChildPath "logs"

# Create required directories if they don't exist
$requiredDirs = @($scriptsDir, $configDir, $dataDir, $outputDir, $logsDir)
foreach ($dir in $requiredDirs) {
    if (-not (Test-Path -Path $dir)) {
        Write-Host "Creating directory $dir..." -ForegroundColor $colors.Info
        New-Item -Path $dir -ItemType Directory | Out-Null
    }
}

# Display banner
Write-Host "=================================================================" -ForegroundColor $colors.Title
Write-Host "                TRADING SYSTEM DEVELOPMENT ENVIRONMENT           " -ForegroundColor $colors.Title
Write-Host "=================================================================" -ForegroundColor $colors.Title
Write-Host ""

# Function to check prerequisites
function Check-Prerequisites {
    Write-Host "Checking prerequisites..." -ForegroundColor $colors.Info

    # Check Docker
    $dockerInstalled = $null -ne (Get-Command "docker" -ErrorAction SilentlyContinue)
    if (-not $dockerInstalled) {
        Write-Host "Docker is not installed or not in PATH. Please install Docker Desktop." -ForegroundColor $colors.Error
        return $false
    }
    
    # Check Docker is running - simplified to just check container service
    try {
        # Better test for Docker: try to list containers
        $containerCheck = docker ps 2>&1
        # If the above command didn't throw, Docker is running
        Write-Host "Docker is running correctly." -ForegroundColor $colors.Success
        
        # Try to get Docker info, but just for information (ignore errors)
        try {
            $dockerInfo = docker info 2>&1 
            # Filter out warnings 
            $warnings = $dockerInfo | Where-Object { $_ -is [System.Management.Automation.ErrorRecord] }
            if ($warnings) {
                Write-Host "Docker warnings (these can be ignored):" -ForegroundColor $colors.Warning
                foreach ($warn in $warnings) {
                    Write-Host "  $warn" -ForegroundColor $colors.Warning
                }
            }
            
            # Check WSL2 backend (Windows only)
            if ($IsWindows -or $env:OS -match "Windows") {
                $wslCheck = docker info | Select-String "WSL"
                if ($wslCheck) {
                    Write-Host "Docker is using WSL2 backend - Good!" -ForegroundColor $colors.Success
                }
                else {
                    Write-Host "Docker might not be using WSL2 backend. For best performance, configure Docker Desktop to use WSL2." -ForegroundColor $colors.Warning
                }
            }
        }
        catch {
            # Ignore errors from docker info, as long as docker ps worked
            Write-Host "Docker info command had warnings (these can be ignored)." -ForegroundColor $colors.Warning
        }
    }
    catch {
        Write-Host "Docker is not running. Please start Docker Desktop." -ForegroundColor $colors.Error
        return $false
    }
    
    # Check Python
    $pythonInstalled = $null -ne (Get-Command "python" -ErrorAction SilentlyContinue)
    if (-not $pythonInstalled) {
        Write-Host "Python is not installed or not in PATH. Please install Python 3.10 or later." -ForegroundColor $colors.Error
        return $false
    }
    
    $pythonVersion = python --version
    Write-Host "Found $pythonVersion" -ForegroundColor $colors.Success
    
    # All checks passed
    Write-Host "All prerequisites met!" -ForegroundColor $colors.Success
    return $true
}

# Function to check GPU availability in Docker
function Check-GPU {
    Write-Host ""
    Write-Host "Checking GPU support in Docker..." -ForegroundColor $colors.Info
    
    # First check if nvidia-smi is available on the host
    try {
        $nvidiaSmi = & nvidia-smi 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "NVIDIA GPU detected on host system!" -ForegroundColor $colors.Success
        } else {
            Write-Host "nvidia-smi command failed. NVIDIA GPU may not be available or drivers not installed." -ForegroundColor $colors.Warning
            Write-Host "The system will still work, but without GPU acceleration." -ForegroundColor $colors.Warning
            return $false
        }
    } catch {
        Write-Host "nvidia-smi command not found. NVIDIA GPU may not be available." -ForegroundColor $colors.Warning
        Write-Host "The system will still work, but without GPU acceleration." -ForegroundColor $colors.Warning
        return $false
    }
    
    # Run Docker with nvidia-smi to check container GPU access
    try {
        $dockerGpuTest = & docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "GPU access in Docker containers verified!" -ForegroundColor $colors.Success
            return $true
        } else {
            Write-Host "Docker containers cannot access the GPU." -ForegroundColor $colors.Warning
            Write-Host "The system will still work, but without GPU acceleration." -ForegroundColor $colors.Warning
            Write-Host "Make sure you have installed the NVIDIA Container Toolkit and Docker is properly configured." -ForegroundColor $colors.Info
            return $false
        }
    } catch {
        Write-Host "Error testing GPU in Docker: $_" -ForegroundColor $colors.Error
        Write-Host "The system will still work, but without GPU acceleration." -ForegroundColor $colors.Warning
        return $false
    }
}

# Function to build and start Docker containers
function Start-DockerContainers {
    param (
        [bool]$withGPU
    )
    
    Write-Host ""
    Write-Host "Starting Docker containers..." -ForegroundColor $colors.Info
    Write-Host "GPU support enabled: $withGPU" -ForegroundColor $colors.Info
    
    # Check if docker-compose.yml exists
    $dockerComposeFile = Join-Path -Path $rootDir -ChildPath "docker-compose.yml"
    if (-not (Test-Path -Path $dockerComposeFile)) {
        Write-Host "docker-compose.yml not found. Cannot start containers." -ForegroundColor $colors.Error
        return $false
    }
    
    # Build the containers
    Write-Host "Building Docker containers..." -ForegroundColor $colors.Info
    docker-compose build
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to build Docker containers." -ForegroundColor $colors.Error
        return $false
    }
    
    # Start the containers
    Write-Host "Starting Docker containers..." -ForegroundColor $colors.Info
    docker-compose up -d
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to start Docker containers." -ForegroundColor $colors.Error
        return $false
    }
    
    # Wait for containers to be ready
    Write-Host "Waiting for containers to be ready..." -ForegroundColor $colors.Info
    Start-Sleep -Seconds 5
    
    # Check container status
    $containers = docker-compose ps
    Write-Host "Container status:" -ForegroundColor $colors.Info
    Write-Host $containers
    
    # Check if all containers are running
    $runningContainers = docker-compose ps | Select-String "Up"
    if ($runningContainers.Count -lt 3) {  # We expect at least 3 containers (api, worker, redis)
        Write-Host "Some containers are not running. Check the logs for details." -ForegroundColor $colors.Warning
        return $false
    }
    
    Write-Host "All containers are running!" -ForegroundColor $colors.Success
    return $true
}

# Function to display system information
function Show-SystemInfo {
    Write-Host ""
    Write-Host "System Information:" -ForegroundColor $colors.Info
    
    # Get CPU info
    $cpuInfo = Get-WmiObject -Class Win32_Processor | Select-Object -First 1
    Write-Host "CPU: $($cpuInfo.Name) with $($cpuInfo.NumberOfCores) cores" -ForegroundColor $colors.Info
    
    # Get memory info
    $memoryInfo = Get-WmiObject -Class Win32_OperatingSystem
    $totalMemoryGB = [math]::Round($memoryInfo.TotalVisibleMemorySize / 1MB, 2)
    $availableMemoryGB = [math]::Round($memoryInfo.FreePhysicalMemory / 1MB, 2)
    Write-Host "Memory: $availableMemoryGB GB available of $totalMemoryGB GB total" -ForegroundColor $colors.Info
    
    # Get GPU info if nvidia-smi is available
    try {
        $nvidiaSmi = nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
        if ($LASTEXITCODE -eq 0) {
            Write-Host "GPU: $nvidiaSmi" -ForegroundColor $colors.Info
        }
    }
    catch {
        # nvidia-smi not available, ignore
    }
    
    # Get disk info
    $diskInfo = Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='C:'"
    $totalDiskGB = [math]::Round($diskInfo.Size / 1GB, 2)
    $availableDiskGB = [math]::Round($diskInfo.FreeSpace / 1GB, 2)
    Write-Host "Disk: $availableDiskGB GB available of $totalDiskGB GB total" -ForegroundColor $colors.Info
    
    # Get Docker info
    Write-Host ""
    Write-Host "Docker Resource Allocation:" -ForegroundColor $colors.Info
    docker info | Select-String "CPUs|Memory|Running"
}

# Function to show available services
function Show-AvailableServices {
    Write-Host ""
    Write-Host "Available Services:" -ForegroundColor $colors.Info
    Write-Host "- API: http://localhost:8000" -ForegroundColor $colors.Success
    Write-Host "- Flower Dashboard: http://localhost:5555" -ForegroundColor $colors.Success
    Write-Host ""
    Write-Host "To view logs:" -ForegroundColor $colors.Info
    Write-Host "docker-compose logs -f [service_name]" -ForegroundColor $colors.Info
    Write-Host ""
    Write-Host "To stop the environment:" -ForegroundColor $colors.Info
    Write-Host "./scripts/automation/stop_dev_environment.ps1" -ForegroundColor $colors.Info
    Write-Host "or" -ForegroundColor $colors.Info
    Write-Host "docker-compose down" -ForegroundColor $colors.Info
}

# Function to verify data directories
function Verify-DataDirectories {
    Write-Host ""
    Write-Host "Verifying data directories..." -ForegroundColor $colors.Info
    
    $dataFolders = @(
        (Join-Path -Path $dataDir -ChildPath "raw"),
        (Join-Path -Path $dataDir -ChildPath "processed"),
        (Join-Path -Path $outputDir -ChildPath "backtest"),
        (Join-Path -Path $outputDir -ChildPath "models"),
        (Join-Path -Path $logsDir -ChildPath "api"),
        (Join-Path -Path $logsDir -ChildPath "worker")
    )
    
    foreach ($folder in $dataFolders) {
        if (-not (Test-Path -Path $folder)) {
            Write-Host "Creating $folder..." -ForegroundColor $colors.Info
            New-Item -Path $folder -ItemType Directory -Force | Out-Null
        }
    }
    
    # Check if data exists
    $rawDataExists = (Get-ChildItem -Path (Join-Path -Path $dataDir -ChildPath "raw") -File | Measure-Object).Count -gt 0
    if (-not $rawDataExists) {
        Write-Host "No raw data found in data/raw directory." -ForegroundColor $colors.Warning
        Write-Host "You may need to download or generate data before running the system." -ForegroundColor $colors.Warning
    }
    else {
        Write-Host "Raw data found in data/raw directory." -ForegroundColor $colors.Success
    }
}

# Main execution flow
$prerequisites = Check-Prerequisites
if (-not $prerequisites) {
    Write-Host "Prerequisites check failed. Please fix the issues and try again." -ForegroundColor $colors.Error
    exit 1
}

# Check GPU availability
$gpuAvailable = $false  # Default to false
$gpuAvailable = Check-GPU  # This will correctly set to true or false

# Verify data directories
Verify-DataDirectories

# Start Docker containers
$containersStarted = Start-DockerContainers -withGPU $gpuAvailable

if ($containersStarted) {
    Show-SystemInfo
    Show-AvailableServices
    
    Write-Host ""
    Write-Host "=================================================================" -ForegroundColor $colors.Title
    Write-Host "          DEVELOPMENT ENVIRONMENT STARTED SUCCESSFULLY           " -ForegroundColor $colors.Title
    Write-Host "=================================================================" -ForegroundColor $colors.Title
    
    # If running without GPU, remind the user
    if (-not $gpuAvailable) {
        Write-Host ""
        Write-Host "NOTE: The system is running without GPU acceleration." -ForegroundColor $colors.Warning
        Write-Host "For instructions on enabling GPU support, see docs/docker_gpu_setup.md" -ForegroundColor $colors.Info
    }
    
    exit 0
}
else {
    Write-Host ""
    Write-Host "Failed to start the development environment." -ForegroundColor $colors.Error
    Write-Host "Check the logs for more details." -ForegroundColor $colors.Error
    exit 1
} 