# PowerShell script to submit tasks to the containerized API
# This script makes it easy to submit long-running tasks from the command line

param (
    [Parameter(Mandatory=$true)]
    [ValidateSet("train-models", "train-regime-classifier", "backtest", "optimize-parameters")]
    [string]$TaskType,
    
    [Parameter(Mandatory=$false)]
    [string]$Pair = "",
    
    [Parameter(Mandatory=$false)]
    [string]$Timeframe = "",
    
    [Parameter(Mandatory=$false)]
    [string]$StartDate = "",
    
    [Parameter(Mandatory=$false)]
    [string]$EndDate = "",
    
    [Parameter(Mandatory=$false)]
    [string]$PairsFile = "",
    
    [Parameter(Mandatory=$false)]
    [string[]]$Tickers = @(),
    
    [Parameter(Mandatory=$false)]
    [string[]]$Pairs = @(),
    
    [Parameter(Mandatory=$false)]
    [int]$NRegimes = 3,
    
    [Parameter(Mandatory=$false)]
    [switch]$QuickMode = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$UseML = $false,
    
    [Parameter(Mandatory=$false)]
    [string]$ConfigPath = $null,
    
    [Parameter(Mandatory=$false)]
    [switch]$WaitForCompletion = $false
)

$ErrorActionPreference = "Stop"
$ApiUrl = "http://localhost:8000"

# Helper function to check if the API is available
function Test-ApiAvailable {
    try {
        $response = Invoke-RestMethod -Uri "$ApiUrl/health" -Method Get
        if ($response.status -eq "healthy") {
            return $true
        }
        return $false
    } catch {
        return $false
    }
}

# Helper function to check task status
function Get-TaskStatus {
    param (
        [string]$TaskId
    )
    
    try {
        $response = Invoke-RestMethod -Uri "$ApiUrl/tasks/$TaskId" -Method Get
        return $response
    } catch {
        Write-Host "Error checking task status: $_" -ForegroundColor Red
        return $null
    }
}

# Check if API is available
if (-not (Test-ApiAvailable)) {
    Write-Host "API is not available at $ApiUrl" -ForegroundColor Red
    Write-Host "Please ensure the Docker containers are running with scripts/start-containers.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "API is available. Submitting task..." -ForegroundColor Green

# Prepare request body based on task type
$requestBody = @{}

switch ($TaskType) {
    "train-models" {
        if ([string]::IsNullOrEmpty($Pair) -or [string]::IsNullOrEmpty($Timeframe) -or 
            [string]::IsNullOrEmpty($StartDate) -or [string]::IsNullOrEmpty($EndDate)) {
            Write-Host "Error: Pair, Timeframe, StartDate, and EndDate are required for train-models" -ForegroundColor Red
            exit 1
        }
        
        $requestBody = @{
            pair = $Pair
            timeframe = $Timeframe
            start_date = $StartDate
            end_date = $EndDate
        }
        
        if ($ConfigPath) {
            $requestBody.config_path = $ConfigPath
        }
        
        $endpoint = "$ApiUrl/tasks/train-models"
    }
    
    "train-regime-classifier" {
        if ($Tickers.Count -eq 0 -or [string]::IsNullOrEmpty($Timeframe)) {
            Write-Host "Error: Tickers and Timeframe are required for train-regime-classifier" -ForegroundColor Red
            exit 1
        }
        
        $requestBody = @{
            tickers = $Tickers
            timeframe = $Timeframe
            n_regimes = $NRegimes
        }
        
        if ($ConfigPath) {
            $requestBody.config_path = $ConfigPath
        }
        
        $endpoint = "$ApiUrl/tasks/train-regime-classifier"
    }
    
    "backtest" {
        if ($Pairs.Count -eq 0 -or [string]::IsNullOrEmpty($StartDate) -or 
            [string]::IsNullOrEmpty($EndDate) -or [string]::IsNullOrEmpty($Timeframe)) {
            Write-Host "Error: Pairs, StartDate, EndDate, and Timeframe are required for backtest" -ForegroundColor Red
            exit 1
        }
        
        $requestBody = @{
            pairs = $Pairs
            start_date = $StartDate
            end_date = $EndDate
            timeframe = $Timeframe
            use_ml = [bool]$UseML
        }
        
        if ($ConfigPath) {
            $requestBody.config_path = $ConfigPath
        }
        
        $endpoint = "$ApiUrl/tasks/backtest"
    }
    
    "optimize-parameters" {
        if ([string]::IsNullOrEmpty($PairsFile) -or [string]::IsNullOrEmpty($StartDate) -or 
            [string]::IsNullOrEmpty($EndDate)) {
            Write-Host "Error: PairsFile, StartDate, and EndDate are required for optimize-parameters" -ForegroundColor Red
            exit 1
        }
        
        $requestBody = @{
            pairs_file = $PairsFile
            start_date = $StartDate
            end_date = $EndDate
            quick_mode = [bool]$QuickMode
        }
        
        if ($ConfigPath) {
            $requestBody.config_path = $ConfigPath
        }
        
        $endpoint = "$ApiUrl/tasks/optimize-parameters"
    }
}

# Submit the task
try {
    $jsonBody = $requestBody | ConvertTo-Json
    $response = Invoke-RestMethod -Uri $endpoint -Method Post -Body $jsonBody -ContentType "application/json"
    
    $taskId = $response.task_id
    Write-Host "Task submitted successfully!" -ForegroundColor Green
    Write-Host "Task ID: $taskId" -ForegroundColor Cyan
    Write-Host "Task Type: $($response.task_type)" -ForegroundColor Cyan
    Write-Host "Initial Status: $($response.status)" -ForegroundColor Cyan
    
    # If waiting for completion, poll for status
    if ($WaitForCompletion) {
        Write-Host "`nWaiting for task completion..." -ForegroundColor Yellow
        
        $status = "PENDING"
        $completed = $false
        $retryCount = 0
        $maxRetries = 300  # 5 minutes at 1 second intervals
        
        while (-not $completed -and $retryCount -lt $maxRetries) {
            $taskStatus = Get-TaskStatus -TaskId $taskId
            
            if ($taskStatus -ne $null) {
                $status = $taskStatus.status
                
                if ($status -in @("SUCCESS", "FAILURE", "REVOKED")) {
                    $completed = $true
                } else {
                    Start-Sleep -Seconds 1
                    $retryCount++
                    
                    # Show progress every 10 seconds
                    if ($retryCount % 10 -eq 0) {
                        Write-Host "Task still running... Current status: $status" -ForegroundColor Yellow
                    }
                }
            } else {
                Write-Host "Could not retrieve task status" -ForegroundColor Red
                break
            }
        }
        
        if ($completed) {
            if ($status -eq "SUCCESS") {
                Write-Host "`nTask completed successfully!" -ForegroundColor Green
                Write-Host "Result: $($taskStatus.result | ConvertTo-Json -Depth 5)" -ForegroundColor Cyan
            } else {
                Write-Host "`nTask failed with status: $status" -ForegroundColor Red
                if ($taskStatus.traceback) {
                    Write-Host "Error: $($taskStatus.traceback)" -ForegroundColor Red
                }
            }
        } else {
            Write-Host "`nTask is still running after $maxRetries seconds. Check status manually." -ForegroundColor Yellow
        }
    }
    
    Write-Host "`nTo check task status later, run:" -ForegroundColor Yellow
    Write-Host "Invoke-RestMethod -Uri '$ApiUrl/tasks/$taskId' -Method Get | ConvertTo-Json -Depth 5" -ForegroundColor Yellow
    
} catch {
    Write-Host "Error submitting task: $_" -ForegroundColor Red
    exit 1
} 