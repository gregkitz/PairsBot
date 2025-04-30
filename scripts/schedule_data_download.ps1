# PowerShell script to schedule data download task
#
# This script creates a Windows Task Scheduler task to run the download_data.py
# script daily after market close (5:30 PM EST / 17:30 local time).

# Task parameters
$taskName = "QuantTrader_DataDownload"
$taskDescription = "Download daily market data from Interactive Brokers after market close"
$workingDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$pythonExe = "python"  # Use system Python
$scriptPath = Join-Path (Join-Path $workingDir "scripts") "download_data.py"
$logPath = Join-Path (Join-Path $workingDir "logs") "data_download_task.log"

# Create logs directory if it doesn't exist
$logsDir = Join-Path $workingDir "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}

# Command to execute (with output redirection)
$scriptCommand = "$pythonExe `"$scriptPath`" --sequential >> `"$logPath`" 2>&1"

Write-Host "Creating scheduled task: $taskName"
Write-Host "Working directory: $workingDir"
Write-Host "Command: $scriptCommand"

# Create a new scheduled task action
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"$scriptCommand`"" -WorkingDirectory $workingDir

# Create a trigger (daily at 5:30 PM / 17:30)
$trigger = New-ScheduledTaskTrigger -Daily -At "17:30"

# Create a principal (run with highest privileges)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Highest

# Create settings (allow running on demand, start task if missed)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2)

# Register the task
Register-ScheduledTask -TaskName $taskName -Description $taskDescription -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force

Write-Host "Task scheduled successfully! It will run daily at 5:30 PM (17:30)."
Write-Host "You can run the task manually from Task Scheduler or via this command:"
Write-Host "Start-ScheduledTask -TaskName $taskName" 