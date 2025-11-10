# PowerShell script to refresh WhatsApp media ID
# This is designed to be run by Windows Task Scheduler

param(
    [string]$LogLevel = "INFO"
)

# Change to script directory
Set-Location -Path $PSScriptRoot

# Log function
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    Write-Output $logEntry
    Add-Content -Path "task_scheduler.log" -Value $logEntry
}

try {
    Write-Log "Starting WhatsApp media refresh..." "INFO"
    Write-Log "Working directory: $(Get-Location)" "INFO"
    Write-Log "PowerShell version: $($PSVersionTable.PSVersion)" "INFO"
    
    # Run the Python script
    $result = python refresh_media.py
    $exitCode = $LASTEXITCODE
    
    Write-Log "Python script output:" "INFO"
    $result | ForEach-Object { Write-Log "  $_" "INFO" }
    
    if ($exitCode -eq 0) {
        Write-Log "Media refresh completed successfully" "INFO"
    } else {
        Write-Log "Media refresh failed with exit code: $exitCode" "ERROR"
    }
    
    Write-Log "Task completed" "INFO"
    exit $exitCode
}
catch {
    Write-Log "PowerShell error: $($_.Exception.Message)" "ERROR"
    exit 1
}