param(
    [int]$Port = 5000
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$dashboardUrl = "http://127.0.0.1:$Port/"
Write-Host "Starting dashboard server on port $Port"
Write-Host "Opening: $dashboardUrl"
Write-Host ""

$venvPython = Join-Path $scriptDir ".venv\Scripts\python.exe"
$startupLog = Join-Path $scriptDir "config\dashboard-startup.log"
$startupErrorLog = Join-Path $scriptDir "config\dashboard-startup.error.log"

Remove-Item $startupLog, $startupErrorLog -ErrorAction SilentlyContinue

$pythonCmd = if (Test-Path $venvPython) {
    $venvPython
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    'python'
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    'py'
} else {
    Write-Error 'Python is not available in PATH. Please install Python or add it to PATH.'
    exit 1
}

$env:PORT = "$Port"
$env:PYTHONIOENCODING = 'utf-8'
$serverProcess = Start-Process $pythonCmd -ArgumentList @('app.py') -WorkingDirectory $scriptDir -PassThru -RedirectStandardOutput $startupLog -RedirectStandardError $startupErrorLog

$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline) {
    if ($serverProcess.HasExited) {
        Write-Error "Dashboard server exited during startup. Check $startupErrorLog for details."
        if (Test-Path $startupErrorLog) {
            Get-Content $startupErrorLog
        }
        exit 1
    }

    try {
        Invoke-WebRequest -Uri $dashboardUrl -UseBasicParsing -TimeoutSec 2 | Out-Null
        Start-Process $dashboardUrl
        Write-Host "Dashboard is ready."
        exit 0
    } catch {
        continue
    }
}

Write-Error "Dashboard server did not respond on port $Port within 60 seconds. Check $startupLog and $startupErrorLog."
exit 1