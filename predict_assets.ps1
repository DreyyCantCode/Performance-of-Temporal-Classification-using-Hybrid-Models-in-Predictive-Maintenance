param(
    [string]$DataPath = "data/lab_maintenance_synthetic_daily.csv",
    [string]$ModelDir = "models",
    [string]$OutputCsv = "data/predictions_debug.csv",
    [int]$TopN = 20,
    [switch]$AllowStandalone
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Virtual environment not found at .venv\\Scripts\\python.exe"
    exit 1
}

Set-Location $projectRoot
$argsList = @(
    "-m", "src.predict_xgb",
    "--input-csv", $DataPath,
    "--model-dir", $ModelDir,
    "--output-csv", $OutputCsv,
    "--top-n", $TopN
)

if ($AllowStandalone) {
    $argsList += "--allow-standalone"
}

& $pythonExe @argsList
exit $LASTEXITCODE
