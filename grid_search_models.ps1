param(
    [string]$DataPath = "data/lab_maintenance_synthetic_daily.csv",
    [string]$Target = "will_fail_24h",
    [ValidateSet("classification", "regression")]
    [string]$Task = "classification",
    [ValidateSet("random", "time")]
    [string]$SplitMethod = "time",
    [string]$TimeColumn = "timestamp",
    [double]$ValSize = 0.1,
    [int]$Epochs = 8,
    [int]$Patience = 1,
    [double]$MaxFpRate = 0.10,
    [int]$MaxRuns = 5,
    [string]$OutputDir = "models"
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Virtual environment not found at .venv\Scripts\python.exe"
    exit 1
}

Set-Location $projectRoot
$argsList = @(
    "-m", "src.grid_search_models",
    "--data", $DataPath,
    "--target", $Target,
    "--task", $Task,
    "--split-method", $SplitMethod,
    "--time-column", $TimeColumn,
    "--val-size", $ValSize,
    "--epochs", $Epochs,
    "--patience", $Patience,
    "--max-fp-rate", $MaxFpRate,
    "--max-runs", $MaxRuns,
    "--output-dir", $OutputDir
)

& $pythonExe @argsList
exit $LASTEXITCODE
