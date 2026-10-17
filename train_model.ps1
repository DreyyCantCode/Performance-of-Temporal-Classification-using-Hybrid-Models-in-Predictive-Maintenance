param(
    [string]$DataPath = "data/lab_maintenance_synthetic_daily.csv",
    [string]$Target = "will_fail_24h",
    [ValidateSet("classification", "regression")]
    [string]$Task = "classification",
    [ValidateSet("bilstm", "standalone-bilstm", "xgboost", "standalone-xgboost", "hybrid", "hybrid-v1", "hybrid-v2", "hybrid-v3")]
    [string]$ModelMode = "hybrid",
    [ValidateSet("random", "time")]
    [string]$SplitMethod = "time",
    [string]$TimeColumn = "timestamp",
    [double]$ValSize = 0.1,
    [int]$Epochs = 10,
    [int]$BatchSize = 16,
    [int]$Patience = 2,
    [int]$Depth = 4,
    [int]$Trees = 250,
    [double]$MaxFpRate = 0.05,
    [string]$OutputDir = "models",
    [double]$BilstmLearningRate = 0.005,
    [double]$XgbLearningRate = 0.03,
    [int]$BilstmUnits = 48,
    [int]$BilstmHiddenLayers = 2,
    [ValidateSet("auto", "stacked")]
    [string]$RuntimeLock = "stacked"
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Virtual environment not found at .venv\\Scripts\\python.exe"
    exit 1
}

Set-Location $projectRoot
$argsList = @(
    "-m", "src.train",
    "--data", $DataPath,
    "--target", $Target,
    "--task", $Task,
    "--model-mode", $ModelMode,
    "--split-method", $SplitMethod,
    "--val-size", $ValSize,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--patience", $Patience,
    "--output-dir", $OutputDir,
    "--xgb-learning-rate", $XgbLearningRate,
    "--bilstm-learning-rate", $BilstmLearningRate,
    "--bilstm-units", $BilstmUnits,
    "--bilstm-hidden-layers", $BilstmHiddenLayers,
    "--xgb-max-depth", $Depth,
    "--xgb-n-estimators", $Trees,
    "--max-fp-rate", $MaxFpRate,
    "--threshold-strategy", "cost",
    "--fbeta-beta", "2.0",
    "--fn-cost", "5.0",
    "--fp-cost", "1.0",
    "--calibration", "platt",
    "--runtime-lock", $RuntimeLock
)

if ($TimeColumn -and $TimeColumn.Trim().Length -gt 0) {
    $argsList += @("--time-column", $TimeColumn)
}

& $pythonExe @argsList
exit $LASTEXITCODE
