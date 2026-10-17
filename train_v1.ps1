# Version 1 – Standalone BiLSTM
# Uses the BiLSTM branch only.
param(
    [string]$DataPath = "data/lab_maintenance_synthetic_daily.csv",
    [string]$Target = "will_fail_24h",
    [ValidateSet("classification", "regression")]
    [string]$Task = "classification",
    [ValidateSet("random", "time")]
    [string]$SplitMethod = "time",
    [string]$TimeColumn = "timestamp",
    [double]$ValSize = 0.1,
    [int]$Epochs = 5,
    [int]$BatchSize = 16,
    [int]$Patience = 2,
    [int]$Depth = 5,
    [int]$Trees = 50,
    [double]$MaxFpRate = 0.05,
    [string]$OutputDir = "models",
    [double]$BilstmLearningRate = 0.005,
    [double]$XgbLearningRate = 0.002,
    [int]$BilstmUnits = 48,
    [int]$BilstmHiddenLayers = 2
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv310Python = Join-Path $projectRoot ".venv310\Scripts\python.exe"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (Test-Path $venv310Python) {
    $pythonExe = $venv310Python
    Write-Host "Using Python environment: .venv310"
} elseif (Test-Path $venvPython) {
    $pythonExe = $venvPython
    Write-Host "Using Python environment: .venv"
} else {
    Write-Error "Virtual environment not found at .venv310\\Scripts\\python.exe or .venv\\Scripts\\python.exe"
    exit 1
}

$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$cudaBin = Join-Path $cudaRoot "bin"
$cudaLibNvvp = Join-Path $cudaRoot "libnvvp"
if (Test-Path $cudaBin) {
    $env:CUDA_PATH = $cudaRoot
    $env:PATH = "$cudaBin;$cudaLibNvvp;$env:PATH"
    Write-Host "Using CUDA runtime path: $cudaBin"
} else {
    Write-Warning "CUDA bin folder not found at $cudaBin. TensorFlow may run on CPU."
}

Set-Location $projectRoot
$argsList = @(
    "-m", "src.train",
    "--data", $DataPath,
    "--target", $Target,
    "--task", $Task,
    "--model-mode", "bilstm",
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
    "--calibration", "platt"
)

if ($TimeColumn -and $TimeColumn.Trim().Length -gt 0) {
    $argsList += @("--time-column", $TimeColumn)
}

& $pythonExe @argsList
exit $LASTEXITCODE
