[CmdletBinding()]
param(
    [string]$Source = "data/input",
    [string]$Model = "models/electrocom61/best.pt",
    [double]$Conf = 0.25,
    [int]$ImgSize = 640,
    [string]$Device = "auto",
    [string]$RunName = "electrocom61-test",
    [switch]$SaveTxt,
    [switch]$SaveCrops,
    [switch]$Open
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating Python virtual environment in .venv"
    $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pythonLauncher) {
        & py -3 -m venv .venv
    } else {
        & python -m venv .venv
    }

    Write-Host "Installing project requirements"
    & $VenvPython -m pip install --upgrade pip
}

Write-Host "Ensuring project requirements"
& $VenvPython -m pip install -r requirements.txt

if (-not (Test-Path $Model)) {
    throw "Model not found: $Model. Expected the trained weight at models/electrocom61/best.pt."
}

if (-not (Test-Path $Source)) {
    if ($Source -eq "data/input") {
        New-Item -ItemType Directory -Force -Path $Source | Out-Null
    }
    throw "Source not found or empty: $Source. Put your photos in data/input or pass -Source with a file/folder path."
}

$PredictArgs = @(
    "--source", $Source,
    "--model", $Model,
    "--conf", "$Conf",
    "--imgsz", "$ImgSize",
    "--device", $Device,
    "--project", "runs/predict",
    "--name", $RunName
)

if ($SaveTxt) {
    $PredictArgs += "--save-txt"
    $PredictArgs += "--save-conf"
}

if ($SaveCrops) {
    $PredictArgs += "--save-crop"
}

Write-Host "Running prediction on $Source"
& $VenvPython scripts/predict.py @PredictArgs

$OutputDir = Join-Path $RepoRoot "runs\predict\$RunName"
if ($Open) {
    explorer.exe $OutputDir
}
