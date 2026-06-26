[CmdletBinding()]
param(
    [ValidateSet("recommended", "fast")]
    [string]$Preset = "recommended",

    [string]$Dataset = "data_t/ElectroCom-61_v2",
    [string]$RunName = "electrocom61-v1",
    [string]$Device = "0",
    [string]$Model = "",
    [int]$Epochs = 0,

    [switch]$Resume,
    [switch]$SkipTorchInstall,
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu121"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

if ($Preset -eq "fast") {
    if ([string]::IsNullOrWhiteSpace($Model)) { $Model = "yolo11n.pt" }
    if ($Epochs -le 0) { $Epochs = 30 }
} else {
    if ([string]::IsNullOrWhiteSpace($Model)) { $Model = "yolo11s.pt" }
    if ($Epochs -le 0) { $Epochs = 50 }
}

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating Python virtual environment in .venv"
    $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pythonLauncher) {
        & py -3 -m venv .venv
    } else {
        & python -m venv .venv
    }
}

Write-Host "Upgrading pip"
& $VenvPython -m pip install --upgrade pip

if (-not $SkipTorchInstall) {
    Write-Host "Installing PyTorch CUDA packages from $TorchIndexUrl"
    & $VenvPython -m pip install --upgrade torch torchvision --index-url $TorchIndexUrl
}

Write-Host "Installing project requirements"
& $VenvPython -m pip install -r requirements.txt

Write-Host "Checking CUDA"
& $VenvPython scripts/check_cuda.py --require-cuda --device $Device

Write-Host "Preparing and validating dataset"
& $VenvPython scripts/prepare_dataset.py --dataset $Dataset --fix

$DataYaml = Join-Path $Dataset "data.yaml"

$TrainArgs = @(
    "--data", $DataYaml,
    "--model", $Model,
    "--epochs", "$Epochs",
    "--imgsz", "640",
    "--device", $Device,
    "--batch", "auto",
    "--project", "runs/detect",
    "--name", $RunName,
    "--copy-best-to", "models/electrocom61/best.pt"
)

if ($Resume) {
    $TrainArgs += "--resume"
}

Write-Host "Starting YOLO training: model=$Model epochs=$Epochs imgsz=640 device=$Device batch=auto"
& $VenvPython scripts/train.py @TrainArgs
