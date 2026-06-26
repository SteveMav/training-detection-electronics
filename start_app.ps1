[CmdletBinding()]
param()

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

    Write-Host "Upgrading pip"
    & $VenvPython -m pip install --upgrade pip
}

Write-Host "Ensuring project requirements"
& $VenvPython -m pip install -r requirements.txt

Write-Host "Starting ElectroCom-61 tester"
& $VenvPython run_app.py
