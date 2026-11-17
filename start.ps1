# RiftSense · inicio rapido (Windows PowerShell)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Creando entorno virtual..." -ForegroundColor Cyan
    py -m venv .venv
    $python = ".\.venv\Scripts\python.exe"
    & $python -m pip install --upgrade pip
    & $python -m pip install -r requirements.txt
}

if (-not (Test-Path "artifacts\meta.json")) {
    Write-Host "Descargando catalogo de Data Dragon..." -ForegroundColor Cyan
    & $python -m lolanalyzer.scraper.ddragon
}

if (-not (Test-Path "data\processed\dataset.npz")) {
    Write-Warning "Falta el dataset. Ejecuta:  .\.venv\Scripts\python.exe scripts\build_all.py"
}

if (-not (Test-Path "artifacts\draft_net.pt")) {
    Write-Warning "DraftNet sin entrenar. La app funciona en modo datos pro."
}

Write-Host "RiftSense -> http://127.0.0.1:8000" -ForegroundColor Green
& $python -m lolanalyzer.api.server
