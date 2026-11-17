param(
  [string]$Repo = "Fraandm/rift-sense",
  [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$desktop = Join-Path $root "desktop"
$package = Get-Content (Join-Path $desktop "package.json") -Raw | ConvertFrom-Json
$version = $package.version
$tag = "v$version"
$release = Join-Path $root "release"

if (Test-Path $release) {
  Remove-Item $release -Recurse -Force
}
New-Item -ItemType Directory -Path $release | Out-Null

if (-not $SkipBuild) {
  Push-Location $desktop
  npm run dist
  Pop-Location
}

$exe = Join-Path $release "RiftSense.exe"
Copy-Item (Join-Path $desktop "dist\RiftSense.exe") $exe -Force

$stage = Join-Path $release "project"
New-Item -ItemType Directory -Path $stage | Out-Null
foreach ($dir in @("lolanalyzer", "web", "scripts")) {
  Copy-Item (Join-Path $root $dir) (Join-Path $stage $dir) -Recurse -Force
}
foreach ($file in @("README.md", "requirements.txt", "start.ps1", ".env.example")) {
  Copy-Item (Join-Path $root $file) (Join-Path $stage $file) -Force
}

$artifacts = Join-Path $stage "artifacts"
New-Item -ItemType Directory -Path $artifacts | Out-Null
foreach ($file in @("meta.json", "draft_net.pt", "build_net.pt")) {
  $source = Join-Path $root "artifacts\$file"
  if (Test-Path $source) {
    Copy-Item $source $artifacts -Force
  } else {
    Write-Warning "Falta artifacts\$file"
  }
}

$processed = Join-Path $stage "data\processed"
New-Item -ItemType Directory -Path $processed | Out-Null
foreach ($file in @("vocab.json", "priors.json", "summary.json", "timings.json")) {
  $source = Join-Path $root "data\processed\$file"
  if (Test-Path $source) {
    Copy-Item $source $processed -Force
  }
}

$zip = Join-Path $release "RiftSense-project.zip"
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip -Force
Remove-Item $stage -Recurse -Force

$notes = @"
RiftSense v$version

- Actualizaciones automaticas desde el lanzador.
- Conserva data, .venv, .env y la configuracion local.
- Actualiza codigo, modelos y metadatos desde la release.
- Comprueba la version al arrancar mediante GitHub Releases.
"@
$notesFile = Join-Path $root "release-notes.md"
if (Test-Path $notesFile) {
  $notes = Get-Content $notesFile -Raw
}

$exeHash = (Get-FileHash $exe -Algorithm SHA256).Hash.ToLower()
$zipHash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
$manifest = [ordered]@{
  version = $version
  tag = $tag
  notes = $notes
  publishedAt = (Get-Date).ToUniversalTime().ToString("o")
  exe = [ordered]@{
    name = "RiftSense.exe"
    size = (Get-Item $exe).Length
    sha256 = $exeHash
  }
  project = [ordered]@{
    name = "RiftSense-project.zip"
    size = (Get-Item $zip).Length
    sha256 = $zipHash
  }
}
$manifestPath = Join-Path $release "latest.json"
$manifestJson = $manifest | ConvertTo-Json -Depth 6
[System.IO.File]::WriteAllText($manifestPath, $manifestJson, (New-Object System.Text.UTF8Encoding($false)))
$notesPath = Join-Path $release "notes.md"
[System.IO.File]::WriteAllText($notesPath, $notes, (New-Object System.Text.UTF8Encoding($false)))

$existing = $null
try {
  $existing = gh release view $tag --repo $Repo --json tagName 2>$null | ConvertFrom-Json
} catch {
  $existing = $null
}

if ($existing) {
  gh release upload $tag $exe $zip $manifestPath --repo $Repo --clobber
  gh release edit $tag --repo $Repo --notes-file $notesPath --latest
  Write-Host "Release $tag actualizada."
} else {
  gh release create $tag $exe $zip $manifestPath --repo $Repo --title "RiftSense $tag" --notes-file $notesPath --latest
  Write-Host "Release $tag creada."
}

Write-Host ""
Write-Host "Assets:"
Write-Host "  $exe"
Write-Host "  $zip"
Write-Host "  $manifestPath"
Write-Host "Version: $version"
