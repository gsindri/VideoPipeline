param(
  [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Assert-Exists([string]$Path, [string]$Message) {
  if (-not (Test-Path -LiteralPath $Path)) {
    throw $Message
  }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

$resolvedPython = $PythonExe
if (-not $resolvedPython) {
  $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
  if (Test-Path -LiteralPath $venvPython) {
    $resolvedPython = $venvPython
  } else {
    $resolvedPython = "python"
  }
}

Write-Host "PythonExe: $resolvedPython"

# Prefer persisted user env var; fall back to the default install location if present.
$ffmpegSharedBin = [Environment]::GetEnvironmentVariable("FFMPEG_SHARED_BIN", "User")
if (-not $ffmpegSharedBin) {
  $defaultBin = "C:\Tools\ffmpeg-shared\bin"
  if (Test-Path -LiteralPath $defaultBin) {
    $ffmpegSharedBin = $defaultBin
  }
}
if ($ffmpegSharedBin) {
  $env:FFMPEG_SHARED_BIN = $ffmpegSharedBin
  Write-Host "FFMPEG_SHARED_BIN (session): $env:FFMPEG_SHARED_BIN"
} else {
  Write-Host "FFMPEG_SHARED_BIN is not set. Run scripts/setup_ffmpeg_shared.ps1 first."
}

& $resolvedPython -c "import torch; print(torch.__version__)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $resolvedPython -m pip show torchcodec
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $resolvedPython "scripts\diagnose_torchcodec.py" --require-core 7 --require-core 8
exit $LASTEXITCODE

