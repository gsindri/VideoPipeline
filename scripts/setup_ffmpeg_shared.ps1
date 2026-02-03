param(
  [string]$InstallRoot = "C:\Tools\ffmpeg-shared",
  [string]$DownloadUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-win64-gpl-shared-7.1.zip",
  [switch]$SkipDownload,
  [switch]$SkipSetUserEnvVar,
  [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Write-Section([string]$Title) {
  Write-Host ""
  Write-Host "== $Title =="
}

function Assert-Exists([string]$Path, [string]$Message) {
  if (-not (Test-Path -LiteralPath $Path)) {
    throw $Message
  }
}

function Test-HasSharedFfmpegDlls([string]$BinDir) {
  $need = @(
    "avcodec-*.dll",
    "avformat-*.dll",
    "avutil-*.dll"
  )
  foreach ($pat in $need) {
    $found = @(Get-ChildItem -Path $BinDir -Filter $pat -File -ErrorAction SilentlyContinue)
    if ($found.Count -le 0) { return $false }
  }
  return $true
}

Write-Section "Config"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$installScript = Join-Path $repoRoot "scripts\install_ffmpeg_shared.ps1"
Assert-Exists $installScript "Missing script: $installScript"

$finalBin = Join-Path $InstallRoot "bin"
Write-Host "InstallRoot: $InstallRoot"
Write-Host "FinalBin:    $finalBin"
Write-Host "DownloadUrl: $DownloadUrl"
Write-Host "SkipDownload: $SkipDownload"
Write-Host "SkipSetUserEnvVar: $SkipSetUserEnvVar"

Write-Section "Ensure FFmpeg shared install"
if (-not $SkipDownload) {
  $hasExe = Test-Path -LiteralPath (Join-Path $finalBin "ffmpeg.exe")
  $hasDlls = $false
  if (Test-Path -LiteralPath $finalBin) {
    $hasDlls = Test-HasSharedFfmpegDlls $finalBin
  }

  if ($hasExe -and $hasDlls) {
    Write-Host "Already present: ffmpeg.exe and shared DLLs in $finalBin"
  } else {
    Write-Host "Running installer: $installScript"
    $args = @(
      "-ExecutionPolicy", "Bypass",
      "-File", $installScript,
      "-InstallRoot", $InstallRoot,
      "-DownloadUrl", $DownloadUrl
    )
    if ($SkipSetUserEnvVar) {
      $args += "-SkipSetUserEnvVar"
    }
    & powershell.exe @args
    if ($LASTEXITCODE -ne 0) {
      throw "FFmpeg shared installer failed with exit code: $LASTEXITCODE"
    }
  }
} else {
  Write-Host "SkipDownload set: not downloading/installing. Will only validate existing install."
}

Assert-Exists $finalBin "FFmpeg bin directory missing: $finalBin"
Assert-Exists (Join-Path $finalBin "ffmpeg.exe") "Missing ffmpeg.exe in: $finalBin"
if (-not (Test-HasSharedFfmpegDlls $finalBin)) {
  throw "Missing required FFmpeg shared DLLs in $finalBin (need avcodec/avformat/avutil)."
}

Write-Section "Set environment variable"
$env:FFMPEG_SHARED_BIN = $finalBin
Write-Host "Set for current session: FFMPEG_SHARED_BIN=$finalBin"

if (-not $SkipSetUserEnvVar) {
  [Environment]::SetEnvironmentVariable("FFMPEG_SHARED_BIN", $finalBin, "User")
  Write-Host "Set persistently (User): FFMPEG_SHARED_BIN=$finalBin"
  Write-Host "Note: user env vars apply to NEW terminals/processes."
} else {
  Write-Host "Skipped persistent user env var."
}

Write-Section "Verify from PowerShell + Python"
Write-Host "User env var readback: $([Environment]::GetEnvironmentVariable('FFMPEG_SHARED_BIN','User'))"

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

& $resolvedPython -c "import os; print('FFMPEG_SHARED_BIN from python:', os.environ.get('FFMPEG_SHARED_BIN'))"
if ($LASTEXITCODE -ne 0) {
  throw "Python verification failed with exit code: $LASTEXITCODE"
}

Write-Section "Done"
Write-Host "FFmpeg shared is ready."
