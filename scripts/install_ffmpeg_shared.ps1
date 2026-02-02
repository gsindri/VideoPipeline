param(
  [string]$InstallRoot = "C:\Tools\ffmpeg-shared",
  # TorchCodec currently supports FFmpeg 4/5/6/7, so default to an FFmpeg 7.x shared build.
  [string]$DownloadUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-win64-gpl-shared-7.1.zip",
  [switch]$SkipSetUserEnvVar
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

Write-Section "Config"
Write-Host "InstallRoot: $InstallRoot"
Write-Host "DownloadUrl: $DownloadUrl"

Write-Section "Create install directory"
New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
Assert-Exists $InstallRoot "Failed to create install directory: $InstallRoot"

Write-Section "Download"
try {
  # Ensure TLS 1.2+ is enabled for older PowerShell versions.
  [Net.ServicePointManager]::SecurityProtocol = `
    [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch {
  # No-op if not supported.
}

$tmpDir = Join-Path $env:TEMP ("vp_ffmpeg_shared_" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

$zipPath = Join-Path $tmpDir "ffmpeg-shared.zip"
Invoke-WebRequest -Uri $DownloadUrl -OutFile $zipPath
Assert-Exists $zipPath "Download failed (no file): $zipPath"
Write-Host "Downloaded: $zipPath"

Write-Section "Extract"
$extractDir = Join-Path $tmpDir "extract"
New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

$ffmpegExe = Get-ChildItem -Path $extractDir -Recurse -Filter "ffmpeg.exe" -File | Select-Object -First 1
if (-not $ffmpegExe) {
  throw "Could not find ffmpeg.exe inside extracted archive. Check the DownloadUrl."
}

$binDir = $ffmpegExe.Directory.FullName
$pkgRoot = Split-Path -Parent $binDir
Write-Host "Found ffmpeg.exe: $($ffmpegExe.FullName)"
Write-Host "Package root: $pkgRoot"
Write-Host "Package bin:  $binDir"

Write-Section "Install"
# Copy full package root (bin/include/lib/etc.) so the install is self-contained.
Copy-Item -Path (Join-Path $pkgRoot "*") -Destination $InstallRoot -Recurse -Force

$finalBin = Join-Path $InstallRoot "bin"
Assert-Exists $finalBin "Install failed: missing $finalBin"
Assert-Exists (Join-Path $finalBin "ffmpeg.exe") "Install failed: missing ffmpeg.exe in $finalBin"

Write-Section "Verify FFmpeg shared DLLs"
$need = @(
  @{ Name = "avcodec-*.dll";  Files = @(Get-ChildItem -Path $finalBin -Filter "avcodec-*.dll"  -File -ErrorAction SilentlyContinue) },
  @{ Name = "avformat-*.dll"; Files = @(Get-ChildItem -Path $finalBin -Filter "avformat-*.dll" -File -ErrorAction SilentlyContinue) },
  @{ Name = "avutil-*.dll";   Files = @(Get-ChildItem -Path $finalBin -Filter "avutil-*.dll"   -File -ErrorAction SilentlyContinue) }
)

foreach ($entry in $need) {
  if ($entry.Files.Count -le 0) {
    throw "Missing $($entry.Name) in $finalBin (this is not a shared build)."
  }
}

Write-Host "DLL evidence in ${finalBin}:"
foreach ($entry in $need) {
  Write-Host "  $($entry.Name):"
  $entry.Files | Select-Object -First 10 | ForEach-Object { Write-Host ("    - " + $_.Name) }
  if ($entry.Files.Count -gt 10) {
    Write-Host ("    ... (" + ($entry.Files.Count - 10) + " more)")
  }
}

Write-Section "Optional env var"
$ffmpegSharedBin = $finalBin
if (-not $SkipSetUserEnvVar) {
  [Environment]::SetEnvironmentVariable("FFMPEG_SHARED_BIN", $ffmpegSharedBin, "User")
  Write-Host "Set user env var: FFMPEG_SHARED_BIN=$ffmpegSharedBin"
  Write-Host "Note: User env vars apply to NEW terminals/processes."
} else {
  Write-Host "Skipped setting user env var FFMPEG_SHARED_BIN."
}

Write-Section "Next commands"
Write-Host "PowerShell (current session):"
Write-Host "  `$env:FFMPEG_SHARED_BIN = '$ffmpegSharedBin'"
Write-Host "  python scripts/diagnose_torchcodec.py"
Write-Host ""
Write-Host "CMD (current session):"
Write-Host "  set FFMPEG_SHARED_BIN=$ffmpegSharedBin"
Write-Host "  python scripts/diagnose_torchcodec.py"

Write-Section "Done"
Write-Host "Installed to: $InstallRoot"
