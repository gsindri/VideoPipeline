<#
.SYNOPSIS
    Build VideoPipeline as a Windows executable using PyInstaller.

.DESCRIPTION
    This script builds VideoPipeline.exe from source.
    Run from the repo root directory.

.PARAMETER OneFile
    Build a single-file executable (slower startup, more AV false positives).
    Default is a one-directory build.

.PARAMETER Console
    Show console window (useful for debugging).
    Default is no console (windowed mode).

.EXAMPLE
    .\scripts\build_exe.ps1
    # Builds dist\VideoPipeline\VideoPipeline.exe (no console)

.EXAMPLE
    .\scripts\build_exe.ps1 -Console
    # Builds with console visible (for debugging)

.EXAMPLE
    .\scripts\build_exe.ps1 -OneFile
    # Builds dist\VideoPipeline.exe (single file)
#>

param(
    [switch]$OneFile,
    [switch]$Console
)

$ErrorActionPreference = "Stop"

# Ensure we're running from repo root
if (-not (Test-Path "pyproject.toml")) {
    throw "Run this script from the repo root directory."
}

Write-Host "=== VideoPipeline EXE Builder ===" -ForegroundColor Cyan
Write-Host ""

# Activate venv if present
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    . .\.venv\Scripts\Activate.ps1
}

# Upgrade pip and install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
pip install -e . --quiet
pip install pyinstaller --quiet

Write-Host "Dependencies installed." -ForegroundColor Green
Write-Host ""

# Build PyInstaller arguments
$modeArgs = @("--clean", "--name", "VideoPipeline")

if ($OneFile) {
    $modeArgs += "--onefile"
    Write-Host "Build mode: Single file" -ForegroundColor Cyan
} else {
    Write-Host "Build mode: One directory" -ForegroundColor Cyan
}

if (-not $Console) {
    $modeArgs += "--noconsole"
    Write-Host "Console: Hidden (windowed)" -ForegroundColor Cyan
} else {
    Write-Host "Console: Visible" -ForegroundColor Cyan
}

# Include static assets
# NOTE: Windows uses ';' as separator in --add-data
$staticPath = "src\videopipeline\studio\static"
if (Test-Path $staticPath) {
    $modeArgs += @("--add-data", "$staticPath;videopipeline\studio\static")
    Write-Host "Including: Studio static files" -ForegroundColor Gray
}

# Include profiles folder if present
if (Test-Path ".\profiles") {
    $modeArgs += @("--add-data", "profiles;profiles")
    Write-Host "Including: Profiles folder" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Building executable..." -ForegroundColor Yellow

# Build launcher exe
# Using -m to run as module
pyinstaller @modeArgs src\videopipeline\launcher.py

Write-Host ""
Write-Host "=== Build Complete ===" -ForegroundColor Green
Write-Host ""

if ($OneFile) {
    $exePath = "dist\VideoPipeline.exe"
} else {
    $exePath = "dist\VideoPipeline\VideoPipeline.exe"
}

if (Test-Path $exePath) {
    $size = (Get-Item $exePath).Length / 1MB
    Write-Host "Executable: $exePath" -ForegroundColor Cyan
    Write-Host ("Size: {0:N1} MB" -f $size) -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To run: $exePath" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Optional: Copy ffmpeg.exe and ffprobe.exe next to the exe for portability." -ForegroundColor Gray
} else {
    Write-Host "Warning: Expected executable not found at $exePath" -ForegroundColor Red
    Write-Host "Check the PyInstaller output above for errors." -ForegroundColor Red
}
