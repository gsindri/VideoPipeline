param(
  [string]$PythonExe = "",
  [string]$TorchCodecVersion = "",
  [string]$TorchCodecRef = "",
  [string]$TorchCodecRepo = "https://github.com/pytorch/torchcodec.git",
  [string]$CacheRoot = "",
  [string]$VsDevCmd = "",
  [string]$PytorchRocmArch = "",
  [switch]$CleanBuild,
  [switch]$NoBackup
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

function Exec([string]$Exe, [object[]]$Args) {
  & $Exe @Args
  if ($LASTEXITCODE -ne 0) {
    $argStr = ($Args | ForEach-Object { [string]$_ }) -join " "
    throw "Command failed (exit=$LASTEXITCODE): $Exe $argStr"
  }
}

function Resolve-VsDevCmd([string]$Hint) {
  if ($Hint) {
    Assert-Exists $Hint "VsDevCmd.bat not found at: $Hint"
    return $Hint
  }

  $candidates = @()

  $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
  if (Test-Path -LiteralPath $vswhere) {
    try {
      $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
      if ($installPath) {
        $cand = Join-Path $installPath "Common7\Tools\VsDevCmd.bat"
        if (Test-Path -LiteralPath $cand) {
          $candidates += $cand
        }
      }
    } catch {
      # If vswhere exists but fails, keep probing common locations.
    }
  }

  $pf = ${env:ProgramFiles}
  $candidates += @(
    (Join-Path $pf "Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"),
    (Join-Path $pf "Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"),
    (Join-Path $pf "Microsoft Visual Studio\17\Community\Common7\Tools\VsDevCmd.bat"),
    (Join-Path $pf "Microsoft Visual Studio\17\BuildTools\Common7\Tools\VsDevCmd.bat")
  )

  foreach ($cand in $candidates) {
    if (Test-Path -LiteralPath $cand) {
      return $cand
    }
  }

  throw "Could not find VsDevCmd.bat. Install Visual Studio (Desktop development with C++) and re-run, or pass -VsDevCmd <path>."
}

function Get-PipShowVersion([string]$Py, [string]$Pkg) {
  $out = & $Py -m pip show $Pkg 2>$null
  if ($LASTEXITCODE -ne 0) { return "" }
  foreach ($line in ($out -split "`r?`n")) {
    if ($line -match "^Version:\\s*(.+)\\s*$") { return $Matches[1].Trim() }
  }
  return ""
}

function Get-PythonPurelib([string]$Py) {
  return (& $Py -c "import sysconfig, pathlib; print(pathlib.Path(sysconfig.get_path('purelib')).resolve())").Trim()
}

function Get-TorchCmakeDir([string]$Py) {
  return (& $Py -c "import torch, pathlib; print((pathlib.Path(torch.__file__).resolve().parent / 'share' / 'cmake' / 'Torch').resolve())").Trim()
}

function Get-Pybind11Prefix([string]$Py) {
  return (& $Py -c "import pybind11, pathlib; print(pathlib.Path(pybind11.__file__).resolve().parent)").Trim()
}

function Get-RocmDevelRoot([string]$Py) {
  return (& $Py -c "from rocm_sdk import _devel; print(_devel.get_devel_root())").Trim()
}

function Detect-GfxArch([string]$RocmRoot, [string]$Hint) {
  if ($Hint) { return $Hint }
  $hipInfo = Join-Path $RocmRoot "bin\hipInfo.exe"
  Assert-Exists $hipInfo "Could not find hipInfo.exe at: $hipInfo"

  $text = (& $hipInfo) | Out-String
  if ($text -match "gcnArchName:\\s*(gfx\\d+)") {
    return $Matches[1]
  }
  throw "Failed to auto-detect GPU arch from hipInfo.exe. Re-run with -PytorchRocmArch gfxXXXX."
}

function Copy-IfExists([string]$SrcGlob, [string]$DstDir) {
  $items = @(Get-ChildItem -Path $SrcGlob -ErrorAction SilentlyContinue)
  foreach ($i in $items) {
    Copy-Item -LiteralPath $i.FullName -Destination $DstDir -Force
  }
  return $items.Count
}

Write-Section "Config"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) {
  throw "git is not available. Install Git for Windows and re-run."
}

$resolvedPython = $PythonExe
if (-not $resolvedPython) {
  $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
  Assert-Exists $venvPython "PythonExe not provided and repo venv not found at: $venvPython"
  $resolvedPython = $venvPython
}
Assert-Exists $resolvedPython "PythonExe not found: $resolvedPython"

$cacheRoot = $CacheRoot
if (-not $cacheRoot) {
  $cacheRoot = Join-Path $repoRoot "cache\torchcodec"
}

$vsDevCmdPath = Resolve-VsDevCmd $VsDevCmd

Write-Host "RepoRoot: $repoRoot"
Write-Host "PythonExe: $resolvedPython"
Write-Host "CacheRoot: $cacheRoot"
Write-Host "VsDevCmd:  $vsDevCmdPath"
Write-Host "CleanBuild: $CleanBuild"
Write-Host "NoBackup: $NoBackup"

Write-Section "Versions"
& $resolvedPython -c "import sys; print('python =', sys.version.replace('\n',' '))"
& $resolvedPython -c "import torch; print('torch =', torch.__version__)"

$detectedVersion = $TorchCodecVersion
if (-not $detectedVersion) {
  $detectedVersion = Get-PipShowVersion $resolvedPython "torchcodec"
}
if (-not $detectedVersion) {
  $detectedVersion = "0.9.1"
}

$ref = $TorchCodecRef
if (-not $ref) { $ref = "v$detectedVersion" }

Write-Host "torchcodec_version_target: $detectedVersion"
Write-Host "torchcodec_ref: $ref"
Write-Host "torchcodec_repo: $TorchCodecRepo"

Write-Section "Ensure torchcodec wheel installed (no-deps)"
Exec $resolvedPython @("-m", "pip", "install", "--no-deps", "--upgrade", "torchcodec==$detectedVersion")

Write-Section "Resolve build dependencies"
Exec $resolvedPython @("-m", "pip", "install", "--upgrade", "--no-deps", "cmake", "ninja", "pybind11")

$purelib = Get-PythonPurelib $resolvedPython
$torchcodecPkgDir = Join-Path $purelib "torchcodec"
Assert-Exists $torchcodecPkgDir "torchcodec package dir not found: $torchcodecPkgDir"

$torchDir = Get-TorchCmakeDir $resolvedPython
$pybindPrefix = Get-Pybind11Prefix $resolvedPython
Assert-Exists $torchDir "Torch CMake dir not found: $torchDir"
Assert-Exists $pybindPrefix "pybind11 prefix not found: $pybindPrefix"

$rocmRoot = Get-RocmDevelRoot $resolvedPython
Assert-Exists $rocmRoot "ROCm devel root not found: $rocmRoot"

$gfx = Detect-GfxArch $rocmRoot $PytorchRocmArch
Write-Host "PYTORCH_ROCM_ARCH: $gfx"

$clangCl = Join-Path $rocmRoot "lib\llvm\bin\clang-cl.exe"
Assert-Exists $clangCl "clang-cl.exe not found at: $clangCl"

Write-Section "Prepare cache directories"
New-Item -ItemType Directory -Force -Path $cacheRoot | Out-Null
$srcDir = Join-Path $cacheRoot "src"
$buildDir = Join-Path $cacheRoot "build_ninja"

Write-Host "SourceDir: $srcDir"
Write-Host "BuildDir:  $buildDir"

Write-Section "Fetch torchcodec source"
if (-not (Test-Path -LiteralPath (Join-Path $srcDir ".git"))) {
  if (Test-Path -LiteralPath $srcDir) {
    Remove-Item -Recurse -Force $srcDir
  }
  Exec "git" @("clone", $TorchCodecRepo, $srcDir)
} else {
  Exec "git" @("-C", $srcDir, "fetch", "--tags", "--prune")
}

Exec "git" @("-C", $srcDir, "checkout", "--force", $ref)

Write-Section "Configure + build (Ninja + clang-cl)"
if ($CleanBuild -and (Test-Path -LiteralPath $buildDir)) {
  Remove-Item -Recurse -Force $buildDir
}
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

$installPrefix = Join-Path $buildDir "_install"

$cmakeCmdLine = @(
  "cmake",
  "-S", "`"$srcDir`"",
  "-B", "`"$buildDir`"",
  "-G", "Ninja",
  "-DCMAKE_BUILD_TYPE=Release",
  "-DCMAKE_INSTALL_PREFIX=`"$installPrefix`"",
  "-DTorch_DIR=`"$torchDir`"",
  "-DCMAKE_PREFIX_PATH=`"$pybindPrefix`"",
  "-DPython_EXECUTABLE=`"$resolvedPython`"",
  "-DCMAKE_C_COMPILER=`"$clangCl`"",
  "-DCMAKE_CXX_COMPILER=`"$clangCl`"",
  "-DENABLE_CUDA=",
  "-DTORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR=OFF"
) -join " "

$cmd = @"
call "$vsDevCmdPath" -arch=amd64 -host_arch=amd64
set "BUILD_AGAINST_ALL_FFMPEG_FROM_S3=1"
set "ROCM_PATH=$rocmRoot"
set "MIOPEN_PATH=$rocmRoot"
set "PYTORCH_ROCM_ARCH=$gfx"
set "PATH=$rocmRoot\lib\llvm\bin;$rocmRoot\bin;%PATH%"
$cmakeCmdLine
if errorlevel 1 exit /b %errorlevel%
cmake --build "$buildDir" -- -v
exit /b %errorlevel%
"@

Exec "cmd.exe" @("/c", $cmd)

Write-Section "Apply rebuilt binaries"
$coreOut = Join-Path $buildDir "src\torchcodec\_core"
Assert-Exists $coreOut "Build output dir not found: $coreOut"

if (-not $NoBackup) {
  $backupRoot = Join-Path $cacheRoot "backup"
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $backupDir = Join-Path $backupRoot $stamp
  New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
  Copy-Item -Path (Join-Path $torchcodecPkgDir "libtorchcodec_*.*") -Destination $backupDir -Force -ErrorAction SilentlyContinue
  Write-Host "Backed up existing binaries to: $backupDir"
} else {
  Write-Host "Skipping backup (-NoBackup set)."
}

$dllCount = Copy-IfExists (Join-Path $coreOut "libtorchcodec_*.dll") $torchcodecPkgDir
$pydCount = Copy-IfExists (Join-Path $coreOut "libtorchcodec_*.pyd") $torchcodecPkgDir
Write-Host "Copied DLLs: $dllCount"
Write-Host "Copied PYDs: $pydCount"

if (($dllCount + $pydCount) -le 0) {
  throw "No torchcodec binaries copied. Build may have failed to produce outputs."
}

Write-Section "Run diagnostics"
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
  Write-Host "WARNING: FFMPEG_SHARED_BIN is not set. Run scripts/setup_ffmpeg_shared.ps1 first for reliable validation."
}

& $resolvedPython -c "import torch; print('torch =', torch.__version__)"
Exec $resolvedPython @("-m", "pip", "show", "torchcodec")

& $resolvedPython "scripts\diagnose_torchcodec.py" --require-core 7 --require-core 8
$diagExit = $LASTEXITCODE
if ($diagExit -ne 0) {
  exit $diagExit
}

Write-Section "Done"
Write-Host "TorchCodec rebuild/apply finished."
