@echo off
setlocal EnableExtensions

REM === Repo location ===
set "VP_REPO=C:\Users\gsind\Documents\GitHub\VideoPipeline"
set "TOKEN_HELPER=%VP_REPO%\tools\vp_api_token.ps1"
set "VENV_PY=%VP_REPO%\.venv\Scripts\python.exe"
set "VENV_PYW=%VP_REPO%\.venv\Scripts\pythonw.exe"

REM === Basic checks ===
if not exist "%VP_REPO%" (
  echo [studio] Repo not found: %VP_REPO%
  pause
  exit /b 1
)

if not exist "%TOKEN_HELPER%" (
  echo [studio] Token helper not found: %TOKEN_HELPER%
  pause
  exit /b 1
)

if not exist "%VENV_PY%" (
  echo [studio] Repo venv python not found: %VENV_PY%
  echo [studio] Create it with:
  echo   py -3 -m venv "%VP_REPO%\.venv"
  echo   "%VP_REPO%\.venv\Scripts\python.exe" -m pip install -e "%VP_REPO%"
  pause
  exit /b 1
)

set "VENV_LAUNCH_PY=%VENV_PY%"
if exist "%VENV_PYW%" set "VENV_LAUNCH_PY=%VENV_PYW%"

REM === Use repo-local Windows venv explicitly ===
cd /d "%VP_REPO%"

REM === Set VP_API_TOKEN (persisted in %LOCALAPPDATA%\VideoPipeline\vp_api_token.txt) ===
if not defined VP_API_TOKEN (
  for /f "usebackq delims=" %%T in (`
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%TOKEN_HELPER%" -Persist
  `) do set "VP_API_TOKEN=%%T"
)

if not defined VP_API_TOKEN (
  echo [studio] Failed to load/generate VP_API_TOKEN.
  echo [studio] Try running:
  echo powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%TOKEN_HELPER%" -Persist -ShowPath
  pause
  exit /b 1
)

REM === Pin Studio port for stable browser origin/token storage ===
if not defined VP_STUDIO_PORT set "VP_STUDIO_PORT=57820"
if not defined VP_STUDIO_PROFILE if exist "%VP_REPO%\profiles\gaming_assemblyai.yaml" set "VP_STUDIO_PROFILE=%VP_REPO%\profiles\gaming_assemblyai.yaml"
if not defined VP_RUNTIME_HOST (
  for /f "usebackq delims=" %%H in (`
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
      "$wslIp = Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias 'vEthernet (WSL (Hyper-V firewall))' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty IPAddress -First 1; if ($wslIp) { Write-Output $wslIp } else { Write-Output '127.0.0.1' }"
  `) do set "VP_RUNTIME_HOST=%%H"
)
if not defined VP_RUNTIME_HOST set "VP_RUNTIME_HOST=127.0.0.1"
if not defined VP_STUDIO_HOST set "VP_STUDIO_HOST=0.0.0.0"

REM === Launch Studio ===
start "" /min "%VENV_LAUNCH_PY%" -m videopipeline.launcher --host "%VP_STUDIO_HOST%" --runtime-host "%VP_RUNTIME_HOST%"
