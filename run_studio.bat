@echo off
setlocal EnableExtensions

REM === Repo location ===
set "VP_REPO=C:\Users\gsind\Documents\GitHub\VideoPipeline"
set "TOKEN_HELPER=%VP_REPO%\tools\vp_api_token.ps1"

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

REM === Activate venv ===
cd /d "%VP_REPO%"
call .venv\Scripts\activate

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

REM === Launch Studio ===
start "" /min python -m videopipeline.launcher
