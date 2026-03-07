@echo off
setlocal EnableExtensions

set "REPO=%~dp0.."
if not defined VP_STUDIO_PORT set "VP_STUDIO_PORT=57820"
if not defined VP_TUNNEL_WAIT_SECONDS set "VP_TUNNEL_WAIT_SECONDS=300"
set "PROFILE_ARGS="
set "TARGET=%~f0"

if /I "%~1"=="--launched" goto :run_inline

if exist "%WT_EXE%" goto :open_tab
where wt >nul 2>&1
if not errorlevel 1 set "WT_EXE=wt"
if not defined WT_EXE goto :run_inline

:open_tab
"%WT_EXE%" -w 0 new-tab --title "VideoPipeline Actions" cmd /k call "%TARGET%" --launched >nul 2>&1
if not errorlevel 1 exit /b 0

:run_inline
echo [studio-quick-tunnel] Starting tunnel launcher in this terminal tab...
if defined VP_STUDIO_PROFILE (
  echo [studio-quick-tunnel] Studio profile: %VP_STUDIO_PROFILE%
  set "PROFILE_ARGS=-StudioProfile \"%VP_STUDIO_PROFILE%\""
) else (
  echo [studio-quick-tunnel] Studio profile: launcher default
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  -File "%REPO%\tools\studio_quick_tunnel.ps1" ^
  -StartStudio %PROFILE_ARGS% -CopyImportUrl -WaitSeconds %VP_TUNNEL_WAIT_SECONDS%
if errorlevel 1 (
  echo [studio-quick-tunnel] Tunnel launcher exited with an error.
  exit /b 1
)

endlocal
exit /b 0
