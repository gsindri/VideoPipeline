@echo off
setlocal EnableExtensions

set "REPO=%~dp0.."
if not defined VP_STUDIO_PORT set "VP_STUDIO_PORT=57820"
if not defined VP_TUNNEL_WAIT_SECONDS set "VP_TUNNEL_WAIT_SECONDS=300"

echo [studio-quick-tunnel] Starting tunnel launcher in a separate window...
start "VideoPipeline Tunnel" powershell.exe -NoLogo -NoExit -NoProfile -ExecutionPolicy Bypass ^
  -File "%REPO%\tools\studio_quick_tunnel.ps1" ^
  -StartStudio -CopyImportUrl -WaitSeconds %VP_TUNNEL_WAIT_SECONDS%
if errorlevel 1 (
  echo [studio-quick-tunnel] Failed to start tunnel window.
  pause
  exit /b 1
)

echo [studio-quick-tunnel] Tunnel window launched. Press any key to close this launcher.
pause >nul
endlocal
