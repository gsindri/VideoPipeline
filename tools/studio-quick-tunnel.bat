@echo off
setlocal EnableExtensions

set "REPO=%~dp0.."
if not defined VP_STUDIO_PORT set "VP_STUDIO_PORT=57820"

where wt.exe >nul 2>&1
if %errorlevel%==0 (
  rem Prefer a new tab in an existing Windows Terminal window (opens a new window if none exist).
  wt -w 0 nt --title "VideoPipeline Tunnel" powershell.exe -NoLogo -NoExit -NoProfile -ExecutionPolicy Bypass ^
    -File "%REPO%\tools\studio_quick_tunnel.ps1" ^
    -StartStudio -CopyImportUrl
  if %errorlevel%==0 goto :eof
)

rem Fallback: open a standalone PowerShell window.
start "VideoPipeline Tunnel" powershell.exe -NoLogo -NoExit -NoProfile -ExecutionPolicy Bypass ^
  -File "%REPO%\tools\studio_quick_tunnel.ps1" ^
  -StartStudio -CopyImportUrl

endlocal
