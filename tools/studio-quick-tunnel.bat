@echo off
setlocal EnableExtensions

set "REPO=%~dp0.."

powershell.exe -NoProfile -ExecutionPolicy Bypass ^
  -File "%REPO%\tools\studio_quick_tunnel.ps1" ^
  -StartStudio -CopyImportUrl

endlocal

