# Quick Tunnel launcher that auto-detects the Studio port and copies the Actions OpenAPI import URL.
# PowerShell 5.1 compatible.
#
# Requires:
# - `cloudflared` in PATH (winget: winget install -e --id Cloudflare.cloudflared)
# - A VP_API_TOKEN (generated/persisted via tools/vp_api_token.ps1)
#
# Notes:
# - This script prints a masked import URL. Use clipboard for the full URL.
# - The trycloudflare.com hostname changes each time you start the tunnel.

[CmdletBinding()]
param(
  [switch]$StartStudio,
  [switch]$CopyToken,
  [switch]$CopyImportUrl,
  [switch]$OpenImportUrl,
  [switch]$PrintImportUrl,
  [int]$WaitSeconds = 30,
  [string]$Protocol = "http2"
)

$ErrorActionPreference = "Stop"

function Write-Info([string]$msg) { Write-Host $msg }
function Write-Warn([string]$msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Err ([string]$msg) { Write-Host $msg -ForegroundColor Red }

function Copy-ToClipboard([string]$Text) {
  try {
    Set-Clipboard -Value $Text
    return
  } catch {}
  # Fallback: spawn clip.exe and write to stdin (no newline).
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = "clip.exe"
  $psi.UseShellExecute = $false
  $psi.RedirectStandardInput = $true
  $psi.CreateNoWindow = $true
  $p = New-Object System.Diagnostics.Process
  $p.StartInfo = $psi
  if (-not $p.Start()) { throw "Failed to start clip.exe" }
  try {
    $p.StandardInput.Write($Text)
    $p.StandardInput.Close()
    $p.WaitForExit()
    if ($p.ExitCode -ne 0) { throw "clip.exe failed (exit $($p.ExitCode))" }
  } finally {
    try { $p.Dispose() } catch {}
  }
}

try {
  # Repo root from this file's location: <repo>\tools\studio_quick_tunnel.ps1
  $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

  # Token helper
  $tokenHelper = Join-Path $PSScriptRoot "vp_api_token.ps1"
  if (!(Test-Path -LiteralPath $tokenHelper)) {
    throw "Token helper not found: $tokenHelper"
  }

  # Load/persist token (single-line output)
  $token = (& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $tokenHelper -Persist | Select-Object -First 1)
  $token = ([string]$token).Trim()
  if ([string]::IsNullOrWhiteSpace($token)) {
    throw "vp_api_token.ps1 returned an empty token."
  }
  if ($token -notmatch "^[0-9a-fA-F]{64}$") {
    throw "Invalid token format returned by vp_api_token.ps1 (expected 64 hex chars)."
  }

  if ($CopyToken) {
    Copy-ToClipboard -Text $token
    Write-Info "[token] VP_API_TOKEN copied to clipboard."
  }

  function Get-StudioProc {
    # Choose the newest python/pythonw process whose command line contains videopipeline.launcher
    $cands = Get-CimInstance Win32_Process |
      Where-Object {
        $_.CommandLine -and
        ($_.Name -match '^python(w)?\.exe$') -and
        ($_.CommandLine -match 'videopipeline\.launcher')
      }

    if (-not $cands) { return $null }

    $best = $cands | Sort-Object {
      try { [Management.ManagementDateTimeConverter]::ToDateTime($_.CreationDate) } catch { Get-Date 0 }
    } -Descending | Select-Object -First 1

    return $best
  }

  function Start-StudioIfNeeded {
    $existing = Get-StudioProc
    if ($existing) { return $existing }

    if (-not $StartStudio) { return $null }

    $py = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (!(Test-Path -LiteralPath $py)) {
      throw "Venv python not found: $py"
    }

    Write-Info "[studio] Starting Studio (minimized)…"

    $old = $env:VP_API_TOKEN
    $env:VP_API_TOKEN = $token
    try {
      Start-Process -FilePath $py -ArgumentList @("-m", "videopipeline.launcher") -WorkingDirectory $repoRoot -WindowStyle Minimized | Out-Null
    } finally {
      $env:VP_API_TOKEN = $old
    }

    $deadline = (Get-Date).AddSeconds([Math]::Max(5, $WaitSeconds))
    while ((Get-Date) -lt $deadline) {
      $p = Get-StudioProc
      if ($p) { return $p }
      Start-Sleep -Milliseconds 300
    }

    return $null
  }

  function Get-ListeningPorts([int]$pid) {
    try {
      Get-NetTCPConnection -State Listen -OwningProcess $pid |
        Select-Object -ExpandProperty LocalPort -Unique |
        Sort-Object
    } catch {
      @()
    }
  }

  function Probe-OpenApi([int]$port) {
    $url = "http://127.0.0.1:$port/api/actions/openapi.json?token=$token"
    try {
      $resp = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri $url
      return $resp.StatusCode
    } catch {
      # Try to surface HTTP status codes if present
      try {
        $we = $_.Exception
        if ($we.Response -and $we.Response.StatusCode) {
          return [int]$we.Response.StatusCode
        }
      } catch {}
      return $null
    }
  }

  function Find-StudioPort([int]$pid) {
    $deadline = (Get-Date).AddSeconds([Math]::Max(5, $WaitSeconds))
    $saw401 = $false

    while ((Get-Date) -lt $deadline) {
      $ports = @(Get-ListeningPorts $pid)

      foreach ($p in $ports) {
        $code = Probe-OpenApi $p
        if ($code -eq 200 -or $code -eq 429) { return $p } # 429 still proves the endpoint exists.
        if ($code -eq 401 -or $code -eq 403) { $saw401 = $true }
      }

      Start-Sleep -Milliseconds 350
    }

    if ($saw401) {
      throw "Studio responded but auth failed (401/403). Token mismatch? Delete %LOCALAPPDATA%\VideoPipeline\vp_api_token.txt to rotate."
    }

    return $null
  }

  # Ensure cloudflared exists
  if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    throw "cloudflared not found in PATH. Install it (winget install -e --id Cloudflare.cloudflared) and reopen PowerShell."
  }

  # Find or start Studio
  $studio = Start-StudioIfNeeded
  if (-not $studio) {
    throw "Studio process not found. Start Studio first, or rerun with -StartStudio."
  }

  $pid = [int]$studio.ProcessId
  Write-Info "[studio] Found Studio PID: $pid"

  $port = Find-StudioPort $pid
  if (-not $port) {
    throw "Could not identify Studio port within ${WaitSeconds}s. Is Studio healthy?"
  }

  Write-Info "[studio] Detected Studio port: $port"
  Write-Info "[studio] Local: http://127.0.0.1:$port/"

  Write-Info "[tunnel] Starting Cloudflare Quick Tunnel (protocol=$Protocol)…"
  Write-Info "[tunnel] Press Ctrl+C to stop the tunnel."

  $publicBase = $null
  $importUrl  = $null

  # Stream cloudflared output live, but watch for the first trycloudflare URL
  & cloudflared tunnel --url "http://127.0.0.1:$port" --protocol $Protocol 2>&1 | ForEach-Object {
    $line = $_.ToString()

    if (-not $publicBase -and $line -match 'https://[a-z0-9-]+\.trycloudflare\.com') {
      $publicBase = $Matches[0]
      $importUrl = "$publicBase/api/actions/openapi.json?token=$token"

      Write-Host ""
      Write-Host "[tunnel] Public base URL: $publicBase" -ForegroundColor Cyan

      if ($CopyImportUrl) {
        Copy-ToClipboard -Text $importUrl
        Write-Host "[actions] OpenAPI import URL copied to clipboard." -ForegroundColor Cyan
      }

      if ($PrintImportUrl) {
        Write-Host "[actions] OpenAPI import URL:" -ForegroundColor Cyan
        Write-Host $importUrl
      } else {
        $mask = ($token.Substring(0,4) + "…" + $token.Substring($token.Length-4))
        Write-Host "[actions] Import URL is: $publicBase/api/actions/openapi.json?token=$mask" -ForegroundColor DarkCyan
        Write-Host "[actions] (Use clipboard for the full URL.)" -ForegroundColor DarkCyan
      }

      Write-Host "[actions] In Actions auth: API Key → Bearer → paste the SAME token." -ForegroundColor Cyan

      if ($OpenImportUrl) {
        try { Start-Process $importUrl | Out-Null } catch {}
      }
    }

    $line
  }
} catch {
  $msg = $_.Exception.Message
  if ([string]::IsNullOrWhiteSpace($msg)) { $msg = "Unknown error" }
  Write-Err $msg
  exit 1
}

