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
  [switch]$Detach,
  [int]$WaitSeconds = 120,
  [string]$Protocol = "http2"
)

$ErrorActionPreference = "Stop"

function Write-Info([string]$msg) { Write-Host $msg }
function Write-Warn([string]$msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Err ([string]$msg) { Write-Host $msg -ForegroundColor Red }

function Get-EnvInt([string]$name) {
  $raw = [Environment]::GetEnvironmentVariable($name)
  if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
  $n = 0
  if (-not [int]::TryParse(([string]$raw).Trim(), [ref]$n)) { return $null }
  if ($n -lt 1 -or $n -gt 65535) { return $null }
  return $n
}

function Update-WorkerUpstreamKv([string]$publicBase) {
  $acct = $env:CF_ACCOUNT_ID
  $ns   = $env:CF_KV_NAMESPACE_ID
  $tok  = $env:CF_API_TOKEN
  if ([string]::IsNullOrWhiteSpace($acct) -or [string]::IsNullOrWhiteSpace($ns) -or [string]::IsNullOrWhiteSpace($tok)) {
    Write-Warn "[proxy] CF_ACCOUNT_ID / CF_KV_NAMESPACE_ID / CF_API_TOKEN not set; skipping KV update."
    return
  }

  $uri = "https://api.cloudflare.com/client/v4/accounts/$acct/storage/kv/namespaces/$ns/values/base"
  $headers = @{ Authorization = "Bearer $tok" }

  try {
    Invoke-RestMethod -Method Put -Uri $uri -Headers $headers -ContentType "text/plain" -Body $publicBase -TimeoutSec 12 | Out-Null
    Write-Host "[proxy] Updated Workers KV: base = $publicBase" -ForegroundColor Cyan
  } catch {
    Write-Err "[proxy] KV update failed: $($_.Exception.Message)"
  }
}

function Copy-ToClipboard([string]$Text) {
  # Use clip.exe with timeout to avoid occasional Set-Clipboard stalls.
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
    if (-not $p.WaitForExit(5000)) {
      try { $p.Kill() } catch {}
      throw "clip.exe timed out"
    }
    if ($p.ExitCode -ne 0) { throw "clip.exe failed (exit $($p.ExitCode))" }
  } finally {
    try { $p.Dispose() } catch {}
  }
}

try {
  $exitCode = 0
  $proc = $null
  $keepTunnelRunning = $false
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

    Write-Info "[studio] Starting Studio (minimized)..."

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

  function Get-ListeningPorts([int]$processId) {
    try {
      Get-NetTCPConnection -State Listen -OwningProcess $processId |
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

  function Find-StudioPort([int]$processId) {
    $deadline = (Get-Date).AddSeconds([Math]::Max(5, $WaitSeconds))
    $saw401 = $false

    while ((Get-Date) -lt $deadline) {
      $ports = @(Get-ListeningPorts $processId)

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

  $studioPid = [int]$studio.ProcessId
  Write-Info "[studio] Found Studio PID: $studioPid"

  $port = $null

  # If Studio is configured to use a fixed port, prefer that over PID-based scanning.
  $fixedPort = Get-EnvInt "VP_STUDIO_PORT"
  if ($fixedPort) {
    Write-Info "[studio] VP_STUDIO_PORT is set: $fixedPort"
    $deadline = (Get-Date).AddSeconds([Math]::Max(5, $WaitSeconds))
    $saw401 = $false
    while ((Get-Date) -lt $deadline) {
      $code = Probe-OpenApi $fixedPort
      if ($code -eq 200 -or $code -eq 429) { $port = $fixedPort; break }
      if ($code -eq 401 -or $code -eq 403) { $saw401 = $true }
      Start-Sleep -Milliseconds 300
    }
    if (-not $port -and $saw401) {
      throw "Studio responded but auth failed (401/403). Token mismatch? Delete %LOCALAPPDATA%\\VideoPipeline\\vp_api_token.txt to rotate."
    }
  }

  if (-not $port) {
    $port = Find-StudioPort $studioPid
  }
  if (-not $port) {
    throw "Could not identify Studio port within ${WaitSeconds}s. Is Studio healthy?"
  }

  Write-Info "[studio] Detected Studio port: $port"
  Write-Info "[studio] Local: http://127.0.0.1:$port/"

  Write-Info "[tunnel] Starting Cloudflare Quick Tunnel (protocol=$Protocol)..."

  $publicBase = $null
  $importUrl  = $null

  # Avoid piping cloudflared output through PowerShell (it can behave oddly on Windows).
  # Instead, write logs to a file and watch for the first trycloudflare URL.
  $logPath = Join-Path $env:TEMP ("vp_cloudflared_quick_tunnel_{0}.log" -f [int][DateTimeOffset]::UtcNow.ToUnixTimeSeconds())
  try { Remove-Item -LiteralPath $logPath -Force -ErrorAction SilentlyContinue | Out-Null } catch {}
  Write-Info "[tunnel] cloudflared log: $logPath"

  $cfArgs = @(
    "tunnel",
    "--url", "http://127.0.0.1:$port",
    "--protocol", $Protocol,
    "--loglevel", "info",
    "--logfile", $logPath
  )

  # Run cloudflared hidden to avoid terminal UI freezes from heavy stdout updates.
  $proc = Start-Process -FilePath "cloudflared" -ArgumentList $cfArgs -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru

  $deadline = (Get-Date).AddSeconds([Math]::Max(5, $WaitSeconds))
  while ((Get-Date) -lt $deadline -and (-not $publicBase)) {
    if ($proc.HasExited) { break }
    if (Test-Path -LiteralPath $logPath) {
      try {
        $tail = Get-Content -LiteralPath $logPath -Tail 200 -ErrorAction Stop
        $text = ($tail -join "`n")
        if ($text -match 'https://[a-z0-9-]+\.trycloudflare\.com') {
          $publicBase = $Matches[0]
          $importUrl = "$publicBase/api/actions/openapi.json?token=$token"

          Write-Host ""
          Write-Host "[tunnel] Public base URL: $publicBase" -ForegroundColor Cyan

          Update-WorkerUpstreamKv $publicBase

          if ($CopyImportUrl) {
            Copy-ToClipboard -Text $importUrl
            Write-Host "[actions] OpenAPI import URL copied to clipboard." -ForegroundColor Cyan
          }

          if ($PrintImportUrl) {
            Write-Host "[actions] OpenAPI import URL:" -ForegroundColor Cyan
            Write-Host $importUrl
          } else {
            $mask = ($token.Substring(0,4) + "..." + $token.Substring($token.Length-4))
            Write-Host "[actions] Import URL is: $publicBase/api/actions/openapi.json?token=$mask" -ForegroundColor DarkCyan
            Write-Host "[actions] (Use clipboard for the full URL.)" -ForegroundColor DarkCyan
          }

          Write-Host "[actions] In Actions auth: API Key -> Bearer -> paste the SAME token." -ForegroundColor Cyan

          if ($OpenImportUrl) {
            try { Start-Process $importUrl | Out-Null } catch {}
          }
        }
      } catch {}
    }
    Start-Sleep -Milliseconds 250
  }

  if (-not $publicBase) {
    $code = if ($proc.HasExited) { $proc.ExitCode } else { "running" }
    $tailText = ""
    if (Test-Path -LiteralPath $logPath) {
      try { $tailText = (Get-Content -LiteralPath $logPath -Tail 20 -ErrorAction SilentlyContinue | Out-String) } catch {}
    }
    if ([string]::IsNullOrWhiteSpace($tailText)) { $tailText = "(no log output)" }
    throw "cloudflared did not produce a trycloudflare URL (proc=$($proc.Id) exit=$code). Log tail from ${logPath}:`n$tailText"
  }

  if (-not $proc.HasExited) {
    if ($Detach) {
      $keepTunnelRunning = $true
      Write-Host ("[tunnel] Detached. cloudflared PID: {0}" -f $proc.Id) -ForegroundColor Cyan
      Write-Host ("[tunnel] Log file: {0}" -f $logPath) -ForegroundColor DarkCyan
      Write-Host "[tunnel] To stop later: taskkill /F /IM cloudflared.exe" -ForegroundColor DarkCyan
    } else {
      Write-Host ("[tunnel] Running (PID {0}). Press Enter to stop the tunnel." -f $proc.Id) -ForegroundColor Cyan
      try { [void](Read-Host) } catch {}
    }
  }
} catch {
  $exitCode = 1
  $msg = $_.Exception.Message
  if ([string]::IsNullOrWhiteSpace($msg)) { $msg = "Unknown error" }
  Write-Err $msg
} finally {
  # Ctrl+C stops this script, but doesn't necessarily stop the child process.
  # Ensure we clean up cloudflared if it's still running.
  if ($proc -and (-not $proc.HasExited) -and (-not $keepTunnelRunning)) {
    try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue | Out-Null } catch {}
  }
}

exit $exitCode
