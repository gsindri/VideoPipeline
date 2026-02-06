<# 
VideoPipeline API token helper.

Default behavior (no flags):
  - Prints the token to stdout (single line) with no extra noise.

Token storage:
  - %LOCALAPPDATA%\VideoPipeline\vp_api_token.txt

Flags:
  -Persist   Create the token file if missing (and persist the generated token).
  -Copy      Copy token to clipboard.
  -ShowPath  Print token file path to stderr (labeled).
#>

[CmdletBinding()]
param(
  [switch]$Persist,
  [switch]$Copy,
  [switch]$ShowPath
)

$ErrorActionPreference = "Stop"

function New-RandomHexToken([int]$ByteCount) {
  $bytes = New-Object byte[] $ByteCount
  $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
  try {
    $rng.GetBytes($bytes)
  } finally {
    if ($rng -ne $null) { $rng.Dispose() }
  }
  return ([System.BitConverter]::ToString($bytes)).Replace("-", "").ToLowerInvariant()
}

function Read-TokenFile([string]$Path) {
  # Single-line token file. If multiple lines exist, use the first.
  $raw = [System.IO.File]::ReadAllText($Path)
  if ($raw -eq $null) { return "" }
  $line = ($raw -split "(\r\n|\n|\r)")[0]
  return ($line -as [string]).Trim()
}

function Write-TokenFile([string]$Path, [string]$Token) {
  $dir = [System.IO.Path]::GetDirectoryName($Path)
  if (-not [string]::IsNullOrWhiteSpace($dir)) {
    [System.IO.Directory]::CreateDirectory($dir) | Out-Null
  }
  [System.IO.File]::WriteAllText($Path, $Token, [System.Text.Encoding]::ASCII)
}

function Copy-ToClipboard([string]$Text) {
  $cmd = Get-Command -Name Set-Clipboard -ErrorAction SilentlyContinue
  if ($cmd) {
    Set-Clipboard -Value $Text
    return
  }
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

$exitCode = 0

try {
  $base = ($env:LOCALAPPDATA -as [string])
  if ([string]::IsNullOrWhiteSpace($base)) { throw "LOCALAPPDATA is not set." }
  $tokenPath = Join-Path $base "VideoPipeline\vp_api_token.txt"

  if ($ShowPath) {
    [Console]::Error.WriteLine("VP_API_TOKEN_PATH=$tokenPath")
  }

  $token = ""
  if (Test-Path -LiteralPath $tokenPath) {
    $token = Read-TokenFile -Path $tokenPath
    if ($token -notmatch "^[0-9a-fA-F]{64}$") {
      throw "Token file exists but contents are invalid. Delete the file to rotate: $tokenPath"
    }
  } else {
    $token = New-RandomHexToken -ByteCount 32
    if ($Persist) {
      Write-TokenFile -Path $tokenPath -Token $token
    }
  }

  if ($Copy) {
    Copy-ToClipboard -Text $token
  }
} catch {
  $exitCode = 1
  $msg = $_.Exception.Message
  if ([string]::IsNullOrWhiteSpace($msg)) { $msg = "Unknown error" }
  [Console]::Error.WriteLine("vp_api_token.ps1 error: $msg")
}

# Token to stdout (single line, no noise).
if (-not [string]::IsNullOrWhiteSpace($token)) {
  Write-Output $token
}

exit $exitCode
