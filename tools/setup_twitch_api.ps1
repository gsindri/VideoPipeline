[CmdletBinding()]
param(
  [string]$ClientId,
  [string]$ClientSecret,
  [switch]$SetUserEnv,
  [switch]$TestOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-VideoPipelineLocalEnvPath {
  $base = $env:LOCALAPPDATA -as [string]
  if ([string]::IsNullOrWhiteSpace($base)) {
    throw "LOCALAPPDATA is not set."
  }
  $dir = Join-Path $base "VideoPipeline"
  New-Item -ItemType Directory -Path $dir -Force | Out-Null
  return (Join-Path $dir "studio.env")
}

function Read-RequiredValue {
  param(
    [string]$Prompt,
    [switch]$Secret
  )

  if ($Secret) {
    $secure = Read-Host -Prompt $Prompt -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
      return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    } finally {
      if ($bstr -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
      }
    }
  }

  return (Read-Host -Prompt $Prompt)
}

function Test-TwitchClientCredentials {
  param(
    [Parameter(Mandatory = $true)][string]$ResolvedClientId,
    [Parameter(Mandatory = $true)][string]$ResolvedClientSecret
  )

  $tokenResp = Invoke-RestMethod `
    -Method Post `
    -Uri "https://id.twitch.tv/oauth2/token" `
    -ContentType "application/x-www-form-urlencoded" `
    -Body @{
      client_id     = $ResolvedClientId
      client_secret = $ResolvedClientSecret
      grant_type    = "client_credentials"
    }

  if ([string]::IsNullOrWhiteSpace($tokenResp.access_token)) {
    throw "Twitch token response did not include access_token."
  }

  return [ordered]@{
    access_token = [string]$tokenResp.access_token
    expires_in   = [int]$tokenResp.expires_in
    token_type   = [string]$tokenResp.token_type
  }
}

$ClientId = ($ClientId -as [string]).Trim()
$ClientSecret = ($ClientSecret -as [string]).Trim()

if ([string]::IsNullOrWhiteSpace($ClientId)) {
  $ClientId = (Read-RequiredValue -Prompt "Twitch Client ID").Trim()
}
if ([string]::IsNullOrWhiteSpace($ClientSecret)) {
  $ClientSecret = (Read-RequiredValue -Prompt "Twitch Client Secret" -Secret).Trim()
}

if ([string]::IsNullOrWhiteSpace($ClientId)) {
  throw "Client ID is required."
}
if ([string]::IsNullOrWhiteSpace($ClientSecret)) {
  throw "Client Secret is required."
}

$tokenInfo = Test-TwitchClientCredentials -ResolvedClientId $ClientId -ResolvedClientSecret $ClientSecret
$envPath = Get-VideoPipelineLocalEnvPath

if (-not $TestOnly) {
  @(
    "# Local secrets for VideoPipeline Studio. Do not commit."
    "TWITCH_CLIENT_ID=$ClientId"
    "TWITCH_CLIENT_SECRET=$ClientSecret"
  ) | Set-Content -Path $envPath -Encoding UTF8
}

if ($SetUserEnv) {
  [Environment]::SetEnvironmentVariable("TWITCH_CLIENT_ID", $ClientId, "User")
  [Environment]::SetEnvironmentVariable("TWITCH_CLIENT_SECRET", $ClientSecret, "User")
}

$result = [ordered]@{
  ok               = $true
  env_file         = [string]$envPath
  wrote_env_file   = (-not $TestOnly)
  set_user_env     = [bool]$SetUserEnv
  access_token_ttl = [int]$tokenInfo.expires_in
  next_steps       = @(
    "Restart VideoPipeline Studio or use run_studio.bat so the launcher reloads studio.env.",
    "Refresh Scout and verify twitch_api_configured becomes true in the backend environment.",
    "Do not persist TWITCH_APP_ACCESS_TOKEN; VideoPipeline will mint app tokens from the client credentials."
  )
}

$result | ConvertTo-Json -Depth 4
