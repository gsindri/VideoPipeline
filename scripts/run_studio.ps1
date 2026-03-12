param(
  [Parameter(Mandatory=$true)][string]$Video,
  [string]$Profile = "profiles/gaming_assemblyai.yaml"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvDir = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
  $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
  if ($pyLauncher) {
    & $pyLauncher.Source -3 -m venv $venvDir
  } else {
    python -m venv $venvDir
  }
}

Push-Location $repoRoot
try {
  & $venvPython -m pip install -e $repoRoot
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }

  & $venvPython -m videopipeline.cli studio $Video --profile $Profile
  exit $LASTEXITCODE
} finally {
  Pop-Location
}
