param(
  [Parameter(Mandatory=$true)][string]$Video,
  [string]$Profile = "profiles/gaming.yaml"
)

if (-not (Test-Path ".venv")) {
  python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1
pip install -e .

vp studio $Video --profile $Profile
