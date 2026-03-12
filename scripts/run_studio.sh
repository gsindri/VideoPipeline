#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VIDEO="${1:-}"
PROFILE="${2:-profiles/gaming_assemblyai.yaml}"
WIN_VENV_PY="${REPO_ROOT}/.venv/Scripts/python.exe"
POSIX_VENV_PY="${REPO_ROOT}/.venv/bin/python"

if [[ -z "$VIDEO" ]]; then
  echo "Usage: scripts/run_studio.sh /path/to/video.mp4 [profile.yaml]"
  exit 1
fi

if [[ -f "$WIN_VENV_PY" && ! -x "$POSIX_VENV_PY" ]]; then
  echo "Detected a Windows repo venv at .venv/Scripts/python.exe."
  echo "Primary runtime on the main PC uses the Windows venv."
  echo "Launch from Windows with run_studio.bat or scripts/run_studio.ps1 instead of /usr/bin/python3 from WSL."
  exit 1
fi

if [[ ! -x "$POSIX_VENV_PY" ]]; then
  python3 -m venv "${REPO_ROOT}/.venv"
fi

"$POSIX_VENV_PY" -m pip install -e "$REPO_ROOT"
"$POSIX_VENV_PY" -m videopipeline.cli studio "$VIDEO" --profile "$PROFILE"
