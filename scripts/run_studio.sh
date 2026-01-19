#!/usr/bin/env bash
set -euo pipefail

VIDEO="${1:-}"
PROFILE="${2:-profiles/gaming.yaml}"

if [[ -z "$VIDEO" ]]; then
  echo "Usage: scripts/run_studio.sh /path/to/video.mp4 [profile.yaml]"
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  python -m venv .venv
fi

source .venv/bin/activate
pip install -e .

vp studio "$VIDEO" --profile "$PROFILE"
