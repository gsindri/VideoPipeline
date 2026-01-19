# VideoPipeline Studio

## Quickstart

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -e .
vp studio /path/to/video.mp4 --profile profiles/gaming.yaml
```

Open: http://127.0.0.1:8765

## Workflow

- Click Run / Re-run audio analysis
- Review Top candidates
- Load a candidate, fine-tune Start/End with:

  - Space = play/pause
  - I = set start
  - O = set end
  - [ / ] = nudge ±0.5s

- Click Save selection
- Click Export on a selection
- Watch progress in Jobs

## Outputs

Projects live in: `outputs/projects/<project_id>/`

- `project.json` — source video info, selections, export history
- `analysis/audio_features.npz` — cached audio timeline + scores
- `analysis/transcripts/*.json` — cached transcripts (if captions enabled)
- `analysis/subtitles/*.ass` — generated subtitle files (if captions enabled)
- `exports/*.mp4` — rendered exports

## Troubleshooting

- If the video won’t seek in-browser, Range requests may be blocked (should be supported by Studio).
- If export fails, ensure `ffmpeg` and `ffprobe` are installed and on PATH.
- Captions require optional install:

  ```bash
  pip install -e '.[whisper]'
  ```
