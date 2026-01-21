# VideoPipeline Studio

## Quickstart

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -e .
```

## Launch Modes

### Home Mode (no video)

Launch Studio without a video to see the Home screen:

```bash
vp studio
```

The Home screen lets you:
- **Browse for video**: Opens a native Windows file dialog to select a video
- **Paste a path**: Enter a video path directly and click Open
- **Recent Projects**: Shows your recent projects sorted by last modified time

### Direct Mode (with video)

Launch Studio directly with a specific video:

```bash
vp studio /path/to/video.mp4 --profile profiles/gaming.yaml
```

This opens the video immediately in the Studio editor.

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
- Click "← Home" to return to Home screen and open another project

## Publishing from the UI

Studio includes a **Publish** tab for uploading exports to YouTube (and later TikTok).

### Setup: Add Accounts

Accounts are managed via CLI. Add a YouTube account:

```bash
vp accounts add youtube --client-secrets "path/to/client_secret.json" --label "My Channel"
```

This opens an OAuth flow in your browser. Once authorized, the account appears in Studio's Publish tab.

List accounts:

```bash
vp accounts list
```

### Using the Publish Tab

1. **Open a project** from the Home screen or via `vp studio /path/to/video.mp4`
2. **Export clips** from the Edit tab
3. **Switch to the Publish tab**
4. **Select accounts** — click to multi-select which accounts to publish to
5. **Select exports** — click to multi-select which clips to publish
6. **Configure options**:
   - **Privacy**: Private / Unlisted / Public
   - **Title override**: Replace the metadata title
   - **Description override**: Replace the metadata description
   - **Append hashtags**: Add hashtags to the description
   - **Stagger**: Delay between posts (useful for multi-account publishing)
7. **Click "Queue publish"** — creates one job per (export × account) combination

### Monitoring Jobs

The Publish Jobs panel shows:
- **Status badges**: queued → running → succeeded/failed
- **Progress bar** for running uploads
- **Error messages** for failed jobs
- **Remote URL** link when succeeded (click to open on YouTube)
- **Retry** button for failed/canceled jobs
- **Cancel** button for queued/running jobs

Jobs update in real-time via SSE (Server-Sent Events).

### Safety

The publisher only allows uploading files from the active project's `exports/` directory. This prevents accidentally publishing arbitrary files.

### Batch Publishing

Select multiple exports and multiple accounts, then click Queue publish. Studio creates one job for each combination. Use the Stagger option to space out uploads (e.g., 600 seconds = 10 minutes between posts).

## Outputs

Projects live in: `outputs/projects/<project_id>/`

- `project.json` — source video info, selections, export history
- `analysis/audio_features.npz` — cached audio timeline + scores
- `analysis/transcripts/*.json` — cached transcripts (if captions enabled)
- `analysis/subtitles/*.ass` — generated subtitle files (if captions enabled)
- `exports/*.mp4` — rendered exports

## Troubleshooting

- If the video won't seek in-browser, Range requests may be blocked (should be supported by Studio).
- If export fails, ensure `ffmpeg` and `ffprobe` are installed and on PATH.
- Captions require optional install:

  ```bash
  pip install -e '.[whisper]'
  ```
