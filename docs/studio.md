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
- **Download from URL**: Paste a YouTube/Twitch/etc link and download directly
- **Recent Projects**: Shows your recent projects sorted by last modified time
- **Recent Downloads**: Shows recently downloaded videos

### Direct Mode (with video)

Launch Studio directly with a specific video:

```bash
vp studio /path/to/video.mp4 --profile profiles/gaming.yaml
```

This opens the video immediately in the Studio editor.

Open: http://127.0.0.1:8765

## Downloading from URLs (Step 6.5)

Studio integrates yt-dlp for smart downloading from 1000+ supported sites with automatic site detection, adaptive speed, and guaranteed playback.

### How to Download

1. On the Home screen, paste a URL into the "Download from URL" field
2. The app automatically detects the site and shows a badge (e.g., "Detected: Twitch VOD (HLS) — Auto concurrency 16")
3. Click **🔍 Probe** for detailed info (title, duration) - optional
4. Choose quality cap and speed mode if needed
5. Click **Download** (or press Enter)
6. Watch progress in the Jobs panel (shows %, MB/s, ETA)
7. When complete, the video automatically opens as a project
8. Video plays in browser (preview is created if needed)

### Site Detection

| Site | Detection | Strategy |
|------|-----------|----------|
| **Twitch VOD** | `twitch.tv/videos/...` | HLS native + parallel fragments |
| **Twitch Clip** | `clips.twitch.tv/...` | HLS native |
| **YouTube** | `youtube.com`, `youtu.be` | Default DASH strategy |
| **Generic** | Any other site | Conservative defaults |

### Download Options

- **Create preview**: Generate a browser-friendly H.264/AAC proxy for smooth playback (recommended)
- **No playlist**: Download single video only, not entire playlists (default: on)
- **Speed mode**: Controls download concurrency for HLS streams (Twitch optimization)
- **Quality cap**: Limit download resolution (Source, 1080p, 720p, 480p)

### Speed Modes

| Mode | Concurrency (N) | Description |
|------|-----------------|-------------|
| **Auto** | Adaptive | Learns optimal speed per site, backs off on throttling |
| Conservative | 4 | Very safe, for slow/unstable connections |
| Balanced | 8 | Conservative, stable for most connections |
| Fast | 16 | Good for fast connections |
| Aggressive | 32 | Maximum speed, may trigger throttling |

**Auto mode** is recommended. It:
- Starts at N=16 for Twitch sites
- Backs off automatically if throttled (429/403 errors, fragment retries)
- Remembers what worked for each domain
- Learns over time so you get optimal speed without manual tuning

The tuning history is stored in: `%APPDATA%\VideoPipeline\ytdlp_tuning.json`

### Guaranteed Playback

After download, the app automatically:
1. **Probes** the video with ffprobe
2. **Remuxes** MPEG-TS to MP4 if needed (no quality loss)
3. **Creates preview** if codecs aren't browser-friendly (H.264/AAC proxy)

Studio uses:
- `source_path` for analysis and export (best quality)
- `preview_path` for the `<video>` element (guaranteed to play)

### Why Create Preview?

Some video formats (WebM, HEVC, VP9) don't play well in browsers. The preview option automatically creates an H.264/AAC MP4 for smooth playback in Studio while keeping the original full-quality file.

Downloads are saved to: `%LOCALAPPDATA%\VideoPipeline\Workspace\downloads\`

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
