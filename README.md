# VideoPipeline

Personal toolchain to:

1. Ingest long gaming/stream videos (local file or URL)
2. Auto-suggest highlight moments ("top clips")
3. Export social-ready shorts (vertical, captioned, templated)
4. Publish to multiple destination accounts (YouTube, TikTok, etc.)

This repository starts with a clean foundation (README, LICENSE, .gitignore). The implementation will evolve, but the core idea stays the same:

**signals → candidate clips → ranking → exports → publishing**

## Why this exists

- Speed: turn long VODs into multiple shorts quickly.
- Consistency: repeatable exports with stable templates.
- Extensibility: start with gaming/streaming, expand via "profiles".

## Planned scope

### Ingest
- Local video files (primary)
- Optional URL ingest (e.g., download VODs with yt-dlp)

### Signals (gaming/streaming profile)
- Audio excitement / energy peaks
- Scene cuts (shot boundary detection)
- Optional: chat spike timeline (when available)
- Optional: creator markers (best signal if you stream yourself)

### Clip generation + ranking
- Generate many candidates around peak moments
- Rank and surface top N with explanations
- Human review in seconds (approve + tweak start/end)

### Packaging
- Vertical 9:16 templates (gameplay-only, gameplay+facecam)
- Captions (burned-in + optional sidecar)
- Thumbnails / title suggestions

### Publishing
- Pluggable connectors (YouTube first)
- Multiple destination accounts per platform
- Dry-run/export-only mode when APIs are unavailable

## Publisher layer (Step 6)

The publisher layer turns exports (MP4 + metadata JSON from Step 5) into queued uploads with retry support.

### State + security

- Windows state lives in `%APPDATA%\\VideoPipeline\\`
- macOS/Linux state lives in `~/.videopipeline/`
- `accounts.json` stores account metadata (no secrets)
- `publisher.sqlite` stores the publish queue and deduplication records
- OAuth tokens are stored in the OS credential store via `keyring`

### CLI: accounts

```bash
vp accounts list

vp accounts add youtube --client-secrets path/to/client_secrets.json --label "Main YouTube"

vp accounts add tiktok \
  --client-key <client_key> \
  --client-secret <client_secret> \
  --redirect-port 3455 \
  --scopes "user.info.basic,video.upload,video.publish" \
  --label "Main TikTok"

vp accounts remove <account_id>
```

### CLI: publish

```bash
vp publish --account <account_id> --export path/to/clip.mp4 --meta path/to/clip.metadata.json

# Queue all exports from a project
vp publish-project --project <project_id> --account <account_id>

# Review the queue
vp jobs list
vp jobs retry <job_id>
```

### Studio API (minimum)

- `GET /api/publisher/accounts`
- `POST /api/publisher/publish`
- `GET /api/publisher/jobs`
- `GET /api/publisher/jobs/stream` (SSE)

### Defaults + platform notes

- YouTube uploads default to **private** to avoid unverified-project visibility issues.
- TikTok supports both inbox upload (`video.upload`) and direct post (`video.publish`) flows.

## Repo conventions (proposed)



/data/ # ignored - raw inputs (VODs, downloads)
/outputs/ # ignored - rendered clips
/cache/ # ignored - intermediate artifacts (features, transcripts)
/src/ # code (when added)
/scripts/ # helper scripts (download, local runs)
/docs/ # design notes, architecture


> Tip: use Git LFS if you ever want to version sample videos.

## Security notes

- Never commit secrets. This repo ignores `.env` by default.
- OAuth tokens/refresh tokens should be stored encrypted (future work).

## Legal note

Only clip/upload content you have rights to use. Platform rules and copyright enforcement vary.

## Status

Work in progress. First commits focus on repo hygiene and scaffolding.
