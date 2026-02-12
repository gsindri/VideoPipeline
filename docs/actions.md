# ChatGPT Actions + API Token

When exposing Studio over an HTTPS tunnel (ngrok/Cloudflare Tunnel/etc.), you should protect all `/api/*` endpoints with a Bearer token.

## Token location (Windows)

The helper script stores the token at:

- `%LOCALAPPDATA%\\VideoPipeline\\vp_api_token.txt`

The file is **one line**: `64` hex characters (32 bytes).

## Generate / view / copy

- Print the token (stdout only):
  - `powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\\vp_api_token.ps1`
- Persist it (create the file if missing):
  - `powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\\vp_api_token.ps1 -Persist`
- Copy to clipboard:
  - `powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\\vp_api_token.ps1 -Persist -Copy`

## Use with Studio

Studio reads the token from the environment variable:

- `VP_API_TOKEN`
- Optional: set `VP_STUDIO_PORT` (for example `57820`) to keep Studio on a stable local URL.
  This helps browser-stored token memory survive restarts (`http://127.0.0.1:<port>` stays constant).

See `tools\\studio-launch.bat.example` for an end-to-end launcher that:
1) activates `.venv`
2) generates/persists a token (if missing)
3) sets `VP_API_TOKEN`
4) launches Studio

## Quick Tunnel (Cloudflare) + copy import URL

If you use Cloudflare Quick Tunnel, the public `trycloudflare.com` hostname changes every time you restart the tunnel.

Use:

- `tools\\studio-quick-tunnel.bat`

It will:
1) start Studio if needed
2) auto-detect the correct local port
3) start a Cloudflare Quick Tunnel
4) copy the Actions OpenAPI import URL (`/api/actions/openapi.json?token=...`) to your clipboard

Requirements:
- `cloudflared` in PATH (install with `winget install -e --id Cloudflare.cloudflared`)

Security note:
- The copied import URL includes your token in the query string. Treat it like a password and don’t share it.

## Rotate the token

Delete the token file and generate a new one:

- Delete: `%LOCALAPPDATA%\\VideoPipeline\\vp_api_token.txt`
- Recreate: run the script with `-Persist`

## Safety

- Never commit tokens to git.
- Treat the token like a password (anyone with it can call `/api/*` on your Studio server).

## External AI Candidate Defaults

`GET /api/actions/ai/candidates` now defaults to a hybrid candidate feed for ChatGPT Actions:

- Top `30` candidates by fused multi-signal score
- Plus `15` additional chat-spike candidates (deduped from the top set)

If you explicitly pass `top_n` and omit `chat_top_n`, the endpoint still applies the hybrid feed (`+15` chat-spike by default).
To force top-only behavior, pass `chat_top_n=0`.

Compatibility note:
- `chat_top` is accepted as an alias for `chat_top_n`.
