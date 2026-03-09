# Machine Notes

## Purpose

Track differences between the main PC and the laptop so machine-specific fixes are not accidentally removed or generalized into the wrong default.

## Priority

- The main PC is the primary supported environment.
- Laptop support is best-effort only.
- A laptop-only issue is not a blocker unless it also affects the main PC or you explicitly decide to prioritize laptop support again.

## Current Known Differences

### Laptop

- A Windows `cmd` / console freezing or flashing issue was seen during some local runs.
- At least some launcher or subprocess-related settings were adjusted to work around that behavior.

### Main PC

- The same `cmd` / console issue is not currently known to reproduce on the main PC.

## Sensitive Areas

Be cautious when changing behavior in these areas because they may be compensating for machine-specific runtime differences:

- `src/videopipeline/launcher.py`
- `VideoPipeline.spec`
- subprocess creation flags and hidden-console behavior
- local GPU/runtime startup paths
- FFmpeg / ffprobe path discovery

## Change Guardrails

- Optimize for the main PC first.
- Do not remove a Windows-specific workaround just because it is unnecessary on the current machine.
- Prefer environment variables, runtime detection, or machine-local overrides when a fix is only validated on one machine.
- When changing launcher or packaging behavior, verify on both the main PC and the laptop when possible, but the main PC result is the priority.

## Verification Checklist

After touching launcher, packaging, subprocess, GPU startup, or FFmpeg discovery behavior, check:

1. Studio launches from source on the main PC.
2. Studio launches from source on the laptop.
3. No extra console window freeze/flash regression appears on the laptop.
4. If the `.exe` path changed, smoke-test the packaged build on Windows.

## Unknowns

- Other machine-specific differences may exist but are not yet documented here.
- Add concrete symptoms and the files/settings involved when new differences are discovered.
