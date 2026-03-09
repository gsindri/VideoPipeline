# Gondull Control Plane

## Purpose

Define the intended operating model for `VideoPipeline` so both humans and `Gondull` treat it as a backend pipeline service controlled through `Gondull_Platform`, not as a human-first desktop app.

## Primary Architecture

- `Gondull_Platform` is the primary operator surface.
- `Gondull` running through OpenClaw is the primary orchestrator.
- `VideoPipeline` is the local pipeline backend that performs ingest, analysis, export, and publishing work.
- The `VideoPipeline` Studio UI remains available for manual review, debugging, and fallback operation.

## Default Orchestration Mode

- Default profile: `profiles/gaming_assemblyai.yaml`
- Default LLM mode: `gondull` / `external_strict`
- Preferred control surface: `VideoPipeline` Actions API (`/api/actions/*`)

## Work Split

### AssemblyAI

Use AssemblyAI for the heavy cloud workload in the default orchestration profile, especially:

- speech transcription
- diarization in the speech pipeline
- audio-event intelligence

### Gondull

Use `Gondull` for the LLM-driven decision work, including:

- semantic candidate review
- chapter labeling
- director picks
- clip-review decisions
- other external-AI pipeline judgments

Do not assume the local in-process LLM is the default path when the control plane is operating normally.

## Preferred Workflow

1. `Gondull_Platform` issues or relays the task.
2. `Gondull` checks `VideoPipeline` diagnostics/readiness.
3. `Gondull` runs ingest/analyze/export/publish flows through the Actions API.
4. `VideoPipeline` performs the media pipeline work.
5. `Gondull` handles the LLM judgment steps and sends the results back through the same control path.

## Current Integration Reality

- The intended operator skill for end-to-end control is `videopipeline-operator` in `/home/sindri/.openclaw/workspace/skills/videopipeline-operator/SKILL.md`.
- `Gondull_Platform` currently discovers the active local `VideoPipeline` runtime and can queue manual scout URLs into the scout inbox.
- `Gondull_Platform` slash commands currently scaffold the `videopipeline-operator` prompt, but they do not proxy the full `/api/actions/*` workflow yet.
- After a task is issued from `Gondull_Platform`, `Gondull` should drive the remaining preflight, ingest, external-AI apply, export, and publish steps directly through `VideoPipeline` Actions API calls.

## Operator Guidance

- Treat direct Studio use as secondary.
- Prefer backend/API integration work over Studio-only manual flows when choosing where to improve the system.
- When code or docs conflict, align them toward `Gondull_Platform` plus `Gondull` as the primary control plane.

## Related Files

- `docs/actions.md`
- `docs/studio.md`
- `profiles/gaming_assemblyai.yaml`
- `src/videopipeline/studio/actions_api.py`
- `src/videopipeline/studio/static/app.js`
