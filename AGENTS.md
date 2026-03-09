# VideoPipeline Agent Context

## Project Relationship

- `VideoPipeline` exists for the OpenClaw bot `Gondull` to oversee and manage.
- Gondull's OpenClaw home is `/home/sindri/.openclaw` inside WSL.
- From Windows, that same location is `\\wsl.localhost\Ubuntu\home\sindri\.openclaw`.
- Gondull is used through the companion app at `/mnt/c/Users/gsind/Projects/Gondull_Platform`.
- The intended primary control plane is `Gondull_Platform` + `Gondull`/OpenClaw, not direct human-first use of VideoPipeline Studio.
- `VideoPipeline` should be treated as a pipeline service/tooling backend that can still be run locally, but the priority integration path is via `Gondull_Platform`.
- Prefer `/api/actions/*` and other backend interfaces over manual Studio-first workflows when implementing the primary architecture.
- Default orchestrated profile is `profiles/gaming_assemblyai.yaml` unless explicitly overridden.
- In that profile, AssemblyAI is the intended heavy-work backend for speech/audio intelligence.
- `Gondull` is the intended LLM operator for semantic scoring, review, chapter labeling, director picks, and other LLM-driven pipeline tasks via `gondull` / `external_strict` mode.
- When `Gondull` is asked to operate `VideoPipeline` end-to-end, prefer the OpenClaw skill `videopipeline-operator` at `/home/sindri/.openclaw/workspace/skills/videopipeline-operator/SKILL.md`.
- Direct Studio use is secondary and mainly for manual review, debugging, and fallback operation.
- When integration work spans both codebases, it is acceptable to update both `VideoPipeline` and `Gondull_Platform` together.
- See `docs/gondull-control-plane.md` for the intended operating model.

## Cross-Machine Context

- This project has been worked on from both the main PC and a laptop.
- The main PC is the primary target environment and should be optimized first.
- Laptop support is best-effort only and should not block improvements for the main PC.
- The two machines have had different local/runtime quirks and may require different settings or workarounds.
- Do not remove or "clean up" machine-specific Windows fixes just because they are not reproducible on the current machine, unless laptop support is being intentionally dropped.
- Prefer environment-scoped settings, launcher/runtime detection, or machine-specific overrides over changing shared defaults when a fix is only validated on one machine.
- Be especially careful around `launcher.py`, `VideoPipeline.spec`, subprocess/console behavior, GPU/runtime startup, and FFmpeg path discovery.
- A known example is Windows `cmd` / console freezing or flashing behavior that appeared on the laptop but not on the main PC.
- See `docs/machine-notes.md` for tracked cross-machine behavior and guardrails.
