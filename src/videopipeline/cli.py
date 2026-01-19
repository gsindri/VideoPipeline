from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path
from typing import Any, List, Optional

from .analysis_chat import compute_chat_analysis
from .analysis_highlights import compute_highlights_analysis
from .doctor import run_doctor
from .exporter import ExportSpec, HookTextSpec, LayoutPipSpec, run_ffmpeg_export
from .layouts import get_facecam_rect
from .metadata import build_metadata, derive_hook_text, write_metadata
from .profile import load_profile
from .project import add_selection_from_candidate, create_or_load_project, get_project_data
from .transcribe import TranscribeConfig, load_transcript_json, save_transcript_json, transcribe_segment


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60.0
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def _default_out_path(video_path: Path) -> Path:
    return Path("outputs") / video_path.stem / "candidates_highlights.json"


def cmd_suggest(args: argparse.Namespace) -> None:
    proj = create_or_load_project(args.video)
    profile = load_profile(args.profile)
    analysis_cfg = profile.get("analysis", {})

    payload = compute_highlights_analysis(
        proj,
        audio_cfg=analysis_cfg.get("audio", {}),
        motion_cfg=analysis_cfg.get("motion", {}),
        scenes_cfg=analysis_cfg.get("scenes", {}),
        highlights_cfg=analysis_cfg.get("highlights", {}),
        include_chat=True,
    )

    out_path = args.out or _default_out_path(args.video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    candidates = payload.get("candidates", [])

    print(f"Wrote: {out_path}")
    print(f"Project: {proj.project_dir}")
    print()
    print(f"{'Rank':>4}  {'Peak':>10}  {'Start':>10}  {'End':>10}  {'Score':>7}  {'Audio':>7}  {'Motion':>7}  {'Chat':>7}")
    for c in candidates:
        breakdown = c.get("breakdown", {})
        print(
            f"{c.get('rank'):>4}  {_fmt_time(c.get('peak_time_s')):>10}  {_fmt_time(c.get('start_s')):>10}  {_fmt_time(c.get('end_s')):>10}  {c.get('score', 0.0):>7.2f}  {breakdown.get('audio', 0.0):>7.2f}  {breakdown.get('motion', 0.0):>7.2f}  {breakdown.get('chat', 0.0):>7.2f}"
        )


def cmd_studio(args: argparse.Namespace) -> None:
    from .studio.app import create_app

    app = create_app(video_path=args.video, profile_path=args.profile)

    url = f"http://{args.host}:{args.port}"
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


def cmd_export(args: argparse.Namespace) -> None:
    # Headless export (useful in scripts/automation)
    spec = ExportSpec(
        video_path=args.video,
        start_s=args.start,
        end_s=args.end,
        output_path=args.out,
        template=args.template,
        width=args.width,
        height=args.height,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        normalize_audio=args.normalize_audio,
        subtitles_ass=args.subtitles_ass,
    )

    def on_prog(frac: float, msg: str) -> None:
        pct = int(frac * 100)
        print(f"\r{pct:3d}% {msg:10s}", end="", flush=True)

    run_ffmpeg_export(spec, on_progress=on_prog)
    print("\nDone:", spec.output_path)


def cmd_export_top(args: argparse.Namespace) -> None:
    proj = create_or_load_project(args.video)
    profile = load_profile(args.profile)
    proj_data = get_project_data(proj)

    highlights = proj_data.get("analysis", {}).get("highlights", {})
    candidates = highlights.get("candidates") or []
    if not candidates:
        analysis_cfg = profile.get("analysis", {})
        payload = compute_highlights_analysis(
            proj,
            audio_cfg=analysis_cfg.get("audio", {}),
            motion_cfg=analysis_cfg.get("motion", {}),
            scenes_cfg=analysis_cfg.get("scenes", {}),
            highlights_cfg=analysis_cfg.get("highlights", {}),
            include_chat=True,
        )
        candidates = payload.get("candidates", [])

    top_n = max(1, int(args.top))
    template = args.template or profile.get("export", {}).get("template", "vertical_blur")

    facecam = get_facecam_rect(proj_data.get("layout", {}))
    pip_cfg = profile.get("layout", {}).get("pip", {})
    hook_cfg = profile.get("overlay", {}).get("hook_text", {})
    export_cfg = profile.get("export", {})

    created: list[dict[str, Any]] = []
    for cand in candidates[:top_n]:
        sel_id = add_selection_from_candidate(proj, candidate=cand, template=template, title="")
        created.append(
            {
                "id": sel_id,
                "start_s": cand["start_s"],
                "end_s": cand["end_s"],
                "template": template,
                "candidate_rank": cand.get("rank"),
                "candidate_score": cand.get("score"),
                "candidate_peak_time_s": cand.get("peak_time_s"),
            }
        )

    for idx, selection in enumerate(created, start=1):
        start_s = float(selection["start_s"])
        end_s = float(selection["end_s"])
        out_path = proj.exports_dir / (
            f"{selection['id']}_{template}_{export_cfg.get('width', 1080)}x{export_cfg.get('height', 1920)}.mp4"
        )

        segments = None
        if args.with_captions:
            cfg = TranscribeConfig(
                model_size=str(profile.get("captions", {}).get("model_size", "small")),
                language=profile.get("captions", {}).get("language"),
                device=str(profile.get("captions", {}).get("device", "cpu")),
                compute_type=str(profile.get("captions", {}).get("compute_type", "int8")),
            )
            tjson = proj.analysis_dir / "transcripts" / f"{selection['id']}_{int(start_s)}_{int(end_s)}.json"
            if tjson.exists():
                segments = load_transcript_json(tjson)
            else:
                segments = transcribe_segment(proj.video_path, start_s=start_s, end_s=end_s, cfg=cfg)
                save_transcript_json(tjson, segments, cfg)
            ass_path = proj.analysis_dir / "subtitles" / f"{selection['id']}.ass"
            from .subtitles import write_ass

            subtitles_ass = write_ass(
                segments,
                ass_path,
                playres_x=int(export_cfg.get("width", 1080)),
                playres_y=int(export_cfg.get("height", 1920)),
            )
        else:
            subtitles_ass = None

        hook_spec = None
        if hook_cfg.get("enabled", False):
            hook_text = hook_cfg.get("text") or derive_hook_text(selection, segments)
            if hook_text:
                hook_spec = HookTextSpec(
                    enabled=True,
                    duration_seconds=float(hook_cfg.get("duration_seconds", 2.0)),
                    text=str(hook_text),
                    font=str(hook_cfg.get("font", "auto")),
                    fontsize=int(hook_cfg.get("fontsize", 64)),
                    y=int(hook_cfg.get("y", 120)),
                )

        spec = ExportSpec(
            video_path=proj.video_path,
            start_s=start_s,
            end_s=end_s,
            output_path=out_path,
            template=template,
            width=int(export_cfg.get("width", 1080)),
            height=int(export_cfg.get("height", 1920)),
            fps=int(export_cfg.get("fps", 30)),
            crf=int(export_cfg.get("crf", 20)),
            preset=str(export_cfg.get("preset", "veryfast")),
            normalize_audio=bool(export_cfg.get("normalize_audio", False)),
            subtitles_ass=subtitles_ass,
            layout_facecam=facecam,
            layout_pip=LayoutPipSpec(**pip_cfg) if pip_cfg else None,
            hook_text=hook_spec,
        )

        def on_prog(frac: float, msg: str) -> None:
            pct = int(frac * 100)
            print(f"\r[{idx}/{len(created)}] {pct:3d}% {msg:10s}", end="", flush=True)

        run_ffmpeg_export(spec, on_progress=on_prog)
        metadata = build_metadata(
            selection=selection,
            output_path=out_path,
            template=template,
            with_captions=args.with_captions,
            segments=segments,
        )
        write_metadata(out_path.with_suffix(".metadata.json"), metadata)
        print("\nDone:", out_path)


def cmd_doctor(_: argparse.Namespace) -> None:
    rep = run_doctor()
    print("VideoPipeline doctor\n")
    for name, data in rep.checks.items():
        print(f"- {name}:")
        for k, v in data.items():
            print(f"    {k}: {v}")
    print("\nOK" if rep.ok else "\nNOT OK (fix missing requirements above)")


def cmd_attach_chat(args: argparse.Namespace) -> None:
    proj = create_or_load_project(args.video)
    proj.analysis_dir.mkdir(parents=True, exist_ok=True)
    dest = proj.chat_raw_path
    dest.write_bytes(Path(args.chat).read_bytes())

    hop_s = args.hop
    if hop_s is None:
        proj_data = get_project_data(proj)
        hop_s = (
            proj_data.get("analysis", {})
            .get("audio", {})
            .get("config", {})
            .get("hop_seconds", 0.5)
        )

    compute_chat_analysis(
        proj,
        chat_path=dest,
        hop_s=float(hop_s),
        smooth_s=float(args.smooth),
    )
    print(f"Attached chat: {dest}")
    print(f"Project: {proj.project_dir}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="vp", description="VideoPipeline CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("suggest", help="Suggest highlight candidates from combined signals.")
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--profile", type=Path, default=None)
    p.set_defaults(func=cmd_suggest)

    s = sub.add_parser("studio", help="Launch the local review + export studio web app.")
    s.add_argument("video", type=Path)
    s.add_argument("--profile", type=Path, default=None, help="Path to a YAML profile (e.g. profiles/gaming.yaml)")
    s.add_argument("--host", type=str, default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--reload", action="store_true", help="Auto-reload server on code changes (dev)")
    s.add_argument("--no-open", action="store_true", help="Do not open a browser automatically")
    s.set_defaults(func=cmd_studio)

    e = sub.add_parser("export", help="Headless export of a specific clip range.")
    e.add_argument("video", type=Path)
    e.add_argument("--start", type=float, required=True)
    e.add_argument("--end", type=float, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--template", type=str, default="vertical_blur")
    e.add_argument("--width", type=int, default=1080)
    e.add_argument("--height", type=int, default=1920)
    e.add_argument("--fps", type=int, default=30)
    e.add_argument("--crf", type=int, default=20)
    e.add_argument("--preset", type=str, default="veryfast")
    e.add_argument("--normalize-audio", action="store_true")
    e.add_argument("--subtitles-ass", type=Path, default=None)
    e.set_defaults(func=cmd_export)

    et = sub.add_parser("export-top", help="Create selections from top candidates and export them.")
    et.add_argument("video", type=Path)
    et.add_argument("--top", type=int, default=10)
    et.add_argument("--template", type=str, default=None)
    et.add_argument("--profile", type=Path, default=None)
    et.add_argument("--with-captions", action="store_true")
    et.set_defaults(func=cmd_export_top)

    d = sub.add_parser("doctor", help="Check local system dependencies (ffmpeg, optional whisper).")
    d.set_defaults(func=cmd_doctor)

    c = sub.add_parser("attach-chat", help="Attach chat JSON and compute chat features.")
    c.add_argument("video", type=Path)
    c.add_argument("chat", type=Path)
    c.add_argument("--hop", type=float, default=None)
    c.add_argument("--smooth", type=float, default=3.0)
    c.set_defaults(func=cmd_attach_chat)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
