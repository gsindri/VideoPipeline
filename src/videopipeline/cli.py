from __future__ import annotations

import argparse
import json
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .analysis_audio import compute_audio_analysis
from .doctor import run_doctor
from .exporter import ExportSpec, run_ffmpeg_export
from .project import create_or_load_project, get_project_data


@dataclass(frozen=True)
class CandidateClip:
    rank: int
    peak_time_s: float
    start_s: float
    end_s: float
    score: float
    peak_db: float


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60.0
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def _default_out_path(video_path: Path) -> Path:
    return Path("outputs") / video_path.stem / "candidates_audio.json"


def cmd_suggest(args: argparse.Namespace) -> None:
    proj = create_or_load_project(args.video)

    payload = compute_audio_analysis(
        proj,
        sample_rate=args.sample_rate,
        hop_s=args.hop,
        smooth_s=args.smooth,
        top=args.top,
        min_gap_s=args.min_gap,
        pre_s=args.pre,
        post_s=args.post,
        skip_start_s=args.skip_start,
    )

    out_path = args.out or _default_out_path(args.video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    candidates = [CandidateClip(**c) for c in payload.get("candidates", [])]

    print(f"Wrote: {out_path}")
    print(f"Project: {proj.project_dir}")
    print()
    print(f"{'Rank':>4}  {'Peak':>10}  {'Start':>10}  {'End':>10}  {'Score':>7}  {'dB':>7}")
    for c in candidates:
        print(
            f"{c.rank:>4}  {_fmt_time(c.peak_time_s):>10}  {_fmt_time(c.start_s):>10}  {_fmt_time(c.end_s):>10}  {c.score:>7.2f}  {c.peak_db:>7.2f}"
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


def cmd_doctor(_: argparse.Namespace) -> None:
    rep = run_doctor()
    print("VideoPipeline doctor\n")
    for name, data in rep.checks.items():
        print(f"- {name}:")
        for k, v in data.items():
            print(f"    {k}: {v}")
    print("\nOK" if rep.ok else "\nNOT OK (fix missing requirements above)")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="vp", description="VideoPipeline CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("suggest", help="Suggest highlight candidates from audio excitement peaks.")
    p.add_argument("video", type=Path)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--hop", type=float, default=0.5)
    p.add_argument("--smooth", type=float, default=3.0)
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--min-gap", type=float, default=20.0)
    p.add_argument("--pre", type=float, default=8.0)
    p.add_argument("--post", type=float, default=22.0)
    p.add_argument("--skip-start", type=float, default=10.0)
    p.add_argument("--out", type=Path, default=None)
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

    d = sub.add_parser("doctor", help="Check local system dependencies (ffmpeg, optional whisper).")
    d.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
