from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from .audio_features import AudioFeatureConfig, audio_rms_db_timeline
from .ffmpeg import ffprobe_duration_seconds
from .peaks import moving_average, pick_top_peaks, robust_z


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
    stem = video_path.stem
    return Path("outputs") / stem / "candidates_audio.json"


def suggest_audio_candidates(
    video_path: Path,
    *,
    sample_rate: int,
    hop_s: float,
    smooth_s: float,
    top: int,
    min_gap_s: float,
    pre_s: float,
    post_s: float,
    skip_start_s: float,
) -> tuple[List[CandidateClip], dict]:
    video_path = Path(video_path)
    duration_s = ffprobe_duration_seconds(video_path)

    cfg = AudioFeatureConfig(sample_rate=sample_rate, hop_seconds=hop_s)
    timeline_db = audio_rms_db_timeline(video_path, cfg)

    if not timeline_db:
        raise RuntimeError("No audio timeline computed. Is the file valid and does it contain audio?")

    x = np.array(timeline_db, dtype=np.float64)

    smooth_frames = max(1, int(round(smooth_s / hop_s)))
    xs = moving_average(x, smooth_frames)

    scores = robust_z(xs)

    # Optionally ignore the start (intros, silence)
    skip_frames = int(round(skip_start_s / hop_s))
    if skip_frames > 0:
        scores[:skip_frames] = -np.inf

    min_gap_frames = max(1, int(round(min_gap_s / hop_s)))
    peak_idxs = pick_top_peaks(scores, top_k=top, min_gap_frames=min_gap_frames)

    candidates: List[CandidateClip] = []
    for rank, idx in enumerate(peak_idxs, start=1):
        peak_time = float(idx * hop_s)
        start = max(0.0, peak_time - pre_s)
        end = min(duration_s, peak_time + post_s)
        if end - start < 3.0:
            continue
        candidates.append(
            CandidateClip(
                rank=rank,
                peak_time_s=peak_time,
                start_s=start,
                end_s=end,
                score=float(scores[idx]),
                peak_db=float(x[idx]),
            )
        )

    meta = {
        "video": str(video_path),
        "duration_seconds": duration_s,
        "method": "audio_rms_db_peaks",
        "sample_rate": sample_rate,
        "hop_seconds": hop_s,
        "smooth_seconds": smooth_s,
        "top_requested": top,
        "min_gap_seconds": min_gap_s,
        "pre_seconds": pre_s,
        "post_seconds": post_s,
        "skip_start_seconds": skip_start_s,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return candidates, meta


def render_preview_clips(
    video_path: Path,
    candidates: List[CandidateClip],
    out_dir: Path,
    *,
    reencode: bool,
) -> None:
    """
    Optional helper: render small preview clips to quickly review the suggestions.
    Default uses stream copy for speed; --reencode can make cuts more accurate.
    """
    import subprocess

    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(video_path)

    for c in candidates:
        duration = max(0.1, c.end_s - c.start_s)
        out_file = out_dir / f"clip_{c.rank:02d}_{int(c.start_s)}s_{int(c.end_s)}s.mp4"

        if reencode:
            cmd = [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{c.start_s:.3f}",
                "-i",
                str(video_path),
                "-t",
                f"{duration:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(out_file),
            ]
        else:
            cmd = [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{c.start_s:.3f}",
                "-i",
                str(video_path),
                "-t",
                f"{duration:.3f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(out_file),
            ]

        subprocess.check_call(cmd)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="vp", description="VideoPipeline CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("suggest", help="Suggest highlight candidates from audio excitement peaks.")
    p.add_argument("video", type=Path, help="Path to local video file")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--hop", type=float, default=0.5, help="Seconds per loudness frame")
    p.add_argument("--smooth", type=float, default=3.0, help="Smoothing window in seconds")
    p.add_argument("--top", type=int, default=12, help="Number of candidates to return")
    p.add_argument("--min-gap", type=float, default=20.0, help="Minimum seconds between selected peaks")
    p.add_argument("--pre", type=float, default=8.0, help="Seconds before peak to start clip")
    p.add_argument("--post", type=float, default=22.0, help="Seconds after peak to end clip")
    p.add_argument("--skip-start", type=float, default=10.0, help="Ignore first N seconds when picking peaks")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: outputs/<video>/candidates_audio.json)",
    )

    p.add_argument(
        "--render-previews",
        action="store_true",
        help="Render preview clips to review quickly",
    )
    p.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Where to write previews (default: outputs/<video>/previews)",
    )
    p.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode previews (more accurate cutting; requires libx264)",
    )

    args = parser.parse_args(argv)

    if args.cmd == "suggest":
        out_path = args.out or _default_out_path(args.video)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        candidates, meta = suggest_audio_candidates(
            args.video,
            sample_rate=args.sample_rate,
            hop_s=args.hop,
            smooth_s=args.smooth,
            top=args.top,
            min_gap_s=args.min_gap,
            pre_s=args.pre,
            post_s=args.post,
            skip_start_s=args.skip_start,
        )

        payload = {
            **meta,
            "candidates": [asdict(c) for c in candidates],
        }

        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # Print a nice summary
        print(f"Wrote: {out_path}")
        print()
        print(f"{'Rank':>4}  {'Peak':>10}  {'Start':>10}  {'End':>10}  {'Score':>7}  {'dB':>7}")
        for c in candidates:
            print(
                f"{c.rank:>4}  {_fmt_time(c.peak_time_s):>10}  {_fmt_time(c.start_s):>10}  "
                f"{_fmt_time(c.end_s):>10}  {c.score:>7.2f}  {c.peak_db:>7.2f}"
            )

        if args.render_previews:
            preview_dir = args.preview_dir or (out_path.parent / "previews")
            render_preview_clips(args.video, candidates, preview_dir, reencode=args.reencode)
            print()
            print(f"Previews written to: {preview_dir}")


if __name__ == "__main__":
    main()
