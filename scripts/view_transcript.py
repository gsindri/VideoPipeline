#!/usr/bin/env python3
"""View and compare transcription quality.

Usage:
    python scripts/view_transcript.py                    # View latest transcript
    python scripts/view_transcript.py --full             # Show all segments
    python scripts/view_transcript.py --time 120 180     # Show segments from 2:00-3:00
    python scripts/view_transcript.py --compare          # Compare if you have multiple
"""

import argparse
import json
from pathlib import Path


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def load_transcript(path: Path) -> dict:
    """Load transcript JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def print_transcript_info(data: dict, path: Path):
    """Print transcript metadata."""
    print(f"File: {path}")
    print(f"Backend: {data.get('backend_used', 'unknown')}")
    print(f"GPU used: {data.get('gpu_used', False)}")
    print(f"Language: {data.get('detected_language', 'unknown')}")
    print(f"Duration: {format_time(data.get('duration_seconds', 0))}")
    print(f"Segments: {data.get('segment_count', 0)}")
    
    config = data.get("config", {})
    print(f"Model: {config.get('model_size', 'unknown')}")
    print(f"VAD filter: {config.get('vad_filter', 'unknown')}")
    print(f"Word timestamps: {config.get('word_timestamps', 'unknown')}")
    print()


def print_segments(data: dict, start_s: float = 0, end_s: float = float("inf"), limit: int = None):
    """Print transcript segments."""
    segments = data.get("transcript", {}).get("segments", [])
    
    # Filter by time range
    filtered = [s for s in segments if s["end"] > start_s and s["start"] < end_s]
    
    if limit:
        filtered = filtered[:limit]
    
    for seg in filtered:
        time_str = f"[{format_time(seg['start'])} - {format_time(seg['end'])}]"
        print(f"{time_str} {seg['text']}")
        
        # Show word-level if available
        if seg.get("words"):
            words_preview = seg["words"][:5]
            word_strs = [f"{w['word']}({w.get('probability', 1.0):.2f})" for w in words_preview]
            if len(seg["words"]) > 5:
                word_strs.append(f"... +{len(seg['words'])-5} more")
            print(f"         Words: {' '.join(word_strs)}")
    
    print(f"\n[Showing {len(filtered)} of {len(segments)} segments]")


def find_transcripts(projects_dir: Path) -> list[Path]:
    """Find all transcript files."""
    return sorted(
        projects_dir.glob("*/analysis/transcript_full.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )


def main():
    parser = argparse.ArgumentParser(description="View transcription quality")
    parser.add_argument("--full", action="store_true", help="Show all segments")
    parser.add_argument("--time", nargs=2, type=float, metavar=("START", "END"),
                        help="Show segments in time range (seconds)")
    parser.add_argument("--limit", type=int, default=10, help="Max segments to show (default: 10)")
    parser.add_argument("--compare", action="store_true", help="Compare multiple transcripts")
    parser.add_argument("--project", type=str, help="Project hash prefix to select")
    
    args = parser.parse_args()
    
    projects_dir = Path("outputs/projects")
    if not projects_dir.exists():
        print("No projects directory found. Run analysis first.")
        return
    
    transcripts = find_transcripts(projects_dir)
    if not transcripts:
        print("No transcripts found. Run speech analysis first.")
        return
    
    # Select transcript(s)
    if args.project:
        transcripts = [t for t in transcripts if args.project in str(t)]
    
    if not transcripts:
        print(f"No transcript found matching '{args.project}'")
        return
    
    # Load and display
    if args.compare and len(transcripts) > 1:
        print("=" * 60)
        print("COMPARING TRANSCRIPTS")
        print("=" * 60)
        for t in transcripts[:3]:  # Compare up to 3
            data = load_transcript(t)
            print_transcript_info(data, t)
            print("-" * 40)
    else:
        data = load_transcript(transcripts[0])
        print_transcript_info(data, transcripts[0])
        print("=" * 60)
        print("TRANSCRIPT")
        print("=" * 60)
        
        start_s = args.time[0] if args.time else 0
        end_s = args.time[1] if args.time else float("inf")
        limit = None if args.full else args.limit
        
        print_segments(data, start_s, end_s, limit)


if __name__ == "__main__":
    main()
