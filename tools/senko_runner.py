from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Senko diarization and write JSON + RTTM outputs.")
    p.add_argument("--wav", type=Path, required=True, help="Input WAV (16kHz mono 16-bit PCM recommended).")
    p.add_argument("--out-json", type=Path, required=True, help="Output path for normalized segments JSON.")
    p.add_argument("--out-rttm", type=Path, required=True, help="Output path for RTTM.")
    p.add_argument("--device", type=str, default="cpu", help="Senko device (use 'cpu' for this benchmark).")
    p.add_argument("--quiet", action="store_true", help="Suppress non-essential output (stdout stays JSON stats).")
    return p.parse_args()


def _as_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _get_attr_or_key(obj: Any, *names: str) -> Any:
    if isinstance(obj, dict):
        for n in names:
            if n in obj:
                return obj.get(n)
        return None
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def _normalize_segments(raw_segments: Iterable[Any]) -> list[dict[str, Any]]:
    segs: list[dict[str, Any]] = []
    for seg in raw_segments:
        start = _as_float(_get_attr_or_key(seg, "start", "start_s", "start_time"))
        end = _as_float(_get_attr_or_key(seg, "end", "end_s", "end_time"))
        speaker = _get_attr_or_key(seg, "speaker", "label", "speaker_id", "spk")
        if start is None or end is None:
            continue
        if end <= start:
            continue
        segs.append({"speaker": "" if speaker is None else str(speaker), "start": float(start), "end": float(end)})

    segs.sort(key=lambda s: (float(s["start"]), float(s["end"]), str(s["speaker"])))

    # Remap to stable SPEAKER_XX labels (first appearance in time order).
    mapping: dict[str, str] = {}
    next_id = 0
    for s in segs:
        raw = str(s.get("speaker") or "")
        if raw not in mapping:
            mapping[raw] = f"SPEAKER_{next_id:02d}"
            next_id += 1
        s["speaker"] = mapping[raw]

    return segs


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_rttm(path: Path, segments: list[dict[str, Any]], *, file_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for s in segments:
        start = float(s["start"])
        end = float(s["end"])
        dur = max(0.0, end - start)
        spk = str(s["speaker"])
        lines.append(f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"Input WAV not found: {wav_path}", file=sys.stderr)
        return 2

    try:
        import senko  # type: ignore
    except Exception as exc:
        print(f"Failed to import senko: {exc}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    try:
        try:
            diarizer = senko.Diarizer(device=str(args.device), warmup=True, quiet=bool(args.quiet))
        except TypeError:
            diarizer = senko.Diarizer(device=str(args.device), warmup=True)

        try:
            result = diarizer.diarize(str(wav_path), generate_colors=False)
        except TypeError:
            result = diarizer.diarize(str(wav_path))
    except Exception as exc:
        print(f"Senko diarization failed: {exc}", file=sys.stderr)
        return 1
    runtime_s = time.perf_counter() - t0

    merged = None
    if isinstance(result, dict):
        merged = result.get("merged_segments") or result.get("segments") or result.get("diarization")
    if merged is None:
        print("Senko returned no segments (expected key: merged_segments).", file=sys.stderr)
        return 1

    segments = _normalize_segments(merged)
    _write_json(Path(args.out_json), segments)
    _write_rttm(Path(args.out_rttm), segments, file_id=wav_path.stem)

    speakers = sorted({str(s["speaker"]) for s in segments})
    stats = {
        "runtime_sec": float(runtime_s),
        "segment_count": int(len(segments)),
        "speaker_count": int(len(speakers)),
    }
    print(json.dumps(stats, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

