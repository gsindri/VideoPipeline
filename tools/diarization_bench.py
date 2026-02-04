from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


_SCAN_DIRS = ("outputs", "out", "runs", "data", "cache", "downloads")
_SCAN_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".opus", ".mkv", ".mp4", ".webm")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path() -> None:
    repo = _repo_root()
    src = repo / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark speaker diarization backends (pyannote vs Senko).")
    p.add_argument("--input", type=Path, default=None, help="Input audio/video file path.")
    p.add_argument(
        "--auto-latest",
        action="store_true",
        help=f"If set, scan {', '.join(_SCAN_DIRS)} for the most recently modified media file.",
    )
    p.add_argument(
        "--seconds",
        type=int,
        default=600,
        help="If >0, benchmark only the first N seconds (trim during conversion). Default: 600.",
    )
    p.add_argument(
        "--backend",
        choices=("pyannote", "senko", "both"),
        default="both",
        help="Which backend(s) to run. Default: both.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs") / "diarization_bench",
        help="Output directory for benchmark artifacts.",
    )
    p.add_argument(
        "--senko-python",
        type=Path,
        default=None,
        help="Path to the Python interpreter in the Senko venv (separate env).",
    )
    p.add_argument("--keep-wav", action="store_true", help="Keep the converted 16kHz WAV in outdir.")
    return p.parse_args()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _as_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _normalize_segments(raw_segments: Iterable[Any]) -> list[dict[str, Any]]:
    segs: list[dict[str, Any]] = []
    for seg in raw_segments:
        if isinstance(seg, dict):
            speaker = seg.get("speaker")
            start = _as_float(seg.get("start"))
            end = _as_float(seg.get("end"))
        else:
            speaker = getattr(seg, "speaker", None)
            start = _as_float(getattr(seg, "start", None))
            end = _as_float(getattr(seg, "end", None))

        if start is None or end is None:
            continue
        if end <= start:
            continue
        segs.append({"speaker": "" if speaker is None else str(speaker), "start": float(start), "end": float(end)})

    segs.sort(key=lambda s: (float(s["start"]), float(s["end"]), str(s["speaker"])))

    # Stable-ish speaker relabeling for diffs (based on first appearance).
    mapping: dict[str, str] = {}
    next_id = 0
    for s in segs:
        raw = str(s.get("speaker") or "")
        if raw not in mapping:
            mapping[raw] = f"SPEAKER_{next_id:02d}"
            next_id += 1
        s["speaker"] = mapping[raw]

    return segs


def _find_latest_media(repo_root: Path, *, exclude_under: set[Path]) -> Path | None:
    exts = {e.lower() for e in _SCAN_EXTS}
    best: tuple[float, Path] | None = None

    def _excluded(p: Path) -> bool:
        for ex in exclude_under:
            try:
                if ex == p or ex in p.parents:
                    return True
            except Exception:
                continue
        return False

    for d in _SCAN_DIRS:
        base = repo_root / d
        if not base.exists():
            continue
        try:
            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                if _excluded(p):
                    continue
                if p.suffix.lower() not in exts:
                    continue
                try:
                    mtime = p.stat().st_mtime
                except Exception:
                    continue
                if best is None or mtime > best[0]:
                    best = (mtime, p)
        except Exception:
            continue

    return best[1] if best else None


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found in PATH. Install ffmpeg and ensure it is available on PATH "
            "(needed to convert input to 16kHz mono 16-bit WAV)."
        )
    return ffmpeg


def _convert_to_16k_mono_s16_wav(*, ffmpeg: str, input_path: Path, out_wav: Path, seconds: int) -> float:
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(input_path),
    ]
    if int(seconds) > 0:
        cmd += ["-t", str(int(seconds))]
    cmd += [
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if proc.returncode != 0 or not out_wav.exists():
        err = (proc.stderr or proc.stdout or "").strip()
        if err:
            err = "\n" + err
        raise RuntimeError(f"ffmpeg conversion failed (exit={proc.returncode}).{err}")
    return float(dt)


def _resolve_senko_python(repo_root: Path, explicit: Path | None) -> Path | None:
    if explicit:
        return explicit
    candidates = [
        repo_root / "tools" / "senko" / ".venv" / "bin" / "python",
        repo_root / "tools" / "senko" / ".venv" / "Scripts" / "python.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


@dataclass
class BackendResult:
    status: str  # ok | error | skipped
    runtime_sec: float | None = None
    segment_count: int | None = None
    speaker_count: int | None = None
    outputs: dict[str, str] | None = None
    error: str | None = None


def _run_pyannote(*, wav_path: Path, out_json: Path) -> BackendResult:
    try:
        from videopipeline.transcription import diarization as dia
    except Exception as exc:
        return BackendResult(status="error", error=f"Failed to import videopipeline diarization module: {exc}")

    if not getattr(dia, "is_diarization_available", None) or not dia.is_diarization_available():
        return BackendResult(
            status="error",
            error="pyannote diarization is not available. Install optional deps and ensure pyannote-audio imports.",
        )

    try:
        t0 = time.perf_counter()
        diar = dia.diarize_audio(wav_path)
        runtime_s = time.perf_counter() - t0
        segments = _normalize_segments(getattr(diar, "segments", []) or [])
        _json_dump(out_json, segments)
        speakers = sorted({str(s["speaker"]) for s in segments})
        return BackendResult(
            status="ok",
            runtime_sec=float(runtime_s),
            segment_count=int(len(segments)),
            speaker_count=int(len(speakers)),
            outputs={"segments_json": str(out_json)},
        )
    except Exception as exc:
        return BackendResult(status="error", error=str(exc))


def _run_senko(
    *,
    repo_root: Path,
    wav_path: Path,
    out_json: Path,
    out_rttm: Path,
    senko_python: Path | None,
) -> BackendResult:
    if senko_python is None or not Path(senko_python).exists():
        return BackendResult(
            status="skipped",
            error=(
                "Senko venv not found. Create it under tools/senko/ (see docs/diarization_bench.md) "
                "or pass --senko-python PATH."
            ),
        )

    runner = repo_root / "tools" / "senko_runner.py"
    cmd = [
        str(senko_python),
        str(runner),
        "--wav",
        str(wav_path),
        "--out-json",
        str(out_json),
        "--out-rttm",
        str(out_rttm),
        "--device",
        "cpu",
        "--quiet",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    runtime_s = time.perf_counter() - t0

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if err:
            err = "\n" + err
        return BackendResult(status="error", error=f"senko_runner failed (exit={proc.returncode}).{err}")

    try:
        segments = json.loads(out_json.read_text(encoding="utf-8"))
        segments = _normalize_segments(segments)
        # Re-write to ensure stable schema even if runner changes.
        _json_dump(out_json, segments)
        speakers = sorted({str(s["speaker"]) for s in segments})
        return BackendResult(
            status="ok",
            runtime_sec=float(runtime_s),
            segment_count=int(len(segments)),
            speaker_count=int(len(speakers)),
            outputs={"segments_json": str(out_json), "rttm": str(out_rttm)},
        )
    except Exception as exc:
        return BackendResult(status="error", error=f"Failed to read Senko outputs: {exc}")


def _format_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out_lines = []
    for idx, r in enumerate(rows):
        line = "  ".join(c.ljust(widths[i]) for i, c in enumerate(r))
        out_lines.append(line.rstrip())
        if idx == 0:
            out_lines.append("  ".join("-" * w for w in widths).rstrip())
    return "\n".join(out_lines)


def main() -> int:
    _ensure_src_on_path()
    args = _parse_args()

    repo_root = _repo_root()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report_path = outdir / "report.json"

    exclude_under = {outdir.resolve()}
    if args.input is None:
        if not args.auto_latest:
            print("Error: provide --input PATH or set --auto-latest.", file=sys.stderr)
            return 2
        chosen = _find_latest_media(repo_root, exclude_under=exclude_under)
        if chosen is None:
            print(
                "Error: --auto-latest found no media files. Looked under: "
                + ", ".join(_SCAN_DIRS)
                + " (extensions: "
                + ", ".join(_SCAN_EXTS)
                + ").",
                file=sys.stderr,
            )
            return 2
        input_path = chosen
        input_source = "auto-latest"
    else:
        input_path = Path(args.input)
        input_source = "input"

    if not input_path.exists():
        print(f"Error: input path does not exist: {input_path}", file=sys.stderr)
        return 2

    ffmpeg = _require_ffmpeg()

    seconds = int(args.seconds or 0)
    if seconds < 0:
        seconds = 0

    report: dict[str, Any] = {
        "input": {
            "path": str(input_path),
            "source": input_source,
            "slice_seconds": seconds if seconds > 0 else None,
        },
        "system": {
            "platform": platform.platform(),
            "python": {
                "version": platform.python_version(),
                "executable": sys.executable,
            },
        },
        "prep": {},
        "backends": {},
    }
    _json_dump(report_path, report)

    keep_wav = bool(args.keep_wav)
    if keep_wav:
        wav_path = outdir / "input_16k_mono_s16.wav"
        tmp_ctx: Any = None
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="vp_diar_bench_")
        wav_path = Path(tmp_ctx.name) / "input_16k_mono_s16.wav"

    exit_code = 0
    try:
        report["prep"]["ffmpeg"] = ffmpeg
        report["prep"]["convert_runtime_sec"] = _convert_to_16k_mono_s16_wav(
            ffmpeg=ffmpeg,
            input_path=input_path,
            out_wav=wav_path,
            seconds=seconds,
        )
        report["prep"]["wav_path"] = str(wav_path) if keep_wav else None
        _json_dump(report_path, report)

        want_pyannote = args.backend in ("pyannote", "both")
        want_senko = args.backend in ("senko", "both")

        results: dict[str, BackendResult] = {}

        if want_pyannote:
            res = _run_pyannote(wav_path=wav_path, out_json=outdir / "pyannote_segments.json")
            results["pyannote"] = res
            report["backends"]["pyannote"] = res.__dict__
            _json_dump(report_path, report)
            if res.status != "ok":
                exit_code = 1

        if want_senko:
            senko_python = _resolve_senko_python(repo_root, args.senko_python)
            res = _run_senko(
                repo_root=repo_root,
                wav_path=wav_path,
                out_json=outdir / "senko_segments.json",
                out_rttm=outdir / "senko.rttm",
                senko_python=senko_python,
            )
            results["senko"] = res
            report["backends"]["senko"] = res.__dict__
            _json_dump(report_path, report)

            if res.status == "error":
                exit_code = 1
            elif res.status == "skipped" and args.backend == "senko":
                exit_code = 2

        # Console summary (short).
        rows = [["backend", "status", "runtime_s", "segments", "speakers"]]
        for k in ("pyannote", "senko"):
            if k not in results:
                continue
            r = results[k]
            rows.append(
                [
                    k,
                    r.status,
                    "-" if r.runtime_sec is None else f"{r.runtime_sec:.3f}",
                    "-" if r.segment_count is None else str(r.segment_count),
                    "-" if r.speaker_count is None else str(r.speaker_count),
                ]
            )
        print(_format_table(rows))
        print(f"Wrote: {outdir}")

        # Extra guidance when Senko is requested but missing.
        senko_res = results.get("senko")
        if senko_res and senko_res.status == "skipped":
            print(senko_res.error or "Senko skipped.")

        return exit_code
    finally:
        if tmp_ctx is not None:
            try:
                tmp_ctx.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

