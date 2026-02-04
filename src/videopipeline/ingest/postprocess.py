"""Post-processing for downloaded videos.

Handles:
- ffprobe codec/container detection
- MPEG-TS to MP4 remuxing (no re-encode)
- Preview proxy generation for browser playback
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

from ..ffmpeg import _require_cmd
from ..utils import subprocess_flags as _subprocess_flags
from .models import PostprocessResult


ProgressCallback = Callable[[float, str], None]


def probe_video(video_path: Path) -> dict[str, Any]:
    """Probe a video file with ffprobe.
    
    Returns dict with:
        - container: file extension
        - video_codec: video codec name
        - audio_codec: audio codec name
        - duration: duration in seconds
        - streams: full stream info
    """
    try:
        ffprobe = _require_cmd("ffprobe")
        
        result = subprocess.run(
            [
                ffprobe, "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(video_path)
            ],
            capture_output=True,
            text=True,
            timeout=30,
            **_subprocess_flags(),
        )
        
        if result.returncode != 0:
            return {"error": "ffprobe failed"}
        
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        format_info = data.get("format", {})
        
        video_codec = None
        audio_codec = None
        
        for stream in streams:
            codec_type = stream.get("codec_type")
            codec_name = stream.get("codec_name", "").lower()
            
            if codec_type == "video" and video_codec is None:
                video_codec = codec_name
            elif codec_type == "audio" and audio_codec is None:
                audio_codec = codec_name
        
        return {
            "container": video_path.suffix.lower(),
            "video_codec": video_codec or "unknown",
            "audio_codec": audio_codec or "none",
            "duration": float(format_info.get("duration", 0) or 0),
            "format_name": format_info.get("format_name", ""),
            "streams": streams,
        }
    
    except FileNotFoundError:
        return {"error": "ffprobe not found"}
    except Exception as e:
        return {"error": str(e)}


def needs_remux(probe_result: dict[str, Any]) -> bool:
    """Check if video needs remuxing to MP4 container.
    
    Remux is needed for:
    - MPEG-TS container (.ts)
    - MKV with compatible codecs
    - ADTS audio streams
    """
    if "error" in probe_result:
        return False
    
    container = probe_result.get("container", "")
    format_name = probe_result.get("format_name", "")
    audio_codec = probe_result.get("audio_codec", "")
    
    # MPEG-TS needs remux
    if container in (".ts", ".m2ts") or "mpegts" in format_name:
        return True
    
    # MKV can often be remuxed if codecs are compatible
    if container == ".mkv":
        video_codec = probe_result.get("video_codec", "")
        if video_codec in ("h264", "avc", "avc1", "hevc", "h265"):
            if audio_codec in ("aac", "mp4a", "opus", "none"):
                return True
    
    # ADTS audio needs BSF filter
    if audio_codec == "aac" and "adts" in format_name.lower():
        return True
    
    return False


def needs_preview(probe_result: dict[str, Any]) -> bool:
    """Check if video needs a browser-friendly preview.
    
    Preview is needed if:
    - Video codec is not H.264
    - Audio codec is not AAC/none
    - Container is not MP4/M4V
    """
    if "error" in probe_result:
        return True  # If we can't probe, create preview to be safe
    
    container = probe_result.get("container", "")
    video_codec = probe_result.get("video_codec", "")
    audio_codec = probe_result.get("audio_codec", "")
    
    # Browser-friendly: H.264 video + AAC audio in MP4 container
    is_h264 = video_codec in ("h264", "avc", "avc1")
    is_aac_or_none = audio_codec in ("aac", "mp4a", "none")
    is_mp4 = container in (".mp4", ".m4v")
    
    return not (is_h264 and is_aac_or_none and is_mp4)


def remux_to_mp4(
    source_path: Path,
    output_path: Optional[Path] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Optional[Path]:
    """Remux video to MP4 container without re-encoding.
    
    Handles MPEG-TS and ADTS audio properly.
    
    Args:
        source_path: Path to source video
        output_path: Path for output (default: source with .mp4 extension)
        on_progress: Progress callback
    
    Returns:
        Path to remuxed file, or None on failure
    """
    try:
        ffmpeg = _require_cmd("ffmpeg")
        
        if output_path is None:
            output_path = source_path.with_suffix(".mp4")
        
        # Don't overwrite source if it's already mp4
        if output_path == source_path:
            output_path = source_path.parent / f"{source_path.stem}_remux.mp4"
        
        if on_progress:
            on_progress(0.0, "Remuxing to MP4...")
        
        # Use bitstream filter for ADTS AAC
        cmd = [
            ffmpeg, "-y",
            "-i", str(source_path),
            "-c", "copy",
            "-bsf:a", "aac_adtstoasc",
            "-movflags", "+faststart",
            str(output_path),
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            **_subprocess_flags(),
        )
        
        if result.returncode == 0 and output_path.exists():
            if on_progress:
                on_progress(1.0, "Remux complete")
            return output_path
        
        return None
    
    except Exception:
        return None


def create_preview(
    source_path: Path,
    preview_path: Optional[Path] = None,
    height: int = 720,
    crf: int = 28,
    on_progress: Optional[ProgressCallback] = None,
) -> Optional[Path]:
    """Create a browser-friendly H.264/AAC preview.
    
    Attempts hardware acceleration (AMF > NVENC > QSV > CPU) for faster encoding.
    
    Args:
        source_path: Path to source video
        preview_path: Path for preview (default: source_stem_preview.mp4)
        height: Target height (default 720p)
        crf: Quality (higher = smaller file, default 28)
        on_progress: Progress callback
    
    Returns:
        Path to preview file, or None on failure
    """
    try:
        ffmpeg = _require_cmd("ffmpeg")
        
        if preview_path is None:
            preview_path = source_path.parent / f"{source_path.stem}_preview.mp4"
        
        if on_progress:
            on_progress(0.0, "Creating browser preview...")
        
        # Scale to target height while maintaining aspect ratio
        # -2 ensures width is divisible by 2 (required for H.264)
        scale_filter = f"scale=-2:{height}"
        
        # Try hardware encoders in order: AMF (AMD) > NVENC (NVIDIA) > QSV (Intel) > CPU
        # Hardware encoding is 5-10x faster than CPU
        encoder_configs = [
            # AMD AMF
            {
                "codec": "h264_amf",
                "opts": ["-quality", "speed", "-rc", "vbr_latency", "-qp_i", str(crf), "-qp_p", str(crf)],
                "label": "AMF",
            },
            # NVIDIA NVENC
            {
                "codec": "h264_nvenc",
                "opts": ["-preset", "p4", "-rc", "vbr", "-cq", str(crf)],
                "label": "NVENC",
            },
            # Intel Quick Sync
            {
                "codec": "h264_qsv",
                "opts": ["-preset", "fast", "-global_quality", str(crf)],
                "label": "QSV",
            },
            # CPU fallback
            {
                "codec": "libx264",
                "opts": ["-preset", "veryfast", "-crf", str(crf)],
                "label": "CPU",
            },
        ]
        
        for config in encoder_configs:
            cmd = [
                ffmpeg, "-y",
                "-i", str(source_path),
                "-c:v", config["codec"],
                *config["opts"],
                "-vf", scale_filter,
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                str(preview_path),
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,
                **_subprocess_flags(),
            )
            
            if result.returncode == 0 and preview_path.exists():
                if on_progress:
                    on_progress(1.0, f"Preview created ({config['label']})")
                return preview_path
            
            # Clean up partial file before trying next encoder
            if preview_path.exists():
                try:
                    preview_path.unlink()
                except Exception:
                    pass
        
        return None
    
    except Exception:
        return None


def postprocess_download(
    video_path: Path,
    create_preview_if_needed: bool = True,
    preview_height: int = 720,
    preview_crf: int = 28,
    on_progress: Optional[ProgressCallback] = None,
) -> PostprocessResult:
    """Post-process a downloaded video.
    
    Steps:
    1. Probe the video
    2. If MPEG-TS or similar, remux to MP4
    3. If codecs not browser-friendly, create preview
    
    Args:
        video_path: Path to downloaded video
        create_preview_if_needed: Whether to create preview for non-browser codecs
        preview_height: Preview resolution
        preview_crf: Preview quality
        on_progress: Progress callback
    
    Returns:
        PostprocessResult with source and preview paths
    """
    result = PostprocessResult(source_path=video_path)
    
    # Probe the video
    probe = probe_video(video_path)
    
    if "error" not in probe:
        result.source_video_codec = probe.get("video_codec", "")
        result.source_audio_codec = probe.get("audio_codec", "")
        result.source_container = probe.get("container", "")
    
    current_source = video_path
    
    # Step 1: Remux if needed
    if needs_remux(probe):
        if on_progress:
            on_progress(0.5, "Remuxing to MP4...")
        
        remuxed = remux_to_mp4(current_source, on_progress=on_progress)
        if remuxed:
            result.remuxed = True
            # Update source path to remuxed version
            current_source = remuxed
            result.source_path = remuxed
            
            # Re-probe the remuxed file
            probe = probe_video(remuxed)
    
    # Step 2: Create preview if needed
    if create_preview_if_needed and needs_preview(probe):
        if on_progress:
            on_progress(0.7, "Creating browser preview...")
        
        preview = create_preview(
            current_source,
            height=preview_height,
            crf=preview_crf,
            on_progress=on_progress,
        )
        if preview:
            result.preview_created = True
            result.preview_path = preview
    
    return result
