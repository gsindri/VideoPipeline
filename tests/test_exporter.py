from pathlib import Path

import videopipeline.exporter as exporter
from videopipeline.exporter import (
    ExportSpec,
    HookTextSpec,
    _escape_drawtext_text,
    build_ffmpeg_command,
    normalize_camera_plan,
)


def test_escape_drawtext_text_handles_filter_delimiters() -> None:
    raw = "Legendary Spawn, Then It's 100%; [Clip]\nGo:"
    escaped = _escape_drawtext_text(raw)

    assert "\n" not in escaped
    assert "\\," in escaped
    assert "\\'" in escaped
    assert "\\%" in escaped
    assert "\\;" in escaped
    assert "\\[" in escaped and "\\]" in escaped
    assert "\\:" in escaped


def test_build_ffmpeg_command_hook_uses_expansion_none(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(exporter, "_require_cmd", lambda *_: None)

    font_path = tmp_path / "dummy_font.ttf"
    font_path.write_bytes(b"")
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"")
    output_path = tmp_path / "out.mp4"

    spec = ExportSpec(
        video_path=video_path,
        start_s=1.0,
        end_s=3.0,
        output_path=output_path,
        template="vertical_blur",
        hook_text=HookTextSpec(
            enabled=True,
            text="100% clutch, it's real;",
            font=str(font_path),
        ),
    )

    cmd = build_ffmpeg_command(spec)
    vf_value = cmd[cmd.index("-filter_complex") + 1]
    assert "drawtext=" in vf_value
    assert "expansion=none:" in vf_value
    assert "100\\% clutch\\, it\\'s real\\;" in vf_value
    assert ",null[vout]" in vf_value
    assert "enable='lt(t\\,2.00)'[vout]" not in vf_value


def test_normalize_camera_plan_clamps_and_sorts_keyframes() -> None:
    camera_plan = normalize_camera_plan(
        [
            {"at_s": 2.0, "focus_x": 0.7, "focus_y": 0.4, "zoom": 1.8},
            {"at_s": 0.0, "focus_x": -1.0, "focus_y": 2.0, "zoom": 0.5},
            {"at_s": 9.0, "focus_x": 0.3, "focus_y": 0.6, "zoom": 9.0},
        ],
        duration_s=4.0,
    )

    assert [frame.at_s for frame in camera_plan] == [0.0, 2.0, 4.0]
    assert camera_plan[0].focus_x == 0.0
    assert camera_plan[0].focus_y == 1.0
    assert camera_plan[0].zoom == 1.0
    assert camera_plan[-1].zoom == 4.0


def test_build_ffmpeg_command_includes_camera_plan_crop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(exporter, "_require_cmd", lambda *_: None)

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"")
    output_path = tmp_path / "out.mp4"

    spec = ExportSpec(
        video_path=video_path,
        start_s=0.0,
        end_s=4.0,
        output_path=output_path,
        layout_preset="proof_overlay",
        camera_plan=normalize_camera_plan(
            [
                {"at_s": 0.0, "focus_x": 0.45, "focus_y": 0.35, "zoom": 1.0},
                {"at_s": 2.0, "focus_x": 0.62, "focus_y": 0.48, "zoom": 1.6},
            ]
        ),
    )

    cmd = build_ffmpeg_command(spec)
    vf_value = cmd[cmd.index("-filter_complex") + 1]
    assert "crop=w='" in vf_value
    assert "clip((t-0)/2,0,1)" in vf_value
    assert "scale=1080:-2:force_original_aspect_ratio=decrease[fg]" in vf_value
