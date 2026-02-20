from pathlib import Path

import videopipeline.exporter as exporter
from videopipeline.exporter import ExportSpec, HookTextSpec, _escape_drawtext_text, build_ffmpeg_command


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
