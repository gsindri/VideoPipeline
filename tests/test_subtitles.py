from pathlib import Path

from videopipeline.subtitles import caption_theme_style, normalize_caption_theme, write_ass


def test_caption_theme_style_variants_scale_and_normalize() -> None:
    assert normalize_caption_theme("classic") == "clean"
    assert normalize_caption_theme("punch") == "impact"
    assert normalize_caption_theme("highlight-box") == "boxed"

    clean = caption_theme_style("clean", playres_y=1920)
    impact = caption_theme_style("impact", playres_y=1920)
    boxed = caption_theme_style("boxed", playres_y=1920)

    assert impact.fontsize > clean.fontsize
    assert boxed.border_style == 3
    assert boxed.outline == 0


def test_write_ass_uses_caption_theme_defaults(tmp_path: Path) -> None:
    out_path = tmp_path / "captions.ass"
    write_ass([], out_path, theme="boxed")

    payload = out_path.read_text(encoding="utf-8")
    assert "Style: Default,DejaVu Sans" in payload
    assert ",3,0,0,2," in payload
