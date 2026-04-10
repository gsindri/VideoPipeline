from videopipeline.exporter import LayoutPipSpec, filtergraph_for_template, layout_preset_to_template, normalize_layout_preset
from videopipeline.layouts import RectNorm


def test_filtergraph_templates() -> None:
    assert filtergraph_for_template("original", 1080, 1920) == "null"
    assert "overlay" in filtergraph_for_template("vertical_blur", 1080, 1920)
    assert "overlay" in filtergraph_for_template("proof_overlay", 1080, 1920)
    assert "crop" in filtergraph_for_template("vertical_crop_center", 1080, 1920)
    assert "crop" in filtergraph_for_template("single_subject_punch", 1080, 1920)

    rect = RectNorm(x=0.1, y=0.1, w=0.2, h=0.2)
    pip_graph = filtergraph_for_template(
        "vertical_streamer_pip",
        1080,
        1920,
        layout_facecam=rect,
        source_width=1920,
        source_height=1080,
        pip_spec=LayoutPipSpec(),
    )
    assert "overlay" in pip_graph
    assert "overlay" in filtergraph_for_template(
        "speaker_broll",
        1080,
        1920,
        layout_facecam=rect,
        source_width=1920,
        source_height=1080,
        pip_spec=LayoutPipSpec(),
    )

    split_graph = filtergraph_for_template(
        "vertical_streamer_split",
        1080,
        1920,
        layout_facecam=rect,
        source_width=1920,
        source_height=1080,
    )
    assert "vstack" in split_graph
    assert "vstack" in filtergraph_for_template(
        "reaction_stack",
        1080,
        1920,
        layout_facecam=rect,
        source_width=1920,
        source_height=1080,
    )


def test_layout_preset_normalization_accepts_legacy_templates() -> None:
    assert normalize_layout_preset("vertical_blur") == "proof_overlay"
    assert normalize_layout_preset("vertical_streamer_pip") == "speaker_broll"
    assert normalize_layout_preset("vertical_streamer_split") == "reaction_stack"
    assert normalize_layout_preset("vertical_crop_center") == "single_subject_punch"
    assert layout_preset_to_template("speaker_broll") == "vertical_streamer_pip"
