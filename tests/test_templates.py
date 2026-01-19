from videopipeline.exporter import LayoutPipSpec, filtergraph_for_template
from videopipeline.layouts import RectNorm


def test_filtergraph_templates() -> None:
    assert filtergraph_for_template("original", 1080, 1920) == "null"
    assert "overlay" in filtergraph_for_template("vertical_blur", 1080, 1920)
    assert "crop" in filtergraph_for_template("vertical_crop_center", 1080, 1920)

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

    split_graph = filtergraph_for_template(
        "vertical_streamer_split",
        1080,
        1920,
        layout_facecam=rect,
        source_width=1920,
        source_height=1080,
    )
    assert "vstack" in split_graph
