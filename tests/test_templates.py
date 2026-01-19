from videopipeline.exporter import filtergraph_for_template


def test_filtergraph_templates() -> None:
    assert filtergraph_for_template("original", 1080, 1920) == "null"
    assert "overlay" in filtergraph_for_template("vertical_blur", 1080, 1920)
    assert "crop" in filtergraph_for_template("vertical_crop_center", 1080, 1920)
