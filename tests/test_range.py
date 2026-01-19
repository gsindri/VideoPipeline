from videopipeline.studio.range import _parse_range_header


def test_parse_range_basic() -> None:
    assert _parse_range_header("bytes=0-99", 1000) == (0, 99)


def test_parse_range_open_end() -> None:
    assert _parse_range_header("bytes=100-", 1000) == (100, 999)


def test_parse_range_suffix() -> None:
    assert _parse_range_header("bytes=-200", 1000) == (800, 999)


def test_parse_range_invalid() -> None:
    assert _parse_range_header("nope", 1000) is None
    assert _parse_range_header("bytes=999-100", 1000) is None
