import pytest

from videopipeline.layouts import RectNorm


def test_rect_norm_validation() -> None:
    RectNorm(x=0.1, y=0.2, w=0.3, h=0.4)
    with pytest.raises(ValueError):
        RectNorm(x=-0.1, y=0.2, w=0.3, h=0.4)
    with pytest.raises(ValueError):
        RectNorm(x=0.9, y=0.2, w=0.3, h=0.4)
    with pytest.raises(ValueError):
        RectNorm(x=0.1, y=0.2, w=0.0, h=0.4)


def test_rect_norm_to_pixels() -> None:
    rect = RectNorm(x=0.1, y=0.2, w=0.3, h=0.4)
    px = rect.to_pixels(width=1920, height=1080)
    assert px.x == 192
    assert px.y == 216
    assert px.w == 576
    assert px.h == 432
