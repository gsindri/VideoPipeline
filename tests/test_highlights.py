import numpy as np

from videopipeline.analysis_highlights import (
    ClipConfig,
    _merge_candidate_extras,
    resample_series,
    shape_clip_bounds,
    snap_time_to_cuts,
)


def test_resample_series_linear():
    values = np.array([0.0, 10.0, 20.0], dtype=np.float64)
    out = resample_series(values, src_hop_s=1.0, target_hop_s=0.5, target_len=5)
    assert np.allclose(out, [0.0, 5.0, 10.0, 15.0, 20.0])


def test_snap_time_to_cuts():
    cuts = [1.0, 2.2, 5.0]
    assert snap_time_to_cuts(2.0, cuts, 0.3) == 2.2
    assert snap_time_to_cuts(1.6, cuts, 0.3) == 1.6


def test_shape_clip_bounds_uses_valleys():
    scores = np.array([1.0, 0.1, 2.0, 5.0, 2.0, 0.2, 1.0], dtype=np.float64)
    clip_cfg = ClipConfig(
        min_seconds=2.0,
        max_seconds=10.0,
        min_pre_seconds=1.0,
        max_pre_seconds=3.0,
        min_post_seconds=1.0,
        max_post_seconds=3.0,
    )
    bounds = shape_clip_bounds(
        peak_idx=3,
        scores=scores,
        hop_s=1.0,
        duration_s=10.0,
        clip_cfg=clip_cfg,
        scene_cuts=[],
        snap_window_s=0.0,
    )
    assert bounds["start_s"] == 1.0
    assert bounds["end_s"] == 5.0


def test_merge_candidate_extras_preserves_ai_and_enrich_fields():
    prev = [
        {
            "rank": 1,
            "peak_time_s": 100.0,
            "start_s": 95.0,
            "end_s": 110.0,
            "ai": {"title": "MILK FIRST (NO JUDGMENT)", "hook": "MILK FIRST?!"},
            "hook_text": "MILK FIRST?!",
            "quote_text": "I pour the milk first.",
        }
    ]
    new = [
        {
            "rank": 1,
            "peak_time_s": 100.3,
            "start_s": 95.5,
            "end_s": 110.5,
        }
    ]
    merged = _merge_candidate_extras(new, prev)
    assert merged[0]["ai"]["title"] == "MILK FIRST (NO JUDGMENT)"
    assert merged[0]["hook_text"] == "MILK FIRST?!"
    assert merged[0]["quote_text"] == "I pour the milk first."


def test_merge_candidate_extras_does_not_overmatch():
    prev = [
        {
            "rank": 1,
            "peak_time_s": 10.0,
            "start_s": 5.0,
            "end_s": 15.0,
            "ai": {"title": "OLD"},
        }
    ]
    new = [
        {
            "rank": 1,
            "peak_time_s": 100.0,
            "start_s": 95.0,
            "end_s": 110.0,
        }
    ]
    merged = _merge_candidate_extras(new, prev)
    assert "ai" not in merged[0]
