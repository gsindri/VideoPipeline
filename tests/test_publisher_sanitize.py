from videopipeline.publisher.sanitize import sanitize_metadata


def test_sanitize_youtube_truncates_and_normalizes():
    metadata = {
        "title": "  Hello   World  ",
        "caption": "Caption",
        "description": "D" * 6000,
        "hashtags": ["tag1", "#tag2", "  "],
    }
    result = sanitize_metadata("youtube", metadata)
    assert result["title"] == "Hello World"
    assert result["description"].endswith("…")
    assert len(result["description"]) <= 5000
    assert result["hashtags"] == ["#tag1", "#tag2"]


def test_sanitize_tiktok_caption_combines_tags():
    metadata = {
        "title": "Title",
        "caption": "Caption",
        "hashtags": ["tag1", "tag2"],
    }
    result = sanitize_metadata("tiktok", metadata)
    assert "#tag1" in result["caption"]
    assert "#tag2" in result["caption"]
