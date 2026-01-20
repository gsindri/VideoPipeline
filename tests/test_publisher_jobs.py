from pathlib import Path

from videopipeline.publisher.jobs import PublishJobStore


def test_job_store_create_update_and_dedup(tmp_path: Path):
    store = PublishJobStore(path=tmp_path / "publisher.sqlite")
    job = store.create_job(
        job_id="job123",
        platform="youtube",
        account_id="acct",
        file_path="/tmp/video.mp4",
        metadata_path="/tmp/meta.json",
    )
    assert job.status == "queued"

    updated = store.update_job(job.id, status="running", progress=0.5)
    assert updated.status == "running"
    assert updated.progress == 0.5

    store.mark_dedup("youtube", "acct", "hash123", "remote123", "https://youtu.be/remote123")
    dedup = store.lookup_dedup("youtube", "acct", "hash123")
    assert dedup is not None
    assert dedup["remote_id"] == "remote123"


def test_job_store_backoff_skips_recent_retries(tmp_path: Path):
    store = PublishJobStore(path=tmp_path / "publisher.sqlite")
    store.create_job(
        job_id="job123",
        platform="youtube",
        account_id="acct",
        file_path="/tmp/video.mp4",
        metadata_path="/tmp/meta.json",
    )
    store.update_job("job123", attempts=1)
    claimed = store.claim_next(backoff_fn=lambda attempts: 999)
    assert claimed is None
