from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

from fastapi import Request
from fastapi.responses import StreamingResponse


def _parse_range_header(range_header: str, file_size: int) -> Optional[Tuple[int, int]]:
    # Expected format: bytes=start-end
    if not range_header:
        return None
    if not range_header.startswith("bytes="):
        return None
    ranges = range_header.replace("bytes=", "", 1).strip()
    # We only handle a single range
    if "," in ranges:
        ranges = ranges.split(",", 1)[0].strip()
    if "-" not in ranges:
        return None
    start_s, end_s = ranges.split("-", 1)
    start_s = start_s.strip()
    end_s = end_s.strip()

    if start_s == "":
        # suffix bytes: e.g. "-500"
        try:
            length = int(end_s)
        except ValueError:
            return None
        length = min(length, file_size)
        start = file_size - length
        end = file_size - 1
        return start, end

    try:
        start = int(start_s)
    except ValueError:
        return None

    if end_s == "":
        end = file_size - 1
    else:
        try:
            end = int(end_s)
        except ValueError:
            return None

    if start < 0:
        start = 0
    if end >= file_size:
        end = file_size - 1
    if end < start:
        return None

    return start, end


def _iter_file_range(path: Path, start: int, end: int, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    with path.open("rb") as f:
        f.seek(start)
        remaining = (end - start) + 1
        while remaining > 0:
            to_read = min(chunk_size, remaining)
            data = f.read(to_read)
            if not data:
                break
            remaining -= len(data)
            yield data


def ranged_file_response(
    request: Request,
    path: Path,
    *,
    media_type: str,
) -> StreamingResponse:
    """Serve a file with HTTP Range support (critical for HTML5 video seeking)."""
    path = Path(path)
    file_size = path.stat().st_size

    range_header = request.headers.get("range")
    byte_range = _parse_range_header(range_header, file_size) if range_header else None

    if byte_range is None:
        # Full response
        def it() -> Iterator[bytes]:
            yield from _iter_file_range(path, 0, file_size - 1)

        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        }
        return StreamingResponse(it(), status_code=200, media_type=media_type, headers=headers)

    start, end = byte_range
    content_length = (end - start) + 1

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
    }

    return StreamingResponse(
        _iter_file_range(path, start, end),
        status_code=206,
        media_type=media_type,
        headers=headers,
    )
