"""Downloading an image without writing rubbish to disk."""
from __future__ import annotations

from pathlib import Path

import pytest
from conftest import HTML_ERROR_PAGE, JPEG_BYTES

from swipeml.automation import images


class FakeResponse:
    def __init__(self, body: bytes, *, status: int = 200,
                 content_type: str = "image/jpeg"):
        self.body = body
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        self.closed = False

    def iter_content(self, chunk_size: int = 1024):
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start : start + chunk_size]

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, response=None, raises: Exception | None = None):
        self.response = response
        self.raises = raises
        self.calls: list[dict] = []

    def get(self, url, stream=False, timeout=None):  # noqa: ANN001
        self.calls.append({"url": url, "stream": stream, "timeout": timeout})
        if self.raises:
            raise self.raises
        return self.response


def test_a_good_download_is_written(tmp_path: Path) -> None:
    session = FakeSession(FakeResponse(JPEG_BYTES))
    result = images.fetch_image("https://ex.com/a.jpg", tmp_path / "a.jpg",
                                session=session)

    assert result.path.read_bytes() == JPEG_BYTES
    assert result.bytes_written == len(JPEG_BYTES)
    assert result.content_type == "image/jpeg"


def test_a_timeout_is_always_passed(tmp_path: Path) -> None:
    """The original had none, so a slow server hung the run with no way out but
    Ctrl-C."""
    session = FakeSession(FakeResponse(JPEG_BYTES))
    images.fetch_image("https://ex.com/a.jpg", tmp_path / "a.jpg", session=session)
    assert session.calls[0]["timeout"] is not None


def test_an_http_error_does_not_become_a_jpg(tmp_path: Path) -> None:
    """The original had no `raise_for_status`, so a 404 body was written to a
    .jpg, added to the training set, and classified."""
    session = FakeSession(FakeResponse(b"Not Found", status=404,
                                       content_type="text/html"))
    target = tmp_path / "a.jpg"

    with pytest.raises(images.DownloadError, match="HTTP 404"):
        images.fetch_image("https://ex.com/a.jpg", target, session=session)
    assert not target.exists()


def test_an_html_error_page_with_a_200_is_rejected(tmp_path: Path) -> None:
    """A rate-limit or consent page served with a 200, which is what actually
    happens."""
    session = FakeSession(FakeResponse(HTML_ERROR_PAGE, content_type="text/html"))
    target = tmp_path / "a.jpg"

    with pytest.raises(images.DownloadError, match="not an image"):
        images.fetch_image("https://ex.com/a.jpg", target, session=session)
    assert not target.exists()


def test_a_body_that_is_not_image_data_is_rejected(tmp_path: Path) -> None:
    """Even when the Content-Type claims otherwise, the first bytes are checked
    against the format's magic number."""
    session = FakeSession(FakeResponse(HTML_ERROR_PAGE, content_type="image/jpeg"))
    target = tmp_path / "a.jpg"

    with pytest.raises(images.DownloadError, match="did not start with image data"):
        images.fetch_image("https://ex.com/a.jpg", target, session=session)
    assert not target.exists()


def test_an_oversized_body_is_refused(tmp_path: Path) -> None:
    """A cap stops a mis-pointed URL filling a disk during an unattended run."""
    session = FakeSession(FakeResponse(JPEG_BYTES + b"\x00" * 5000))
    target = tmp_path / "a.jpg"

    with pytest.raises(images.DownloadError, match="cap"):
        images.fetch_image("https://ex.com/a.jpg", target, session=session,
                           max_bytes=1024)
    assert not target.exists()


def test_an_empty_body_is_refused(tmp_path: Path) -> None:
    session = FakeSession(FakeResponse(b"", content_type="image/jpeg"))
    with pytest.raises(images.DownloadError, match="empty"):
        images.fetch_image("https://ex.com/a.jpg", tmp_path / "a.jpg",
                           session=session)


def test_a_network_error_is_wrapped(tmp_path: Path) -> None:
    session = FakeSession(raises=OSError("connection reset"))
    with pytest.raises(images.DownloadError, match="could not reach"):
        images.fetch_image("https://ex.com/a.jpg", tmp_path / "a.jpg",
                           session=session)


def test_no_partial_file_is_left_behind(tmp_path: Path) -> None:
    """A half-file with a .jpg name would be counted as training data by the
    next run."""
    session = FakeSession(FakeResponse(HTML_ERROR_PAGE, content_type="image/jpeg"))
    target = tmp_path / "a.jpg"

    with pytest.raises(images.DownloadError):
        images.fetch_image("https://ex.com/a.jpg", target, session=session)

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []  # not even the .part file


def test_the_response_is_closed(tmp_path: Path) -> None:
    response = FakeResponse(JPEG_BYTES)
    images.fetch_image("https://ex.com/a.jpg", tmp_path / "a.jpg",
                       session=FakeSession(response))
    assert response.closed


# -- naming ----------------------------------------------------------------

def test_next_available_does_not_collide(tmp_path: Path) -> None:
    """The original computed `len(files) + 1` and then formatted
    `file_count + 1`, so names were offset by two, and after any deletion the
    next name overwrote an existing file."""
    folder = tmp_path / "liked"
    folder.mkdir()
    for index in (1, 2, 3):
        (folder / f"liked_{index}.jpg").write_bytes(JPEG_BYTES)

    assert images.next_available(folder, "liked").name == "liked_4.jpg"


def test_next_available_survives_a_gap(tmp_path: Path) -> None:
    """Delete one and the count no longer predicts a free name."""
    folder = tmp_path / "liked"
    folder.mkdir()
    for index in (1, 2, 3, 4, 5):
        (folder / f"liked_{index}.jpg").write_bytes(JPEG_BYTES)
    (folder / "liked_2.jpg").unlink()

    chosen = images.next_available(folder, "liked")
    assert not chosen.exists()


def test_next_available_creates_the_folder(tmp_path: Path) -> None:
    chosen = images.next_available(tmp_path / "new" / "deeper", "card")
    assert chosen.parent.is_dir()
    assert chosen.name == "card_1.jpg"
