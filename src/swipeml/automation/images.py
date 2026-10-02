"""Downloading a profile image without writing rubbish to disk.

The original:

    response = requests.get(url, stream=True)
    with open(temp_name, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)

No timeout, so a slow server hangs the run with no way out but Ctrl-C. No
status check, so a 404 or a rate-limit page is written to a `.jpg` and then fed
to the classifier, which happily returns a prediction about an error page. And
the file lands in the training set, where it stays.

Here: a timeout, the status checked, the content type checked, a size cap, and
the first bytes verified against the format's magic number before the file is
kept.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..dataset import MAGIC_NUMBERS


class HttpGetter(Protocol):
    """Just the part of `requests` this uses.

    A Protocol rather than `requests.Session`, so a test can pass a stub and
    nothing here needs requests installed to be type-checked.
    """

    def get(self, url: str, *, stream: bool = ..., timeout: float = ...) -> Any: ...

DEFAULT_TIMEOUT = 15.0
#: A profile photograph is not 50MB. A cap stops a mis-pointed URL filling a
#: disk during an unattended run.
MAX_BYTES = 12 * 1024 * 1024
#: Read in chunks so the cap can be enforced before the whole body is in memory.
CHUNK = 64 * 1024
ALLOWED_TYPES = ("image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp")


class DownloadError(RuntimeError):
    """The URL did not give us an image."""


@dataclass
class Download:
    path: Path
    bytes_written: int
    content_type: str


def looks_like_an_image(head: bytes) -> bool:
    return any(head.startswith(magic) for magic in MAGIC_NUMBERS)


def fetch_image(
    url: str,
    destination: str | Path,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = MAX_BYTES,
    session: HttpGetter | None = None,
) -> Download:
    """Download one image, or raise without leaving a file behind.

    Written to a temporary name and moved into place only once it is known to be
    an image, so a failed download cannot leave a half-file that a later run
    counts as training data.
    """
    import requests

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")

    http: Any = session if session is not None else requests
    try:
        response = http.get(url, stream=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - requests raises a family of these
        raise DownloadError(f"could not reach {url[:80]}: {exc}") from exc

    try:
        status = getattr(response, "status_code", 200)
        if status >= 400:
            raise DownloadError(f"{url[:80]} returned HTTP {status}")

        content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip()
        if content_type and not content_type.startswith("image/"):
            # The case that actually happens: an HTML error or consent page.
            raise DownloadError(
                f"{url[:80]} returned {content_type!r}, not an image"
            )

        written = 0
        head = b""
        with temporary.open("wb") as out:
            for chunk in response.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                if not head:
                    head = chunk[:16]
                    if not looks_like_an_image(head):
                        raise DownloadError(
                            f"{url[:80]} did not start with image data "
                            f"(first bytes: {head[:8]!r})"
                        )
                written += len(chunk)
                if written > max_bytes:
                    raise DownloadError(
                        f"{url[:80]} is over the {max_bytes // 1024 // 1024}MB cap"
                    )
                out.write(chunk)

        if not written:
            raise DownloadError(f"{url[:80]} returned an empty body")

        temporary.replace(target)
        return Download(target, written, content_type or "unknown")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def next_available(folder: str | Path, prefix: str, suffix: str = ".jpg") -> Path:
    """A filename that is not taken.

    The original computed `len(files) + 1` and then formatted `file_count + 1`,
    so names were offset by two, and after any deletion the next name collided
    with an existing file and overwrote training data. Checking is cheaper than
    arithmetic that has to stay correct.
    """
    base = Path(folder)
    base.mkdir(parents=True, exist_ok=True)
    index = sum(1 for p in base.iterdir() if p.is_file()) + 1
    while True:
        candidate = base / f"{prefix}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1
