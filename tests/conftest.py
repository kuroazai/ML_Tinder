"""Shared fixtures.

No real photographs here. The images are the shortest byte strings that count as
their format, which is enough for everything except actually running a model
through them, and a model is not what most of these tests are about.
"""
from __future__ import annotations

from pathlib import Path

import pytest

#: A minimal valid JPEG header, then padding. `dataset.looks_like_an_image`
#: checks the magic number, so this passes and an HTML error page does not.
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
#: What a rate-limited or expired image URL actually returns, and what the
#: original wrote to disk with a .jpg name and then fed to the classifier.
HTML_ERROR_PAGE = b"<!DOCTYPE html>\n<html><body>429 Too Many Requests</body></html>"


def write_images(folder: Path, count: int, prefix: str, payload: bytes = JPEG_BYTES) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (folder / f"{prefix}_{index}.jpg").write_bytes(payload)


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """A usable, slightly unbalanced dataset: 120 liked, 80 disliked."""
    root = tmp_path / "training_data"
    write_images(root / "liked", 120, "liked")
    write_images(root / "disliked", 80, "disliked")
    return root


@pytest.fixture
def balanced_root(tmp_path: Path) -> Path:
    root = tmp_path / "balanced"
    write_images(root / "liked", 100, "liked")
    write_images(root / "disliked", 100, "disliked")
    return root


class FakeElement:
    """Stands in for a Selenium WebElement."""

    def __init__(self, *, displayed: bool = True, style: str = "", stale: bool = False):
        self.displayed = displayed
        self.style = style
        self.stale = stale
        self.clicks = 0

    def is_displayed(self) -> bool:
        if self.stale:
            raise RuntimeError("stale element reference")
        return self.displayed

    def get_attribute(self, name: str) -> str:
        return self.style if name == "style" else ""

    def click(self) -> None:
        self.clicks += 1


class FakeDriver:
    """Answers `find_elements` from a dict, and records what was asked for."""

    def __init__(self, answers: dict[str, list[FakeElement]] | None = None):
        self.answers = answers or {}
        self.asked: list[str] = []
        self.page_source = "<html></html>"

    def find_elements(self, by, what):  # noqa: ANN001 - mirrors selenium
        self.asked.append(what)
        return self.answers.get(what, [])


@pytest.fixture
def fake_driver():
    return FakeDriver
