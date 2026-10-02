"""The swipe loop: budgets, dry runs, and terminating."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from conftest import JPEG_BYTES

from swipeml.automation import locators, swiper


@dataclass
class FakePrediction:
    score: float
    label: str
    threshold: float = 0.5

    @property
    def positive(self) -> bool:
        return self.score >= self.threshold


class FakeClassifier:
    """Returns a scripted sequence of scores."""

    def __init__(self, scores: list[float]):
        self.scores = list(scores)
        self.calls = 0

    def predict(self, path):  # noqa: ANN001
        score = self.scores[self.calls % len(self.scores)]
        self.calls += 1
        return FakePrediction(score, "liked" if score >= 0.5 else "disliked")


class FakeSession:
    def __init__(self, driver):
        self.driver = driver

    def start(self):
        return self.driver


@pytest.fixture
def no_delay(monkeypatch):
    """Make the loop run instantly."""
    monkeypatch.setattr(swiper.time, "sleep", lambda _s: None)


@pytest.fixture
def fake_page(monkeypatch):
    """A page that always offers a card and both buttons."""
    clicked = {"like": 0, "pass": 0}

    monkeypatch.setattr(
        swiper, "current_card_url", lambda _d: "https://ex.com/card.jpg"
    )

    class Button:
        def __init__(self, kind):
            self.kind = kind

        def click(self):
            clicked[self.kind] += 1

    def find_or_none(self, driver):  # noqa: ANN001
        kind = "like" if self is locators.LIKE_BUTTON else "pass"
        return Button(kind), locators.Strategy("css", "stub", stability=10)

    monkeypatch.setattr(locators.Locator, "find_or_none", find_or_none)
    return clicked


@pytest.fixture
def fake_download(monkeypatch):
    def fetch(url, destination, **kwargs):  # noqa: ANN001
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(JPEG_BYTES)
        from swipeml.automation.images import Download

        return Download(target, len(JPEG_BYTES), "image/jpeg")

    monkeypatch.setattr(swiper.images, "fetch_image", fetch)


# -- limits ----------------------------------------------------------------

def test_limits_reject_nonsense() -> None:
    with pytest.raises(ValueError, match="delay_range"):
        swiper.SwipeLimits(delay_range=(5.0, 1.0))
    with pytest.raises(ValueError, match="delay_range"):
        swiper.SwipeLimits(delay_range=(-1.0, 1.0))
    with pytest.raises(ValueError, match="max_actions"):
        swiper.SwipeLimits(max_actions=0)


def test_the_delay_is_sampled_not_fixed() -> None:
    """A fixed `time.sleep(3)` between every action is a metronome."""
    import random

    limits = swiper.SwipeLimits(delay_range=(1.0, 4.0))
    generator = random.Random(1)
    samples = {round(limits.delay(generator), 6) for _ in range(20)}
    assert len(samples) > 1
    assert all(1.0 <= s <= 4.0 for s in samples)


# -- the loop --------------------------------------------------------------

def test_a_dry_run_classifies_and_clicks_nothing(
    tmp_path: Path, no_delay, fake_page, fake_download
) -> None:
    result = swiper.run(
        FakeSession(object()),
        FakeClassifier([0.9, 0.1]),
        liked_dir=tmp_path / "liked",
        disliked_dir=tmp_path / "disliked",
        limits=swiper.SwipeLimits(max_actions=6),
        dry_run=True,
    )

    assert len(result.outcomes) == 6
    assert result.actions == 0
    assert fake_page == {"like": 0, "pass": 0}
    assert all("dry run" in o.note for o in result.outcomes)


def test_a_dry_run_still_files_the_images(
    tmp_path: Path, no_delay, fake_page, fake_download
) -> None:
    """Which is the point: it builds a labelled set you can check by eye."""
    swiper.run(
        FakeSession(object()),
        FakeClassifier([0.9, 0.1]),
        liked_dir=tmp_path / "liked",
        disliked_dir=tmp_path / "disliked",
        limits=swiper.SwipeLimits(max_actions=4),
        dry_run=True,
    )
    assert len(list((tmp_path / "liked").iterdir())) == 2
    assert len(list((tmp_path / "disliked").iterdir())) == 2


def test_a_live_run_clicks_the_matching_button(
    tmp_path: Path, no_delay, fake_page, fake_download
) -> None:
    swiper.run(
        FakeSession(object()),
        FakeClassifier([0.9, 0.9, 0.1]),
        liked_dir=tmp_path / "liked",
        disliked_dir=tmp_path / "disliked",
        limits=swiper.SwipeLimits(max_actions=3),
        dry_run=False,
    )
    assert fake_page == {"like": 2, "pass": 1}


def test_the_budget_is_respected(
    tmp_path: Path, no_delay, fake_page, fake_download
) -> None:
    """The original's loop could not terminate: the default `--validation 1`
    branch did `continue` without decrementing the counter."""
    result = swiper.run(
        FakeSession(object()),
        FakeClassifier([0.9]),
        liked_dir=tmp_path / "liked",
        disliked_dir=tmp_path / "disliked",
        limits=swiper.SwipeLimits(max_actions=5),
        dry_run=True,
    )
    assert len(result.outcomes) == 5
    assert "budget" in result.stopped_because


def test_an_unreadable_page_stops_the_run_rather_than_spinning(
    tmp_path: Path, no_delay, monkeypatch
) -> None:
    """When the selectors stop matching, the loop has to give up and say so."""
    monkeypatch.setattr(swiper, "current_card_url", lambda _d: None)

    result = swiper.run(
        FakeSession(object()),
        FakeClassifier([0.9]),
        liked_dir=tmp_path / "liked",
        disliked_dir=tmp_path / "disliked",
        limits=swiper.SwipeLimits(max_actions=100, max_consecutive_failures=3),
        dry_run=True,
    )

    assert len(result.outcomes) == 3
    assert "could not be read" in result.stopped_because
    assert "selectors" in result.stopped_because  # says what to run next


def test_a_failed_download_does_not_stop_the_run(
    tmp_path: Path, no_delay, fake_page, monkeypatch
) -> None:
    from swipeml.automation.images import DownloadError

    calls = {"n": 0}

    def flaky(url, destination, **kwargs):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] % 2:
            raise DownloadError("429 Too Many Requests")
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(JPEG_BYTES)
        from swipeml.automation.images import Download

        return Download(target, len(JPEG_BYTES), "image/jpeg")

    monkeypatch.setattr(swiper.images, "fetch_image", flaky)

    result = swiper.run(
        FakeSession(object()),
        FakeClassifier([0.9]),
        liked_dir=tmp_path / "liked",
        disliked_dir=tmp_path / "disliked",
        limits=swiper.SwipeLimits(max_actions=6),
        dry_run=True,
    )
    assert any("429" in o.note for o in result.outcomes)
    assert any(o.label == "liked" for o in result.outcomes)


def test_a_missing_button_is_recorded_not_fatal(
    tmp_path: Path, no_delay, fake_download, monkeypatch
) -> None:
    monkeypatch.setattr(
        swiper, "current_card_url", lambda _d: "https://ex.com/card.jpg"
    )
    monkeypatch.setattr(locators.Locator, "find_or_none", lambda self, d: None)

    result = swiper.run(
        FakeSession(object()),
        FakeClassifier([0.9]),
        liked_dir=tmp_path / "liked",
        disliked_dir=tmp_path / "disliked",
        limits=swiper.SwipeLimits(max_actions=10, max_consecutive_failures=2),
        dry_run=False,
    )
    assert result.actions == 0
    assert any("could not find" in o.note for o in result.outcomes)


def test_a_click_that_raises_is_recorded_not_fatal(
    tmp_path: Path, no_delay, fake_download, monkeypatch
) -> None:
    monkeypatch.setattr(
        swiper, "current_card_url", lambda _d: "https://ex.com/card.jpg"
    )

    class Exploding:
        def click(self):
            raise RuntimeError("element click intercepted")

    monkeypatch.setattr(
        locators.Locator, "find_or_none",
        lambda self, d: (Exploding(), locators.Strategy("css", "stub", stability=10)),
    )

    result = swiper.run(
        FakeSession(object()),
        FakeClassifier([0.9]),
        liked_dir=tmp_path / "liked",
        disliked_dir=tmp_path / "disliked",
        limits=swiper.SwipeLimits(max_actions=3),
        dry_run=False,
    )
    assert result.actions == 0
    assert any("click failed" in o.note for o in result.outcomes)


def test_using_a_fragile_selector_is_noted(
    tmp_path: Path, no_delay, fake_download, monkeypatch
) -> None:
    """So drift shows up in the log before it becomes a failure."""
    monkeypatch.setattr(
        swiper, "current_card_url", lambda _d: "https://ex.com/card.jpg"
    )

    class Button:
        def click(self):
            pass

    monkeypatch.setattr(
        locators.Locator, "find_or_none",
        lambda self, d: (Button(), locators.Strategy("css", "div.Bdrs(8px)", stability=90)),
    )

    result = swiper.run(
        FakeSession(object()),
        FakeClassifier([0.9]),
        liked_dir=tmp_path / "liked",
        disliked_dir=tmp_path / "disliked",
        limits=swiper.SwipeLimits(max_actions=1),
        dry_run=False,
    )
    assert "fragile" in result.outcomes[0].note


def test_the_summary_counts_by_label(
    tmp_path: Path, no_delay, fake_page, fake_download
) -> None:
    result = swiper.run(
        FakeSession(object()),
        FakeClassifier([0.9, 0.1, 0.9, 0.1]),
        liked_dir=tmp_path / "liked",
        disliked_dir=tmp_path / "disliked",
        limits=swiper.SwipeLimits(max_actions=4),
        dry_run=True,
    )
    summary = result.summary()
    assert "liked" in summary
    assert "disliked" in summary
