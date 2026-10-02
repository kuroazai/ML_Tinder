"""The loop: look at a card, decide, act, record.

Three things the original's loop got wrong, all of which stop it working rather
than merely slowing it down:

**It could not terminate.** `--validation` defaulted to 1, and the validation
branch did `continue` without decrementing the counter, so the default
invocation downloaded images forever and never swiped. The dislike branch also
`continue`d without decrementing when its button was not found.

**Fixed `time.sleep(3)` between every action**, and a `while True` polling
`keyboard.is_pressed` with no sleep at all, which pegs a CPU core.

**No session limits.** Nothing capped how many actions a run would take.

Here the loop has an explicit budget, a dry-run mode that decides without
clicking anything, and it stops and says why rather than spinning.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import images, locators


@dataclass
class SwipeLimits:
    """What a single run is allowed to do.

    Defaults are deliberately modest. This drives a real account, and the
    sensible default for anything unattended is "not very much".
    """

    max_actions: int = 50
    #: Seconds between actions, sampled from this range rather than fixed, so a
    #: run does not act on a metronome.
    delay_range: tuple[float, float] = (2.0, 5.0)
    #: Stop if this many cards in a row cannot be read.
    max_consecutive_failures: int = 5
    #: Overall wall-clock cap.
    max_minutes: float = 30.0

    def __post_init__(self) -> None:
        low, high = self.delay_range
        if low < 0 or high < low:
            raise ValueError(
                "delay_range must be (low, high) with 0 <= low <= high, got "
                f"{self.delay_range}"
            )
        if self.max_actions < 1:
            raise ValueError("max_actions must be at least 1")

    def delay(self, generator: random.Random | None = None) -> float:
        rng = generator or random
        return rng.uniform(*self.delay_range)


@dataclass
class SwipeOutcome:
    """One card."""

    index: int
    label: str
    score: float | None
    acted: bool
    note: str = ""

    def __str__(self) -> str:
        score = f"{self.score:.3f}" if self.score is not None else "n/a"
        action = "acted" if self.acted else "no action"
        return f"#{self.index} {self.label} (score {score}) - {action}" + (
            f": {self.note}" if self.note else ""
        )


@dataclass
class SwipeSession:
    """A run's results."""

    outcomes: list[SwipeOutcome] = field(default_factory=list)
    stopped_because: str = ""

    @property
    def actions(self) -> int:
        return sum(1 for o in self.outcomes if o.acted)

    def summary(self) -> str:
        lines = [
            f"{len(self.outcomes)} card(s) seen, {self.actions} action(s) taken"
        ]
        if self.stopped_because:
            lines.append(f"stopped: {self.stopped_because}")
        counts: dict[str, int] = {}
        for outcome in self.outcomes:
            counts[outcome.label] = counts.get(outcome.label, 0) + 1
        for label, count in sorted(counts.items()):
            lines.append(f"  {label:<20}{count:>5}")
        return "\n".join(lines)


def current_card_url(driver) -> str | None:
    """The image URL of the card on top of the stack.

    The original took the middle element of everything matching a styling class,
    `divs[int(len(divs) / 2)]`, which is a guess that happened to work on one
    version of the layout. Here the card is located first and its image read
    from inside it, so the relationship is structural rather than positional.
    """
    found = locators.CARD_IMAGE.find_or_none(driver)
    if found is None:
        return None
    element, _strategy = found
    style = element.get_attribute("style") or ""
    return locators.extract_background_url(style)


def run(
    session,
    classifier,
    *,
    liked_dir: str | Path,
    disliked_dir: str | Path,
    limits: SwipeLimits | None = None,
    dry_run: bool = True,
    seed: int | None = None,
) -> SwipeSession:
    """Classify each card and, unless this is a dry run, act on it.

    `dry_run` defaults to True. Deciding without clicking is the useful mode for
    checking a model, and it is also the safe default for a tool that drives a
    real account.
    """
    limits = limits or SwipeLimits()
    generator = random.Random(seed)
    result = SwipeSession()
    driver = session.start()

    liked_path, disliked_path = Path(liked_dir), Path(disliked_dir)
    started = time.monotonic()
    failures = 0

    while True:
        if len(result.outcomes) >= limits.max_actions:
            result.stopped_because = f"reached the {limits.max_actions}-card budget"
            break
        elapsed_minutes = (time.monotonic() - started) / 60
        if elapsed_minutes >= limits.max_minutes:
            result.stopped_because = f"reached the {limits.max_minutes:.0f} minute cap"
            break
        if failures >= limits.max_consecutive_failures:
            result.stopped_because = (
                f"{failures} cards in a row could not be read. The page has "
                "probably changed; run 'swipeml selectors' to see which "
                "selectors are expected to rot."
            )
            break

        index = len(result.outcomes) + 1
        url = current_card_url(driver)
        if not url:
            failures += 1
            result.outcomes.append(
                SwipeOutcome(index, "unreadable", None, False, "no card image found")
            )
            time.sleep(limits.delay(generator))
            continue
        failures = 0

        # Downloaded to a scratch name first; it is only filed under a label
        # once the classification is known.
        scratch = liked_path.parent / ".current.jpg"
        try:
            images.fetch_image(url, scratch)
        except images.DownloadError as exc:
            result.outcomes.append(
                SwipeOutcome(index, "unreadable", None, False, str(exc))
            )
            time.sleep(limits.delay(generator))
            continue

        prediction = classifier.predict(scratch)
        positive = prediction.positive
        destination = liked_path if positive else disliked_path
        prefix = "auto_liked" if positive else "auto_disliked"
        target = images.next_available(destination, prefix)
        scratch.replace(target)

        if dry_run:
            result.outcomes.append(
                SwipeOutcome(index, prediction.label, prediction.score, False,
                             "dry run, no click")
            )
            time.sleep(limits.delay(generator))
            continue

        locator = locators.LIKE_BUTTON if positive else locators.PASS_BUTTON
        found = locator.find_or_none(driver)
        if found is None:
            result.outcomes.append(
                SwipeOutcome(index, prediction.label, prediction.score, False,
                             f"could not find the {locator.name}")
            )
            failures += 1
            time.sleep(limits.delay(generator))
            continue

        element, strategy = found
        try:
            element.click()
        except Exception as exc:  # noqa: BLE001 - several selenium exceptions
            result.outcomes.append(
                SwipeOutcome(index, prediction.label, prediction.score, False,
                             f"click failed: {type(exc).__name__}")
            )
            time.sleep(limits.delay(generator))
            continue

        note = "" if strategy.stability <= 30 else f"via a fragile selector ({strategy.what})"
        result.outcomes.append(
            SwipeOutcome(index, prediction.label, prediction.score, True, note)
        )
        time.sleep(limits.delay(generator))

    return result


def collect(
    session,
    *,
    staging_dir: str | Path,
    limits: SwipeLimits | None = None,
) -> SwipeSession:
    """Save the cards you are shown, for you to label afterwards.

    One folder, not two. Nothing here can read which way you swiped: the
    original polled `keyboard.is_pressed` in a `while True` with no sleep, which
    burns a CPU core, needs root on Linux, and still only knows that *a* key was
    pressed somewhere, not that the page acted on it.

    So this is honest about what it does. It collects the images and you sort
    them into `liked/` and `disliked/`, which takes a couple of minutes and is
    the one step that actually has to be yours, because the labels are the whole
    point.
    """
    limits = limits or SwipeLimits()
    result = SwipeSession()
    driver = session.start()
    staging = Path(staging_dir)

    seen: set[str] = set()
    started = time.monotonic()

    while len(result.outcomes) < limits.max_actions:
        if (time.monotonic() - started) / 60 >= limits.max_minutes:
            result.stopped_because = f"reached the {limits.max_minutes:.0f} minute cap"
            break

        url = current_card_url(driver)
        if not url or url in seen:
            time.sleep(0.5)
            continue
        seen.add(url)

        index = len(result.outcomes) + 1
        target = images.next_available(staging, "card")
        try:
            images.fetch_image(url, target)
            result.outcomes.append(SwipeOutcome(index, "collected", None, True))
        except images.DownloadError as exc:
            result.outcomes.append(
                SwipeOutcome(index, "unreadable", None, False, str(exc))
            )
        time.sleep(0.5)

    if not result.stopped_because:
        result.stopped_because = f"collected {len(result.outcomes)} card(s)"
    return result
