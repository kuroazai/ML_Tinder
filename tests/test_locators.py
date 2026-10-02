"""Finding things on a page, and failing usefully when they move.

This is where the original broke. These tests stub `_find_one`, so no browser
and no Selenium install is needed to check the strategy ordering, the fallback
behaviour and the error messages.
"""
from __future__ import annotations

import pytest
from conftest import FakeDriver, FakeElement

from swipeml.automation import locators as L


@pytest.fixture(autouse=True)
def stub_find(monkeypatch):
    """Replace the one function that touches Selenium."""
    def fake_find_one(driver, strategy):
        for element in driver.find_elements(strategy.how, strategy.what):
            try:
                if element.is_displayed():
                    return element
            except Exception:  # noqa: BLE001 - a stale element is simply not it
                continue
        return None

    monkeypatch.setattr(L, "_find_one", fake_find_one)


# -- recognising a selector that will rot ----------------------------------

@pytest.mark.parametrize(
    ("name", "atomic"),
    [
        ("Bdrs(8px)", True),
        ("Bgz(cv)", True),
        ("Bgp(c)", True),
        ("Mb(20px)", True),
        ("StretchedBox", False),
        ("profile-card", False),
        ("btn", False),
        ("", False),
    ],
)
def test_atomic_class_names_are_recognised(name: str, atomic: bool) -> None:
    """These are Atomic CSS: a property abbreviation and a value. They encode
    styling, so a selector built on them breaks when the design changes. The
    original matched on four of them at once."""
    assert L.is_atomic_class(name) is atomic


def test_an_escaped_atomic_class_in_a_css_selector_is_still_recognised() -> None:
    """A CSS selector has to escape the brackets, so the check has to see past
    the backslashes or it never fires on the selectors it exists to catch."""
    found = L.atomic_classes(r"div.Bdrs\(8px\).Bgz\(cv\)")
    assert "Bdrs(8px)" in found
    assert "Bgz(cv)" in found


def test_an_absolute_xpath_is_fragile() -> None:
    """One extra wrapper element anywhere above it and the path is wrong. The
    original used six of these."""
    strategy = L.Strategy("xpath", "/html/body/div[1]/div/div[1]/main/button")
    assert strategy.fragile
    assert "wrapper" in strategy.why_fragile()


def test_a_semantic_selector_is_not_fragile() -> None:
    assert not L.Strategy("css", '[data-testid="gamepadLike"]').fragile
    assert not L.Strategy("css", 'button[aria-label*="Like" i]').fragile


def test_the_shipped_locators_know_which_of_them_will_rot() -> None:
    report = L.fragility_report()
    assert "card image" in report
    assert "Bdrs(8px)" in report


# -- finding, and the order it is tried in ---------------------------------

def test_the_most_stable_strategy_is_tried_first() -> None:
    driver = FakeDriver({
        'button[aria-label*="Like" i]': [FakeElement()],
        '[data-testid="gamepadLike"]': [FakeElement()],
    })
    _element, strategy = L.LIKE_BUTTON.find(driver)
    assert strategy.stability == 10
    assert driver.asked == ['button[aria-label*="Like" i]']


def test_it_falls_back_when_the_first_choice_is_gone() -> None:
    driver = FakeDriver({'[data-testid="gamepadLike"]': [FakeElement()]})
    _element, strategy = L.LIKE_BUTTON.find(driver)

    assert strategy.what == '[data-testid="gamepadLike"]'
    # The more stable one was tried first and did not match.
    assert driver.asked[0] == 'button[aria-label*="Like" i]'


def test_a_hidden_element_does_not_count() -> None:
    """An offscreen or hidden button is present in the DOM and not clickable."""
    driver = FakeDriver({
        'button[aria-label*="Like" i]': [FakeElement(displayed=False)],
        '[data-testid="gamepadLike"]': [FakeElement()],
    })
    _element, strategy = L.LIKE_BUTTON.find(driver)
    assert strategy.what == '[data-testid="gamepadLike"]'


def test_a_stale_element_does_not_count() -> None:
    """Cards are replaced constantly, so a reference can go stale between
    finding it and asking about it."""
    driver = FakeDriver({
        'button[aria-label*="Like" i]': [FakeElement(stale=True)],
        '[data-testid="gamepadLike"]': [FakeElement()],
    })
    _element, strategy = L.LIKE_BUTTON.find(driver)
    assert strategy.what == '[data-testid="gamepadLike"]'


def test_the_first_displayed_element_of_several_is_taken() -> None:
    wanted = FakeElement()
    driver = FakeDriver({
        'button[aria-label*="Like" i]': [FakeElement(displayed=False), wanted],
    })
    element, _strategy = L.LIKE_BUTTON.find(driver)
    assert element is wanted


def test_failing_to_find_lists_everything_tried() -> None:
    """A run that says "the like button moved, here is what I tried" is fixable
    in minutes. `NoSuchElementException` with an absolute XPath is an
    afternoon."""
    with pytest.raises(L.LocatorError) as caught:
        L.LIKE_BUTTON.find(FakeDriver({}))

    message = str(caught.value)
    assert "like button" in message
    assert "locators.py" in message  # says where to fix it
    for strategy in L.LIKE_BUTTON.strategies:
        assert strategy.what in message


def test_find_or_none_does_not_raise() -> None:
    assert L.LIKE_BUTTON.find_or_none(FakeDriver({})) is None


def test_every_shipped_locator_has_more_than_one_strategy() -> None:
    """One selector is how the original ended up permanently broken."""
    for locator in L.ALL_LOCATORS:
        assert len(locator.strategies) >= 2, locator.name


def test_an_unknown_strategy_type_is_refused(monkeypatch) -> None:
    monkeypatch.undo()
    with pytest.raises(ValueError, match="unknown strategy"):
        L._find_one(FakeDriver({}), L.Strategy("telepathy", "button"))


# -- the url out of a style attribute --------------------------------------

@pytest.mark.parametrize(
    ("style", "expected"),
    [
        ('background-image: url("https://ex.com/a.jpg")', "https://ex.com/a.jpg"),
        ("background-image: url('https://ex.com/b.jpg')", "https://ex.com/b.jpg"),
        ("background-image: url(https://ex.com/c.jpg)", "https://ex.com/c.jpg"),
        # A signed URL can contain brackets, and matching to the closing quote
        # rather than the first ')' is what gets this right.
        ('background-image: url("https://ex.com/d.jpg?sig=a)b")',
         "https://ex.com/d.jpg?sig=a)b"),
        ('background-image: url("  https://ex.com/e.jpg  ")', "https://ex.com/e.jpg"),
        ("background-color: red", None),
        ('background-image: url("")', None),
        ("", None),
    ],
)
def test_background_url_extraction(style: str, expected) -> None:
    assert L.extract_background_url(style) == expected


def test_the_original_string_replacement_left_the_quotes_attached() -> None:
    """`.replace('url(', '').replace(')', '')` on a quoted value leaves the
    quotes, and a quoted URL passed to requests returns a 404 rather than an
    error anyone can read."""
    style = 'url("https://ex.com/a.jpg")'
    naive = style.replace("url(", "").replace(")", "")
    assert naive == '"https://ex.com/a.jpg"'
    assert L.extract_background_url(style) == "https://ex.com/a.jpg"
