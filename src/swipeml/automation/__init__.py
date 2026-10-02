"""Browser automation: the session, resilient locators, and the swipe loop."""
from .images import Download, DownloadError, fetch_image, next_available
from .locators import (
    ALL_LOCATORS,
    CARD_IMAGE,
    LIKE_BUTTON,
    PASS_BUTTON,
    PROFILE_CARD,
    Locator,
    LocatorError,
    Strategy,
    extract_background_url,
    fragility_report,
    is_atomic_class,
)
from .session import BrowserSession, SessionError
from .swiper import SwipeLimits, SwipeOutcome, SwipeSession, collect, current_card_url, run

__all__ = [
    "BrowserSession", "SessionError",
    "Locator", "Strategy", "LocatorError", "is_atomic_class",
    "extract_background_url", "fragility_report",
    "LIKE_BUTTON", "PASS_BUTTON", "PROFILE_CARD", "CARD_IMAGE", "ALL_LOCATORS",
    "fetch_image", "next_available", "Download", "DownloadError",
    "SwipeLimits", "SwipeOutcome", "SwipeSession", "run", "collect",
    "current_card_url",
]
