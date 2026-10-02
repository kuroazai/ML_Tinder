"""The browser, on Selenium 4.

The original cannot run on any currently installed Selenium:

    self.profile = FirefoxProfile(firefox_profile)
    self.browser = webdriver.Firefox(self.profile)

A positional profile argument was removed in Selenium 4; profiles go through
`Options` now. And every method then reached for `self.browser.driver`, which
does not exist, because `self.browser` *is* the driver. So
`check_exists_by_xpath`, `get_page_source`, `refresh_page` and `close_browser`
all raised `AttributeError`, and `auto_tinder` referred to `src.driver` which
was never set at all. That is most of the automation path.

`find_element_by_xpath` was also removed in Selenium 4, in favour of
`find_element(By.XPATH, ...)`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SessionError(RuntimeError):
    """The browser could not be started, or is not logged in."""


@dataclass
class BrowserSession:
    """A Firefox or Chrome session, using an existing logged-in profile.

    An existing profile is how this avoids touching anyone's credentials: the
    browser is already signed in, and nothing here ever sees a password.
    """

    url: str
    profile_dir: str | Path | None = None
    browser: str = "firefox"
    headless: bool = False
    page_load_timeout: float = 30.0
    driver: Any = None

    def start(self) -> Any:
        if self.driver is not None:
            return self.driver
        self.driver = self._build_driver()
        self.driver.set_page_load_timeout(self.page_load_timeout)
        return self.driver

    def _build_driver(self) -> Any:
        from selenium import webdriver

        if self.browser == "firefox":
            options = webdriver.FirefoxOptions()
            if self.profile_dir:
                directory = Path(self.profile_dir)
                if not directory.is_dir():
                    raise SessionError(f"no Firefox profile directory at {directory}")
                # Selenium 4: the profile is an option, not a constructor
                # argument.
                options.add_argument("-profile")
                options.add_argument(str(directory))
            if self.headless:
                options.add_argument("-headless")
            try:
                return webdriver.Firefox(options=options)
            except Exception as exc:  # noqa: BLE001 - selenium raises several
                raise SessionError(
                    f"could not start Firefox: {exc}. Is geckodriver installed "
                    "and on PATH, and is the profile closed in any other window?"
                ) from exc

        if self.browser == "chrome":
            options = webdriver.ChromeOptions()
            if self.profile_dir:
                options.add_argument(f"--user-data-dir={self.profile_dir}")
            if self.headless:
                options.add_argument("--headless=new")
            try:
                return webdriver.Chrome(options=options)
            except Exception as exc:  # noqa: BLE001
                raise SessionError(f"could not start Chrome: {exc}") from exc

        raise SessionError(
            f"unknown browser {self.browser!r}; use 'firefox' or 'chrome'"
        )

    # -- the driver, not driver.driver -------------------------------------
    def open(self, settle: float = 2.0) -> None:
        driver = self.start()
        driver.get(self.url)
        time.sleep(settle)

    @property
    def page_source(self) -> str:
        return self.start().page_source

    def refresh(self) -> None:
        self.start().refresh()

    def close(self) -> None:
        """Quit, not close.

        `driver.close()` shuts one window and leaves the process running, which
        over a few interrupted runs leaves a handful of orphaned geckodriver
        processes holding the profile open. The original called `close`.
        """
        if self.driver is not None:
            try:
                self.driver.quit()
            finally:
                self.driver = None

    def __enter__(self) -> BrowserSession:
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
