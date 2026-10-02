"""Settings, from the environment.

The original put a Firefox profile path in a tracked `config.py`. It happened to
be committed empty, but the same pattern in another of these repositories
shipped a real Windows username and Firefox profile ID to a public repository.
A browser profile path identifies a person and a machine, so it does not belong
in version control.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    """The configuration is incomplete."""


def load_env_file(path: str = ".env", *, override: bool = False) -> dict[str, str]:
    """Read KEY=value lines into the environment.

    Existing environment variables win unless `override`, so a real setup is
    never clobbered by a stray file.
    """
    loaded: dict[str, str] = {}
    source = Path(path)
    if not source.is_file():
        return loaded

    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


@dataclass
class Settings:
    data_root: Path = Path("training_data")
    model_path: Path = Path("models/model.keras")
    browser: str = "firefox"
    browser_profile: Path | None = None
    target_url: str = ""
    headless: bool = False

    @property
    def liked_dir(self) -> Path:
        return self.data_root / "liked"

    @property
    def disliked_dir(self) -> Path:
        return self.data_root / "disliked"

    @property
    def staging_dir(self) -> Path:
        """Where collected, unlabelled images go until you sort them."""
        return self.data_root / "unsorted"

    @property
    def automation_configured(self) -> bool:
        return bool(self.target_url)

    @classmethod
    def from_env(cls, env_file: str | None = ".env") -> Settings:
        if env_file:
            load_env_file(env_file)
        profile = os.environ.get("SWIPEML_BROWSER_PROFILE", "").strip()
        return cls(
            data_root=Path(os.environ.get("SWIPEML_DATA_ROOT", "training_data")),
            model_path=Path(os.environ.get("SWIPEML_MODEL", "models/model.keras")),
            browser=os.environ.get("SWIPEML_BROWSER", "firefox").strip().lower(),
            browser_profile=Path(profile) if profile else None,
            target_url=os.environ.get("SWIPEML_URL", "").strip(),
            headless=os.environ.get("SWIPEML_HEADLESS", "").strip().lower()
            in {"1", "true", "yes"},
        )

    def require_url(self) -> str:
        if not self.target_url:
            raise ConfigError(
                "no SWIPEML_URL configured. The automation commands need the site "
                "to drive; set it in .env. See .env.example."
            )
        return self.target_url

    def describe(self) -> str:
        """What is configured. A browser profile path identifies a person, so
        only whether it is set and valid is printed, never the path itself."""
        def state(path: Path | None, *, directory: bool) -> str:
            if path is None:
                return "not set"
            exists = path.is_dir() if directory else path.exists()
            return f"{'found' if exists else 'MISSING'} ({path})"

        return "\n".join([
            f"data root        {state(self.data_root, directory=True)}",
            f"model            {state(self.model_path, directory=False)}",
            f"browser          {self.browser}"
            + (" (headless)" if self.headless else ""),
            f"browser profile  {'set' if self.browser_profile else 'not set'}"
            + ("" if not self.browser_profile
               else f", {'found' if self.browser_profile.is_dir() else 'MISSING'}"),
            f"automation       {'configured' if self.automation_configured else 'not configured'}",
        ])
