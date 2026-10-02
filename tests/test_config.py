"""Settings, and keeping a browser profile path out of the repository."""
from __future__ import annotations

from pathlib import Path

import pytest

from swipeml.config import ConfigError, Settings, load_env_file


def test_defaults_need_no_configuration() -> None:
    settings = Settings()
    assert settings.liked_dir == Path("training_data/liked")
    assert settings.disliked_dir == Path("training_data/disliked")
    assert not settings.automation_configured


def test_the_automation_commands_need_a_url() -> None:
    with pytest.raises(ConfigError, match="SWIPEML_URL"):
        Settings().require_url()


def test_env_file_is_read(tmp_path: Path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n\nSWIPEML_DATA_ROOT=/data\n"
        'SWIPEML_BROWSER="chrome"\nNOT_A_PAIR\n',
        encoding="utf-8",
    )
    for key in ("SWIPEML_DATA_ROOT", "SWIPEML_BROWSER", "SWIPEML_URL",
                "SWIPEML_BROWSER_PROFILE", "SWIPEML_MODEL", "SWIPEML_HEADLESS"):
        monkeypatch.delenv(key, raising=False)

    loaded = load_env_file(str(env))
    assert loaded["SWIPEML_BROWSER"] == "chrome"  # quotes stripped
    assert "NOT_A_PAIR" not in loaded
    assert Settings.from_env(env_file=None).browser == "chrome"


def test_the_real_environment_beats_an_env_file(tmp_path: Path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("SWIPEML_BROWSER=chrome\n", encoding="utf-8")
    monkeypatch.setenv("SWIPEML_BROWSER", "firefox")

    load_env_file(str(env))
    assert Settings.from_env(env_file=None).browser == "firefox"


def test_a_missing_env_file_is_fine(tmp_path: Path) -> None:
    assert load_env_file(str(tmp_path / "absent.env")) == {}


def test_describe_does_not_print_the_browser_profile_path() -> None:
    """A browser profile path identifies a person and a machine. The equivalent
    pattern in another of these repositories published a Windows username and a
    Firefox profile ID."""
    settings = Settings(browser_profile=Path("/home/someone/.mozilla/abc123.default"))
    described = settings.describe()

    assert "someone" not in described
    assert "abc123" not in described
    assert "set" in described


def test_describe_flags_missing_paths(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path / "absent",
                        model_path=tmp_path / "absent.keras")
    assert described_count(settings.describe(), "MISSING") == 2


def described_count(text: str, needle: str) -> int:
    return text.count(needle)


def test_headless_is_read_from_the_environment(monkeypatch) -> None:
    for value, expected in [("1", True), ("true", True), ("yes", True),
                            ("0", False), ("", False), ("no", False)]:
        monkeypatch.setenv("SWIPEML_HEADLESS", value)
        assert Settings.from_env(env_file=None).headless is expected
