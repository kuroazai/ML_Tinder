"""The command line, called directly."""
from __future__ import annotations

from pathlib import Path

import pytest
from conftest import write_images

from swipeml.cli import build_parser, main


def test_every_subcommand_has_a_handler() -> None:
    parser = build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    assert actions
    for action in actions:
        if not isinstance(action.choices, dict):
            continue
        for name, sub in action.choices.items():
            assert sub.get_default("handler") is not None, f"{name} has no handler"


def test_backbones_needs_nothing(capsys) -> None:
    assert main(["backbones"]) == 0
    printed = capsys.readouterr().out
    assert "inceptionv3" in printed
    assert "299x299" in printed


def test_config_reports_the_wiring(capsys, tmp_path: Path) -> None:
    assert main(["--env-file", str(tmp_path / "absent.env"), "config"]) == 0
    assert "data root" in capsys.readouterr().out


def test_selectors_reports_fragility(capsys) -> None:
    assert main(["selectors"]) == 0
    printed = capsys.readouterr().out
    assert "fragile" in printed
    assert "Bdrs(8px)" in printed


def test_data_reports_a_good_dataset(capsys, tmp_path: Path) -> None:
    root = tmp_path / "data"
    write_images(root / "liked", 100, "liked")
    write_images(root / "disliked", 100, "disliked")

    assert main(["--data-root", str(root), "data"]) == 0
    printed = capsys.readouterr().out
    assert "200 image(s)" in printed
    assert "majority baseline" in printed


def test_data_exits_non_zero_on_a_problem(capsys, tmp_path: Path) -> None:
    """So it can gate a script."""
    root = tmp_path / "data"
    write_images(root / "liked", 10, "liked")
    write_images(root / "disliked", 10, "disliked")

    assert main(["--data-root", str(root), "data"]) == 2
    assert "too few" in capsys.readouterr().out


def test_data_on_a_missing_folder(capsys, tmp_path: Path) -> None:
    assert main(["--data-root", str(tmp_path / "absent"), "data"]) == 1
    assert "no dataset folder" in capsys.readouterr().out


def test_train_checks_the_dataset_before_importing_tensorflow(
    capsys, tmp_path: Path
) -> None:
    """Importing TensorFlow takes seconds and can fail outright, and neither is
    a useful thing to do in order to discover the data folder is empty."""
    assert main(["--data-root", str(tmp_path / "absent"), "train"]) == 1
    printed = capsys.readouterr().out
    assert "no dataset folder" in printed
    assert "TensorFlow" not in printed


def test_train_rejects_an_unknown_backbone() -> None:
    with pytest.raises(SystemExit):
        main(["train", "--backbone", "resnet9000"])


def test_train_rejects_nonsense_settings(capsys, tmp_path: Path) -> None:
    root = tmp_path / "data"
    write_images(root / "liked", 100, "liked")
    write_images(root / "disliked", 100, "disliked")

    assert main(["--data-root", str(root), "train", "--epochs", "0"]) == 1
    assert "at least 1" in capsys.readouterr().out


def test_model_on_a_missing_file(capsys, tmp_path: Path) -> None:
    assert main(["--model", str(tmp_path / "absent.keras"), "model"]) == 1
    assert "no model at" in capsys.readouterr().out


def test_model_on_a_file_with_no_manifest(capsys, tmp_path: Path) -> None:
    model = tmp_path / "model.keras"
    model.write_bytes(b"not a model")
    assert main(["--model", str(model), "model"]) == 1
    assert "no manifest" in capsys.readouterr().out


def test_swipe_requires_a_url(capsys, tmp_path: Path) -> None:
    assert main(["--env-file", str(tmp_path / "absent.env"), "swipe"]) == 1
    assert "SWIPEML_URL" in capsys.readouterr().out


def test_swipe_rejects_a_backwards_delay_range(capsys, tmp_path: Path) -> None:
    code = main(["--env-file", str(tmp_path / "absent.env"), "swipe",
                 "--min-delay", "9", "--max-delay", "1"])
    assert code == 1
    assert "delay_range" in capsys.readouterr().out


def test_an_unknown_command_exits() -> None:
    with pytest.raises(SystemExit):
        main(["nonsense"])


def test_output_is_ascii(capsys) -> None:
    """A pound sign or an em-dash crashes a legacy Windows console code page."""
    main(["backbones"])
    capsys.readouterr().out.encode("ascii")
    main(["selectors"])
    capsys.readouterr().out.encode("ascii")
