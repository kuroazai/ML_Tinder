"""Dataset inspection: what is there, and what is wrong with it."""
from __future__ import annotations

from pathlib import Path

import pytest
from conftest import HTML_ERROR_PAGE, JPEG_BYTES, PNG_BYTES, write_images

from swipeml import dataset


def test_counts_each_class(dataset_root: Path) -> None:
    report = dataset.inspect(dataset_root)
    assert report.total == 200
    assert {c.name: c.images for c in report.classes} == {"liked": 120, "disliked": 80}


def test_class_names_are_alphabetical(dataset_root: Path) -> None:
    """Keras assigns class indices by sorted folder name, so the order recorded
    in the manifest has to match or the labels are swapped at serving time."""
    assert dataset.inspect(dataset_root).class_names == ["disliked", "liked"]


def test_majority_baseline_is_the_number_to_beat(dataset_root: Path) -> None:
    """120 of 200, so always answering "liked" scores 60%. A model reporting
    65% has barely learned anything, and the original reported 85% with no
    baseline beside it at all."""
    assert dataset.inspect(dataset_root).majority_baseline == 0.6


def test_a_balanced_dataset_has_a_fifty_percent_baseline(balanced_root: Path) -> None:
    assert dataset.inspect(balanced_root).majority_baseline == 0.5


def test_imbalance_is_reported(dataset_root: Path) -> None:
    assert dataset.inspect(dataset_root).imbalance == 1.5


def test_class_weights_equalise_the_loss(dataset_root: Path) -> None:
    weights = dataset.inspect(dataset_root).class_weights()
    # disliked is index 0 and the smaller class, so it is weighted up.
    assert weights[0] > weights[1]
    assert weights[0] * 80 == pytest.approx(weights[1] * 120)


def test_an_html_error_page_saved_as_a_jpg_is_caught(tmp_path: Path) -> None:
    """Exactly what the original produced: `requests.get` with no
    `raise_for_status`, so a 429 page was written to a .jpg, added to the
    training set, and fed to the model."""
    root = tmp_path / "data"
    write_images(root / "liked", 60, "liked")
    write_images(root / "disliked", 60, "disliked")
    (root / "liked" / "rate_limited.jpg").write_bytes(HTML_ERROR_PAGE)

    report = dataset.inspect(root)
    liked = next(c for c in report.classes if c.name == "liked")
    assert liked.corrupt == ["rate_limited.jpg"]
    assert liked.images == 60  # not counted
    assert any("not images despite the extension" in p for p in report.problems())


def test_non_image_files_are_reported_not_ignored(tmp_path: Path) -> None:
    """Keras skips them silently, which makes a dataset smaller than it looks."""
    root = tmp_path / "data"
    write_images(root / "liked", 60, "liked")
    write_images(root / "disliked", 60, "disliked")
    (root / "liked" / "notes.txt").write_text("hello", encoding="utf-8")

    report = dataset.inspect(root)
    assert any("non-image file" in p for p in report.problems())


def test_png_is_accepted_too(tmp_path: Path) -> None:
    root = tmp_path / "data"
    write_images(root / "liked", 60, "liked", payload=PNG_BYTES)
    write_images(root / "disliked", 60, "disliked", payload=JPEG_BYTES)
    assert dataset.inspect(root).total == 120


def test_a_tiny_dataset_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "data"
    write_images(root / "liked", 5, "liked")
    write_images(root / "disliked", 5, "disliked")

    with pytest.raises(dataset.DatasetError, match="too few"):
        dataset.require_trainable(dataset.inspect(root))


def test_an_empty_class_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "data"
    write_images(root / "liked", 60, "liked")
    (root / "disliked").mkdir(parents=True)

    with pytest.raises(dataset.DatasetError, match="no images"):
        dataset.require_trainable(dataset.inspect(root))


def test_one_class_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "data"
    write_images(root / "liked", 60, "liked")
    with pytest.raises(dataset.DatasetError, match="two are needed"):
        dataset.require_trainable(dataset.inspect(root))


def test_imbalance_is_a_warning_not_a_blocker(dataset_root: Path) -> None:
    """It is workable with class weights, so it must not stop a run."""
    report = dataset.inspect(dataset_root)
    assert any("unbalanced" in p for p in report.problems())
    dataset.require_trainable(report)  # does not raise


def test_a_missing_folder_says_so(tmp_path: Path) -> None:
    with pytest.raises(dataset.DatasetError, match="no dataset folder"):
        dataset.inspect(tmp_path / "absent")


def test_class_weights_on_an_empty_class_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "data"
    write_images(root / "liked", 10, "liked")
    (root / "disliked").mkdir(parents=True)
    with pytest.raises(dataset.DatasetError, match="empty class"):
        dataset.inspect(root).class_weights()


def test_describe_includes_the_baseline(dataset_root: Path) -> None:
    described = dataset.inspect(dataset_root).describe()
    assert "majority baseline" in described
    assert "learned anything" in described
