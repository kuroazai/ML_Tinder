"""The manifest, which is what stops train/serve preprocessing skew."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from swipeml.manifest import Manifest, ManifestError


def make(**overrides) -> Manifest:
    base = dict(
        backbone="inceptionv3",
        input_size=(299, 299),
        class_names=["disliked", "liked"],
        threshold=0.62,
    )
    base.update(overrides)
    return Manifest(**base)


def test_round_trips_through_a_file(tmp_path: Path) -> None:
    model = tmp_path / "model.keras"
    original = make()
    original.save(model)

    loaded = Manifest.load(model)
    assert loaded.backbone == "inceptionv3"
    assert loaded.input_size == (299, 299)
    assert loaded.class_names == ["disliked", "liked"]
    assert loaded.threshold == pytest.approx(0.62)


def test_a_model_without_a_manifest_is_refused(tmp_path: Path) -> None:
    """Serving with a guessed preprocessing is invisible when the guess is
    wrong: the model returns confident nonsense and looks merely mediocre."""
    with pytest.raises(ManifestError, match="no manifest"):
        Manifest.load(tmp_path / "model.keras")


def test_the_positive_class_is_the_last_one() -> None:
    """Keras sorts class folder names, so `liked` is index 1 with these two.
    Recording the order rather than relying on remembering it is the point."""
    assert make().positive_class == "liked"
    assert make(class_names=["a", "b", "c"]).positive_class == "c"


def test_the_manifest_carries_the_input_size_training_used() -> None:
    """Which is how `predict` cannot resize differently from `train`. The
    original resized to 224 in training and 320 in prediction, then reshaped to
    224, which raised ValueError on every call."""
    assert make(backbone="xception", input_size=(299, 299)).input_size == (299, 299)


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"backbone": "nonsense"}, "unknown backbone"),
        ({"threshold": 0.0}, "between 0 and 1"),
        ({"threshold": 1.0}, "between 0 and 1"),
        ({"threshold": -0.5}, "between 0 and 1"),
        ({"class_names": ["only"]}, "at least two classes"),
        ({"input_size": (299, 299, 3)}, "height, width"),
        ({"input_size": (299,)}, "height, width"),
    ],
)
def test_nonsense_is_refused(overrides: dict, fragment: str) -> None:
    with pytest.raises((ManifestError, KeyError), match=fragment):
        make(**overrides)


def test_invalid_json_says_so(tmp_path: Path) -> None:
    model = tmp_path / "model.keras"
    Manifest.path_for(model).write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestError, match="not valid JSON"):
        Manifest.load(model)


def test_a_newer_schema_version_is_refused(tmp_path: Path) -> None:
    """Rather than loading it and ignoring fields it does not understand, which
    could silently drop the threshold."""
    model = tmp_path / "model.keras"
    payload = {"backbone": "vgg16", "input_size": [224, 224],
               "class_names": ["a", "b"], "schema_version": 99}
    Manifest.path_for(model).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError, match="schema version 99"):
        Manifest.load(model)


def test_unknown_fields_from_a_newer_writer_are_dropped(tmp_path: Path) -> None:
    """Forwards-compatible within a schema version: a field this build does not
    know is better dropped than refused."""
    model = tmp_path / "model.keras"
    payload = {"backbone": "vgg16", "input_size": [224, 224],
               "class_names": ["a", "b"], "threshold": 0.4,
               "something_new": "ignored"}
    Manifest.path_for(model).write_text(json.dumps(payload), encoding="utf-8")

    loaded = Manifest.load(model)
    assert loaded.threshold == pytest.approx(0.4)


def test_missing_required_fields_say_which(tmp_path: Path) -> None:
    model = tmp_path / "model.keras"
    Manifest.path_for(model).write_text(json.dumps({"backbone": "vgg16"}),
                                        encoding="utf-8")
    with pytest.raises(ManifestError, match="missing required fields"):
        Manifest.load(model)


def test_the_manifest_sits_beside_the_model(tmp_path: Path) -> None:
    model = tmp_path / "nested" / "model.keras"
    model.parent.mkdir(parents=True)
    written = make().save(model)
    assert written.parent == model.parent
    assert model.name in written.name


def test_describe_names_the_preprocessing(tmp_path: Path) -> None:
    described = make(backbone="vgg16", input_size=(224, 224)).describe()
    assert "caffe" in described
    assert "224x224" in described
