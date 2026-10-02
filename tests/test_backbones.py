"""The backbone registry.

The whole reason this module exists is that each backbone wants its own
preprocessing and its own input size, and the original used one of each for all
five.
"""
from __future__ import annotations

import pytest

from swipeml import backbones


def test_the_five_backbones_are_registered() -> None:
    assert set(backbones.BACKBONES) == {
        "vgg16", "resnet50", "inceptionv3", "xception", "inceptionresnetv2",
    }


def test_each_backbone_declares_its_native_input_size() -> None:
    """The inception family is 299x299, not 224x224. The original resized
    everything to 224 and threw away a fifth of the resolution the weights
    expect."""
    assert backbones.get("inceptionv3").input_size == (299, 299)
    assert backbones.get("xception").input_size == (299, 299)
    assert backbones.get("inceptionresnetv2").input_size == (299, 299)
    assert backbones.get("vgg16").input_size == (224, 224)
    assert backbones.get("resnet50").input_size == (224, 224)


def test_preprocessing_style_differs_between_families() -> None:
    """This is the bug that cannot be seen. A tf-style network fed caffe-style
    inputs still trains and is quietly worse, with no error anywhere."""
    assert backbones.get("vgg16").preprocess_style == "caffe"
    assert backbones.get("resnet50").preprocess_style == "caffe"
    assert backbones.get("inceptionv3").preprocess_style == "tf"
    assert backbones.get("xception").preprocess_style == "tf"


def test_input_shape_appends_the_channels() -> None:
    assert backbones.get("inceptionv3").input_shape == (299, 299, 3)


def test_lookup_is_case_and_space_insensitive() -> None:
    assert backbones.get("  VGG16  ").name == "vgg16"


def test_an_unknown_backbone_lists_the_real_ones() -> None:
    with pytest.raises(KeyError, match="inceptionv3"):
        backbones.get("resnet9000")


def test_describe_lists_every_backbone() -> None:
    described = backbones.describe()
    for name in backbones.BACKBONES:
        assert name in described


def test_the_registry_does_not_import_tensorflow() -> None:
    """Looking up a backbone must not pull in TensorFlow, or the dataset and
    metric code stops being usable on its own."""
    import sys

    backbones.get("vgg16")
    backbones.describe()
    assert "tensorflow" not in sys.modules
