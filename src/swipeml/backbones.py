"""The pretrained backbones, each with the preprocessing it was trained with.

This is the single most important module here, because getting it wrong is
invisible. Every ImageNet backbone expects its inputs scaled a particular way:

    vgg16               caffe style, BGR channel order, per-channel mean
                        subtracted, values roughly -124..151
    inception_v3        tf style, scaled to -1..1
    xception            tf style, scaled to -1..1
    inception_resnet_v2 tf style, scaled to -1..1
    resnet50            caffe style, like vgg16

Feed a tf-style network caffe-style inputs and it still trains, still reports a
plausible accuracy, and is quietly worse than it should be. Nothing errors.

The original selected the right `preprocess_input` per architecture, returned
it, and then never used it. Training rescaled everything by 1/255 instead, which
is correct for none of the five.

So the preprocessing is part of the backbone's identity here, and
`manifest.py` persists which one was used so that serving cannot diverge from
training.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

#: How a backbone wants its pixels. "tf" is -1..1, "caffe" is BGR with the
#: ImageNet channel means subtracted, "scale" is plain 0..1.
PreprocessStyle = str


@dataclass(frozen=True)
class Backbone:
    """One pretrained feature extractor."""

    name: str
    #: Dotted path to the Keras application, resolved lazily so importing this
    #: module does not pull in TensorFlow.
    module: str
    constructor: str
    input_size: tuple[int, int]
    preprocess_style: PreprocessStyle
    #: Roughly how many parameters, for choosing one on a given machine.
    approximate_params_m: float

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (*self.input_size, 3)

    def load(self, *, weights: str = "imagenet") -> Any:
        """Build the backbone without its classifier head."""
        import importlib

        module = importlib.import_module(self.module)
        constructor = getattr(module, self.constructor)
        return constructor(
            weights=weights, include_top=False, input_shape=self.input_shape
        )

    def preprocess_function(self) -> Callable[[Any], Any]:
        """Keras' own `preprocess_input` for this backbone.

        Taken from the backbone's own module rather than reimplemented, because
        the constants are part of how the weights were trained and a typo in a
        channel mean is not something you would notice.
        """
        import importlib

        module = importlib.import_module(self.module)
        return module.preprocess_input


BACKBONES: dict[str, Backbone] = {
    "vgg16": Backbone(
        name="vgg16",
        module="tensorflow.keras.applications.vgg16",
        constructor="VGG16",
        input_size=(224, 224),
        preprocess_style="caffe",
        approximate_params_m=14.7,
    ),
    "resnet50": Backbone(
        name="resnet50",
        module="tensorflow.keras.applications.resnet50",
        constructor="ResNet50",
        input_size=(224, 224),
        preprocess_style="caffe",
        approximate_params_m=23.6,
    ),
    "inceptionv3": Backbone(
        name="inceptionv3",
        module="tensorflow.keras.applications.inception_v3",
        constructor="InceptionV3",
        input_size=(299, 299),
        preprocess_style="tf",
        approximate_params_m=21.8,
    ),
    "xception": Backbone(
        name="xception",
        module="tensorflow.keras.applications.xception",
        constructor="Xception",
        input_size=(299, 299),
        preprocess_style="tf",
        approximate_params_m=20.9,
    ),
    "inceptionresnetv2": Backbone(
        name="inceptionresnetv2",
        module="tensorflow.keras.applications.inception_resnet_v2",
        constructor="InceptionResNetV2",
        input_size=(299, 299),
        preprocess_style="tf",
        approximate_params_m=54.3,
    ),
}

#: The default. Smallest of the five and the quickest to get a baseline from.
DEFAULT_BACKBONE = "vgg16"


def get(name: str) -> Backbone:
    """Look up a backbone, or say which exist."""
    try:
        return BACKBONES[name.strip().lower()]
    except KeyError as exc:
        raise KeyError(
            f"unknown backbone {name!r}. Available: " + ", ".join(sorted(BACKBONES))
        ) from exc


def describe() -> str:
    """A table, for choosing one."""
    lines = [f"{'backbone':<20}{'input':>12}{'preprocess':>12}{'params':>10}"]
    for backbone in sorted(BACKBONES.values(), key=lambda b: b.approximate_params_m):
        size = f"{backbone.input_size[0]}x{backbone.input_size[1]}"
        lines.append(
            f"{backbone.name:<20}{size:>12}{backbone.preprocess_style:>12}"
            f"{backbone.approximate_params_m:>9.1f}M"
        )
    return "\n".join(lines)


#: The note worth repeating wherever a size is chosen. InceptionV3 at 224x224
#: works and throws away a fifth of the resolution the weights expect.
SIZE_NOTE = (
    "Each backbone has a native input size. Using a different one still trains, "
    "and wastes part of what the pretrained weights learned."
)
