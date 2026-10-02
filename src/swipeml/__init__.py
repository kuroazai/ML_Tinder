"""swipeml - learn what you like from your own labels, then apply it.

A binary image classifier built on a pretrained backbone, plus the browser
automation to gather the labels and act on the predictions.

    from swipeml import Classifier, TrainingConfig, train

    train("training_data", "models/model.keras", TrainingConfig(backbone="xception"))

    classifier = Classifier("models/model.keras")
    print(classifier.predict("photo.jpg"))

The dataset, metric and automation code work without TensorFlow installed;
Keras is imported inside the functions that need it.

The classifier learns the labels *you* gave it. It is not a judgement about
anyone, it is a model of one person's preferences, and it is only as good as
the few hundred examples behind it.
"""
from .backbones import BACKBONES, DEFAULT_BACKBONE, Backbone
from .config import ConfigError, Settings, load_env_file
from .dataset import DatasetError, DatasetReport, inspect, require_trainable
from .evaluate import (
    ConfusionMatrix,
    Report,
    ThresholdChoice,
    choose_threshold,
    confusion_at,
    report,
    roc_auc,
)
from .manifest import Manifest, ManifestError
from .predict import Classifier, PredictError, Prediction
from .training import Split, TrainingConfig, make_split, train

__version__ = "1.0.0"

__all__ = [
    "Backbone", "BACKBONES", "DEFAULT_BACKBONE",
    "Settings", "ConfigError", "load_env_file",
    "inspect", "DatasetReport", "DatasetError", "require_trainable",
    "Manifest", "ManifestError",
    "TrainingConfig", "train", "Split", "make_split",
    "Classifier", "Prediction", "PredictError",
    "ConfusionMatrix", "Report", "ThresholdChoice",
    "confusion_at", "roc_auc", "choose_threshold", "report",
    "__version__",
]
