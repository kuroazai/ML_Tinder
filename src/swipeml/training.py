"""Training, with the mistakes the original made written down.

Keras is imported inside the functions, so `import swipeml` works without
TensorFlow installed and the dataset, metric and automation code can be used and
tested on their own.

Four things fixed here, in rough order of how much they cost:

**The preprocessing was never applied.** `build_model` chose the right
`preprocess_input` for each architecture, returned it, and the caller ignored
it in favour of `rescale=1./255`. That is correct for none of the five
backbones, and nothing errors when it is wrong.

**The best weights were thrown away.** `EarlyStopping(patience=3)` without
`restore_best_weights=True` leaves the model holding the weights from three
epochs past the best. The run then did `model.save('tinder_model_main.h5')`,
overwriting the `ModelCheckpoint` file that did hold the best weights, and the
predictor loaded that overwritten file. So the reported accuracy belonged to a
model nobody ever used.

**No augmentation.** On a few hundred personal photographs, a frozen backbone
with a fresh dense head overfits in a handful of epochs.

**No class weights.** Whatever you happened to swipe is not a balanced dataset.

The validation split is also made by file, not by image generator order, so the
same images stay in validation between runs. `ImageDataGenerator`'s
`validation_split` divides each class by directory order, which means the split
changes the moment a file is added and the "validation" accuracy of two runs is
not comparable.
"""
from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import backbones, dataset
from .manifest import Manifest

#: Held back from training. Twenty percent of a few hundred images is small
#: enough already.
DEFAULT_VALIDATION_FRACTION = 0.2
#: Fixed so a split is reproducible and two runs are comparable.
DEFAULT_SEED = 1337


@dataclass
class TrainingConfig:
    backbone: str = backbones.DEFAULT_BACKBONE
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    #: Unfreezing the top of the backbone after the head has settled. Worth
    #: several points, and pointless before the head has converged, which is why
    #: it is a second phase rather than a flag on the first.
    fine_tune_epochs: int = 0
    fine_tune_layers: int = 30
    fine_tune_learning_rate: float = 1e-5
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    seed: int = DEFAULT_SEED
    use_class_weights: bool = True
    augment: bool = True
    #: Objective for picking the decision threshold from validation scores.
    threshold_objective: str = "f1"
    minimum_precision: float = 0.0

    def __post_init__(self) -> None:
        backbones.get(self.backbone)
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if not 0 < self.validation_fraction < 1:
            raise ValueError(
                f"validation_fraction must be between 0 and 1, got "
                f"{self.validation_fraction}"
            )
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass
class Split:
    """A train/validation division, recorded by filename.

    Written to disk so the same images stay in validation across runs. Without
    this, adding one photograph reshuffles the split and two runs' validation
    numbers cannot be compared.
    """

    train: dict[str, list[str]] = field(default_factory=dict)
    validation: dict[str, list[str]] = field(default_factory=dict)

    def counts(self) -> str:
        train = sum(len(v) for v in self.train.values())
        validation = sum(len(v) for v in self.validation.values())
        return f"{train} training, {validation} validation"

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(
            json.dumps({"train": self.train, "validation": self.validation}, indent=2),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> Split:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(train=payload["train"], validation=payload["validation"])


def make_split(
    report: dataset.DatasetReport,
    *,
    fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SEED,
) -> Split:
    """Divide each class independently, so both appear in validation.

    Splitting the pooled set can leave a small class absent from validation
    entirely, and then its recall is undefined and the metrics mislead.
    """
    split = Split()
    generator = random.Random(seed)

    for item in report.classes:
        names = sorted(
            p.name for p in item.path.iterdir()
            if p.is_file()
            and p.suffix.lower() in dataset.IMAGE_SUFFIXES
            and dataset.looks_like_an_image(p)
        )
        generator.shuffle(names)
        cut = max(1, int(len(names) * fraction)) if names else 0
        split.validation[item.name] = sorted(names[:cut])
        split.train[item.name] = sorted(names[cut:])

    return split


def materialise(
    report: dataset.DatasetReport, split: Split, destination: str | Path
) -> tuple[Path, Path]:
    """Lay the split out as two folders Keras can read.

    Copies rather than links: a symlink needs a privileged account on Windows,
    which is where this runs.
    """
    base = Path(destination)
    train_dir, validation_dir = base / "train", base / "validation"
    for folder in (train_dir, validation_dir):
        if folder.exists():
            shutil.rmtree(folder)

    for class_name, names in split.train.items():
        target = train_dir / class_name
        target.mkdir(parents=True, exist_ok=True)
        source = report.root / class_name
        for name in names:
            shutil.copy2(source / name, target / name)

    for class_name, names in split.validation.items():
        target = validation_dir / class_name
        target.mkdir(parents=True, exist_ok=True)
        source = report.root / class_name
        for name in names:
            shutil.copy2(source / name, target / name)

    return train_dir, validation_dir


def build_model(config: TrainingConfig, class_count: int) -> Any:
    """A frozen backbone with a small head on top."""
    from tensorflow.keras import layers, models

    backbone = backbones.get(config.backbone)
    base = backbone.load()
    base.trainable = False  # phase one: train the head only

    inputs = layers.Input(shape=backbone.input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    # Dropout before the head, because a few hundred images and a 14-million
    # parameter feature extractor overfit quickly.
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    if class_count == 2:
        # One unit and a sigmoid, not two and a softmax. Mathematically the
        # same for two classes, and it gives a single probability that can be
        # thresholded, which is what `evaluate.choose_threshold` needs.
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(class_count, activation="softmax")(x)

    return models.Model(inputs, outputs), base


def _generators(config: TrainingConfig, train_dir: Path, validation_dir: Path):
    """Keras generators, using each backbone's own preprocessing.

    `preprocessing_function` rather than `rescale`. This is the line the
    original was missing.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    backbone = backbones.get(config.backbone)
    preprocess = backbone.preprocess_function()
    class_mode = "binary" if len(list(train_dir.iterdir())) == 2 else "categorical"

    augmentation = dict(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=(0.85, 1.15),
        fill_mode="nearest",
    ) if config.augment else {}

    train_generator = ImageDataGenerator(
        preprocessing_function=preprocess, **augmentation
    ).flow_from_directory(
        train_dir,
        target_size=backbone.input_size,
        batch_size=config.batch_size,
        class_mode=class_mode,
        shuffle=True,
        seed=config.seed,
    )

    # No augmentation on validation, and no shuffling, so scores line up with
    # labels when they are read back for threshold selection.
    validation_generator = ImageDataGenerator(
        preprocessing_function=preprocess
    ).flow_from_directory(
        validation_dir,
        target_size=backbone.input_size,
        batch_size=config.batch_size,
        class_mode=class_mode,
        shuffle=False,
    )

    return train_generator, validation_generator


def train(
    data_root: str | Path,
    output: str | Path = "model.keras",
    config: TrainingConfig | None = None,
    *,
    work_dir: str | Path | None = None,
) -> tuple[Any, Manifest]:
    """Train, choose a threshold, and save the model with its manifest."""
    # The dataset is checked before TensorFlow is imported. Importing it takes
    # several seconds and can fail outright, and neither is a useful thing to do
    # in order to discover that the data folder is empty or absent.
    config = config or TrainingConfig()
    report = dataset.inspect(data_root)
    dataset.require_trainable(report)

    import numpy as np
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )
    from tensorflow.keras.optimizers import Adam

    from . import evaluate

    split = make_split(
        report, fraction=config.validation_fraction, seed=config.seed
    )
    base_work = Path(work_dir) if work_dir else Path(output).parent / "_split"
    train_dir, validation_dir = materialise(report, split, base_work)
    split.save(Path(output).parent / "split.json")

    train_generator, validation_generator = _generators(
        config, train_dir, validation_dir
    )
    class_names = sorted(train_generator.class_indices,
                         key=lambda k: train_generator.class_indices[k])
    model, base = build_model(config, len(class_names))

    loss = "binary_crossentropy" if len(class_names) == 2 else "categorical_crossentropy"
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )

    checkpoint = Path(output).with_suffix(".best.keras")
    callbacks = [
        ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True),
        # restore_best_weights is the point. Without it the model in memory
        # after training is `patience` epochs worse than the best one seen.
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7),
    ]

    class_weight = report.class_weights() if config.use_class_weights else None
    model.fit(
        train_generator,
        epochs=config.epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    # -- phase two: fine-tune the top of the backbone ----------------------
    if config.fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[: -config.fine_tune_layers]:
            layer.trainable = False
        model.compile(
            optimizer=Adam(learning_rate=config.fine_tune_learning_rate),
            loss=loss,
            metrics=["accuracy"],
        )
        model.fit(
            train_generator,
            epochs=config.fine_tune_epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            class_weight=class_weight,
        )

    # -- the threshold, from validation scores -----------------------------
    validation_generator.reset()
    scores = model.predict(validation_generator).ravel()
    labels = validation_generator.classes

    threshold = 0.5
    if len(class_names) == 2:
        choice = evaluate.choose_threshold(
            labels, scores,
            objective=config.threshold_objective,
            minimum_precision=config.minimum_precision,
        )
        threshold = choice.threshold

    performance = evaluate.report(
        labels, scores, threshold=threshold, class_names=class_names
    )

    model.save(output)
    manifest = Manifest(
        backbone=config.backbone,
        input_size=backbones.get(config.backbone).input_size,
        class_names=class_names,
        threshold=threshold,
        notes={
            "split": split.counts(),
            "class_weights": class_weight,
            "augmented": config.augment,
            "fine_tuned": config.fine_tune_epochs > 0,
            "validation_accuracy": round(performance.matrix.accuracy, 4),
            "validation_precision": round(performance.matrix.precision, 4),
            "validation_recall": round(performance.matrix.recall, 4),
            "validation_roc_auc": None if np.isnan(performance.auc)
            else round(performance.auc, 4),
            "majority_baseline": performance.majority_baseline,
            "beats_baseline": performance.beats_baseline,
        },
    )
    manifest.save(output)

    print()
    print(performance.describe())
    return model, manifest
