"""Prediction, using the preprocessing the model was actually trained with.

The manifest is not optional. A model with no manifest is refused rather than
served with a guess, because the guess is invisible when it is wrong: the model
returns confident nonsense and looks like a model that merely underperforms.

The original guessed, and guessed inconsistently. Training resized to 224x224
and divided by 255. `predictor.py` resized to 320x320 and then called
`.reshape(1, 224, 224, 3)` on it, which raises

    ValueError: cannot reshape array of size 307200 into shape (1,224,224,3)

every time it is called. It also loaded the model into a module-level global at
import time, so `model_predict` raised `NameError` when the file was absent
rather than saying the model was missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import backbones
from .manifest import Manifest


class PredictError(RuntimeError):
    """The model or the image cannot be used."""


@dataclass
class Prediction:
    """One image's verdict."""

    path: str
    score: float
    label: str
    threshold: float

    @property
    def positive(self) -> bool:
        return self.score >= self.threshold

    @property
    def confidence(self) -> float:
        """How far from the threshold, scaled to 0..1.

        Reported separately from the score because a score of 0.51 against a
        threshold of 0.5 is a coin toss, and `positive` alone does not say so.
        """
        span = max(self.threshold, 1 - self.threshold)
        return round(min(1.0, abs(self.score - self.threshold) / span), 4)

    def __str__(self) -> str:
        return (f"{Path(self.path).name}: {self.label} "
                f"(score {self.score:.3f}, confidence {self.confidence:.2f})")


class Classifier:
    """A trained model, loaded with its manifest.

    Loaded on construction rather than at import, so a missing file is an error
    the caller can handle at a sensible moment.
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise PredictError(
                f"no model at {self.model_path}. Train one first: "
                "swipeml train <data folder>"
            )
        self.manifest = Manifest.load(self.model_path)
        self.backbone = backbones.get(self.manifest.backbone)
        self._preprocess = self.backbone.preprocess_function()
        self._model: Any = None

    @property
    def model(self) -> Any:
        """Loaded on first use, because importing TensorFlow is not quick."""
        if self._model is None:
            from tensorflow.keras.models import load_model

            self._model = load_model(self.model_path)
        return self._model

    # -- the part that must match training ---------------------------------
    def prepare(self, image_path: str | Path) -> np.ndarray:
        """Load one image exactly as training loaded it.

        The size and the preprocessing both come from the manifest, so there is
        no second place for them to be written down differently.
        """
        from tensorflow.keras.preprocessing import image as keras_image

        source = Path(image_path)
        if not source.is_file():
            raise PredictError(f"no image at {source}")

        try:
            loaded = keras_image.load_img(
                source, target_size=self.manifest.input_size
            )
        except Exception as exc:  # noqa: BLE001
            # Pillow raises several different things for "this is not an image",
            # and the caller only needs to know which file.
            raise PredictError(f"{source.name} could not be read as an image: {exc}") from exc

        array = keras_image.img_to_array(loaded)
        array = np.expand_dims(array, axis=0)
        return self._preprocess(array)

    def predict(self, image_path: str | Path) -> Prediction:
        batch = self.prepare(image_path)
        raw = self.model.predict(batch, verbose=0)
        return self._interpret(str(image_path), raw[0])

    def predict_many(self, image_paths: list[str | Path], batch_size: int = 16):
        """Several images in batches, which is markedly faster per image."""
        predictions = []
        for start in range(0, len(image_paths), batch_size):
            chunk = image_paths[start : start + batch_size]
            arrays, usable = [], []
            for path in chunk:
                try:
                    arrays.append(self.prepare(path)[0])
                    usable.append(path)
                except PredictError as exc:
                    # One unreadable file must not take down a batch of fifty.
                    print(f"skipped {exc}")
            if not arrays:
                continue
            raw = self.model.predict(np.stack(arrays), verbose=0)
            predictions += [
                self._interpret(str(path), row)
                for path, row in zip(usable, raw, strict=False)
            ]
        return predictions

    def _interpret(self, path: str, row: np.ndarray) -> Prediction:
        """Turn a model output into a labelled score.

        Handles both heads: one sigmoid unit, or N softmax units. The original
        rounded the output to an int and returned 0, 1 or - when neither
        matched - `None`, which the caller then compared with `== 0`, so a None
        silently became a dislike.
        """
        row = np.asarray(row).ravel()
        if row.size == 1:
            score = float(row[0])
            index = 1 if score >= self.manifest.threshold else 0
        else:
            index = int(np.argmax(row))
            score = float(row[index])

        names = self.manifest.class_names
        label = names[index] if index < len(names) else f"class_{index}"
        return Prediction(
            path=path,
            score=score if row.size == 1 else score,
            label=label,
            threshold=self.manifest.threshold,
        )

    def describe(self) -> str:
        return f"{self.model_path.name}\n{self.manifest.describe()}"
