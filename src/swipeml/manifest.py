"""What a saved model needs to remember about how it was trained.

The failure this exists to prevent is train/serve skew, and it is the kind that
does not announce itself. Train with Inception preprocessing at 299x299, then
predict with `img / 255.0` at 224x224, and every prediction is wrong in a way
that looks like a merely mediocre model. No exception, no warning.

The original had exactly this. Training resized to 224x224 and rescaled by
1/255; `predictor.py` resized to 320x320 and then reshaped to 224x224, which
raises `ValueError: cannot reshape array of size 307200 into shape
(1,224,224,3)` on every call. It could never have run.

So a model is saved with a manifest beside it, and `predict.py` reads the
manifest rather than taking anyone's word for the preprocessing. If the manifest
is missing, prediction refuses instead of guessing.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import backbones

#: Saved next to the model file, same stem.
MANIFEST_SUFFIX = ".manifest.json"
#: Bumped if the fields change in a way older manifests cannot satisfy.
SCHEMA_VERSION = 1


class ManifestError(RuntimeError):
    """The manifest is missing, unreadable, or describes a different model."""


def _now() -> str:
    """Timezone-aware UTC. `datetime.utcnow` is naive and deprecated, and
    `datetime.UTC` is 3.11+ while this package supports 3.10."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Manifest:
    """Everything needed to feed a saved model correctly."""

    backbone: str
    input_size: tuple[int, int]
    class_names: list[str]
    #: The probability above which the positive class is chosen. Not assumed to
    #: be 0.5: see `evaluate.choose_threshold`.
    threshold: float = 0.5
    schema_version: int = SCHEMA_VERSION
    created: str = field(default_factory=_now)
    #: Free-form, for whatever is worth recording about the run.
    notes: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.input_size = tuple(self.input_size)  # type: ignore[assignment]
        if len(self.input_size) != 2:
            raise ManifestError(f"input_size must be (height, width), got {self.input_size!r}")
        if len(self.class_names) < 2:
            raise ManifestError("a classifier needs at least two classes")
        if not 0.0 < self.threshold < 1.0:
            raise ManifestError(f"threshold must be between 0 and 1, got {self.threshold}")
        backbones.get(self.backbone)  # raises with the list if unknown

    @property
    def positive_class(self) -> str:
        """The class a score above the threshold means.

        Keras' `flow_from_directory` assigns indices alphabetically, so with
        folders named `liked` and `disliked`, index 1 is `liked`. Recording the
        order rather than relying on remembering it is the point.
        """
        return self.class_names[-1]

    # -- persistence -------------------------------------------------------
    @staticmethod
    def path_for(model_path: str | Path) -> Path:
        model = Path(model_path)
        return model.with_suffix(model.suffix + MANIFEST_SUFFIX)

    def save(self, model_path: str | Path) -> Path:
        target = self.path_for(model_path)
        payload = asdict(self)
        payload["input_size"] = list(self.input_size)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, model_path: str | Path) -> Manifest:
        target = cls.path_for(model_path)
        if not target.is_file():
            raise ManifestError(
                f"no manifest at {target}. A model saved without one cannot be "
                "served safely, because the preprocessing it was trained with is "
                "not recorded anywhere. Retrain, or write the manifest by hand."
            )
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{target.name} is not valid JSON: {exc}") from exc

        version = int(payload.get("schema_version", 0))
        if version > SCHEMA_VERSION:
            raise ManifestError(
                f"{target.name} is schema version {version}, this build "
                f"understands up to {SCHEMA_VERSION}"
            )

        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            # Forwards-compatible: a newer writer may add fields, and dropping
            # them is better than refusing to load.
            payload = {k: v for k, v in payload.items() if k in known}
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ManifestError(f"{target.name} is missing required fields: {exc}") from exc

    def describe(self) -> str:
        return "\n".join([
            f"backbone     {self.backbone}",
            f"input size   {self.input_size[0]}x{self.input_size[1]}",
            f"preprocess   {backbones.get(self.backbone).preprocess_style}",
            f"classes      {', '.join(self.class_names)}"
            f"  (positive: {self.positive_class})",
            f"threshold    {self.threshold:.3f}",
            f"trained      {self.created}",
        ])
