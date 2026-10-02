"""The labelled images, and whether they are fit to train on.

The dataset is two folders:

    training_data/
      liked/
      disliked/

Simple, and it has failure modes that are worth catching before a training run
rather than after it. A class with eleven images. A 95/5 split that makes
"always say no" a 95%-accurate model. Files that are not images because a
download wrote an error page with a .jpg name, which is exactly what the
original's `requests.get` without `raise_for_status` did.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: What `flow_from_directory` will read. Anything else in the folder is ignored
#: by Keras silently, so it is reported here instead.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"})
#: Below this, a class cannot support a meaningful validation split.
MINIMUM_PER_CLASS = 50
#: At or above this ratio between the largest and smallest class, accuracy
#: stops meaning much and class weights are needed. Inclusive, because 1.5:1 is
#: already the point where it matters.
IMBALANCE_WARNING = 1.5
#: The first few bytes of each format, for checking a file is what it claims.
MAGIC_NUMBERS = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"BM": "bmp",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
    b"RIFF": "webp",
}


class DatasetError(RuntimeError):
    """The dataset cannot be trained on."""


@dataclass
class ClassCount:
    name: str
    path: Path
    images: int
    #: Files with an image suffix whose contents are not an image.
    corrupt: list[str]
    #: Files that are not images at all, which Keras would skip silently.
    other_files: list[str]


@dataclass
class DatasetReport:
    """What is in the dataset, and what is wrong with it."""

    root: Path
    classes: list[ClassCount]

    @property
    def total(self) -> int:
        return sum(c.images for c in self.classes)

    @property
    def imbalance(self) -> float:
        """Largest class over smallest. 1.0 is perfectly balanced."""
        counts = [c.images for c in self.classes if c.images]
        if len(counts) < 2:
            return float("inf")
        return round(max(counts) / min(counts), 2)

    @property
    def majority_baseline(self) -> float:
        """The accuracy of always predicting the biggest class.

        The number any trained model has to beat to have learned anything. The
        original reported 85% accuracy with no baseline to compare it against,
        and on an unbalanced two-class set that could have been worse than
        guessing the majority.
        """
        if not self.total:
            return 0.0
        return round(max(c.images for c in self.classes) / self.total, 4)

    def problems(self) -> list[str]:
        """Everything that would make a training run misleading or fail."""
        found = []
        if len(self.classes) < 2:
            found.append(
                f"found {len(self.classes)} class folder(s) in {self.root}; "
                "two are needed, named after the labels"
            )
        for item in self.classes:
            if item.images == 0:
                found.append(f"{item.name!r} has no images")
            elif item.images < MINIMUM_PER_CLASS:
                found.append(
                    f"{item.name!r} has only {item.images} image(s); "
                    f"under {MINIMUM_PER_CLASS} is too few to validate against"
                )
            if item.corrupt:
                found.append(
                    f"{item.name!r} has {len(item.corrupt)} file(s) that are not "
                    f"images despite the extension: {', '.join(item.corrupt[:3])}"
                )
            if item.other_files:
                found.append(
                    f"{item.name!r} has {len(item.other_files)} non-image file(s), "
                    "which Keras skips without saying so"
                )
        if len(self.classes) >= 2 and self.imbalance >= IMBALANCE_WARNING:
            found.append(
                f"classes are unbalanced {self.imbalance}:1, so always predicting "
                f"the majority scores {self.majority_baseline:.1%}. Train with "
                "class weights and judge on precision and recall, not accuracy"
            )
        return found

    def class_weights(self) -> dict[int, float]:
        """Weights that make each class contribute equally to the loss.

        Indices follow `flow_from_directory`, which sorts class folder names
        alphabetically. Computed the same way as sklearn's "balanced" mode,
        without the dependency.
        """
        counts = [c.images for c in self.classes]
        if not counts or min(counts) == 0:
            raise DatasetError("cannot compute class weights with an empty class")
        total = sum(counts)
        # Not rounded. These multiply a loss, and rounding to a few decimal
        # places stops the classes contributing exactly equally, which is the
        # one thing the weights exist to do.
        return {
            index: total / (len(counts) * count)
            for index, count in enumerate(counts)
        }

    @property
    def class_names(self) -> list[str]:
        """Alphabetical, matching how Keras assigns indices."""
        return [c.name for c in self.classes]

    def describe(self) -> str:
        lines = [f"{self.total} image(s) in {self.root}"]
        for item in self.classes:
            lines.append(f"  {item.name:<16}{item.images:>6}")
        if len(self.classes) >= 2:
            lines.append(f"  imbalance        {self.imbalance}:1")
            lines.append(f"  majority baseline {self.majority_baseline:.1%}"
                         "  <- a model must beat this to have learned anything")
        problems = self.problems()
        if problems:
            lines.append(f"\n{len(problems)} problem(s):")
            lines += [f"  {p}" for p in problems]
        return "\n".join(lines)


def looks_like_an_image(path: Path, *, read_bytes: int = 16) -> bool:
    """Whether a file starts with a known image magic number.

    Cheap, and it catches the case that actually happens: a download that wrote
    an HTML error page to a .jpg because nothing checked the status code.
    """
    try:
        head = path.open("rb").read(read_bytes)
    except OSError:
        return False
    return any(head.startswith(magic) for magic in MAGIC_NUMBERS)


def inspect(root: str | Path) -> DatasetReport:
    """Look at a dataset folder without loading a single image into memory."""
    base = Path(root)
    if not base.is_dir():
        raise DatasetError(f"no dataset folder at {base}")

    classes = []
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        images, corrupt, other = 0, [], []
        for item in sorted(folder.iterdir()):
            if not item.is_file():
                continue
            if item.suffix.lower() in IMAGE_SUFFIXES:
                if looks_like_an_image(item):
                    images += 1
                else:
                    corrupt.append(item.name)
            else:
                other.append(item.name)
        classes.append(ClassCount(folder.name, folder, images, corrupt, other))

    return DatasetReport(root=base, classes=classes)


def require_trainable(report: DatasetReport) -> None:
    """Raise unless the dataset can be trained on.

    Imbalance is a warning, not a blocker: it is workable with class weights.
    Too few images, or a class with none, is a blocker.
    """
    blocking = [
        problem
        for problem in report.problems()
        if "unbalanced" not in problem and "skips without saying so" not in problem
    ]
    if blocking:
        raise DatasetError(
            "this dataset cannot be trained on:\n"
            + "\n".join(f"  {p}" for p in blocking)
        )
