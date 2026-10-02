"""Metrics that mean something, and choosing a threshold.

Accuracy on its own is close to useless here. The original reported 85% with no
baseline beside it, and on a two-class set that is 80/20 a model can reach 80%
by always answering the same way. Whether 85% is good depends entirely on the
split, and the split was never reported.

So: precision, recall, F1 and ROC-AUC, plus the majority baseline to compare
against. And the decision threshold is chosen from the validation scores rather
than left at 0.5, because 0.5 is only right when the classes are balanced and
the two kinds of mistake cost the same.

Implemented on numpy alone. The metrics are twenty lines, scikit-learn is a
large dependency to add for them, and writing them out makes the definitions
visible instead of implied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

#: Any float or int array. Recent numpy stubs infer very precise generic types
#: that mypy will not index or call `.max()` on, and naming the type plainly is
#: clearer than scattering ignores through the arithmetic.
Array: TypeAlias = np.ndarray


@dataclass(frozen=True)
class ConfusionMatrix:
    """Counts at one threshold."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def total(self) -> int:
        return (self.true_positive + self.false_positive
                + self.true_negative + self.false_negative)

    @property
    def accuracy(self) -> float:
        if not self.total:
            return 0.0
        return (self.true_positive + self.true_negative) / self.total

    @property
    def precision(self) -> float:
        """Of the ones it said yes to, how many were right.

        Zero when it never says yes, which is the honest answer: a model that
        makes no positive predictions has no precision, and defining it as 1.0
        would flatter exactly the degenerate case worth catching.
        """
        predicted = self.true_positive + self.false_positive
        return self.true_positive / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        """Of the ones that were yes, how many it found."""
        actual = self.true_positive + self.false_negative
        return self.true_positive / actual if actual else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        if not denominator:
            return 0.0
        return 2 * self.precision * self.recall / denominator

    def describe(self) -> str:
        return "\n".join([
            f"{'':>14}{'said yes':>12}{'said no':>12}",
            f"{'was yes':>14}{self.true_positive:>12}{self.false_negative:>12}",
            f"{'was no':>14}{self.false_positive:>12}{self.true_negative:>12}",
            "",
            f"accuracy   {self.accuracy:.3f}",
            f"precision  {self.precision:.3f}",
            f"recall     {self.recall:.3f}",
            f"f1         {self.f1:.3f}",
        ])


def confusion_at(labels: Array, scores: Array, threshold: float) -> ConfusionMatrix:
    """Counts at a given threshold. Labels are 0 or 1, scores are probabilities."""
    label_array: Array = np.asarray(labels).astype(int).ravel()
    score_array: Array = np.asarray(scores, dtype=float).ravel()
    if label_array.shape != score_array.shape:
        raise ValueError(
            f"{label_array.size} label(s) against {score_array.size} score(s)"
        )

    predicted: Array = np.asarray(score_array >= threshold)
    actual: Array = np.asarray(label_array == 1)
    missed: Array = np.logical_not(predicted)
    negative: Array = np.logical_not(actual)
    return ConfusionMatrix(
        true_positive=int(np.sum(np.logical_and(predicted, actual))),
        false_positive=int(np.sum(np.logical_and(predicted, negative))),
        true_negative=int(np.sum(np.logical_and(missed, negative))),
        false_negative=int(np.sum(np.logical_and(missed, actual))),
    )


def roc_auc(labels: Array, scores: Array) -> float:
    """Area under the ROC curve, threshold-free.

    Computed from the rank-sum identity rather than by integrating the curve,
    which handles ties correctly by giving them their average rank. Integrating
    a curve built from unique score values quietly mishandles a model that
    outputs the same probability for many inputs, and a barely-trained model
    does exactly that.
    """
    label_array: Array = np.asarray(labels).astype(int).ravel()
    score_array: Array = np.asarray(scores, dtype=float).ravel()
    positives = int(np.sum(label_array == 1))
    negatives = int(np.sum(label_array == 0))
    if not positives or not negatives:
        # Undefined with one class present. NaN rather than a number that would
        # be read as a result.
        return float("nan")

    order: Array = np.asarray(np.argsort(score_array, kind="mergesort"))
    ranks: Array = np.empty(order.shape, dtype=float)
    ranks[order] = np.arange(1, score_array.size + 1, dtype=float)

    # Average the ranks within each group of tied scores.
    sorted_scores: Array = np.asarray(score_array[order])
    start = 0
    for index in range(1, sorted_scores.size + 1):
        at_end = index == sorted_scores.size
        if at_end or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                tied: Array = np.asarray(order[start:index])
                ranks[tied] = float(np.mean(ranks[tied]))
            start = index

    positive_rank_sum = float(np.sum(ranks[label_array == 1]))
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    matrix: ConfusionMatrix
    objective: str
    score: float


def choose_threshold(
    labels: Array,
    scores: Array,
    *,
    objective: str = "f1",
    minimum_precision: float = 0.0,
    steps: int = 101,
) -> ThresholdChoice:
    """Pick the decision threshold from validation scores.

    0.5 is the default everywhere and is only correct when the classes are
    balanced and a false positive costs the same as a false negative. Neither
    holds here: the dataset is whatever you happened to swipe, and the two
    mistakes are not equally annoying.

    `minimum_precision` is the useful knob. Set it to 0.8 and the threshold is
    chosen from only those that keep precision at or above 0.8, maximising
    recall within that. If none can, the constraint is reported rather than
    silently ignored.
    """
    label_array: Array = np.asarray(labels).astype(int).ravel()
    score_array: Array = np.asarray(scores, dtype=float).ravel()
    if label_array.size == 0:
        raise ValueError("cannot choose a threshold with no validation data")

    objectives = {
        "f1": lambda m: m.f1,
        "accuracy": lambda m: m.accuracy,
        "precision": lambda m: m.precision,
        "recall": lambda m: m.recall,
    }
    if objective not in objectives:
        raise ValueError(
            f"unknown objective {objective!r}. Available: " + ", ".join(objectives)
        )
    measure = objectives[objective]

    # Candidates span the open interval. 0.0 and 1.0 are excluded because a
    # threshold at either end always predicts one class, which maximises recall
    # or precision trivially and is never what was wanted.
    # Plain Python rather than np.linspace. It is the same arithmetic, there is
    # no array involved, and the type is obvious to a reader and to a checker.
    candidates: list[float] = [index / (steps - 1) for index in range(1, steps - 1)]

    best: ThresholdChoice | None = None
    constrained_best: ThresholdChoice | None = None
    for threshold in candidates:
        matrix = confusion_at(label_array, score_array, float(threshold))
        value = measure(matrix)
        choice = ThresholdChoice(float(threshold), matrix, objective, value)

        if best is None or value > best.score:
            best = choice
        clears_precision = matrix.precision >= minimum_precision
        better_recall = (
            constrained_best is None
            or matrix.recall > constrained_best.matrix.recall
        )
        if clears_precision and better_recall:
            constrained_best = choice

    assert best is not None
    if minimum_precision > 0:
        if constrained_best is None:
            best_available = max(
                confusion_at(label_array, score_array, value).precision
                for value in candidates
            )
            raise ValueError(
                f"no threshold reaches precision {minimum_precision:.2f}; "
                f"the best available is {best_available:.3f}"
            )
        return constrained_best
    return best


@dataclass
class Report:
    """A model's validation performance, with the baseline to judge it against."""

    matrix: ConfusionMatrix
    auc: float
    threshold: float
    majority_baseline: float
    class_names: list[str]

    @property
    def beats_baseline(self) -> bool:
        """Whether the model learned anything at all."""
        return self.matrix.accuracy > self.majority_baseline

    def describe(self) -> str:
        lines = [
            f"threshold {self.threshold:.3f}, positive class {self.class_names[-1]!r}",
            "",
            self.matrix.describe(),
            "",
            f"roc auc    {self.auc:.3f}"
            + ("  (undefined: only one class present)" if np.isnan(self.auc) else ""),
            f"baseline   {self.majority_baseline:.3f}  (always predict the majority)",
        ]
        if not self.beats_baseline:
            lines.append(
                "\nThis model does not beat always predicting the majority class. "
                "Its accuracy is not evidence that it learned anything."
            )
        return "\n".join(lines)


def report(
    labels: Array,
    scores: Array,
    *,
    threshold: float = 0.5,
    class_names: list[str] | None = None,
) -> Report:
    label_array: Array = np.asarray(labels).astype(int).ravel()
    counts: Array = np.asarray(np.bincount(label_array, minlength=2))
    total = int(np.sum(counts))
    baseline = float(int(np.max(counts)) / total) if total else 0.0
    return Report(
        matrix=confusion_at(label_array, scores, threshold),
        auc=roc_auc(label_array, scores),
        threshold=threshold,
        majority_baseline=round(baseline, 4),
        class_names=class_names or ["negative", "positive"],
    )
