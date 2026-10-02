"""Metrics, and choosing a threshold.

The point of this module is that accuracy alone is misleading on an unbalanced
dataset, which is what "85% accuracy" in the original README was.
"""
from __future__ import annotations

import numpy as np
import pytest

from swipeml import evaluate as ev


def test_a_perfect_classifier() -> None:
    matrix = ev.confusion_at([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert matrix.accuracy == 1.0
    assert matrix.precision == 1.0
    assert matrix.recall == 1.0
    assert matrix.f1 == 1.0


def test_a_classifier_that_always_says_no() -> None:
    """Scores well on accuracy when the classes are unbalanced, and is useless.

    Precision is 0.0, not 1.0. Defining it as 1.0 when nothing was predicted
    positive would flatter exactly the degenerate case worth catching.
    """
    labels = [0] * 90 + [1] * 10
    scores = [0.1] * 100
    matrix = ev.confusion_at(labels, scores, 0.5)

    assert matrix.accuracy == 0.9
    assert matrix.precision == 0.0
    assert matrix.recall == 0.0
    assert matrix.f1 == 0.0


def test_the_report_says_when_a_model_fails_to_beat_the_baseline() -> None:
    labels = [0] * 90 + [1] * 10
    scores = [0.1] * 100
    report = ev.report(labels, scores, threshold=0.5)

    assert report.majority_baseline == 0.9
    assert not report.beats_baseline
    assert "does not beat" in report.describe()


def test_a_model_that_beats_the_baseline_is_not_warned_about() -> None:
    labels = [0] * 50 + [1] * 50
    scores = [0.2] * 50 + [0.8] * 50
    report = ev.report(labels, scores, threshold=0.5)
    assert report.beats_baseline
    assert "does not beat" not in report.describe()


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="label"):
        ev.confusion_at([0, 1], [0.5], 0.5)


# -- roc auc ---------------------------------------------------------------

def test_auc_of_a_perfect_ranking_is_one() -> None:
    assert ev.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_auc_of_an_inverted_ranking_is_zero() -> None:
    assert ev.roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_auc_with_every_score_tied_is_a_half() -> None:
    """A barely-trained network outputs the same probability for everything.

    Integrating a curve built from unique score values gets this wrong; the
    rank-sum form with averaged tied ranks gets it right, and 0.5 is the honest
    answer for a model carrying no information.
    """
    assert ev.roc_auc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auc_with_partial_ties() -> None:
    assert ev.roc_auc([0, 1], [0.5, 0.5]) == 0.5
    assert 0.5 < ev.roc_auc([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]) < 1.0


def test_auc_with_one_class_present_is_nan() -> None:
    """Undefined, so NaN rather than a number that would be read as a score."""
    assert np.isnan(ev.roc_auc([1, 1, 1], [0.2, 0.5, 0.9]))
    assert np.isnan(ev.roc_auc([0, 0], [0.2, 0.5]))


def test_auc_is_threshold_free() -> None:
    """Which is why it is reported next to accuracy: it says whether the model
    ranks correctly regardless of where the cut is placed."""
    labels = [0] * 50 + [1] * 50
    scores = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    assert ev.roc_auc(labels, scores) == 1.0


# -- threshold selection ---------------------------------------------------

def test_the_threshold_is_chosen_not_assumed() -> None:
    """0.5 is only right when the classes are balanced and both mistakes cost
    the same. Here the positive class is rare and the scores are shifted."""
    rng = np.random.default_rng(3)
    labels = (rng.random(400) < 0.2).astype(int)
    scores = np.clip(rng.normal(0.3, 0.15, 400) + labels * 0.35, 0.001, 0.999)

    choice = ev.choose_threshold(labels, scores, objective="f1")
    at_half = ev.confusion_at(labels, scores, 0.5)
    assert choice.matrix.f1 >= at_half.f1


def test_each_objective_moves_the_threshold_the_expected_way() -> None:
    rng = np.random.default_rng(11)
    labels = (rng.random(400) < 0.3).astype(int)
    scores = np.clip(rng.normal(0.4, 0.2, 400) + labels * 0.3, 0.001, 0.999)

    for_precision = ev.choose_threshold(labels, scores, objective="precision")
    for_recall = ev.choose_threshold(labels, scores, objective="recall")

    assert for_precision.threshold > for_recall.threshold
    assert for_precision.matrix.precision >= for_recall.matrix.precision
    assert for_recall.matrix.recall >= for_precision.matrix.recall


def test_a_minimum_precision_constraint_is_honoured() -> None:
    """"Only act when you are fairly sure" is the useful knob here, and it is a
    constraint on precision with recall maximised inside it."""
    rng = np.random.default_rng(5)
    labels = (rng.random(600) < 0.35).astype(int)
    scores = np.clip(rng.normal(0.4, 0.2, 600) + labels * 0.35, 0.001, 0.999)

    choice = ev.choose_threshold(labels, scores, minimum_precision=0.7)
    assert choice.matrix.precision >= 0.7


def test_an_unreachable_precision_is_reported_not_ignored() -> None:
    rng = np.random.default_rng(5)
    labels = (rng.random(300) < 0.3).astype(int)
    scores = rng.random(300)  # no signal at all

    with pytest.raises(ValueError, match="no threshold reaches"):
        ev.choose_threshold(labels, scores, minimum_precision=0.999)


def test_thresholds_never_reach_the_endpoints() -> None:
    """A threshold of 0 or 1 always predicts one class, which maximises recall
    or precision trivially and is never what was wanted."""
    labels = [0, 1] * 50
    scores = list(np.linspace(0.01, 0.99, 100))
    for objective in ("f1", "accuracy", "precision", "recall"):
        choice = ev.choose_threshold(labels, scores, objective=objective)
        assert 0.0 < choice.threshold < 1.0


def test_an_unknown_objective_lists_the_real_ones() -> None:
    with pytest.raises(ValueError, match="f1"):
        ev.choose_threshold([0, 1], [0.2, 0.8], objective="vibes")


def test_choosing_with_no_data_is_refused() -> None:
    with pytest.raises(ValueError, match="no validation data"):
        ev.choose_threshold([], [])
