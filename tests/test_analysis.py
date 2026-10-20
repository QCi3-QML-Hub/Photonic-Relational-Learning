"""AI-generated (Claude, Anthropic): tests for cv_relational_learning.analysis."""

import numpy as np

from cv_relational_learning.analysis import best_threshold


def test_perfect_separation():
    thresh, acc = best_threshold(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
    assert acc == 1.0 and 0.2 < thresh <= 0.8


def test_best_threshold_maximises_accuracy():
    preds = np.array([0.1, 0.4, 0.35, 0.8, 0.3, 0.9])
    targets = np.array([0, 0, 1, 1, 0, 1])
    thresh, acc = best_threshold(preds, targets)
    brute = max(np.mean((preds >= t) == targets) for t in np.r_[preds, np.inf])
    assert np.isclose(acc, brute)
    assert np.isclose(np.mean((preds >= thresh) == targets), acc)
