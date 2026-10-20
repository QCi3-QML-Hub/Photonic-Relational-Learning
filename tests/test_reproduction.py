"""
AI-generated (Claude, Anthropic): retraining with this package reproduces the stored paper results.

Training of the Fig. 3/5/7 models and of the Fig. 8 shadow classifiers is deterministic, so test-set
predictions must match the stored ones. Fig. 5 needs ~3 GB of RAM. The seeded Fig. 8 adaptive runs are
checked for one realisation.
"""

import numpy as np
import pytest

from cv_relational_learning.experiments import (
    RELATIONAL_EXPERIMENTS,
    adaptive_seed,
    run_adaptive_shot_experiment,
    run_relational_experiment,
    run_shadow_experiment,
)
from cv_relational_learning.paths import RESULTS_DIR


@pytest.mark.slow
@pytest.mark.parametrize(
    "fig, model",
    [("fig3", "SWAP"), ("fig3", "NUMBER"), ("fig5", "CYCLE"), ("fig5", "NUMBER"), ("fig7", "SWAP"), ("fig7", "NUMBER")],
)
def test_relational_models_reproduce(fig, model, tmp_path):
    preds, targets = run_relational_experiment(RELATIONAL_EXPERIMENTS[fig], model, tmp_path, verbose=False)
    stored = np.load(RESULTS_DIR / fig / model / "preds.npy")
    assert np.array_equal(targets, np.load(RESULTS_DIR / fig / model / "targets.npy"))
    assert np.allclose(preds, stored, rtol=1e-6, atol=1e-8)


@pytest.mark.slow
def test_shadow_baseline_reproduces(tmp_path):
    preds = run_shadow_experiment(500, 1, tmp_path)
    assert np.allclose(preds, np.load(RESULTS_DIR / "fig8" / "shadow" / "500shots_fold1.npy"), rtol=1e-6, atol=1e-8)


@pytest.mark.slow
def test_adaptive_run_reproduces(tmp_path):
    preds, _ = run_adaptive_shot_experiment(500, tmp_path, seed=adaptive_seed(500, 1))
    assert np.allclose(preds, np.load(RESULTS_DIR / "fig8" / "adaptive" / "500shots" / "stat1" / "preds.npy"))
