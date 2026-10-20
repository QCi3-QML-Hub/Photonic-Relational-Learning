"""
AI-generated (Claude, Anthropic): tests for cv_relational_learning.circuits, using the trained
parameters stored with the paper results.
"""

import numpy as np
import torch

from cv_relational_learning.analysis import best_threshold
from cv_relational_learning.circuits import (
    ChiralityCycleCircuit,
    RotationSwapCircuit,
    RotationSwapShotCircuit,
    SensingSwapCircuit,
)
from cv_relational_learning.datasets import load_dataset
from cv_relational_learning.fock import cyclic_operator, swap_operator
from cv_relational_learning.paths import RESULTS_DIR


def _trained(fig, model):
    return torch.load(RESULTS_DIR / fig / model / "model_state.pth")["params"]


def _outputs(circuit, params, dataset):
    with torch.no_grad():
        return np.array([circuit.forward(params, x).item() for x in dataset.data])


def test_swap_undoes_hidden_rotation():
    """The trained SWAP model rotates by ~ -1, so class-1 pairs become identical: <SWAP> ~ 1."""
    params = _trained("fig3", "SWAP")
    assert np.isclose(params.item(), -1.0, atol=0.05)
    _, _, test = load_dataset("hidden_rotation", 25, 10, 2, 50)
    out = _outputs(RotationSwapCircuit(25, swap_operator(25)), params, test)
    labels = test.targets.numpy()
    assert out[labels == 1].min() > 0.95
    assert np.isclose(RotationSwapCircuit(25, swap_operator(25)).forward(torch.tensor([-1.0]), test.data[0]).item(),
                      1.0, atol=1e-3)
    assert best_threshold(out, labels)[1] == 1.0


def test_cycle_model_separates_chirality_classes():
    """CYCLE test with the trained squeeze (~ +0.7, undoing sf's Sgate(0.7); cutoff 10 for speed)."""
    params = _trained("fig5", "CYCLE")
    test = load_dataset("chirality", 10, 0, 0, 10)[2]
    out = _outputs(ChiralityCycleCircuit(10, cyclic_operator(10, 3)), params, test)
    assert best_threshold(out, test.targets.numpy())[1] >= 0.9


def test_sensing_swap_model_separates_classes():
    params = _trained("fig7", "SWAP")
    test = load_dataset("sensing", 20, 0, 0, 10)[2]
    out = _outputs(SensingSwapCircuit(20, swap_operator(20)), params, test)
    assert best_threshold(out, test.targets.numpy())[1] == 1.0


def test_shot_noise_is_unbiased_with_clt_variance():
    torch.manual_seed(0)
    cutoff, n_shots = 10, 500
    x = load_dataset("hidden_rotation", cutoff, 1, 0, 0)[0].data[0]
    params = torch.tensor([-0.5], dtype=torch.double)
    exact = RotationSwapCircuit(cutoff, swap_operator(cutoff)).forward(params, x).item()
    noisy = RotationSwapShotCircuit(cutoff, swap_operator(cutoff), n_shots)
    samples = np.array([noisy.forward(params, x).item() for _ in range(4000)])
    assert np.isclose(samples.mean(), exact, atol=4 * samples.std() / np.sqrt(len(samples)))
    assert np.isclose(samples.std(), np.sqrt((1 - exact**2) / n_shots), rtol=0.1)
