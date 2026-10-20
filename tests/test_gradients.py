"""AI-generated (Claude, Anthropic): tests for cv_relational_learning.gradients."""

import numpy as np
import torch

from cv_relational_learning.circuits import RotationSwapCircuit
from cv_relational_learning.datasets import load_dataset
from cv_relational_learning.fock import swap_operator
from cv_relational_learning.gradients import generalised_parameter_shift, parameter_shift


def test_two_term_shift_is_exact_for_single_frequency():
    f = lambda p: np.cos(p[0]) + 0.3 * np.sin(p[0])
    p = np.array([0.4])
    assert np.isclose(parameter_shift(f, p, 0), -np.sin(0.4) + 0.3 * np.cos(0.4))


def test_generalised_shift_matches_autograd_on_paper_circuit():
    """The Fig. 8 gradient rule is exact for the rotation ansatz on cutoff-10 Fock states."""
    cutoff = 10
    x = load_dataset("hidden_rotation", cutoff, 1, 0, 0)[0].data[0]
    circuit = RotationSwapCircuit(cutoff, swap_operator(cutoff))

    theta = torch.tensor([-0.7], dtype=torch.double, requires_grad=True)
    circuit.forward(theta, x).backward()

    f = lambda p, x: circuit.forward(torch.tensor(p), x).item()
    shift_grad = generalised_parameter_shift(f, np.array([-0.7]), 0, circuit_args=(x,), n_frequencies=cutoff)
    assert np.isclose(shift_grad, theta.grad.item(), atol=1e-10)
