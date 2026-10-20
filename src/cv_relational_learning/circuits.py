"""
The trainable circuits used in the paper, implemented with torch linear algebra (see fock.py).

Each circuit maps (params, datapoint) -> scalar model output via `forward(params, x)`:
    - "relational" circuits apply shallow local gates to the input states and measure a global
      SWAP / CYCLE test (an n-state Bargmann invariant);
    - "local" circuits are the baselines: the same kind of gates plus an entangling
      beamsplitter/tritter, but measuring the photon number of a single mode.

Task               Relational circuit          Local baseline          Paper
hidden rotation    RotationSwapCircuit         RotationLocalCircuit    Fig. 3, Fig. 8 (with shots)
chirality          ChiralityCycleCircuit       ChiralityLocalCircuit   Fig. 5
sensing            SensingSwapCircuit          SensingLocalCircuit     Fig. 7
"""

import warnings

import numpy as np
import torch

from cv_relational_learning.fock import (
    apply_tritter,
    beamsplitter,
    expectation,
    identity,
    rotation,
    squeezing,
)
from cv_relational_learning.gradients import generalised_parameter_shift


def _check_truncation(state, cutoff, threshold=1e-2):
    """Warn if a two-mode state has significant population at the Fock cutoff."""
    probs_2d = (torch.abs(state) ** 2).view(cutoff, cutoff)
    edge_pop = probs_2d[-1, :].sum() + probs_2d[:, -1].sum()
    if edge_pop > threshold:
        warnings.warn(f"Fock cutoff edges highly populated: {edge_pop:.3e}")


# ---------------------------------------------------------------------------
# Hidden rotation (two-mode pairs)
# ---------------------------------------------------------------------------

class RotationSwapCircuit:
    """
    Rotate the second state by params[0], then measure `observable` (the SWAP operator).
    (formerly RParamMatrixCirc)
    """

    def __init__(self, cutoff, observable):
        self.cutoff = cutoff
        self.observable = observable

    def forward(self, params, x):
        x = x.reshape(-1)
        R = rotation(self.cutoff, theta=params[0])
        x = torch.kron(identity(self.cutoff), R) @ x
        return torch.real(expectation(x, self.observable))


class RotationSwapShotCircuit:
    """
    RotationSwapCircuit with finite-shot noise, trained with the generalised parameter shift.
    (formerly RParamMatrixShotCirc)

    Shot noise is a central-limit approximation of the destructive SWAP test of
    Volkoff & Subasi (arXiv:2202.09923): the estimator averages +-1 outcomes, so its variance
    is (1 - <SWAP>^2) / n_shots.
    """

    def __init__(self, cutoff, observable, n_shots):
        self.cutoff = cutoff
        self.observable = observable
        self.n_shots = n_shots

    def forward(self, params, x):
        x = torch.as_tensor(x).reshape(-1)
        R = rotation(self.cutoff, theta=params[0])
        x = torch.kron(identity(self.cutoff), R) @ x
        obs_expval = expectation(x, self.observable)
        std = torch.sqrt((1 - torch.real(obs_expval**2)) / self.n_shots)
        noise = torch.randn(1) * std
        return torch.real(obs_expval) + noise

    def grad(self, params, x):
        full_grad = np.zeros_like(params)
        for i in range(len(params)):
            full_grad[i] = generalised_parameter_shift(
                self.forward, params, i, circuit_args=(x,), n_frequencies=self.cutoff
            )
        return full_grad


class RotationLocalCircuit:
    """
    Local baseline for the hidden rotation task: rotate the second state by params[0],
    beamsplitter with angle params[1], measure `observable` (photon number of mode 0).
    (formerly RParamMCompetitorCirc)
    """

    def __init__(self, cutoff, observable):
        self.cutoff = cutoff
        self.observable = observable

    def forward(self, params, x):
        x = x.reshape(-1)
        R = rotation(self.cutoff, theta=params[0])
        BS = beamsplitter(self.cutoff, theta=params[1], phi=0)
        x = BS @ torch.kron(identity(self.cutoff), R) @ x
        # NOTE: somewhat arbitrary scaling such that outputs are roughly between 0 and 1
        return torch.real(expectation(x, self.observable)) / 10


# ---------------------------------------------------------------------------
# Chirality (three-mode triplets)
# ---------------------------------------------------------------------------

class ChiralityCycleCircuit:
    """
    Squeeze the third state by params[0], then measure `observable` (the 3-mode cyclic shift).
    Returns the phase of the Bargmann invariant (the geometric phase), which is real-valued.
    (formerly BargmannMatrixCirc)
    """

    def __init__(self, cutoff, observable):
        self.cutoff = cutoff
        self.observable = observable

    def forward(self, params, x):
        x = x.reshape(-1)
        S = squeezing(params[0], self.cutoff)
        ansatz = torch.kron(torch.kron(identity(self.cutoff), identity(self.cutoff)), S)
        x = ansatz @ x
        return torch.angle(expectation(x, self.observable))


class ChiralityLocalCircuit:
    """
    Local baseline for the chirality task: squeeze the third state by params[0], apply a
    tritter-like interferometer parametrised by params[1], measure `observable`
    (photon number of mode 0). (formerly BargmannMCompetitorCirc)
    """

    def __init__(self, cutoff, observable):
        self.cutoff = cutoff
        self.observable = observable

    def forward(self, params, x):
        d = self.cutoff
        x = x.reshape(-1)
        S = squeezing(params[0], d)
        # apply S to mode 2 without building the d^3 x d^3 matrix
        x = torch.einsum("lk,ijk->ijl", S, x.view(d, d, d)).reshape(-1)
        x = apply_tritter(x, d, [params[1], 0, np.pi / 4, 2 * np.pi / 3, params[1], -2 * np.pi / 3])
        return torch.real(expectation(x, self.observable))


# ---------------------------------------------------------------------------
# Sensing (two copies, data = the unknown interaction)
# ---------------------------------------------------------------------------

class SensingSwapCircuit:
    """
    Two-copy sensing protocol. Starting from vacuum, squeeze copy 0 by params[0], apply the
    datapoint (a two-mode interaction: signal on copy 1, shared nuisance rotation on both),
    then measure `observable` (SWAP). (formerly SqueezingInteractionEchoMatrixCirc)
    """

    def __init__(self, cutoff, observable):
        self.cutoff = cutoff
        self.observable = observable

    def forward(self, params, x):
        interaction = x.to(torch.complex128)
        state = torch.zeros(self.cutoff**2, dtype=torch.complex128)
        state[0] = 1.0  # vacuum

        S = squeezing(params[0], self.cutoff)
        state = torch.kron(S, identity(self.cutoff)) @ state
        state = interaction @ state

        _check_truncation(state, self.cutoff)
        return torch.real(expectation(state, self.observable))


class SensingLocalCircuit:
    """
    Local baseline for sensing: squeeze copy 0 (params[0]), apply the interaction,
    beamsplitter (params[1]), squeeze copy 0 again (params[2]), measure `observable`
    (photon number of mode 0). (formerly SqueezingInteractionEchoCompetitorCirc)
    """

    def __init__(self, cutoff, observable):
        self.cutoff = cutoff
        self.observable = observable

    def forward(self, params, x):
        interaction = x.to(torch.complex128)
        state = torch.zeros(self.cutoff**2, dtype=torch.complex128)
        state[0] = 1.0  # vacuum

        S1 = squeezing(params[0], self.cutoff)
        BS = beamsplitter(self.cutoff, theta=params[1], phi=0)
        S2 = squeezing(params[2], self.cutoff)

        state = torch.kron(S1, identity(self.cutoff)) @ state
        state = interaction @ state
        state = BS @ state
        state = torch.kron(S2, identity(self.cutoff)) @ state

        _check_truncation(state, self.cutoff)
        return torch.real(expectation(state, self.observable))
