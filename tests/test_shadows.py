"""AI-generated (Claude, Anthropic): tests for cv_relational_learning.shadows."""

import numpy as np
import torch

from cv_relational_learning.datasets import load_dataset
from cv_relational_learning.fock import expectation, fock_state, number_operator
from cv_relational_learning.shadows import (
    compute_psi,
    hidden_rotation_observables,
    reconstruct_state,
    sample_homodyne_shots,
    shadow_expvals,
)


def test_oscillator_eigenfunctions_are_orthonormal():
    x = np.linspace(-12, 12, 6001)
    psi = compute_psi(8, x)
    gram = psi @ psi.T * (x[1] - x[0])
    assert np.allclose(gram, np.eye(9), atol=1e-8)


def test_shadow_reconstructs_fock_state():
    np.random.seed(0)
    cutoff = 5
    xs, thetas = sample_homodyne_shots(fock_state(1, cutoff).numpy(), 1, cutoff, 3000)
    rho = reconstruct_state(xs, thetas, 1, cutoff)
    assert np.isclose(np.trace(rho).real, 1, atol=0.1)
    assert np.isclose(rho[1, 1].real, 1, atol=0.15)
    assert np.abs(rho - np.diag([0, 1, 0, 0, 0])).max() < 0.25


def test_shadow_expvals_estimate_photon_numbers():
    np.random.seed(1)
    cutoff = 5
    state = torch.kron(fock_state(0, cutoff), fock_state(2, cutoff)).numpy().reshape(cutoff, cutoff)
    xs, thetas = sample_homodyne_shots(state, 2, cutoff, 3000)
    observables = [number_operator(cutoff, 2, 0).to(torch.complex128), number_operator(cutoff, 2, 1).to(torch.complex128)]
    est = shadow_expvals(xs, thetas, observables, 2, cutoff)
    assert np.allclose(est, [0, 2], atol=0.35)


def test_hidden_rotation_feature_peaks_at_hidden_rotation():
    """Exact <O(r)> on a class-1 pair is ~1 at r = 1 (the hidden rotation) and smaller elsewhere."""
    pair = load_dataset("hidden_rotation", 25, 1, 0, 0)[0].data[0].reshape(-1)
    values = [expectation(pair, obs).real.item() for obs in hidden_rotation_observables(25)]
    assert np.isclose(values[2], 1, atol=1e-3)
    assert np.argmax(values) == 2


def test_extended_phi_matches_exact_irregular_solution():
    """phi_0 = 2 pi^(-3/4) e^(x^2/2) D(x) (Dawson), also beyond the band where the original asymptotics are used."""
    from scipy.special import dawsn
    from cv_relational_learning.shadows import bohr_sommerfeld_band, compute_phi

    x = np.linspace(0.5, 11, 40)
    exact = 2 * np.pi ** (-3 / 4) * np.exp(x**2 / 2) * dawsn(x)
    assert np.allclose(compute_phi(25, x)[0] / exact, 1, atol=1e-4)
    beyond = x > bohr_sommerfeld_band(25)
    assert np.abs(compute_phi(25, x[beyond], "asymptotic")[0] / exact[beyond] - 1).max() > 1e-3


def test_gaussian_sampler_matches_moments():
    from cv_relational_learning.shadows import gaussian_homodyne_shots

    means = np.array([1.0, -0.5, 0.3, 2.0])  # xxpp
    cov = np.array([[2.0, 0.0, 0.4, 0.0], [0.0, 1.0, 0.0, -0.2], [0.4, 0.0, 0.8, 0.0], [0.0, -0.2, 0.0, 1.5]])
    xs, thetas = gaussian_homodyne_shots(means, cov, 400_000, np.random.default_rng(0))
    c, s = np.cos(thetas), np.sin(thetas)
    resid = xs - (c * means[:2] + s * means[2:])
    var = c**2 * np.diag(cov)[:2] + 2 * c * s * np.array([cov[0, 2], cov[1, 3]]) + s**2 * np.diag(cov)[2:]
    assert np.allclose(resid.mean(0), 0, atol=0.01)
    assert np.allclose((resid**2 / var).mean(0), 1, atol=0.01)


def test_fast_hidden_rotation_features_match_dense_traces():
    from cv_relational_learning.shadows import compute_shadow_features, hidden_rotation_features, hidden_rotation_observables

    rng = np.random.default_rng(2)
    xs, thetas = rng.normal(1, 2, (2, 300, 2)), rng.uniform(0, np.pi, (2, 300, 2))
    fast = hidden_rotation_features(xs, thetas, 12, progress=False)
    dense = compute_shadow_features(xs, thetas, hidden_rotation_observables(12), 2, 12, progress=False)
    assert np.allclose(fast, dense, atol=1e-9)


def test_generic_block_estimates_match_problem_specific_formula():
    from cv_relational_learning.shadows import hidden_rotation_observables, hidden_rotation_shot_features, shadow_expval_blocks

    rng = np.random.default_rng(5)
    xs, thetas = rng.normal(0.5, 2, (400, 2)), rng.uniform(0, np.pi, (400, 2))
    generic = shadow_expval_blocks(xs, thetas, hidden_rotation_observables(10), 2, 10, block_size=100)
    specific = hidden_rotation_shot_features(xs, thetas, 10).reshape(4, 100, -1).mean(1)
    assert np.allclose(generic, specific, atol=1e-9)


def test_hidden_rotation_sampling_methods():
    from cv_relational_learning.shadows import sample_hidden_rotation_shots

    for method in ("sf", "gaussian_moments"):
        xs, thetas, labels, params = sample_hidden_rotation_shots(1, 30, seed=0, return_params=True, method=method)
        assert xs.shape == thetas.shape == (2, 30, 2) and labels.tolist() == [1, 0] and params.shape == (2, 5)
    # same seed -> same pairs, whichever sampler
    p_sf = sample_hidden_rotation_shots(1, 5, seed=3, return_params=True, method="sf")[3]
    p_gm = sample_hidden_rotation_shots(1, 5, seed=3, return_params=True, method="gaussian_moments")[3]
    assert np.allclose(p_sf[0], p_gm[0])
