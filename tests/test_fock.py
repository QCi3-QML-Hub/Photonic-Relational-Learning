"""AI-generated (Claude, Anthropic): tests for cv_relational_learning.fock."""

import numpy as np
import pytest
import strawberryfields as sf
import torch
from strawberryfields import ops

from conftest import random_ket
from cv_relational_learning.fock import (
    apply_tritter,
    aswap_observable,
    beamsplitter,
    coherent_state,
    cyclic_operator,
    displacement,
    expectation,
    fock_state,
    number_operator,
    rotation,
    squeezing,
    swap_operator,
)

D = 12


def _is_unitary(U):
    U = U.numpy()
    return np.allclose(U.conj().T @ U, np.eye(len(U)), atol=1e-10)


def test_gates_are_unitary():
    assert _is_unitary(displacement(D, torch.tensor(0.4 + 0.3j)))
    assert _is_unitary(rotation(D, torch.tensor(0.7)))
    assert _is_unitary(squeezing(torch.tensor(0.5), D, 0.3))
    assert _is_unitary(beamsplitter(D, 0.6, 0.2))


def test_swap_test_gives_overlap(rng):
    psi, phi = random_ket(D, rng), random_ket(D, rng)
    value = expectation(torch.kron(psi, phi), swap_operator(D))
    assert np.isclose(value.real, abs(torch.vdot(psi, phi)) ** 2)
    assert np.isclose(value.imag, 0)


def test_cycle_test_gives_bargmann_invariant(rng):
    d = 4
    a, b, c = (random_ket(d, rng) for _ in range(3))
    value = expectation(torch.kron(torch.kron(a, b), c), cyclic_operator(d, 3))
    bargmann = torch.vdot(a, c) * torch.vdot(b, a) * torch.vdot(c, b)
    assert torch.allclose(value, bargmann)


def test_aswap_observable(rng):
    psi1, psi2 = random_ket(D, rng), random_ket(D, rng)
    B = rotation(D, torch.tensor(0.9))
    value = expectation(torch.kron(psi1, psi2), aswap_observable(B, D))
    assert np.isclose(value.real, abs(torch.vdot(psi2, B @ psi1)) ** 2)


def test_number_operator_counts_photons_in_chosen_mode():
    d = 5
    state = torch.kron(torch.kron(fock_state(1, d), fock_state(3, d)), fock_state(2, d))
    for mode, n in enumerate([1, 3, 2]):
        assert np.isclose(expectation(state, number_operator(d, 3, mode)).real, n)


def test_apply_tritter_matches_dense_beamsplitters(rng):
    d = 4
    params = [0.3, 0.1, np.pi / 4, 2 * np.pi / 3, 0.3, -2 * np.pi / 3]
    x = random_ket(d**3, rng)
    eye = torch.eye(d, dtype=torch.complex128)
    bs01 = lambda t, p: torch.kron(beamsplitter(d, t, p), eye)
    bs12 = lambda t, p: torch.kron(eye, beamsplitter(d, t, p))
    dense = bs01(params[4], params[5]) @ bs12(params[2], params[3]) @ bs01(params[0], params[1])
    assert torch.allclose(apply_tritter(x, d, params), dense @ x)


def test_coherent_state_is_displaced_vacuum():
    alpha = 0.8 - 0.3j
    displaced = displacement(25, torch.tensor(alpha)) @ fock_state(0, 25)
    assert np.allclose(displaced.numpy(), coherent_state(alpha, 25), atol=1e-8)


# --- agreement with strawberryfields -------------------------------------------------------------

def _sf_ket(gate, n_modes=1, init=None, cutoff=D):
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    prog = sf.Program(n_modes)
    with prog.context as q:
        if init is not None:
            ops.Ket(init) | q
        gate(q)
    return eng.run(prog).state.ket()


def test_displacement_matches_sf():
    a = 0.3 + 0.2j
    ours = displacement(D, torch.tensor(a)).numpy()[:, 0]
    assert np.allclose(ours, _sf_ket(lambda q: ops.Dgate(abs(a), np.angle(a)) | q[0]), atol=1e-6)


def test_rotation_matches_sf():
    one = fock_state(1, D).numpy()
    ours = rotation(D, torch.tensor(0.7)).numpy() @ one
    assert np.allclose(ours, _sf_ket(lambda q: ops.Rgate(0.7) | q[0], init=one), atol=1e-6)


@pytest.mark.parametrize("phi", [0.0, 0.3, np.pi / 2])
def test_squeezing_convention_vs_sf(phi):
    """squeezing(r, phi) here == sf Sgate(-r, -phi) == Sgate(r, pi - phi)."""
    r, cutoff = 0.4, 20
    ours = squeezing(torch.tensor(r), cutoff, phi).numpy()[:, 0]
    assert np.allclose(ours, _sf_ket(lambda q: ops.Sgate(-r, -phi) | q[0], cutoff=cutoff), atol=1e-5)
    if not np.isclose(phi, np.pi / 2):  # the conventions only coincide at phi = pi/2
        assert not np.allclose(ours, _sf_ket(lambda q: ops.Sgate(r, phi) | q[0], cutoff=cutoff), atol=1e-2)


def test_beamsplitter_matches_sf():
    two = np.kron(fock_state(1, D).numpy(), fock_state(0, D).numpy()).reshape(D, D)
    ours = (beamsplitter(D, 0.6, 0.2).numpy() @ two.reshape(-1)).reshape(D, D)
    assert np.allclose(ours, _sf_ket(lambda q: ops.BSgate(0.6, 0.2) | (q[0], q[1]), n_modes=2, init=two))


def test_observable_objects_match_dense_matrices(rng):
    from cv_relational_learning.fock import CycleObservable, NumberObservable

    d = 5
    for n_modes in (2, 3):
        psi = random_ket(d**n_modes, rng)
        dense_cycle = swap_operator(d) if n_modes == 2 else cyclic_operator(d, n_modes)
        assert torch.allclose(expectation(psi, CycleObservable(d, n_modes)), expectation(psi, dense_cycle))
        for mode in range(n_modes):
            assert torch.allclose(expectation(psi, NumberObservable(d, n_modes, mode)),
                                  expectation(psi, number_operator(d, n_modes, mode)))
