"""
Truncated Fock-space linear algebra in torch.

In the small-system regime of the paper, plain (autograd-able) matrix multiplication is
faster than simulating with strawberryfields, so all trained models are built from these
operators. Conventions follow strawberryfields (hbar = 2) unless stated otherwise.

Multi-mode states are flattened kets of length cutoff**n_modes with mode 0 as the most
significant index, i.e. |n_0, n_1, ...> sits at index n_0 * cutoff**(M-1) + n_1 * cutoff**(M-2) + ...
"""

import math

import numpy as np
import torch
from scipy.special import factorial

dtype = torch.complex128


# ---------------------------------------------------------------------------
# Single-mode building blocks
# ---------------------------------------------------------------------------

def annihilation(d):
    a = torch.zeros((d, d), dtype=dtype)
    for n in range(1, d):
        a[n - 1, n] = torch.sqrt(torch.tensor(n, dtype=torch.float64))
    return a


def creation(d):
    return annihilation(d).conj().T


def identity(d):
    diag = torch.tensor([1 for _ in range(d)], dtype=torch.float64)
    return torch.diag(diag).to(dtype)


def parity(d):
    diag = torch.tensor([(-1) ** n for n in range(d)], dtype=torch.float64)
    return torch.diag(diag).to(dtype)


def fock_state(n, d):
    ket = torch.zeros(d, dtype=dtype)
    ket[n] = 1.0
    return ket


def coherent_state(alpha, cutoff):
    """Coherent-state ket |alpha> (strawberryfields convention), renormalised after truncation."""
    n = np.arange(cutoff)
    ket = np.exp(-0.5 * np.abs(alpha) ** 2) * (alpha**n / np.sqrt(factorial(n)))
    ket = ket.astype(np.complex128)
    return ket / np.linalg.norm(ket)


def cat_state(alpha, cutoff, p=0):
    """Cat-state ket |alpha> + (-1)^p |-alpha> (p=0 even, p=1 odd), normalised."""
    ket = coherent_state(alpha, cutoff) + (-1) ** p * coherent_state(-alpha, cutoff)
    return ket / np.linalg.norm(ket)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def displacement(d, alpha):
    a = annihilation(d)
    adag = creation(d)
    generator = alpha * adag - torch.conj(alpha) * a
    return torch.matrix_exp(generator)


def rotation(d, theta):
    """Phase rotation R(theta) = exp(i theta a^dag a), diagonal in the Fock basis."""
    n = torch.arange(d, dtype=torch.float64)
    phases = torch.exp(1j * theta * n)
    return torch.diag(phases).to(dtype)


def squeezing(r, d, phi=0):
    """
    Single-mode squeezer exp((z* a^dag^2 - z a^2) / 2) with z = r exp(i phi). `r` must be a tensor.

    NOTE: differs from strawberryfields: squeezing(r, d, phi) here equals sf.ops.Sgate(-r, -phi),
    i.e. Sgate(r, pi - phi). E.g. undoing sf's Sgate(0.7) needs squeezing(+0.7) here.
    """
    a = annihilation(d)
    adag = creation(d)
    z = r * torch.exp(1j * torch.tensor(phi, dtype=torch.float64, device=r.device))
    exponent = 0.5 * (z.conj() * (adag @ adag) - z * (a @ a))
    return torch.matrix_exp(exponent)


def beamsplitter(d, theta, phi):
    """
    Two-mode beamsplitter matching sf.ops.BSgate(theta, phi):
        B = exp(theta * (exp(i phi) a1 a2^dag - exp(-i phi) a1^dag a2))
    """
    dim = d * d
    G = torch.zeros((dim, dim), dtype=dtype)

    ephi = complex(math.cos(phi), math.sin(phi))
    ephic = complex(math.cos(phi), -math.sin(phi))

    for n in range(d):
        for m in range(d):
            i = n * d + m  # input |n>_1 |m>_2

            # a1 a2^dag: |n,m> -> sqrt(n (m+1)) |n-1, m+1>
            if n > 0 and m + 1 < d:
                j = (n - 1) * d + (m + 1)
                G[j, i] += ephi * math.sqrt(n * (m + 1))

            # -a1^dag a2: |n,m> -> -sqrt((n+1) m) |n+1, m-1>
            if n + 1 < d and m > 0:
                j = (n + 1) * d + (m - 1)
                G[j, i] -= ephic * math.sqrt((n + 1) * m)

    return torch.linalg.matrix_exp(theta * G)


def apply_tritter(x, d, params):
    """
    Apply BS_01(t3,p3) BS_12(t2,p2) BS_01(t1,p1) to a flat 3-mode ket without building
    the d^3 x d^3 matrix. With suitable parameters this is the balanced tritter used for
    the three-state CYCLE test. params = (t1, p1, t2, p2, t3, p3).
    """
    t1, p1, t2, p2, t3, p3 = params

    BS1 = beamsplitter(d, t1, p1).view(d, d, d, d)
    BS2 = beamsplitter(d, t2, p2).view(d, d, d, d)
    BS3 = beamsplitter(d, t3, p3).view(d, d, d, d)

    x_tensor = x.view(d, d, d)
    x_tensor = torch.einsum("ijmn,mnk->ijk", BS1, x_tensor)  # modes 0, 1
    x_tensor = torch.einsum("jkmn,imn->ijk", BS2, x_tensor)  # modes 1, 2
    x_tensor = torch.einsum("ijmn,mnk->ijk", BS3, x_tensor)  # modes 0, 1
    return x_tensor.reshape(-1)


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

def swap_operator(d):
    """SWAP on two modes. Its expectation on |psi>|phi> is |<psi|phi>|^2 (the SWAP test)."""
    dim = d * d
    S = torch.zeros((dim, dim), dtype=dtype)
    for n in range(d):
        for m in range(d):
            S[n * d + m, m * d + n] = 1.0
    return S


def cyclic_operator(d, M):
    """
    Cyclic shift of M modes, (i1, i2, ..., iM) -> (i2, ..., iM, i1). Its expectation on a
    product state is the M-th order Bargmann invariant (the CYCLE test).
    NOTE: dense d^M x d^M matrix; ~1 GB for d=20, M=3.
    """
    dim = d**M
    C = torch.zeros((dim, dim), dtype=dtype)

    for idx in range(dim):
        indices = []
        tmp = idx
        for _ in range(M):
            indices.append(tmp % d)
            tmp //= d
        indices = indices[::-1]

        shifted = indices[1:] + [indices[0]]

        j = 0
        for k in range(M):
            j = j * d + shifted[k]
        C[idx, j] = 1.0

    return C


def number_operator(d, M, mode=0):
    """Photon number of `mode` in an M-mode system (the local "NUMBER" observable)."""
    dim = d**M
    N = torch.zeros((dim, dim))

    for idx in range(dim):
        occupations = []
        tmp = idx
        for _ in range(M):
            occupations.append(tmp % d)
            tmp //= d
        occupations = occupations[::-1]
        N[idx, idx] = float(occupations[mode])

    return N


def aswap_observable(B, cutoff):
    """
    ASWAP observable (B^dag (x) B) SWAP. On a product state |psi_1>|psi_2> its expectation is
    |<psi_2|B|psi_1>|^2, i.e. a SWAP test after applying B to the first state.
    """
    A = torch.kron(B.conj().T.contiguous(), B)
    return A @ swap_operator(cutoff)


def expectation(psi, O):
    """<psi|O|psi> for a dense matrix O, or for an observable object (CycleObservable, NumberObservable)."""
    if hasattr(O, "expectation"):
        return O.expectation(psi)
    O = O.to(torch.complex128)
    return torch.vdot(psi, O @ psi)


# ---------------------------------------------------------------------------
# The same observables without dense matrices (a d^M x d^M matrix is ~1 GB for d = 20, M = 3)
# ---------------------------------------------------------------------------

class CycleObservable:
    """
    SWAP (n_modes = 2) or CYCLE (n_modes > 2) as a permutation of the mode indices,
    (C psi)[i_1, ..., i_M] = psi[i_2, ..., i_M, i_1]. Same as swap_operator / cyclic_operator.
    """

    def __init__(self, cutoff, n_modes):
        self.cutoff = cutoff
        self.n_modes = n_modes

    def expectation(self, psi):
        shape = (self.cutoff,) * self.n_modes
        shifted = psi.reshape(shape).permute(self.n_modes - 1, *range(self.n_modes - 1)).reshape(-1)
        return torch.vdot(psi, shifted)


class NumberObservable:
    """Photon number of `mode` in an n_modes system, a diagonal observable. Same as number_operator."""

    def __init__(self, cutoff, n_modes, mode=0):
        shape = [1] * n_modes
        shape[mode] = cutoff
        n = torch.arange(cutoff, dtype=torch.float64).reshape(shape)
        self.weights = n.expand((cutoff,) * n_modes).reshape(-1)

    def expectation(self, psi):
        return torch.vdot(psi, self.weights * psi)
