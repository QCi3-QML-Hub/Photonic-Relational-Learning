"""
Continuous-variable classical shadows (Gandhari et al., PRX Quantum 2024) from random-quadrature
homodyne measurements, used for the "measure-first" baseline of Sec. IV / Fig. 8.

Each shot measures every mode along a uniformly random quadrature angle theta in [0, pi), giving
(x, theta) per mode. A shot is turned into a single-mode shadow element with Leonhardt's pattern
functions f_mn (Leonhardt et al., PRA 1996),
    rho_mn ~ pi * f_mn(x) * exp(i (m - n) theta),
and a multi-mode shot into the tensor product of its single-mode elements. Averaging shadow elements
estimates the density matrix; averaging Tr(O * element) estimates <O>.
Quadratures follow strawberryfields (hbar = 2), hence X_SCALE.

The generic pipeline (any state, any observable) samples shots with strawberryfields and estimates
<O> = Tr(O rho_shadow) from the reconstructed shadow state; pattern functions and reconstructions are
vectorised over shots. The last section contains PROBLEM-SPECIFIC speedups that are exact only for the
hidden-rotation task of Fig. 8 (Gaussian states; ASWAP observables with a diagonal rotation).
"""

import warnings

import numpy as np
from tqdm import tqdm

from cv_relational_learning.fock import aswap_observable, rotation

X_SCALE = 1 / np.sqrt(2)  # sf quadratures (hbar = 2) -> hbar = 1 used by the pattern functions

# Candidate rotations r for the hidden_rotation shadow features O(r) = ASWAP(R(r)) (Fig. 8)
SHADOW_ANGLES = (0.2, 0.6, 1.0, 1.4, 1.8)


# ---------------------------------------------------------------------------
# Regular (psi_n) and irregular (phi_n) oscillator solutions, hbar = 1 units
# ---------------------------------------------------------------------------

def compute_psi(max_n, x):
    """Harmonic-oscillator eigenfunctions psi_0..psi_max_n at x, shape (max_n + 1, len(x))."""
    x = np.asarray(x, dtype=float)
    psi = np.zeros((max_n + 1, len(x)))
    psi[0] = np.pi ** (-1 / 4) * np.exp(-(x**2) / 2)
    if max_n == 0:
        return psi
    psi[1] = np.sqrt(2) * x * psi[0]
    for n in range(2, max_n + 1):
        psi[n] = np.sqrt(2.0 / n) * x * psi[n - 1] - np.sqrt((n - 1.0) / n) * psi[n - 2]
    return psi


def bohr_sommerfeld_radius(n):
    return np.sqrt(2 * n + 1)


def bohr_sommerfeld_band(n):
    """Eq. 57: inside this region the irregular solution behaves like the regular one."""
    a_n = bohr_sommerfeld_radius(n)
    return a_n - (2 * a_n) ** (-1 / 3)


def phi0_asymptotic(x):
    """Leading-order irregular phi_0 for large |x| (used by the original code outside the band)."""
    x = np.asarray(x, dtype=float)
    return np.pi ** (-3 / 4) * (1 / x) * np.exp(x**2 / 2)


def semiclassical_phi(m, x):
    """Semiclassical high-m phi_m inside the band; recursed backward from here."""
    am = bohr_sommerfeld_radius(m)
    x = np.asarray(x, dtype=float)
    if np.any(np.abs(x) / am > 1):
        warnings.warn("semiclassical_phi outside its domain, will produce NaN")
    t_m = np.arccos(x / am)
    return np.sqrt(2 / (np.pi * am * np.sin(t_m))) * np.sin((am**2 / 4) * (np.sin(2 * t_m) - 2 * t_m) + np.pi / 4)


def compute_phi(max_n, x, outside_band="extended"):
    """
    Irregular solutions phi_0..phi_max_n at x, shape (max_n + 1, len(x)).

    Semiclassical initialisation at m = 4*max_n followed by backward recursion (eq. 58), which is stable
    because phi grows when recursing down. Leonhardt et al. use it only inside the Bohr-Sommerfeld band of
    max_n and the leading-order asymptotics (phi_0 ~ pi^(-3/4) e^(x^2/2) / x, phi_n = sqrt(n/2) phi_(n-1) / x,
    eq. 61) outside it; those are off by ~1-3% just outside the band and bias the estimator for states with
    weight there. Options:
        outside_band="extended":   backward recursion for |x| inside the band of 4*max_n, asymptotics beyond
                                   (agrees with the exact phi_0 = 2 pi^(-3/4) e^(x^2/2) D(x) to ~1e-5);
        outside_band="asymptotic": the original treatment (used for the first version of the paper's features).
    """
    x = np.asarray(x, dtype=float)
    phi = np.zeros((max_n + 1, len(x)))
    m_init = 4 * max_n
    band_edge = {"extended": bohr_sommerfeld_band(m_init), "asymptotic": bohr_sommerfeld_band(max_n)}[outside_band]
    outside = np.abs(x) >= band_edge

    xo = x[outside]
    if xo.size:
        phi[0, outside] = phi0_asymptotic(xo)
        for n in range(1, max_n + 1):
            phi[n, outside] = np.sqrt(n / 2) * (1 / xo) * phi[n - 1, outside]

    xi = x[~outside]
    if xi.size:
        upper, lower = semiclassical_phi(m_init, xi), semiclassical_phi(m_init - 1, xi)  # phi_(m+2), phi_(m+1)
        for m in range(m_init - 2, -1, -1):
            upper, lower = lower, (np.sqrt(2) * xi * lower - np.sqrt(m + 2) * upper) / np.sqrt(m + 1)
            if m <= max_n:
                phi[m, ~outside] = lower
    return phi


# ---------------------------------------------------------------------------
# Pattern functions and shadow elements
# ---------------------------------------------------------------------------

def pattern_functions(x, cutoff, outside_band="extended"):
    """
    Pattern functions f_nm(x) (eq. 52 of Leonhardt et al.) of every shot: shape (n_shots, cutoff, cutoff),
    real and symmetric. `x` are homodyne outcomes in sf units.
    """
    xs = np.atleast_1d(np.asarray(x, dtype=float)) * X_SCALE
    psi = compute_psi(cutoff, xs).T                              # (N, cutoff + 1)
    phi = compute_phi(cutoff, xs, outside_band).T
    k = np.sqrt(2 * (np.arange(cutoff) + 1))
    f = (2 * xs[:, None, None] * psi[:, :cutoff, None] * phi[:, None, :cutoff]
         - k[None, :, None] * psi[:, 1:, None] * phi[:, None, :cutoff]
         - k[None, None, :] * psi[:, :cutoff, None] * phi[:, None, 1:])
    upper = np.triu(f)                                           # entries n <= m are the ones of eq. 52
    return upper + np.swapaxes(np.triu(f, 1), 1, 2)


def shadow_elements(x, theta, cutoff, outside_band="extended"):
    """Single-mode shadow elements pi * f_mn(x) * exp(i (m - n) theta), shape (n_shots, cutoff, cutoff)."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    diff = np.arange(cutoff)[:, None] - np.arange(cutoff)[None, :]
    return np.pi * np.exp(1j * diff[None] * theta[:, None, None]) * pattern_functions(x, cutoff, outside_band)


def shot_to_shadow_element(x, theta, cutoff, outside_band="extended"):
    """Single-mode shadow element of one (x, theta) shot (eq. 11 of Gandhari et al.)."""
    return shadow_elements([x], [theta], cutoff, outside_band)[0]


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def reconstruct_state(sampled_x, sampled_thetas, n_modes, cutoff, chunk=20000, outside_band="extended"):
    """
    Shadow estimate of the (truncated) density matrix, the mean of all shadow elements, for 1 or 2 modes.
    sampled_x, sampled_thetas have shape (n_shots, n_modes).
    """
    sampled_x, sampled_thetas = np.asarray(sampled_x), np.asarray(sampled_thetas)
    if n_modes not in (1, 2):
        raise NotImplementedError("reconstruct_state supports 1 or 2 modes")
    d = cutoff
    total = np.zeros((d**n_modes, d**n_modes), dtype=complex)
    for start in range(0, len(sampled_x), chunk):
        sl = slice(start, start + chunk)
        rho0 = shadow_elements(sampled_x[sl, 0], sampled_thetas[sl, 0], d, outside_band)
        if n_modes == 1:
            total += rho0.sum(0)
        else:
            rho1 = shadow_elements(sampled_x[sl, 1], sampled_thetas[sl, 1], d, outside_band)
            # sum_i rho0_i (x) rho1_i as one matrix product: (a b),(c e) <- sum_i rho0[i,a,c] rho1[i,b,e]
            n = rho0.shape[0]
            prod = rho0.reshape(n, d * d).T @ rho1.reshape(n, d * d)          # (a c), (b e)
            total += prod.reshape(d, d, d, d).transpose(0, 2, 1, 3).reshape(d * d, d * d)
    return total / len(sampled_x)


def shadow_expvals(sampled_x, sampled_thetas, observables, n_modes, cutoff, outside_band="extended"):
    """Shadow estimates of Tr(O rho) for each observable (by linearity, from the reconstructed state)."""
    rho = reconstruct_state(sampled_x, sampled_thetas, n_modes, cutoff, outside_band=outside_band)
    return np.array([np.real(np.trace(np.asarray(O) @ rho)) for O in observables])


def hidden_rotation_observables(cutoff, angles=SHADOW_ANGLES):
    """ASWAP observables O(r) = (R(r)^dag x R(r)) SWAP as dense matrices; class 1 maximises <O(1)> = 1."""
    return [aswap_observable(rotation(cutoff, theta), cutoff) for theta in angles]


def shadow_expval_blocks(sampled_x, sampled_thetas, observables, n_modes, cutoff, block_size, outside_band="extended"):
    """
    Generic: shadow estimates of Tr(O rho) from consecutive blocks of `block_size` shots, shape (n_blocks, n_obs).
    Averaging block estimates of equal size gives the estimate from all their shots, so budgets and realisations
    that are multiples of block_size can be assembled from the blocks.
    """
    obs_t = [np.asarray(O).T for O in observables]  # Tr(O rho) = sum(O^T * rho)
    n_blocks = len(sampled_x) // block_size
    out = np.empty((n_blocks, len(obs_t)))
    for b in range(n_blocks):
        sl = slice(b * block_size, (b + 1) * block_size)
        rho = reconstruct_state(sampled_x[sl], sampled_thetas[sl], n_modes, cutoff, outside_band=outside_band)
        out[b] = [np.real(np.sum(Ot * rho)) for Ot in obs_t]
    return out


def compute_shadow_features(xs, thetas, observables, n_modes, cutoff, progress=True, outside_band="extended"):
    """Generic features (n_datapoints, n_observables) for arbitrary dense observables (slower)."""
    iterator = tqdm(range(len(xs))) if progress else range(len(xs))
    return np.array([shadow_expvals(xs[i], thetas[i], observables, n_modes, cutoff, outside_band) for i in iterator])


# ---------------------------------------------------------------------------
# Sampling (simulated experiment)
# ---------------------------------------------------------------------------

def sample_homodyne_shots(state, n_modes, cutoff, n_shots):
    """Random-quadrature homodyne shots of a Fock-basis ket with the sf Fock backend (slow). Returns (xs, thetas)."""
    import strawberryfields as sf
    from strawberryfields import ops

    engine = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    xs, thetas = [], []
    for _ in range(n_shots):
        directions = np.random.uniform(0, np.pi, n_modes)
        prog = sf.Program(n_modes)
        with prog.context as q:
            ops.Ket(state) | q
            for mode in range(n_modes):
                ops.MeasureHomodyne(directions[mode]) | q[mode]
        xs.append(engine.run(prog).samples[0])
        engine.reset()
        thetas.append(directions)
    return np.asarray(xs), np.asarray(thetas)


def sample_hidden_rotation_shots(n_per_class, n_shots, seed=None, return_params=False, method="sf"):
    """
    Draw fresh hidden_rotation pairs (same distribution as datasets.make_hidden_rotation_dataset) and sample
    `n_shots` random-quadrature homodyne shots of each SINGLE pair. Returns xs, thetas of shape
    (2 * n_per_class, n_shots, 2) and labels (class 1 first), plus the pair parameters (squeeze, squeeze angle,
    displacement, rotation mode 0, rotation mode 1) if requested.

    method="sf":               one strawberryfields Gaussian-backend run per shot (the original method; ~ms per shot);
    method="gaussian_moments": PROBLEM-SPECIFIC, see gaussian_homodyne_shots (exact for these Gaussian states).
    """
    from cv_relational_learning.datasets import sample_hidden_rotation_params

    if seed is not None:
        np.random.seed(seed)
    rng = np.random.default_rng(seed)
    xs, thetas, labels, params = [], [], [], []
    for label in (1, 0):
        for _ in range(n_per_class):
            p = sample_hidden_rotation_params(label)
            if method == "sf":
                x, t = _sf_hidden_rotation_shots(p, n_shots)
            elif method == "gaussian_moments":
                x, t = gaussian_homodyne_shots(*hidden_rotation_gaussian_moments(p), n_shots, rng)
            else:
                raise ValueError(method)
            xs.append(x)
            thetas.append(t)
            labels.append(label)
            params.append(p)
    out = (np.asarray(xs), np.asarray(thetas), np.asarray(labels))
    return out + (np.asarray(params),) if return_params else out


def _sf_hidden_rotation_shots(params, n_shots):
    """One sf Gaussian-backend program per shot, random quadrature per mode (formerly data_utils.rot_gaussian_cs)."""
    import strawberryfields as sf
    from strawberryfields import ops

    from cv_relational_learning.datasets import prepare_hidden_rotation_gates

    eng = sf.Engine("gaussian")
    xs, thetas = [], []
    for _ in range(n_shots):
        directions = np.random.uniform(0, np.pi, 2)
        prog = sf.Program(2)
        with prog.context as q:
            prepare_hidden_rotation_gates(q, *params)
            ops.MeasureHomodyne(directions[0]) | q[0]
            ops.MeasureHomodyne(directions[1]) | q[1]
        xs.append(eng.run(prog).samples[0])
        eng.reset()
        thetas.append(directions)
    return np.asarray(xs), np.asarray(thetas)


# ---------------------------------------------------------------------------
# PROBLEM-SPECIFIC speedups for the hidden-rotation task (Fig. 8). They are exact for that task (validated in
# tests/test_shadows.py against the generic pipeline above) but do not apply to general states or observables.
# ---------------------------------------------------------------------------

def gaussian_homodyne_shots(means, cov, n_shots, rng):
    """
    PROBLEM-SPECIFIC (Gaussian states only). Exact random-quadrature homodyne shots of an M-mode Gaussian state (sf conventions: xxpp ordering,
    hbar = 2). Each shot draws theta_m ~ U[0, pi) per mode and samples x_m cos(theta_m) + p_m sin(theta_m)
    jointly from the Gaussian marginal. Returns (xs, thetas), each of shape (n_shots, M).
    """
    means, cov = np.asarray(means), np.asarray(cov)
    M = len(means) // 2
    thetas = rng.uniform(0, np.pi, size=(n_shots, M))
    c, s = np.cos(thetas), np.sin(thetas)
    mu = c * means[:M] + s * means[M:]
    Vxx, Vxp, Vpx, Vpp = cov[:M, :M], cov[:M, M:], cov[M:, :M], cov[M:, M:]
    sigma = (c[:, :, None] * c[:, None, :] * Vxx + c[:, :, None] * s[:, None, :] * Vxp
             + s[:, :, None] * c[:, None, :] * Vpx + s[:, :, None] * s[:, None, :] * Vpp)
    L = np.linalg.cholesky(sigma)
    xs = mu + np.einsum("nij,nj->ni", L, rng.standard_normal((n_shots, M)))
    return xs, thetas


def hidden_rotation_gaussian_moments(params):
    """PROBLEM-SPECIFIC helper for gaussian_homodyne_shots. Means (xxpp) and covariance of a hidden_rotation pair with the given parameters (sf Gaussian backend)."""
    import strawberryfields as sf

    from cv_relational_learning.datasets import prepare_hidden_rotation_gates

    prog = sf.Program(2)
    with prog.context as q:
        prepare_hidden_rotation_gates(q, *params)
    state = sf.Engine("gaussian").run(prog).state
    return state.means(), state.cov()


def hidden_rotation_shot_features(sampled_x, sampled_thetas, cutoff, angles=SHADOW_ANGLES, chunk=20000,
                                  outside_band="extended"):
    """
    PROBLEM-SPECIFIC (the ASWAP observables O(r) with B = R(r) diagonal, two-mode product shadow elements); the
    generic equivalent is shadow_expval_blocks with hidden_rotation_observables. Single-shot estimates of <O(r)> for a two-mode datapoint: shape (n_shots, len(angles)).

    For a product shadow element, Tr[(B^dag x B) SWAP (rho_0 x rho_1)] = Tr[rho_0 B^dag rho_1 B]. With B = R(r)
    diagonal and rho_m = pi e^{i(a-b) theta_m} f_m[a, b], this is
        pi^2 sum_{j,k} f_0[j,k] f_1[j,k] cos((k - j)(theta_0 - theta_1 + r))
      = pi^2 [ g_0 + 2 sum_{D>0} g_D cos(D (theta_0 - theta_1 + r)) ],   g_D = sum_{k-j=D} f_0[j,k] f_1[j,k],
    i.e. O(cutoff^2) per shot instead of a dense (cutoff^2 x cutoff^2) matrix product.
    """
    sampled_x, sampled_thetas = np.asarray(sampled_x), np.asarray(sampled_thetas)
    angles = np.asarray(angles, dtype=float)
    deltas = np.arange(cutoff)
    weights = np.where(deltas == 0, 1.0, 2.0)
    out = np.empty((len(sampled_x), len(angles)))
    for start in range(0, len(sampled_x), chunk):
        sl = slice(start, start + chunk)
        prod = pattern_functions(sampled_x[sl, 0], cutoff, outside_band) * pattern_functions(sampled_x[sl, 1], cutoff, outside_band)
        g = np.stack([np.diagonal(prod, offset=D, axis1=1, axis2=2).sum(-1) for D in deltas], axis=1)  # (n, cutoff)
        phase = (sampled_thetas[sl, 0] - sampled_thetas[sl, 1])[:, None, None] + angles[None, None, :]
        out[sl] = np.pi**2 * np.einsum("nd,ndr->nr", g * weights, np.cos(deltas[None, :, None] * phase))
    return out


def hidden_rotation_features(xs, thetas, cutoff, angles=SHADOW_ANGLES, progress=True, outside_band="extended"):
    """PROBLEM-SPECIFIC, see hidden_rotation_shot_features. Features (n_datapoints, len(angles)): mean single-shot estimates; xs, thetas of shape (n_datapoints, n_shots, 2)."""
    iterator = tqdm(range(len(xs))) if progress else range(len(xs))
    return np.array([
        hidden_rotation_shot_features(xs[i], thetas[i], cutoff, angles, outside_band=outside_band).mean(0)
        for i in iterator
    ])
