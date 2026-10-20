"""
Photonic simulation with StrawberryFields: the destructive CV SWAP test estimated from
photon-counting samples, as it would be run on hardware.

The SWAP test follows Volkoff & Subasi, "Ancilla-free continuous-variable SWAP test"
(arXiv:2202.09923): a 50:50 beamsplitter maps SWAP onto the photon-number parity of one output
mode, so each shot contributes (-1)^(n_0); shots with n_0 + n_1 > 2*cutoff contribute 0 (eq. 15). In
simulation that never happens, as sf returns at most cutoff - 1 photons per mode.
"""

import numpy as np
import strawberryfields as sf
from strawberryfields import ops

from cv_relational_learning.gradients import generalised_parameter_shift


def swap_test_shot(state, eng):
    """
    One shot of the destructive SWAP test on a two-mode ket `state` of shape (cutoff, cutoff).
    Returns the photon counts (n_0, n_1).
    NOTE: the beamsplitter in arXiv:2202.09923 is the dagger of sf's BSgate(pi/4, 0).
    """
    prog = sf.Program(2)
    with prog.context as q:
        ops.Ket(state) | (q[0], q[1])
        ops.BSgate(np.pi / 4, 0) | (q[0], q[1])
        ops.MeasureFock() | q[0]
        ops.MeasureFock() | q[1]
    result = eng.run(prog)
    eng.reset()
    return np.asarray(result.samples).reshape(-1)


def estimate_swap_from_counts(counts, cutoff):
    """
    Eq. 15 of arXiv:2202.09923: the mean of (-1)^(n_0) * Heaviside(2*cutoff - n_0 - n_1),
    with Heaviside(0) = 1. `counts` has shape (n_shots, 2).
    """
    counts = np.asarray(counts)
    saturation = 2 * cutoff - counts[:, 0] - counts[:, 1]
    return np.mean((-1.0) ** counts[:, 0] * np.heaviside(saturation, 1))


def run_swap_test(state, cutoff, shots):
    """Estimate <SWAP> of a two-mode ket from `shots` simulated photon-counting shots."""
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    counts = [swap_test_shot(state, eng) for _ in range(shots)]
    return estimate_swap_from_counts(counts, cutoff)


class SFRotationSwapCircuit:
    """
    StrawberryFields counterpart of circuits.RotationSwapCircuit with a shot-sampled SWAP test,
    trained with the generalised parameter shift through models.ParamShiftModel (2 * cutoff circuit
    evaluations per gradient). (formerly RParamASWAPCirc)
    """

    def __init__(self, cutoff, shots):
        self.cutoff = cutoff
        self.shots = shots
        self.engine = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})

    def forward(self, params, x):
        prog = sf.Program(2)
        r_param = prog.params("r_param")
        with prog.context as q:
            ops.Ket(np.asarray(x)) | q
            ops.Rgate(r_param) | q[1]
        prog.bind_params({"r_param": params[0]})

        state = self.engine.run(prog).state.ket()
        self.engine.reset()
        return run_swap_test(state, self.cutoff, self.shots)

    def grad(self, params, x):
        full_grad = np.zeros_like(params)
        for i in range(len(params)):
            full_grad[i] = generalised_parameter_shift(
                self.forward, params, i, circuit_args=(x,), n_frequencies=self.cutoff
            )
        return full_grad
