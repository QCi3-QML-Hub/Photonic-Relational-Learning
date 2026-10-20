"""
Hardware-style gradients for circuits whose outputs are estimated from shots, so torch
autograd cannot be used. Gradients come from parameter-shift rules and are fed back into
torch through a custom autograd.Function.
"""

import numpy as np
import torch


def parameter_shift(expval_circuit, params, param_idx, circuit_args=(), circuit_kwargs={}):
    """
    Two-term shift rule (Schuld et al., "Evaluating analytic gradients on quantum hardware").
    Only exact for generators with two eigenvalue gaps, so it is an approximation for Rgates
    on Fock states; see `generalised_parameter_shift`.
    """
    params_minus = params.copy()
    params_minus[param_idx] -= np.pi / 2
    expval_minus = expval_circuit(params_minus, *circuit_args, **circuit_kwargs)

    params_plus = params.copy()
    params_plus[param_idx] += np.pi / 2
    expval_plus = expval_circuit(params_plus, *circuit_args, **circuit_kwargs)

    return 0.5 * (expval_plus - expval_minus)


def generalised_parameter_shift(
    expval_circuit, params, param_idx, circuit_args=(), circuit_kwargs={}, n_frequencies=25
):
    """
    Equation 24 of Wierichs et al., "General parameter-shift rules for quantum gradients" (2022),
    for an expectation value that is a trigonometric polynomial with integer frequencies
    1..n_frequencies in the parameter. For an Rgate in a cutoff-d Fock space, d frequencies suffice.
    Costs 2 * n_frequencies circuit evaluations per parameter.
    """
    R = n_frequencies
    evals = []
    for i in range(1, 2 * R + 1):
        eval_params = params.copy()
        eval_params[param_idx] = params[param_idx] + (2 * i - 1) * np.pi / (2 * R)

        sin_term = np.sin((2 * i - 1) * np.pi / (4 * R)) ** 2
        coeff = ((-1) ** (i - 1)) / (4 * R * sin_term)

        evals.append(expval_circuit(eval_params, *circuit_args, **circuit_kwargs) * coeff)

    return np.sum(evals)


class ParamShiftFunction(torch.autograd.Function):
    """
    Wraps any circuit exposing `forward(params, x)` and `grad(params, x)` so it can be trained
    with torch optimisers. Gate/ansatz agnostic; mimics how an experiment would be trained.
    """

    @staticmethod
    def forward(ctx, params, x, circuit):
        params_np = params.detach().cpu().numpy()
        x_cpu = x.detach().cpu().numpy()

        output = circuit.forward(params_np, x_cpu)
        ctx.circuit = circuit
        ctx.params = params_np
        ctx.x = x_cpu

        return torch.as_tensor(output, device=params.device)

    @staticmethod
    def backward(ctx, grad_output):
        grad_params = ctx.circuit.grad(ctx.params, ctx.x)
        grad_params = torch.as_tensor(grad_params, device=grad_output.device)
        return grad_output * grad_params, None, None
