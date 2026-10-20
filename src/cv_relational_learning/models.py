"""
torch.nn.Module wrappers around circuits (see circuits.py) and the classical-shadow baseline.
"""

import torch
from torch import nn

from cv_relational_learning.gradients import ParamShiftFunction


def _as_param(init_params):
    if init_params is None:
        return nn.Parameter(torch.tensor([0.0], dtype=torch.double))
    return nn.Parameter(torch.as_tensor(init_params, dtype=torch.double).clone())


class MatrixModel(nn.Module):
    """
    Exact (noise-free) model; gradients flow through the circuit with torch autograd.
    (formerly MatrixBasedModel)
    """

    def __init__(self, circuit_cls, circuit_args, init_params=None):
        super().__init__()
        self.params = _as_param(init_params)
        self.circuit = circuit_cls(*circuit_args)

    def forward(self, x):
        return torch.stack([self.circuit.forward(self.params, xi) for xi in x]).view(-1)


class ParamShiftModel(nn.Module):
    """
    Model whose circuit is evaluated from (simulated) shots, so gradients come from the
    circuit's parameter-shift rule rather than autograd. (formerly OpticsParamShiftModel)
    """

    def __init__(self, circuit_cls, circuit_args, init_params=None):
        super().__init__()
        self.params = _as_param(init_params)
        self.circuit = circuit_cls(*circuit_args)

    def forward(self, x):
        outputs = [ParamShiftFunction.apply(self.params, xi, self.circuit) for xi in x]
        return torch.stack(outputs).view(-1)


class ShadowModel(nn.Module):
    """
    "Measure-first" baseline: a classical network acting on classical-shadow estimates of
    observables, precomputed for each datapoint (see shadows.py). (formerly CSModel)
    """

    def __init__(self, nn_architecture):
        super().__init__()
        self.model = nn_architecture

    def forward(self, x):
        return self.model(x).view(-1)
