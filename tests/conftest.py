"""AI-generated (Claude, Anthropic): shared pytest configuration."""

import numpy as np
import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: re-runs training; deselected unless `-m slow` is given")


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m"):
        return
    skip_slow = pytest.mark.skip(reason="slow; run with `pytest -m slow`")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def random_ket(d, rng, n_occupied=None):
    """Random normalised single-mode ket with support on the lowest n_occupied Fock states."""
    n_occupied = n_occupied or d
    v = np.zeros(d, dtype=np.complex128)
    v[:n_occupied] = rng.normal(size=n_occupied) + 1j * rng.normal(size=n_occupied)
    return torch.from_numpy(v / np.linalg.norm(v))


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
