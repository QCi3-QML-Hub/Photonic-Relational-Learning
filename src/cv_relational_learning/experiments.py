"""
The numerical experiments of the paper: hyperparameters and runners.

Figs. 3, 5, 7 compare a relational model (SWAP / CYCLE test) with a local-observable model on the
three tasks. Fig. 8 compares the adaptive SWAP model under finite shots with a linear classifier on
classical-shadow features (the "measure-first" baseline).
"""

import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cv_relational_learning.circuits import (
    ChiralityCycleCircuit,
    ChiralityLocalCircuit,
    RotationLocalCircuit,
    RotationSwapCircuit,
    RotationSwapShotCircuit,
    SensingLocalCircuit,
    SensingSwapCircuit,
)
from cv_relational_learning.datasets import load_dataset, load_shadow_features
from cv_relational_learning.fock import CycleObservable, NumberObservable
from cv_relational_learning.models import MatrixModel, ParamShiftModel, ShadowModel
from cv_relational_learning.training import load_model, predict, test_model, train_model


@dataclass
class ModelSpec:
    circuit: type
    observable: callable  # cutoff -> observable matrix
    init_params: list


@dataclass
class Experiment:
    dataset: str
    cutoff: int
    n_train: int  # per class; also the batch size
    n_val: int    # per class
    n_test: int   # per class
    n_epochs: int
    eval_interval: int
    lr: float
    models: dict = field(default_factory=dict)


RELATIONAL_EXPERIMENTS = {
    "fig3": Experiment(
        dataset="hidden_rotation", cutoff=25, n_train=10, n_val=2, n_test=50,
        n_epochs=50, eval_interval=2, lr=0.1,
        models={
            "SWAP": ModelSpec(RotationSwapCircuit, lambda c: CycleObservable(c, 2), [0.0]),
            "NUMBER": ModelSpec(RotationLocalCircuit, lambda c: NumberObservable(c, 2), [0.0, 0.0]),
        },
    ),
    "fig5": Experiment(
        dataset="chirality", cutoff=20, n_train=10, n_val=1, n_test=50,
        n_epochs=20, eval_interval=2, lr=0.04,
        models={
            "CYCLE": ModelSpec(ChiralityCycleCircuit, lambda c: CycleObservable(c, 3), [0.0]),
            "NUMBER": ModelSpec(ChiralityLocalCircuit, lambda c: NumberObservable(c, 3), [0.0, 0.0]),
        },
    ),
    "fig7": Experiment(
        dataset="sensing", cutoff=20, n_train=10, n_val=10, n_test=50,
        n_epochs=50, eval_interval=2, lr=0.015,
        models={
            "SWAP": ModelSpec(SensingSwapCircuit, lambda c: CycleObservable(c, 2), [0.0]),
            # NOTE: an all-zero initialisation gives zero gradients for this circuit
            "NUMBER": ModelSpec(SensingLocalCircuit, lambda c: NumberObservable(c, 2), [0.1, 0.1, 0.1]),
        },
    ),
}


def run_relational_experiment(exp, model_name, out_dir, verbose=True):
    """Train one model of a Fig. 3/5/7 experiment and store losses, model and test predictions in out_dir."""
    spec = exp.models[model_name]
    train, val, test = load_dataset(exp.dataset, exp.cutoff, exp.n_train, exp.n_val, exp.n_test)
    model = MatrixModel(spec.circuit, (exp.cutoff, spec.observable(exp.cutoff)), spec.init_params)
    train_model(model, train, val, exp.n_epochs, exp.eval_interval, exp.lr, exp.n_train, out_dir, verbose)
    return test_model(model, test, out_dir)


# ---------------------------------------------------------------------------
# Fig. 8
# ---------------------------------------------------------------------------

SHOT_BUDGETS = (100, 500, 1000, 5000, 10000, 20000, 40000, 50000)
N_REALISATIONS = 5

# Adaptive SWAP model with simulated shot noise on the hidden_rotation task
ADAPTIVE = Experiment(
    dataset="hidden_rotation", cutoff=25, n_train=20, n_val=5, n_test=20,
    n_epochs=50, eval_interval=2, lr=0.1,
    models={"SWAP": ModelSpec(RotationSwapShotCircuit, lambda c: CycleObservable(c, 2), [0.0])},
)

# Linear classifier on the shadow features; several seeds per realisation, accuracies averaged
SHADOW = dict(n_epochs=2000, eval_interval=50, lr=1e-3, batch_size=40, seeds=range(10), n_features=5)


def adaptive_seed(n_shots, realisation):
    """Seed of the simulated shot noise of one adaptive training run."""
    return 10 * n_shots + realisation


def run_adaptive_shot_experiment(n_shots, out_dir, seed=None, verbose=False):
    """
    Train the adaptive SWAP model from theta_0 = 0 and test it, with `n_shots` per SWAP-test estimate
    during both training (every parameter-shifted evaluation) and inference.
    """
    if seed is not None:
        torch.manual_seed(seed)
    exp, spec = ADAPTIVE, ADAPTIVE.models["SWAP"]
    train, val, test = load_dataset(exp.dataset, exp.cutoff, exp.n_train, exp.n_val, exp.n_test)
    model = ParamShiftModel(spec.circuit, (exp.cutoff, spec.observable(exp.cutoff), n_shots), spec.init_params)
    train_model(model, train, val, exp.n_epochs, exp.eval_interval, exp.lr, exp.n_train, out_dir, verbose)
    return test_model(model, test, out_dir)


def _adaptive_task(task):
    n_shots, realisation, out_dir, single_thread = task
    if single_thread:
        torch.set_num_threads(1)
    run_adaptive_shot_experiment(n_shots, out_dir, seed=adaptive_seed(n_shots, realisation))
    return n_shots, realisation


def run_adaptive_shot_experiments(budgets, out_dir, n_workers=1):
    """
    N_REALISATIONS independent, seeded training runs per shot budget, stored in out_dir/{n}shots/stat{i}.
    Runs are independent, so they can be spread over `n_workers` processes.
    """
    tasks = [(n, i, Path(out_dir) / f"{n}shots" / f"stat{i}", n_workers > 1)
             for n in budgets for i in range(N_REALISATIONS)]
    if n_workers == 1:
        for task in tasks:
            print("adaptive model: {} shots, realisation {} done".format(*_adaptive_task(task)))
        return
    with ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn")) as pool:
        for n, i in pool.map(_adaptive_task, tasks):
            print(f"adaptive model: {n} shots, realisation {i} done")


def run_shadow_experiment(n_shots, fold, out_dir, verbose=False):
    """
    Train the shadow baseline on the features of (n_shots, fold) once per seed; returns test-set
    probabilities of shape (n_seeds, n_test). Only the last seed's model is kept in out_dir.
    """
    cfg = SHADOW
    preds = []
    for seed in cfg["seeds"]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = ShadowModel(nn.Linear(cfg["n_features"], 1).double())
        train, val, test = load_shadow_features(n_shots, fold)
        train_model(model, train, val, cfg["n_epochs"], cfg["eval_interval"], cfg["lr"], cfg["batch_size"],
                    out_dir, verbose)
        model = load_model(model, out_dir)
        preds.append(torch.sigmoid(predict(model, test)).numpy())
    return np.asarray(preds)
