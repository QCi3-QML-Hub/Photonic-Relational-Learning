"""
Multi-instance quantum datasets of the paper, and their torch Dataset wrappers.

Every dataset has two classes; the files list class 1 first, then class 0.

    hidden_rotation  (Sec. III.1, Figs. 2-3, 8)  pairs of identical squeezed-displaced states; in
                     class 1 the second state is rotated by a fixed r_hidden = 1 relative to the
                     first, in class 0 by a random angle outside [0.5, 1.5].
    chirality        (Sec. III.2, Figs. 4-5)     coherent-state triangles with a hidden squeeze on
                     the third state; class 1 clockwise, class 0 counter-clockwise.
    sensing          (Sec. III.3, Figs. 6-7)     two-mode interactions: a squeezing signal on copy 1 with
                     a real (class 1) or imaginary (class 0) squeezing parameter, i.e. squeezing axes
                     45 degrees apart, and an unknown rotation shared by both copies.

The hidden_rotation and chirality states are prepared with the strawberryfields Fock backend and
stored as kets. Sensing datapoints are stored as their generating parameters and the
(cutoff^2 x cutoff^2) interaction matrices are rebuilt on load.
"""

import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from cv_relational_learning.fock import identity, rotation, squeezing
from cv_relational_learning.paths import DATA_DIR

HIDDEN_ROTATION = 1.0


# ---------------------------------------------------------------------------
# torch Datasets
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """
    Labelled multi-instance datapoints: a two/three-mode ket or a two-mode interaction matrix.
    (formerly SiamData)
    """

    def __init__(self, class_1, class_0):
        data = [torch.as_tensor(np.asarray(x)).cdouble() for x in list(class_1) + list(class_0)]
        self.data = torch.stack(data) if data else torch.empty(0, dtype=torch.cdouble)
        self.targets = torch.tensor([1] * len(class_1) + [0] * len(class_0)).double()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class ShadowFeatureDataset(Dataset):
    """Classical-shadow expectation values (features) per datapoint. (formerly CSData)"""

    def __init__(self, features, labels):
        self.data = torch.as_tensor(features).double()
        self.targets = torch.as_tensor(labels).double()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _split(class_1, class_0, n_train, n_val, n_test):
    """Sizes are per class; consecutive slices [train | val | test] of each class."""
    bounds = [(0, n_train), (n_train, n_train + n_val), (n_train + n_val, n_train + n_val + n_test)]
    return tuple(PairDataset(class_1[a:b], class_0[a:b]) for a, b in bounds)


def load_dataset(name, cutoff, n_train, n_val, n_test, data_dir=DATA_DIR):
    """
    Returns (train, val, test) PairDatasets for `name` in {"hidden_rotation", "chirality", "sensing"}.
    Sizes are per class.
    """
    if name == "sensing":
        class_1, class_0 = load_sensing_interactions(cutoff, n_train + n_val + n_test, data_dir)
    else:
        f = np.load(data_dir / name / f"cutoff{cutoff}.npz")
        class_1, class_0 = f["class_1"], f["class_0"]
    return _split(class_1, class_0, n_train, n_val, n_test)


def load_shadow_features(n_shots, fold, data_dir=DATA_DIR):
    """(train, val, test) ShadowFeatureDatasets for the Fig. 8 shadow baseline."""
    f = np.load(data_dir / "shadow_features" / f"{n_shots}shots_fold{fold}.npz")
    return tuple(ShadowFeatureDataset(f[f"{s}_features"], f[f"{s}_labels"]) for s in ("train", "val", "test"))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _sf_ket(n_modes, cutoff, gates):
    """Prepare a pure state with the sf Fock backend. `gates(q)` applies ops to the registers."""
    import strawberryfields as sf

    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    prog = sf.Program(n_modes)
    with prog.context as q:
        gates(q)
    state = eng.run(prog).state
    return state.ket(), state.trace()


def _report_traces(traces, cutoff):
    traces = np.asarray(traces)
    if traces.min() < 0.95:
        warnings.warn(f"{np.sum(traces < 0.95)} states have trace < 0.95 at cutoff {cutoff}")
    print(f"State traces at cutoff {cutoff}: min={traces.min():.4f}, mean={traces.mean():.4f}")


def sample_relative_rotation(label):
    """
    Relative rotation between the two states of a hidden_rotation pair.
    Class 1: exactly HIDDEN_ROTATION. Class 0: with probability 1/2 uniform in [0, 0.5], otherwise
    uniform in [1.5, 2 pi), i.e. outside the window HIDDEN_ROTATION +- 0.5.
    """
    if label == 1:
        return HIDDEN_ROTATION
    rot_small = np.random.uniform(0, 0.5)
    rot_big = np.random.uniform(1.5, 2 * np.pi)
    return rot_big if np.random.uniform(0, 2 * np.pi - 1) >= 2.64 else rot_small


def sample_hidden_rotation_params(label):
    """Squeezing r, squeezing angle, displacement, and the rotations of the two modes."""
    squeeze = np.random.rand() * 1                # 0 to 1
    squeeze_angle = np.random.rand() * np.pi      # 0 to pi
    common_rotation = np.random.rand() * 2 * np.pi
    displacement = np.random.uniform(0.5, 2)
    return squeeze, squeeze_angle, displacement, common_rotation, common_rotation + sample_relative_rotation(label)


def prepare_hidden_rotation_gates(q, squeeze, squeeze_angle, displacement, rot_0, rot_1):
    from strawberryfields import ops

    for mode, rot in ((0, rot_0), (1, rot_1)):
        ops.Sgate(squeeze, squeeze_angle) | q[mode]
        ops.Dgate(displacement) | q[mode]
        ops.Rgate(rot) | q[mode]


def make_hidden_rotation_dataset(n_per_class, cutoff, seed=None, data_dir=DATA_DIR):
    if seed is not None:
        np.random.seed(seed)
    classes, traces = {}, []
    for label in (1, 0):
        kets = []
        for _ in range(n_per_class):
            params = sample_hidden_rotation_params(label)
            ket, tr = _sf_ket(2, cutoff, lambda q: prepare_hidden_rotation_gates(q, *params))
            kets.append(ket)
            traces.append(tr)
        classes[f"class_{label}"] = np.array(kets)
    _report_traces(traces, cutoff)
    out = data_dir / "hidden_rotation" / f"cutoff{cutoff}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **classes)
    return out


def make_chirality_dataset(n_per_class, cutoff, seed=None, data_dir=DATA_DIR):
    """
    Triplets |a_0>|a_1>|a_2> of coherent states forming a triangle in phase space (shifted and rotated
    globally at random) whose orientation is the class; the third state also carries a fixed hidden
    squeeze of 0.7 that the model has to learn to undo.
    """
    from strawberryfields import ops

    if seed is not None:
        np.random.seed(seed)
    classes, traces = {}, []
    for label, rot_range in ((1, (-np.pi / 2, -0.4)), (0, (0.4, np.pi / 2))):
        kets = []
        for _ in range(n_per_class):
            d = np.random.uniform(0.25, 0.5)
            r = np.random.uniform(*rot_range)
            d_global = np.random.uniform(0, 0.5)
            r_global = np.random.uniform(0, 2 * np.pi)

            def gates(q):
                ops.Dgate(d) | q[1]
                ops.Dgate(2 * d, r) | q[2]
                for mode in range(3):
                    ops.Dgate(d_global) | q[mode]
                for mode in range(3):
                    ops.Rgate(r_global) | q[mode]
                ops.Sgate(0.7) | q[2]

            ket, tr = _sf_ket(3, cutoff, gates)
            kets.append(ket)
            traces.append(tr)
        classes[f"class_{label}"] = np.array(kets)
    _report_traces(traces, cutoff)
    out = data_dir / "chirality" / f"cutoff{cutoff}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **classes)
    return out


SENSING_SQUEEZE_ANGLE = {1: 0.0, 0: np.pi / 2}


def sensing_interaction(squeeze, rot, label, cutoff):
    """
    (R(rot) x R(rot)) (I x S(squeeze, angle)): signal squeezing on copy 1 with a real (label 1) or
    imaginary (label 0) squeezing parameter, then a nuisance rotation shared by both copies.
    `squeeze` and `rot` are float32, as drawn by torch.rand when the paper data was generated, so the
    matrices are rebuilt bit-exactly.
    """
    signal = squeezing(torch.tensor([squeeze], dtype=torch.float32), cutoff, SENSING_SQUEEZE_ANGLE[label])
    noise = rotation(cutoff, torch.tensor([rot], dtype=torch.float32))
    return torch.kron(noise, noise) @ torch.kron(identity(cutoff), signal)


def make_sensing_dataset(n_per_class, seed=None, data_dir=DATA_DIR):
    """Signal squeezing uniform in [0.5, 1), nuisance rotation uniform in [0, 2 pi)."""
    if seed is not None:
        torch.manual_seed(seed)
    params = {}
    for label in (1, 0):
        params[f"squeezing_{label}"] = (torch.rand(n_per_class) * 0.5 + 0.5).numpy()
        params[f"rotation_{label}"] = (torch.rand(n_per_class) * 2 * torch.pi).numpy()
    out = data_dir / "sensing" / "params.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **params)
    return out


def load_sensing_interactions(cutoff, n_per_class=None, data_dir=DATA_DIR):
    """Rebuild the first n_per_class interaction matrices of each class (all if None)."""
    f = np.load(data_dir / "sensing" / "params.npz")
    return tuple(
        [
            sensing_interaction(s, r, label, cutoff)
            for s, r in zip(f[f"squeezing_{label}"][:n_per_class], f[f"rotation_{label}"][:n_per_class])
        ]
        for label in (1, 0)
    )
