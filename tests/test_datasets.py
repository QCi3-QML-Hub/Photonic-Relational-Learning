"""AI-generated (Claude, Anthropic): tests for cv_relational_learning.datasets."""

import numpy as np
import torch

from cv_relational_learning.datasets import (
    load_dataset,
    load_sensing_interactions,
    load_shadow_features,
    make_chirality_dataset,
    make_hidden_rotation_dataset,
    make_sensing_dataset,
    sensing_interaction,
)


def test_paper_splits_have_expected_shapes_and_labels():
    train, val, test = load_dataset("hidden_rotation", 25, 10, 2, 50)
    assert (len(train), len(val), len(test)) == (20, 4, 100)
    assert train.data.shape[1:] == (25, 25)
    # class 1 first, then class 0
    assert train.targets.tolist() == [1.0] * 10 + [0.0] * 10

    train, _, _ = load_dataset("chirality", 20, 10, 1, 50)
    assert train.data.shape[1:] == (20, 20, 20)


def test_splits_do_not_overlap():
    train, val, test = load_dataset("hidden_rotation", 25, 10, 2, 50)
    all_points = torch.view_as_real(torch.cat([train.data, val.data, test.data])).reshape(124, -1)
    assert len(torch.unique(all_points, dim=0)) == 124


def test_sensing_interactions_are_unitary_and_class_dependent():
    class_1, class_0 = load_sensing_interactions(10, n_per_class=2)
    for U in class_1 + class_0:
        U = U.numpy()
        assert np.allclose(U.conj().T @ U, np.eye(100), atol=1e-10)
    # same parameters, different class -> different squeezing direction
    assert not torch.allclose(sensing_interaction(np.float32(0.7), np.float32(1.0), 1, 10),
                              sensing_interaction(np.float32(0.7), np.float32(1.0), 0, 10))


def test_shadow_features_files():
    train, val, test = load_shadow_features(500, 0)
    assert train.data.shape == (40, 5) and val.data.shape == (10, 5) and test.data.shape == (40, 5)
    assert test.targets.tolist() == [1.0] * 20 + [0.0] * 20


def test_generation_is_seeded(tmp_path):
    a = np.load(make_hidden_rotation_dataset(2, cutoff=6, seed=3, data_dir=tmp_path / "a"))
    b = np.load(make_hidden_rotation_dataset(2, cutoff=6, seed=3, data_dir=tmp_path / "b"))
    assert np.array_equal(a["class_1"], b["class_1"]) and np.array_equal(a["class_0"], b["class_0"])

    f = np.load(make_chirality_dataset(1, cutoff=4, seed=0, data_dir=tmp_path))
    assert f["class_1"].shape == (1, 4, 4, 4)

    f = np.load(make_sensing_dataset(3, seed=0, data_dir=tmp_path))
    assert f["squeezing_1"].dtype == np.float32 and np.all((f["squeezing_1"] >= 0.5) & (f["squeezing_1"] < 1))
