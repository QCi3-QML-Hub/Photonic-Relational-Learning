"""
Accuracy evaluation and the figures of the paper.

Accuracies are best-threshold accuracies: the decision threshold on the model output is the one
(among the ROC thresholds) that maximises accuracy on the evaluated set.
"""

from decimal import Decimal
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve

from cv_relational_learning.paths import REPO_ROOT


def best_threshold(preds, targets):
    """Returns (threshold, accuracy) maximising accuracy of `preds >= threshold`."""
    preds, targets = np.asarray(preds), np.asarray(targets)
    _, _, thresholds = roc_curve(targets, preds)
    best_acc, best_thresh = 0, 0
    for thresh in thresholds:
        acc = accuracy_score(targets, (preds >= thresh).astype(int))
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh
    return best_thresh, best_acc


def load_run(run_dir):
    """Stored outputs of one trained model (see training.py)."""
    run_dir = Path(run_dir)
    return {name: np.load(run_dir / f"{name}.npy") for name in ("train_losses", "val_losses", "preds", "targets")}


def plot_prediction_comparison(runs, titles, out_path=None, seed=0):
    """
    Figs. 3, 5 and 7: one panel per model with test-set outputs per true class, coloured by whether
    they are correctly classified at the best threshold (dashed line). `runs` are dicts from load_run.
    """
    rng = np.random.default_rng(seed)  # vertical jitter only
    fig, axes = plt.subplots(len(runs), 1, figsize=(5, 2.5 * len(runs)), sharey=True)
    axes = np.atleast_1d(axes)
    spacing = 0.04

    for ax, run, title in zip(axes, runs, titles):
        preds, targets = run["preds"], run["targets"]
        thresh, acc = best_threshold(preds, targets)
        is_correct = (preds >= thresh).astype(int) == targets

        for class_idx in (0, 1):
            mask = targets == class_idx
            y = class_idx + rng.uniform(-spacing, spacing, mask.sum())
            correct_handle = ax.scatter(preds[mask][is_correct[mask]], y[is_correct[mask]], c="blue", alpha=0.6, s=30)
            incorrect_handle = ax.scatter(preds[mask][~is_correct[mask]], y[~is_correct[mask]], c="red", alpha=0.6, s=30)
        threshold_line = ax.axvline(x=thresh, linestyle="--", linewidth=1.5, color="black", alpha=0.7)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(
            0.5, 0.50, f"Accuracy = {acc:.3f}", transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5),
        )
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Class 0", "Class 1"])
        ax.set_ylim(-0.4, 1.4)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("True label")
        ax.grid(True, alpha=0.3)

    fig.legend(
        handles=[correct_handle, incorrect_handle, threshold_line],
        labels=["Correct", "Misclassified", "Best threshold"],
        loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=True, fontsize=9,
    )
    fig.tight_layout()
    return _save(fig, out_path)


def plot_accuracy_vs_shots(shot_budgets, adaptive_accs, shadow_accs, out_path=None):
    """
    Fig. 8: test accuracy against shots per estimate. `adaptive_accs[n]` / `shadow_accs[n]` are lists
    of accuracies (one per realisation; may be empty). A linear fit is drawn through the shadow means.
    """
    fig, ax = plt.subplots()
    first = {"Adaptive model": True, "Shadow model": True}

    def scatter(n, accs, colour, label):
        if len(accs) == 0:
            return
        ax.scatter([n] * len(accs), accs, c=colour, alpha=0.5, s=20, label=label if first[label] else None)
        first[label] = False

    for n in shot_budgets:
        scatter(n, adaptive_accs.get(n, []), "r", "Adaptive model")
        scatter(n, shadow_accs.get(n, []), "b", "Shadow model")

    fit_x = [n for n in shot_budgets if len(shadow_accs.get(n, [])) > 0]
    fit_y = [np.mean(shadow_accs[n]) for n in fit_x]
    if len(fit_x) >= 2:
        slope, intercept = np.polyfit(fit_x, fit_y, 1)
        ax.plot(
            fit_x, slope * np.array(fit_x) + intercept, c="b", linestyle="--", alpha=0.7,
            label=f"Fit to shadow model mean:\ny = {Decimal(str(slope)):.1e}x + {Decimal(str(intercept)):.1e}",
        )

    ax.set_xlabel("Number of shots")
    ax.set_ylabel("Accuracy")
    ax.legend()
    return _save(fig, out_path)


def plot_losses(runs, labels, out_path=None):
    """Training (solid) and validation (dashed, at evaluation epochs) losses of several runs."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run, label in zip(runs, labels):
        train = run["train_losses"]
        mask = np.isfinite(train)
        line, = ax.plot(np.arange(len(train))[mask], train[mask], label=f"{label} (train)")
        val = run["val_losses"]
        val_epochs = np.linspace(0, len(train) - 1, len(val))
        ax.plot(val_epochs, val, linestyle="--", color=line.get_color(), label=f"{label} (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.6)
    fig.tight_layout()
    return _save(fig, out_path)


def _save(fig, out_path):
    """Save (if out_path is given) and close the figure; returns it for display, e.g. in notebooks."""
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        shown = out_path.resolve().relative_to(REPO_ROOT) if out_path.resolve().is_relative_to(REPO_ROOT) else out_path.name
        print(f"Saved {shown}")
    plt.close(fig)
    return fig
