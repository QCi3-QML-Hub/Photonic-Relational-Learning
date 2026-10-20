"""
Fig. 8: test accuracy against shots per estimate on the hidden-relationship task, for the adaptive
SWAP model (red; five independent training runs per budget) and the classical-shadow baseline (blue; five
realisations of the shadow measurement statistics of the same states per budget, each the mean accuracy over
10 classifier seeds).

    python examples/fig8_shot_scaling.py                     # plot the stored paper results
    python examples/fig8_shot_scaling.py --retrain-shadow    # retrain the shadow classifiers from the
                                                             # stored shadow features (minutes)
    python examples/fig8_shot_scaling.py --retrain-adaptive --workers 8
                                                             # retrain the adaptive model for every budget
                                                             # and realisation (~1 min per run; use --budgets)

Every adaptive realisation is an independent training run from theta_0 = 0 with seeded shot noise
(experiments.adaptive_seed), so the stored results are reproduced exactly.

Recomputing the shadow features themselves from simulated shots is done by
examples/make_shadow_features.py.
"""

import argparse

import numpy as np

from cv_relational_learning.analysis import best_threshold, plot_accuracy_vs_shots
from cv_relational_learning.datasets import load_shadow_features
from cv_relational_learning.experiments import (
    N_REALISATIONS,
    SHOT_BUDGETS,
    run_adaptive_shot_experiments,
    run_shadow_experiment,
)
from cv_relational_learning.paths import FIGURES_DIR, OUTPUTS_DIR, RESULTS_DIR

SHADOW_BUDGETS = SHOT_BUDGETS


def adaptive_accuracies(results_dir, budgets):
    accs = {}
    for n in budgets:
        accs[n] = []
        for i in range(N_REALISATIONS):
            run_dir = results_dir / f"{n}shots" / f"stat{i}"
            if (run_dir / "preds.npy").exists():
                accs[n].append(best_threshold(np.load(run_dir / "preds.npy"), np.load(run_dir / "targets.npy"))[1])
    return accs


def shadow_accuracies(results_dir, budgets):
    accs = {}
    for n in budgets:
        accs[n] = []
        for fold in range(N_REALISATIONS):
            path = results_dir / f"{n}shots_fold{fold}.npy"
            if path.exists():
                targets = load_shadow_features(n, fold)[2].targets.numpy()
                accs[n].append(np.mean([best_threshold(p, targets)[1] for p in np.load(path)]))
    return accs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrain-shadow", action="store_true", help="retrain into outputs/fig8/shadow")
    parser.add_argument("--retrain-adaptive", action="store_true", help="retrain into outputs/fig8/adaptive")
    parser.add_argument("--workers", type=int, default=1, help="parallel processes for --retrain-adaptive")
    parser.add_argument("--budgets", type=int, nargs="+", default=list(SHOT_BUDGETS), help="shot budgets to retrain")
    args = parser.parse_args()

    adaptive_dir = (OUTPUTS_DIR if args.retrain_adaptive else RESULTS_DIR) / "fig8" / "adaptive"
    shadow_dir = (OUTPUTS_DIR if args.retrain_shadow else RESULTS_DIR) / "fig8" / "shadow"

    if args.retrain_adaptive:
        run_adaptive_shot_experiments(args.budgets, adaptive_dir, args.workers)

    if args.retrain_shadow:
        shadow_dir.mkdir(parents=True, exist_ok=True)
        for n in [b for b in args.budgets if b in SHADOW_BUDGETS]:
            for fold in range(N_REALISATIONS):
                print(f"shadow model: {n} shots, fold {fold}")
                preds = run_shadow_experiment(n, fold, OUTPUTS_DIR / "fig8" / "shadow_scratch")
                np.save(shadow_dir / f"{n}shots_fold{fold}.npy", preds)

    adaptive = adaptive_accuracies(adaptive_dir, SHOT_BUDGETS)
    shadow = shadow_accuracies(shadow_dir, SHADOW_BUDGETS)
    print(f"{'shots':>6}  {'adaptive (mean)':>15}  {'shadow (mean)':>13}")
    for n in SHOT_BUDGETS:
        a = f"{np.mean(adaptive[n]):.3f}" if adaptive.get(n) else "-"
        s = f"{np.mean(shadow[n]):.3f}" if shadow.get(n) else "-"
        print(f"{n:>6}  {a:>15}  {s:>13}")
    plot_accuracy_vs_shots(SHOT_BUDGETS, adaptive, shadow, FIGURES_DIR / "fig8_shot_scaling.pdf")


if __name__ == "__main__":
    main()
