"""
Fig. 5: chirality task (three-state triplets). CYCLE-test (Bargmann invariant) relational model vs
a model with a tritter-like interferometer measuring a local photon-number observable.

    python examples/fig5_chirality.py             # plot the stored paper results (seconds)
    python examples/fig5_chirality.py --retrain   # retrain both models first (~10 minutes, ~3 GB RAM)
"""

import argparse

from cv_relational_learning.analysis import best_threshold, load_run, plot_prediction_comparison
from cv_relational_learning.experiments import RELATIONAL_EXPERIMENTS, run_relational_experiment
from cv_relational_learning.paths import FIGURES_DIR, OUTPUTS_DIR, RESULTS_DIR

FIG = "fig5"
MODELS = ["CYCLE", "NUMBER"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrain", action="store_true", help=f"retrain into outputs/{FIG} and plot those")
    args = parser.parse_args()

    results_dir = (OUTPUTS_DIR if args.retrain else RESULTS_DIR) / FIG
    if args.retrain:
        for model in MODELS:
            run_relational_experiment(RELATIONAL_EXPERIMENTS[FIG], model, results_dir / model)

    runs = [load_run(results_dir / model) for model in MODELS]
    for model, run in zip(MODELS, runs):
        print(f"{model:>6}: test accuracy {best_threshold(run['preds'], run['targets'])[1]:.3f}")
    plot_prediction_comparison(runs, MODELS, FIGURES_DIR / f"{FIG}_predictions.pdf")


if __name__ == "__main__":
    main()
