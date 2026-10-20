"""
Simulate the "measure-first" experiment behind the Fig. 8 shadow baseline.

1. Draw fresh hidden-rotation pairs (train/val/test = 20/5/20 per class, seeded per split).
2. Sample N_REALISATIONS * max(budgets) random-quadrature homodyne shots of every pair.
3. Estimate the ASWAP features O(r), r in SHADOW_ANGLES, from the classical shadows.
4. Realisation k of budget n uses shots [k*n, (k+1)*n) of the SAME pair, so realisations differ only in
   measurement statistics.

Generic pipeline (default): one strawberryfields run per shot and features Tr(O rho_shadow) from shadow
states reconstructed per block of shots (shadows.shadow_expval_blocks). Works for any state and observable but
is slow at the paper's scale (~1-2 ms per shot for sampling, i.e. many hours for 250,000 shots x 90 pairs).

--fast: PROBLEM-SPECIFIC speedups, exact for this task only: shots drawn from each pair's Gaussian moments
(shadows.gaussian_homodyne_shots) and O(cutoff^2) single-shot features (shadows.hidden_rotation_shot_features).
About 10 minutes in total. The stored data/shadow_features were generated with --fast --overwrite-paper-data.

Output: {data_dir}/shadow_features/{n}shots_fold{k}.npz and states.npz (pair parameters and seeds).
By default it writes to outputs/data/ so the stored features are not overwritten.

    python examples/make_shadow_features.py --budgets 100 200 --seed 0   # generic, small
    python examples/make_shadow_features.py --fast                       # -> outputs/data/shadow_features
"""

import argparse

import numpy as np
from tqdm import tqdm

from cv_relational_learning.experiments import N_REALISATIONS, SHOT_BUDGETS
from cv_relational_learning.paths import DATA_DIR, OUTPUTS_DIR
from cv_relational_learning.shadows import (
    SHADOW_ANGLES,
    hidden_rotation_observables,
    hidden_rotation_shot_features,
    sample_hidden_rotation_shots,
    shadow_expval_blocks,
)

CUTOFF = 25
SPLITS = {"train": 20, "val": 5, "test": 20}  # pairs per class


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(SHOT_BUDGETS))
    parser.add_argument("--seed", type=int, default=0, help="split i uses seed + i")
    parser.add_argument("--fast", action="store_true", help="problem-specific speedups (exact for this task only)")
    parser.add_argument("--overwrite-paper-data", action="store_true", help="write into data/ instead of outputs/data/")
    args = parser.parse_args()

    out_dir = (DATA_DIR if args.overwrite_paper_data else OUTPUTS_DIR / "data") / "shadow_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    total_shots = max(args.budgets) * N_REALISATIONS
    block = int(np.gcd.reduce(args.budgets))  # generic path: shadow estimates per block of shots
    observables = None if args.fast else hidden_rotation_observables(CUTOFF)

    features = {(n, k): {} for n in args.budgets for k in range(N_REALISATIONS)}
    states = {"seeds": [], "n_shots": total_shots, "cutoff": CUTOFF, "angles": np.asarray(SHADOW_ANGLES)}
    for i, (split, n_per_class) in enumerate(SPLITS.items()):
        seed = args.seed + i
        xs, thetas, labels, params = sample_hidden_rotation_shots(
            n_per_class, total_shots, seed=seed, return_params=True, method="gaussian_moments" if args.fast else "sf"
        )
        states["seeds"].append(seed)
        states[f"{split}_params"], states[f"{split}_labels"] = params, labels
        per_window = {key: [] for key in features}
        for x, t in tqdm(zip(xs, thetas), total=len(xs), desc=f"{split} features"):
            if args.fast:
                per_shot = hidden_rotation_shot_features(x, t, CUTOFF)
                window_mean = lambda lo, hi: per_shot[lo:hi].mean(0)
            else:
                blocks = shadow_expval_blocks(x, t, observables, 2, CUTOFF, block)
                window_mean = lambda lo, hi: blocks[lo // block:hi // block].mean(0)
            for n, k in features:
                per_window[(n, k)].append(window_mean(k * n, (k + 1) * n))
        for key in features:
            features[key][f"{split}_features"] = np.array(per_window[key])
            features[key][f"{split}_labels"] = labels.astype(np.float64)

    for (n, k), arrays in features.items():
        np.savez(out_dir / f"{n}shots_fold{k}.npz", **arrays)
    np.savez(out_dir / "states.npz", **states)
    print(f"wrote {len(features)} feature files and states.npz to {out_dir}")


if __name__ == "__main__":
    main()
