"""
Regenerate the datasets of Figs. 3, 5 and 7 (100 datapoints per class each).

The paper's own datasets are stored in data/ and were drawn without a seed, so regenerated data
(and anything trained on it) will differ from the paper in detail but not in distribution.
By default the new data goes to outputs/data/ so the paper data is never overwritten.

    python examples/make_datasets.py --seed 0
"""

import argparse

from cv_relational_learning.datasets import make_chirality_dataset, make_hidden_rotation_dataset, make_sensing_dataset
from cv_relational_learning.paths import DATA_DIR, OUTPUTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-per-class", type=int, default=100)
    parser.add_argument("--overwrite-paper-data", action="store_true", help="write into data/ instead of outputs/data/")
    args = parser.parse_args()

    data_dir = DATA_DIR if args.overwrite_paper_data else OUTPUTS_DIR / "data"
    print(make_hidden_rotation_dataset(args.n_per_class, cutoff=25, seed=args.seed, data_dir=data_dir))
    print(make_chirality_dataset(args.n_per_class, cutoff=20, seed=args.seed, data_dir=data_dir))
    print(make_sensing_dataset(args.n_per_class, seed=args.seed, data_dir=data_dir))


if __name__ == "__main__":
    main()
