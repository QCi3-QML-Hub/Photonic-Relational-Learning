# Adaptive relational learning on multi-instance quantum data

Code for

> M. Jastrzebski, S. Yu, R. B. Patel, O. Kyriienko,
> *Adaptive Relational Learning on Multi-instance Quantum Data with Photonic Processors*,
> [arXiv:2609.17352](https://arxiv.org/abs/2609.17352).

The models apply shallow trainable transformations locally to each of n input states and then measure a
global SWAP (n = 2) or CYCLE (n > 2) test, i.e. an n-state Bargmann invariant. They are benchmarked on
continuous-variable photonic tasks against models with local observables and against a "measure-first"
baseline built on continuous-variable classical shadows.

## Repository layout

```
src/cv_relational_learning/   the package
    fock.py            truncated Fock-space operators, gates and observables (torch)
    circuits.py        the trainable circuits of Figs. 3, 5, 7, 8
    sf_simulation.py   shot-sampled SWAP test with StrawberryFields
    gradients.py       parameter-shift rules
    models.py          torch wrappers (exact, parameter-shift, shadow baseline)
    datasets.py        the three multi-instance datasets
    shadows.py         continuous-variable classical shadows
    training.py        training / evaluation loop
    experiments.py     the paper's hyperparameters and experiment runners
    analysis.py        accuracies and figures
examples/              scripts reproducing Figs. 3, 5, 7, 8 and regenerating data
notebooks/             tutorials
data/                  the datasets used in the paper (see data/README.md)
results/               stored outputs of the trained models behind the paper's figures
tests/                 tests (AI-generated, see tests/README.md)
```

## Installation

```bash
conda env create -f environment.yml
conda activate cvrl
pip install -e ".[dev]"
```

## Reproducing the figures

Each script plots from the stored results in `results/` by default. With `--retrain` it retrains into
`outputs/` (not version controlled) and plots those results instead. Figures are written to `figures/`.

| Figure | Task | Script | Retraining time* |
|---|---|---|---|
| 3 | hidden relationship (pairs) | `python examples/fig3_hidden_rotation.py [--retrain]` | ~13 min |
| 5 | chirality (triplets, CYCLE test) | `python examples/fig5_chirality.py [--retrain]` | ~10 min, ~3 GB RAM |
| 7 | sensing with a shared nuisance | `python examples/fig7_sensing.py [--retrain]` | ~3 min |
| 8 | shot scaling vs classical shadows | `python examples/fig8_shot_scaling.py [--retrain-shadow] [--retrain-adaptive --workers 8]` | shadow classifiers ~8 min; adaptive ~40 min |

\* On a laptop CPU. Most of the time goes to the local-observable baselines.

Training is deterministic for Figs. 3, 5, 7 and for the Fig. 8 shadow baseline, and the simulated shot
noise of the Fig. 8 adaptive model is seeded, so retraining reproduces the stored predictions (checked for
Figs. 3, 5, 7 and the shadow baseline by `pytest -m slow`).

`examples/make_datasets.py` and `examples/make_shadow_features.py` regenerate the data with fixed seeds. The stored
shadow features were made with `make_shadow_features.py --fast` (~10 min), which uses two problem-specific speedups
for this task; the default, generic pipeline is statistically equivalent but takes many hours at the paper's scale.

## Tutorials

| Notebook | Topic |
|---|---|
| `01_cv_swap_test` | The continuous-variable SWAP test |
| `02_multi_instance_datasets` | The three datasets in phase space (Figs. 2 and 4) |
| `03_matrix_emulation_and_gradients` | Matrix emulation vs StrawberryFields, shot noise, parameter-shift rules |

## Reproducibility notes

- Accuracies are best-threshold accuracies: the decision threshold is the one that maximises accuracy on
  the evaluated (test) set.
- The stored models are those of the final training epoch (no early stopping or model selection).
- Batches are not shuffled, and the batch size equals the number of training points per class.
- Fig. 8 setup used by the code: both models have 20 + 20 training and 20 + 20 test points. The adaptive
  model is trained for 50 epochs. Every adaptive realisation is an independent run from θ₀ = 0 with seeded shot noise. Each shadow realisation is the mean accuracy of 10 linear classifiers
  (seeds 0–9), each trained for 2000 epochs. Shadow features use rotations r ∈ {0.2, 0.6, 1.0, 1.4, 1.8}.
- Shadow features: every datapoint is a single pair measured with 250,000 shots; realisation k of budget n uses
  shots [k·n, (k+1)·n) (see `data/README.md`).

## Citation

```bibtex
@article{jastrzebski2026adaptive,
  title   = {Adaptive Relational Learning on Multi-instance Quantum Data with Photonic Processors},
  author  = {Jastrzebski, Marcin and Yu, Shang and Patel, Raj B. and Kyriienko, Oleksandr},
  journal = {arXiv preprint arXiv:2609.17352},
  year    = {2026}
}
```
