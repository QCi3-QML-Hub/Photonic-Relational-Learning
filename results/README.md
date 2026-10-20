# Stored results

Outputs of the trained models behind the paper's figures, as produced by the original research code.
The `examples/fig*.py` scripts plot from here by default.

| Path | Contents |
|---|---|
| `fig3/{SWAP,NUMBER}/` | hidden-relationship models (Fig. 3) |
| `fig5/{CYCLE,NUMBER}/` | chirality models (Fig. 5) |
| `fig7/{SWAP,NUMBER}/` | sensing models (Fig. 7) |
| `fig8/adaptive/{n}shots/stat{i}/` | adaptive SWAP model trained and tested with n shots per estimate, realisations i = 0..4 |
| `fig8/shadow/{n}shots_fold{k}.npy` | shadow-baseline test probabilities, shape (10 seeds, 40 test points), realisation k |

Each model directory has `model_state.pth` (the trained parameters), `preds.npy` / `targets.npy` (test-set
outputs and labels, class 1 first), and `train_losses.npy` / `val_losses.npy`.

The original trainer recorded losses only at evaluation epochs and saved the loss files before adding the
current epoch. As a result the stored loss arrays hold `inf` at non-evaluation epochs and at the final
epoch. Re-running with this package records every epoch. The models are unaffected: retraining
Figs. 3, 5 and 7 with this package reproduces the stored `model_state.pth` and `preds.npy` (to ≤ 1e-9).

Fig. 8 adaptive runs (`fig8/adaptive`) were produced with this package: each realisation is an
independent training run from θ₀ = 0 with seeded shot noise (`experiments.adaptive_seed`), so
`examples/fig8_shot_scaling.py --retrain-adaptive` reproduces them exactly. (In the first version of the
results, realisations 1–4 of each budget had been warm-started from the previous realisation's
parameters through an in-place update of the initial-parameter tensor.)
