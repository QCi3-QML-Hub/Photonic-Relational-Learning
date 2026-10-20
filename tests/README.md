# Tests

**These tests were written by an AI assistant (Claude, Anthropic)**, not by the paper's authors.
They check the physics conventions and the code paths behind the paper's results. They are not
part of the scientific contribution.

```bash
pip install -e ".[dev]"
pytest                    # fast checks (~2-3 minutes)
pytest -m slow            # re-run training and compare against the stored paper results (~40 minutes, ~3 GB RAM)
```

| File | What is checked |
|---|---|
| `test_fock.py` | Gate unitarity, SWAP / CYCLE / ASWAP expectation values on product states, observable objects against dense matrices, agreement of gates with strawberryfields (including the squeezing sign convention) |
| `test_gradients.py` | Two-term and generalised parameter-shift rules against exact derivatives |
| `test_circuits.py` | Trained paper models separate the classes; shot-noise circuit is unbiased |
| `test_datasets.py` | Data splits and labels, sensing interactions rebuilt from parameters, seeded generation |
| `test_shadows.py` | Oscillator functions (incl. the exact irregular solution), the Gaussian homodyne sampler, shadow state reconstruction, expectation values, and the problem-specific speedups against the generic pipeline |
| `test_analysis.py` | Best-threshold accuracy |
| `test_reproduction.py` (slow) | Retraining reproduces the stored predictions of Figs. 3, 5, 7 and of the Fig. 8 shadow baseline |
