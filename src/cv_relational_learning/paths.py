"""
Repository locations, resolved relative to this file so that scripts and notebooks
work from any working directory (requires an editable install: `pip install -e .`).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"   # stored results behind the paper's figures
OUTPUTS_DIR = REPO_ROOT / "outputs"   # results of re-running experiments (not version controlled)
FIGURES_DIR = REPO_ROOT / "figures"   # generated figures (not version controlled)
