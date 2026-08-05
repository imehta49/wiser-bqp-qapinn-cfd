# Reproducibility Guide

## 1. Record the environment

Use Python 3.10 or 3.11 and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

The experiment automatically writes Python, NumPy, SciPy, pandas, PyTorch, platform, device, and GPU information to `environment.json`.

## 2. Run the smoke test

```bash
python scripts/run_burgers_cpinn.py --profile smoke --seed 0 --output-root outputs
```

Confirm that:

- the reference checks pass;
- the hard initial-condition error is below `1e-10`;
- the hard boundary-condition error is below `1e-10`;
- the run produces figures, metrics, checkpoints, and a ZIP archive.

## 3. Run the smooth baseline

```bash
python scripts/run_burgers_cpinn.py --profile baseline --seed 0 --output-root outputs
```

Repeat without changing any other setting:

```bash
python scripts/run_burgers_cpinn.py --profile baseline --seed 1 --output-root outputs
python scripts/run_burgers_cpinn.py --profile baseline --seed 2 --output-root outputs
```

## 4. Aggregate the seeds

```python
import pandas as pd

frame = pd.read_csv("outputs/experiment_summary.csv")
selected = frame[
    (frame["profile"] == "baseline")
    & (frame["viscosity_label"] == "nu_0p1pi")
]
print(selected[[
    "seed",
    "relative_l2_error",
    "pde_residual_rms",
    "total_training_seconds",
    "trainable_parameters",
]])
print(selected[[
    "relative_l2_error",
    "pde_residual_rms",
    "total_training_seconds",
]].agg(["mean", "std", "min", "max"]))
```

## 5. Preserve fair comparisons

For parameter-matched classical, Fourier-feature, and QAPINN studies, reuse the saved collocation points and evaluation grid. Do not silently change the sampling distribution, number of training points, random seed, error definition, or optimizer budget.

## 6. Large artefacts

The `.gitignore` excludes generated `*.pt`, `*.npy`, `*.npz`, and ZIP files. Publish large artefacts through a GitHub Release, Git LFS, or an archival repository rather than adding them to ordinary Git history.
