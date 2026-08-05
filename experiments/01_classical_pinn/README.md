# Section 01 — Classical Burgers PINN

## Objective

Establish a numerically accurate and reproducible classical PINN baseline before introducing a quantum layer.

## Model

\[
\hat u_\theta(x,t)
=-\sin(\pi x)+t(1-x^2)N_\theta(x,t).
\]

The transformation imposes the initial and boundary conditions exactly. Training minimizes only the Burgers-equation residual.

## Commands

```bash
python scripts/run_burgers_cpinn.py --profile smoke --seed 0 --output-root outputs
python scripts/run_burgers_cpinn.py --profile baseline --seed 0 --output-root outputs
python scripts/run_burgers_cpinn.py --profile baseline --seed 1 --output-root outputs
python scripts/run_burgers_cpinn.py --profile baseline --seed 2 --output-root outputs
```

## Completion criteria

- exact-condition checks below `1e-10`;
- stable results across three seeds;
- saved reference, prediction, error, time-slice, loss, and gradient figures;
- aggregated relative \(L_2\), PDE residual, runtime, memory, and parameter statistics.
