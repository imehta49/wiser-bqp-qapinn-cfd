# Contributing

## Branches

Use short-lived feature branches such as:

```text
feature/parameter-matched-pinn
feature/qapinn-4qubit
analysis/fourier-spectrum
cfd/projectile-euler
```

## Before opening a pull request

1. Run `pytest -q`.
2. Run the smoke experiment if training code changed.
3. Do not commit secrets or large generated artefacts.
4. Update the relevant experiment README.
5. Record configuration changes and random seeds.
6. Explain whether a change affects fairness against existing baselines.

## Commit messages

Use descriptive messages, for example:

```text
Add hard-constraint unit tests
Add smooth Burgers seeds 1 and 2
Implement four-qubit ring-entangled QAPINN
Add quantum feature Fourier analysis
```
