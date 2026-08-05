# Project Roadmap

## Phase 1 — Classical foundation

- Validate the smooth Burgers reference solution.
- Train the hard-constrained full classical PINN for seeds 0, 1, and 2.
- Aggregate mean and standard deviation.
- Train a parameter-matched smaller classical network.
- Train a classical Fourier-feature PINN.

## Phase 2 — Quantum-assisted PINNs

- Implement 3-, 4-, and 5-qubit variational quantum layers.
- Compare angle encoding and data re-uploading.
- Compare no-entanglement, chain, and ring circuits.
- Compare local expectation values, correlation observables, and probability-vector outputs.

## Phase 3 — Explainability

- Fourier-spectrum analysis of hidden features and predicted fields.
- Gradient-norm and gradient-variance analysis.
- Neural tangent-kernel spectra and effective rank.
- Loss-landscape sections around trained solutions.
- Entanglement entropy and entangler-removal ablations.

## Phase 4 — Additional PDE and CFD benchmark

- Heat-equation benchmark with an analytical solution.
- Low-viscosity Burgers equation.
- Reduced two-dimensional projectile-flow problem.
- Aerodynamic observables and held-out Mach/angle generalization.

## Phase 5 — Submission

- Reproducibility audit from a clean environment.
- Technical report.
- Presentation slides.
- Summary of findings and recommendations.
- Team-contribution and AI-tool-use statements.
