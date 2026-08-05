# Data Directory

This directory is reserved for reproducible input data.

- `reference/`: self-generated analytical or numerical reference solutions.
- `collocation/`: saved PINN collocation points used for fair model comparisons.

Generated `*.npy`, `*.npz`, HDF5, CFD mesh, and large field files are excluded from ordinary Git tracking. Store large public artefacts through a release, Git LFS, or an archival data repository.
