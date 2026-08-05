# Section 07 — Reduced Projectile CFD

## Scope

A reduced, openly reproducible CFD problem will connect the PINN/QAPINN study to projectile aerodynamics without attempting the original full three-dimensional multi-fidelity database.

## Initial target

\[
(x,y,M_\infty,\alpha)
\longrightarrow
(\rho,u,v,p),
\]

followed by integrated aerodynamic observables such as axial force, normal force, and pitching moment.

## Development sequence

1. simplified geometry;
2. subsonic two-dimensional Euler benchmark;
3. held-out Mach/angle generalization;
4. optional supersonic or multi-fidelity extension.
