# Section 02 — Parameter-Matched Classical PINN

## Purpose

Construct a smaller classical network with a trainable-parameter count comparable to the QAPINN. This prevents an unfair comparison between a large classical model and a much smaller hybrid model.

## Required controls

- same PDE and hard constraints;
- same saved collocation points;
- same seeds and evaluation grid;
- same precision where practical;
- documented optimizer and wall-clock budgets.

## Planned first architecture

A starting point is a three-hidden-layer network with width 32. Its exact parameter count should be reported and adjusted after the QAPINN architecture is fixed.
