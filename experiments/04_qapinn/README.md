# Section 04 — Quantum-Assisted PINNs

## Planned architecture

\[
(x,t)
\rightarrow \text{classical embedding}
\rightarrow \text{variational quantum circuit}
\rightarrow \text{classical decoder}
\rightarrow \hat u(x,t).
\]

## Controlled variants

- 3, 4, and 5 qubits;
- angle encoding and data re-uploading;
- no entanglement, chain entanglement, and ring entanglement;
- local expectation values, correlation observables, and probability-vector outputs;
- shallow and deeper circuits.

No QAPINN result should be reported until derivatives with respect to both inputs and circuit parameters have passed numerical tests.
