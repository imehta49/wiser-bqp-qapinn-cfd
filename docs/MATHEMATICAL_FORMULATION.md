# Mathematical Formulation

## Burgers equation

\[
u_t+u u_x=\nu u_{xx},
\qquad x\in[-1,1],\quad t\in[0,1].
\]

The prescribed conditions are

\[
u(x,0)=-\sin(\pi x),
\qquad u(-1,t)=u(1,t)=0.
\]

## Hard-constrained PINN

Let `N_theta(x,t)` be the unconstrained neural-network output. Define

\[
\hat u_\theta(x,t)
=-\sin(\pi x)+t(1-x^2)N_\theta(x,t).
\]

Then

\[
\hat u_\theta(x,0)=-\sin(\pi x)
\]

and

\[
\hat u_\theta(\pm1,t)=0
\]

for every value of the trainable parameters.

The optimized residual is

\[
R_\theta
=\partial_t\hat u_\theta
+\hat u_\theta\partial_x\hat u_\theta
-\nu\partial_x^2\hat u_\theta,
\]

with objective

\[
\mathcal L_{\mathrm{PDE}}
=\frac{1}{N_r}\sum_{i=1}^{N_r}|R_\theta(x_i,t_i)|^2.
\]

Initial- and boundary-condition losses are retained only as numerical diagnostics.
