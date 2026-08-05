"""Classical PINN baseline for the one-dimensional viscous Burgers equation.

The model uses the hard-constrained output

    u_hat(x, t) = -sin(pi x) + t (1 - x**2) N_theta(x, t),

so the prescribed initial and boundary conditions are satisfied exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy
from scipy.interpolate import interp1d
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.stats import qmc
import torch
from torch import nn


@dataclass(frozen=True)
class BurgersConfig:
    """Configuration for one reproducible Burgers PINN experiment."""

    profile: str
    viscosity_label: str
    nu: float
    seed: int
    output_root: str = "outputs"

    # Reference solution
    nx_reference: int = 512
    nt_reference: int = 201
    nx_internal: int = 2049
    t_max: float = 1.0

    # Collocation points
    n_r_bank: int = 20_000
    n_r_batch: int = 4_096
    n_ic: int = 512
    n_bc: int = 512

    # Neural network
    width: int = 64
    hidden_layers: int = 5
    dtype: str = "float64"

    # Optimisation
    adam_epochs: int = 10_000
    adam_lr: float = 1.0e-3
    adam_min_lr: float = 1.0e-5
    lbfgs_iterations: int = 300
    lbfgs_points: int = 5_000
    lbfgs_history_size: int = 50
    grad_clip: float = 100.0

    # Logging and evaluation
    log_every: int = 200
    checkpoint_every: int = 2_000
    prediction_batch: int = 16_384
    pde_evaluation_points: int = 10_000


def make_config(profile: str, seed: int, output_root: str) -> BurgersConfig:
    """Return a controlled experiment profile.

    ``baseline`` reproduces the validated smooth-viscosity experiment.
    ``main`` is the more difficult low-viscosity case and should only be run
    after the baseline has been reproduced across multiple seeds.
    """

    base = BurgersConfig(
        profile=profile,
        viscosity_label="nu_0p1pi",
        nu=0.1 / math.pi,
        seed=seed,
        output_root=output_root,
    )

    if profile == "smoke":
        return replace(
            base,
            n_r_bank=2_000,
            n_r_batch=512,
            width=32,
            hidden_layers=3,
            adam_epochs=300,
            lbfgs_iterations=20,
            lbfgs_points=1_000,
            log_every=50,
            checkpoint_every=300,
            nx_reference=192,
            nt_reference=81,
            nx_internal=513,
            pde_evaluation_points=1_000,
        )

    if profile == "baseline":
        return base

    if profile == "main":
        return replace(
            base,
            viscosity_label="nu_0p01pi",
            nu=0.01 / math.pi,
            n_r_bank=50_000,
            n_r_batch=8_192,
            width=96,
            hidden_layers=6,
            adam_epochs=15_000,
            adam_lr=5.0e-4,
            adam_min_lr=1.0e-6,
            lbfgs_iterations=800,
            lbfgs_points=20_000,
            log_every=250,
            checkpoint_every=2_500,
            pde_evaluation_points=30_000,
        )

    raise ValueError(f"Unknown profile {profile!r}; choose smoke, baseline, or main.")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}.")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def process_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0**2)


def environment_metadata(device: torch.device) -> Dict[str, object]:
    return {
        "created_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


# ---------------------------------------------------------------------------
# Cole-Hopf reference solution
# ---------------------------------------------------------------------------


def _neumann_laplacian(nx: int, dx: float):
    matrix = lil_matrix((nx, nx), dtype=np.float64)
    for index in range(1, nx - 1):
        matrix[index, index - 1] = 1.0
        matrix[index, index] = -2.0
        matrix[index, index + 1] = 1.0

    matrix[0, 0] = -2.0
    matrix[0, 1] = 2.0
    matrix[-1, -2] = 2.0
    matrix[-1, -1] = -2.0
    return matrix.tocsr() / (dx * dx)


def _first_derivative_fourth_order(values: np.ndarray, dx: float) -> np.ndarray:
    derivative = np.zeros_like(values)
    derivative[:, 2:-2] = (
        values[:, :-4]
        - 8.0 * values[:, 1:-3]
        + 8.0 * values[:, 3:-1]
        - values[:, 4:]
    ) / (12.0 * dx)
    derivative[:, 1] = (values[:, 2] - values[:, 0]) / (2.0 * dx)
    derivative[:, -2] = (values[:, -1] - values[:, -3]) / (2.0 * dx)
    derivative[:, 0] = 0.0
    derivative[:, -1] = 0.0
    return derivative


def generate_reference_solution(
    nu: float,
    nx_output: int = 512,
    nt: int = 201,
    nx_internal: int = 2049,
    t_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a high-resolution reference using the Cole-Hopf transform."""

    if nu <= 0.0:
        raise ValueError("Viscosity must be positive.")
    if nx_internal < nx_output:
        raise ValueError("nx_internal must be greater than or equal to nx_output.")

    x_internal = np.linspace(-1.0, 1.0, nx_internal, dtype=np.float64)
    times = np.linspace(0.0, t_max, nt, dtype=np.float64)
    dx = x_internal[1] - x_internal[0]

    # This scaled form is numerically stable and differs only by a multiplicative
    # constant, which cancels in u = -2 nu phi_x / phi.
    phi_initial = np.exp(
        -(np.cos(np.pi * x_internal) + 1.0) / (2.0 * np.pi * nu)
    )

    laplacian = _neumann_laplacian(nx_internal, dx)
    phi = expm_multiply(
        nu * laplacian,
        phi_initial,
        start=0.0,
        stop=t_max,
        num=nt,
        endpoint=True,
    )
    phi = np.maximum(phi, np.finfo(np.float64).tiny)
    phi_x = _first_derivative_fourth_order(phi, dx)
    solution_internal = -2.0 * nu * phi_x / phi

    solution_internal[:, 0] = 0.0
    solution_internal[:, -1] = 0.0
    solution_internal[0, :] = -np.sin(np.pi * x_internal)

    x_output = np.linspace(-1.0, 1.0, nx_output, dtype=np.float64)
    interpolator = interp1d(
        x_internal,
        solution_internal,
        axis=1,
        kind="cubic",
        bounds_error=True,
        assume_sorted=True,
    )
    solution = np.asarray(interpolator(x_output), dtype=np.float64)
    solution[:, 0] = 0.0
    solution[:, -1] = 0.0
    solution[0, :] = -np.sin(np.pi * x_output)
    return x_output, times, solution


def reference_quality_checks(x: np.ndarray, solution: np.ndarray) -> Dict[str, float]:
    return {
        "initial_condition_max_abs_error": float(
            np.max(np.abs(solution[0] + np.sin(np.pi * x)))
        ),
        "boundary_condition_max_abs_error": float(
            max(np.max(np.abs(solution[:, 0])), np.max(np.abs(solution[:, -1])))
        ),
        "antisymmetry_max_abs_error": float(
            np.max(np.abs(solution + solution[:, ::-1]))
        ),
    }


# ---------------------------------------------------------------------------
# Sampling and model
# ---------------------------------------------------------------------------


def make_collocation_points(config: BurgersConfig) -> Dict[str, np.ndarray]:
    """Create one reproducible Latin-hypercube collocation bank."""

    sampler = qmc.LatinHypercube(d=2, seed=config.seed)
    unit = sampler.random(n=config.n_r_bank)
    x_r = -1.0 + 2.0 * unit[:, 0:1]
    t_r = config.t_max * unit[:, 1:2]

    x_ic = np.linspace(-1.0, 1.0, config.n_ic, dtype=np.float64)[:, None]
    t_ic = np.zeros_like(x_ic)
    u_ic = -np.sin(np.pi * x_ic)

    t_bc = np.linspace(0.0, config.t_max, config.n_bc, dtype=np.float64)[:, None]
    return {
        "x_r": x_r.astype(np.float64),
        "t_r": t_r.astype(np.float64),
        "x_ic": x_ic,
        "t_ic": t_ic,
        "u_ic": u_ic,
        "x_left": -np.ones_like(t_bc),
        "t_left": t_bc,
        "u_left": np.zeros_like(t_bc),
        "x_right": np.ones_like(t_bc),
        "t_right": t_bc.copy(),
        "u_right": np.zeros_like(t_bc),
    }


class HardConstrainedBurgersPINN(nn.Module):
    """Fully connected PINN with exact initial and boundary conditions."""

    def __init__(self, width: int = 64, hidden_layers: int = 5) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        input_features = 2
        for _ in range(hidden_layers):
            linear = nn.Linear(input_features, width)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.Tanh()])
            input_features = width

        output = nn.Linear(input_features, 1)
        nn.init.xavier_normal_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

        self.register_buffer("lower", torch.tensor([[-1.0, 0.0]]))
        self.register_buffer("upper", torch.tensor([[1.0, 1.0]]))

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        x = coordinates[:, 0:1]
        t = coordinates[:, 1:2]
        scaled = 2.0 * (coordinates - self.lower) / (self.upper - self.lower) - 1.0
        raw = self.network(scaled)
        return -torch.sin(torch.pi * x) + t * (1.0 - x**2) * raw


def verify_hard_constraints(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    tolerance: float = 1.0e-10,
) -> Dict[str, float]:
    model.eval()
    x = torch.linspace(-1.0, 1.0, 501, device=device, dtype=dtype)[:, None]
    t_zero = torch.zeros_like(x)
    with torch.no_grad():
        initial_prediction = model(torch.cat([x, t_zero], dim=1))
        initial_exact = -torch.sin(torch.pi * x)

    times = torch.linspace(0.0, 1.0, 501, device=device, dtype=dtype)[:, None]
    with torch.no_grad():
        left = model(torch.cat([-torch.ones_like(times), times], dim=1))
        right = model(torch.cat([torch.ones_like(times), times], dim=1))

    initial_error = float(torch.max(torch.abs(initial_prediction - initial_exact)).cpu())
    boundary_error = float(max(torch.max(torch.abs(left)), torch.max(torch.abs(right))).cpu())
    if initial_error > tolerance or boundary_error > tolerance:
        raise RuntimeError(
            "Hard-constraint test failed: "
            f"IC error={initial_error:.3e}, BC error={boundary_error:.3e}."
        )
    return {
        "initial_condition_max_abs_error": initial_error,
        "boundary_condition_max_abs_error": boundary_error,
    }


def _as_tensor(
    array: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
) -> torch.Tensor:
    return torch.tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def burgers_residual(
    model: nn.Module,
    coordinates: torch.Tensor,
    nu: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coordinates = coordinates.clone().detach().requires_grad_(True)
    solution = model(coordinates)
    first = torch.autograd.grad(
        solution,
        coordinates,
        grad_outputs=torch.ones_like(solution),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_x = first[:, 0:1]
    u_t = first[:, 1:2]
    second = torch.autograd.grad(
        u_x,
        coordinates,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_xx = second[:, 0:1]
    residual = u_t + solution * u_x - float(nu) * u_xx
    return residual, solution, u_x, u_t, u_xx


def diagnostic_constraint_losses(
    model: nn.Module,
    tensors: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    initial_prediction = model(torch.cat([tensors["x_ic"], tensors["t_ic"]], dim=1))
    initial_loss = torch.mean((initial_prediction - tensors["u_ic"]) ** 2)
    left = model(torch.cat([tensors["x_left"], tensors["t_left"]], dim=1))
    right = model(torch.cat([tensors["x_right"], tensors["t_right"]], dim=1))
    boundary_loss = 0.5 * (torch.mean(left**2) + torch.mean(right**2))
    return initial_loss, boundary_loss


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def train_model(
    config: BurgersConfig,
    model: nn.Module,
    points: Dict[str, np.ndarray],
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    tensors = {key: _as_tensor(value, device, dtype) for key, value in points.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config.adam_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.adam_epochs, 1),
        eta_min=config.adam_min_lr,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 12_345)
    n_bank = tensors["x_r"].shape[0]
    history: List[Dict[str, object]] = []

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    model.train()
    for epoch in range(1, config.adam_epochs + 1):
        if config.n_r_batch >= n_bank:
            indices_cpu = torch.arange(n_bank)
        else:
            indices_cpu = torch.randperm(n_bank, generator=generator)[: config.n_r_batch]
        indices = indices_cpu.to(device=device)
        coordinates = torch.cat(
            [tensors["x_r"][indices], tensors["t_r"][indices]], dim=1
        )

        optimizer.zero_grad(set_to_none=True)
        residual, _, _, _, _ = burgers_residual(model, coordinates, config.nu)
        pde_loss = torch.mean(residual**2)
        pde_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        if epoch == 1 or epoch % config.log_every == 0 or epoch == config.adam_epochs:
            with torch.no_grad():
                initial_loss, boundary_loss = diagnostic_constraint_losses(model, tensors)
            row = {
                "stage": "adam",
                "step": epoch,
                "pde_loss": float(pde_loss.detach().cpu()),
                "ic_loss": float(initial_loss.detach().cpu()),
                "bc_loss": float(boundary_loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - start,
                "rss_memory_mb": process_memory_mb(),
            }
            history.append(row)
            print(
                f"[Adam {epoch:6d}/{config.adam_epochs}] "
                f"PDE={row['pde_loss']:.4e} IC={row['ic_loss']:.2e} "
                f"BC={row['bc_loss']:.2e} |g|={row['gradient_norm']:.3e}"
            )

        if epoch % config.checkpoint_every == 0 or epoch == config.adam_epochs:
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(config),
                },
                checkpoint_dir / f"adam_epoch_{epoch:06d}.pt",
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    adam_seconds = time.perf_counter() - start

    # Fixed random subset for deterministic L-BFGS refinement.
    lbfgs_seconds = 0.0
    if config.lbfgs_iterations > 0:
        lbfgs_generator = torch.Generator(device="cpu")
        lbfgs_generator.manual_seed(config.seed + 54_321)
        count = min(config.lbfgs_points, n_bank)
        indices = torch.randperm(n_bank, generator=lbfgs_generator)[:count].to(device)
        coordinates = torch.cat(
            [tensors["x_r"][indices], tensors["t_r"][indices]], dim=1
        )
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=config.lbfgs_iterations,
            max_eval=max(int(config.lbfgs_iterations * 1.25), config.lbfgs_iterations + 1),
            history_size=config.lbfgs_history_size,
            tolerance_grad=1.0e-9,
            tolerance_change=1.0e-12,
            line_search_fn="strong_wolfe",
        )
        closure_calls = 0
        lbfgs_start = time.perf_counter()

        def closure() -> torch.Tensor:
            nonlocal closure_calls
            optimizer_lbfgs.zero_grad(set_to_none=True)
            residual, _, _, _, _ = burgers_residual(model, coordinates, config.nu)
            loss = torch.mean(residual**2)
            loss.backward()
            closure_calls += 1
            if closure_calls == 1 or closure_calls % 25 == 0:
                history.append(
                    {
                        "stage": "lbfgs",
                        "step": closure_calls,
                        "pde_loss": float(loss.detach().cpu()),
                        "ic_loss": 0.0,
                        "bc_loss": 0.0,
                        "gradient_norm": float("nan"),
                        "learning_rate": 1.0,
                        "elapsed_seconds": adam_seconds + time.perf_counter() - lbfgs_start,
                        "rss_memory_mb": process_memory_mb(),
                    }
                )
                print(f"[L-BFGS closure {closure_calls:5d}] PDE={float(loss.detach().cpu()):.4e}")
            return loss

        model.train()
        optimizer_lbfgs.step(closure)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        lbfgs_seconds = time.perf_counter() - lbfgs_start

    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
        if device.type == "cuda"
        else 0.0
    )
    return history, {
        "adam_seconds": adam_seconds,
        "lbfgs_seconds": lbfgs_seconds,
        "total_training_seconds": adam_seconds + lbfgs_seconds,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "final_rss_memory_mb": process_memory_mb(),
    }


def predict_grid(
    model: nn.Module,
    x: np.ndarray,
    times: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> np.ndarray:
    xx, tt = np.meshgrid(x, times)
    points = np.column_stack([xx.ravel(), tt.ravel()])
    outputs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, points.shape[0], batch_size):
            batch = _as_tensor(points[start : start + batch_size], device, dtype)
            outputs.append(model(batch).detach().cpu().numpy())
    return np.vstack(outputs).reshape(times.size, x.size)


def evaluate_pde_rms(
    model: nn.Module,
    nu: float,
    n_points: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 2_000,
) -> float:
    sampler = qmc.LatinHypercube(d=2, seed=seed + 777)
    unit = sampler.random(n=n_points)
    points = np.column_stack([-1.0 + 2.0 * unit[:, 0], unit[:, 1]])
    squared_sum = 0.0
    count = 0
    model.eval()
    for start in range(0, n_points, batch_size):
        batch = _as_tensor(points[start : start + batch_size], device, dtype)
        residual, _, _, _, _ = burgers_residual(model, batch, nu)
        squared_sum += float(torch.sum(residual.detach() ** 2).cpu())
        count += residual.numel()
    return math.sqrt(squared_sum / max(count, 1))


def _save_field_plot(
    x: np.ndarray,
    times: np.ndarray,
    field: np.ndarray,
    title: str,
    colorbar_label: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(
        field,
        extent=[float(x.min()), float(x.max()), float(times.min()), float(times.max())],
        origin="lower",
        aspect="auto",
    )
    axis.set_xlabel("x")
    axis.set_ylabel("t")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _save_time_slices(
    x: np.ndarray,
    times: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 7))
    for target in [0.0, 0.25, 0.5, 0.75, 1.0]:
        index = int(np.argmin(np.abs(times - target)))
        axis.plot(x, reference[index], label=f"Reference t={times[index]:.2f}")
        axis.plot(x, prediction[index], "--", label=f"PINN t={times[index]:.2f}")
    axis.set_xlabel("x")
    axis.set_ylabel("u(x,t)")
    axis.set_title("Reference and classical PINN time slices")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _save_training_plots(history: List[Dict[str, object]], figure_dir: Path) -> None:
    if not history:
        return
    frame = pd.DataFrame(history)
    adam = frame[frame["stage"] == "adam"]

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.semilogy(adam["step"], adam["pde_loss"], label="PDE loss")
    axis.semilogy(adam["step"], np.maximum(adam["ic_loss"], 1.0e-30), label="IC diagnostic")
    axis.semilogy(adam["step"], np.maximum(adam["bc_loss"], 1.0e-30), label="BC diagnostic")
    axis.set_xlabel("Adam epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training losses")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "05_training_losses.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.semilogy(adam["step"], np.maximum(adam["gradient_norm"], 1.0e-30))
    axis.set_xlabel("Adam epoch")
    axis.set_ylabel("Parameter-gradient L2 norm")
    axis.set_title("Gradient norm during Adam training")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "06_gradient_norm.png", dpi=200)
    plt.close(figure)


def run_experiment(config: BurgersConfig) -> Dict[str, object]:
    """Run a complete classical Burgers PINN experiment and save all artefacts."""

    set_global_seed(config.seed)
    device = get_device()
    dtype = get_dtype(config.dtype)
    torch.set_default_dtype(dtype)

    output_root = Path(config.output_root).resolve()
    run_name = f"{config.profile}_{config.viscosity_label}_seed_{config.seed}"
    run_dir = output_root / "runs" / run_name
    data_dir = run_dir / "data"
    figure_dir = run_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    (run_dir / "environment.json").write_text(
        json.dumps(environment_metadata(device), indent=2)
    )

    print("=" * 80)
    print("WISER/BQP classical Burgers PINN")
    print(f"Profile={config.profile} seed={config.seed} nu={config.nu:.8e}")
    print(f"Device={device} dtype={dtype}")
    print(f"Output={run_dir}")
    print("=" * 80)

    reference_start = time.perf_counter()
    x, times, reference = generate_reference_solution(
        config.nu,
        nx_output=config.nx_reference,
        nt=config.nt_reference,
        nx_internal=config.nx_internal,
        t_max=config.t_max,
    )
    reference_seconds = time.perf_counter() - reference_start
    reference_checks = reference_quality_checks(x, reference)
    np.savez_compressed(
        data_dir / "reference_solution.npz",
        x=x,
        t=times,
        u=reference,
        nu=np.array(config.nu),
    )

    points = make_collocation_points(config)
    np.savez_compressed(data_dir / "collocation_points.npz", **points)

    model = HardConstrainedBurgersPINN(config.width, config.hidden_layers).to(
        device=device, dtype=dtype
    )
    constraint_checks = verify_hard_constraints(model, device, dtype)
    parameter_count = count_trainable_parameters(model)
    print(f"Trainable parameters: {parameter_count}")
    print(f"Constraint checks: {constraint_checks}")

    history, timing = train_model(config, model, points, run_dir, device, dtype)
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    prediction = predict_grid(
        model, x, times, device, dtype, batch_size=config.prediction_batch
    )
    absolute_error = np.abs(prediction - reference)
    relative_l2 = float(
        np.linalg.norm((prediction - reference).ravel())
        / np.linalg.norm(reference.ravel())
    )
    pde_rms = evaluate_pde_rms(
        model,
        config.nu,
        config.pde_evaluation_points,
        config.seed,
        device,
        dtype,
    )

    # Final diagnostic losses.
    tensor_points = {key: _as_tensor(value, device, dtype) for key, value in points.items()}
    with torch.no_grad():
        final_ic, final_bc = diagnostic_constraint_losses(model, tensor_points)

    # Approximate final training residual on the deterministic L-BFGS subset.
    count = min(config.lbfgs_points, points["x_r"].shape[0])
    coordinates = np.column_stack([points["x_r"][:count], points["t_r"][:count]])
    final_residual, _, _, _, _ = burgers_residual(
        model, _as_tensor(coordinates, device, dtype), config.nu
    )
    final_pde_loss = float(torch.mean(final_residual**2).detach().cpu())

    metrics: Dict[str, object] = {
        "run_name": run_name,
        "created_utc": utc_now(),
        "profile": config.profile,
        "seed": config.seed,
        "nu": config.nu,
        "viscosity_label": config.viscosity_label,
        "relative_l2_error": relative_l2,
        "mean_absolute_error": float(np.mean(absolute_error)),
        "maximum_absolute_error": float(np.max(absolute_error)),
        "pde_residual_rms": pde_rms,
        "final_pde_loss": final_pde_loss,
        "final_ic_loss": float(final_ic.cpu()),
        "final_bc_loss": float(final_bc.cpu()),
        "trainable_parameters": parameter_count,
        "reference_generation_seconds": reference_seconds,
        **timing,
        "device": str(device),
        "dtype": config.dtype,
        "reference_checks": reference_checks,
        "constraint_checks": constraint_checks,
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame([{key: value for key, value in metrics.items() if not isinstance(value, dict)}]).to_csv(
        run_dir / "metrics.csv", index=False
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "trainable_parameters": parameter_count,
        },
        run_dir / "model_final.pt",
    )
    np.savez_compressed(
        run_dir / "evaluation_fields.npz",
        x=x,
        t=times,
        reference=reference,
        prediction=prediction,
        absolute_error=absolute_error,
    )

    _save_field_plot(x, times, reference, "Reference Burgers solution", "u", figure_dir / "01_reference_solution.png")
    _save_field_plot(x, times, prediction, "Classical PINN prediction", "u", figure_dir / "02_pinn_prediction.png")
    _save_field_plot(x, times, absolute_error, "Absolute pointwise error", "|u_PINN-u_ref|", figure_dir / "03_absolute_error.png")
    _save_time_slices(x, times, reference, prediction, figure_dir / "04_time_slices.png")
    _save_training_plots(history, figure_dir)

    summary_path = output_root / "experiment_summary.csv"
    summary_row = pd.DataFrame(
        [{key: value for key, value in metrics.items() if not isinstance(value, dict)}]
    )
    if summary_path.exists():
        summary_row = pd.concat([pd.read_csv(summary_path), summary_row], ignore_index=True)
    summary_row.to_csv(summary_path, index=False)

    bundle_path = run_dir.parent / f"{run_name}_reproducibility_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(run_dir.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(run_dir.parent))

    print(json.dumps(metrics, indent=2))
    print(f"Reproducibility bundle: {bundle_path}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "baseline", "main"], default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", default="outputs")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    config = make_config(arguments.profile, arguments.seed, arguments.output_root)
    run_experiment(config)


if __name__ == "__main__":
    main()
