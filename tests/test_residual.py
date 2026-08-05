import math

import torch

from wiser_qapinn.burgers import HardConstrainedBurgersPINN, burgers_residual


def test_burgers_residual_has_expected_shape():
    torch.set_default_dtype(torch.float64)
    model = HardConstrainedBurgersPINN(width=8, hidden_layers=2).to(dtype=torch.float64)
    coordinates = torch.tensor(
        [[-0.5, 0.1], [0.0, 0.5], [0.5, 0.9]],
        dtype=torch.float64,
    )
    residual, solution, u_x, u_t, u_xx = burgers_residual(
        model,
        coordinates,
        nu=0.1 / math.pi,
    )
    for tensor in (residual, solution, u_x, u_t, u_xx):
        assert tensor.shape == (3, 1)
        assert torch.isfinite(tensor).all()
