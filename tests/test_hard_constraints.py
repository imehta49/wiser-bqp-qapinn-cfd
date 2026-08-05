import torch

from wiser_qapinn.burgers import HardConstrainedBurgersPINN, verify_hard_constraints


def test_hard_constraints_are_exact():
    torch.set_default_dtype(torch.float64)
    model = HardConstrainedBurgersPINN(width=8, hidden_layers=2).to(dtype=torch.float64)
    checks = verify_hard_constraints(
        model,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert checks["initial_condition_max_abs_error"] < 1.0e-12
    assert checks["boundary_condition_max_abs_error"] < 1.0e-12
