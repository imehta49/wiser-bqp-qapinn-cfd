import math

import numpy as np

from wiser_qapinn.burgers import generate_reference_solution, reference_quality_checks


def test_reference_satisfies_conditions_and_symmetry():
    x, times, solution = generate_reference_solution(
        nu=0.1 / math.pi,
        nx_output=64,
        nt=21,
        nx_internal=129,
        t_max=0.25,
    )
    checks = reference_quality_checks(x, solution)
    assert solution.shape == (times.size, x.size)
    assert np.all(np.isfinite(solution))
    assert checks["initial_condition_max_abs_error"] < 1.0e-12
    assert checks["boundary_condition_max_abs_error"] < 1.0e-12
    assert checks["antisymmetry_max_abs_error"] < 1.0e-8
