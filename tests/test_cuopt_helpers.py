import numpy as np
import scipy.sparse as sp

import jaddle.jaddle_linear as jl
from jaddle.cuopt_helpers import (
    cuopt_to_standard_form_sparse,
    jaddle_to_cuopt_dict,
    warmstart_cuopt_problem,
)
from jaddle.jaddle_basic_types import SaddleState


def test_cuopt_dict_to_standard_form():
    # A = [[1, 1], [2, -1]], rows: first equality, second upper inequality
    cuopt_dict = {
        "csr_constraint_matrix": {
            "offsets": [0, 2, 4],
            "indices": [0, 1, 0, 1],
            "values": [1.0, 1.0, 2.0, -1.0],
        },
        "constraint_bounds": {
            "lower_bounds": [3.0, "-inf"],
            "upper_bounds": [3.0, 4.0],
        },
        "objective_data": {
            "coefficients": [1.0, 2.0],
            "offset": 0.0,
        },
        "variable_bounds": {
            "lower_bounds": [0.0, 0.0],
            "upper_bounds": ["inf", "inf"],
        },
        "maximize": False,
    }

    lp = cuopt_to_standard_form_sparse(cuopt_dict)

    assert lp.A_eq.shape == (1, 2)
    assert lp.A_ineq.shape == (1, 2)
    assert np.allclose(lp.b_eq, np.array([3.0], dtype=np.float32))
    assert np.allclose(lp.b_ineq, np.array([4.0], dtype=np.float32))


def test_jaddle_to_cuopt_dict_roundtrip_layout():
    c = np.array([1.0, 2.0], dtype=np.float32)
    A_eq = sp.csc_matrix([[1.0, 1.0]], dtype=np.float32)
    b_eq = np.array([3.0], dtype=np.float32)
    A_ineq = sp.csc_matrix([[2.0, -1.0]], dtype=np.float32)
    b_ineq = np.array([4.0], dtype=np.float32)
    lb = np.array([0.0, 0.0], dtype=np.float32)
    ub = np.array([np.inf, np.inf], dtype=np.float32)

    lp = jl.LP(c, A_eq, b_eq, A_ineq, b_ineq, lb, ub)
    out = jaddle_to_cuopt_dict(lp)

    assert out["maximize"] is False
    assert out["constraint_bounds"]["lower_bounds"][0] == 3.0
    assert np.isneginf(out["constraint_bounds"]["lower_bounds"][1])
    assert out["constraint_bounds"]["upper_bounds"] == [3.0, 4.0]


def test_warmstart_cuopt_problem_uses_primal_start_method():
    class MockCuOptProblem:
        def __init__(self):
            self.seen = None

        def setPrimalStart(self, values):
            self.seen = values

    prob = MockCuOptProblem()
    sol = SaddleState(
        primal=np.array([1.5, -0.25], dtype=np.float32),
        dual_ineq=np.array([], dtype=np.float32),
        dual_eq=np.array([], dtype=np.float32),
    )

    ok = warmstart_cuopt_problem(prob, sol)
    assert ok is True
    assert np.allclose(np.asarray(prob.seen, dtype=np.float32), sol.primal)
