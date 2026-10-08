"""Tests for jaddle.presolve.eliminate_defined_variables.

Each case checks that the reduced LP plus its objective offset has the same
optimum as the original, and that postsolve maps a reduced optimum back to a
feasible, optimal point of the original problem.
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from jaddle.jaddle_basic_types import LP
from jaddle.presolve import eliminate_defined_variables


def lp_solve(lp):
    bounds = [
        (None if np.isinf(lo) else lo, None if np.isinf(hi) else hi)
        for lo, hi in zip(lp.lower_bounds, lp.upper_bounds)
    ]
    has_eq = lp.A_eq.shape[0] > 0
    has_ineq = lp.A_ineq.shape[0] > 0
    res = linprog(
        lp.c,
        A_ub=lp.A_ineq if has_ineq else None,
        b_ub=lp.b_ineq if has_ineq else None,
        A_eq=lp.A_eq if has_eq else None,
        b_eq=lp.b_eq if has_eq else None,
        bounds=bounds,
        method="highs",
    )
    assert res.status == 0
    return res


def assert_equivalent(full, reduced, offset, postsolve):
    full_res, red_res = lp_solve(full), lp_solve(reduced)
    assert np.isclose(red_res.fun + offset, full_res.fun, atol=1e-8)

    x = postsolve.primal(red_res.x)
    assert x.shape == (len(full.c),)
    assert np.isclose(full.c @ x, full_res.fun, atol=1e-8)
    np.testing.assert_allclose(full.A_eq @ x, full.b_eq, atol=1e-8)
    assert np.all(full.A_ineq @ x <= full.b_ineq + 1e-8)
    assert np.all(x >= full.lower_bounds - 1e-8)
    assert np.all(x <= full.upper_bounds + 1e-8)


def test_objective_row_is_eliminated():
    # Variables (z, x1, x2): min z  s.t.  z = x1 + 2 x2,  x1 + x2 >= 1,
    # 0 <= z <= 5, x >= 0. The objective hides in the equality row.
    full = LP(
        c=np.array([1.0, 0.0, 0.0]),
        A_eq=sp.csc_matrix([[1.0, -1.0, -2.0]]),
        b_eq=np.array([0.0]),
        A_ineq=sp.csc_matrix([[0.0, -1.0, -1.0]]),
        b_ineq=np.array([-1.0]),
        lower_bounds=np.zeros(3),
        upper_bounds=np.array([5.0, np.inf, np.inf]),
    )
    reduced, offset, postsolve = eliminate_defined_variables(full)
    assert postsolve is not None
    assert len(reduced.c) == 2 and reduced.A_eq.shape[0] == 0
    # z's upper bound is not implied by x >= 0, so it becomes an inequality row.
    assert reduced.A_ineq.shape[0] == 2
    np.testing.assert_allclose(reduced.c, [1.0, 2.0])
    assert_equivalent(full, reduced, offset, postsolve)


def test_zero_cost_aggregate_is_eliminated():
    # Variables (z, x1, x2, x3): z = x1 + x2 + x3 + 0.5 has no cost, is used by
    # an inequality row, and the row has a nonzero right-hand side.
    full = LP(
        c=np.array([0.0, -1.0, -2.0, -3.0]),
        A_eq=sp.csc_matrix([[1.0, -1.0, -1.0, -1.0]]),
        b_eq=np.array([0.5]),
        A_ineq=sp.csc_matrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]]),
        b_ineq=np.array([2.0, 1.0]),
        lower_bounds=np.zeros(4),
        upper_bounds=np.full(4, np.inf),
    )
    # The default fill cap (max_fill * nnz) is too tight for a 6-nonzero LP.
    reduced, offset, postsolve = eliminate_defined_variables(
        full, min_row_nnz=4, max_fill=10.0
    )
    assert postsolve is not None
    assert len(reduced.c) == 3 and reduced.A_eq.shape[0] == 0
    assert_equivalent(full, reduced, offset, postsolve)


def test_nothing_to_eliminate():
    full = LP(
        c=np.array([1.0, 1.0]),
        A_eq=sp.csc_matrix([[1.0, 1.0]]),
        b_eq=np.array([1.0]),
        A_ineq=sp.csc_matrix((0, 2)),
        b_ineq=np.zeros(0),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, np.inf),
    )
    reduced, offset, postsolve = eliminate_defined_variables(full)
    assert reduced is full and offset == 0.0 and postsolve is None
