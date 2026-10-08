"""Tests for the LP solver, jaddle.jaddle_linear."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import linprog

import jaddle.jaddle_linear as jl

TOL = 1e-6


def toy_lp():
    # minimise 3 x1 + 2 x2  s.t.  x1 + x2 >= 4,  x >= 0.  Optimum (0, 4), obj 8.
    return jl.LP(
        c=np.array([3.0, 2.0]),
        A_eq=sp.csc_matrix((0, 2)),
        b_eq=np.zeros(0),
        A_ineq=sp.csc_matrix([[-1.0, -1.0]]),
        b_ineq=np.array([-4.0]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, np.inf),
    )


def random_lp(seed=0, m_eq=5, m_ineq=15, n=30):
    """A feasible, bounded LP with equality and inequality rows and a mix of
    boxed and half-bounded variables."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.1, 0.9, n)
    A_eq = rng.standard_normal((m_eq, n))
    A_ineq = rng.standard_normal((m_ineq, n))
    upper = np.ones(n)
    upper[: n // 3] = np.inf
    return jl.LP(
        c=rng.standard_normal(n),
        A_eq=sp.csc_matrix(A_eq),
        b_eq=A_eq @ x0,
        A_ineq=sp.csc_matrix(A_ineq),
        b_ineq=A_ineq @ x0 + rng.uniform(0.1, 1.0, m_ineq),
        lower_bounds=np.zeros(n),
        upper_bounds=upper,
    )


def reference_objective(lp):
    bounds = [
        (lo, None if np.isinf(hi) else hi)
        for lo, hi in zip(lp.lower_bounds, lp.upper_bounds)
    ]
    res = linprog(
        lp.c,
        A_ub=lp.A_ineq,
        b_ub=lp.b_ineq,
        A_eq=lp.A_eq if lp.A_eq.shape[0] else None,
        b_eq=lp.b_eq if lp.A_eq.shape[0] else None,
        bounds=bounds,
        method="highs",
    )
    assert res.status == 0
    return res.fun


def solve(lp, tol=TOL, **kwargs):
    return jl.solve(
        lp,
        primal_feasibility_tolerance=tol,
        dual_feasibility_tolerance=tol,
        dual_gap_tolerance=tol,
        max_epochs=kwargs.pop("max_epochs", 500),
        **kwargs,
    )


def assert_certified(lp, result, ref_obj, tol=TOL):
    assert result["converged"]
    assert result["stop_reason"] == "certificate"
    sol = result["solution"]
    obj = float(lp.objective(sol.primal))
    assert abs(obj - ref_obj) / (1 + abs(ref_obj)) < 10 * tol
    # Independently re-check the certificate in the original problem's units.
    cert = jl.evaluate_lp_certificate(
        jl.to_jaddle_sparse(lp), sol.primal, sol.dual_eq, sol.dual_ineq
    )
    assert float(cert["relative_gap"]) < 10 * tol
    assert float(cert["relative_primal_feasibility_residual"]) < 10 * tol
    assert float(cert["relative_dual_feasibility_residual"]) < 10 * tol


def test_toy_lp():
    lp = toy_lp()
    result = solve(lp, tol=1e-4)
    assert result["stop_reason"] == "certificate"
    np.testing.assert_allclose(result["solution"].primal, [0.0, 4.0], atol=1e-3)


@pytest.mark.parametrize("update_mode", ["alternating", "pdhg", "halpern"])
def test_random_lp_matches_reference(update_mode):
    lp = random_lp()
    assert_certified(lp, solve(lp, update_mode=update_mode), reference_objective(lp))


@pytest.mark.parametrize("seed", [1, 2])
def test_other_random_lps(seed):
    lp = random_lp(seed=seed, m_eq=8, m_ineq=25, n=50)
    assert_certified(lp, solve(lp), reference_objective(lp))


@pytest.mark.parametrize("adaptive_eta", [0.3, 0.0, "auto"])
def test_adaptive_eta_seeds(adaptive_eta):
    lp = random_lp()
    result = solve(lp, adaptive_eta=adaptive_eta)
    assert_certified(lp, result, reference_objective(lp))


@pytest.mark.parametrize("adaptive_eta", ["bogus", -1.0, None])
def test_invalid_adaptive_eta(adaptive_eta):
    with pytest.raises(ValueError):
        jl.solve(toy_lp(), adaptive_eta=adaptive_eta)


def test_invalid_update_mode():
    with pytest.raises(ValueError):
        jl.solve(toy_lp(), update_mode="synchronous")


def test_unscaled_solve():
    lp = random_lp()
    assert_certified(lp, solve(lp, scale=False), reference_objective(lp))


def test_accepts_jaddle_lp():
    lp = random_lp()
    result = solve(jl.to_jaddle_sparse(lp))
    assert_certified(lp, result, reference_objective(lp))


def test_epoch_budget():
    result = solve(random_lp(), tol=1e-12, max_epochs=1)
    assert not result["converged"]
    assert result["stop_reason"] == "max_epochs"
    assert result["epochs"] == 1


def test_time_budget():
    # The budget is spent before the first epoch boundary, where it is checked.
    result = solve(random_lp(), tol=1e-12, max_epochs=None, max_seconds=1e-3)
    assert not result["converged"]
    assert result["stop_reason"] == "time_limit"


def test_warm_start():
    lp = random_lp()
    loose = solve(lp, tol=1e-3)
    assert loose["stop_reason"] == "certificate"
    tight = solve(
        lp,
        initial_solution=loose["solution"],
        initial_opt_state=loose["opt_state"],
    )
    assert_certified(lp, tight, reference_objective(lp))
