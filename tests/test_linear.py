"""Tests for the LP solver, jaddle.jaddle_linear."""

import jax
import jax.numpy as jnp
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


def solve_certificate(lp, sol):
    """The certificate solve() terminates on (l2 norm, PDLP dual residual)."""
    # solve() gives an empty constraint class one zero row, and its duals.
    padded = getattr(jl, "__pad_empty_blocks")(jl.to_jaddle_sparse(lp))
    cert = jl.evaluate_lp_certificate(
        padded,
        sol.primal,
        sol.dual_eq,
        sol.dual_ineq,
        norm="l2",
        dual_residual="pdlp",
    )
    return {k: float(v) for k, v in cert.items()}


def test_certificate_matches_solve():
    lp = random_lp()
    cert = solve_certificate(lp, solve(lp)["solution"])
    assert cert["relative_primal_feasibility_residual"] <= TOL
    assert cert["relative_dual_feasibility_residual"] <= TOL
    assert cert["relative_gap_abs"] <= TOL


def test_warm_start_from_optimum_stays_put():
    # A certified point must certify again after one epoch from it. The duals
    # were once warm-started off by the objective scale c_max.
    lp = random_lp()
    sol = solve(lp)["solution"]
    again = solve(lp, initial_solution=sol, max_epochs=1, iterations_per_epoch=8)
    assert again["stop_reason"] == "certificate"
    y = np.concatenate([sol.dual_eq, sol.dual_ineq])
    y_again = np.concatenate([again["solution"].dual_eq, again["solution"].dual_ineq])
    assert np.linalg.norm(y_again - y) <= 1e-3 * (1 + np.linalg.norm(y))


def perturbed_optimum(seed=0, size=1e-3):
    lp = random_lp()
    sol = solve(lp)["solution"]
    rng = np.random.default_rng(seed)

    def noise(v):
        return v + size * rng.standard_normal(v.shape)

    return lp, sol, noise


def test_primal_polish_restores_feasibility():
    lp, sol, noise = perturbed_optimum()
    bad = jl.SaddleState(
        primal=noise(sol.primal), dual_ineq=sol.dual_ineq, dual_eq=sol.dual_eq
    )
    before = solve_certificate(lp, bad)["relative_primal_feasibility_residual"]
    x = jl.primal_polish(jl.to_jaddle_sparse(lp), bad)
    polished = jl.SaddleState(primal=x, dual_ineq=sol.dual_ineq, dual_eq=sol.dual_eq)
    after = solve_certificate(lp, polished)["relative_primal_feasibility_residual"]
    assert after < 1e-2 * before


def test_dual_polish_restores_feasibility():
    lp, sol, noise = perturbed_optimum()
    bad = jl.SaddleState(
        primal=sol.primal, dual_ineq=noise(sol.dual_ineq), dual_eq=noise(sol.dual_eq)
    )
    before = solve_certificate(lp, bad)["relative_dual_feasibility_residual"]
    dual_eq, dual_ineq = jl.dual_polish(jl.to_jaddle_sparse(lp), bad)
    polished = jl.SaddleState(primal=sol.primal, dual_ineq=dual_ineq, dual_eq=dual_eq)
    after = solve_certificate(lp, polished)["relative_dual_feasibility_residual"]
    assert after < 1e-2 * before
    # A feasibility projection, not a jump across the dual feasible set.
    y = np.concatenate([sol.dual_eq, sol.dual_ineq])
    moved = np.linalg.norm(np.concatenate([dual_eq, dual_ineq]) - y)
    perturbation = np.linalg.norm(np.concatenate([bad.dual_eq, bad.dual_ineq]) - y)
    assert moved <= perturbation


@pytest.mark.parametrize("make_lp", [toy_lp, random_lp])
def test_solve_with_polishing(make_lp):
    # Tight feasibility, loose gap; polish from the first epoch on.
    lp = make_lp()
    result = jl.solve_with_polishing(
        lp,
        tol=1e-9,
        dual_gap_tolerance=1e-4,
        first_polish_epoch=1,
        iterations_per_epoch=16,
        max_epochs=2000,
    )
    assert result["stop_reason"] == "certificate"
    assert set(result["polish"]) == {"attempts", "epochs", "polished"}
    cert = solve_certificate(lp, result["solution"])
    assert cert["relative_primal_feasibility_residual"] <= 1e-9
    assert cert["relative_dual_feasibility_residual"] <= 1e-9
    assert cert["relative_gap_abs"] <= 1e-4
    ref = reference_objective(lp)
    assert abs(cert["objective"] - ref) / (1 + abs(ref)) < 1e-3


def small_lp(c, A_ineq=None, b_ineq=None, A_eq=None, b_eq=None, ub=None):
    n = len(c)

    def block(A):
        return sp.csc_matrix(np.array(A, float)) if A is not None else sp.csc_matrix((0, n))

    def rhs(b):
        return np.array(b, float) if b is not None else np.zeros(0)

    return jl.LP(
        c=np.array(c, float),
        A_eq=block(A_eq),
        b_eq=rhs(b_eq),
        A_ineq=block(A_ineq),
        b_ineq=rhs(b_ineq),
        lower_bounds=np.zeros(n),
        upper_bounds=np.full(n, np.inf) if ub is None else np.array(ub, float),
    )


def contradictory_random_lp():
    # A random feasible LP plus the pair a.x <= 0 and a.x >= 1.
    lp = random_lp(seed=4)
    a = sp.csc_matrix(np.random.default_rng(3).standard_normal(lp.c.shape[0]))
    return jl.LP(
        c=lp.c,
        A_eq=lp.A_eq,
        b_eq=lp.b_eq,
        A_ineq=sp.vstack([lp.A_ineq, a, -a]).tocsc(),
        b_ineq=np.concatenate([lp.b_ineq, [0.0, -1.0]]),
        lower_bounds=lp.lower_bounds,
        upper_bounds=lp.upper_bounds,
    )


@pytest.mark.parametrize(
    "lp, status",
    [
        (small_lp([1, 1], [[1, 1], [-1, -1]], [1, -3]), "primal_infeasible"),
        (small_lp([1], [[-1]], [-2], ub=[1]), "primal_infeasible"),
        (small_lp([1, 1], A_eq=[[1, 1], [1, 1]], b_eq=[1, 2]), "primal_infeasible"),
        (contradictory_random_lp(), "primal_infeasible"),
        (small_lp([-1, 0], [[1, -1]], [1]), "dual_infeasible"),
        (random_lp(), "optimal"),
        (toy_lp(), "optimal"),
    ],
)
def test_detect_infeasibility(lp, status):
    result = jl.detect_infeasibility(lp, max_epochs=2000)
    assert result["status"] == status
    if status == "optimal":
        return
    # The returned rays certify in their own right.
    cert = jl.evaluate_infeasibility_certificate(
        getattr(jl, "__pad_empty_blocks")(jl.to_jaddle_sparse(lp)),
        result["primal_ray"],
        result["dual_ray_eq"],
        result["dual_ray_ineq"],
    )
    key = (
        "primal_infeasibility_ratio"
        if status == "primal_infeasible"
        else "dual_infeasibility_ratio"
    )
    assert float(cert[key]) <= 1e-8


@pytest.mark.parametrize("seed", range(6))
def test_detect_infeasibility_no_false_positive(seed):
    lp = random_lp(seed=seed, m_eq=seed % 4, m_ineq=10 + seed, n=20 + 2 * seed)
    result = jl.detect_infeasibility(lp, max_epochs=300, first_check_epoch=1)
    assert result["status"] in ("optimal", "undetermined")


def test_infeasibility_certificate_ignores_rounding():
    # A zero dual ray must not certify, whatever the rounding in its objective.
    lp = jl.to_jaddle_sparse(random_lp())
    y_eq = np.zeros(lp.n_eq)
    y_ineq = np.full(lp.A_ineq.shape[0], 1e-300)
    cert = jl.evaluate_infeasibility_certificate(
        lp, np.zeros(lp.c.shape[0]), y_eq, y_ineq
    )
    assert float(cert["primal_infeasibility_ratio"]) == np.inf
    assert float(cert["dual_infeasibility_ratio"]) == np.inf


def test_solve_with_polishing_epoch_budget():
    result = jl.solve_with_polishing(
        random_lp(), tol=1e-12, first_polish_epoch=1, max_epochs=3
    )
    assert result["stop_reason"] == "max_epochs"
    assert result["epochs"] == 3


@pytest.mark.parametrize(
    "options",
    [{}, {"update_mode": "pdhg"}, {"update_mode": "halpern"}, {"scale": False}],
)
def test_make_solver_matches_solve(options):
    # One traced solve reproduces solve() exactly (CPU, float64).
    lp = random_lp()
    tolerances = dict(
        primal_feasibility_tolerance=TOL,
        dual_feasibility_tolerance=TOL,
        dual_gap_tolerance=TOL,
    )
    reference = jl.solve(lp, **tolerances, **options)
    result = jax.jit(jl.make_solver(lp, **tolerances, **options))()
    assert int(result.epochs) == reference["epochs"]
    for a, b in zip(result.solution, reference["solution"]):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_make_solver_vmap_batch():
    # Batch over the cost, matrix values and right-hand sides: every member
    # certifies and matches HiGHS. (Members are not bit-identical to separate
    # solves: XLA rounds batched reductions differently, and the adaptive
    # steps amplify it.)
    lp = jl.to_jaddle_sparse(random_lp())
    base = lp.values()
    rng = np.random.default_rng(0)
    size = 4

    def jitter(v):
        return v * (1 + 0.1 * rng.uniform(-1, 1, (size,) + v.shape))

    batch = jl.LPValues(
        c=jitter(base.c),
        A_eq_data=jitter(base.A_eq_data),
        b_eq=jitter(base.b_eq),
        A_ineq_data=jitter(base.A_ineq_data),
        b_ineq=jitter(base.b_ineq),
        lower_bounds=np.broadcast_to(base.lower_bounds, (size,) + base.lower_bounds.shape),
        upper_bounds=np.broadcast_to(base.upper_bounds, (size,) + base.upper_bounds.shape),
    )
    solve_fn = jl.make_solver(
        lp,
        primal_feasibility_tolerance=TOL,
        dual_feasibility_tolerance=TOL,
        dual_gap_tolerance=TOL,
    )
    result = jax.jit(jax.vmap(solve_fn))(batch)
    assert np.asarray(result.converged).all()
    for i in range(size):
        member = lp.with_values(jax.tree.map(lambda a: a[i], batch))
        member_lp = jl.LP(
            c=np.asarray(member.c),
            A_eq=getattr(jl, "__convert_to_scipy")(member.A_eq),
            b_eq=np.asarray(member.b_eq),
            A_ineq=getattr(jl, "__convert_to_scipy")(member.A_ineq),
            b_ineq=np.asarray(member.b_ineq),
            lower_bounds=np.asarray(member.lower_bounds),
            upper_bounds=np.asarray(member.upper_bounds),
        )
        ref = reference_objective(member_lp)
        obj = float(member.c @ result.solution.primal[i])
        assert abs(obj - ref) / (1 + abs(ref)) < 10 * TOL


def test_solve_batch_broadcasts_shared_fields():
    # Only c and b_ineq batched; matrices, b_eq and bounds shared.
    lp = random_lp()
    base = jl.to_jaddle_sparse(lp).values()
    costs = np.stack([np.asarray(base.c) * s for s in (1.0, 1.1, 0.9)])
    rhs = np.stack([np.asarray(base.b_ineq) + d for d in (0.0, 0.05, 0.1)])
    result = jl.solve_batch(
        lp,
        base._replace(c=costs, b_ineq=rhs),
        primal_feasibility_tolerance=TOL,
        dual_feasibility_tolerance=TOL,
        dual_gap_tolerance=TOL,
    )
    assert result.solution.primal.shape == (3, base.c.shape[0])
    assert np.asarray(result.converged).all()
    for i in range(3):
        member = jl.LP(
            c=costs[i],
            A_eq=lp.A_eq,
            b_eq=lp.b_eq,
            A_ineq=lp.A_ineq,
            b_ineq=rhs[i],
            lower_bounds=lp.lower_bounds,
            upper_bounds=lp.upper_bounds,
        )
        ref = reference_objective(member)
        obj = float(costs[i] @ result.solution.primal[i])
        assert abs(obj - ref) / (1 + abs(ref)) < 10 * TOL


def test_solve_batch_accepts_unpadded_values():
    # toy_lp has no equality rows; its own values() must fit the padded pattern.
    lp = toy_lp()
    base = jl.to_jaddle_sparse(lp).values()
    result = jl.solve_batch(lp, base._replace(c=np.stack([base.c, 2 * base.c])))
    assert np.asarray(result.converged).all()


def test_optimal_value_gradient_matches_finite_differences():
    tight = dict(
        primal_feasibility_tolerance=1e-10,
        dual_feasibility_tolerance=1e-10,
        dual_gap_tolerance=1e-10,
    )
    lp = random_lp()
    jlp = jl.to_jaddle_sparse(lp)
    values = jlp.values()
    value_fn = jl.make_optimal_value(jlp, **tight)
    assert abs(float(value_fn(values)) - reference_objective(lp)) < 1e-8
    grad = jax.grad(value_fn)(values)
    to_scipy = getattr(jl, "__convert_to_scipy")

    def highs_value(v):
        m = jlp.with_values(v)
        return reference_objective(
            jl.LP(
                np.asarray(m.c),
                to_scipy(m.A_eq),
                np.asarray(m.b_eq),
                to_scipy(m.A_ineq),
                np.asarray(m.b_ineq),
                np.asarray(m.lower_bounds),
                np.asarray(m.upper_bounds),
            )
        )

    # The value is piecewise linear, so a small central difference that keeps
    # the optimal basis is exact. Check each field's largest entry.
    h = 1e-5
    for field in jl.LPValues._fields:
        g = np.asarray(getattr(grad, field))
        k = int(np.argmax(np.abs(g)))
        arr = np.asarray(getattr(values, field))
        if not np.isfinite(arr[k]):
            continue

        def bumped(d):
            a = arr.copy()
            a[k] += d
            return values._replace(**{field: a})

        fd = (highs_value(bumped(h)) - highs_value(bumped(-h))) / (2 * h)
        assert abs(fd - g[k]) <= 1e-6 * (1 + abs(fd)), field


def test_optimal_value_gradient_vmaps():
    lp = jl.to_jaddle_sparse(random_lp())
    base = lp.values()
    value_fn = jl.make_optimal_value(
        lp,
        primal_feasibility_tolerance=TOL,
        dual_feasibility_tolerance=TOL,
        dual_gap_tolerance=TOL,
    )
    costs = np.stack([np.asarray(base.c), 1.2 * np.asarray(base.c)])
    grads = jax.vmap(jax.grad(lambda c: value_fn(base._replace(c=c))))(costs)
    # dz*/dc = x*: each row is that member's optimal primal.
    for i in range(2):
        x = jl.make_solver(
            lp,
            primal_feasibility_tolerance=TOL,
            dual_feasibility_tolerance=TOL,
            dual_gap_tolerance=TOL,
        )(base._replace(c=costs[i])).solution.primal
        np.testing.assert_allclose(grads[i], x, atol=1e-4)


def test_perturbed_solution_matches_closed_form():
    # min c.x over {x1 + x2 = 4, x >= 0}: x1* = 4 [c1 < c2], so the perturbed
    # solution is x1 = 4 Φ((c2 - c1) / (σ√2)), with a closed-form derivative.
    from scipy.stats import norm

    lp = small_lp([3, 2], A_eq=[[1, 1]], b_eq=[4])
    sigma = 1.0
    x_fn = jl.make_perturbed_solution(
        lp,
        sigma=sigma,
        num_samples=2048,
        primal_feasibility_tolerance=1e-6,
        dual_feasibility_tolerance=1e-6,
        dual_gap_tolerance=1e-6,
        max_epochs=200,
    )
    c = np.array([3.0, 2.0])
    key = jax.random.PRNGKey(0)
    d = (c[1] - c[0]) / (sigma * np.sqrt(2))
    x1 = 4 * norm.cdf(d)
    dx1 = 4 * norm.pdf(d) / (sigma * np.sqrt(2))
    x = jax.jit(x_fn)(c, key)
    jac = jax.jit(jax.jacrev(x_fn))(c, key)
    np.testing.assert_allclose(x, [x1, 4 - x1], atol=0.05)
    np.testing.assert_allclose(jac[0], [-dx1, dx1], atol=0.1)


def test_solution_vjp_matches_finite_differences():
    tight = dict(
        primal_feasibility_tolerance=1e-11,
        dual_feasibility_tolerance=1e-11,
        dual_gap_tolerance=1e-11,
    )
    lp = jl.to_jaddle_sparse(random_lp())
    values = lp.values()
    to_scipy = getattr(jl, "__convert_to_scipy")

    def highs_x(v):
        m = lp.with_values(v)
        res = linprog(
            np.asarray(m.c),
            A_ub=to_scipy(m.A_ineq),
            b_ub=np.asarray(m.b_ineq),
            A_eq=to_scipy(m.A_eq),
            b_eq=np.asarray(m.b_eq),
            bounds=[
                (lo, None if np.isinf(hi) else hi)
                for lo, hi in zip(np.asarray(m.lower_bounds), np.asarray(m.upper_bounds))
            ],
            method="highs-ds",
        )
        return res.x

    x_fn = jl.make_solution(lp, **tight)
    g = np.random.default_rng(1).standard_normal(values.c.shape[0])
    vjp = jax.grad(lambda v: jnp.dot(x_fn(v), g))(values)
    assert not np.any(np.asarray(vjp.c))  # x* is piecewise constant in c
    h = 1e-6
    for field in jl.LPValues._fields[1:]:
        gv = np.asarray(getattr(vjp, field))
        k = int(np.argmax(np.abs(gv)))
        arr = np.asarray(getattr(values, field))

        def bumped(d):
            a = arr.copy()
            a[k] += d
            return values._replace(**{field: a})

        fd = g @ (highs_x(bumped(h)) - highs_x(bumped(-h))) / (2 * h)
        assert abs(fd - gv[k]) <= 1e-6 * (1 + abs(fd)), field


def test_solution_vjp_rejects_optimal_face():
    # Every point of x1 + x2 = 1, x >= 0 is optimal; PDHG returns its middle,
    # where dx*/dθ is undefined.
    lp = small_lp([1, 1], [[-1, -1]], [-1])
    x_fn = jl.make_solution(
        lp,
        primal_feasibility_tolerance=1e-10,
        dual_feasibility_tolerance=1e-10,
        dual_gap_tolerance=1e-10,
    )
    np.testing.assert_allclose(x_fn(), [0.5, 0.5], atol=1e-6)
    with pytest.raises(Exception, match="not a nondegenerate vertex"):
        jax.grad(lambda v: x_fn(v).sum())(jl.to_jaddle_sparse(lp).values())


PRESOLVE_TOL = dict(
    primal_feasibility_tolerance=1e-8,
    dual_feasibility_tolerance=1e-8,
    dual_gap_tolerance=1e-8,
)


def reducible_lp():
    # random_lp plus a fixed column (lower = upper = 0.5) that appears in the
    # rows, and a singleton row x0 <= 0.7: HiGHS presolve removes both.
    lp = random_lp()
    rng = np.random.default_rng(5)
    m_eq, m_ineq = lp.A_eq.shape[0], lp.A_ineq.shape[0]
    fixed_eq = sp.csc_matrix(rng.standard_normal((m_eq, 1)))
    fixed_ineq = sp.csc_matrix(rng.standard_normal((m_ineq, 1)))
    singleton = sp.csc_matrix(([1.0], ([0], [0])), shape=(1, lp.c.shape[0] + 1))
    return jl.LP(
        c=np.append(lp.c, 1.0),
        A_eq=sp.hstack([lp.A_eq, fixed_eq]).tocsc(),
        b_eq=lp.b_eq + 0.5 * fixed_eq.toarray().ravel(),
        A_ineq=sp.vstack([sp.hstack([lp.A_ineq, fixed_ineq]), singleton]).tocsc(),
        b_ineq=np.append(lp.b_ineq + 0.5 * fixed_ineq.toarray().ravel(), 0.7),
        lower_bounds=np.append(lp.lower_bounds, 0.5),
        upper_bounds=np.append(lp.upper_bounds, 0.5),
    )


@pytest.mark.parametrize("make_lp", [random_lp, reducible_lp])
def test_solve_with_presolve(make_lp):
    lp = make_lp()
    result = jl.solve_with_presolve(lp, **PRESOLVE_TOL)
    if make_lp is reducible_lp:
        assert result["presolve"]["status"] == "kReduced"
    assert result["converged"]
    ref = reference_objective(lp)
    assert abs(result["objective"] - ref) / (1 + abs(ref)) < 1e-6
    # The solution and certificate are for the LP as given, in its own rows.
    s = result["solution"]
    assert s.primal.shape == lp.c.shape
    assert s.dual_eq.shape == lp.b_eq.shape
    assert s.dual_ineq.shape == lp.b_ineq.shape
    cert = solve_certificate(lp, s)
    assert cert["relative_primal_feasibility_residual"] <= 1e-6
    assert cert["relative_dual_feasibility_residual"] <= 1e-6


def test_solve_with_presolve_reduced_to_empty():
    result = jl.solve_with_presolve(toy_lp())
    assert result["stop_reason"] == "presolve_solved"
    assert result["converged"]
    np.testing.assert_allclose(result["solution"].primal, [0.0, 4.0], atol=1e-9)


def test_solve_with_presolve_detects_infeasibility():
    lp = small_lp([1, 1], [[1, 1], [-1, -1]], [1, -3])
    result = jl.solve_with_presolve(lp)
    assert result["stop_reason"] == "presolve_infeasible"
    assert not result["converged"]
    assert result["solution"] is None


def test_solve_with_presolve_reads_files(tmp_path):
    import highspy

    from jaddle.highs_helpers import jaddle_lp_to_highs

    lp = reducible_lp()
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.passModel(jaddle_lp_to_highs(lp))
    path = str(tmp_path / "reducible.mps")
    highs.writeModel(path)
    result = jl.solve_with_presolve(path, **PRESOLVE_TOL)
    assert result["converged"]
    ref = reference_objective(lp)
    assert abs(result["objective"] - ref) / (1 + abs(ref)) < 1e-6


def test_warm_start_without_padding_rows():
    # toy_lp has no equality rows; a warm start in its own row layout (no
    # dual for the padding row solve() adds) must be accepted.
    lp = toy_lp()
    cold = solve(lp)
    s = cold["solution"]
    warm = solve(
        lp,
        initial_solution=jl.SaddleState(
            primal=s.primal, dual_ineq=s.dual_ineq, dual_eq=np.zeros(0)
        ),
    )
    assert warm["stop_reason"] == "certificate"
