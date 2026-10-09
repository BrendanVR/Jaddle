"""Tests for the convex solver, jaddle.jaddle_convex."""

import jax.numpy as jnp
import numpy as np
import pytest

import jaddle.jaddle_convex as jc

TOLS = dict(
    primal_grad_norm_tolerance=1e-6,
    primal_feasibility_tolerance=1e-6,
    complementarity_slack_tolerance=1e-6,
)


def project_onto_simplex(v):
    """Exact Euclidean projection onto {x >= 0, sum x = 1}."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > css - 1)[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def isotonic_fit(y, lo, hi):
    """Exact box-constrained isotonic regression: pool-adjacent-violators, then
    clip to the box (clipping preserves optimality for this problem)."""
    blocks = []  # [mean, size]
    for value in y:
        blocks.append([value, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, s2 = blocks.pop()
            m1, s1 = blocks.pop()
            blocks.append([(m1 * s1 + m2 * s2) / (s1 + s2), s1 + s2])
    fit = np.concatenate([np.full(s, m) for m, s in blocks])
    return np.clip(fit, lo, hi)


def simplex_problem(n=50, seed=0):
    a = np.random.default_rng(seed).standard_normal(n)
    cp = jc.JaddleCP(
        num_variables=n,
        objective=lambda x: jnp.sum((x - a) ** 2),
        constraints_eq=lambda x: jnp.array([jnp.sum(x) - 1.0]),
        constraints_ineq=lambda x: jnp.zeros(0),
        lower_bounds=jnp.zeros(n),
        upper_bounds=jnp.full(n, jnp.inf),
    )
    return cp, project_onto_simplex(a)


def solve(cp, **kwargs):
    kwargs.setdefault("max_epochs", 300)
    return jc.solve(cp, iterations_per_epoch=200, **TOLS, **kwargs)


@pytest.mark.parametrize(
    "update_mode", ["extragradient", "forward_reflected", "alternating"]
)
def test_simplex_projection(update_mode):
    cp, expected = simplex_problem()
    result = solve(cp, update_mode=update_mode)
    assert result["converged"]
    assert result["stop_reason"] == "converged"
    np.testing.assert_allclose(result["solution"].primal, expected, atol=1e-5)


def test_isotonic_regression():
    n = 60
    x = np.linspace(-1, 1, n)
    y = x**3 + 0.15 * np.random.default_rng(0).standard_normal(n)
    cp = jc.JaddleCP(
        num_variables=n,
        objective=lambda p: jnp.sum((p - y) ** 2),
        constraints_eq=lambda p: jnp.zeros(0),
        constraints_ineq=lambda p: p[:-1] - p[1:],
        lower_bounds=-jnp.ones(n),
        upper_bounds=jnp.ones(n),
    )
    result = solve(cp)
    assert result["converged"]
    np.testing.assert_allclose(
        result["solution"].primal, isotonic_fit(y, -1.0, 1.0), atol=1e-4
    )


def test_epoch_budget():
    cp, _ = simplex_problem()
    result = jc.solve(cp, max_epochs=1, iterations_per_epoch=5, **TOLS)
    assert not result["converged"]
    assert result["stop_reason"] == "max_epochs"


def test_invalid_update_mode():
    cp, _ = simplex_problem()
    with pytest.raises(ValueError):
        jc.solve(cp, update_mode="pdhg")


def test_forward_reflected_requires_adaptive_eta():
    cp, _ = simplex_problem()
    with pytest.raises(ValueError):
        jc.solve(cp, update_mode="forward_reflected", adaptive_eta=None)


# --- Traceable solves, batching and derivatives ------------------------------

import jax  # noqa: E402

TIGHT = dict(
    primal_grad_norm_tolerance=1e-11,
    primal_feasibility_tolerance=1e-11,
    complementarity_slack_tolerance=1e-11,
    iterations_per_epoch=500,
    max_epochs=3000,
)


def build_scaled_simplex(params):
    # Projection of a onto {x >= 0, sum x = s}: x* = s * P(a / s).
    a, s = params
    n = a.shape[0]
    return jc.JaddleCP(
        num_variables=n,
        objective=lambda x: jnp.sum((x - a) ** 2),
        constraints_eq=lambda x: jnp.array([jnp.sum(x) - s]),
        constraints_ineq=lambda x: jnp.zeros(0),
        lower_bounds=jnp.zeros(n),
        upper_bounds=jnp.full(n, jnp.inf),
    )


def exact_scaled_projection(a, s):
    return s * project_onto_simplex(np.asarray(a) / s)


@pytest.mark.parametrize(
    "options",
    [{}, {"update_mode": "forward_reflected"}, {"restarts": True, "average": True}],
)
def test_make_solver_matches_solve(options):
    cp, _ = simplex_problem(n=30)
    kwargs = dict(TOLS, iterations_per_epoch=200, max_epochs=300, **options)
    reference = jc.solve(cp, **kwargs)
    # Called without an outer jit (which may round differently), the traced
    # solve is solve() exactly.
    result = jc.make_solver(lambda _: cp, **kwargs)(jnp.zeros(()))
    assert int(result.epochs) == reference["epochs"]
    for a, b in zip(result.solution, reference["solution"]):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_make_solver_vmap_batch():
    a = np.random.default_rng(1).standard_normal((6, 30))
    solve_fn = jc.make_solver(build_scaled_simplex, **TIGHT)
    result = jax.jit(jax.vmap(solve_fn))((jnp.asarray(a), jnp.ones(6)))
    assert np.asarray(result.converged).all()
    for i in range(6):
        np.testing.assert_allclose(
            result.solution.primal[i], exact_scaled_projection(a[i], 1.0), atol=1e-7
        )


def test_optimal_value_gradient():
    # z*(a, s) = ||x* - a||^2: dz/da = 2 (a - x*) and dz/ds = mu*, checked
    # against central differences of the exact projection.
    rng = np.random.default_rng(2)
    a, s = jnp.asarray(rng.standard_normal(10)), jnp.asarray(1.5)
    value_fn = jc.make_optimal_value(build_scaled_simplex, return_solution=True, **TIGHT)
    (z, result), (grad_a, grad_s) = jax.value_and_grad(value_fn, has_aux=True)((a, s))
    x = exact_scaled_projection(a, float(s))
    assert abs(float(z) - np.sum((x - np.asarray(a)) ** 2)) < 1e-8
    np.testing.assert_allclose(grad_a, 2 * (np.asarray(a) - x), atol=1e-7)

    def z_exact(s_):
        x_ = exact_scaled_projection(a, s_)
        return np.sum((x_ - np.asarray(a)) ** 2)

    h = 1e-6
    fd = (z_exact(float(s) + h) - z_exact(float(s) - h)) / (2 * h)
    assert abs(float(grad_s) - fd) < 1e-6


def test_optimal_value_gradient_through_bounds():
    # Upper bounds as a parameter: the active ones carry -nu_upper.
    rng = np.random.default_rng(3)
    n = 8

    def build(params):
        a, u = params
        return jc.JaddleCP(
            num_variables=n,
            objective=lambda x: jnp.sum((x - a) ** 2),
            constraints_eq=lambda x: jnp.array([jnp.sum(x) - 2.0]),
            constraints_ineq=lambda x: jnp.zeros(0),
            lower_bounds=jnp.zeros(n),
            upper_bounds=u,
        )

    a = jnp.asarray(rng.standard_normal(n) + 0.5)
    u = jnp.full(n, 0.45)
    value_fn = jax.jit(jc.make_optimal_value(build, **TIGHT))
    grad_u = jax.grad(value_fn)((a, u))[1]
    k = int(np.argmin(np.asarray(grad_u)))  # most active upper bound
    assert float(grad_u[k]) < 0
    h = 1e-5
    fd = (float(value_fn((a, u.at[k].add(h)))) - float(value_fn((a, u.at[k].add(-h))))) / (2 * h)
    assert abs(float(grad_u[k]) - fd) < 1e-6


def test_solution_vjp_matches_finite_differences():
    rng = np.random.default_rng(4)
    a, s = jnp.asarray(rng.standard_normal(10)), jnp.asarray(1.5)
    x_fn = jc.make_solution(build_scaled_simplex, **TIGHT)
    cot = rng.standard_normal(10)
    grad_a, grad_s = jax.grad(lambda p: jnp.dot(x_fn(p), cot))((a, s))
    h = 1e-6

    def fd(bump):
        return cot @ (bump(h) - bump(-h)) / (2 * h)

    for j in range(10):
        e = np.zeros(10)
        e[j] = 1.0
        expected = fd(lambda d: exact_scaled_projection(np.asarray(a) + d * e, float(s)))
        assert abs(float(grad_a[j]) - expected) < 1e-6
    expected = fd(lambda d: exact_scaled_projection(a, float(s) + d))
    assert abs(float(grad_s) - expected) < 1e-6


def test_solution_vjp_is_nan_when_undefined():
    # A linear objective over a face of optima: the KKT system is singular,
    # so dx*/dθ is undefined and the VJP says so with NaN.
    def build(c):
        return jc.JaddleCP(
            num_variables=2,
            objective=lambda x: c @ x,
            constraints_eq=lambda x: jnp.array([jnp.sum(x) - 1.0]),
            constraints_ineq=lambda x: jnp.zeros(0),
            lower_bounds=jnp.zeros(2),
            upper_bounds=jnp.full(2, jnp.inf),
        )

    x_fn = jc.make_solution(build, **dict(TIGHT, max_epochs=50))
    # sum(x) = 1 everywhere, so its derivative is genuinely 0 and the singular
    # system is consistent for that cotangent; x[0] depends on where in the
    # face the solution lies, which is undefined.
    grad_sum = jax.grad(lambda c: x_fn(c).sum())(jnp.array([1.0, 1.0]))
    np.testing.assert_allclose(grad_sum, 0.0, atol=1e-9)
    grad = jax.grad(lambda c: x_fn(c)[0])(jnp.array([1.0, 1.0]))
    assert np.all(np.isnan(np.asarray(grad)))
