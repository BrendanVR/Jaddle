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
