"""Second-order and PSD cones: the projections, the LP solver on cone-
constrained variables, and the CVXPY conic solver against Clarabel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

import jaddle.jaddle_linear as jl
from jaddle.cones import Cones, smat, svec

TIGHT = dict(
    primal_feasibility_tolerance=1e-8,
    dual_feasibility_tolerance=1e-8,
    dual_gap_tolerance=1e-8,
)

CONES = Cones(soc=(3, 1, 5), psd=(3, 2, 3))


# %%
# Projections


def test_moreau_decomposition():
    # v = Π_K(v) - Π_K(-v), with the two parts orthogonal (K self-dual).
    v = np.random.default_rng(0).standard_normal(CONES.size)
    p = np.asarray(CONES.project(jnp.asarray(v)))
    q = np.asarray(CONES.project(jnp.asarray(-v)))
    np.testing.assert_allclose(p - q, v, atol=1e-12)
    assert abs(p @ q) < 1e-12
    np.testing.assert_allclose(CONES.project(jnp.asarray(p)), p, atol=1e-12)


def test_soc_projection_cases():
    K = Cones(soc=(3, 3, 3))
    inside, polar, between = [2.0, 1.0, 1.0], [-2.0, 1.0, 1.0], [0.0, 3.0, 4.0]
    p = np.asarray(K.project(jnp.asarray(inside + polar + between)))
    np.testing.assert_allclose(p[:3], inside)
    np.testing.assert_allclose(p[3:6], 0.0)
    np.testing.assert_allclose(p[6:], [2.5, 1.5, 2.0])


def test_psd_projection_matches_eigh():
    v = np.random.default_rng(1).standard_normal(CONES.size)
    p = np.asarray(CONES.project(jnp.asarray(v)))
    offset = CONES.soc_size
    for d in CONES.psd:
        length = d * (d + 1) // 2
        lam, U = np.linalg.eigh(smat(v[offset : offset + length]))
        expected = svec(U @ np.diag(np.maximum(lam, 0.0)) @ U.T)
        np.testing.assert_allclose(p[offset : offset + length], expected, atol=1e-12)
        offset += length


def test_svec_round_trip_and_inner_product():
    rng = np.random.default_rng(2)
    A, B = (rng.standard_normal((4, 4)) for _ in range(2))
    A, B = A + A.T, B + B.T
    np.testing.assert_allclose(smat(svec(A)), A)
    assert svec(A) @ svec(B) == pytest.approx(np.trace(A @ B))


def test_tie_is_uniform_per_block():
    d = jnp.exp(jnp.asarray(np.random.default_rng(3).standard_normal(CONES.size + 2)))
    tied = np.asarray(CONES.tie(d))
    np.testing.assert_array_equal(tied[:2], np.asarray(d)[:2])
    offset = 2
    for size in CONES.block_sizes():
        block = tied[offset : offset + size]
        assert np.ptp(block) < 1e-12
        offset += size


def test_cone_variables_must_be_unbounded():
    n = 4
    with pytest.raises(ValueError, match="infinite bounds"):
        jl.LP(
            np.zeros(n),
            sp.csc_matrix((0, n)),
            np.zeros(0),
            sp.csc_matrix((0, n)),
            np.zeros(0),
            np.zeros(n),
            np.full(n, np.inf),
            cones=Cones(soc=(3,)),
        )


# %%
# The LP solver with cones


def unit_ball_lp(c, radius=1.0):
    """min cᵀx s.t. ‖x‖ <= radius, as [t, x] ∈ SOC with t = radius."""
    n = c.shape[0] + 1
    return jl.LP(
        jnp.concatenate([jnp.zeros(1), c]),
        sp.csc_matrix(np.eye(1, n)),
        jnp.reshape(radius, (1,)),
        sp.csc_matrix((0, n)),
        jnp.zeros(0),
        jnp.full(n, -jnp.inf),
        jnp.full(n, jnp.inf),
        cones=Cones(soc=(n,)),
    )


def test_soc_closed_form():
    c = np.random.default_rng(4).standard_normal(6)
    result = jl.solve(unit_ball_lp(c), **TIGHT)
    assert result["converged"]
    x = np.asarray(result["solution"].primal)[1:]
    np.testing.assert_allclose(x, -c / np.linalg.norm(c), atol=1e-6)


def test_min_eigenvalue_sdp_with_linear_variable():
    # min z + <C, X>  s.t.  tr X = 1,  X PSD,  0 <= z <= 2   ->  λ_min(C).
    rng = np.random.default_rng(5)
    d = 6
    B = rng.standard_normal((d, d))
    C = B + B.T
    L = d * (d + 1) // 2
    lp = jl.LP(
        np.r_[1.0, svec(C)],
        sp.csc_matrix(np.r_[0.0, svec(np.eye(d))][None, :]),
        np.ones(1),
        sp.csc_matrix((0, 1 + L)),
        np.zeros(0),
        np.r_[0.0, np.full(L, -np.inf)],
        np.r_[2.0, np.full(L, np.inf)],
        cones=Cones(psd=(d,)),
    )
    result = jl.solve(lp, **TIGHT)
    assert result["converged"]
    x = np.asarray(result["solution"].primal)
    assert x[0] == pytest.approx(0.0, abs=1e-8)
    assert svec(C) @ x[1:] == pytest.approx(np.linalg.eigvalsh(C)[0], abs=1e-6)


def test_scaling_keeps_cone_blocks_uniform():
    rng = np.random.default_rng(6)
    n_lin = 3
    n = n_lin + CONES.size
    A = sp.random(5, n, density=0.5, random_state=7, format="csc")
    lp = jl.LP(
        rng.standard_normal(n),
        A,
        np.ones(5),
        sp.csc_matrix((0, n)),
        np.zeros(0),
        np.r_[np.zeros(n_lin), np.full(CONES.size, -np.inf)],
        np.full(n, np.inf),
        cones=CONES,
    )
    _, _, col_scale, _ = jl.scale_problem(jl.to_jaddle_sparse(lp))
    col_scale = np.asarray(col_scale)
    offset = n_lin
    for size in CONES.block_sizes():
        assert np.ptp(col_scale[offset : offset + size]) == 0.0
        offset += size


def test_optimal_value_gradient_and_batch():
    # z*(c, r) = -r ‖c‖, so dz/dc = -r c / ‖c‖ and dz/dr = -‖c‖.
    def build(params):
        return unit_ball_lp(*params)

    c, r = jnp.array([1.0, -2.0, 0.5, 3.0]), 2.0
    z, (grad_c, grad_r) = jax.value_and_grad(jl.make_optimal_value(build, **TIGHT))(
        (c, r)
    )
    norm = jnp.linalg.norm(c)
    assert z == pytest.approx(-r * norm, rel=1e-6)
    np.testing.assert_allclose(grad_c, -r * c / norm, atol=1e-6)
    assert grad_r == pytest.approx(-norm, rel=1e-6)

    cs, rs = jnp.stack([c, -c, 2 * c]), jnp.array([1.0, 2.0, 3.0])
    batch = jl.solve_batch(build, (cs, rs), **TIGHT)
    values = jnp.einsum("bi,bi->b", batch.solution.primal[:, 1:], cs)
    np.testing.assert_allclose(values, -rs * jnp.linalg.norm(cs, axis=1), rtol=1e-6)


def test_unsupported_routes_raise():
    lp = unit_ball_lp(jnp.ones(3))
    with pytest.raises(NotImplementedError):
        jl.solve_with_presolve(lp)
    with pytest.raises(NotImplementedError):
        jl.detect_infeasibility(lp)


# %%
# CVXPY conic solver, against Clarabel


def _cvxpy_problems():
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(0)

    n = 12
    mu = rng.normal(0.05, 0.05, n)
    F = rng.standard_normal((n, n)) / np.sqrt(n)
    w = cp.Variable(n)
    # The minimum achievable risk is 0.216, so the 0.3 limit binds.
    portfolio = [cp.sum(w) == 1, w >= 0, cp.norm(F.T @ w) <= 0.3]
    yield "portfolio", cp.Problem(cp.Maximize(mu @ w), portfolio), portfolio, [w]

    A, b = rng.standard_normal((20, 6)), rng.standard_normal(20)
    x = cp.Variable(6)
    ball = [cp.norm(x, 2) <= 0.5]
    yield "robust_ls", cp.Problem(cp.Minimize(cp.norm(A @ x - b)), ball), ball, [x]

    d = 5
    B = rng.standard_normal((d, d))
    X = cp.Variable((d, d), symmetric=True)
    trace_one = [X >> 0, cp.trace(X) == 1]
    yield (
        "min_eigenvalue",
        cp.Problem(cp.Minimize(cp.trace((B + B.T) @ X)), trace_one),
        trace_one,
        [X],
    )

    # Max-Cut's relaxation has many optimal X, so only the value and the duals
    # are compared (no variables).
    d = 6
    W = np.triu(rng.random((d, d)) < 0.5, 1).astype(float)
    W = W + W.T
    L = np.diag(W.sum(1)) - W
    X = cp.Variable((d, d), symmetric=True)
    unit_diag = [X >> 0, cp.diag(X) == 1]
    yield (
        "maxcut",
        cp.Problem(cp.Maximize(cp.trace(L @ X) / 4), unit_diag),
        unit_diag,
        [],
    )


@pytest.mark.parametrize("name", ["portfolio", "robust_ls", "min_eigenvalue", "maxcut"])
def test_cvxpy_conic_matches_clarabel(name):
    cp = pytest.importorskip("cvxpy")
    import jaddle.jaddle_cvxpy as jcv

    problem, constraints, variables = next(
        p[1:] for p in _cvxpy_problems() if p[0] == name
    )
    problem.solve(solver="CLARABEL")
    value = problem.value
    primals = [np.copy(v.value) for v in variables]
    duals = [np.copy(c.dual_value) for c in constraints]

    # The epoch budget turns a non-converging case into a failure, not a hang.
    problem.solve(solver=jcv.JaddleConicSolver(), tol=1e-8, max_epochs=500)
    assert problem.status == cp.OPTIMAL
    assert problem.value == pytest.approx(value, abs=1e-6 * (1 + abs(value)))
    for v, x in zip(variables, primals):
        np.testing.assert_allclose(v.value, x, atol=1e-4)
    for c, y in zip(constraints, duals):
        np.testing.assert_allclose(c.dual_value, y, atol=1e-4)


def test_cvxpy_conic_warm_start():
    # A fresh solver instance per solve must still warm-start: CVXPY clears
    # its cache whenever the solver it is given changes.
    cp = pytest.importorskip("cvxpy")
    import jaddle.jaddle_cvxpy as jcv

    problem = next(p[1] for p in _cvxpy_problems() if p[0] == "portfolio")
    problem.solve(solver=jcv.JaddleConicSolver(), tol=1e-8, max_epochs=500)
    cold = problem.solver_stats.num_iters
    problem.solve(solver=jcv.JaddleConicSolver(), tol=1e-8, max_epochs=500)
    assert problem.status == cp.OPTIMAL
    assert problem.solver_stats.num_iters < cold
