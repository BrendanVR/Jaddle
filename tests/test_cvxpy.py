"""CVXPY front end: both routes checked against Clarabel (bundled with
CVXPY), primal, objective and every constraint's dual."""

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")

import jax.numpy as jnp  # noqa: E402

import jaddle.jaddle_cvxpy as jcv  # noqa: E402

TOL = dict(tol=1e-8)


def reference(problem, constraints, variables):
    """Clarabel's value, primal and duals."""
    problem.solve(solver="CLARABEL")
    assert problem.status == cp.OPTIMAL
    return (
        problem.value,
        [np.copy(v.value) for v in variables],
        [np.copy(c.dual_value) for c in constraints],
    )


def assert_matches(problem, constraints, variables, ref, atol=1e-6, dual_atol=1e-5):
    value, primals, duals = ref
    assert problem.status == cp.OPTIMAL
    assert problem.value == pytest.approx(value, abs=atol * (1 + abs(value)))
    for v, x in zip(variables, primals):
        np.testing.assert_allclose(v.value, x, atol=atol)
    for c, y in zip(constraints, duals):
        np.testing.assert_allclose(c.dual_value, y, atol=dual_atol)


def lp(seed=3):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((4, 8))
    c = rng.uniform(1, 2, 8)
    x = cp.Variable(8, bounds=[-1, 4])
    constraints = [A[:2] @ x == 1, A[2] @ x >= -1, A[3] @ x <= 2]
    return cp.Problem(cp.Minimize(c @ x), constraints), constraints, [x]


def simplex_qp(n=6, seed=0):
    a = np.random.default_rng(seed).standard_normal(n)
    x = cp.Variable(n)
    constraints = [cp.sum(x) == 1, x >= 0]
    return cp.Problem(cp.Minimize(cp.sum_squares(x - a)), constraints), constraints, [x]


def l1_logistic(n=60, p=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = np.sign(X @ rng.standard_normal(p) + 0.3 * rng.standard_normal(n))
    w, b = cp.Variable(p), cp.Variable()
    loss = cp.sum(cp.logistic(-cp.multiply(y, X @ w + b))) / n
    return cp.Problem(cp.Minimize(loss + 0.05 * cp.norm1(w))), [], [w, b]


def lse_budget(explicit_bound=False):
    x = cp.Variable(5, nonneg=not explicit_bound)
    constraints = [cp.sum(x) == 1, cp.maximum(x[0], x[1]) <= 0.3]
    if explicit_bound:
        constraints.append(x >= 0)
    objective = cp.log_sum_exp(np.arange(5.0) - 3 * x) + cp.sum_squares(x)
    return cp.Problem(cp.Minimize(objective), constraints), constraints, [x]


# %%
# Route 1: JaddleSolver


@pytest.mark.parametrize("presolve", [True, False])
def test_lp(presolve):
    problem, constraints, variables = lp()
    ref = reference(problem, constraints, variables)
    problem.solve(solver=jcv.JaddleSolver(), presolve=presolve, **TOL)
    assert_matches(problem, constraints, variables, ref)
    stats = problem.solver_stats.extra_stats
    assert ("presolve" in stats) == presolve


def test_lp_maximisation():
    x = cp.Variable(4, nonneg=True)
    constraints = [cp.sum(x) <= 2, x[0] - x[1] == 0.5]
    problem = cp.Problem(cp.Maximize(np.arange(1, 5) @ x - 1), constraints)
    ref = reference(problem, constraints, [x])
    problem.solve(solver=jcv.JaddleSolver(), **TOL)
    assert_matches(problem, constraints, [x], ref)


def test_qp():
    problem, constraints, variables = simplex_qp()
    ref = reference(problem, constraints, variables)
    problem.solve(solver=jcv.JaddleSolver(), **TOL)
    assert_matches(problem, constraints, variables, ref)


def test_qp_warm_start():
    # A fresh solver instance per solve must still warm-start: CVXPY clears
    # its cache whenever the solver it is given changes.
    problem, _, _ = simplex_qp(n=30)
    problem.solve(solver=jcv.JaddleSolver(), **TOL)
    cold = problem.solver_stats.num_iters
    problem.solve(solver=jcv.JaddleSolver(), warm_start=True, **TOL)
    assert problem.status == cp.OPTIMAL
    assert problem.solver_stats.num_iters < cold


def test_presolve_infeasible():
    x = cp.Variable(3, nonneg=True)
    problem = cp.Problem(cp.Minimize(cp.sum(x)), [cp.sum(x) >= 2, cp.sum(x) <= 1])
    problem.solve(solver=jcv.JaddleSolver())
    assert problem.status == cp.INFEASIBLE


def test_budget_stop_keeps_point():
    problem, _, (x,) = simplex_qp(n=30)
    with pytest.warns(UserWarning):
        problem.solve(
            solver=jcv.JaddleSolver(), max_epochs=1, iterations_per_epoch=2, tol=1e-14
        )
    assert problem.status == cp.USER_LIMIT
    assert x.value is not None


# %%
# Route 2: the compiled JaddleCP


def test_compiled_l1_logistic():
    problem, constraints, variables = l1_logistic()
    assert not problem.is_qp()
    ref = reference(problem, constraints, variables)
    jcv.solve(problem, **TOL)
    # The l1 penalty leaves the minimiser ill-conditioned along the support.
    assert_matches(problem, constraints, variables, ref, atol=2e-5)
    assert problem.solver_stats.solver_name == "JADDLE"


@pytest.mark.parametrize("explicit_bound", [False, True])
def test_compiled_constraints_and_bounds(explicit_bound):
    # A maximum in a constraint (epigraph rows), an equality row, and x >= 0
    # as an attribute or as a constraint held as a box bound, whose dual is
    # recovered from the reduced gradient.
    problem, constraints, variables = lse_budget(explicit_bound)
    ref = reference(problem, constraints, variables)
    jcv.solve(problem, **TOL)
    assert_matches(problem, constraints, variables, ref, atol=2e-5, dual_atol=5e-5)
    compiled = jcv.compile_problem(problem)
    assert len(compiled.bounds) == int(explicit_bound)
    assert compiled.num_epigraph == 1


def test_compiled_maximisation():
    x = cp.Variable(4)
    constraints = [cp.sum(x) == 1, cp.norm_inf(x) <= 0.6]
    problem = cp.Problem(cp.Maximize(-cp.log_sum_exp(x) - cp.sum_squares(x)), constraints)
    ref = reference(problem, constraints, [x])
    jcv.solve(problem, **TOL)
    assert_matches(problem, constraints, [x], ref, atol=2e-5, dual_atol=5e-5)


def test_compiled_route_on_a_qp():
    # Huber regression is QP-representable; qp_route=False compiles it.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 5))
    y = X @ rng.standard_normal(5) + rng.standard_t(2, 40)
    w = cp.Variable(5)
    constraints = [cp.norm_inf(w) <= 2]
    problem = cp.Problem(cp.Minimize(cp.sum(cp.huber(X @ w - y, 1.0))), constraints)
    ref = reference(problem, constraints, [w])
    for qp_route in (True, False):
        jcv.solve(problem, qp_route=qp_route, **TOL)
        assert_matches(problem, constraints, [w], ref)


def test_without_epigraph():
    problem, constraints, variables = lse_budget()
    ref = reference(problem, constraints, variables)
    jcv.solve(problem, epigraph=False, tol=1e-6, max_epochs=2000)
    assert jcv.compile_problem(problem, epigraph=False).num_epigraph == 0
    assert problem.value == pytest.approx(ref[0], abs=1e-4)


def test_compiled_values_match_cvxpy():
    # The compiled functions against CVXPY's own evaluation, over the affine
    # atoms and a sparse constant matrix.
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    X = cp.Variable((3, 4))
    v = cp.Variable(4)
    s = cp.Variable()
    S = sp.random(5, 4, density=0.5, random_state=0)
    expression = cp.hstack(
        [
            cp.sum(X, axis=0) @ np.arange(4.0),
            cp.trace(X[:, :3]) / 2,
            cp.sum(cp.reshape(X.T, (2, 6), order="F")[1]),
            cp.sum(cp.vstack([X, X[[0, 2]]]), axis=1)[3],
            cp.sum(S @ v) - s,
            cp.sum(cp.multiply(np.arange(4.0), v) + s),
            cp.sum(cp.diag(v)[1:, 2]),
            cp.pnorm(v, 3),
            cp.quad_form(v, np.eye(4) * 2),
            cp.max(cp.abs(X)),
        ]
    )
    problem = cp.Problem(cp.Minimize(cp.sum(expression)))
    compiled = jcv.compile_problem(problem, epigraph=False)
    for _ in range(3):
        point = rng.standard_normal(compiled.cp.num_variables)
        for var in problem.variables():
            offset, shape = compiled.variables[var.id]
            var.value = point[offset : offset + int(np.prod(shape))].reshape(shape)
        got = float(compiled.cp.objective(jnp.asarray(point)))
        assert got == pytest.approx(float(problem.objective.value), rel=1e-10)


def test_entr_values_and_extension():
    # Above ENTR_FLOOR the compiled entr is CVXPY's; below it, the tangent
    # line keeps value and gradient finite where entr is undefined.
    import jax

    rng = np.random.default_rng(0)
    A = rng.uniform(0.1, 1.0, (5, 3))
    v = cp.Variable(3)
    problem = cp.Problem(cp.Maximize(cp.sum(cp.entr(A @ v))))
    compiled = jcv.compile_problem(problem)
    for _ in range(3):
        v.value = rng.uniform(0.01, 2.0, 3)
        got = -float(compiled.cp.objective(jnp.asarray(v.value)))
        assert got == pytest.approx(float(problem.objective.value), rel=1e-10)
    for point in (0.0, -1.0):
        x = jnp.full(3, point)
        assert np.isfinite(float(compiled.cp.objective(x)))
        assert np.all(np.isfinite(jax.grad(compiled.cp.objective)(x)))


def test_channel_capacity():
    # Binary symmetric channel: capacity C = 1 - H2(0.25) at the uniform
    # input, and the simplex row's dual is C - log2(e) (closed forms; Clarabel
    # is only ~1e-5 accurate on this dual).
    P = np.array([[0.75, 0.25], [0.25, 0.75]])
    x = cp.Variable(2)
    c = np.sum(P * np.log(P), axis=0)
    total = cp.sum(x) == 1
    problem = cp.Problem(
        cp.Maximize((c @ x + cp.sum(cp.entr(P @ x))) / np.log(2)), [total, x >= 0]
    )
    jcv.solve(problem, **TOL)
    capacity = 1 + 0.25 * np.log2(0.25) + 0.75 * np.log2(0.75)
    assert problem.status == cp.OPTIMAL
    assert problem.value == pytest.approx(capacity, abs=1e-8)
    np.testing.assert_allclose(x.value, [0.5, 0.5], atol=1e-6)
    assert total.dual_value == pytest.approx(capacity - np.log2(np.e), abs=1e-6)


@pytest.mark.parametrize(
    "build",
    [
        lambda x: cp.Problem(cp.Minimize(-cp.sum(cp.log(x)))),
        lambda x: cp.Problem(cp.Minimize(cp.sum(x)), [cp.SOC(cp.sum(x), x)]),
    ],
)
def test_unsupported(build):
    with pytest.raises(NotImplementedError):
        jcv.compile_problem(build(cp.Variable(3)))
