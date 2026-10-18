# %% [markdown]
# # Modelling with CVXPY, solving with Jaddle
# CVXPY can serve as Jaddle's modelling language. This example fits two sparse
# models on the same data, each written in a few lines of CVXPY:
#
# * the lasso, least squares plus an L1 penalty. CVXPY can put it in QP form,
#   so `prob.solve(solver=jcv.JaddleSolver())` hands Jaddle the matrices and
#   `jc.quadratic_program` solves it with its KKT scaling;
# * L1-penalised logistic regression. CVXPY would rewrite the logistic loss
#   into exponential cones, which Jaddle does not support, so `jcv.solve`
#   compiles the expression tree into JAX functions for `jc.solve` instead:
#   the loss stays smooth and only the L1 term becomes linear epigraph rows.
#
# Both are checked against Clarabel, the interior-point solver CVXPY ships with.

# %%
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cvxpy as cp
import numpy as np

import jaddle.jaddle_optimisers as jo

jo.configure_jax("float64")

import jaddle.jaddle_cvxpy as jcv  # noqa: E402

# %% [markdown]
# ## Data
# `n` samples of `p` correlated features; the true coefficients are nonzero on
# only `k` of them.

# %%
rng = np.random.default_rng(0)
n, p, k = 200, 50, 5
correlation = 0.5 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
X = rng.standard_normal((n, p)) @ np.linalg.cholesky(correlation).T
w_true = np.zeros(p)
support = rng.choice(p, k, replace=False)
w_true[support] = rng.choice([-1, 1], k) * rng.uniform(1, 3, k)
y = X @ w_true + 0.5 * rng.standard_normal(n)
labels = np.sign(X @ w_true + rng.standard_normal(n))


def report(name, problem, w, reference_value, reference_w):
    stats = problem.solver_stats
    found = set(np.flatnonzero(np.abs(w.value) > 1e-3))
    print(
        f"{name}: {problem.status}, objective {problem.value:.8f} "
        f"(Clarabel {reference_value:.8f}), max |w - w_Clarabel| "
        f"{np.max(np.abs(w.value - reference_w)):.1e}, {stats.num_iters} epochs, "
        f"{len(found)} nonzero coefficients, true support "
        f"{'recovered' if set(support) <= found else 'missed'}"
    )


# %% [markdown]
# ## Lasso: the QP route
# `tol` sets all three of Jaddle's relative tolerances. The other keyword
# arguments of `prob.solve` go to Jaddle (`max_epochs`, `verbose`, ...).

# %%
w = cp.Variable(p)
lasso = cp.Problem(cp.Minimize(cp.sum_squares(X @ w - y) / n + 0.1 * cp.norm1(w)))
lasso.solve(solver="CLARABEL")
reference_value, reference_w = lasso.value, w.value.copy()

lasso.solve(solver=jcv.JaddleSolver(), tol=1e-8)
report("lasso", lasso, w, reference_value, reference_w)

# %% [markdown]
# ## L1-penalised logistic regression: the compiled route
# `jcv.solve` writes the result back into the problem as CVXPY does, so
# `w.value`, `problem.value` and the constraints' `dual_value` read as usual.
# Its tolerances are absolute.

# %%
w = cp.Variable(p)
b = cp.Variable()
logistic = cp.Problem(
    cp.Minimize(
        cp.sum(cp.logistic(-cp.multiply(labels, X @ w + b))) / n + 0.02 * cp.norm1(w)
    )
)
logistic.solve(solver="CLARABEL")
reference_value, reference_w = logistic.value, w.value.copy()

jcv.solve(logistic, tol=1e-8)
report("logistic", logistic, w, reference_value, reference_w)

compiled = jcv.compile_problem(logistic)
print(
    f"compiled: {compiled.cp.num_variables} variables ({compiled.num_epigraph} "
    f"epigraph), {compiled.cp.num_ineq_constraints()} inequality rows"
)
data, _, _ = logistic.get_problem_data(cp.CLARABEL)
print(
    f"CVXPY's conic form for Clarabel: {data['A'].shape[1]} variables, "
    f"{data['A'].shape[0]} rows ({data['dims'].exp} exponential cones)"
)
