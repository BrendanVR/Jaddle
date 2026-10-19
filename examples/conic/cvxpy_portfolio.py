# %% [markdown]
# # A second-order cone program in CVXPY: a risk-limited portfolio
# Choose long-only weights `w` for 50 assets to maximise expected return while
# keeping the portfolio's volatility under a limit `σ`:
#
# ```
# maximise    μᵀw
# subject to  1ᵀw = 1,  w >= 0
#             ‖ [Fᵀw ; d ∘ w] ‖ <= σ
# ```
#
# Returns follow a factor model: `F` (assets x factors) carries the risk the
# assets share and `d` their own risk, so the volatility is the norm of the
# factor exposures stacked on the idiosyncratic ones. That norm limit is a
# second-order cone constraint, which an LP solver cannot take.
#
# `jcv.JaddleConicSolver` solves it with `jl.solve` on cone-constrained
# variables: each step projects onto the cone instead of a box. Sweeping `σ`
# traces the efficient frontier, warm-starting each solve from the last. The
# results are checked against Clarabel (an interior-point solver bundled with
# CVXPY).

# %%
import os
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cvxpy as cp
import numpy as np

import jaddle.jaddle_optimisers as jo

jo.configure_jax("float64")

import jaddle.jaddle_cvxpy as jcv  # noqa: E402

# %% [markdown]
# ## Data

# %%
rng = np.random.default_rng(0)
n_assets, n_factors = 500, 50
mu = rng.normal(0.06, 0.04, n_assets)  # expected annual returns
F = rng.normal(0.0, 0.08, (n_assets, n_factors))  # factor loadings
d = rng.uniform(0.05, 0.25, n_assets)  # idiosyncratic volatilities

# %% [markdown]
# ## Model
# `σ` is a CVXPY parameter, so the sweep below only changes its value. Declaring
# `w` nonnegative, rather than adding the constraint `w >= 0`, makes CVXPY pass
# it to Jaddle as bounds on `w`, not as 5000 inequality rows.

# %%
w = cp.Variable(n_assets, nonneg=True)
sigma = cp.Parameter(nonneg=True)
volatility = cp.norm(cp.hstack([F.T @ w, cp.multiply(d, w)]))
risk_limit = volatility <= sigma
constraints = [cp.sum(w) == 1, risk_limit]
problem = cp.Problem(cp.Maximize(mu @ w), constraints)
# SCS gets its own copy: solving one problem with two solvers in turn would
# clear CVXPY's cache, and with it Jaddle's warm start.
reference = cp.Problem(cp.Maximize(mu @ w), constraints)

# %% [markdown]
# ## The efficient frontier
# A risk limit's dual is the return gained per unit of extra volatility allowed,
# the slope of the frontier there. Each solve after the first warm-starts from
# the last (CVXPY's `warm_start=True` is the default).

# %%
print(
    f"{'σ':>6s}{'return':>10s}{'CLARABEL':>10s}{'slope':>8s}{'assets':>8s}{'epochs':>8s}"
    f"{'JADDLE s':>10s}{'CLARABEL s':>10s}"
)
for limit in [0.06, 0.08, 0.10, 0.14, 0.20]:
    sigma.value = limit
    start = time.perf_counter()
    reference.solve(solver="CLARABEL")
    clarabel_time = time.perf_counter() - start
    reference_return, reference_w = reference.value, w.value.copy()

    start = time.perf_counter()
    problem.solve(
        solver=jcv.JaddleConicSolver(),
        tol=1e-6,
        warm_start=True,
        diag_clamp=1e8,
    )
    jaddle_time = time.perf_counter() - start
    held = int(np.sum(w.value > 1e-4))
    epochs = problem.solver_stats.extra_stats["epochs"]
    print(
        f"{limit:6.2f}{problem.value:10.5f}{reference_return:10.5f}"
        f"{float(risk_limit.dual_value):8.3f}{held:8d}{epochs:8d}"
        f"{jaddle_time:10.3f}{clarabel_time:10.3f}"
    )

    assert problem.status == cp.OPTIMAL
    assert abs(problem.value - reference_return) <= 1e-5 * (1 + abs(reference_return))
    assert np.allclose(w.value, reference_w, atol=1e-3)

# %% [markdown]
# As the limit loosens, the portfolio concentrates in fewer, higher-return
# assets and each extra unit of volatility buys less return.
