# %% [markdown]
# # An exponential cone program in CVXPY: the Kelly portfolio
# A Kelly (log-optimal) investor chooses long-only weights `w` to maximise the
# expected logarithm of wealth after one period, which maximises the long-run
# growth rate when the bet is repeated:
#
# ```
# maximise    (1/N) Σᵢ log(rᵢᵀw)
# subject to  1ᵀw = 1,  0 <= w <= cap
# ```
#
# Each `rᵢ` is a scenario of gross returns (1.05 means +5%). CVXPY writes
# each `log` as one exponential cone, `(uᵢ, 1, rᵢᵀw) ∈ K_exp`, i.e.
# `uᵢ <= log(rᵢᵀw)`, so the model has one cone per scenario. `jcv.solve`'s
# compiled route cannot take `log` (its iterates respect only the box and
# would leave the domain); `jcv.JaddleConicSolver` keeps the cone variables
# inside the cone at every step instead.
#
# Full Kelly is famously aggressive, so the sweep below caps each asset's
# weight. Only the cap's value changes, so every solve after the first reuses
# the compiled program. The results are checked against Clarabel.

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
# ## Scenarios
# 20 assets with different means and volatilities over 1000 one-period
# scenarios, driven by a common market factor.

# %%
rng = np.random.default_rng(0)
n_assets, n_scenarios = 20, 1000
mean = rng.uniform(0.00, 0.08, n_assets)
vol = rng.uniform(0.05, 0.40, n_assets)
beta = rng.uniform(0.3, 1.2, n_assets)
market = rng.normal(0.0, 0.15, (n_scenarios, 1))
noise = rng.standard_normal((n_scenarios, n_assets))
R = 1.0 + mean + beta * market + vol * noise
R = np.maximum(R, 0.05)  # a gross return can't fall below zero

# %% [markdown]
# ## Model

# %%
w = cp.Variable(n_assets, nonneg=True)
cap = cp.Parameter(nonneg=True)
growth = cp.sum(cp.log(R @ w)) / n_scenarios
weight_cap = w <= cap
problem = cp.Problem(cp.Maximize(growth), [cp.sum(w) == 1, weight_cap])
# Clarabel gets its own copy, so solving the two in turn does not clear
# CVXPY's cache (and with it Jaddle's warm start).
reference = cp.Problem(cp.Maximize(growth), [cp.sum(w) == 1, weight_cap])

# %% [markdown]
# ## Growth against concentration
# The growth rate is the expected log return per period. The cap's dual is the
# growth given up per unit of cap, summed over the assets it binds.

# %%
print(
    f"{'cap':>6s}{'growth':>10s}{'Clarabel':>10s}{'assets':>8s}{'largest':>9s}"
    f"{'shadow':>8s}{'epochs':>8s}{'Jaddle s':>10s}{'Clarabel s':>12s}"
)
for limit in [1.0, 0.5, 0.3, 0.2, 0.1]:
    cap.value = limit
    start = time.perf_counter()
    reference.solve(solver="CLARABEL")
    clarabel_time = time.perf_counter() - start
    reference_growth, reference_w = reference.value, w.value.copy()

    start = time.perf_counter()
    problem.solve(solver=jcv.JaddleConicSolver(), tol=1e-7)
    jaddle_time = time.perf_counter() - start
    held = int(np.sum(w.value > 1e-4))
    print(
        f"{limit:6.2f}{problem.value:10.5f}{reference_growth:10.5f}{held:8d}"
        f"{w.value.max():9.3f}{float(np.sum(weight_cap.dual_value)):8.3f}"
        f"{problem.solver_stats.extra_stats['epochs']:8d}"
        f"{jaddle_time:10.3f}{clarabel_time:12.3f}"
    )

    assert problem.status == cp.OPTIMAL
    assert abs(problem.value - reference_growth) <= 1e-5 * (1 + abs(reference_growth))
    assert np.allclose(w.value, reference_w, atol=1e-3)

# %% [markdown]
# A tighter cap spreads the bet over more assets at a small cost in growth;
# the first solve pays for compiling, and the rest reuse it.
