# %% [markdown]
# # A semidefinite program in CVXPY: Max-Cut
# Split the nodes of a graph into two sides so that as many edges as possible
# cross between them. With `xᵢ = ±1` for the side of node `i`, an edge `(i, j)`
# is cut when `xᵢ xⱼ = -1`, so the cut size is `¼ Σ wᵢⱼ (1 - xᵢ xⱼ) = ¼ xᵀLx`
# with `L` the graph Laplacian. Finding the best `x` is NP-hard.
#
# Goemans and Williamson relax `X = xxᵀ` to any positive semidefinite matrix
# with a unit diagonal:
#
# ```
# maximise    ¼ tr(L X)
# subject to  diag(X) = 1,  X ⪰ 0
# ```
#
# The relaxation's optimum bounds the best cut from above. Cutting the vectors
# of a factorisation `X = VᵀV` with a random hyperplane rounds it to a cut that
# is, in expectation, at least 0.878 times the optimum.
#
# `jcv.JaddleConicSolver` solves the SDP with `jl.solve` on a PSD-constrained
# block of variables, projecting onto the PSD cone (one eigendecomposition)
# each step. The result is checked against Clarabel.

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
# ## A random graph
# 30 nodes, each pair joined with probability 0.3.

# %%
rng = np.random.default_rng(0)
n_nodes = 30
W = np.triu(rng.random((n_nodes, n_nodes)) < 0.3, 1).astype(float)
W = W + W.T
L = np.diag(W.sum(axis=1)) - W
print(f"{n_nodes} nodes, {int(W.sum() / 2)} edges")

# %% [markdown]
# ## The relaxation

# %%
X = cp.Variable((n_nodes, n_nodes), symmetric=True)
unit_diagonal = cp.diag(X) == 1
problem = cp.Problem(cp.Maximize(cp.trace(L @ X) / 4), [X >> 0, unit_diagonal])

problem.solve(solver="CLARABEL")
reference_bound = problem.value

problem.solve(solver=jcv.JaddleConicSolver(), tol=1e-6)
stats = problem.solver_stats.extra_stats
print(
    f"Status {problem.status}: bound {problem.value:.4f} "
    f"(Clarabel {reference_bound:.4f}), {stats['epochs']} epochs"
)
assert problem.status == cp.OPTIMAL
assert abs(problem.value - reference_bound) <= 1e-5 * (1 + abs(reference_bound))

# %% [markdown]
# ## Rounding to a cut
# Factor `X = VᵀV` (through its eigendecomposition, clipping the tiny negative
# eigenvalues a first-order solution can have) and put node `i` on the side of
# a random hyperplane its column `vᵢ` falls on. Keep the best of 100 tries.

# %%
lam, U = np.linalg.eigh(X.value)
V = (U * np.sqrt(np.maximum(lam, 0.0))).T


def cut_size(x):
    return float(x @ L @ x) / 4


best = max(
    (np.sign(rng.standard_normal(n_nodes) @ V) for _ in range(100)), key=cut_size
)
print(
    f"Best rounded cut: {cut_size(best):.0f} edges, "
    f"{cut_size(best) / problem.value:.3f} of the SDP bound"
)
assert cut_size(best) >= 0.878 * problem.value

# %% [markdown]
# The duals `y` of `diag(X) = 1` are a per-node certificate: by weak duality
# their sum bounds every cut, and at the optimum it equals the SDP bound.

# %%
dual_bound = float(unit_diagonal.dual_value.sum())
print(f"Dual bound: {dual_bound:.4f}")
assert abs(dual_bound - problem.value) <= 1e-4 * (1 + abs(problem.value))
