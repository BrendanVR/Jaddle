# %% [markdown]
# # Channel capacity with CVXPY and Jaddle
# A discrete memoryless channel takes an input symbol `X ∈ {1, ..., n}` to an
# output symbol `Y ∈ {1, ..., m}` with transition probabilities
# `P[i, j] = Prob(Y = i | X = j)`, so each column of `P` sums to one. Its
# capacity is the largest mutual information `I(X; Y)` over input
# distributions `x` (Boyd & Vandenberghe, *Convex Optimization*, exercise 4.57).
# With `y = P x` the output distribution,
#
#     I(X; Y) = Σ_j x_j Σ_i P_ij log₂ P_ij  −  Σ_i y_i log₂ y_i
#             = c @ x + Σ_i entr(y_i) / log 2,
#
# linear plus concave in `x`, so the capacity is a concave maximisation over
# the probability simplex. This is CVXPY's "channel capacity" example.
#
# CVXPY would rewrite `entr` into exponential cones, which Jaddle does not
# support, so `jcv.solve` compiles the expression tree into JAX functions
# instead and hands `jc.solve` the smooth objective as it is. Jaddle's
# iterates respect only the box `x >= 0`, not `entr`'s domain, so the
# compiler continues `entr` below `jcv.ENTR_FLOOR` (1e-8) by its tangent line.
# That changes nothing at a solution whose outputs all have probability above
# the floor, which holds for any channel without zero entries.

# %%
import os
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cvxpy as cp
import numpy as np
from scipy.special import xlogy

import jaddle.jaddle_optimisers as jo

jo.configure_jax("float64")

import jaddle.jaddle_cvxpy as jcv  # noqa: E402

# %% [markdown]
# ## The model
# As on CVXPY's page: `c[j] = Σ_i P_ij log₂ P_ij` (with `0 log 0 = 0`), the
# mutual information as the objective, and the simplex as constraints.


# %%
def channel_capacity(P):
    n = P.shape[1]
    x = cp.Variable(n)
    y = P @ x
    c = np.sum(xlogy(P, P), axis=0) / np.log(2)
    mutual_information = c @ x + cp.sum(cp.entr(y)) / np.log(2)
    total = cp.sum(x) == 1
    problem = cp.Problem(cp.Maximize(mutual_information), [total, x >= 0])
    return problem, x, total


def blahut_arimoto(P, iterations=10_000):
    """The classical fixed-point algorithm for the capacity, as a check."""
    x = np.full(P.shape[1], 1 / P.shape[1])
    for _ in range(iterations):
        y = P @ x
        divergence = np.sum(xlogy(P, P) - xlogy(P, y[:, None]), axis=0)
        x = x * np.exp(divergence)
        x /= x.sum()
    y = P @ x
    divergence = np.sum(xlogy(P, P) - xlogy(P, y[:, None]), axis=0)
    return x @ divergence / np.log(2), x


def report(name, problem, x, reference, reference_x):
    print(
        f"{name}: {problem.status}, capacity {problem.value:.10f} bits, "
        f"|C - C_ref| {abs(problem.value - reference):.1e}, max |x - x_ref| "
        f"{np.max(np.abs(x.value - reference_x)):.1e}"
    )


# %% [markdown]
# ## A binary symmetric channel
# Each bit is flipped with probability 0.25. By symmetry the uniform input is
# optimal and the capacity is `1 − H₂(0.25)`, with `H₂` the binary entropy.

# %%
np.set_printoptions(precision=4)
P = np.array([[0.75, 0.25], [0.25, 0.75]])
problem, x, total = channel_capacity(P)
jcv.solve(problem, tol=1e-8)

flip = 0.25
exact = 1 + flip * np.log2(flip) + (1 - flip) * np.log2(1 - flip)
print(f"1 - H2(0.25) = {exact:.10f} bits")
report("Jaddle", problem, x, exact, np.array([0.5, 0.5]))
print("optimal input distribution x =", x.value)

# %% [markdown]
# ## A random channel
# Eight inputs and six outputs, each column drawn from a Dirichlet
# distribution. The reference is the Blahut–Arimoto algorithm run to
# convergence; Clarabel, the interior-point solver CVXPY ships with, solves
# the same problem through its exponential-cone form for comparison.
#
# The capacity has a well-known optimality condition: every input the optimal
# `x` uses has the same divergence `D(P_j ‖ y)` from the output distribution,
# equal to the capacity, and the unused ones have less. It is the KKT
# condition of the problem above, so the dual of `sum(x) == 1` (the value of
# one more unit of probability) is `C − log₂ e`.

# %%
rng = np.random.default_rng(0)
input, output = 6, 10
P = rng.dirichlet(np.full(output, 0.5), size=input).T
problem, x, total = channel_capacity(P)

reference, reference_x = blahut_arimoto(P)
print(f"Blahut-Arimoto: capacity {reference:.10f} bits")

start = time.perf_counter()
problem.solve(solver="CLARABEL")
clarabel_time = time.perf_counter() - start
report("Clarabel", problem, x, reference, reference_x)
print(f"Clarabel solve time: {clarabel_time:.4f} s")

start = time.perf_counter()
jcv.solve(problem, tol=1e-8)
jaddle_time = time.perf_counter() - start
report("Jaddle", problem, x, reference, reference_x)
print(f"Jaddle solve time (including JAX compilation): {jaddle_time:.4f} s")

y = P @ x.value
divergence = np.sum(xlogy(P, P) - xlogy(P, y[:, None]), axis=0) / np.log(2)
print("optimal input distribution x =", x.value)
print("divergence D(P_j || y), bits =", divergence)
print(
    f"dual of sum(x) == 1: {total.dual_value:.6f}, "
    f"C - log2(e) = {problem.value - np.log2(np.e):.6f}"
)


# %%
