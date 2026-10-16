# %% [markdown]
# # Kernel SVM with Jaddle
# This example trains a support vector machine with an RBF kernel on a small
# two-dimensional problem that no straight line can separate.
# The SVM's dual is a quadratic program, so we solve it with `jc.quadratic_program`.

# %%
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=5"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax

jax.config.update("jax_optimization_level", "O3")


import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaddle.jaddle_convex as jc
import jaddle.jaddle_optimisers as jo
from functools import partial

jo.configure_jax("float32")

# %% [markdown]
# ## Generate Synthetic Data
# Two interleaved half-moons, one per class, with labels `y` in {-1, +1}.
rng = np.random.default_rng(0)
n = 1000
theta = rng.uniform(0, np.pi, n // 2)
upper_moon = np.column_stack([np.cos(theta), np.sin(theta)])
lower_moon = np.column_stack([1 - np.cos(theta), 0.5 - np.sin(theta)])
X = np.vstack([upper_moon, lower_moon]) + 0.15 * rng.standard_normal((n, 2))
y = np.concatenate([np.ones(n // 2), -np.ones(n // 2)])


# %% [markdown]
# ## Define the Quadratic Program
# With the RBF kernel `K(a, b) = exp(-gamma * ||a - b||^2)`, the SVM dual is
#
#     minimise    ½ αᵀQα − 1ᵀα,   Q[i, j] = y[i] y[j] K(X[i], X[j])
#     subject to  yᵀα = 0
#                 0 <= α <= C
#
# with one variable per training point. `C` limits how much any one point can
# pull on the boundary, and `gamma` sets how local the kernel is.
def rbf_kernel(A, B, gamma):
    squared_distances = (
        np.sum(A**2, axis=1)[:, None] - 2 * A @ B.T + np.sum(B**2, axis=1)[None, :]
    )
    return np.exp(-gamma * squared_distances)


# %% [markdown]
# ## Solve the problem using Jaddle
C = 1
gamma = 2.0
Q = np.outer(y, y) * rbf_kernel(X, X, gamma)
Q = jnp.array(Q)
y = jnp.array(y)


# Low rank approximation Q ≈ F Fᵀ using the Nystroem method:
# Q ≈ C W⁺ Cᵀ, where C = Q[:, idx] and W = Q[idx, idx] for k sampled landmarks.
@partial(jax.jit, static_argnames=("k",))
def nystroem(A, k, key=jax.random.PRNGKey(0)):
    n = A.shape[0]
    idx = jax.random.choice(key, n, (k,), replace=False)
    C = A[:, idx]  # n x k
    W = C[idx, :]  # k x k
    W = (W + W.T) / 2
    w, V = jnp.linalg.eigh(W)
    tol = jnp.max(jnp.abs(w)) * k * jnp.finfo(A.dtype).eps
    inv_sqrt = jnp.where(w > tol, 1.0 / jnp.sqrt(jnp.maximum(w, tol)), 0.0)
    return C @ (V * inv_sqrt)  # n x k, so that Q ≈ F Fᵀ


rank = 30
F = nystroem(Q, rank)

objective = lambda alpha: 0.5 * jnp.linalg.multi_dot([alpha, F, F.T, alpha]) - jnp.sum(
    alpha
)
constraints_eq = lambda alpha: jnp.array([jnp.dot(y, alpha)])
constraints_ineq = lambda _: jnp.zeros(0)
lower_bounds = jnp.zeros(n)
upper_bounds = jnp.full(n, C)

problem = jc.JaddleCP(
    num_variables=n,
    objective=objective,
    constraints_eq=constraints_eq,
    constraints_ineq=constraints_ineq,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
)

# %%
result = jc.solve(
    problem,
    verbose=True,
    update_mode="extragradient",
)

alpha = np.asarray(result["solution"].primal)
# The intercept is the multiplier of the constraint yᵀα = 0.
intercept = float(result["solution"].dual_eq[0])


# %% [markdown]
# ## The Classifier
# A new point is classified by the sign of
# `f(x) = sum_i alpha[i] y[i] K(X[i], x) + intercept`. Only the support
# vectors, the points with `alpha > 0`, take part.
def decision_function(points):
    return rbf_kernel(points, X, gamma) @ (alpha * y) + intercept


support = alpha > 1e-6 * C
print("Converged:", result["converged"])
print("Support vectors:", int(support.sum()), "of", n)
print("Training accuracy:", np.mean(np.sign(decision_function(X)) == y))

# %%
grid_x, grid_y = np.meshgrid(np.linspace(-1.7, 2.7, 300), np.linspace(-1.3, 1.8, 300))
scores = decision_function(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
scores = scores.reshape(grid_x.shape)

plt.figure(figsize=(10, 6))
plt.contourf(
    grid_x, grid_y, scores, levels=[-np.inf, 0, np.inf], alpha=0.15, colors=["C0", "C1"]
)
plt.contour(
    grid_x, grid_y, scores, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"]
)
plt.scatter(*X[y > 0].T, color="C1", label="Class +1")
plt.scatter(*X[y < 0].T, color="C0", label="Class -1")
plt.scatter(
    *X[support].T, s=120, facecolors="none", edgecolors="k", label="Support vectors"
)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("RBF Kernel SVM using Jaddle")
plt.legend()
plt.show()

# %%
