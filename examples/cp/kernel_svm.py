# %% [markdown]
# # Kernel SVM with Jaddle
# This example trains a support vector machine with an RBF kernel on a small
# two-dimensional problem that no straight line can separate.
# The SVM's dual is a quadratic program, so we solve it with `jc.quadratic_program`.

# %%
import numpy as np
import matplotlib.pyplot as plt
import jaddle.jaddle_convex as jc
import jaddle.jaddle_optimisers as jo

jo.configure_jax("float64")

# %% [markdown]
# ## Generate Synthetic Data
# Two interleaved half-moons, one per class, with labels `y` in {-1, +1}.
rng = np.random.default_rng(0)
n = 500
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


C = 1.0
gamma = 2.0
Q = np.outer(y, y) * rbf_kernel(X, X, gamma)

# %% [markdown]
# ## Solve the problem using Jaddle
result = jc.quadratic_program(
    Q,
    -np.ones(n),
    A_eq=y[None, :],
    b_eq=np.zeros(1),
    lower_bounds=np.zeros(n),
    upper_bounds=np.full(n, C),
    primal_feasibility_tolerance=1e-4,
    dual_feasibility_tolerance=1e-4,
    dual_gap_tolerance=1e-4,
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
print("Relative duality gap:", result["certificate"]["relative_gap"])

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
