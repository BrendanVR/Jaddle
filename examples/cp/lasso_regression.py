# %% [markdown]
# # Lasso Regression with Jaddle
# This example fits a sparse linear model with the lasso: least squares plus an
# L1 penalty that drives most coefficients exactly to zero. The data come from a
# model where only a handful of the features matter, and we check that the
# lasso finds them.

# %%
import jax
from jax import vmap
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import jaddle.jaddle_convex as jc
import jaddle.jaddle_optimisers as jo
from sklearn.linear_model import Lasso
import time
import functools as ft

jo.configure_jax("float32")

# %% [markdown]
# ## Generate Synthetic Data
# `n` samples of `p` correlated features, with a true coefficient vector that
# is nonzero on only `k` of them.
rng = np.random.default_rng(0)
n, p, k = 200, 50, 5

correlation = 0.5 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
X = rng.standard_normal((n, p)) @ np.linalg.cholesky(correlation).T
w_true = np.zeros(p)
w_true[rng.choice(p, k, replace=False)] = rng.choice([-1, 1], k) * rng.uniform(1, 3, k)
y = X @ w_true + 0.5 * rng.standard_normal(n)


# %% [markdown]
# ## Define the Convex Problem
# The lasso is
#
#     minimise    1/(2n) ||Xw − y||² + α ||w||₁
#
# The L1 norm has a kink at zero, so we split each coefficient into its
# positive and negative parts, `w = u − v` with `u, v >= 0`. At the optimum at
# most one of `u[i]`, `v[i]` is nonzero, so `||w||₁ = sum(u + v)`, and the
# problem becomes smooth with only bound constraints:
#
#     minimise    1/(2n) ||X(u − v) − y||² + α sum(u + v)
#     subject to  u, v >= 0
alpha = 0.1

X_jax = jnp.array(X)
y_jax = jnp.array(y)


def objective(z):
    u, v = z[:p], z[p:]
    residual = X_jax @ (u - v) - y_jax
    return 0.5 * jnp.mean(residual**2) + alpha * jnp.sum(z)


cp = jc.JaddleCP(
    num_variables=2 * p,
    objective=objective,
    constraints_eq=lambda z: jnp.zeros(0),
    constraints_ineq=lambda z: jnp.zeros(0),
    lower_bounds=jnp.zeros(2 * p),
    upper_bounds=jnp.full(2 * p, jnp.inf),
)

# %% [markdown]
# ## Solve the problem using Jaddle
result = jc.solve(
    cp,
    primal_grad_norm_tolerance=1e-4,
    verbose=True,
)

z = np.asarray(result["solution"].primal)
w_jaddle = z[:p] - z[p:]

# %% [markdown]
# ## Solve the problem using scikit-learn's Lasso
# scikit-learn's `Lasso` minimises the same objective, so the two fits should
# agree.
start_time = time.time()
sklearn_lasso = Lasso(
    alpha=alpha,
    fit_intercept=False,
    tol=1e-8,
)
sklearn_lasso.fit(X, y)
end_time = time.time()
print("Scikit-learn Lasso time:", end_time - start_time)
w_sklearn = sklearn_lasso.coef_


def lasso_objective(w):
    return 0.5 * np.mean((X @ w - y) ** 2) + alpha * np.sum(np.abs(w))


print("Converged:", result["converged"])
print("Jaddle objective:      ", lasso_objective(w_jaddle))
print("Scikit-learn objective:", lasso_objective(w_sklearn))
print("Max coefficient difference:", np.max(np.abs(w_jaddle - w_sklearn)))
print("True support:  ", np.flatnonzero(w_true))
print("Jaddle support:", np.flatnonzero(np.abs(w_jaddle) > 1e-3))

# %%
plt.figure(figsize=(10, 6))
plt.stem(w_true, linefmt="C7-", markerfmt="C7o", basefmt=" ", label="True")
plt.plot(
    w_sklearn,
    "s",
    color="C0",
    markersize=9,
    fillstyle="none",
    label="Scikit-learn Lasso",
)
plt.plot(w_jaddle, "x", color="C3", markersize=7, label="Jaddle Lasso")
plt.axhline(0, color="k", linewidth=0.5)
plt.xlabel("Feature index")
plt.ylabel("Coefficient")
plt.title("Lasso Regression using Jaddle")
plt.legend()
plt.show()


# %% [markdown]
# ## The whole regularization path in one batched solve
# To solve many problems with the same shapes, write the problem as a function
# of its parameters, `build_cp(alpha)`. `jc.make_solver` turns it into a pure
# JAX function `params -> result` that can be `vmap`-ed over a batch of
# parameters and `jit`-ed. Batch members run in lockstep until the last one
# converges, so set `max_epochs` as a budget.
#
# Under `vmap`, `X @ (u - v)` becomes a matrix-matrix product, which the
# float32 profile runs in TF32 (about 3 significant digits). That noise puts a
# floor on the gradient norm, so we ask for full-precision matmuls here.
#
# `alpha` can be a scalar or one penalty per feature (shape `(p,)`); the
# penalty term broadcasts either way. The per-feature form is used at the end.


def build_cp(alpha):
    def objective(z):
        u, v = z[:p], z[p:]
        residual = jnp.matmul(X_jax, u - v, precision="highest") - y_jax
        return 0.5 * jnp.mean(residual**2) + jnp.sum(alpha * (u + v))

    return jc.JaddleCP(
        num_variables=2 * p,
        objective=objective,
        constraints_eq=lambda z: jnp.zeros(0),
        constraints_ineq=lambda z: jnp.zeros(0),
        lower_bounds=jnp.zeros(2 * p),
        upper_bounds=jnp.full(2 * p, jnp.inf),
    )


solve_path = jax.jit(vmap(jc.make_solver(build_cp)))
alphas = jnp.logspace(-1, 1, 100)
path = solve_path(alphas)

Z = np.asarray(path.solution.primal)  # shape (len(alphas), 2p)
W_path = Z[:, :p] - Z[:, p:]
print("Converged:", int(path.converged.sum()), "/", len(alphas))

# %%
plt.figure(figsize=(10, 6))
plt.plot(alphas, W_path)
plt.xscale("log")
plt.xlabel("alpha")
plt.ylabel("Coefficient")
plt.title("Lasso regularization path using Jaddle")
plt.show()


# %% [markdown]
# ## Choosing alpha: grid search
# The fits above use a hand-picked `alpha`. The usual way to choose it is to
# fit on the training data and score each fit on held-out validation data. We
# also draw a large test set from the same model, used only to report the
# final results. The noise variance is 0.25, so no model can get a test loss
# `1/2 mean((Xw − y)²)` below 0.125.
#
# The tuning solves need tighter tolerances than the path above: the
# gradients in the next section are only as accurate as the solutions.
def sample(m):
    X = rng.standard_normal((m, p)) @ np.linalg.cholesky(correlation).T
    return X, X @ w_true + 0.5 * rng.standard_normal(m)


X_val, y_val = sample(n)
X_test, y_test = sample(5 * n)


def loss(X, y, W):
    """Half mean squared error of each row of `W` (or of a single `w`)."""
    return 0.5 * np.mean((X @ np.atleast_2d(W).T - y[:, None]) ** 2, axis=0)


tight = dict(primal_grad_norm_tolerance=1e-5, max_epochs=200)
solve_grid = jax.jit(vmap(jc.make_solver(build_cp, **tight)))
alpha_grid = jnp.logspace(-3, 0, 60)
grid = solve_grid(alpha_grid)

Z = np.asarray(grid.solution.primal)
val_grid = loss(X_val, y_val, Z[:, :p] - Z[:, p:])
alpha_best = float(alpha_grid[np.argmin(val_grid)])
print("Converged:", int(grid.converged.sum()), "/", len(alpha_grid))
print(f"Grid search: alpha = {alpha_best:.4f}, validation loss = {val_grid.min():.4f}")


# %% [markdown]
# ## Choosing alpha: gradient descent
# `jc.make_solution(build_cp)` returns the solution `z*(alpha)` as a function
# that JAX can differentiate. The gradient does not backpropagate through the
# solver's iterations. It differentiates the optimality (KKT) conditions at
# the solution instead: the coefficients that are zero stay at their bound,
# and the nonzero ones move with `alpha` as the solution of a small linear
# system. So the validation loss is a differentiable function of `alpha`,
# smooth wherever the set of selected features does not change, and we can
# minimise it with gradient descent. We work with `log(alpha)` so that alpha
# stays positive.
#
# First a check that the gradient matches a finite difference.
lasso_solution = jc.make_solution(build_cp, **tight)
X_val_jax, y_val_jax = jnp.array(X_val), jnp.array(y_val)


def validation_loss(log_alpha):
    z = lasso_solution(jnp.exp(log_alpha))
    residual = jnp.matmul(X_val_jax, z[:p] - z[p:], precision="highest") - y_val_jax
    return 0.5 * jnp.mean(residual**2)


validation_loss_and_grad = jax.jit(jax.value_and_grad(validation_loss))

log_alpha, eps = jnp.log(0.1), 1e-2
finite_difference = (
    validation_loss(log_alpha + eps) - validation_loss(log_alpha - eps)
) / (2 * eps)
print("Gradient:         ", float(validation_loss_and_grad(log_alpha)[1]))
print("Finite difference:", float(finite_difference))


# %%
def tune(log_alpha, steps, learning_rate):
    """Minimise the validation loss over `log_alpha` with Adam. The step size
    decays to zero: the minimum sits in a flat valley, where Adam at a fixed
    step size overshoots and then creeps back."""
    optimiser = optax.adam(optax.cosine_decay_schedule(learning_rate, steps))
    opt_state = optimiser.init(log_alpha)
    history = []
    for _ in range(steps):
        value, grad = validation_loss_and_grad(log_alpha)
        history.append((np.exp(np.asarray(log_alpha)), float(value)))
        updates, opt_state = optimiser.update(grad, opt_state)
        log_alpha = optax.apply_updates(log_alpha, updates)
    return log_alpha, history


start_time = time.time()
log_alpha_scalar, history = tune(jnp.log(1.0), steps=50, learning_rate=0.3)
print("Tuning time:", time.time() - start_time)
print(
    f"Gradient descent: alpha = {float(jnp.exp(log_alpha_scalar)):.4f},"
    f" validation loss = {history[-1][1]:.4f}"
)


# %% [markdown]
# ## One alpha per feature
# Gradient descent finds the same alpha as the grid search. Its real advantage
# is that it scales to many hyperparameters. Give every feature its own
# penalty `alpha_j` (the adaptive lasso): with 50 penalties a grid search is
# out of the question, but the gradient costs about the same as before.
# We start from the best single alpha.
#
# A feature whose coefficient is zero has zero gradient with respect to its
# own penalty: a small change in `alpha_j` does not move the solution. So only
# the penalties of selected features move, and the others keep their starting
# value.
#
# Fifty hyperparameters fitted on 200 validation points could overfit the
# validation set, so we judge the result on the test set.
start_time = time.time()
log_alpha_features, _ = tune(
    jnp.full(p, jnp.log(alpha_best)), steps=100, learning_rate=0.1
)
print("Tuning time:", time.time() - start_time)

fits = {
    "Grid search, one alpha": alpha_best,
    "Gradient descent, one alpha": jnp.exp(log_alpha_scalar),
    "Gradient descent, alpha per feature": jnp.exp(log_alpha_features),
}
print(f"{'':37s} {'validation':>10s} {'test':>7s} {'features':>8s}")
for name, alpha in fits.items():
    z = np.asarray(lasso_solution(alpha))
    w = z[:p] - z[p:]
    print(
        f"{name:37s} {loss(X_val, y_val, w)[0]:10.4f}"
        f" {loss(X_test, y_test, w)[0]:7.4f}"
        f" {np.sum(np.abs(w) > 1e-3):8d}"
    )
print(f"{'Best possible':37s} {'':10s} {0.125:7.4f} {k:8d}")

# %% [markdown]
# On the left, gradient descent walks down the validation curve to the grid
# search's minimum. On the right, irrelevant features that the single-alpha
# fit selected get larger penalties until they drop out, and most true
# features get smaller ones, so their coefficients are shrunk less. Features
# left unselected throughout keep the starting value.

# %%
fig, (ax_alpha, ax_penalty) = plt.subplots(1, 2, figsize=(14, 5))
alphas_seen, losses_seen = np.array(history).T
ax_alpha.plot(alpha_grid, val_grid, color="C7", label="Grid search")
ax_alpha.scatter(
    alphas_seen,
    losses_seen,
    c=np.arange(len(history)),
    cmap="viridis",
    zorder=3,
    label="Gradient descent (colour = step)",
)
ax_alpha.set_xscale("log")
ax_alpha.set_yscale("log")
ax_alpha.set_xlabel("alpha")
ax_alpha.set_ylabel("Validation loss")
ax_alpha.set_title("Tuning one alpha")
ax_alpha.legend()

alpha_features = np.exp(np.asarray(log_alpha_features))
in_support = w_true != 0
ax_penalty.bar(
    np.flatnonzero(~in_support),
    alpha_features[~in_support],
    color="C7",
    label="Irrelevant feature",
)
ax_penalty.bar(
    np.flatnonzero(in_support),
    alpha_features[in_support],
    color="C3",
    label="True feature",
)
ax_penalty.axhline(
    alpha_best,
    color="k",
    linestyle="--",
    linewidth=1,
    label="Starting value (best single alpha)",
)
ax_penalty.set_yscale("log")
ax_penalty.set_xlabel("Feature index")
ax_penalty.set_ylabel("alpha_j")
ax_penalty.set_title("Tuned penalty per feature")
ax_penalty.legend()
plt.show()

# %%
