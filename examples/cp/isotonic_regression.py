# %% [markdown]
# # Isotonic Regression with Jaddle
# This example demonstrates how to perform isotonic regression using Jaddle.
# Isotonic regression is a type of regression that fits a non-decreasing function to the data.

# %%
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=5"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax
import jaddle.jaddle_optimisers as jo

jo.configure_jax("float64")

import jax.experimental.sparse as jsp
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import jaddle.jaddle_convex as jc
import jaddle.jaddle_linear as jl

from sklearn.isotonic import IsotonicRegression

# %% [markdown]
# ## Generate Synthetic Data
# We will create synthetic data that follows a cubic relationship with some added noise.
n = 500
x = np.linspace(-1, 1, n)
y = x**3
y += 0.1 * np.random.randn(n)  # add noise


# %% [markdown]
# ## Define the Convex Problem
# The objective is to minimize the squared error between the predicted values and the observed values,
# subject to the isotonic constraint (non-decreasing order).
def objective(y_pred):
    return jnp.sum((y_pred - y) ** 2)


def constraints_ineq(y_pred):
    return y_pred[:-1] - y_pred[1:]  # y_pred[i] <= y_pred[i+1]


def constraints_eq(y_pred):
    return jnp.zeros(0)  # No equality constraints in this example


lower_bounds = -jnp.ones(n)
upper_bounds = jnp.ones(n)

cp = jc.JaddleCP(
    num_variables=n,
    objective=objective,
    constraints_eq=constraints_eq,
    constraints_ineq=constraints_ineq,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
)

# %% [markdown]
# ## Solve the problem using Jaddle Convex SPS optimizer
solution = jc.solve(
    cp,
    primal_feasibility_tolerance=1e-6,
)["solution"]

# %% [markdown]
# ## Solve the problem using scikit-learn's Isotonic Regression
ir = IsotonicRegression(increasing=True)
start_time = time.time()
y_pred_sklearn = ir.fit_transform(x, y)
end_time = time.time()
print("Scikit-learn Isotonic Regression Time:", end_time - start_time)

# %%
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Noisy data", marker="o", linestyle="", alpha=0.2)
plt.plot(
    x,
    y_pred_sklearn,
    label="Scikit-learn Isotonic Regression",
    color="blue",
    linewidth=2,
)
plt.plot(
    x,
    solution.primal,
    label="Jaddle Isotonic Regression solution",
    color="red",
    linewidth=2,
)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Isotonic Regression using Jaddle")
plt.legend()
plt.show()

# %% [markdown]
# Verify that the solution satisfies the constraints
y_pred = solution.primal
ineq_violations = constraints_ineq(y_pred)
print("Max Inequality Constraint Violation (should be <= 0):", cp.ineq_slack(y_pred))
print("Optimal Objective Value:", objective(y_pred))


# %% [markdown]
# ## Isotonic Quantile Regression using Jaddle
# Minimize the pinball (quantile) loss subject to a non-decreasing constraint.
def isotonic_quantile_regression(tau):
    # LP formulation: z = [y_pred, u, v], with u, v >= 0 the positive/negative
    # parts of the residual: y - y_pred = u - v
    # minimize tau * sum(u) + (1 - tau) * sum(v)
    # Written in standard LP form (assumed API):
    #   min c^T z  s.t.  A_eq z = b_eq,  A_ineq z <= b_ineq,  lb <= z <= ub
    eye = jnp.eye(n)
    diff = eye[:-1] - eye[1:]  # y_pred[i] - y_pred[i+1] <= 0
    c = jnp.concatenate([jnp.zeros(n), tau * jnp.ones(n), (1 - tau) * jnp.ones(n)])
    A_eq = jnp.hstack([eye, eye, -eye])
    b_eq = jnp.asarray(y)
    A_ineq = jnp.hstack([diff, jnp.zeros((n - 1, 2 * n))])
    b_ineq = jnp.zeros(n - 1)

    problem = jl.LP(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ineq=A_ineq,
        b_ineq=b_ineq,
        lower_bounds=jnp.concatenate([lower_bounds, jnp.zeros(2 * n)]),
        upper_bounds=jnp.concatenate([upper_bounds, 10.0 * jnp.ones(2 * n)]),
    )
    return jl.to_jaddle_sparse(problem)


quantile_fitter = jax.jit(jl.make_solver(isotonic_quantile_regression))

# %%
quantiles = [0.01, 0.5, 0.99]
quantile_problems = [isotonic_quantile_regression(tau) for tau in quantiles]

start = time.time()
quantile_fits = jax.vmap(quantile_fitter)(jnp.array(quantiles))
end = time.time()
print("Isotonic Quantile Regression using Jaddle Time:", end - start)
quantile_predictions = quantile_fits.solution.primal[:, :n]

# %%
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Noisy data", marker="o", linestyle="", alpha=0.2)
for i, tau in enumerate(quantiles):
    plt.plot(
        x,
        quantile_predictions[i],
        label=f"Isotonic quantile {tau:.1f}",
        linewidth=2,
    )
plt.xlabel("x")
plt.ylabel("y")
plt.title("Isotonic Quantile Regression using Jaddle")
plt.legend()
plt.show()

# %%
