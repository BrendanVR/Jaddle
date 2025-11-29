# %% [markdown]
# # Isotonic Regression with SPS Optimizer
# This example demonstrates how to perform modified isotonic regression using the SPS optimizer from the Jaddle Convex library.
# Isotonic regression is a type of regression that fits a non-decreasing function to the data. Here we impose an extra mean constraint
# on the solution.

# %%
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaddle.jaddle_convex as jc

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
    return jnp.array([y_pred.mean() - 0.1])  # mean of y_pred should be 0.1


lower_bounds = -jnp.inf * jnp.ones(n)
upper_bounds = jnp.inf * jnp.ones(n)

cp = jc.CP(
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
    iterations_per_epoch=int(1e4),
    cp=cp,
    initial_solution=cp.initial_solution(),
    constraint_tolerance=1e-5,
    progress_tolerance=1e-4,
    complementarity_tolerance=1e-3,
)

# %%
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Noisy data", marker="o", linestyle="", alpha=0.2)
plt.plot(
    x,
    solution["primal"],
    label="Modified isotonic regression solution",
    color="red",
    linewidth=2,
)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Modified Isotonic Regression using Jaddle")
plt.legend()
plt.show()

# %% [markdown]
# Verify that the solution satisfies the constraints
y_pred = solution["primal"]
ineq_violations = constraints_ineq(y_pred)
eq_violations = constraints_eq(y_pred)
print("Max Inequality Constraint Violation (should be <= 0):", ineq_violations.max())
print("Equality Constraint Violation (should be == 0):", eq_violations)
print("Optimal Objective Value:", objective(y_pred))

# %%
