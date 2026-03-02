# %% [markdown]
# # Isotonic Regression with Jaddle
# This example demonstrates how to perform modified isotonic regression using Jaddle.
# Isotonic regression is a type of regression that fits a non-decreasing function to the data. Here we impose an extra mean constraint
# on the solution.

# %%
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaddle.jaddle_convex as jc
import jaddle.jaddle_optimisers as jo

jo.configure_jax("max_speed")

import optax

# %% [markdown]
# ## Generate Synthetic Data
# We will create synthetic data that follows a cubic relationship with some added noise.
n = 1000
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
    return jnp.array([0.0])  # No equality constraints in this example


lower_bounds = -np.ones(n)
upper_bounds = jnp.ones(n)

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
primal_lr = optax.exponential_decay(
    init_value=1e0,
    transition_steps=1000,
    decay_rate=0.9,
    end_value=1e-5,
    staircase=True,
)

optimiser = jo.create_saddle_optimiser(
    optax.optimistic_adam_v2(primal_lr, alpha=0.05),
    optax.optimistic_adam_v2(primal_lr, alpha=0.05),
)

solution, _ = jc.solve(
    cp,
    optimiser=optimiser,
    average=False,
    update_mode="synchronous",
)


# %%
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Noisy data", marker="o", linestyle="", alpha=0.2)
plt.plot(
    x,
    solution.primal,
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
y_pred = solution.primal
ineq_violations = constraints_ineq(y_pred)
eq_violations = constraints_eq(y_pred)
print("Max Inequality Constraint Violation (should be <= 0):", cp.ineq_slack(y_pred))
print("Max Equality Constraint Violation (should be == 0):", cp.eq_slack(y_pred))
print("Optimal Objective Value:", objective(y_pred))

# %%
