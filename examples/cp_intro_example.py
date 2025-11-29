# %% [markdown]
# # Introduction to Jaddle Convex Problem Solver
# This example demonstrates how to define and solve a simple convex problem using the Jaddle Convex library.
# We will formulate a convex problem with inequality and equality constraints and solve it using the SPS optimizer.

# %%
import jax.numpy as jnp
import jaddle.jaddle_convex as jc


# %% [markdown]
# ## Defining the Convex Problem
# We will define a simple convex problem:
# Minimize:     f(x) = 5 * sum_{i=1}^{4} x_i - 5 * sum_{i=1}^{4} x_i^2 - sum_{i=5}^{13} x_i
# Subject to:   g_i(x) <= 0 for i = 1,...,9
#               h(x) = 0
# where g_i(x) are a set of linear inequality constraints, and h(x) is a linear equality constraint.
def objective(x):
    # x is a vector of length 13
    return 5 * jnp.sum(x[:4]) - 5 * jnp.sum(x[:4] ** 2) - jnp.sum(x[4:13])


def contraints_ineq(x):
    return jnp.array(
        [
            2 * x[0] + 2 * x[1] + x[9] + x[10] - 10,
            2 * x[0] + 2 * x[2] + x[9] + x[11] - 10,
            2 * x[1] + 2 * x[2] + x[10] + x[11] - 10,
            -8 * x[0] + x[9],
            -8 * x[1] + x[10],
            -8 * x[2] + x[11],
            -2 * x[3] - x[4] + x[9],
            -2 * x[5] - x[6] + x[10],
            -2 * x[7] - x[8] + x[11],
        ]
    )


def constraints_eq(x):
    return jnp.array([x[1] + x[12] - 0.5])  # x[1] + x[12] = 0.5


lower_bounds = jnp.zeros(13)
upper_bounds = jnp.array([1.0] * 9 + [100.0] * 3 + [1.0])

# %% [markdown]
# ## Creating the Convex Problem Model
cp = jc.CP(
    num_variables=13,
    objective=objective,
    constraints_eq=constraints_eq,
    constraints_ineq=contraints_ineq,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
)

# %% [markdown]
# ## Solving the Convex Problem
solution = jc.solve(
    iterations_per_epoch=int(1e4),
    cp=cp,
    initial_solution=cp.initial_solution(),
    constraint_tolerance=1e-5,
    progress_tolerance=1e-5,
)

# %% [markdown
# ## Displaying the Solution
print("Optimal Objective Value:", objective(solution["primal"]))
print("Optimal Solution:", solution["primal"])
print(
    "Inequality Constraints (should be <= 0):",
    contraints_ineq(solution["primal"]).max(),
)
print("Equality Constraints (should be == 0):", constraints_eq(solution["primal"]))
# %%
