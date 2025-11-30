# %% [markdown]
# # Introduction to Jaddle Linear Programming Solver
# This example demonstrates how to define and solve a simple linear programming problem using the Jaddle library.

# %%
import jax
import numpy as np
import jaddle.jaddle_linear as jl

jax.config.update("jax_platform_name", "cpu")  # Using CPU for toy problem

# %% [markdown]
# ## Defining the Linear Programming Problem
# We will define a simple LP problem in standard form.
# Minimize:     c^T x
# Subject to:   A_eq x = b_eq
#               A_ineq x <= b_ineq

c = np.array([3, 2])  # Objective coefficients

A_eq = np.array([[0, 0]])  # No equality constraints
b_eq = np.array([0])

A_ineq = np.array([[-1, -1]])  # Inequality constraints
b_ineq = np.array([-4])  # x1 + x2 >= 4 in standard form

lower_bounds = np.array([0, 0])  # Non-negativity constraints
upper_bounds = np.array([np.inf, np.inf])  # No upper bounds

# %% [markdown]
# ## Creating the LP Model
lp = jl.LP(
    c=c,
    A_eq=A_eq,
    b_eq=b_eq,
    A_ineq=A_ineq,
    b_ineq=b_ineq,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
)

# %% [markdown]
# ## Solving the LP Problem
solution = jl.solve(lp)
# %%
print(f"x1 = {solution['primal'][0]:.4f}, x2 = {solution['primal'][1]:.4f}")
print(f"Optimal objective value: {lp.objective(solution['primal']):.4f}")

# %%
