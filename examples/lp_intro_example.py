# %% [markdown]
# # Introduction to Jaddle Linear Programming Solver
# This example demonstrates how to define and solve a simple linear programming problem using the Jaddle library.

# %%
import jax
import numpy as np
import optax
import scipy.sparse as sp
import jaddle.jaddle_linear as jl
import jaddle.jaddle_optimisers as jo

jax.config.update("jax_platform_name", "cpu")  # Using CPU for toy problem

# %% [markdown]
# ## Defining the Linear Programming Problem
# We will define a simple LP problem in standard form.
# Minimize:     c^T x
# Subject to:   A_eq x = b_eq
#               A_ineq x <= b_ineq

c = np.array([3, 2])  # Objective coefficients

A_eq = sp.csc_matrix([[0, 0]])  # No equality constraints
b_eq = np.array([0])

A_ineq = sp.csc_matrix([[-1, -1]])  # Inequality constraints
b_ineq = np.array([-4])  # x1 + x2 >= 4 in standard form

lower_bounds = np.array([0, 0])  # x >= 0
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
lp = jl.to_jaddle_sparse(lp)  # Convert to Jaddle sparse format


# %% [markdown]
# ## Solving the LP Problem
lr = optax.exponential_decay(
    init_value=1e0,
    transition_steps=1000,
    decay_rate=0.9,
    end_value=1e-5,
)

solution, _ = jl.solve(
    lp,
    optimiser=jo.optimistic_adam_saddle(lr, lr),
    verbose=True,
    average=False,
    update_mode="alternating",
)
# %%
print(f"x1 = {solution.primal[0]:.4f}, x2 = {solution.primal[1]:.4f}")
print(f"Optimal objective value: {lp.objective(solution.primal):.4f}")
# %%
