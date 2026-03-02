# %% [markdown]
# # Solve a Scaled LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# We will use the `highspy` library to load a MIPLIB LP from an MPS file.

# %%
import os

# Suppress INFO and WARNING logs from XLA/JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax
import jax.numpy as jnp

# jax.config.update(
#     "jax_platform_name", "cpu"
# )  # Using CPU for this example, as the problem is not large and we want to avoid GPU overhead
# jax.config.update(
#     "jax_enable_x64", True
# )  # Use 64-bit precision for better numerical stability

import jaddle.jaddle_linear as jl
import jaddle.highs_helpers as hh
import jaddle.jaddle_optimisers as jo

jo.configure_jax("max_speed")

import highspy as hspy
import optax

# %% [markdown]
# ## Load the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
highs = hspy.Highs()
highs.readModel("/home/brendanvr/python/Jaddle/data/ns1758913.mps")  # path to MPS file

# %% [markdown]
# We convert the LP to Jaddle's sparse format, before applying ruiz scaling.
highs_lp = highs.getLp()
jaddle_lp = jl.to_jaddle_sparse(hh.highs_to_standard_form_sparse(highs_lp))

# %%
lr = optax.exponential_decay(
    1e1,
    transition_steps=1000,
    decay_rate=0.9,
    end_value=1e-4,
)

optimiser = jo.optimistic_adam_saddle(lr, lr)

# %% [markdown]
# ## Solve the presolved LP using Jaddle's saddle point solver
# Using warm restarts: 5 restart cycles, starting with 10 epochs per cycle,
# doubling the cycle length each time (geometric growth).
solution, _ = jl.solve(
    lp=jaddle_lp,
    optimiser=optimiser,
    # scale="ruiz+pc",
    update_mode="synchronous",
    weight_function=lambda i: jax.lax.select(i < int(3e4), 1e-8, 1.0),
)

# %%
print(f"Primal Equality Residual: {jaddle_lp.eq_slack(solution.primal)}")
print(f"Primal Inequality Residual: {jaddle_lp.ineq_slack(solution.primal)}")
print("----------------------------------------------")

# %%
