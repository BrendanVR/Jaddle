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

jax.config.update("eager_constant_folding", True)

import jaddle.jaddle_linear as jl
import jaddle.highs_helpers as hh
import jaddle.jaddle_optimisers as jo
import highspy as hspy
import optax

# %%
PROBLEM_NAME = "stp3d"
jax_mode = "max_speed"
gpu = "y"
scale = "ruiz+pc"

if jax_mode in ["balanced", "safe", "max_speed"]:
    jo.configure_jax(jax_mode)
else:
    print("Invalid JAX mode. Using default precision.")

if gpu == "y":
    jax.config.update("jax_platform_name", "gpu")
else:
    jax.config.update("jax_platform_name", "cpu")


# %% [markdown]
# ## Load the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
highs = hspy.Highs()
highs.readModel(
    f"/home/brendanvr/python/Jaddle/data/{PROBLEM_NAME}.mps"
)  # path to MPS file


# %% [markdown]
# We convert the LP to Jaddle's sparse format, before applying the selected scaling strategy.
highs_lp = highs.getLp()
jaddle_lp = jl.to_jaddle_sparse(hh.highs_to_standard_form_sparse(highs_lp))

# %% [markdown]
# ## Solve the presolved LP using Jaddle's saddle point solver
jl.lp_summary_statistics(jaddle_lp)

learning_rate = optax.cosine_decay_schedule(
    init_value=5e-1,
    decay_steps=int(5e5),
    alpha=1e-4,
)

optimiser = jo.create_saddle_optimiser(
    optax.optimistic_gradient_descent(learning_rate=learning_rate)
)
solution, _ = jl.solve(
    lp=jaddle_lp,
    optimiser=optimiser,
    scale=scale,
    scaled_objective=True,
    dual_gap_tolerance=1e-2,
    update_mode="alternating",
    average="off",
)

# %%
print(f"Primal Equality Residual: {jaddle_lp.eq_slack(solution.primal)}")
print(f"Primal Inequality Residual: {jaddle_lp.ineq_slack(solution.primal)}")
print("----------------------------------------------")

# %%
