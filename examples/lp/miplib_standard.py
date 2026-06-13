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

jax.config.update("eager_constant_folding", True)
jax.config.update("jax_bcoo_cusparse_lowering", True)

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
highs.presolve()
highs_lp = highs.getPresolvedLp()
jaddle_lp = jl.to_jaddle_sparse(hh.highs_to_standard_form_sparse(highs_lp))

# %% [markdown]
# ## Solve the presolved LP using Jaddle's saddle point solver
jl.lp_summary_statistics(jaddle_lp)


learning_rate = optax.exponential_decay(
    init_value=1e-1, transition_steps=5000, decay_rate=0.95, end_value=1e-6
)

optimiser = jo.create_saddle_optimiser(
    optax.optimistic_adam_v2(learning_rate=learning_rate, alpha=0.1),
    optax.optimistic_adam_v2(learning_rate=learning_rate, alpha=0.1),
)
solution, _ = jl.solve(
    lp=jaddle_lp,
    optimiser=optimiser,
    scale=scale,
    scaled_objective=True,
    restarts=30,
    epochs_per_restart=10,
    restart_multiplier=1.3,
    dual_gap_tolerance=1e0,
    verbose=True,
    log_every=1,
)

# %%
print(f"Primal Equality Residual: {jaddle_lp.eq_slack(solution.primal)}")
print(f"Primal Inequality Residual: {jaddle_lp.ineq_slack(solution.primal)}")
print("----------------------------------------------")

# %%
