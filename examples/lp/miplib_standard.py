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
PROBLEM_NAME = input("Enter the MIPLIB problem name: ")
jax_mode = input("Set JAX mode (balanced/safe/max_speed): ")

if jax_mode in ["balanced", "safe", "max_speed"]:
    jo.configure_jax(jax_mode)
else:
    print("Invalid JAX mode. Using default precision.")
gpu = input("Use GPU? (y/n): ").lower() == "y"
if gpu:
    jax.config.update("jax_platform_name", "gpu")
else:
    jax.config.update("jax_platform_name", "cpu")

float_precision = input("Use 32-bit precision? (y/n): ").lower() == "y"
if float_precision:
    jax.config.update("jax_enable_x64", False)
else:
    jax.config.update("jax_enable_x64", True)

scale = input("Scaling strategy (ruiz/pc/ruiz+pc): ").lower()

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

# %%


def slingshot_lr(step):
    return 1e-1 * jnp.cos(jnp.pi * step / 1000) * jnp.exp(-step / 1000) + 1e-3


optimiser = jo.create_saddle_optimiser(
    optax.sgd(learning_rate=slingshot_lr),
    optax.sgd(learning_rate=lambda step: -slingshot_lr(step)),
)

# %% [markdown]
# ## Solve the presolved LP using Jaddle's saddle point solver
jl.lp_summary_statistics(jaddle_lp)


solution, _ = jl.solve(
    lp=jaddle_lp,
    optimiser=optimiser,
    verbose=True,
    log_every=1,
    scale=scale,
    scaled_objective=True,
)

# %%
print(f"Primal Equality Residual: {jaddle_lp.eq_slack(solution.primal)}")
print(f"Primal Inequality Residual: {jaddle_lp.ineq_slack(solution.primal)}")
print("----------------------------------------------")

# %%
