# %% [markdown]
# # Solve a MIPLIB LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# We will use the `highspy` library to load a MIPLIB LP from an MPS file.
# The LP will then be presolved to simplify the problem before applying saddle point optimisation.

# %%
import os

# Suppress INFO and WARNING logs from XLA/JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax
import jax.numpy as jnp

# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_default_dtype_bits", "64")

# Ensure JAX is properly initialized
_ = jax.random.normal(jax.random.PRNGKey(0), (1,)).block_until_ready()

import time
import jaddle.jaddle_linear as jl
import jaddle.jaddle_optimisers as jo
import jaddle.highs_helpers as hh
import highspy as hspy
import optax

# %% [markdown]
# ## Load and presolve the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
# The LP is then presolved to reduce its size and complexity.
# Finally, we convert the presolved LP into a format compatible with Jaddle.
highs = hspy.Highs()
highs.readModel("/home/brendanvr/python/Jaddle/data/stp3d.mps")  # path to MPS file
# highs.presolve()
highs_lp = highs.getLp()
jaddle_lp = hh.highs_to_standard_form_sparse(highs_lp)
highs_lp = hh.highs_from_standard_form_sparse(jaddle_lp)

# %% [markdown]
# ## Solve the scaled, presolved LP using Jaddle's saddle point solver
lr_primal = optax.exponential_decay(
    init_value=1e0,
    transition_steps=1000,
    decay_rate=0.9,
    end_value=1e-4,
)


def lr_dual(step):
    return 1 / lr_primal(step)


solution = jl.solve(
    lp=jaddle_lp,
    optimiser=jo.adamdelta_saddle(
        lr_primal=lr_primal,
        lr_dual=1e0,
    ),
    iterations_per_epoch=1000,
    max_epochs=5000,
    verbose=False,
    exponential_weighting=0.01,
)

print("--------------------------------")
print("Saddle point solver objective:", jaddle_lp.objective(solution.primal))
print("Inequality violation:", jaddle_lp.ineq_slack(solution.primal))
print("Equality violation:", jaddle_lp.eq_slack(solution.primal))
print("--------------------------------")


# %%
