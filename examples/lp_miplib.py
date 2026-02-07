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

# Ensure JAX is properly initialized
_ = jax.random.normal(jax.random.PRNGKey(0), (1,)).block_until_ready()

import time
import jaddle.jaddle_linear as jl
import jaddle.highs_helpers as hh
import highspy as hspy
import optax

# %% [markdown]
# ## Load and presolve the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
# The LP is then presolved to reduce its size and complexity.
# Finally, we convert the presolved LP into a format compatible with Jaddle.
highs = hspy.Highs()
highs.readModel("/home/brendanvr/python/Jaddle/data/bab1.mps")  # path to MPS file
highs.presolve()
highs_lp = highs.getPresolvedLp()
jaddle_lp = hh.highs_to_standard_form_sparse(highs_lp)

# %%
primal_optimiser = optax.chain(
    optax.optimistic_adam_v2(learning_rate=1e0, alpha=5e-2),
    optax.contrib.reduce_on_plateau(
        factor=0.9,
        patience=500,
        min_scale=1e-4,
        cooldown=100,
    ),
)

dual_optimiser = optax.adadelta(1.0)


# %% [markdown]
# ## Solve the scaled, presolved LP using Jaddle's saddle point solver
start_time = time.time()
solution_primal, solution_dual = jl.solve(
    primal_optimiser=primal_optimiser,
    dual_optimiser=dual_optimiser,
    iterations_per_epoch=500,
    # constraint_tolerance=1e-6,
    lp=jaddle_lp,
    scale_A=True,
    scale_b=True,
    scale_c=True,
    max_epochs=5000,
    verbose=False,
)

print("--------------------------------")
print("Time to solution:", time.time() - start_time)
print("Saddle point solver objective:", jaddle_lp.objective(solution_primal.primal))
print("Inequality violation:", jaddle_lp.ineq_slack(solution_primal.primal))
print("Equality violation:", jaddle_lp.eq_slack(solution_primal.primal))
print("--------------------------------")

# %%
