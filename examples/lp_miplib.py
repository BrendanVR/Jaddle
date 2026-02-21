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

jax.config.update("jax_platform_name", "cpu")

# Ensure JAX is properly initialized
_ = jax.random.normal(jax.random.PRNGKey(0), (1,)).block_until_ready()

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
highs.readModel("/home/brendanvr/python/Jaddle/data/boeing.mps")  # path to MPS file

model = highs.getModel().lp_
model.integrality_ = [
    hspy.HighsVarType.kContinuous
] * model.num_col_  # Remove integer constraints to get the LP relaxation
highs.setOptionValue("solver", "pdlp")
highs.setOptionValue("primal_feasibility_tolerance", 1e-3)
highs.setOptionValue("pdlp_optimality_tolerance", 1e-3)

highs.passModel(model)
highs.solve()

# %%
highs_lp = highs.getLp()
jaddle_lp = hh.highs_to_standard_form_sparse(highs_lp)


# %% [markdown]
# ## Solve the presolved LP using Jaddle's saddle point solver
lr_primal = optax.exponential_decay(
    init_value=1e0,
    transition_steps=10000,
    decay_rate=0.9,
    end_value=1e-4,
    staircase=True,
)

optimiser = jo.create_saddle_optimiser(
    optax.optimistic_adam_v2(
        learning_rate=lr_primal,
        alpha=0.05,
    ),
    optax.adadelta(1.0),
)

tol = 1e-4
solution = jl.solve(
    lp=jaddle_lp,
    optimiser=optimiser,
    iterations_per_epoch=10000,
    max_epochs=100,
    progress_tolerance=tol,
    complementarity_tolerance=tol,
    constraint_tolerance=tol,
    weight_function=lambda i: jnp.sqrt(i),
)

# %%
print("--------------------------------")
print("Saddle point solver objective:", jaddle_lp.objective(solution.primal))
print("Inequality violation:", jaddle_lp.ineq_slack(solution.primal))
print("Equality violation:", jaddle_lp.eq_slack(solution.primal))
print("--------------------------------")

# %%

# %%
solution.primal
# %%
