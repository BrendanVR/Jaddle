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

import optax
import highspy as hspy
import jaddle.jaddle_optimisers as jo
import jaddle.jaddle_linear as jl
import jaddle.highs_helpers as hh

# %%
jo.configure_jax("x64")

# %% [markdown]
# ## Load the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
PROBLEM_NAME = "ns1758913"  # name of MIPLIB problem (without .mps extension)
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


def optimiser(lr):
    primal = optax.inject_hyperparams(optax.sgd)(
        learning_rate=lr,
    )

    dual = optax.inject_hyperparams(optax.sgd)(
        learning_rate=lr,
    )

    return jo.create_saddle_optimiser(
        primal,
        dual,
    )


lr = optax.exponential_decay(
    init_value=1e0, decay_rate=0.9, transition_steps=5000, end_value=1e-4
)


# %%
print("Solving Problem:", PROBLEM_NAME)
jl.lp_summary_statistics(jaddle_lp)

# %%
solution_jaddle, _ = jl.solve(
    lp=jaddle_lp,
    optimiser=optimiser(1 / 2),
    scaled_objective=True,
    iterations_per_epoch=1000,
    scale="ruiz+pc",
    update_mode="per_iterate_k",
    per_iterate_k_hi=1e5,
    per_iterate_k_lo=1e-5,
    average=True,
    verbose=True,
    log_every=1,
)

print(f"Primal Equality Residual: {jaddle_lp.eq_slack(solution_jaddle.primal)}")
print(f"Primal Inequality Residual: {jaddle_lp.ineq_slack(solution_jaddle.primal)}")
print("----------------------------------------------")

# %%
