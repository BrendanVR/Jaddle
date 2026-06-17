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
PROBLEM_NAME = "stp3d"  # name of MIPLIB problem (without .mps extension)
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


def primal(lr):
    return optax.inject_hyperparams(optax.optimistic_gradient_descent)(
        learning_rate=lr,
    )


def dual(lr):
    return optax.inject_hyperparams(optax.optimistic_gradient_descent)(
        learning_rate=lr,
    )


k = 1e2
eta = 1e4
base_lr = 1e-4
optimiser = jo.create_saddle_optimiser(
    primal(base_lr),
    dual(base_lr),
)


jl.lp_summary_statistics(jaddle_lp)
solution, _ = jl.solve(
    lp=jaddle_lp,
    optimiser=optimiser,
    iterations_per_epoch=1000,
    scale="ruiz+pc",
    verbose=True,
    log_every=10,
    per_iterate_k=True,
    per_iterate_k_theta=0.1,
    per_iterate_k_lo=1 / k,
    per_iterate_k_hi=k,
    per_iterate_eta=True,
    per_iterate_eta_theta=0.1,
    per_iterate_eta_lo=1 / eta,
    per_iterate_eta_hi=eta,
    eta_init=1.0,
    restarts=100,
    epochs_per_restart=10,
    restart_multiplier=1.0,
    iterations_per_epoch_decay=0.9,
    iterations_per_epoch_min=100,
    average="polyak",
)

print(f"Primal Equality Residual: {jaddle_lp.eq_slack(solution.primal)}")
print(f"Primal Inequality Residual: {jaddle_lp.ineq_slack(solution.primal)}")
print("----------------------------------------------")

# %%
jaddle_lp.objective(solution.primal)

# %%
