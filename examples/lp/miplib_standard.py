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
PROBLEM_NAME = "momentum1"  # name of MIPLIB problem (without .mps extension)
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


def primal(lr):
    return optax.inject_hyperparams(optax.sgd)(
        learning_rate=lr,
        nesterov=True,
    )


def dual(lr):
    return optax.inject_hyperparams(optax.sgd)(
        learning_rate=lr,
        nesterov=True,
        momentum=0.5,
    )


# PDHG convergence requires σ·τ·‖A‖² ≤ 1; with symmetric steps η = σ = τ
# the safe upper bound is η ≤ 1/‖A‖.
k_max = 1e3
eta_max = 1e3
base_lr = 1.0 / 2
optimiser = jo.create_saddle_optimiser(
    primal(base_lr),
    dual(base_lr),
)

# %%
jl.lp_summary_statistics(jaddle_lp)
solution, _ = jl.solve(
    lp=jaddle_lp,
    optimiser=optimiser,
    iterations_per_epoch=10000,
    scale="ruiz+pc",
    verbose=True,
    log_every=1,
    extragradient=True,
    per_iterate_k_theta=0.001,
    per_iterate_k_lo=1 / k_max,
    per_iterate_k_hi=k_max,
    # per_iterate_eta=True,
    # per_iterate_eta_theta=0.001,
    # per_iterate_eta_lo=1 / eta_max,
    # per_iterate_eta_hi=eta_max,
    restarts=100,
    epochs_per_restart=10,
    restart_multiplier=1.1,
    iterations_per_epoch_decay=0.9,
    iterations_per_epoch_min=100,
    dual_gap_tolerance=1,
    primal_grad_norm_tolerance=1e-4,
    complementarity_slack_tolerance=1e-4,
    average="polyak",
)

print(f"Primal Equality Residual: {jaddle_lp.eq_slack(solution.primal)}")
print(f"Primal Inequality Residual: {jaddle_lp.ineq_slack(solution.primal)}")
print("----------------------------------------------")

# %%
jaddle_lp.objective(solution.primal)

# %%
sol = jl.project_onto_eq(jaddle_lp, solution.primal, 1e-3)

# %%
print(f"Primal Equality Residual: {jaddle_lp.eq_slack(sol)}")
print(f"Primal Inequality Residual: {jaddle_lp.ineq_slack(sol)}")
print("----------------------------------------------")
print(f"Objective: {jaddle_lp.objective(sol)}")

# %%
