# %% [markdown]
# # Solve a MIPLIB LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# %%
import os

import jax
import jax.numpy as jnp

jax.config.update("eager_constant_folding", True)
jax.config.update("jax_bcoo_cusparse_lowering", True)

import jaddle.jaddle_linear as jl
import jaddle.jaddle_optimisers as jo
import jaddle.highs_helpers as hh
import optax
import highspy as hspy

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

presolve = input("Presolve the problem using Highs? (y/n): ").lower() == "y"

# %% [markdown]
# ## Load and presolve the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
highs = hspy.Highs()
highs.readModel(
    f"/home/brendanvr/python/Jaddle/data/{PROBLEM_NAME}.mps"
)  # path to MPS file


# %% [markdown]
# We convert the LP to Jaddle's sparse format.
if presolve:
    highs.presolve()
    highs_lp = highs.getPresolvedLp()
else:
    highs_lp = highs.getLp()
jaddle_lp = jl.to_jaddle_sparse(hh.highs_to_standard_form_sparse(highs_lp))

# %%
learning_rates = [
    optax.cosine_decay_schedule(
        init_value=1e-1,
        decay_steps=decay_steps,
        exponent=exponent,
        alpha=1e-5,
    )
    for exponent in [1, 2, 3]
    for decay_steps in [int(1e3), int(1e4), int(1e5)]
]

HEDGE_ETA = 1.0
LOSS_CLIP = 20.0
EXPLORATION_RATE = 0.05

primal_experts = [optax.sgd(learning_rate=lr) for lr in learning_rates]
dual_experts = [optax.sgd(learning_rate=lr) for lr in learning_rates]
ensemble_optimiser = jo.hedge_ensemble_saddle(
    lp=jaddle_lp,
    primal_experts=primal_experts,
    dual_experts=dual_experts,
    primal_eta=HEDGE_ETA,
    dual_eta=HEDGE_ETA,
    loss_clip=LOSS_CLIP,
    exploration_rate=EXPLORATION_RATE,
    center_losses=True,
)

solution, _ = jl.solve(
    lp=jaddle_lp,
    optimiser=ensemble_optimiser,
    average="polyak",
    weight_function=lambda i: jax.lax.select(i <= int(5e4), 1e-16, 1.0),
    scale="ruiz",
    scaled_objective=True,
    update_mode="alternating",
    verbose=True,
    log_every=1,
    expert_diagnostics=True,
)

# %%
print(f"Primal Equality Residual: {jaddle_lp.eq_slack(solution.primal)}")
print(f"Primal Inequality Residual: {jaddle_lp.ineq_slack(solution.primal)}")
print("----------------------------------------------")

# %%
