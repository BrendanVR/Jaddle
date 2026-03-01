# %% [markdown]
# # Solve a MIPLIB LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# We will use the `highspy` library to load a MIPLIB LP from an MPS file.
# %%
import os

# Suppress INFO and WARNING logs from XLA/JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=slow-operation-alarm"

import jax
import jax.numpy as jnp
import jaddle.jaddle_linear as jl
import jaddle.jaddle_optimisers as jo
import jaddle.highs_helpers as hh
import highspy as hspy
import optax

# %%
PROBLEM_NAME = "buildingenergy"  # name of the MIPLIB problem to load


# %% [markdown]
# ## Load and presolve the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
highs = hspy.Highs()
highs.readModel(
    f"/home/brendanvr/python/Jaddle/data/{PROBLEM_NAME}.mps"
)  # path to MPS file

# %% [markdown]
# We convert the LP to Jaddle's sparse format.
highs_lp = highs.getLp()
jaddle_lp = jl.to_jaddle_sparse(hh.highs_to_standard_form_sparse(highs_lp))


# %%
def lr(decay_rate, transition_steps=int(5e4)):
    return optax.exponential_decay(
        init_value=1e0,
        transition_steps=transition_steps,
        decay_rate=decay_rate,
        end_value=1e-5,
    )


learning_rates = [
    lr(decay_rate, transition_steps)
    for decay_rate in [0.5, 0.9, 0.99]
    for transition_steps in [int(5e4), int(1e5), int(1e3)]
]
primal_experts = [
    optax.optimistic_adam_v2(learning_rate=lr, alpha=0.05) for lr in learning_rates
]
dual_experts = [optax.adadelta(learning_rate=1.0)]

ensemble_optimiser = jo.hedge_ensemble_saddle(
    primal_experts=primal_experts,
    dual_experts=dual_experts,
    primal_eta=1e-3,
    dual_eta=1e-3,
    loss_clip=1e3,
)

# %%
solution = jl.solve(
    lp=jaddle_lp,
    optimiser=ensemble_optimiser,
    verbose=True,
    expert_diagnostics=True,
    iterations_per_epoch=int(5e4),
    weight_function=lambda i: jax.lax.select(i <= int(5e4), 1e-16, 1.0),
    scale="pc",
)

# %%
print(f"Primal Equality Residual: {jaddle_lp.eq_slack(solution.primal)}")
print(f"Primal Inequality Residual: {jaddle_lp.ineq_slack(solution.primal)}")
print("----------------------------------------------")

# %%
