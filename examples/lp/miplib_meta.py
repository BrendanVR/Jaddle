# %% [markdown]
# # Solve a MIPLIB LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# %%
import os

import jax
import jax.numpy as jnp

jax.config.update("eager_constant_folding", True)

import jaddle.jaddle_linear as jl
import jaddle.jaddle_optimisers as jo
import jaddle.highs_helpers as hh
import optax
import highspy as hspy

# %%
jo.configure_jax("x64")

# %% [markdown]
# ## Load and presolve the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
PROBLEM_NAME = "stp3d"
highs = hspy.Highs()
highs.readModel(
    f"/home/brendanvr/python/Jaddle/data/{PROBLEM_NAME}.mps"
)  # path to MPS file


# %% [markdown]
# We convert the LP to Jaddle's sparse format.
highs_lp = highs.getLp()
jaddle_lp = jl.to_jaddle_sparse(hh.highs_to_standard_form_sparse(highs_lp))

# %%
k = 1e4

HEDGE_ETA = 1.0
LOSS_CLIP = 1000
EXPLORATION_RATE = 0.0

# Each player is a joint (primal, dual) pair sharing one learning-rate
# schedule; the ensemble selects coherent (primal, dual) strategies as units.
experts = [
    (
        optax.optimistic_gradient_descent(learning_rate=lr),
        optax.optimistic_gradient_descent(learning_rate=lr),
    )
    for lr in [1e-1]
]

ensemble_optimiser = jo.hedge_ensemble_saddle(
    lp=jaddle_lp,
    experts=experts,
    eta=HEDGE_ETA,
    loss_clip=LOSS_CLIP,
    exploration_rate=EXPLORATION_RATE,
    center_losses=True,
    per_expert_k=True,
    per_expert_k_lo=1 / k,
    per_expert_k_hi=k,
    per_expert_k_theta=0.1,
)

solution, _ = jl.solve(
    lp=jaddle_lp,
    optimiser=ensemble_optimiser,
    scale="ruiz+pc",
    scaled_objective=True,
    update_mode="synchronous",
    verbose=True,
    log_every=10,
    expert_diagnostics=True,
    # restarts=40,
    average="polyak",
    weight_function=lambda i: jax.lax.cond(
        i < int(5e4),
        lambda _: 1e-5,
        lambda _: 1.0,
        operand=None,
    ),
)

# %%
print(f"Primal Equality Residual: {jaddle_lp.eq_slack(solution.primal)}")
print(f"Primal Inequality Residual: {jaddle_lp.ineq_slack(solution.primal)}")
print("----------------------------------------------")

# %%
jaddle_lp.objective(solution.primal)

# %%
