# %% [markdown]
# # Solve a MIPLIB LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# We will use the `highspy` library to load a MIPLIB LP from an MPS file.
# %%
import os

# Suppress INFO and WARNING logs from XLA/JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax
import jaddle.jaddle_linear as jl
import jaddle.jaddle_optimisers as jo
import jaddle.highs_helpers as hh
import highspy as hspy
import optax

# %% [markdown]
# ## Load and presolve the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
highs = hspy.Highs()
highs.readModel("/home/brendanvr/python/Jaddle/data/stp3d.mps")  # path to MPS file

# %% [markdown]
# We convert the LP to Jaddle's sparse format.
highs_lp = highs.getLp()
jaddle_lp = jl.to_jaddle_sparse(hh.highs_to_standard_form_sparse(highs_lp))

# %%
lr_fast = optax.exponential_decay(
    init_value=1e0,
    transition_steps=1000,
    decay_rate=0.9,
    end_value=1e-4,
)
lr_slow = optax.exponential_decay(
    init_value=1e0,
    transition_steps=1000,
    decay_rate=0.99,
    end_value=1e-4,
)

primal_experts = [
    optax.adagrad(learning_rate=lr_fast),
    optax.adagrad(learning_rate=lr_slow),
]

dual_experts = [
    optax.adagrad(learning_rate=lr_fast),
    optax.adagrad(learning_rate=lr_slow),
]

ensemble_optimiser = jo.hedge_ensemble_saddle(
    primal_experts=primal_experts,
    dual_experts=dual_experts,
    mode="bandit",  # options: "all_experts", "bandit"
)

# %% [markdown]
# ## Solve the presolved LP using Jaddle's saddle point solver
solution = jl.solve(
    lp=jaddle_lp,
    optimiser=ensemble_optimiser,
    verbose=True,
    expert_diagnostics=True,
)

# %%
print("----------------------------------------------")
print(f"Primal objective value: {jaddle_lp.objective(solution.primal)}")
print(f"Primal Equality Residual: {jaddle_lp.eq_slack(solution.primal)}")
print(f"Primal Inequality Residual: {jaddle_lp.ineq_slack(solution.primal)}")
print("----------------------------------------------")

# %%
