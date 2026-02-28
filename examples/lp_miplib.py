# %% [markdown]
# # Solve a MIPLIB LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# We will use the `highspy` library to load a MIPLIB LP from an MPS file.
# %%
import os

# Suppress INFO and WARNING logs from XLA/JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax
import jax.numpy as jnp
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
def lr(decay_rate):
    return optax.exponential_decay(
        init_value=1e0,
        transition_steps=1000,
        decay_rate=decay_rate,
        end_value=1e-4,
    )


learning_rates = [lr(decay_rate) for decay_rate in [0.6, 0.7, 0.8, 0.9, 0.99]]
primal_experts = [
    optax.optimistic_adam_v2(learning_rate=lr, alpha=0.05) for lr in learning_rates
]
dual_optimiser = optax.adadelta(learning_rate=1.0)
ensemble_optimiser = jo.hedge_ensemble_saddle(
    primal_experts=primal_experts,
    dual_experts=[dual_optimiser],
    lp=jaddle_lp,
)

# %%
solution, opt_state = jl.solve(
    lp=jaddle_lp,
    optimiser=ensemble_optimiser,
    output_opt_state=True,
    max_epochs=1,
    iterations_per_epoch=5000,
)

# Evaluate all experts and find the best
print("Evaluating experts...")
print(f"Primal Weights = {jax.nn.softmax(opt_state.primal.log_weights)}")
print(f"Dual Weights = {jax.nn.softmax(opt_state.dual.log_weights)}")
best_expert_idx_primal = jnp.argmax(opt_state.primal.log_weights)
best_expert_idx_dual = jnp.argmax(opt_state.dual.log_weights)
print(f"Best primal expert index: {best_expert_idx_primal}")
print(f"Best dual expert index: {best_expert_idx_dual}")

# Solve with the best performing expert
solution = jl.solve(
    jaddle_lp,
    initial_solution=solution,
    optimiser=jo.create_saddle_optimiser(
        primal_experts[best_expert_idx_primal], dual_experts[best_expert_idx_dual]
    ),
    weight_function=lambda i: jax.lax.select(i <= int(5e4), 1e-16, 1.0),
)

# %%
print(f"Primal Equality Residual: {jaddle_lp.eq_slack(solution.primal)}")
print(f"Primal Inequality Residual: {jaddle_lp.ineq_slack(solution.primal)}")
print("----------------------------------------------")

# %%
