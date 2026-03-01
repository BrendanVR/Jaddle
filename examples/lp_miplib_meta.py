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
PROBLEM_NAME = "stp3d"  # name of the MIPLIB problem to load


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
def lr(decay_rate, init_value, end_value):
    return optax.exponential_decay(
        init_value=init_value,
        transition_steps=int(1e4),
        decay_rate=decay_rate,
        end_value=end_value,
    )


learning_rates = [
    lr(decay_rate, init_value, end_value)
    for decay_rate in [0.5, 0.9, 99]
    for init_value in [1e0]
    for end_value in [1e-5]
]
primal_experts = [
    optax.optimistic_adam_v2(learning_rate=lr, alpha=alpha, beta=beta)
    for lr in learning_rates
    for alpha in [0.01, 0.05, 0.1]
    for beta in [0.9, 0.99, 0.999]
]

dual_experts = [optax.adadelta(learning_rate=1.0)]

ensemble_optimiser = jo.hedge_ensemble_saddle(
    lp=jaddle_lp,
    primal_experts=primal_experts,
    dual_experts=dual_experts,
)
is_converged = False
solution = jaddle_lp.initial_solution()
opt_state = None

while (len(primal_experts) > 1) and (not is_converged):
    solution, is_converged, opt_state = jl.solve(
        lp=jaddle_lp,
        optimiser=ensemble_optimiser,
        initial_opt_state=opt_state,
        initial_solution=solution,
        verbose=True,
        expert_diagnostics=True,
        iterations_per_epoch=int(1e4),
        output_opt_state=True,
        max_epochs=1,
    )

    opt_state, primal_idx, dual_idx = opt_state.prune(threshold=1e-4)
    primal_experts = [primal_experts[i] for i in primal_idx]

    print(f"Pruned to {len(primal_experts)} primal experts.")

    ensemble_optimiser = jo.hedge_ensemble_saddle(
        lp=jaddle_lp,
        primal_experts=primal_experts,
        dual_experts=dual_experts,
    )

optimiser = jo.create_saddle_optimiser(
    primal_optimizer=primal_experts[0],
    dual_optimizer=dual_experts[0],
)

solution, is_converged = jl.solve(
    lp=jaddle_lp,
    optimiser=optimiser,
    initial_solution=solution,
    verbose=True,
    iterations_per_epoch=int(1e4),
    weight_function=lambda i: jax.lax.select(i <= int(5e4), 1e-16, 1.0),
)

# %%
print(f"Primal Equality Residual: {jaddle_lp.eq_slack(solution.primal)}")
print(f"Primal Inequality Residual: {jaddle_lp.ineq_slack(solution.primal)}")
print("----------------------------------------------")

# %%
