# %% [markdown]
# # Prediction with Expert Advice over Saddle Optimisers
# This example demonstrates a Hedge-style optimiser ensemble that learns online
# over separate pools of primal and dual experts.

# %%
import jax
import numpy as np
import optax
import scipy.sparse as sp

import jaddle.jaddle_linear as jl
import jaddle.jaddle_optimisers as jo

jax.config.update("jax_platform_name", "cpu")

# %%
# Simple LP:
#   min 3 x1 + 2 x2
#   s.t. x1 + x2 >= 4, x >= 0
c = np.array([3.0, 2.0])
A_eq = sp.csc_matrix([[0.0, 0.0]])
b_eq = np.array([0.0])
A_ineq = sp.csc_matrix([[-1.0, -1.0]])
b_ineq = np.array([-4.0])
lower_bounds = np.array([0.0, 0.0])
upper_bounds = np.array([np.inf, np.inf])

lp = jl.LP(
    c=c,
    A_eq=A_eq,
    b_eq=b_eq,
    A_ineq=A_ineq,
    b_ineq=b_ineq,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
)
lp = jl.to_jaddle_sparse(lp)

# %%
# Build two expert pools with different step-size behavior.
lr_fast = optax.exponential_decay(
    init_value=1e-1,
    transition_steps=1000,
    decay_rate=0.9,
    end_value=1e-3,
)
lr_slow = optax.exponential_decay(
    init_value=3e-2,
    transition_steps=1000,
    decay_rate=0.99,
    end_value=1e-4,
)

primal_experts = [
    optax.optimistic_adam_v2(learning_rate=lr_fast, alpha=0.05),
    optax.optimistic_adam_v2(learning_rate=lr_slow, alpha=0.05),
]

dual_experts = [
    optax.optimistic_adam_v2(learning_rate=lr_fast, alpha=0.05),
    optax.optimistic_adam_v2(learning_rate=lr_slow, alpha=0.05),
    optax.sgd(1e-1),
]

ensemble_optimiser = jo.hedge_ensemble_saddle(
    primal_experts=primal_experts,
    dual_experts=dual_experts,
    primal_eta=5e-2,
    dual_eta=5e-2,
    expert_loss_mode="projected_kkt_merit",
    lp=lp,
)

# %%
solution = jl.solve(
    lp,
    optimiser=ensemble_optimiser,
    verbose=True,
    expert_diagnostics=True,
)

# %%
print(f"x1 = {solution.primal[0]:.4f}, x2 = {solution.primal[1]:.4f}")
print(f"Objective value: {lp.objective(solution.primal):.4f}")
# %%
