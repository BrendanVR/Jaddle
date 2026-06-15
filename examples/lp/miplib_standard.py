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

import jaddle.jaddle_linear as jl
import jaddle.highs_helpers as hh
import jaddle.jaddle_optimisers as jo
import highspy as hspy
import optax

# %%
jo.configure_jax("max_speed")

# %% [markdown]
# ## Load the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
PROBLEM_NAME = "boeing"  # name of MIPLIB problem (without .mps extension)
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


def ogd(lr):
    return optax.inject_hyperparams(optax.optimistic_gradient_descent)(learning_rate=lr)


k = 1e1
eta = 1e4
base_lr = 1 / (eta)
optimiser = jo.create_saddle_optimiser(
    ogd(base_lr),
    ogd(base_lr),
)

jl.lp_summary_statistics(jaddle_lp)
solution, _ = jl.solve(
    lp=jaddle_lp,
    optimiser=optimiser,
    scale="ruiz+pc",
    dual_gap_tolerance=1e-1,
    primal_feasibility_tolerance=1e-2,
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
    restarts=40,
    average="polyak",
    weight_function=lambda i: jax.lax.cond(
        i < int(5e4),
        lambda _: 1e-5,
        lambda _: 1.0,
        operand=None,
    ),
)

print(f"Primal Equality Residual: {jaddle_lp.eq_slack(solution.primal)}")
print(f"Primal Inequality Residual: {jaddle_lp.ineq_slack(solution.primal)}")
print("----------------------------------------------")

# %%
jaddle_lp.objective(solution.primal)


# %%
