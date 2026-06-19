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
# highs.presolve()
highs_lp = highs.getLp()
jaddle_lp = jl.to_jaddle_sparse(hh.highs_to_standard_form_sparse(highs_lp))

# %% [markdown]
# ## Solve the presolved LP using Jaddle's saddle point solver


k_max = 1e3


def opt(lr):
    primal = optax.inject_hyperparams(optax.adadelta)(
        learning_rate=lr,
    )

    dual = optax.inject_hyperparams(optax.sgd)(
        learning_rate=lr,
        momentum=0.3,
        nesterov=True,
    )

    return jo.create_saddle_optimiser(
        primal,
        dual,
    )


def polisher(lr):
    return jo.create_saddle_optimiser(
        optax.inject_hyperparams(optax.amsgrad)(
            learning_rate=lr,
            b1=0,
        ),
        optax.inject_hyperparams(optax.amsgrad)(
            learning_rate=lr,
            b1=0,
        ),
    )


# %%
print("Solving Problem:", PROBLEM_NAME)
jl.lp_summary_statistics(jaddle_lp)
solution, _ = jl.solve(
    lp=jaddle_lp,
    optimiser=opt(1),
    iterations_per_epoch=1000,
    scale="ruiz+pc",
    verbose=True,
    log_every=1,
    extragradient=True,
    per_iterate_k_theta=0.1,
    per_iterate_k_lo=1 / k_max,
    per_iterate_k_hi=k_max,
    restarts=20,
    epochs_per_restart=10,
    restart_multiplier=1.5,
    iterations_per_epoch_decay=0.9,
    iterations_per_epoch_min=10,
    average="polyak",
    polish_optimiser=polisher(1e-3),
    polish_merit_threshold=1e-2,
    dual_gap_tolerance=1e0,
)

print(f"Primal Equality Residual: {jaddle_lp.eq_slack(solution.primal)}")
print(f"Primal Inequality Residual: {jaddle_lp.ineq_slack(solution.primal)}")
print("----------------------------------------------")

# %%
jaddle_lp.objective(solution.primal)

# %%
sol = jl.project_onto_eq(jaddle_lp, solution.primal, 1e-5)

# %%
print(f"Primal Equality Residual (Projected): {jaddle_lp.eq_slack(sol)}")
print(f"Primal Inequality Residual (Projected): {jaddle_lp.ineq_slack(sol)}")
print("----------------------------------------------")

# %%
