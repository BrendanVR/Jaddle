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

jax.config.update("jax_log_compiles", True)

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
PROBLEM_NAME = "nug"  # name of MIPLIB problem (without .mps extension)
highs = hspy.Highs()
highs.readModel(
    f"/home/brendanvr/python/Jaddle/data/{PROBLEM_NAME}.mps"
)  # path to MPS file

# %%
# Relax integrality
for col in range(highs.numVariables):
    highs.changeColIntegrality(col, hspy.HighsVarType.kContinuous)

# %%
highs.setOptionValue("presolve", "off")
highs.setOptionValue("solver", "pdlp")
highs.solve()
solution_highs = highs.getSolution()

# %% [markdown]
# We convert the LP to Jaddle's sparse format, before applying the selected scaling strategy.
# highs.presolve()
highs_lp = highs.getLp()
jaddle_lp = jl.to_jaddle_sparse(hh.highs_to_standard_form_sparse(highs_lp))

# %% [markdown]
# ## Solve the presolved LP using Jaddle's saddle point solver


def sgd(lr):
    primal = optax.inject_hyperparams(optax.sgd)(
        learning_rate=lr,
    )

    dual = optax.inject_hyperparams(optax.sgd)(
        learning_rate=lr,
        nesterov=True,
        momentum=0.3,
    )

    return jo.create_saddle_optimiser(
        primal,
        dual,
    )


def optimistic_sgd(lr):
    primal = optax.inject_hyperparams(optax.optimistic_gradient_descent)(
        learning_rate=lr,
    )

    dual = optax.inject_hyperparams(optax.optimistic_gradient_descent)(
        learning_rate=lr,
    )

    return jo.create_saddle_optimiser(
        primal,
        dual,
    )


# %%
print("Solving Problem:", PROBLEM_NAME)
jl.lp_summary_statistics(jaddle_lp)
solution_jaddle, _ = jl.solve(
    lp=jaddle_lp,
    optimiser=optimistic_sgd(1 / 2),
    iterations_per_epoch=1000,
    scale="ruiz+pc",
    update_mode="synchronous",
    average=False,
    precompile=True,
    verbose=True,
    log_every=1,
)

print(f"Primal Equality Residual: {jaddle_lp.eq_slack(solution_jaddle.primal)}")
print(f"Primal Inequality Residual: {jaddle_lp.ineq_slack(solution_jaddle.primal)}")
print("----------------------------------------------")

# %%
