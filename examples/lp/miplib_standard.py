# %% [markdown]
# # Solve a Scaled LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# We will use the `highspy` library to load a MIPLIB LP from an MPS file.

# %%
import os

# Suppress INFO and WARNING logs from XLA/JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax
import optax
import highspy as hspy
import jaddle.jaddle_optimisers as jo
import jaddle.jaddle_linear as jl
import jaddle.highs_helpers as hh

# %%
jo.configure_jax("float32")

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
highs.setOptionValue("primal_feasibility_tolerance", 1e-3)
highs.setOptionValue("dual_feasibility_tolerance", 1e-3)
highs.setOptionValue("pdlp_optimality_tolerance", 1e-3)
highs.setOptionValue("solver", "pdlp")
highs.solve()


# %% [markdown]
# We convert the LP to Jaddle's sparse format, before applying the selected scaling strategy.
highs_lp = highs.getLp()
lp = hh.highs_to_standard_form_sparse(highs_lp)

# %% [markdown]
# ## Solve the presolved LP using Jaddle's saddle point solver


def sgd_dual_momentum(lr):
    primal = optax.inject_hyperparams(optax.sgd)(
        learning_rate=lr,
    )
    dual = optax.inject_hyperparams(optax.sgd)(
        learning_rate=lr,
        nesterov=True,
        momentum=0.5,
    )
    return jo.create_saddle_optimiser(
        primal,
        dual,
    )


def optimisitic_sgd(lr):
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


def polisher(lr):
    primal = optax.inject_hyperparams(optax.amsgrad)(learning_rate=lr, b1=0.0)
    dual = optax.inject_hyperparams(optax.amsgrad)(learning_rate=lr, b1=0.0)
    return jo.create_saddle_optimiser(
        primal,
        dual,
    )


# %%
print("Solving Problem:", PROBLEM_NAME)
jl.lp_summary_statistics(lp)

# %%
solution_jaddle, _ = jl.solve(
    lp=lp,
    optimiser=optimisitic_sgd(1 / 2),
    scale="ruiz+pc",
    update_mode="alternating",
    average=False,
    verbose=True,
    log_every=1,
)

# %%
