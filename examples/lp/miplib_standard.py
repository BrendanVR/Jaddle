# %% [markdown]
# # Solve a Scaled LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# We will use the `highspy` library to load a MIPLIB LP from an MPS file. We do not ship the MPS files with Jaddle, but they can be downloaded from the [MIPLIB website](https://miplib.zib.de/).

# %%
import os

# Suppress INFO and WARNING logs from XLA/JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import highspy as hspy
import numpy as np
import jax
import jaddle.jaddle_optimisers as jo
import jaddle.jaddle_linear as jl
import jaddle.highs_helpers as hh
import jax.numpy as jnp

# %%
jo.configure_jax("float64")

# %% [markdown]
# ## Load the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
PROBLEM_NAME = "neos8"  # name of MIPLIB problem (without .mps extension)
# Download the MPS file from the MIPLIB website (https://miplib.zib.de/) and
# place it in the `data/` directory at the repo root, or override PATH_TO_MPS.
PATH_TO_MPS = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", f"{PROBLEM_NAME}.mps"
)
highs = hspy.Highs()
highs.readModel(PATH_TO_MPS)  # path to MPS file

# %%
# Relax integrality (one batched call; a per-column loop is very slow on large models)
n = highs.numVariables
highs.changeColsIntegrality(
    n, np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.uint8)
)

# %% [markdown]
# We convert the LP to Jaddle's sparse format, before applying the selected scaling strategy.
highs_lp = highs.getLp()
lp = hh.highs_to_standard_form_sparse(highs_lp)

# %%
values = jl.to_jaddle_sparse(lp).values()

tol = dict(
    primal_feasibility_tolerance=1e-6,
    dual_feasibility_tolerance=1e-6,
    dual_gap_tolerance=1e-6,
)

value_fn = jl.make_optimal_value(lp, **tol)
grads = jax.jit(jax.grad(value_fn))(values)
own = jl.make_solver(lp, **tol)(values).solution  # same settings as value_fn
jnp.linalg.norm(grads.b_ineq + own.dual_ineq)

# %%
