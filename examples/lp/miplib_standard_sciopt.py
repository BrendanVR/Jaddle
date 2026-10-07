# %% [markdown]
# # Solve a Scaled LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# We will use the `highspy` library to load a MIPLIB LP from an MPS file. We do not ship the MPS files with Jaddle, but they can be downloaded from the [MIPLIB website](https://miplib.zib.de/).

# %%
import os

# Suppress INFO and WARNING logs from XLA/JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import highspy as hspy
import jaddle.jaddle_optimisers as jo
import jaddle.jaddle_linear as jl
import jaddle.sciopt_helpers as sciopt
from pyscipopt import Model, SCIP_PARAMSETTING

# %%
jo.configure_jax("float64")

# %% [markdown]
# ## Load the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
PROBLEM_NAME = "stp3d"  # name of MIPLIB problem (without .mps extension)
# Download the MPS file from the MIPLIB website (https://miplib.zib.de/) and
# place it in the `data/` directory at the repo root, or override PATH_TO_MPS.
PATH_TO_MPS = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", f"{PROBLEM_NAME}.mps"
)

model = sciopt.read_relaxed_model(PATH_TO_MPS)

# %%
model.setPresolve(SCIP_PARAMSETTING.OFF)
model.setParam("presolving/maxrounds", -1)
model.setParam("presolving/milp/maxrounds", -1)
model.setParam("misc/allowstrongdualreds", False)
model.setParam("misc/allowweakdualreds", False)
model.presolve()

# %% [markdown]
# We convert the LP to Jaddle's sparse format, before applying the selected scaling strategy.
lp, offset = sciopt.scip_to_standard_form_sparse(model, transformed=True)

# %% [markdown]
# ## Solve the presolved LP using Jaddle's saddle point solver
print("Problem:", PROBLEM_NAME)
jl.lp_summary_statistics(lp)
solution_jaddle = jl.solve(
    lp,
    verbose=True,
    iterations_per_epoch=128 * 10,
    restarts=True,
    epochs_per_restart=20,
)["solution"]

print("Solution (Jaddle):", lp.objective(solution_jaddle.primal) + offset)

# %%
