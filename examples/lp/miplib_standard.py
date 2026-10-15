# %% [markdown]
# # Solve a Scaled LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# We will use the `highspy` library to load a MIPLIB LP from an MPS file. We do not ship the MPS files with Jaddle, but they can be downloaded from the [MIPLIB website](https://miplib.zib.de/).

# %%
import os
from time import time
from tracemalloc import start

# Suppress INFO and WARNING logs from XLA/JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax
import jaddle.jaddle_optimisers as jo
import jaddle.jaddle_linear as jl
import highspy as hspy
import numpy as np

# %%
jo.configure_jax("float64")

# %% [markdown]
# ## Path to the LP
PROBLEM_NAME = "stp3d"  # name of MIPLIB problem (without .mps extension)
PATH_TO_MPS = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", f"{PROBLEM_NAME}.mps"
)

# %% [markdown]
# ## Read in the problem with HiGHS
highs = hspy.Highs()
highs.readModel(PATH_TO_MPS)

n = highs.getNumCol()
highs.changeColsIntegrality(
    n, np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.uint8)
)

# %% [markdown]
# ## Solve the LP with Jaddle
# Here we solve the LP using Jaddle, crossing over to the simplex method to render the solution basic.
problem_options = dict(
    primal_feasibility_tolerance=1e-8,
    dual_feasibility_tolerance=1e-4,
    dual_gap_tolerance=1e-4,
    scaled_augmented=False,
    cost_col_floor=1e-2,
)

start_time = time()
solution = jl.solve_with_presolve(
    highs,
    run_crossover=True,
    polish=True,
    verbose=True,
    highs_options=dict(output_flag="true"),
    **problem_options,
)
end_time = time()
print("Jaddle solve time:", end_time - start_time)

# %%
print("Jaddle objective value:", solution["objective"])
print(
    "Jaddle primal feasibility:", solution["certificate"]["primal_feasibility_residual"]
)
print("Jaddle duality gap:", solution["certificate"]["duality_gap"])

# %%
# %%
