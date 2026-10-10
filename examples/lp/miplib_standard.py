# %% [markdown]
# # Solve a Scaled LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# We will use the `highspy` library to load a MIPLIB LP from an MPS file. We do not ship the MPS files with Jaddle, but they can be downloaded from the [MIPLIB website](https://miplib.zib.de/).

# %%
import os

# Suppress INFO and WARNING logs from XLA/JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jaddle.jaddle_optimisers as jo
import jaddle.jaddle_linear as jl
import highspy as hspy
import numpy as np

# %%
jo.configure_jax("float64")

# %% [markdown]
# ## Path to the LP
PROBLEM_NAME = "sct1"  # name of MIPLIB problem (without .mps extension)
PATH_TO_MPS = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", f"{PROBLEM_NAME}.mps"
)

# %% [markdown]
# ## Solve he LP with HiGHS
highs = hspy.Highs()
highs.setOptionValue("output_flag", "true")
highs.readModel(PATH_TO_MPS)

n = highs.getNumCol()
highs.changeColsIntegrality(
    n, np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.uint8)
)

highs.run()

print("HiGHS status:", highs.getModelStatus())
print("HiGHS objective value:", highs.getObjectiveValue())

# %% [markdown]
# ## Solve the LP with Jaddle
# Here we solve the LP using Jaddle, crossing over to the simplex method to render the solution basic.
tol = 1e-6
problem_options = dict(
    primal_feasibility_tolerance=tol,
    dual_feasibility_tolerance=tol,
    dual_gap_tolerance=tol,
)

solution = jl.solve_with_presolve(
    PATH_TO_MPS,
    run_crossover=True,
    verbose=True,
    highs_options=dict(output_flag="true"),
    **problem_options,
)

# %%
# %%
solution
# %%
for k in solution.keys():
    print(k)
# %%
