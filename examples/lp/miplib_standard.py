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
import jaddle.highs_helpers as hh
import jax.numpy as jnp

# %%
jo.configure_jax("float64")

# %% [markdown]
# ## Load the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
PROBLEM_NAME = "app1-2"  # name of MIPLIB problem (without .mps extension)
PATH_TO_MPS = f"/home/brendanvr/python/Jaddle/data/{PROBLEM_NAME}.mps"  # The path to the MPS file. You will need to download the MPS file from the MIPLIB website and change this path.
highs = hspy.Highs()
highs.readModel(PATH_TO_MPS)  # path to MPS file

# %%
# Relax integrality
for col in range(highs.numVariables):
    highs.changeColIntegrality(col, hspy.HighsVarType.kContinuous)

# %%
highs.setOptionValue("presolve", "off")
highs.setOptionValue("primal_feasibility_tolerance", 1e-3)
highs.setOptionValue("dual_feasibility_tolerance", 1e-3)
highs.setOptionValue("pdlp_optimality_tolerance", 1e-5)
highs.setOptionValue("solver", "pdlp")
highs.solve()


# %% [markdown]
# We convert the LP to Jaddle's sparse format, before applying the selected scaling strategy.
highs_lp = highs.getLp()
lp = hh.highs_to_standard_form_sparse(highs_lp)

# %% [markdown]
# ## Solve the presolved LP using Jaddle's saddle point solver
print("Problem:", PROBLEM_NAME)
jl.lp_summary_statistics(lp)
solution_jaddle, _ = jl.solve(
    lp,
    verbose=True,
    k_scale=1e2,
    adaptive_eta=1,
)

# %%
