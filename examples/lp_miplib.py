# %% [markdown]
# # Solve a MIPLIB LP using Jaddle Saddle Point Optimisation
# This example demonstrates how to load and solve a MIPLIB linear program (LP) using saddle point optimisation methods implemented in Jaddle.
# We will use the `highspy` library to load a MIPLIB LP from an MPS file.
# The LP will then be presolved to simplify the problem before applying saddle point optimisation.

# %%
import os

# Suppress INFO and WARNING logs from XLA/JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import jaddle.jaddle_linear as jl
import jaddle.highs_helpers as hh
import highspy as hspy

# %% [markdown]
# ## Load and presolve the LP
# We load a MIPLIB LP from an MPS file using the `highspy` library.
# The LP is then presolved to reduce its size and complexity.
# Finally, we convert the presolved LP into a format compatible with Jaddle.
highs = hspy.Highs()
highs.readModel("../data/bab1.mps")  # path to MPS file
highs.presolve()
highs_lp = highs.getPresolvedLp()
jaddle_lp = hh.highs_to_standard_form_sparse(highs_lp)

# %% [markdown]
# ## Solve the scaled, presolved LP using Jaddle's saddle point solver
start_time = time.time()
solution_primal, solution_dual = jl.solve(
    iterations_per_epoch=1000,
    lp=jaddle_lp,
    scale_A=True,
    scale_b=True,
    scale_c=True,
    max_epochs=5000,
)
print("--------------------------------")
print("Saddle point solver objective:", jaddle_lp.objective(solution_primal.primal))
print("Inequality violation:", jaddle_lp.ineq_slack(solution_primal.primal))
print("Equality violation:", jaddle_lp.eq_slack(solution_primal.primal))
print("Time to solution:", time.time() - start_time)
print("--------------------------------")

# %%
